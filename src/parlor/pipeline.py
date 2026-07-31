"""The streaming turn pipeline: message-content helpers, the incremental
response/transcript parser, and run_turn (decode → sentences → TTS, all
pipelined), plus speculative cache priming."""

import asyncio
import base64
import io
import json
import re
import time
import wave

import numpy as np

from parlor import llama

# Parsed tolerantly ("### TRANSCRIPT : ..." happens) — but the colon is
# REQUIRED and only [ \t] may follow: this regex runs against a partially
# streamed buffer, so an optional colon matches before the ":" token
# arrives (leaking it into the transcript) and \s* would let a newline
# delta terminate an empty transcript line.
TRANSCRIPT_TAG_RE = re.compile(r"#{2,}[ \t]*TRANSCRIPT[ \t]*:[ \t]*", re.IGNORECASE)
SENTENCE_END_RE = re.compile(r"[.!?]+\s")
MAX_OUTPUT_TOKENS = 256

# Appended (inside the final WAV) before the LLM sees the utterance: audio
# that stops abruptly at the VAD cutoff makes the encoder hallucinate a
# confident completion of the last word; a beat of silence fixes it.
TAIL_SILENCE_S = 0.3

# Rough per-part token costs for the context-rotation estimate.
AUDIO_TOKENS_PER_SEC = 32
IMAGE_TOKENS = 300

_DONE = object()


# ── message content ───────────────────────────────────────────────────────

def image_part(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}


def audio_part(b64: str) -> dict:
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def valid_audio(b64: str | None) -> bool:
    """At least ~100ms of 16kHz s16 WAV — llama-server 400s on empty audio,
    and one bad message in history would poison every later request."""
    return bool(b64) and len(b64) * 3 // 4 > 44 + 3200


def user_content(image_b64: str | None, audio_b64s: list[str]) -> list:
    """Media parts of the current user turn, in canonical (cache-stable)
    order: image first, then audio segments oldest-to-newest."""
    parts = [image_part(image_b64)] if image_b64 else []
    parts += [audio_part(b) for b in audio_b64s]
    return parts


def estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content) // 4 + 8
            continue
        for p in content:
            if p["type"] == "text":
                total += len(p["text"]) // 4
            elif p["type"] == "input_audio":
                wav_bytes = len(p["input_audio"]["data"]) * 3 // 4
                total += (wav_bytes // 32000) * AUDIO_TOKENS_PER_SEC  # 16kHz s16
            else:
                total += IMAGE_TOKENS
        total += 8
    return total


def wav_to_float32(b64: str) -> np.ndarray:
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def pad_tail_silence(b64: str) -> str:
    """Append TAIL_SILENCE_S of silence inside the WAV. Must be in the same
    WAV as the speech — a separate silence part doesn't stop the encoder
    hallucinating a completion of an abruptly-cut last word."""
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    frames += b"\x00" * (params.sampwidth * params.nchannels
                         * int(TAIL_SILENCE_S * params.framerate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode()


# ── websocket protocol ────────────────────────────────────────────────────

async def send_json(ws, payload: dict) -> None:
    await ws.send_text(json.dumps(payload))


async def release_client(ws) -> None:
    """Terminal frame for a turn that produced nothing. Without it the client
    sits in 'processing' until its watchdog fires."""
    await send_json(ws, {"type": "turn_final", "transcription": None,
                         "timings": {}, "spoke": False})


# ── streaming turn parser ─────────────────────────────────────────────────

class TagFilter:
    """Streams response text through, extracting complete
    '<name>value</name>' control elements into .tags. XML elements (not
    ###NAME: lines) because the model emits them more reliably —
    benchmarks/tagbench.py measured recall 0.667 vs 0.417 on E4B — and
    the explicit close tag means a half-streamed value can never fire.

    Parsed tolerantly ('< delegate >x</delegate>' extracts — firing the
    intended action beats suppressing it) and the value may contain '<'
    ('flights under <$500'); only the close tag terminates it. Anything
    that names a control tag without being a clean element (a stray
    '</delegate>', '<delegate x=1>') suppresses all further speech,
    mirroring the ##-markup rule: it is model error, and speaking it —
    task text included — is the worst outcome. Ordinary prose '<'
    ('5 < 10') passes through. Markup still open when the stream ends is
    dropped, never spoken and never fired: a delegation with half its
    task is worse than none."""

    def __init__(self, names: tuple[str, ...]):
        alt = "|".join(names)  # names are plain lowercase words
        self._re = re.compile(
            rf"<\s*(?P<name>{alt})\s*>\s*(?P<value>.*?)\s*<\s*/\s*(?P=name)\s*>",
            re.IGNORECASE | re.DOTALL)
        # An element left unclosed at end of stream (see finalize), and
        # the same shape anywhere-to-end-of-text (for strip).
        self._unclosed_re = re.compile(
            rf"^<\s*(?P<name>{alt})\s*>\s*(?P<value>\S.*?)\s*$",
            re.IGNORECASE | re.DOTALL)
        self._tail_open_re = re.compile(rf"<\s*(?:{alt})\s*>.*$",
                                        re.IGNORECASE | re.DOTALL)
        # A complete opening bracket: the element is committed, hold until
        # its close tag (or end of stream) decides it.
        self._open_re = re.compile(rf"^<\s*(?:{alt})\s*>", re.IGNORECASE)
        # Names a control tag (with optional '/' and spacing) — but only
        # consulted after _re and _open_re failed, so it is a near-miss.
        self._miss_re = re.compile(rf"^<\s*/?\s*(?:{alt})\b", re.IGNORECASE)
        # Too short to judge: '<', '</', '< dele' — letters (a name prefix)
        # possibly followed by whitespace, still awaiting a decisive char.
        self._forming_re = re.compile(r"^<[\s/]*([a-zA-Z]*)\s*$")
        self._names = [n.lower() for n in names]
        self._held = ""
        self._dead = False
        self.tags: list[tuple[str, str]] = []  # (NAME, value), stream order

    def feed(self, delta: str) -> str:
        """Returns the speakable text released by this delta."""
        if self._dead:
            return ""
        buf = self._held + delta
        self._held = ""
        out = []
        while buf:
            lt = buf.find("<")
            if lt == -1:
                out.append(buf)
                break
            out.append(buf[:lt])
            m = self._re.match(buf, lt)
            if m:
                self.tags.append((m.group("name").upper(), m.group("value").strip()))
                out.append("\n")  # keep a boundary where the element sat
                buf = buf[m.end():]
                continue
            rest = buf[lt:]
            forming = self._forming_re.match(rest)
            if self._open_re.match(rest) or (
                    forming and any(n.startswith(forming.group(1).lower())
                                    for n in self._names)):
                self._held = rest  # a clean element may still complete
                break
            if self._miss_re.match(rest):
                self._dead = True  # markup, not speech — nothing more is spoken
                break
            out.append("<")  # literal '<', not ours
            buf = buf[lt + 1:]
        return "".join(out)

    def finalize(self) -> None:
        """End of stream. An element still open here is the model hitting
        EOS before the close tag — measured live, a third of exit-command
        '<mode>conversation' switches end exactly like that. The value is
        as complete as it will ever be, so extract it (still never
        spoken). A mid-stream half value can never fire — this only runs
        when no more text is coming — and the residual truncation risk is
        benign: an incomplete mode value no-ops, a delegation task has
        the cap/clamp guards."""
        m = self._unclosed_re.match(self._held)
        if m and not self._dead:
            self.tags.append((m.group("name").upper(), m.group("value").strip()))
        self._held = ""

    def strip(self, text: str) -> str:
        """Remove control elements, closed or unclosed-at-end, from `text`
        (for storing an interrupted turn: history must not claim an action
        fired). After closed elements are gone, any remaining open tag runs
        to the end of the text by construction."""
        return self._tail_open_re.sub("", self._re.sub("", text))


class StreamParser:
    """Incrementally parses '###TRANSCRIPT: <words>\\n<response>'.

    The transcript line LEADS: the model commits to what it heard before
    answering. Generating it after the response instead makes it a
    paraphrase from memory — measured WER 0.39 vs 0.00 on a clean 33-word
    utterance — and the leading line lets the client show what was heard
    while the response is still decoding.

    feed() returns complete response sentences as they become available
    (transcript-line deltas return none); finalize() returns the trailing
    partial sentence and the transcript. With expect_transcript=False
    (text/image turns) the reply streams directly and any imitated
    trailing tag is cut, never spoken.

    control_tags names '<name>value</name>' elements the model may emit
    as actions (e.g. delegate, mode). A recognized element is excised by
    a TagFilter — appended to self.tags, never spoken — and speech
    resumes after it; '##…' markup keeps the terminal cut above.
    """

    def __init__(self, expect_transcript: bool = True,
                 control_tags: tuple[str, ...] = ()):
        self.response = ""
        self.transcript: str | None = None
        self._filter = TagFilter(control_tags) if control_tags else None
        self._awaiting = expect_transcript
        self._got_tag = False
        self._buf = ""
        self._before_tag = ""  # stray text before the tag → response prefix
        self._emitted = 0

    @property
    def tags(self) -> list[tuple[str, str]]:
        return self._filter.tags if self._filter else []

    def _release(self, text: str) -> str:
        """Response text passes through the control-tag filter (if any)
        before it can be spoken."""
        return self._filter.feed(text) if self._filter else text

    def feed(self, delta: str) -> list[str]:
        if self._awaiting:
            self._buf += delta
            if not self._got_tag:
                m = TRANSCRIPT_TAG_RE.search(self._buf)
                if not m:
                    return []
                self._got_tag = True
                self._before_tag = self._buf[:m.start()]
                self._buf = self._buf[m.end():]
            # A leading "\n" delta must not terminate an empty transcript.
            self._buf = self._buf.lstrip()
            newline = self._buf.find("\n")
            if newline == -1:
                if len(self._buf) < 600:
                    return []
                # Runaway transcript line: take the first sentence and stream
                # the rest, rather than holding TTS hostage.
                m = SENTENCE_END_RE.search(self._buf)
                newline = m.end() - 1 if m else len(self._buf) - 1
            self.transcript = self._buf[:newline].strip() or None
            self.response = self._release(
                (self._before_tag + self._buf[newline + 1:]).lstrip())
            self._awaiting = False
            self._buf = ""
        else:
            self.response += self._release(delta)
        return self._complete_sentences()

    def _complete_sentences(self) -> list[str]:
        # The model occasionally imitates tag-like "##…" markup — never
        # speak anything from one onwards.
        end = len(self.response)
        hash_pos = self.response.find("##", self._emitted)
        if hash_pos != -1:
            end = min(end, hash_pos)
        sentences = []
        while True:
            m = SENTENCE_END_RE.search(self.response, self._emitted, end)
            if not m:
                break
            sentence = self.response[self._emitted:m.end()].strip()
            self._emitted = m.end()
            if sentence:
                sentences.append(sentence)
        return sentences

    def finalize(self) -> tuple[list[str], str | None]:
        if self._awaiting:
            if self._got_tag:
                # No newline ever arrived (truncated stream / model ran the
                # reply onto the tag line): first sentence is the transcript,
                # the rest is the reply — never swallow it all silently.
                m = SENTENCE_END_RE.search(self._buf)
                cut = m.end() if m else len(self._buf)
                self.transcript = self._buf[:cut].strip() or None
                self.response = self._release(self._before_tag + self._buf[cut:])
            else:
                self.response = self._release(self._buf)
            self._awaiting = False
        if self._filter:
            self._filter.finalize()  # an element left open at EOS still fires
        sentences = self._complete_sentences()
        # Cut any imitated tag markup — never speak it.
        tail = re.split(r"#{2,}", self.response[self._emitted:])[0].strip()
        return sentences + ([tail] if tail else []), self.transcript


# ── turn execution ────────────────────────────────────────────────────────

# Spoken when a turn produced a control tag but no speech at all — an
# action must never feel like the assistant ignored the user. Generic on
# purpose: it stands in for a delegation ack or a mode-switch confirmation.
TAG_ACK = "Okay — one moment."


def _norm_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


# A transcript that is entirely a bracketed annotation — the sanctioned
# "(no speech)" from the turn prompts, or free-form variants the model
# produces on its own ("(noise)", "[Silence]", "*sigh*", "(background
# noise)") — reports that there were no words. It must never be shown or
# stored as user words. Real transcripts are plain words, never fully
# bracketed; the length cap keeps a genuine parenthesized ramble out.
NO_SPEECH_RE = re.compile(
    r"^(?:\(\s*[^)]{1,40}\)|\[\s*[^\]]{1,40}\]|\*\s*[^*]{1,40}\*"
    r"|no speech)[.!\s]*$",
    re.IGNORECASE)


def echoes_instruction(transcript: str, instruction: str, n: int = 5) -> bool:
    """True when the model's transcript line is an echo of the turn's own
    instruction text rather than the user's words — e.g. a flush turn's
    '###TRANSCRIPT: The user paused mid-thought, so on a new line: …',
    which the client would display as something the user said. Any run of
    n consecutive transcript words appearing verbatim in the instruction
    is an echo; genuine speech doesn't reproduce 5-word runs of prompt
    text, and shorter overlaps ('what you see') are common English.
    Double-quoted spans are stripped from the instruction first: a prompt
    quotes phrases the user is EXPECTED to say (the translate prompt's
    exit examples like "go back to normal conversation"), and a genuine
    utterance reproducing one must not read as an echo."""
    words = _norm_words(transcript)
    if len(words) < n:
        return False
    instruction = re.sub(r'"[^"]*"', " ", instruction)
    haystack = " " + " ".join(_norm_words(instruction)) + " "
    return any(" " + " ".join(words[i:i + n]) + " " in haystack
               for i in range(len(words) - n + 1))


def _instruction_text(messages: list) -> str:
    content = messages[-1].get("content", "")
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text")


async def run_turn(ws, messages: list, interrupted: asyncio.Event,
                   active: dict, tts_backend, expect_transcript: bool = True,
                   p_complete: float | None = None,
                   control_tags: tuple[str, ...] = (),
                   tts_voice: str = "af_heart",
                   proactive: bool = False,
                   fallback: str | None = None
                   ) -> tuple[str, list, int | None, bool]:
    """Stream one model turn: decode → sentences → TTS, all pipelined. The
    transcript line is pushed to the client the moment it completes, while
    the response is still decoding. Returns the raw generated text, any
    control tags the model emitted — empty if interrupted, so an aborted
    turn never fires an action — the request's real prompt token count
    when llama-server reported one (drives history rotation), and
    no_speech: True when the model wrote a transcript line that had to be
    rejected (a no-speech annotation, or an echo of the instruction) — no
    user words stand behind this turn, so the caller must not store it or
    act on its tags. A turn that merely OMITS the transcript marker is
    not no_speech: real words were heard and answered, only the format
    slipped.

    proactive marks a server-initiated turn (delegation delivery): the
    model's transcript line is its own echo, not the user's words, so no
    transcript frame is sent and turn_final carries proactive=True — the
    client must never fill a user bubble from it."""
    loop = asyncio.get_event_loop()
    t0 = time.time()
    timings: dict = {}

    chunk_q: asyncio.Queue = asyncio.Queue()
    stream = llama.ChatStream(messages, MAX_OUTPUT_TOKENS)
    active["stream"] = stream
    raw = {"text": ""}

    def produce():
        try:
            def on_delta(text):
                raw["text"] += text
                loop.call_soon_threadsafe(chunk_q.put_nowait, text)
            stream.run(on_delta)
            loop.call_soon_threadsafe(chunk_q.put_nowait, _DONE)
        except Exception as e:  # surfaced to the consumer loop
            loop.call_soon_threadsafe(chunk_q.put_nowait, e)

    producer = loop.run_in_executor(None, produce)

    sentence_q: asyncio.Queue = asyncio.Queue()
    audio_state = {"started": False, "first_audio_at": None, "chunks": 0}

    async def tts_worker():
        while True:
            sentence = await sentence_q.get()
            if sentence is _DONE:
                return
            if interrupted.is_set():
                continue  # keep draining
            pcm = await loop.run_in_executor(
                None, lambda s=sentence: tts_backend.generate(s, voice=tts_voice))
            if interrupted.is_set():
                continue
            if not audio_state["started"]:
                audio_state["started"] = True
                audio_state["first_audio_at"] = time.time()
                await send_json(ws, {"type": "audio_start",
                                     "sample_rate": tts_backend.sample_rate})
            pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
            await send_json(ws, {
                "type": "audio_chunk",
                "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                "index": audio_state["chunks"],
            })
            audio_state["chunks"] += 1

    tts_task = asyncio.create_task(tts_worker())
    parser = StreamParser(expect_transcript, control_tags)
    tts_started_at = None
    transcript_sent = False
    instruction = _instruction_text(messages)

    def clean_transcript(text: str | None) -> str | None:
        """The client shows this as the user's words — never instruction
        text the model echoed (a live flush-turn bug on real mics), and
        never a non-speech annotation ('(no speech)', '[Silence]'): that
        is the model reporting there were no words to transcribe."""
        if not text:
            return None
        if NO_SPEECH_RE.match(text):
            print(f"Transcript is a no-speech report: {text!r}")
            return None
        if echoes_instruction(text, instruction):
            print(f"Transcript suppressed (instruction echo): {text!r}")
            return None
        return text

    async def dispatch(sentences: list[str]):
        nonlocal tts_started_at
        for sentence in sentences:
            # A sentence that reproduces a run of the turn's own instruction
            # is the model echoing its prompt, not speech — observed live as
            # 'CRIPT: Begin your reply with one line:' read aloud after a
            # degenerate transcript loop was cut mid-tag. Never on proactive
            # turns (the delivery prompt embeds the answer being relayed),
            # and at n=6 so short quoted phrases the user may legitimately
            # trigger ('go back to normal conversation') still speak.
            # Suppression is TTS/display-only: the sentence stays in the
            # raw text, so a mixed turn (real transcript + one echoed
            # sentence) stores the echo — an all-echo turn is dropped
            # wholesale via no_speech, which is the poisoning case.
            if not proactive and echoes_instruction(sentence, instruction, n=6):
                print(f"Sentence suppressed (instruction echo): {sentence!r}")
                continue
            if tts_started_at is None:
                tts_started_at = time.time()
            await send_json(ws, {"type": "text_delta", "text": sentence + " "})
            sentence_q.put_nowait(sentence)

    try:
        while True:
            item = await chunk_q.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            if "prefill_s" not in timings:
                timings["prefill_s"] = round(time.time() - t0, 3)
            if not interrupted.is_set():
                await dispatch(parser.feed(item))
                if parser.transcript and not transcript_sent:
                    transcript_sent = True
                    timings["transcript_s"] = round(time.time() - t0, 3)
                    shown = clean_transcript(parser.transcript)
                    if shown and not proactive:
                        await send_json(ws, {"type": "transcript",
                                             "transcription": shown,
                                             "p_complete": p_complete})

        tail, transcript = parser.finalize()
        timings["llm_time"] = round(time.time() - t0, 3)
        timings["decode_s"] = round(timings["llm_time"] - timings.get("prefill_s", 0), 3)

        if not interrupted.is_set():
            await dispatch(tail)
            # A turn may still end with nothing spoken: only a transcript
            # line (models of every size do this for cut-off audio), or only
            # a control tag. Neither may end in silence — speak the ack or
            # the caller's fallback, and keep history coherent with what was
            # actually said.
            # A proactive (delivery) turn always prefers its fallback: it IS
            # the answer, and a stray tag must not replace it with an ack.
            say = fallback if proactive else (TAG_ACK if parser.tags else fallback)
            if tts_started_at is None and say:
                raw["text"] += "\n" + say
                await dispatch([say])
    finally:
        active["stream"] = None
        sentence_q.put_nowait(_DONE)
        try:
            await tts_task
        finally:
            # The producer thread must be done before anyone reuses the slot.
            await producer

    if audio_state["first_audio_at"]:
        timings["ttfa_s"] = round(audio_state["first_audio_at"] - t0, 3)
    if tts_started_at:
        timings["tts_time"] = round(time.time() - tts_started_at, 3)

    print(
        f"LLM ({timings['llm_time']:.2f}s, prefill {timings.get('prefill_s')}s) "
        f"heard: {transcript!r} → {parser.response.strip()!r}"
    )

    def turn_no_speech() -> bool:
        return (not proactive and parser.transcript is not None
                and clean_transcript(parser.transcript) is None)

    if interrupted.is_set():
        print("Interrupted mid-turn")
        # An aborted turn fires nothing, so its stored text must not carry
        # a tag either — the model must not believe it already delegated.
        text = parser._filter.strip(raw["text"]) if parser._filter else raw["text"]
        return text, [], stream.prompt_tokens, turn_no_speech()

    await send_json(ws, {
        "type": "turn_final",
        "transcription": None if proactive else clean_transcript(transcript),
        "proactive": proactive,
        "timings": timings,
        "p_complete": p_complete,
        "spoke": audio_state["started"],  # False → client must not wait for audio_end
    })
    if audio_state["started"]:
        await send_json(ws, {"type": "audio_end",
                             "tts_time": timings.get("tts_time", 0)})
    return raw["text"], parser.tags, stream.prompt_tokens, turn_no_speech()


async def prime_cache(messages: list) -> None:
    """Fire-and-discard request that pushes a prompt prefix (camera frame,
    speech chunks) through llama-server's cache while the user is talking.
    Content must be media-only appends — a trailing text block would diverge
    the prefix and kill reuse. Failure is not worth reporting: the turn still
    works, it just pays full prefill."""
    t0 = time.time()
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: llama.chat_blocking(messages, max_tokens=1))
        print(f"Primed cache ({time.time() - t0:.2f}s)")
    except Exception as e:
        print(f"Cache priming failed: {e}")
