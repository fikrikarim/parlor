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

    The reply is pure speech — actions are decided by a separate request
    (see actions.py), so there are no control tags to excise; '##…'
    markup keeps the terminal cut below.
    """

    def __init__(self, expect_transcript: bool = True):
        self.response = ""
        self.transcript: str | None = None
        self._awaiting = expect_transcript
        self._got_tag = False
        self._buf = ""
        self._before_tag = ""  # stray text before the tag → response prefix
        self._emitted = 0

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
            self.response = (self._before_tag + self._buf[newline + 1:]).lstrip()
            self._awaiting = False
            self._buf = ""
        else:
            self.response += delta
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
                self.response = self._before_tag + self._buf[cut:]
            else:
                self.response = self._buf
            self._awaiting = False
        sentences = self._complete_sentences()
        # Cut any imitated tag markup — never speak it.
        tail = re.split(r"#{2,}", self.response[self._emitted:])[0].strip()
        return sentences + ([tail] if tail else []), self.transcript


# ── turn execution ────────────────────────────────────────────────────────

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
                   tts_voice: str = "af_heart",
                   proactive: bool = False,
                   fallback: str | None = None
                   ) -> tuple[str, int | None, bool, bool]:
    """Stream one model turn: decode → sentences → TTS, all pipelined. The
    transcript line is pushed to the client the moment it completes, while
    the response is still decoding. Returns the raw generated text, the
    request's real prompt token count when llama-server reported one
    (drives history rotation), no_speech — True when the model wrote a
    transcript line that had to be rejected (a no-speech annotation, or
    an echo of the instruction): no user words stand behind this turn, so
    the caller must not store it or act on it (a turn that merely OMITS
    the transcript marker is not no_speech: real words were heard and
    answered, only the format slipped) — and spoke: whether any TTS audio
    was sent; the caller's floor bookkeeping must key on it, not on the
    text (a transcript-only turn has text but plays nothing).

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
            try:
                pcm = await loop.run_in_executor(
                    None, lambda s=sentence: tts_backend.generate(s, voice=tts_voice))
            except Exception as e:
                # A sentence the voice can't speak (a script its G2P can't
                # phonemize — e.g. Japanese through the English fallback
                # voice) costs that sentence its audio, never the turn:
                # the text already reached the client's transcript.
                print(f"TTS failed on {sentence[:40]!r}: {type(e).__name__}: {e}")
                continue
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
    parser = StreamParser(expect_transcript)
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
            # The model sometimes ANNOTATES instead of speaking — a reply
            # of literally '(no speech)' (observed at temp 0 in listen
            # mode, imitating the transcript clause). An annotation is a
            # report that there is nothing to say; NO_SPEECH_RE already
            # pins the shape, and no genuine reply is one bracketed
            # aside. Suppressing it lets a turn end silent instead of
            # speaking markup.
            if NO_SPEECH_RE.match(sentence):
                print(f"Sentence suppressed (annotation): {sentence!r}")
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
            # A turn may still end with nothing spoken — only a transcript
            # line (models of every size do this for cut-off audio). It must
            # not end in silence: speak the caller's fallback, and keep
            # history coherent with what was actually said.
            if tts_started_at is None and fallback:
                raw["text"] += "\n" + fallback
                await dispatch([fallback])
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
        return (raw["text"], stream.prompt_tokens, turn_no_speech(),
                audio_state["started"])

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
    return (raw["text"], stream.prompt_tokens, turn_no_speech(),
            audio_state["started"])


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
