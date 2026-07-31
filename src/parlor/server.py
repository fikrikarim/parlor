"""Parlor — on-device, real-time multimodal AI (voice + vision).

LLM inference runs on llama.cpp (llama-server, spawned as a subprocess —
see llama.py). This server owns the conversation history and re-sends it
every request; llama-server's prefix cache makes that cheap, and it also
enables two speculative tricks (see pipeline.py): the camera frame and the
user's speech (in ~3s chunks) are pushed through cache-priming requests
WHILE the user is still talking, so the final request only pays for the
tail of the utterance.
"""

import asyncio
import itertools
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # logs stream even when piped

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

try:  # raised by ws.send_text after the client goes away (uvicorn-specific)
    from uvicorn.protocols.utils import ClientDisconnected
except ImportError:  # pragma: no cover
    class ClientDisconnected(OSError):
        pass

DISCONNECT_ERRORS = (WebSocketDisconnect, ClientDisconnected)

from dotenv import load_dotenv
load_dotenv()  # before importing llama — it reads its config at import time

from parlor import llama, reasoner, tts
from parlor.modes import MODES
from parlor.pipeline import (estimate_tokens, pad_tail_silence, prime_cache,
                             release_client, run_turn, send_json, text_part,
                             user_content, valid_audio, wav_to_float32)

# Turn completeness is judged by the smart-turn audio classifier before the
# LLM is involved, so the prompt carries no FINISHED/WAIT machinery at all.
# Asking Gemma to judge it instead scores at chance on audio — see
# benchmarks/turnbench.py, which still reproduces those two variants.
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user talks to you "
    "through a microphone and may show you their camera. Your replies are "
    "spoken aloud, so write plain conversational text without formatting. "
    "If an audio message is just your own previous reply playing back "
    "(echo), don't answer it — briefly ask what they'd like to talk about."
)

# Appended to the system prompt only when a reasoner endpoint is
# configured (.env REASONER_*). An XML element, not a ###-style line:
# benchmarks/tagbench.py measured much higher tag recall for it on E4B.
DELEGATE_INSTRUCTION = (
    " You also have a background research assistant with web access. When "
    "the user asks you to search, look up, find, or research something, or "
    "asks about anything current or changing (weather, news, prices, "
    "scores, openings, \"right now\", \"today\"), you MUST hand the task "
    "over instead of answering from memory — your knowledge is stale and a "
    "guess is worse than handing over. Sports, elections, and rankings "
    "count as current. To hand over: say one short "
    "sentence telling the user you're on it, then append <delegate>the "
    "task, restated to stand alone</delegate> — never speak or mention "
    "that tag; the result arrives later and you can share it then. "
    "Everything else, answer yourself and don't use the tag."
)

# The mode-switch instruction lives in the per-turn prompt, not the
# system prompt: E4B acts on the instruction adjacent to the audio (like
# the ###TRANSCRIPT line, 6/6 in probing) and ignores the same words in
# the system prompt entirely (0/6, even with few-shot examples) — for
# translation it believes it already has the capability, so only the
# always-attended slot makes it emit the switch.
MODE_SUFFIX = (
    " If the audio asks you to translate everything they say from now on, "
    "confirm briefly and end with <mode>translate</mode>."
)

# The per-utterance instruction in translate mode. It carries the exit
# path itself: in this mode every utterance is content to translate, so
# the one exception (a command addressed to the assistant about the
# session) must be spelled out where the translating happens.
TRANSLATE_PROMPT = (
    "Live translation mode. Begin your reply with one line: ###TRANSCRIPT: "
    "followed by the exact words the user said in this message's audio, in "
    "their original language. Then, on a new line, write ONLY the English "
    "translation of those words — no commentary, no answers, no opinions. "
    "If they already spoke English, restate their words in clear English. "
    "ONE exception: if their words are a command TO YOU to stop or leave "
    "translation — like \"stop translating\", \"go back to normal "
    "conversation\", \"berhenti menerjemahkan\", \"deja de traducir\" — do "
    "NOT translate it: confirm in one short sentence and end with "
    "<mode>conversation</mode>. If the audio has no clear words, write "
    "###TRANSCRIPT: (no speech) and nothing else."
)

# A finished background task is delivered by the voice model, not read out
# raw: it ties the answer back to the conversation and keeps one voice.
# "System note (not user audio)" leads both prompts because the model
# occasionally misread the text-only turn as its own reply playing back
# and gave the anti-echo response instead of the answer (observed live).
DELIVER_PROMPT = (
    "System note (not user audio): your background research assistant "
    'just finished the task "{task}". Its answer:\n{answer}\n\n'
    "Tell the user now: one short lead-in sentence tying it back to what "
    "they asked, then the answer itself word-for-word. Never drop or "
    "change a name, number, or place — the words above are already "
    "spoken-style. Plain spoken text."
)
DELIVER_FAILED_PROMPT = (
    "System note (not user audio): your background research assistant "
    'could not finish the task "{task}". Briefly tell the user you '
    "couldn't get that answer, and that they're welcome to ask again. "
    "Don't start new research now."
)
DELIVER_FALLBACK = "Sorry — I couldn't get an answer for that one."

# One session can only usefully consume a few pending research tasks, and
# an off-the-rails reply must not queue an HTTP call per imagined tag.
MAX_PENDING_DELEGATIONS = 3

# The stream filter must ALWAYS know every control-tag name any prompt can
# incite — the system prompt (with its delegate instruction) is the
# prefix-cache prefix and survives mode switches, and a name the filter
# doesn't know is spoken aloud, task text included. Modes gate ACTING on a
# tag (spawn_delegations, apply_mode_tags), never parsing it.
CONTROL_TAGS = ("delegate", "mode")


def strip_unfired_tags(text: str, tags: list) -> str:
    """Remove tags from a turn's stored text when they did NOT fire (task
    fragment skipped, cap hit, unknown mode, mode tag on a delivery). A
    tag left in history without its action teaches the model a state the
    server isn't in — observed live as the model 'translating' from
    context while the server never switched and the mode chip never
    appeared."""
    for name, value in tags:
        text = re.sub(
            rf"<\s*{name.lower()}\s*>\s*{re.escape(value)}\s*(<\s*/\s*{name.lower()}\s*>)?",
            "", text, flags=re.IGNORECASE)
    return text

# The transcript line LEADS the reply: transcribing after the response
# turns the transcript into a paraphrase from memory (WER 0.39 vs 0.00 on a
# clean 33-word utterance), and the leading line reaches the client while
# the response still decodes. Costs its decode time (~0.2s short / ~0.7s
# long utterances) before first audio — measured worth it. Grammar-forced
# JSON ({transcript, response}) was also measured: format breaks 1-3/3 on
# degraded audio and 3/3 on chunked — don't go back to structured output.
# The no-speech clause gives the model a sanctioned out for VAD false
# triggers (breath, cough, room noise). Without it the prompt DEMANDS
# words, so on speechless audio the model invents some — measured at
# temp 0.7: a breath came back as "Hi, can you help me with my homework?"
# (answered), "can you translate everything I say from now on?" (mode
# switched!), or an earlier turn's question copied verbatim. "this
# message's audio" (not "their audio message") points the transcription
# at the newest clip among the many in history; measured clean with this
# wording at temp 0 and 0.7 (WER 0.0 on speech, fresh and late).
NO_SPEECH_CLAUSE = (
    " If the audio has no clear words — noise, breathing, silence — write "
    "###TRANSCRIPT: (no speech) instead; never guess words or repeat "
    "earlier ones."
)

RESPOND_PROMPT = (
    "Begin your reply with one line: ###TRANSCRIPT: followed by the exact "
    "words the user said in this message's audio." + NO_SPEECH_CLAUSE +
    " Then, on a new line, respond "
    "to them: 1-4 short sentences, spoken aloud.{camera}"
)

# Spoken when a turn yields no reply at all (models of every size
# occasionally emit only the transcript line) — silence would leave the
# user hanging, and a stored transcript-only reply teaches the model to
# do it again next turn.
FLUSH_FALLBACK = "Take your time — I'm listening."
AUDIO_FALLBACK = "Sorry, I didn't catch that — could you say it again?"

# A "flush" turn: the classifier judged the utterance unfinished, the user
# then stayed silent, so answer what we have — the model decides whether it
# is answerable or needs encouragement to continue. Dedicated prompt: with
# a bolted-on suffix the model would sometimes emit the transcript line and
# stop, leaving the turn silent.
FLUSH_PROMPT = (
    "Begin your reply with one line: ###TRANSCRIPT: followed by the exact "
    "words the user said in this message's audio." + NO_SPEECH_CLAUSE +
    " The user paused mid-thought, "
    "so on a new line: if their words feel unfinished, write one short, warm "
    "sentence encouraging them to continue; otherwise respond to them in 1-4 "
    "short sentences, spoken aloud.{camera}"
)

# Rotate history before the llama context fills. Rough token estimates are
# fine here — the guard just needs to fire before generation degrades.
# Scaled to the context size: a fixed 2000 against the suite's small
# LLAMA_CTX left a near-zero threshold, rotating on every turn — history
# never accumulated, so the e2e suite silently stopped exercising
# multi-turn context at all.
CONTEXT_HEADROOM = max(512, min(2000, llama.CTX // 8))


def rotate_history(history: list) -> list:
    """Drop the oldest quarter of messages, keeping the system prompt. The
    kept slice must start on a user message: dropping a user turn while
    keeping its reply would leave an orphaned assistant message about
    words that no longer exist. A history with at most one exchange
    returns unchanged (nothing droppable — and slicing a bare [system]
    would duplicate the system prompt)."""
    if len(history) <= 3:
        return history
    keep = 1 + max(2, 3 * (len(history) - 1) // 4)
    while keep > 3 and history[-(keep - 1)].get("role") != "user":
        keep -= 1
    return [history[0]] + history[-(keep - 1):]

tts_backend = None
detector = None  # smart-turn end-of-turn classifier

# Reasoner calls block for up to REASONER_TIMEOUT — keep them off the
# default executor, which serves the latency-critical path (llama
# streaming, TTS, the turn classifier, cache priming).
REASONER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reasoner")


def load_models():
    global tts_backend, detector
    llama.start()
    from parlor.turn_detector import TurnDetector
    detector = TurnDetector()
    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield
    llama.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "web" / "static"), name="static")


@app.get("/")
async def root():
    html = (Path(__file__).parent / "web" / "index.html").read_text()
    return HTMLResponse(content=html.replace("{{model}}", llama.model_label()))


def turn_instruction(msg: dict, has_image: bool, has_audio: bool) -> str:
    if has_audio:
        camera = " Mention what you see on their camera if relevant." if has_image else ""
        prompt = FLUSH_PROMPT if msg.get("type") == "flush" else RESPOND_PROMPT
        return prompt.format(camera=camera)
    if has_image:
        return "The user is showing you their camera. Describe what you see."
    return msg.get("text", "Hello!")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    delegation = reasoner.enabled()
    system = SYSTEM_PROMPT + (DELEGATE_INSTRUCTION if delegation else "")
    history: list = [{"role": "system", "content": system}]
    mode = MODES["conversation"]

    interrupted = asyncio.Event()
    active = {"stream": None}
    msg_queue = asyncio.Queue()

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    stream = active.get("stream")
                    if stream:
                        stream.cancel()  # actually aborts generation
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except DISCONNECT_ERRORS:
            pass
        finally:
            # Always unblock the main loop, even on unexpected errors.
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    frame_image: str | None = None   # camera frame held for the current utterance
    speech_chunks: list[str] = []    # streamed-in speech, already cache-primed
    held_audio: list[str] = []       # incomplete-turn segments awaiting continuation

    def remember(user_msg: dict, raw_text: str, no_speech: bool) -> None:
        """Store a finished turn verbatim (same bytes → full prefix-cache
        hit on the next request). Two kinds of turn are never stored,
        because a degenerate message poisons every later request: one the
        model produced nothing for, and a no_speech turn (the transcript
        line was a no-speech annotation or an instruction echo — no user
        words stand behind it; one stored echo loop came back as invented
        or copied user words on every turn after it). Voice turns must
        keep their raw audio: an experiment storing the transcript as
        user text instead made the model copy the PREVIOUS turn's text as
        the new turn's transcript, deterministically at temp 0 —
        user-role text reads as 'what the user said' more strongly than
        the current audio does."""
        if not raw_text.strip() or no_speech:
            return
        history.append(user_msg)
        history.append({"role": "assistant", "content": raw_text})

    delegation_ids = itertools.count(1)
    delegation_tasks: set[asyncio.Task] = set()
    ready_delegations: list[dict] = []  # finished while the floor was busy
    playing_since = {"t": 0.0}  # last spoken reply is (probably) still playing
    prompt_tokens = {"last": 0}  # real context size, from llama-server usage

    def floor_busy() -> bool:
        """The floor is held by the user (audio held for a continuation, a
        mid-utterance chunk stream, a just-fired barge-in) or by our own
        voice: a spoken reply counts as playing until the client's 'ready'
        says playback ended — a delivery starting mid-playback would cut
        the reply off mid-sentence. The 30s staleness escape means a lost
        'ready' can only delay a result, never strand it."""
        playing = playing_since["t"] and time.time() - playing_since["t"] < 30
        return bool(held_audio or speech_chunks or interrupted.is_set() or playing)

    def drain_ready() -> None:
        """Requeue one finished delegation whenever the floor frees up —
        called at every point that releases it, because msg_queue only
        wakes for client traffic and a result must not wait for one.
        Results also wait out translation mode: an English research answer
        must not barge into an interpreting session."""
        if mode.allows_delegation and ready_delegations and not floor_busy():
            msg_queue.put_nowait(ready_delegations.pop(0))

    async def switch_mode(name: str) -> bool:
        """Enter a mode by name (model tag or UI escape hatch). Unknown and
        already-current names are no-ops, so callers can pass raw values.
        Returns whether a switch happened."""
        nonlocal mode, frame_image, speech_chunks, held_audio
        name = name.strip().lower()
        if name == mode.name:
            return True  # already there — the tag "fired" in every sense
        if name not in MODES:
            # The model confirmed something out loud and then emitted junk —
            # make that diagnosable instead of a silent nothing.
            print(f"Ignoring unknown mode {name!r}")
            return False
        mode = MODES[name]
        # Start the new mode clean: a switch through the UI escape hatch can
        # fire with audio held under the OLD mode's gating, and translate
        # mode never resolves holds — stale held state would keep
        # floor_busy() true and block deliveries for the whole session.
        frame_image = None
        speech_chunks = []
        held_audio = []
        print(f"Mode → {mode.name}")
        await send_json(ws, {"type": "mode_changed", "mode": mode.name})
        drain_ready()  # leaving translation frees deferred deliveries
        return True

    async def apply_mode_tags(tags: list[tuple[str, str]]) -> list:
        """Returns the mode tags that did NOT fire (for history stripping)."""
        unfired = []
        for name, value in tags:
            if name == "MODE" and not await switch_mode(value):
                unfired.append((name, value))
        return unfired

    async def run_delegation(task_id: int, task: str) -> None:
        """Background reasoner call; its outcome re-enters the main loop
        through msg_queue so delivery is serialized with real turns."""
        try:
            answer = await asyncio.get_event_loop().run_in_executor(
                REASONER_POOL, reasoner.ask, task)
            # Clamp a verbose answer: it is interpolated into the delivery
            # prompt and stored in history under a small LLAMA_CTX.
            if len(answer) > 1500:
                answer = answer[:1500].rsplit(". ", 1)[0] + "."
            outcome = {"ok": True, "answer": answer}
        except Exception as e:
            print(f"Delegation #{task_id} failed: {e}")
            outcome = {"ok": False, "answer": ""}
        await msg_queue.put({"type": "delegation_done", "id": task_id,
                             "task": task, **outcome})

    async def spawn_delegations(tags: list[tuple[str, str]]) -> list:
        """Returns the delegate tags that did NOT fire (for history
        stripping — the model must not believe research is underway)."""
        delegate_tags = [(n, v) for n, v in tags if n == "DELEGATE"]
        if not delegation or not mode.allows_delegation:
            # The filter still parses <delegate> here (so it is never
            # spoken); without a reasoner, or mid-translation, it must
            # also never fire.
            return delegate_tags
        unfired = []
        for name, value in delegate_tags:
            task = value.strip()
            if len(task.split()) < 3:
                # An EOS-truncated fragment ('<delegate>search' cut mid-
                # element) — a one-word task sends the reasoner nothing to
                # work with (observed live: task 'search' came back as a
                # clarification request, delivered verbatim).
                print(f"Delegation skipped (fragment): {task!r}")
                unfired.append((name, value))
                continue
            if len(delegation_tasks) >= MAX_PENDING_DELEGATIONS:
                print(f"Delegation skipped (cap {MAX_PENDING_DELEGATIONS}): {task!r}")
                unfired.append((name, value))
                continue
            task_id = next(delegation_ids)
            print(f"Delegation #{task_id}: {task!r}")
            await send_json(ws, {"type": "delegation_started",
                                 "id": task_id, "task": task})
            t = asyncio.create_task(run_delegation(task_id, task))
            delegation_tasks.add(t)
            t.add_done_callback(delegation_tasks.discard)
        return unfired

    async def deliver_delegation(done: dict) -> None:
        """A server-initiated turn: the result goes into history and the
        voice model speaks it. Failures are delivered too — a delegation
        must never end in silence, which is also why the fallback is the
        reasoner's own answer: if the model's relay yields nothing
        speakable (##-markup relapse, transcript-only reply), TTS speaks
        the answer directly."""
        interrupted.clear()
        await send_json(ws, {"type": "delegation_resolved",
                             "id": done["id"], "ok": done["ok"]})
        prompt = (DELIVER_PROMPT if done["ok"] else DELIVER_FAILED_PROMPT).format(
            task=done["task"], answer=done["answer"])
        user_msg = {"role": "user", "content": [text_part(prompt)]}
        # expect_transcript=True although there is no audio: the model
        # often opens a delivery with an imitated ###TRANSCRIPT: line, and
        # the transcript parser consumes it and streams the real delivery;
        # with False the ##-markup cut would swallow the entire reply
        # (observed live).
        raw_text, tags, pt, _ = await run_turn(ws, history + [user_msg], interrupted,
                                               active, tts_backend,
                                               expect_transcript=True,
                                               control_tags=CONTROL_TAGS,
                                               tts_voice=mode.tts_voice,
                                               proactive=True,
                                               fallback=done["answer"] if done["ok"]
                                               else DELIVER_FALLBACK)
        if pt:
            prompt_tokens["last"] = pt
        if raw_text.strip():
            playing_since["t"] = time.time()  # this delivery is now playing
        unfired = await spawn_delegations(tags)  # a delivery may chain research
        unfired += [(n, v) for n, v in tags if n == "MODE"]  # never applied here
        remember(user_msg, strip_unfired_tags(raw_text, unfired), no_speech=False)
        drain_ready()                  # more results may already be waiting

    async def prime(audio_b64s: list[str]) -> None:
        """Warm the cache for the turn as it stands so far — reads the live
        history and held camera frame, so it must stay in this scope."""
        await prime_cache(history + [
            {"role": "user", "content": user_content(frame_image, audio_b64s)}])

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            # Rotate history before the llama context fills: keep the system
            # prompt and the most recent exchanges. The REAL count from the
            # last request wins over the estimate — when the estimate
            # undershoots (camera turns), llama-server silently truncates
            # the oldest turns first, which reads as the model "forgetting".
            # Double headroom because the incoming turn isn't counted yet;
            # drop a quarter (not half) so a rotation is barely noticeable.
            est = estimate_tokens(history)
            used = max(est, prompt_tokens["last"])
            if used > llama.CTX - 2 * CONTEXT_HEADROOM:
                rotated = rotate_history(history)
                if len(rotated) < len(history):
                    print(f"Context near limit (est {est}, real {prompt_tokens['last']}) "
                          f"— dropping {len(history) - len(rotated)} oldest messages")
                    history = rotated
                    prompt_tokens["last"] = 0  # stale after a rotation

            if msg.get("type") == "ready":
                # The client returned to idle listening: playback finished,
                # or a false barge-in never became an utterance. A sticky
                # interrupted flag must not strand queued deliveries — and
                # the client cleared its own frame-discard flag before
                # sending this, so delivering now is safe.
                playing_since["t"] = 0.0
                interrupted.clear()
                drain_ready()
                continue

            if msg.get("type") == "set_mode":
                # The UI escape hatch (mode chip's stop button): a switch
                # that must work even when the model mistranslates the
                # spoken exit command.
                await switch_mode(str(msg.get("mode", "")))
                continue

            if msg.get("type") == "delegation_done":
                if not mode.allows_delegation:
                    # Parked until the user exits translation — tell the
                    # client so its chip stops implying work in progress.
                    await send_json(ws, {"type": "delegation_parked",
                                         "id": msg["id"]})
                    ready_delegations.append(msg)
                    continue
                if floor_busy():
                    ready_delegations.append(msg)  # deliver at the next idle moment
                    continue
                try:
                    await deliver_delegation(msg)
                except DISCONNECT_ERRORS:
                    raise
                except Exception:
                    traceback.print_exc()  # keep the session alive
                    if not msg.get("redelivered"):
                        msg["redelivered"] = True  # one more try at idle
                        ready_delegations.append(msg)
                    await release_client(ws)
                continue

            if msg.get("type") == "frame":
                if msg.get("image") and mode.wants_camera:
                    frame_image = msg["image"]
                    speech_chunks = []
                    await prime(held_audio)
                continue

            if msg.get("type") == "speech_chunk":
                if msg.get("seq") == 0:
                    speech_chunks = []
                if valid_audio(msg.get("audio")):
                    speech_chunks.append(msg["audio"])
                    await prime(held_audio + speech_chunks)
                continue

            interrupted.clear()
            audio_b64s = held_audio + (speech_chunks if msg.get("chunked") else [])
            speech_chunks = []
            if valid_audio(msg.get("audio")):
                audio_b64s.append(msg["audio"])
            image = (msg.get("image") or frame_image) if mode.wants_camera else None
            has_audio = bool(audio_b64s)
            is_flush = msg.get("type") == "flush"

            if not has_audio and not image and not msg.get("text"):
                # Mic glitch (or a flush with nothing held) produced no usable media.
                await release_client(ws)
                drain_ready()  # the floor is provably free here
                continue

            # From here on any failure (malformed WAV included — valid_audio
            # only checks length) must release the client, not kill the
            # session loop.
            p_complete = None  # smart-turn probability, surfaced in the UI
            try:
                # The audio classifier judges completeness before the LLM is
                # involved at all. Incomplete → hold the segments (they stay
                # in the next turn's content AND warm in the cache) and wait.
                # A flush turn skips the check: the client waited out the
                # hold and now wants an answer to whatever we have. Translate
                # mode skips it entirely — an interpreter renders on a short
                # silence, it does not wait out thinking pauses.
                if has_audio and not is_flush and mode.uses_smart_turn:
                    pcm = np.concatenate([wav_to_float32(b) for b in audio_b64s])
                    t0 = time.time()
                    complete, prob = await asyncio.get_event_loop().run_in_executor(
                        None, detector.predict, pcm)
                    decision_s = round(time.time() - t0, 3)
                    p_complete = round(prob, 2)
                    if not complete and not interrupted.is_set():
                        held_audio = audio_b64s
                        # Release the client BEFORE the (slow) cache priming:
                        # until turn_incomplete arrives it can't capture a
                        # resumed utterance, and the flush timer's live-speech
                        # guard can't see the user talking.
                        await send_json(ws, {
                            "type": "turn_incomplete",
                            "decision_s": decision_s, "p_complete": p_complete,
                        })
                        await prime(held_audio)
                        continue

                # Padding the final segment diverges it from its primed bytes
                # on flush turns (≤3s re-prefilled) — the honest-transcript
                # win beats that; the continuation path keeps its cache hits.
                if audio_b64s:
                    audio_b64s[-1] = pad_tail_silence(audio_b64s[-1])
                content = user_content(image, audio_b64s)
                frame_image = None
                held_audio = []

                if mode.name == "translate" and has_audio:
                    instruction = TRANSLATE_PROMPT
                else:
                    instruction = turn_instruction(msg, bool(image), has_audio)
                    if has_audio:
                        instruction += MODE_SUFFIX
                user_msg = {"role": "user", "content": content + [text_part(instruction)]}
                raw_text, tags, pt, no_speech = await run_turn(
                    ws, history + [user_msg], interrupted, active,
                    tts_backend, expect_transcript=has_audio,
                    p_complete=p_complete,
                    control_tags=CONTROL_TAGS,
                    tts_voice=mode.tts_voice,
                    fallback=FLUSH_FALLBACK if is_flush
                    else AUDIO_FALLBACK if has_audio else None)
                if pt:
                    prompt_tokens["last"] = pt
                if raw_text.strip():
                    playing_since["t"] = time.time()  # reply now playing client-side
                if no_speech:
                    # No user words stand behind this turn: a control tag
                    # born from noise must not act (measured live — a breath
                    # transcribed as "translate everything I say" switched
                    # the session into translate mode).
                    unfired = tags
                else:
                    unfired = await spawn_delegations(tags)
                    unfired += await apply_mode_tags(tags)
                remember(user_msg, strip_unfired_tags(raw_text, unfired), no_speech)
            except DISCONNECT_ERRORS:
                raise
            except Exception:
                traceback.print_exc()  # keep the session alive
                await release_client(ws)

            # A result that finished while the floor was busy delivers now.
            drain_ready()
    except DISCONNECT_ERRORS:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        for t in delegation_tasks:
            t.cancel()


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    # localhost, not 0.0.0.0: browsers treat http://localhost as a secure
    # context but not http://0.0.0.0, and without one getUserMedia (mic,
    # camera) doesn't exist.
    uvicorn.run(app, host="localhost", port=port)


if __name__ == "__main__":
    main()
