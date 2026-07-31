"""E2E conversation coverage: turn-taking, chunked speech
overlap, camera grounding, transcript accuracy, response quality.

Not automatable here (needs a browser + live mic): echo /
AEC behavior, VAD feel and barge-in prosody, non-English speech.
"""

import re
import time

import fixtures as fx
import pytest
from util import audio, wer

# Transcripts of clean synthesized speech should be near-verbatim; degraded
# audio gets more slack.
WER_CLEAN = 0.15
WER_DEGRADED = 0.25


def ref_text(name: str) -> str:
    return fx.FIXTURES[name][0]


def assert_spoken_prose(text: str) -> None:
    """Replies are spoken aloud: no leaked tags, no markdown, sane length."""
    assert text.strip(), "empty response"
    assert "###" not in text and "TRANSCRIPT" not in text
    assert "**" not in text and not re.search(r"^\s*[-*•]\s", text, re.M)
    assert len(re.findall(r"[.!?]", text)) <= 8, f"too long for speech: {text!r}"


# ── A. Turn-taking ────────────────────────────────────────────────────────

def test_short_question(session):
    t = session.turn(audio("capital_france"))
    assert t.marker == "complete"
    assert t.audio_chunks > 0
    assert t.p_complete is not None and t.p_complete > 0.5, \
        "turn-confidence missing from the protocol"
    assert wer(ref_text("capital_france"), t.transcription or "") <= WER_CLEAN
    assert "paris" in t.text.lower()
    assert_spoken_prose(t.text)


def test_incomplete_utterance_is_held(session):
    t = session.turn(audio("incomplete_cutoff"))
    assert t.marker == "incomplete"
    assert t.audio_chunks == 0, "spoke a response to an unfinished thought"
    assert t.p_complete is not None and t.p_complete <= 0.5


def test_continuation_merges_held_audio(session):
    t = session.turn(audio("incomplete_cutoff"))
    assert t.marker == "incomplete"
    t = session.turn(audio("continuation"))
    assert t.marker == "complete"
    merged = (t.transcription or "").lower()
    assert "ask" in merged and "paris" in merged, f"lost part of the utterance: {merged!r}"
    # A weather question is delegation bait (DELEGATE_INSTRUCTION): the
    # reply may answer directly or acknowledge and hand it to the reasoner
    # — either proves the model worked from the full merged utterance.
    reply = t.text.lower()
    on_topic = any(w in reply for w in ("weather", "paris", "trip", "forecast"))
    assert on_topic or session.wait_for("delegation_started", timeout=5), reply


@pytest.mark.skip(reason="the TTS fixture has finished-sounding falling prosody, which "
                         "the acoustic classifier correctly reads as complete — real "
                         "hesitation must be judged live")
def test_thinking_pause_is_held(session):
    t = session.turn(audio("thinking_pause"))
    assert t.marker == "incomplete"


def test_flush_answers_a_held_turn(session):
    """After a hold the client flushes; the model answers what it has.
    Covers the false-hold path: a finished question must not die in silence."""
    t = session.turn(audio("long_question_male"))
    assert t.marker == "incomplete", \
        "fixture no longer trips the classifier — pick a voice that does"
    t = session.turn({"type": "flush"})
    assert t.marker == "complete"
    assert wer(ref_text("long_question"), t.transcription or "") <= WER_DEGRADED
    assert_spoken_prose(t.text)


def test_flush_on_genuine_trailoff(session):
    """Flushing a truly unfinished thought: any spoken reply is fine (answer
    or encouragement), but the session must move on cleanly."""
    t = session.turn(audio("incomplete_cutoff"))
    assert t.marker == "incomplete"
    t = session.turn({"type": "flush"})
    assert t.marker == "complete"
    assert_spoken_prose(t.text)
    t = session.turn(audio("capital_france"))
    assert t.marker == "complete" and "paris" in t.text.lower()


# ── B. Speech overlap (chunk streaming during the utterance) ──────────────

def test_chunked_speech_transcript_intact(server, session):
    chunks = fx.load_wav_chunks_b64("long_question")
    primed_before = server.log().count("Primed cache")
    session.send_speech_chunks(chunks)
    t = session.turn({"audio": chunks[-1], "chunked": True}, timeout=120)
    assert t.marker == "complete"
    assert wer(ref_text("long_question"), t.transcription or "") <= WER_CLEAN
    assert_spoken_prose(t.text)
    if server.log_path:
        primed = server.log().count("Primed cache") - primed_before
        assert primed >= len(chunks) - 1, f"only {primed} chunks primed the cache"


# ── D. Camera ─────────────────────────────────────────────────────────────

def test_camera_grounding_with_prefetched_frame(session):
    session.send({"type": "frame", "image": fx.make_image_b64()})
    time.sleep(1.0)
    t = session.turn(audio("describe_scene"))
    assert t.marker == "complete"
    assert any(w in t.text.lower() for w in ("red", "circle", "sky", "green", "blue", "field"))


def test_frame_freshness_across_turns(session):
    session.send({"type": "frame", "image": fx.make_image_b64()})
    time.sleep(1.0)
    t = session.turn(audio("describe_scene"))
    assert t.marker == "complete"
    session.send({"type": "frame", "image": fx.make_image_b64(scene="night")})
    time.sleep(1.0)
    t = session.turn(audio("describe_scene"))
    assert t.marker == "complete"
    assert any(w in t.text.lower() for w in ("moon", "dark", "night", "crescent", "star")), \
        f"described a stale frame: {t.text!r}"


# ── F. Quality ────────────────────────────────────────────────────────────

def test_long_question_transcript_and_response(session):
    t = session.turn(audio("long_question"))
    assert t.marker == "complete"
    assert wer(ref_text("long_question"), t.transcription or "") <= WER_CLEAN
    assert any(w in t.text.lower() for w in ("pronunciation", "pronounce", "speak", "practice"))
    assert_spoken_prose(t.text)


def test_clipped_ending_not_hallucinated(session):
    """The regression that motivated tail-silence padding: audio cut at the
    VAD boundary made the model invent a different question (and answer it)."""
    t = session.turn(audio("capital_france_clipped"))
    if t.marker == "incomplete":
        t = session.turn({"type": "flush"})
    assert t.marker == "complete"
    # Pre-fix this transcribed as "the capital of the United States" (WER 0.5)
    # and answered Washington. Near-miss phonetics ("frant") are acceptable.
    assert wer(ref_text("capital_france"), t.transcription or "") <= WER_DEGRADED, \
        f"hallucinated transcript: {t.transcription!r}"
    assert "paris" in t.text.lower(), f"answered a different question: {t.text!r}"


def test_noisy_audio(session):
    t = session.turn(audio("capital_france_noisy"))
    if t.marker == "incomplete":
        t = session.turn({"type": "flush"})
    assert t.marker == "complete"
    assert "paris" in t.text.lower()


def test_multi_turn_memory(session):
    t = session.turn(audio("name_intro"))
    assert t.marker == "complete"
    t = session.turn(audio("name_recall"))
    assert t.marker == "complete"
    assert "willow" in t.text.lower(), f"forgot the name: {t.text!r}"


def test_text_only_turn(session):
    t = session.turn({"text": "Say hello in one short sentence."})
    assert t.marker == "complete"
    assert t.audio_chunks > 0
    assert_spoken_prose(t.text)
