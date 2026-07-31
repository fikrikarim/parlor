"""E2E transcript stability: the model must transcribe the CURRENT audio,
not hallucinate from conversation context — especially on later turns.

Two mechanisms push a long session toward transcript hallucination:
- every finished turn keeps its raw audio in history, so by turn N the
  model transcribes against a context holding N audio clips and N of its
  own '###TRANSCRIPT:' lines — plenty of material to blend or copy;
- history rotation drops oldest messages, and a dropped user message can
  orphan its assistant reply, leaving a '###TRANSCRIPT:' line whose audio
  no longer exists — an in-context example of transcribing from nothing.

Each turn here uses fixtures with marker words unique across the fixture
set (see benchmarks/fixtures.py), so contamination is detectable: a
transcript containing another turn's marker was written from history, not
from the audio. Early turns double as their own controls — the same
fixtures transcribe cleanly against a near-empty history.
"""

import fixtures as fx
import pytest
from util import audio, norm_words, wer

WER_CLEAN = 0.15

FILLERS = ["filler_color", "filler_pet", "filler_baking",
           "filler_novel", "filler_jazz", "filler_garden"]

# fixture -> words that appear in no other fixture's reference text.
MARKERS = {
    "filler_color": {"turquoise"},
    "filler_pet": {"biscuit", "retriever"},
    "filler_baking": {"sourdough", "rosemary"},
    "filler_novel": {"lisbon", "violin"},
    "filler_jazz": {"jazz"},
    "filler_garden": {"basil", "tomatoes"},
    "capital_france": {"france"},
    "long_question": {"pronunciation"},
}


def resolved_turn(session, payload, timeout=120):
    """A turn as the browser drives it: a held (incomplete) utterance is
    flushed, so every call ends in a terminal answer."""
    t = session.turn(payload, timeout=timeout)
    if t.marker == "incomplete":
        t = session.turn({"type": "flush"}, timeout=timeout)
    return t


def foreign_markers(name: str, transcript: str | None) -> set:
    """Marker words in the transcript that belong to OTHER fixtures."""
    words = set(norm_words(transcript or ""))
    return {m for other, marks in MARKERS.items() if other != name
            for m in words & marks}


def test_transcripts_stay_faithful_as_audio_accumulates(server, session):
    """Six distinct utterances, then a probe whose fresh-session accuracy
    the suite already asserts (test_short_question): every turn must
    transcribe as cleanly late in the session as it does at turn one, with
    no words borrowed from earlier turns. Rotation must NOT fire here —
    this test measures accumulation, so history has to actually grow."""
    dropped_before = server.log().count("dropping")
    rows, failures = [], []
    for i, name in enumerate([*FILLERS, "capital_france"]):
        t = resolved_turn(session, audio(name))
        w = wer(fx.FIXTURES[name][0], t.transcription or "")
        rows.append(f"turn {i + 1} {name}: marker={t.marker} wer={w} "
                    f"heard={t.transcription!r}")
        if t.marker != "complete":
            failures.append(f"turn {i + 1} ({name}): no answer ({t.marker})")
            continue
        if w > WER_CLEAN:
            failures.append(f"turn {i + 1} ({name}): wer {w} > {WER_CLEAN}")
        hits = foreign_markers(name, t.transcription)
        if hits:
            failures.append(f"turn {i + 1} ({name}): transcript contains other "
                            f"turns' words {sorted(hits)}")
    detail = "\n".join(failures + ["", "session:"] + rows)
    assert not failures, f"transcripts drifted on later turns:\n{detail}"
    if server.log_path:
        assert server.log().count("dropping") == dropped_before, \
            "history rotated mid-test — it must accumulate here, or this " \
            "test no longer measures what it claims (check CONTEXT_HEADROOM " \
            "against the suite's LLAMA_CTX)"


def test_nonspeech_audio_never_becomes_user_words(server, session):
    """A VAD false trigger (breath, cough, room noise) must not put words
    in the user's mouth, leak prompt text into speech, fire actions, or
    poison the turns after it. All of these happened before the no-speech
    escape and the storage/tag guards: at temp 0 a fresh-session breath
    got 'CRIPT: Begin your reply with one line:' spoken aloud and a
    mid-session breath displayed '[Silence]' as the user's words; at temp
    0.7 a breath transcribed as an earlier turn's question (re-answered
    verbatim) or as 'translate everything I say from now on' — which
    switched the session into translate mode."""
    modes_before = server.log().count("Mode →")
    for name in FILLERS[:3]:
        assert resolved_turn(session, audio(name)).marker == "complete"
    t = resolved_turn(session, {"audio": fx.breath_wav_b64()})
    assert not t.transcription, \
        f"non-speech audio shown as user words: {t.transcription!r}"
    assert ("transcript" not in t.text.lower() and "##" not in t.text
            and "begin your reply" not in t.text.lower()), \
        f"prompt text leaked into speech: {t.text!r}"
    # The turn after must be pristine — one stored degenerate turn used to
    # poison everything downstream.
    t2 = resolved_turn(session, audio("capital_france"))
    assert t2.marker == "complete"
    assert wer(fx.FIXTURES["capital_france"][0], t2.transcription or "") <= WER_CLEAN, \
        f"turn after non-speech audio drifted: {t2.transcription!r}"
    # The model may answer itself or hand the question to the reasoner —
    # either proves it heard the real question, not a poisoned history.
    assert ("paris" in t2.text.lower()
            or session.wait_for("delegation_started", timeout=5)), t2.text
    if server.log_path:
        assert server.log().count("Mode →") == modes_before, \
            "non-speech audio switched the session mode"


def test_transcripts_stay_faithful_after_rotation(server, session):
    """Drive history past a rotation (camera turns inflate the token
    estimate fastest), then check a fresh utterance still transcribes
    cleanly. Guards the rotation path: dropping oldest messages must not
    leave history that teaches the model to transcribe from context."""
    server.require_managed()
    dropped_before = server.log().count("dropping")
    for i in range(14):
        t = resolved_turn(session, {**audio("capital_france"),
                                    "image": fx.make_image_b64()})
        assert t.marker == "complete", f"turn {i + 1} failed before rotation"
        if server.log().count("dropping") > dropped_before:
            break
    else:
        pytest.fail("rotation never triggered — raise the turn count or "
                    "shrink the suite's LLAMA_CTX")
    t = resolved_turn(session, audio("long_question"))
    assert t.marker == "complete"
    w = wer(fx.FIXTURES["long_question"][0], t.transcription or "")
    hits = foreign_markers("long_question", t.transcription)
    assert w <= WER_CLEAN and not hits, \
        (f"post-rotation transcript drifted: wer={w} "
         f"borrowed={sorted(hits)} heard={t.transcription!r}")
