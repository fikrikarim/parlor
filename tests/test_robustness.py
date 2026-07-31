"""E2E robustness coverage: interrupts, glitch audio, queued
turns, per-connection isolation, context rotation."""

import fixtures as fx
from util import Session, audio, silence_wav_b64


def test_glitch_audio_releases_client(session):
    """A mic glitch (no usable audio) must release the client immediately and
    must not poison the history for later turns."""
    t = session.turn({"audio": silence_wav_b64()}, timeout=15)
    assert t.marker == "released"
    t = session.turn(audio("capital_france"))
    assert t.marker == "complete" and "paris" in t.text.lower()


def test_interrupt_aborts_generation(session):
    """Barge-in: interrupt after first audio → the stream stops without a
    terminal frame, and the next turn works on the same conversation."""
    session.send({"text": "Count from one to thirty, one number per sentence."})
    saw_audio = False
    for _ in range(200):
        msg = session.recv(timeout=30)
        assert msg is not None, "no response before interrupt"
        if msg.get("type") == "audio_chunk":
            saw_audio = True
            break
    assert saw_audio
    session.send({"type": "interrupt"})
    session.drain(quiet_s=3.0)
    t = session.turn(audio("capital_france"))
    assert t.marker == "complete" and "paris" in t.text.lower()


def test_turns_queue_while_processing(session):
    """Talking while the previous turn is still processing: both turns are
    answered, in order."""
    session.send(audio("capital_france"))
    session.send(audio("long_question"))
    t1 = session.collect_turn()
    t2 = session.collect_turn(timeout=120)
    assert t1.marker == "complete" and "paris" in t1.text.lower()
    assert t2.marker == "complete"
    assert "pronunciation" in (t2.transcription or "").lower()


def test_fresh_conversation_per_connection(server):
    """A reload gets a brand-new conversation: no history bleed-through."""
    with Session(server.url) as s:
        t = s.turn(audio("name_intro"))
        assert t.marker == "complete"
    with Session(server.url) as s:
        t = s.turn(audio("name_recall"))
        assert t.marker == "complete"
        assert "willow" not in t.text.lower(), "history leaked across connections"


def test_context_rotation_survives(server, session):
    """Camera turns until the history rotates (the suite server runs with a
    small LLAMA_CTX to make this cheap) — the session must keep working."""
    server.require_managed()
    rotated = False
    # Generous turn budget: how fast history grows depends on the token
    # estimate and CONTEXT_HEADROOM tuning, and the loop exits early the
    # moment rotation fires.
    for _ in range(14):
        t = session.turn({**audio("capital_france"), "image": fx.make_image_b64()}, timeout=120)
        assert t.marker == "complete", f"turn failed before/during rotation: {t}"
        if "dropping" in server.log():
            rotated = True
            break
    assert rotated, "context rotation never triggered"
    t = session.turn(audio("capital_france"))
    assert t.marker == "complete" and "paris" in t.text.lower()
