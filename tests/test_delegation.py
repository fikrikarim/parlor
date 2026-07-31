"""Delegation e2e: the model hands research tasks to the background
reasoner, and the answer comes back as a proactive spoken turn. The
reasoner is the deterministic mock in conftest.py — no network, no API
key — so these tests exercise the full production path: audio in →
<delegate> tag → background HTTP call → idle-gated delivery turn.
"""

import util
from conftest import MOCK_ANSWER


def delegate_turn(session, fixture):
    """Send a delegate-worthy utterance; at temperature 0.7 the model
    occasionally acks without emitting the tag. The retry uses an
    ALTERNATE phrasing ({fixture}_alt): re-sending identical audio tends
    to reproduce the identical tagless completion (cached prefix), so
    only different words give an independent sample. Two misses across
    two phrasings of a MUST-delegate ask is a real regression."""
    for name in (fixture, f"{fixture}_alt"):
        t = session.turn(util.audio(name))
        if t.marker == "incomplete":
            # smart-turn held it: flush rather than re-speak, which would
            # merge two copies of the utterance into one turn.
            t = session.turn({"type": "flush"})
        started = session.wait_for("delegation_started", timeout=10)
        if started:
            return t, started
    raise AssertionError(
        f"model never delegated {fixture!r} (either phrasing) — last ack {t.text!r}")


def test_research_question_is_delegated_and_delivered(server, session, reasoner_mock):
    server.require_managed()  # the mock env is wired by the suite's server
    requests_before = len(reasoner_mock.requests)
    t, started = delegate_turn(session, "delegate_pizza")
    # The acknowledgment turn speaks, and no tag markup may reach the client.
    assert t.marker == "complete", t
    assert "<" not in t.text and "delegate" not in t.text.lower()
    assert started["task"].strip()

    resolved = session.wait_for("delegation_resolved", timeout=30)
    assert resolved and resolved["ok"] is True

    delivery = session.collect_turn(timeout=60)
    assert delivery.marker == "complete", delivery  # spoken, not just text
    # The reasoner's facts survived Gemma's near-verbatim relay.
    assert "bonci" in delivery.text.lower() or "pizzarium" in delivery.text.lower(), (
        f"delivery lost the answer: {delivery.text!r} (mock said: {MOCK_ANSWER!r})")
    # This turn's task reached the mock (the list outlives other tests).
    assert len(reasoner_mock.requests) > requests_before
    assert "pizza" in str(reasoner_mock.requests[-1]).lower()


def test_conversation_continues_while_delegation_runs(server, session, reasoner_mock):
    server.require_managed()
    delegate_turn(session, "delegate_naples")  # mock stalls this task 8s

    # Ask something else while the background task is still running: the
    # answer must come back BEFORE the delegation delivery.
    t2 = session.turn(util.audio("capital_france"))
    assert t2.marker == "complete" and "paris" in t2.text.lower()

    resolved = session.wait_for("delegation_resolved", timeout=30)
    assert resolved and resolved["ok"] is True
    delivery = session.collect_turn(timeout=60)
    assert delivery.marker == "complete", delivery
    assert "twenty-nine" in delivery.text.lower() or "sunny" in delivery.text.lower(), (
        f"delivery lost the answer: {delivery.text!r}")


def test_failed_delegation_is_spoken_not_silent(server, session):
    server.require_managed()
    delegate_turn(session, "delegate_stock")  # 'stock' → mock 500s

    resolved = session.wait_for("delegation_resolved", timeout=30)
    assert resolved and resolved["ok"] is False

    apology = session.collect_turn(timeout=60)
    assert apology.marker == "complete" and apology.text.strip(), apology

    # The session survives a failed delegation.
    t2 = session.turn(util.audio("capital_france"))
    assert t2.marker == "complete"
    assert "paris" in t2.text.lower()


def test_plain_question_is_not_delegated(server, session):
    server.require_managed()
    t = session.turn(util.audio("capital_france"))
    assert "paris" in t.text.lower()
    assert session.wait_for("delegation_started", timeout=3) is None
