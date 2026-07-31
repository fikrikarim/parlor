"""rotate_history unit tests — no server, runs in milliseconds.

Rotation is pure list logic but only reachable e2e after ~15 real-model
turns, so its invariants are pinned here: the system prompt survives
exactly once, the kept slice starts on a user message (an orphaned
assistant reply describes words that no longer exist — an in-context
lesson in transcribing from nothing), and tiny histories pass through
untouched (slicing a bare [system] would duplicate the system prompt).
"""

import pytest

from parlor.server import rotate_history


def make_history(pairs: int) -> list:
    h = [{"role": "system", "content": "sys"}]
    for i in range(pairs):
        h.append({"role": "user", "content": f"u{i}"})
        h.append({"role": "assistant", "content": f"a{i}"})
    return h


@pytest.mark.parametrize("pairs", range(2, 12))
def test_rotation_never_orphans_a_reply(pairs):
    h = rotate_history(make_history(pairs))
    assert h[0]["role"] == "system"
    assert h[1]["role"] == "user", f"orphaned assistant at {pairs} pairs: {h[1]}"
    roles = [m["role"] for m in h[1:]]
    assert roles == ["user", "assistant"] * (len(roles) // 2)


def test_rotation_drops_oldest_and_keeps_newest():
    h = make_history(8)
    out = rotate_history(h)
    assert len(out) < len(h)
    assert out[-1] == h[-1] and out[1]["role"] == "user"
    assert sum(m["role"] == "system" for m in out) == 1


@pytest.mark.parametrize("pairs", [0, 1])
def test_tiny_history_is_untouched(pairs):
    # Nothing droppable — and a sliced bare [system] would come back as
    # [system, system], growing a copy per turn.
    h = make_history(pairs)
    assert rotate_history(h) == h
