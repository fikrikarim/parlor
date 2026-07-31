"""Destructive last test: llama-server dies mid-session.

Runs last (file name order) because it kills the suite server's llama.cpp
child; every turn after that can only fail gracefully.
"""

import subprocess

from util import audio


def test_llama_death_fails_gracefully(server, session):
    server.require_managed()
    out = subprocess.run(["pgrep", "-P", str(server.proc.pid), "llama-server"],
                         capture_output=True, text=True).stdout.split()
    assert out, "could not find the spawned llama-server"
    subprocess.run(["kill", "-9", out[0]])

    t = session.turn(audio("capital_france"), timeout=30)
    assert t.marker == "released", "client left hanging after llama-server died"
    # The server itself must survive to release the next attempt too.
    t = session.turn(audio("capital_france"), timeout=30)
    assert t.marker == "released"
