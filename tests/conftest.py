"""E2E suite server management.

By default each pytest session spawns its own server (real llama.cpp, TTS,
turn detector) on a dedicated port with a small context window so history
rotation is cheap to trigger, and captures its log for assertions. Set
PARLOR_TEST_URL=ws://host:port/ws to test an already-running server instead
(log- and process-based tests skip).
"""

import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

# Captured BEFORE any parlor import runs load_dotenv() (importing fixtures
# below pulls in parlor.tts, and test modules import parlor.server during
# collection): suite overrides must come from the shell, and once .env has
# been folded into os.environ the two are indistinguishable.
SHELL_ENV = dict(os.environ)

import fixtures  # resolved via pythonpath = ["benchmarks"] (pyproject.toml)

ROOT = Path(__file__).resolve().parent.parent

TEST_PORT = 8821
TEST_LLAMA_PORT = 8822
TEST_REASONER_PORT = 8823
STARTUP_TIMEOUT_S = 300

# Distinctive so a delivery turn provably carries the reasoner's facts.
# The stalled weather task gets a weather-shaped answer: the delivery
# prompt demands word-for-word relay, and a mismatched answer (pizza for
# a weather question) makes the model reject it and hallucinate instead.
MOCK_ANSWER = ("The clear winner right now is Pizzarium Bonci near the "
               "Vatican, famous for its crispy square pizza slices.")
MOCK_WEATHER_ANSWER = ("The weather in Naples right now is sunny and "
                       "twenty-nine degrees, with clear skies all afternoon.")


class _ReasonerHandler(http.server.BaseHTTPRequestHandler):
    """OpenAI-compatible chat endpoint the suite's server delegates to.
    Task-text triggers keep the paths deterministic: 'stock' → 500 (the
    failure path, see delegate_stock), 'naples' → an 8s stall (the
    conversation-continues-while-pending path, see delegate_naples). It
    also insists on the real request shape — a dropped auth header or a
    wrong path must fail loudly, not pass silently."""

    def do_POST(self):
        if (not self.path.endswith("/chat/completions")
                or not self.headers.get("Authorization", "").startswith("Bearer ")):
            self.send_response(401)
            self.end_headers()
            return
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append(json.loads(raw))
        low = raw.lower()
        if b"stock" in low:
            self.send_response(500)
            self.end_headers()
            return
        answer = MOCK_ANSWER
        if b"naples" in low:
            time.sleep(8)
            answer = MOCK_WEATHER_ANSWER
        payload = json.dumps(
            {"choices": [{"message": {"content": answer}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture(scope="session")
def reasoner_mock():
    if os.environ.get("PARLOR_TEST_URL"):
        yield None  # the external server has its own reasoner config
        return
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", TEST_REASONER_PORT)) == 0:
            pytest.fail(f"port {TEST_REASONER_PORT} is already in use — kill "
                        f"the stale mock (lsof -ti :{TEST_REASONER_PORT}) and rerun")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", TEST_REASONER_PORT),
                                          _ReasonerHandler)
    srv.requests = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


@dataclass
class Server:
    url: str
    proc: subprocess.Popen | None = None
    log_path: Path | None = None

    def log(self) -> str:
        return self.log_path.read_text() if self.log_path else ""

    def require_managed(self) -> None:
        if self.proc is None:
            pytest.skip("needs a suite-managed server (unset PARLOR_TEST_URL)")


@pytest.fixture(scope="session")
def server(tmp_path_factory, reasoner_mock) -> Server:
    fixtures.generate_all()

    external = os.environ.get("PARLOR_TEST_URL")
    if external:
        yield Server(url=external)
        return

    # A stale process on either port would make the readiness probe pass
    # against the WRONG server and the whole run would test old code.
    # fail (not exit): the server-backed tests error out, but the fast
    # parser unit tests have no server dependency and must keep running.
    for port in (TEST_PORT, TEST_LLAMA_PORT):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                pytest.fail(f"port {port} is already in use — kill the stale "
                            f"server (lsof -ti :{port}) and rerun")

    log_path = tmp_path_factory.mktemp("server") / "server.log"
    # MODEL pinned so changing the product default can't silently change
    # what the suite measures ("e2b" also passes and is ~2x faster if you
    # are iterating on tests). The reasoner points at the suite's mock, so
    # every e2e test runs under the delegation-enabled system prompt —
    # exactly what production runs when a key is configured.
    # TEMPERATURE pinned to 0: the suite verifies machinery and must be
    # deterministic — at 0.7 every delegation/mode-switch ask is a ~0.8
    # coin flip and some run always loses one. Production recall at real
    # temperature is tagbench --production's job, not the suite's.
    env = {**os.environ, "PORT": str(TEST_PORT), "LLAMA_PORT": str(TEST_LLAMA_PORT),
           "LLAMA_CTX": "4096", "MODEL": SHELL_ENV.get("MODEL", "e4b"),
           "TEMPERATURE": SHELL_ENV.get("TEMPERATURE", "0"),
           "REASONER_BASE_URL": f"http://127.0.0.1:{TEST_REASONER_PORT}/v1",
           "REASONER_API_KEY": "test-key", "REASONER_MODEL": "mock-model",
           "REASONER_TIMEOUT": "20"}
    # A developer's .env must not change what the suite measures: every
    # model/backend key is taken from SHELL_ENV (deliberate override) or
    # pinned, never from .env — which by now is inside os.environ and
    # would otherwise leak through the spread above. Set to "" rather
    # than popped: the child runs load_dotenv() itself, which fills in
    # missing keys from .env but never overrides an existing (even
    # empty) one. LLAMA_SERVER_URL is cleared unconditionally — the
    # suite owns its server topology (external servers go through
    # PARLOR_TEST_URL instead).
    for k in ("MODEL_PATH", "MMPROJ_PATH", "KOKORO_ONNX"):
        env[k] = SHELL_ENV.get(k, "")
    env["LLAMA_SERVER_URL"] = ""
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "parlor.server"], cwd=ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)

    def shutdown():
        # server.py's lifespan stops its llama-server child on SIGTERM, but a
        # crashed server orphans it — sweep the llama port regardless.
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["bash", "-c", f"kill $(lsof -ti :{TEST_LLAMA_PORT}) 2>/dev/null"])

    try:
        deadline = time.time() + STARTUP_TIMEOUT_S
        while True:
            if proc.poll() is not None:
                pytest.fail(f"server exited during startup:\n{log_path.read_text()[-3000:]}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{TEST_PORT}/", timeout=2)
                break
            except OSError:
                if time.time() > deadline:
                    pytest.fail(f"server not ready in {STARTUP_TIMEOUT_S}s:\n"
                                f"{log_path.read_text()[-3000:]}")
                time.sleep(2)

        yield Server(url=f"ws://127.0.0.1:{TEST_PORT}/ws", proc=proc, log_path=log_path)
    finally:
        shutdown()


@pytest.fixture
def session(server):
    from util import Session

    with Session(server.url) as s:
        yield s
