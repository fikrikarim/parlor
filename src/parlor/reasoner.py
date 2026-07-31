"""Background reasoner: hands a task from the voice model to a frontier
text model on any OpenAI-compatible chat endpoint and returns a short
speakable answer.

Disabled unless REASONER_API_KEY is set — without it the delegation
instruction is simply absent from the system prompt and Parlor stays
fully on-device. Defaults to OpenRouter, where REASONER_WEB_SEARCH=1
(the default) appends the ':online' suffix so any model gets
provider-side web search; other endpoints (a direct provider URL, a
second local llama-server) ignore the flag.
"""

import json
import os
import urllib.request

BASE_URL = os.environ.get("REASONER_BASE_URL",
                          "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.environ.get("REASONER_API_KEY", "")
MODEL = os.environ.get("REASONER_MODEL", "anthropic/claude-sonnet-4.5")
try:
    TIMEOUT_S = float(os.environ.get("REASONER_TIMEOUT", "90"))
except ValueError:
    print("REASONER_TIMEOUT is not a number — using 90s")
    TIMEOUT_S = 90.0

# On OpenRouter web search is requested through the model id itself, so it
# has to be baked in here; other endpoints would reject the suffix.
if (os.environ.get("REASONER_WEB_SEARCH", "1") == "1"
        and "openrouter.ai" in BASE_URL and not MODEL.endswith(":online")):
    MODEL += ":online"

# The answer is relayed aloud by the voice model near-verbatim, so it must
# already be speech-shaped when it comes back.
SYSTEM_PROMPT = (
    "You are the background research assistant behind a real-time voice "
    "AI. Answer the task directly, in plain conversational text that will "
    "be read aloud: no markdown, no lists, no headings, no URLs. Lead "
    "with the answer, keep it to 2-6 short sentences, and include the "
    "concrete facts (names, numbers, places) the task asks for."
)


def enabled() -> bool:
    return bool(API_KEY)


def ask(task: str) -> str:
    """Blocking chat-completion call — run it in an executor. Raises on
    transport/HTTP errors and empty answers; the caller turns any failure
    into a spoken apology."""
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": task}],
            "max_tokens": 1024,  # the answer is spoken aloud — cap essays
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        body = json.load(resp)
    answer = (body["choices"][0]["message"]["content"] or "").strip()
    if not answer:
        raise RuntimeError("reasoner returned an empty answer")
    return answer
