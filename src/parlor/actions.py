"""The action decider: what the user asked the assistant to DO, decided by
a separate grammar-forced JSON request instead of in-band control tags.

Why a second request (benchmarks/archbench.py, e4b, 19 spoken cases x 2):
in-band tags scored recall 0.955 with the miss being an ack-without-action
("I will be quiet", no tag) — a spoken promise the server never keeps,
the worst failure a voice assistant has. The decoupled head scored recall
1.0, its only error an unwanted-but-cancellable timer misfire (0.062).
Structurally, the head can never leak into speech (there is nothing to
excise), runs at temperature 0 while speech keeps its sampling
temperature, sees the model's own confirmation as evidence, and does the
duration math itself ("twenty minutes" → 1200) in any language. History
stays pure speech — no stored tag can teach the model a state the server
isn't in.

Cost: the head decodes ~35 JSON tokens over the cached prefix (~2s GPU on
e4b, hidden under TTS playback), every turn. A cheaper 1-token yes/no
pre-gate was measured (archbench B_gated): its recall is a perfect 1.0,
but it answers "yes" on nearly every turn — including "how are you" —
so it adds ~0.9s and then runs the head anyway. Not worth its
complexity; revisit only if per-turn GPU cost shows up in practice.

The head must run on the SAME llama-server as speech: a separate model
would pay full prefill of history + audio every turn — the shared prefix
cache is what makes deciding cheap.
"""

import json
from dataclasses import dataclass

from parlor import llama

# One prompt per session mode: the head must know what "no change" means
# ("conversation" reported while already conversing is state, not an
# action) and, mid-translation/listening, what an exit command sounds
# like. The listen wording carries the clearly-ask-YOU nuance measured in
# the e2e suite: a think-aloud quasi-question ("the thing I wanted to ask
# you about is…") is still thinking, not an exit.
_HEAD_COMMON = (
    "System note (not user audio): you are the action decider. From the "
    "user's last message (the assistant's reply above may help), report "
    "what the user asked the assistant to DO, as JSON. timer_seconds: "
    "the countdown duration in seconds if they asked for a timer or a "
    "timed reminder, else 0 — asking the assistant to be quiet for some "
    "time is a mode request, never a timer. timer_label: a "
    "two-or-three-word label for it, else empty. {mode_clause} "
    "research_task: if they asked to "
    "search, look up, or research something, or asked about anything "
    "current or changing (weather, news, prices, scores, openings, "
    "\"right now\", \"today\"), the task restated to stand alone, else "
    "empty. A duration or topic merely mentioned in passing is NOT a "
    "request: report an action only when the user asked for it."
)
_MODE_CLAUSES = {
    "conversation": (
        "mode: the mode they asked to SWITCH TO — 'translate' (translate "
        "everything they say from now on, or interpret two-way between "
        "two languages), 'listen' (any ask for the "
        "assistant to just listen, stay silent or quiet, or stop "
        "responding for a while) — the session is already in normal "
        "conversation, so mode is 'none' unless they asked to change it."),
    "translate": (
        "mode: 'conversation' ONLY if their words are a command to the "
        "assistant to stop translating and go back to normal "
        "conversation (in any language); everything else they say is "
        "content being translated, so mode is 'none'."),
    "listen": (
        "mode: 'conversation' ONLY if their words clearly ask the "
        "assistant to start responding again — like \"okay, I'm done\", "
        "\"what do you think?\", \"stop listening\"; if they are still "
        "thinking out loud, even wondering about something or mentioning "
        "the assistant, mode is 'none'."),
}
# The translate target(s), measured in benchmarks/translatebench.py
# (13/14 with this wording and placement): one entry is a one-way target
# ('english' when unnamed keeps the original behavior), two entries a
# two-way interpreting pair.
_LANG_CLAUSE = (
    " languages: when mode is 'translate', the language or languages "
    "involved, as lowercase English words: ONE entry — the language "
    "everything the user says should be rendered into ('english' if they "
    "didn't name one) — or TWO entries if they asked for two-way "
    "interpreting between two languages (like between English and "
    "Spanish); an empty list when mode is not 'translate'."
)
HEAD_SCHEMA = {
    "type": "object",
    "properties": {
        "timer_seconds": {"type": "integer"},
        "timer_label": {"type": "string"},
        "mode": {"type": "string",
                 "enum": ["none", "translate", "listen", "conversation"]},
        "languages": {"type": "array", "items": {"type": "string"},
                      "maxItems": 2},
        "research_task": {"type": "string"},
    },
    "required": ["timer_seconds", "timer_label", "mode", "languages",
                 "research_task"],
}


@dataclass(frozen=True)
class ActionDecision:
    """What the user asked for, typed. None/empty means 'nothing'."""
    timer: tuple[int, str] | None = None   # (seconds, label)
    mode: str | None = None                # target mode, already ≠ current
    languages: tuple[str, ...] = ()        # translate target(s): 1 one-way, 2 pair
    research: str | None = None            # self-contained task

    def any(self) -> bool:
        return bool(self.timer or self.mode or self.research)


NONE = ActionDecision()


def decide_after(messages: list, current_mode: str) -> ActionDecision:
    """Judge a finished conversation turn (blocking — run in an executor).
    `messages` ends with the assistant reply, byte-identical to the turn
    request plus the reply — the whole prefix is already in the slot
    cache, so only the decider prompt pays prefill. The reply is
    evidence: a confirmation like "three minutes, I'll let you know" is
    exactly what the head keys on."""
    return _decide(lambda prompt: messages + [{"role": "user", "content": prompt}],
                   current_mode)


def decide_before(history: list, content: list, current_mode: str) -> ActionDecision:
    """Judge an utterance BEFORE any reply exists (blocking). Used in
    translate/listen, where what the reply IS depends on the decision
    (content to render, or a command to the assistant). The decider
    prompt rides the same user message as the audio — the production
    instruction shape, and it extends the primed prefix instead of
    opening a second user turn (chat templates dislike consecutive
    same-role messages)."""
    def build(prompt: str) -> list:
        text = {"type": "text", "text": prompt}
        return history + [{"role": "user", "content": content + [text]}]
    return _decide(build, current_mode)


def _decide(build, current_mode: str) -> ActionDecision:
    """One grammar-forced head call. Failures return NONE: a lost
    decision is a no-op turn, never an exception in the turn loop."""
    try:
        head_prompt = (_HEAD_COMMON.format(mode_clause=_MODE_CLAUSES[current_mode])
                       + _LANG_CLAUSE)
        # max_tokens must clear the JSON skeleton (~30 tokens) PLUS a long
        # restated research task — truncated JSON fails json.loads and
        # silently drops an action the reply already promised (review
        # finding). The grammar bounds the shape, so the cap is generous.
        raw = llama.chat_blocking(build(head_prompt), max_tokens=192,
                                  temperature=0.0, json_schema=HEAD_SCHEMA)
        head = json.loads(raw)
    except Exception as e:
        print(f"Action decider failed: {e}")
        return NONE
    timer = None
    seconds = head.get("timer_seconds") or 0
    if isinstance(seconds, int) and seconds > 0:
        timer = (seconds, str(head.get("timer_label", "")).strip())
    target = head.get("mode")
    # Mid-translate/listen the only sanctioned transition is OUT — the
    # clauses ask only about exits, so any other target is a misread
    # (a phantom listen→translate jump would answer aloud mid-listen).
    allowed = (("translate", "listen", "conversation")
               if current_mode == "conversation" else ("conversation",))
    mode = target if target in allowed and target != current_mode else None
    # Normalized translate target(s); order kept (a pair reads naturally
    # in the order it was asked), duplicates and blanks dropped.
    languages: tuple[str, ...] = ()
    if mode == "translate":
        seen: list[str] = []
        for lang in head.get("languages") or []:
            lang = str(lang).strip().lower()
            if lang and lang not in seen:
                seen.append(lang)
        languages = tuple(seen[:2])
    research = str(head.get("research_task", "")).strip() or None
    decision = ActionDecision(timer=timer, mode=mode, languages=languages,
                              research=research)
    if decision.any():
        print(f"Action decision: {decision}")
    return decision
