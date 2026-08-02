"""Session modes: what kind of interaction the session is in decides how
turns are gated and prompted. The default is open conversation; 'translate'
turns Parlor into a consecutive interpreter — every utterance rendered in
English on a short silence window, no conversational replies; 'listen'
turns it into a silent scribe — every utterance transcribed, nothing
spoken back until the user asks for a reply again.

A mode is data, not behavior: server.py consults these flags instead of
hardcoding, so a future mode (an interpreter pair, camera narration) is a
new entry here plus whatever new trigger it needs, not a rewrite of the
turn loop. Switching is decided by the action head (actions.py) judging
each utterance, with a UI escape hatch (the client's mode chip sends
set_mode directly).
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Mode:
    name: str
    uses_smart_turn: bool     # acoustic completeness gate + hold/flush
    allows_delegation: bool   # background-reasoner tag and deliveries
    wants_camera: bool        # accept + cache-prime camera frames
    wants_time_note: bool     # elapsed-quiet note on the turn instruction
    speaks_fallback: bool     # a turn that yields no speech still says a line
    tts_voice: str            # per-mode Kokoro voice (carries the language)
    languages: tuple[str, ...] = ()  # translate target(s): 1 one-way, 2 pair


MODES = {m.name: m for m in [
    Mode("conversation", uses_smart_turn=True, allows_delegation=True,
         wants_camera=True, wants_time_note=True, speaks_fallback=True,
         tts_voice="af_heart"),
    # An interpreter waits only for a short silence, never for a "complete
    # thought" — a mid-sentence thinking pause must not stall translation —
    # and never mixes research or camera chatter into the rendering. No time
    # note either: an interpreter would render it into the translation.
    Mode("translate", uses_smart_turn=False, allows_delegation=False,
         wants_camera=False, wants_time_note=False, speaks_fallback=True,
         tts_voice="af_heart", languages=("english",)),
    # A silent scribe: the user thinks out loud, Parlor transcribes and
    # stays quiet until spoken TO (the exit path lives in LISTEN_PROMPT).
    # VAD-only segmentation — nothing gets answered, so utterance
    # completeness never matters, and a silent turn is the POINT, so it
    # gets no fallback line. Research deliveries park like translate; the
    # time note stays, feeding the exit question ("how long was I at it?").
    Mode("listen", uses_smart_turn=False, allows_delegation=False,
         wants_camera=False, wants_time_note=True, speaks_fallback=False,
         tts_voice="af_heart"),
]}

# Kokoro voice per one-way translate target. Only languages whose G2P
# works on a stock install are mapped (Japanese/Chinese voices need the
# misaki[ja]/[zh] extras); anything unmapped falls back to the English
# voice — accented, but never a crash (and pipeline.py skips sentences a
# voice can't phonemize at all).
TRANSLATE_VOICES = {"english": "af_heart", "spanish": "ef_dora",
                    "french": "ff_siwis", "italian": "if_sara",
                    "portuguese": "pf_dora", "hindi": "hf_alpha"}


def translate_mode(languages: tuple[str, ...] = ()) -> Mode:
    """The translate mode parameterized by its target(s): one language is
    one-way rendering, two an interpreting pair (the model picks the
    direction per utterance — benchmarks/translatebench.py). Voice
    policy: a pair keeps the English voice for both directions (decided
    for English pairs — Kokoro's strongest voice — and the least-bad
    default for the rest); a one-way target gets its own voice."""
    languages = tuple(languages) or ("english",)
    voice = ("af_heart" if len(languages) > 1
             else TRANSLATE_VOICES.get(languages[0], "af_heart"))
    return replace(MODES["translate"], languages=languages, tts_voice=voice)
