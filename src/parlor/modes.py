"""Session modes: what kind of interaction the session is in decides how
turns are gated and prompted. The default is open conversation; 'translate'
turns Parlor into a consecutive interpreter — every utterance is rendered
in English on a short silence window, with no conversational replies.

A mode is data, not behavior: server.py consults these flags instead of
hardcoding, so a future mode (an interpreter pair, camera narration) is a
new entry here plus whatever new trigger it needs, not a rewrite of the
turn loop. Switching is model-driven — the voice model emits a
<mode>name</mode> control tag when the user asks — with a UI escape hatch
(the client's mode chip sends set_mode directly).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    name: str
    uses_smart_turn: bool     # acoustic completeness gate + hold/flush
    allows_delegation: bool   # background-reasoner tag and deliveries
    wants_camera: bool        # accept + cache-prime camera frames
    tts_voice: str            # per-mode Kokoro voice (carries the language)


MODES = {m.name: m for m in [
    Mode("conversation", uses_smart_turn=True, allows_delegation=True,
         wants_camera=True, tts_voice="af_heart"),
    # An interpreter waits only for a short silence, never for a "complete
    # thought" — a mid-sentence thinking pause must not stall translation —
    # and never mixes research or camera chatter into the rendering.
    Mode("translate", uses_smart_turn=False, allows_delegation=False,
         wants_camera=False, tts_voice="af_heart"),
]}
