"""StreamParser unit tests — no server, runs in milliseconds.

The parser consumes a partially-streamed buffer, so every case is exercised
at delta boundaries too: whole-string parsing can hide bugs that only occur
when the tag, colon, or newline arrives as its own token (which is exactly
how llama.cpp streams them).
"""

import pytest

from parlor.pipeline import StreamParser, echoes_instruction


def run(deltas, expect_transcript=True):
    """-> (spoken sentences, transcript, unspoken response text)"""
    p = StreamParser(expect_transcript)
    spoken = []
    for d in deltas:
        spoken += p.feed(d)
    tail, transcript = p.finalize()
    return spoken + tail, transcript, p.response


CANONICAL = "###TRANSCRIPT: What is the capital of France?\nParis is the capital. It is lovely."


def char_deltas(text, n=3):
    return [text[i:i + n] for i in range(0, len(text), n)]


@pytest.mark.parametrize("deltas", [
    [CANONICAL],                # single delta
    char_deltas(CANONICAL),     # arbitrary 3-char boundaries
    ["###", "TRANSCRIPT", ":", " What is the capital of France?", "\n",
     "Paris is the capital.", " It is lovely."],  # per-token, colon alone
], ids=["batch", "char3", "tokenwise"])
def test_canonical_forms(deltas):
    spoken, transcript, _ = run(deltas)
    assert transcript == "What is the capital of France?"
    assert spoken == ["Paris is the capital.", "It is lovely."]


def test_tag_with_spaces_around_colon():
    spoken, transcript, _ = run(["### TRANSCRIPT : hello there\n", "Hi. "])
    assert transcript == "hello there"
    assert spoken == ["Hi."]


def test_newline_directly_after_tag_does_not_empty_transcript():
    # A "\n" delta right after the tag must not terminate an empty
    # transcript line and read the user's words back through TTS.
    spoken, transcript, _ = run(["###TRANSCRIPT:", "\n",
                                 "What is the capital of France?", "\n",
                                 "Paris is the capital."])
    assert transcript == "What is the capital of France?"
    assert spoken == ["Paris is the capital."]


def test_missing_newline_does_not_swallow_the_reply():
    # Model ran the reply onto the tag line (or the stream truncated):
    # first sentence is the transcript, the rest must still be spoken.
    spoken, transcript, _ = run(["###TRANSCRIPT: What is the capital of France? ",
                                 "Paris is the capital of France."])
    assert transcript == "What is the capital of France?"
    assert spoken == ["Paris is the capital of France."]


def test_missing_tag_falls_back_to_plain_response():
    spoken, transcript, _ = run(["The capital of France is Paris. ", "It is lovely. "])
    assert transcript is None
    assert spoken == ["The capital of France is Paris.", "It is lovely."]


def test_stray_text_before_tag_becomes_response_prefix():
    spoken, transcript, _ = run(["Sure! ###TRANSCRIPT: hi\n", "Hello. "])
    assert transcript == "hi"
    assert spoken == ["Sure!", "Hello."]


def test_tag_split_mid_word():
    spoken, transcript, _ = run(["##", "#TRANS", "CRIPT", ":", " hi", "\n", "Hello. "])
    assert transcript == "hi"
    assert spoken == ["Hello."]


def test_no_transcript_mode_streams_directly_and_cuts_imitated_tag():
    spoken, transcript, _ = run(["I see a red circle. ", "###TRANSCRIPT: fake"],
                                expect_transcript=False)
    assert transcript is None
    assert spoken == ["I see a red circle."]
    assert not any("#" in s for s in spoken)


def test_imitated_markup_in_response_is_never_spoken():
    spoken, transcript, _ = run(["###TRANSCRIPT: hi\n", "Hello. ", "## Notes: stuff. "])
    assert transcript == "hi"
    assert spoken == ["Hello."]


def test_runaway_transcript_line_is_cut_not_hoarded():
    # No newline, >600 chars: the parser must give TTS something instead of
    # buffering the entire generation as "transcript". The cut lands
    # wherever the buffer crossed the bound — bounded, not pretty.
    long_line = "###TRANSCRIPT: " + ("word " * 130) + "end? And here is the reply. "
    spoken, transcript, _ = run(char_deltas(long_line, 20))
    assert transcript and len(transcript) <= 650
    assert any("reply" in s for s in spoken)


def test_streaming_matches_batch_for_every_split_point():
    # The systemic check: no single split boundary may change the result.
    batch = run([CANONICAL])
    for i in range(1, len(CANONICAL)):
        assert run([CANONICAL[:i], CANONICAL[i:]]) == batch, f"diverged at split {i}"


# ── instruction-echo guard ────────────────────────────────────────────────
# Live bug: on a flush turn the model sometimes echoes the instruction
# text into its ###TRANSCRIPT: line, and the client displays it as the
# user's words. The guard suppresses any transcript sharing a 5-word run
# with the turn's instruction.

def test_instruction_echo_is_detected_against_production_prompts():
    from parlor import server
    flush = server.FLUSH_PROMPT.format(camera="")
    # The observed leak: instruction text verbatim (with or without the
    # model tacking invented words on the end).
    assert echoes_instruction("The user paused mid-thought, so on a new line:", flush)
    assert echoes_instruction(
        "The user paused mid-thought, so on a new line: hello there", flush)
    respond = server.RESPOND_PROMPT.format(camera="")
    assert echoes_instruction(
        "followed by the exact words the user said in their audio message", respond)


def test_genuine_transcripts_pass_the_echo_guard():
    from parlor import server
    for prompt in (server.FLUSH_PROMPT.format(camera=""),
                   server.RESPOND_PROMPT.format(camera=""),
                   server.TRANSLATE_PROMPT.format(language="English"),
                   server.TWO_WAY_PROMPT.format(a="English", b="Spanish")):
        for said in ("What is the capital of France?",
                     "So the thing I wanted to ask you about is the weather "
                     "in Paris for my trip next week.",
                     "I have been trying to learn English for a few months now.",
                     "Please respond to them, all of them, by email."):
            assert not echoes_instruction(said, prompt), (said, prompt[:40])
    # Too short to judge stays visible.
    assert not echoes_instruction("On a new line", "so on a new line: x")


def test_xml_like_text_is_ordinary_speech():
    # Replies are pure speech (actions live in actions.py): an XML-looking
    # element is just text; the ##-markup cut still applies.
    spoken, _, _ = run(["###TRANSCRIPT: hi\n", "Hello. ",
                        "<delegate>x</delegate> ## notes"])
    assert spoken == ["Hello.", "<delegate>x</delegate>"]


def test_no_speech_annotations_are_not_user_words():
    # A transcript that is entirely a bracketed annotation reports that
    # there were no words — shown/stored as user speech it becomes a
    # hallucination ('[Silence]' appeared as a user bubble, live).
    from parlor.pipeline import NO_SPEECH_RE
    for t in ["(no speech)", "(No Speech)", "no speech", "No speech.",
              "(noise)", "[Silence]", "*sigh*", "(background noise)",
              "(static hum)", "[inaudible]"]:
        assert NO_SPEECH_RE.match(t), t
    # Real utterances — including ones that merely start with or contain
    # such words — must pass through untouched.
    for t in ["Silence is golden, don't you think?",
              "What is the capital of France?",
              "I heard a noise in the garden (I think).",
              "(", "(a very long parenthesized ramble that goes on and on "
              "far past any plausible annotation length, word after word)"]:
        assert not NO_SPEECH_RE.match(t), t


def test_quoted_prompt_phrases_are_not_echoes():
    # The translate prompt QUOTES the exit phrases users actually say — a
    # genuine "go back to normal conversation" must not read as an
    # instruction echo (pre-fix it did, and the no-transcript tag gate
    # then dropped the <mode>conversation</mode> exit).
    from parlor import server
    said = "Okay, stop translating now and go back to normal conversation."
    assert not echoes_instruction(said, server.TRANSLATE_PROMPT.format(language="English"))
    # Unquoted instruction prose still reads as an echo.
    assert echoes_instruction(
        "If the audio has no clear words write no speech instead",
        server.TRANSLATE_PROMPT.format(language="English"))
