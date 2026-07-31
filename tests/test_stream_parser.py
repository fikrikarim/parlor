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


# ── control tags ──────────────────────────────────────────────────────────
# Why XML elements rather than ###NAME: lines: see TagFilter in pipeline.py.

TAGS = ("delegate", "mode")


def run_tags(deltas, expect_transcript=True):
    """-> (spoken sentences, transcript, extracted tags)"""
    p = StreamParser(expect_transcript, control_tags=TAGS)
    spoken = []
    for d in deltas:
        spoken += p.feed(d)
    tail, transcript = p.finalize()
    return spoken + tail, transcript, p.tags


DELEGATED = ("###TRANSCRIPT: What's the best pizza in Rome?\n"
             "Great question — let me dig into that. "
             "<delegate>best pizza places in Rome right now</delegate>")


@pytest.mark.parametrize("deltas", [
    [DELEGATED],
    char_deltas(DELEGATED),
    ["###TRANSCRIPT: What's the best pizza in Rome?", "\n",
     "Great question — let me dig into that.", " <", "delegate", ">",
     "best pizza places", " in Rome right now", "</", "delegate", ">"],
], ids=["batch", "char3", "tokenwise"])
def test_delegate_tag_extracted_never_spoken(deltas):
    spoken, transcript, tags = run_tags(deltas)
    assert transcript == "What's the best pizza in Rome?"
    assert spoken == ["Great question — let me dig into that."]
    assert tags == [("DELEGATE", "best pizza places in Rome right now")]


def test_open_tag_value_is_never_extracted_or_spoken_early():
    # The value may still be streaming: extracting at the first feed would
    # fire a delegation with half the task, and none of it may reach TTS.
    p = StreamParser(control_tags=TAGS)
    assert p.feed("###TRANSCRIPT: hi\nOk. <delegate>first half") == ["Ok."]
    assert p.tags == []
    p.feed(" second half</delegate>")
    assert p.tags == [("DELEGATE", "first half second half")]


def test_unclosed_tag_at_stream_end_fires_but_is_never_spoken():
    # The model often hits EOS before the close tag (measured live: a
    # third of '<mode>conversation' exits ended unclosed). At end of
    # stream the value is as complete as it will ever be — extract it,
    # still never speak it.
    spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n", "Sure. ",
                                "<delegate>look something up"])
    assert spoken == ["Sure."]
    assert tags == [("DELEGATE", "look something up")]
    spoken, _, tags = run_tags(["###TRANSCRIPT: ok\n",
                                "Back to normal. <mode>conversation"])
    assert spoken == ["Back to normal."]
    assert tags == [("MODE", "conversation")]


def test_half_open_tag_at_stream_end_still_drops():
    # Only a complete opening bracket with a value fires at EOS; a
    # fragment ('<dele', '<mode>') never does.
    for tail in ["<dele", "<mode>", "<mode >  "]:
        spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n", "Okay. ", tail])
        assert spoken == ["Okay."], tail
        assert tags == [], tail


def test_speech_resumes_after_a_tag():
    spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n",
                                "One moment. <mode>translate target=en</mode>",
                                " Switching now. "])
    assert spoken == ["One moment.", "Switching now."]
    assert tags == [("MODE", "translate target=en")]


def test_unrecognized_markup_keeps_the_terminal_cut():
    spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n", "Hello. ",
                                "## Notes: stuff. Never spoken. "])
    assert spoken == ["Hello."]
    assert tags == []


def test_literal_angle_bracket_is_released():
    spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n",
                                "Five < 10 is true. <delegate>x</delegate>"])
    assert spoken == ["Five < 10 is true."]
    assert tags == [("DELEGATE", "x")]


def test_unknown_element_is_released_as_text():
    # Only configured names are control tags; other markup-ish text the
    # model produces is ordinary (spoken) output.
    spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n", "I like <b>bold</b> text. "])
    assert spoken == ["I like <b>bold</b> text."]
    assert tags == []


def test_tag_only_reply_yields_tag_and_no_speech():
    spoken, _, tags = run_tags(["<delegate>solo task</delegate>"])
    assert spoken == []
    assert tags == [("DELEGATE", "solo task")]


def test_no_transcript_mode_extracts_tags():
    spoken, transcript, tags = run_tags(
        ["I see a cat. ", "<mode>conversation</mode>"], expect_transcript=False)
    assert transcript is None
    assert spoken == ["I see a cat."]
    assert tags == [("MODE", "conversation")]


def test_back_to_back_tags():
    spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n", "On it. ",
                                "<delegate>task one</delegate>",
                                "<mode>translate target=en</mode>"])
    assert spoken == ["On it."]
    assert tags == [("DELEGATE", "task one"), ("MODE", "translate target=en")]


def test_tolerant_spacing_still_extracts():
    # '< delegate >' variants are the model reaching for the tag — firing
    # the intended action beats suppressing it.
    spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n", "On it. ",
                                "< delegate >find X</ delegate >"])
    assert spoken == ["On it."]
    assert tags == [("DELEGATE", "find X")]


def test_value_may_contain_angle_bracket():
    # 'flights under <$500' must not wedge the match: the close tag still
    # terminates the value, and speech resumes after it.
    spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n", "Sure. ",
                                "<delegate>flights under <$500</delegate>",
                                " Give me a sec. "])
    assert spoken == ["Sure.", "Give me a sec."]
    assert tags == [("DELEGATE", "flights under <$500")]


@pytest.mark.parametrize("markup", [
    '<delegate task="find X">',   # attribute form
    "<delegate/> find X",         # self-closing
    "</delegate> find X",         # orphan close
], ids=["attribute", "self-closing", "orphan-close"])
def test_near_miss_markup_is_suppressed_not_spoken(markup):
    # Names a control tag without being a clean element: model error —
    # suppress from there on (like ## markup); the task text and anything
    # after it must never reach TTS, and no action may fire.
    spoken, _, tags = run_tags(["###TRANSCRIPT: hi\n", "On it. ", markup])
    assert spoken == ["On it."]
    assert tags == []


@pytest.mark.parametrize("text", [
    DELEGATED,
    "###TRANSCRIPT: q\nOk. < delegate >find X</delegate> Done. ",
    "###TRANSCRIPT: q\nOk. <delegate>a <$5 b</delegate> Done. ",
    '###TRANSCRIPT: q\nOk. <delegate x="y">TASK never spoken. ',
    "###TRANSCRIPT: q\nOk. </delegate> TASK never spoken. ",
    "###TRANSCRIPT: q\nOk. 5 < 10 < 20 holds. <b>b</b>. ",
], ids=["clean", "tolerant", "inner-lt", "attribute", "orphan", "literal"])
def test_streaming_matches_batch_for_every_split_point_with_tags(text):
    batch = run_tags([text])
    for i in range(1, len(text)):
        assert run_tags([text[:i], text[i:]]) == batch, f"diverged at split {i}"


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
                   server.TRANSLATE_PROMPT):
        for said in ("What is the capital of France?",
                     "So the thing I wanted to ask you about is the weather "
                     "in Paris for my trip next week.",
                     "I have been trying to learn English for a few months now.",
                     "Please respond to them, all of them, by email."):
            assert not echoes_instruction(said, prompt), (said, prompt[:40])
    # Too short to judge stays visible.
    assert not echoes_instruction("On a new line", "so on a new line: x")


def test_production_filter_knows_every_control_tag():
    # The filter must ALWAYS be built with every tag name the prompts can
    # incite: a name it doesn't know is released as speech, so a narrowed
    # per-mode set would read task text aloud (found in review — modes must
    # gate acting on tags, never parsing them).
    from parlor import server
    assert set(n.lower() for n in server.CONTROL_TAGS) == {"delegate", "mode"}


def test_parser_without_control_tags_is_unchanged():
    # No control_tags configured → a delegate element is just text; the
    # ##-markup cut still applies to hash markup.
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
    assert not echoes_instruction(said, server.TRANSLATE_PROMPT)
    # Unquoted instruction prose still reads as an echo.
    assert echoes_instruction(
        "If the audio has no clear words write no speech instead",
        server.TRANSLATE_PROMPT)
