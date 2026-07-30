# Handoff: latency rebuild + llama.cpp port

Branch: `perf-latency`. Everything below happened in one working session on
2026-07-29; this doc is the state of the world, what is verified, and what
still needs human testing. Delete this file before merging.

## What changed (commit order tells the story)

1. **E2E benchmark harness** (`src/benchmarks/`) — real synthesized speech
   fixtures, perf + correctness suites, JSON results, `compare.py`.
2. **Streaming pipeline** — the `respond_to_user` tool call (which silently
   cost a second full inference round-trip per turn) replaced by streamed
   decoding: response sentences go to TTS while the model still generates,
   transcript is produced last. VAD silence cutoff 600ms → 200ms.
3. **Speculative prefill** — the camera frame is processed while the user is
   still talking; after the llama.cpp port, the speech itself also streams in
   ~3s chunks through the prompt cache.
4. **llama.cpp port** — litert-lm fully replaced by a spawned `llama-server`
   (official Google QAT q4_0 GGUF + mmproj). The server owns conversation
   history; prefix caching makes re-sending it cheap. Real barge-in abort.
   litert-lm can be restored by reverting two commits (`2284193`, `37caaaf`).
5. **Turn detection** — judgment belongs to pipecat's smart-turn-v3.2 audio
   classifier (~20ms CPU), and the LLM prompt carries no format instructions
   at all. The inline-marker and separate-request variants were measured
   (`benchmarks/turnbench.py`) at chance accuracy on E2B, E4B *and* 12B, so
   `TURN_MODE` is gone and only the classifier path remains. The benchmark
   still reproduces both variants against any future model.
6. **Live-session fixes** — history poisoning by invalid audio, echo
   parroting (AEC reference path + sustained-speech barge-in), capture leak.

## Measured (M3 Pro, `benchmarks/results/`)

End of utterance → first audio heard; add ~200ms VAD on top. Baseline is the
pre-session litert build.

| Turn                        | Baseline | Now       |
| --------------------------- | -------- | --------- |
| Short question              | 1.52s    | ~0.6s     |
| Short + camera              | 1.94s    | ~0.7s     |
| Long question (9.4s speech) | 2.91s    | ~0.7-0.9s |
| Long + camera               | 2.98s    | ~0.8-1.0s |

All bench correctness checks pass except `thinking_suppressed` (see Known
limitations). Reproduce: `uv run server.py`, then
`uv run python benchmarks/bench.py --label X --out benchmarks/results/X.json`.

## Test checklist

Run with logs captured: `uv run python server.py 2>&1 | tee /tmp/parlor.log`.
Hard-refresh the browser (Cmd+Shift+R) after every server restart.

### A. Turn-taking (the part benchmarks cannot judge — real prosody)

- [ ] Finish a sentence cleanly → response starts in well under a second.
- [ ] Trail off mid-sentence ("So what I wanted to ask is…") → stays quiet,
      log shows `p(complete)` near 0, gentle nudge after ~5s of silence.
- [ ] "Hmm, let me think about that…" with genuine hesitation tone → stays
      quiet. **Watch the `p(complete)` values** — if your speaking style
      lands on the wrong side, the 0.5 threshold in `turn_detector.py` is
      the tuning knob.
- [ ] Continue after an incomplete pause → the eventual answer accounts for
      BOTH parts of your utterance, and the transcript shows the whole thing.
- [ ] Nudges stop after 2 (it must not pester an empty room).
- [ ] Natural fast back-and-forth conversation feels right at 200ms VAD; if
      it clips you, `redemptionMs` in index.html.

### B. Speech overlap (long utterances)

- [ ] Ask an 8-10s question → `Primed cache (...)` lines appear WHILE you
      talk, and the response starts about as fast as for a short question.
- [ ] If no `Primed cache` lines appear during speech, the vad-web
      `onFrameProcessed` fallback kicked in — chunk streaming is silently
      off. Report the vad-web version.
- [ ] Transcript of a long chunked utterance is verbatim-accurate (this is
      the chunk-boundary integrity check).

### C. Echo and barge-in

- [ ] Speakers at normal volume: the assistant must never answer its own
      voice (the parrot bug). Check `heard:` lines for its own phrasings.
- [ ] Speakers loud: same. (Three layers should hold: AEC via the media
      element, 250ms sustained-speech gate, echo rule in the prompt.)
- [ ] Deliberate barge-in mid-reply: it stops within a beat (~250ms of you
      speaking) and handles what you said next.
- [ ] Barge-in within the first ~800ms of it speaking is intentionally
      ignored (echo grace period) — confirm that feels okay.

### D. Camera

- [ ] Ask about what it sees → grounded, correct description.
- [ ] Move/change the scene between turns → it references the NEW scene
      (frame freshness; the frame is captured at speech start).
- [ ] Camera toggled off → conversation still works, no "with camera" label.
- [ ] A cough/misfire followed by a real question → fresh frame, not stale.

### E. Robustness

- [ ] Coughs, taps, mic bumps → no error cascade, session keeps working
      (the old failure mode was a poisoned history requiring reload).
- [ ] Talk while it's still processing the previous turn → queued, handled.
- [ ] Reload the page mid-reply → reconnects, fresh conversation, works.
- [ ] Long session (15+ camera exchanges) → context rotation kicks in
      ("dropping N oldest messages" in log) and the session survives it.
- [ ] Kill llama-server manually → next turn fails gracefully (client
      returns to listening), and a server restart recovers.

### F. Quality (the original complaint)

- [ ] Responses feel at least as good as the pre-worktree version — the
      marker instructions are gone from the prompt, which was the main
      suspected quality tax.
- [ ] Transcripts accurate enough to be pedagogically useful.
- [ ] Multi-turn memory: reference something from a few turns back.
- [ ] **Multilingual** (the Bule-AI use case): speak Indonesian or another
      language → understanding and transcript quality. Note: Kokoro TTS
      voice `af_heart` is English; non-English TTS output is a known gap.
      smart-turn-v3 is trained on 23 languages, but verify turn-taking
      feels right in non-English speech.
- [ ] If quality is still lacking: try E4B via
      `MODEL_PATH`/`MMPROJ_PATH` (unsloth/gemma-4-E4B-it-GGUF) — expect
      roughly 2x latency.

### G. Platforms (untested — needs hardware)

- [ ] **Linux**: llama.cpp must be installed manually (no brew); TTS falls
      back to kokoro-onnx; `onnxruntime` and vad-web CDN paths. Entirely
      unverified.
- [ ] Safari / Firefox: vad-web, the AEC media-element routing, and audio
      autoplay policies all behave differently. Chrome is the only tested
      browser.
- [ ] Lower-RAM Macs: model + mmproj ≈ 4GB + TTS; 8GB machines are dubious.

## Known limitations / accepted trade-offs

- `thinking_suppressed` bench check fails **by construction**: the fixture is
  TTS-synthesized with finished-sounding falling prosody, and the acoustic
  classifier (correctly) reads the acoustics. Real hesitation is what it is
  trained on — judge from live testing, not this fixture.
- Image-only turns sometimes invent a "transcript" of the instruction text
  (cosmetic; real clients always send audio).
- llama-server output goes to DEVNULL; un-silence in `start_llama_server()`
  when debugging.
- llama.cpp marks Gemma audio input "experimental stage".
- Temperature is 0 (deterministic): nudge phrasing repeats verbatim.
- Context-size guard uses a token *estimate* (`estimate_tokens`), not exact
  counts.

## Upstream issues worth filing

- **litert-lm**: `cancel_process()` permanently wedges the Conversation or
  Session it is called on (engine survives; next send never returns).
- **mlx-vlm**: (1) more than one audio segment per turn crashes the feature
  extractor (paths never decoded); (2) `multimodal_token_ids_from_config`
  omits `audio_token_id`, so the prompt-cache media guard silently fails to
  fire for audio — a corruption bug once multi-audio works.

## Experiment record

`src/benchmarks/experiment_*.py` + commit messages document every dead end
with numbers: session-level litert prefill (infeasible), chunked audio
across closed turns (destroys transcript), MLX overlap (blocked upstream),
LLM turn markers (all five prompt variants), visual_token_budget
(hallucinates). Read these before re-attempting any of them.
