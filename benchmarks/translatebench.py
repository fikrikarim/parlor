"""Translate-mode x→y benchmark: can translation stop hardcoding English?

Production's translate mode renders every utterance into English. Going
x→y needs three things to hold, each measured here:

  1. Head capture — the action decider must extract the TARGET language
     from the spoken request ("translate everything I say into Spanish"),
     via one added string field on the production schema.
  2. Rendering — Gemma must actually translate into the target when the
     per-utterance prompt names it (TRANSLATE_PROMPT with "English"
     swapped out), judged by target-language keywords / script.
  3. Speech — Kokoro must have a voice for the target. The probe runs
     every candidate voice on this install and writes the WAVs to
     benchmarks/results/translate_voices/ — synthesis succeeding is
     measurable, pronunciation quality still needs ears.

Two-way interpreting (en ↔ es) is measured on top: the head captures a
language PAIR (the languages field grows a second entry), and a
pair-aware prompt lets the model pick the direction per utterance from
the language it hears — judged with real Spanish speech synthesized on
the Spanish voice. Voice policy: a pair containing English keeps the
English voice for both directions.

    uv run python benchmarks/translatebench.py --repeat 2 \
        --out benchmarks/results/translatebench.json

Runs its own llama-server on port 8099 (like archbench); WAVs cache in
fixtures/tagbench/.

Measured (e4b, M3 Pro, 2026-08-02 — results/translatebench.json): x→y is
fully feasible, and so is two-way. Head capture 13/14 under the
languages-array schema: one-way targets stay perfect, "translate between
English and X" pairs are captured, the how-do-you-say trap is ignored;
the single miss was one repeat of the looser "interpret between us for
us" phrasing returning no action. Rendering 12/12 one-way (Spanish,
French, Japanese; English control intact) and 5/6 two-way — the model
picked the correct direction every time it rendered (en audio → Spanish,
es audio → English, from real Spanish speech), and the one miss was a
transcript-only reply, the known failure shape production already covers
with fallback lines, not a direction error. Voice policy decided: a pair
containing English keeps the English voice for both directions. TTS was
the only real gap: tts.py never forwarded the voice's language prefix,
so every non-English voice was G2P'd as English — fixed by passing
lang_code (en/es/fr/it/pt/hi synthesize on their own pipelines; ja/zh
additionally need the misaki[ja]/[zh] extras). Pronunciation quality
still needs ears: results/translate_voices/*.wav.
"""

import argparse
import base64
import http.client
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("LLAMA_PORT", "8099")

import fixtures  # noqa: E402
from tagbench import CACHE_DIR, ensure_fixtures  # noqa: E402
from parlor import actions  # noqa: E402
from parlor import llama  # noqa: E402
from parlor import server  # noqa: E402
from parlor.pipeline import audio_part, text_part  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

# ── 1. head capture ───────────────────────────────────────────────────────
# Production schema + one field: the language(s) involved. One entry is a
# one-way target ("english" when unnamed keeps today's behavior the
# default); two entries is a two-way interpreting pair.
LANG_CLAUSE = (
    " languages: when mode is 'translate', the language or languages "
    "involved, as lowercase English words: ONE entry — the language "
    "everything the user says should be rendered into ('english' if they "
    "didn't name one) — or TWO entries if they asked for two-way "
    "interpreting between two languages (like between English and "
    "Spanish); an empty list when mode is not 'translate'."
)
HEAD_PROMPT = actions._HEAD_COMMON.format(
    mode_clause=actions._MODE_CLAUSES["conversation"]) + LANG_CLAUSE
HEAD_SCHEMA = {
    **actions.HEAD_SCHEMA,
    "properties": {**actions.HEAD_SCHEMA["properties"],
                   "languages": {"type": "array",
                                 "items": {"type": "string"},
                                 "maxItems": 2}},
    "required": actions.HEAD_SCHEMA["required"] + ["languages"],
}

# name -> (spoken command, expected (mode, languages) or None); languages
# compare as a set — pair order is not meaningful.
HEAD_CASES = {
    "xl_spanish": ("From now on, please translate everything I say into "
                   "Spanish.", ("translate", ["spanish"])),
    "xl_french": ("Please translate everything I say into French from now "
                  "on.", ("translate", ["french"])),
    "xl_japanese": ("Can you translate everything I say into Japanese from "
                    "now on?", ("translate", ["japanese"])),
    "xl_plain_english": ("From now on, please translate everything I say "
                         "into English.", ("translate", ["english"])),
    "xl_pair_spanish": ("Please translate between English and Spanish for "
                        "us.", ("translate", ["english", "spanish"])),
    "xl_pair_japanese": ("Can you interpret between English and Japanese "
                         "for us?", ("translate", ["english", "japanese"])),
    # Trap: a question ABOUT a language is not a mode request.
    "xl_trap_french": ("How do you say good morning in French?", None),
}

SYSTEM = server.SYSTEM_PROMPT + server.CAPABILITY_NOTE
RESPOND = server.RESPOND_PROMPT.format(camera="")

# ── 2. rendering ──────────────────────────────────────────────────────────

def translate_prompt(language: str) -> str:
    """Production's per-utterance prompt, imported so drift is impossible."""
    return server.TRANSLATE_PROMPT.format(language=language)


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[぀-ヿ一-鿿]", text))


# (utterance fixture, target language, judge) — judge takes the rendered
# translation (transcript line already stripped) and says if it landed in
# the target language with the content words intact.
RENDER_CASES = {
    "render_es_capital": ("capital_france", "Spanish",
                          lambda t: "francia" in t.lower()),
    "render_fr_capital": ("capital_france", "French",
                          lambda t: "capitale" in t.lower()
                          or "france" in t.lower()),
    "render_es_long": ("long_question", "Spanish",
                       lambda t: "ingl" in t.lower()
                       or "pronunciaci" in t.lower()),
    "render_fr_long": ("long_question", "French",
                       lambda t: "angl" in t.lower()
                       or "prononciation" in t.lower()),
    "render_ja_capital": ("capital_france", "Japanese", has_cjk),
    # Control: today's behavior, must keep working.
    "render_en_capital": ("capital_france", "English",
                          lambda t: "capital" in t.lower()
                          and "france" in t.lower()),
}

# ── 2b. two-way rendering ─────────────────────────────────────────────────
# Production's pair-aware prompt: the model decides the direction from
# the language it hears.
TWO_WAY_PROMPT = server.TWO_WAY_PROMPT

# Spoken Spanish, synthesized with the Spanish voice on its own pipeline —
# the es→en direction must be tested with real foreign audio.
FOREIGN_CASES = {
    "es_station": ("¿Dónde está la estación de tren más cercana?", "ef_dora"),
    "es_coffee": ("Me gustaría dos cafés con leche, por favor.", "ef_dora"),
}

# name -> (wav key, (a, b), judge): the direction each judge implies is
# opposite to the audio's language, which is the whole point.
RENDER2_CASES = {
    "two_way_en_capital": ("capital_france", ("English", "Spanish"),
                           lambda t: "francia" in t.lower()),
    "two_way_es_station": ("es_station", ("English", "Spanish"),
                           lambda t: "station" in t.lower()
                           or "train" in t.lower()),
    "two_way_es_coffee": ("es_coffee", ("English", "Spanish"),
                          lambda t: "coffee" in t.lower()
                          or "milk" in t.lower()),
}


def ensure_foreign_fixtures() -> dict[str, str]:
    """name -> WAV base64 for the non-English clips, synthesizing any
    missing ones with their per-language voice."""
    missing = [n for n in FOREIGN_CASES
               if not (CACHE_DIR / f"{n}.wav").exists()]
    if missing:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from parlor import tts
        backend = tts.load()
        for name in missing:
            text, voice = FOREIGN_CASES[name]
            pcm = fixtures._synthesize(backend, text, voice=voice)
            fixtures._write_wav(CACHE_DIR / f"{name}.wav", pcm,
                                fixtures.TARGET_SR)
            print(f"fixture {name}: {len(pcm) / fixtures.TARGET_SR:.1f}s speech")
    return {n: base64.b64encode((CACHE_DIR / f"{n}.wav").read_bytes()).decode()
            for n in FOREIGN_CASES}

# ── 3. voices ─────────────────────────────────────────────────────────────
# Kokoro's non-English voices, one per candidate target. Synthesis
# succeeding proves the voice exists on this install; listen to the WAVs
# for quality (non-English G2P may need espeak-ng / extra misaki extras).
VOICE_PROBES = {
    "english": ("af_heart", "Hello there, how are you doing today?"),
    "spanish": ("ef_dora", "Hola, ¿cómo estás? Me alegra mucho verte hoy."),
    "french": ("ff_siwis", "Bonjour, comment ça va aujourd'hui ?"),
    "italian": ("if_sara", "Ciao, come stai oggi? Sono felice di vederti."),
    "portuguese": ("pf_dora", "Olá, como você está hoje?"),
    "hindi": ("hf_alpha", "नमस्ते, आप आज कैसे हैं?"),
    "japanese": ("jf_alpha", "こんにちは、今日は元気ですか。"),
    "chinese": ("zf_xiaobei", "你好，今天过得怎么样？"),
}


def chat(messages: list, *, max_tokens: int = 256,
         temperature: float | None = None,
         json_schema: dict | None = None) -> str:
    body: dict = {"messages": messages, "max_tokens": max_tokens,
                  "cache_prompt": True,
                  "chat_template_kwargs": {"enable_thinking": False}}
    body["temperature"] = llama.TEMPERATURE if temperature is None else temperature
    if json_schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"schema": json_schema}}
    conn = http.client.HTTPConnection(*llama.host_port(), timeout=300)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    data = json.loads(conn.getresponse().read())
    conn.close()
    if "error" in data:
        raise RuntimeError(f"llama-server: {data['error']}")
    return data["choices"][0]["message"].get("content") or ""


def run_head_case(wav: str) -> tuple[dict, int, str]:
    """Production's decide_after shape: speech reply first, head second."""
    speech = [{"role": "system", "content": SYSTEM},
              {"role": "user", "content": [audio_part(wav), text_part(RESPOND)]}]
    reply = chat(speech)
    t0 = time.time()
    raw = chat(speech + [{"role": "assistant", "content": reply},
                         {"role": "user", "content": HEAD_PROMPT}],
               max_tokens=64, temperature=0.0, json_schema=HEAD_SCHEMA)
    ms = round((time.time() - t0) * 1000)
    try:
        return json.loads(raw), ms, reply
    except ValueError:
        return {}, ms, reply


def run_render_case(wav: str, prompt: str) -> tuple[str, str, int]:
    """One utterance through translate mode under `prompt`. Returns
    (transcript line, translation, wall ms)."""
    t0 = time.time()
    raw = chat([{"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": [audio_part(wav), text_part(prompt)]}])
    ms = round((time.time() - t0) * 1000)
    body = re.sub(r"^#{2,}[ \t]*TRANSCRIPT[ \t]*:[ \t]*", "", raw.strip(),
                  flags=re.IGNORECASE)
    first_nl = body.find("\n")
    transcript, translation = ((body[:first_nl], body[first_nl + 1:])
                               if first_nl != -1 else (body, ""))
    return transcript.strip(), translation.strip(), ms


def probe_voices() -> dict:
    from parlor import tts
    backend = tts.load()
    out_dir = RESULTS_DIR / "translate_voices"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for lang, (voice, text) in VOICE_PROBES.items():
        try:
            t0 = time.time()
            pcm = backend.generate(text, voice=voice)
            seconds = round(len(pcm) / backend.sample_rate, 2)
            fixtures._write_wav(out_dir / f"{lang}_{voice}.wav", pcm,
                                backend.sample_rate)
            report[lang] = {"voice": voice, "ok": seconds > 0.3,
                            "seconds": seconds,
                            "gen_ms": round((time.time() - t0) * 1000)}
            print(f"✓ voice {lang} ({voice}): {seconds}s audio")
        except Exception as e:
            report[lang] = {"voice": voice, "ok": False,
                            "error": f"{type(e).__name__}: {e}"}
            print(f"✗ voice {lang} ({voice}): {type(e).__name__}: {e}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--skip-voices", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out: dict = {"model": llama.MODEL}
    if not args.skip_voices:
        print("── Kokoro voice probe ──")
        out["voices"] = probe_voices()
        print()

    if not args.skip_llm:
        wavs = ensure_fixtures({n: t for n, (t, _) in HEAD_CASES.items()})
        fixtures.generate_all()  # render cases reuse the base e2e fixtures
        render_wavs = {n: fixtures.load_wav_b64(f)
                       for n, (f, _, _) in RENDER_CASES.items()}
        foreign_wavs = ensure_foreign_fixtures()
        llama.start()
        try:
            print("── head language capture ──")
            head_results = []
            for name, (_, expected) in HEAD_CASES.items():
                for _ in range(args.repeat):
                    head, ms, _reply = run_head_case(wavs[name])
                    got_mode = head.get("mode")
                    got_langs = sorted(str(l).strip().lower()
                                       for l in head.get("languages") or [])
                    if expected is None:
                        hit = got_mode in (None, "none")
                    else:
                        hit = (got_mode == expected[0]
                               and got_langs == sorted(expected[1]))
                    head_results.append({"case": name, "hit": hit,
                                         "mode": got_mode,
                                         "languages": got_langs,
                                         "head_ms": ms})
                    print(f"{'✓' if hit else '✗'} {name}: mode={got_mode} "
                          f"languages={got_langs} ({ms}ms)")
            out["head"] = {
                "accuracy": round(sum(r["hit"] for r in head_results)
                                  / len(head_results), 3),
                "results": head_results}
            print(f"\nhead: {out['head']['accuracy']}\n")

            print("── rendering ──")
            render_results = []
            for name, (fixture, language, judge) in RENDER_CASES.items():
                for _ in range(args.repeat):
                    transcript, translation, ms = run_render_case(
                        render_wavs[name], translate_prompt(language))
                    ok = bool(translation) and judge(translation)
                    render_results.append({
                        "case": name, "language": language, "ok": ok,
                        "transcript": transcript,
                        "translation": translation, "ms": ms})
                    print(f"{'✓' if ok else '✗'} {name}: {translation[:80]!r} "
                          f"({ms}ms)")
            out["render"] = {
                "accuracy": round(sum(r["ok"] for r in render_results)
                                  / len(render_results), 3),
                "results": render_results}
            print(f"\nrender: {out['render']['accuracy']}\n")

            print("── two-way rendering ──")
            r2 = []
            for name, (key, (a, b), judge) in RENDER2_CASES.items():
                wav = foreign_wavs.get(key) or fixtures.load_wav_b64(key)
                for _ in range(args.repeat):
                    transcript, translation, ms = run_render_case(
                        wav, TWO_WAY_PROMPT.format(a=a, b=b))
                    ok = bool(translation) and judge(translation)
                    r2.append({"case": name, "pair": [a, b], "ok": ok,
                               "transcript": transcript,
                               "translation": translation, "ms": ms})
                    print(f"{'✓' if ok else '✗'} {name}: {translation[:80]!r} "
                          f"({ms}ms)")
            out["two_way"] = {
                "accuracy": round(sum(r["ok"] for r in r2) / len(r2), 3),
                "results": r2}
            print(f"\ntwo_way: {out['two_way']['accuracy']}\n")
        finally:
            llama.stop()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
