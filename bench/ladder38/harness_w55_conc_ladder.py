#!/usr/bin/env python3
"""
PR #388 definitive concurrency ladder — C=1..128, one client, both engines.

Pinned by recipes/qwen3.6/qwen3.6-27b-w55-sweep-dev.yaml. Every measurement
knob is a constant or a required argument; nothing is defaulted silently.

Methodology (identical for Atlas and vLLM — the SAME client drives both, which
is the point: two harnesses measuring two engines is not an A/B):
  * regime decode_short: ISL 128 / OSL 1024
  * one rep = one batch of C concurrent streaming requests; wall = batch wall
  * reps per rung recorded individually (raw series), never only the mean
  * temperature 0, seed 42, token-matched via a forcing suffix + max_tokens cap
  * chat_template_kwargs.enable_thinking=false on BOTH engines
  * per-request nonce, based at a RANDOM per-process value, so
    enable_prefix_caching cannot serve a repeat from cache — not within a run
    and not across two runs against the same server (see make_prompt)
  * completion_tokens/prompt_tokens read from the usage frame, not counted deltas
    (Atlas batches a short reply into ONE SSE delta)
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import sys
import time

# OPTIONAL on purpose: `--check-shapes` grades prompt construction and needs no
# server, so it must run on a box (or a CI runner) that has no HTTP client
# installed. Every measurement path still refuses to start without it — see the
# guard in main().
try:
    import aiohttp
except ImportError:  # pragma: no cover - exercised by the --check-shapes gate
    aiohttp = None

# ── pinned constants (recipe: benchmark.prompt / benchmark.sampling) ──
#
# VARIED filler, byte-identical to the corpus in
# crates/atlas-plugin/src/benchmarks/stats.rs. Its comment states the reason and
# this run confirmed it the hard way: UNIFORM repetition ("The quick brown fox…"
# over and over, the corpus bench-atlas-concurrency.py uses) drives the model
# into degenerate repetitive output. On Atlas that trips the SimHash
# semantic-loop watchdog, which ENDS the stream — one C=2 request finished at
# 213 of 1024 tokens. vLLM has no such watchdog, so the two engines would have
# emitted wildly different token counts and the ladder would have been
# uninterpretable. Uniform filler is not a valid decode workload.
FILLER = (
    "The quick brown fox jumped over the lazy dog near a river bank. "
    "Mountains rise above the clouds while birds sing their morning songs. "
    "Science explores the universe through careful observation and experiment. "
    "Ancient civilizations built remarkable structures that still stand today. "
    "Music fills the air with rhythm and harmony across every culture. "
    "Technology advances rapidly changing how people communicate and work. "
    "Forests provide shelter for countless species of plants and animals. "
    "Ocean waves crash upon the shore under the light of the moon. "
)
# Output-forcing policy. `count` is the built-in benchmark's PromptMode::Count.
# `essay` asks for long varied prose instead — see the probe in the report for
# which one actually holds the full budget on BOTH engines without looping.
SUFFIX_COUNT = " Count from 1 upward, one number per line, until told to stop."
SUFFIX_ESSAY = (" Using the text above only as a starting point, write a long, richly detailed "
                "essay that keeps introducing new specifics, examples and vocabulary. "
                "Never repeat a sentence or paraphrase one you have already written. "
                "Do not summarise and do not stop early.")
TEMPERATURE = 0.0
SEED = 42
ENABLE_THINKING = False  # set from --enable-thinking in main()
REQUEST_TIMEOUT_S = 900

PROMPT_MODE = os.environ.get("W55_PROMPT_MODE", "essay")

# ── the per-request nonce ────────────────────────────────────────────────────
#
# ★ THE LADDER IS FROZEN. Its prompts are compared across rounds, so the nonce
# must not change the PROMPT TOKEN LENGTH: it is a FIXED-WIDTH field, exactly
# NONCE_WIDTH digits, and every nonce is taken modulo NONCE_MODULUS so it can
# never widen. `--check-shapes` asserts that property rather than trusting it.
#
# ★ THE BUG THIS FIXES (H100 rounds 10-12, `h100-round12-report.md` stage 5a).
# The counter used to start at 0 in every new python process, so a pre-window
# WARMUP invocation and a later PHASE-A invocation against the SAME server sent
# byte-identical prompts and `--enable-prefix-caching true` did its job: round
# 12's phase-A capture reported `ttft_p50_ms: 37.18` against a cold 184 ms, and
# its prefill table is a cached prefill plus a decode step. Round 10's phase-A
# TTFT of 49 ms is the same defect, uncaught at the time.
#
# It never touched a LADDER measurement: a ladder invocation runs its warmup and
# all its reps in ONE process, where the counter keeps advancing and no two
# prompts repeat. What it broke is every measurement whose warmup was a
# SEPARATE invocation — which is exactly what the nsys captures did.
#
# Basing the counter at a random per-process value closes both: within a run the
# counter still advances, and two runs no longer start at the same place.
NONCE_WIDTH = 6
NONCE_MODULUS = 10**NONCE_WIDTH

_seq = 0
_nonce_base = 0


def default_nonce_base() -> int:
    """A per-process base no other process is likely to have used.

    `SystemRandom` rather than pid+clock arithmetic: a pid is 5 digits on Linux
    and recycles, and two ladder invocations started in the same second from a
    script would sit close together. The chosen base is printed and written to
    the run JSON (`nonce_base`), so a receipt records exactly which prompts a
    run sent even though the value is random.
    """
    return random.SystemRandom().randrange(NONCE_MODULUS)


def set_nonce_base(base: int) -> int:
    """Pin the base and restart the counter. Returns the base actually used."""
    global _nonce_base, _seq
    _nonce_base = base % NONCE_MODULUS
    _seq = 0
    return _nonce_base


def make_prompt(isl_tokens: int) -> str:
    """Word-for-word the shape of atlas-plugin's `stats::make_prompt`: the chat
    template contributes ~12 tokens, the rest is `needed` filler words, and the
    nonce prefix forces a prefix-cache MISS so every request does real prefill.

    The nonce is `(base + n) % NONCE_MODULUS` rendered at NONCE_WIDTH digits, so
    the prompt's CHARACTER length — and, for a digit-splitting tokenizer such as
    Qwen's, its TOKEN length — does not depend on the base. `--check-shapes`
    is the assertion."""
    global _seq
    _seq += 1
    nonce = (_nonce_base + _seq) % NONCE_MODULUS
    needed = max(1, isl_tokens - 12)
    words = FILLER.split()
    out = f"[req {nonce:0{NONCE_WIDTH}d}] " + " ".join(
        words[i % len(words)] for i in range(needed)
    )
    return out + (SUFFIX_COUNT if PROMPT_MODE == "count" else SUFFIX_ESSAY)


def percentile(data, p):
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


async def one_request(session, url, model, prompt, osl):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": osl,
        "temperature": TEMPERATURE,
        # ★ PARITY (2026-08-17): both engines must apply the SAME sampling work.
        # Atlas's MODEL.toml non_thinking preset injects presence_penalty=1.5 when
        # the request omits it; vLLM defaults to 0. That is not a like-for-like
        # comparison — Atlas was doing extra per-token logit work AND emitting
        # different text. Sending these explicitly pins both engines to identical
        # sampling. (Measured worth to Atlas at C=8: +7.8%, because the penalty
        # path disables four fast-greedy sampling paths.)
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": SEED,
        "stream": True,
        "stream_options": {"include_usage": True},
        # ★ the ONLY key that toggles thinking on vLLM. {"thinking": false} is
        # silently ignored. Sent to both engines so the bodies are identical.
        # Default False (the GB10 campaign's setting); --enable-thinking flips
        # it for a think-on cell, and the run header records which was sent.
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    completion_tokens = 0
    prompt_tokens = 0
    deltas = 0
    finish_reason = None
    try:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)) as resp:
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}: {(await resp.text())[:300]}"}
            buf = ""
            async for chunk in resp.content.iter_any():
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for ch in ev.get("choices") or []:
                        content = (ch.get("delta") or {}).get("content")
                        if content:
                            now = time.perf_counter()
                            if t_first is None:
                                t_first = now
                            t_last = now
                            deltas += 1
                        if ch.get("finish_reason"):
                            finish_reason = ch["finish_reason"]
                    usage = ev.get("usage")
                    if usage:
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    except Exception as e:  # transport / timeout
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    t_end = time.perf_counter()
    e2e = t_end - t0
    ttft = (t_first - t0) if t_first else e2e
    decode = (t_last - t_first) if (t_first and t_last) else 0.0
    tpot = (decode / (completion_tokens - 1)) if completion_tokens > 1 and decode > 0 else 0.0
    return {
        "ttft_ms": ttft * 1000.0,
        "tpot_ms": tpot * 1000.0,
        "e2e_s": e2e,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "sse_deltas": deltas,
        "finish_reason": finish_reason,
    }


async def run_rep(session, url, model, conc, isl, osl):
    prompts = [make_prompt(isl) for _ in range(conc)]
    t0 = time.perf_counter()
    outs = await asyncio.gather(*[one_request(session, url, model, p, osl) for p in prompts])
    wall = time.perf_counter() - t0
    good = [o for o in outs if "error" not in o]
    errs = [o for o in outs if "error" in o]
    ctok = sum(o["completion_tokens"] for o in good)
    ptok = sum(o["prompt_tokens"] for o in good)
    return {
        "wall_s": wall,
        "completion_tokens": ctok,
        "prompt_tokens": ptok,
        "prompt_tokens_per_req": sorted({o["prompt_tokens"] for o in good}),
        "tok_s": (ctok / wall) if wall > 0 else 0.0,
        "n_ok": len(good),
        "n_err": len(errs),
        "errors": [e["error"] for e in errs][:3],
        "ttft_p50_ms": percentile([o["ttft_ms"] for o in good], 50),
        "ttft_p99_ms": percentile([o["ttft_ms"] for o in good], 99),
        "tpot_p50_ms": percentile([o["tpot_ms"] for o in good if o["tpot_ms"] > 0], 50),
        "e2e_p50_s": percentile([o["e2e_s"] for o in good], 50),
        "finish_reasons": sorted({str(o["finish_reason"]) for o in good}),
        "completion_tokens_per_req": sorted(o["completion_tokens"] for o in good),
    }


def load_tokenizer(path):
    """A Qwen tokenizer from a LOCAL path, or None.

    Never downloads: this runs on benchmark boxes that are offline by policy and
    inside CI. `None` is a legitimate answer — `check_shapes` then grades the
    character-length and digit-width invariant, which is the property the token
    count actually rests on for a digit-splitting tokenizer.
    """
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:  # missing files, wrong dir, unsupported revision
        return None


def check_shapes(isl, tokenizer_path):
    """`--check-shapes`: the nonce base must not change the prompt's SHAPE.

    The ladder is frozen for cross-round comparability, so a per-process nonce
    base is only admissible if a prompt built at base A tokenises to the same
    length as the same prompt built at base B. Three assertions, in the order
    they bite:

      1. the nonce field is exactly NONCE_WIDTH digits at every base, including
         the ones that would otherwise widen it (0 and NONCE_MODULUS - 1);
      2. two bases give byte-equal prompts outside that field, and equal
         character length;
      3. with a Qwen tokenizer present, equal TOKEN counts — the property (1)
         and (2) are a proxy for.

    Also asserts the bug is gone: two DIFFERENT bases must not produce the same
    first prompt, which is what the pre-window warmup and phase A used to do.
    """
    failures = []
    # Distinct MOD NONCE_MODULUS — bases that differ by a whole modulus alias
    # to the same nonce by construction, which assertion (4) states separately.
    bases = [0, 1, 42, 770_001, NONCE_MODULUS - 1]
    prompts = {}
    for base in bases:
        used = set_nonce_base(base)
        prompts[base] = make_prompt(isl)
        field = prompts[base][len("[req "):len("[req ") + NONCE_WIDTH]
        if not (len(field) == NONCE_WIDTH and field.isdigit()):
            failures.append(f"base {base} (used {used}): nonce field {field!r} is not "
                            f"{NONCE_WIDTH} digits")
        if prompts[base][len("[req ") + NONCE_WIDTH:len("[req ") + NONCE_WIDTH + 2] != "] ":
            failures.append(f"base {base}: nonce field is not closed by '] '")

    ref = prompts[bases[0]]
    head = len("[req ") + NONCE_WIDTH
    for base, prompt in prompts.items():
        if len(prompt) != len(ref):
            failures.append(f"base {base}: prompt is {len(prompt)} chars, base "
                            f"{bases[0]} is {len(ref)} — the ladder is not frozen")
        if prompt[:len("[req ")] != ref[:len("[req ")] or prompt[head:] != ref[head:]:
            failures.append(f"base {base}: prompt differs outside the nonce field")

    distinct = {p[:head] for p in prompts.values()}
    if len(distinct) != len(prompts):
        failures.append(f"two bases produced the same nonce: {sorted(distinct)} — the "
                        "prefix-cache collision this flag exists to prevent")

    # (4) the base is normalised into the field, and the counter restarts with
    # it, so an out-of-range --nonce-base narrows rather than widening the field.
    if set_nonce_base(NONCE_MODULUS + 7) != 7:
        failures.append("set_nonce_base does not reduce modulo NONCE_MODULUS")
    if make_prompt(isl)[:head] != f"[req {8:0{NONCE_WIDTH}d}]"[:head]:
        failures.append("set_nonce_base did not restart the per-process counter")

    tok = load_tokenizer(tokenizer_path)
    if tok is None:
        token_note = ("SKIPPED (no local tokenizer; pass --tokenizer <dir> or set "
                      "W55_TOKENIZER). The character-length and digit-width "
                      "assertions above are the standing invariant.")
    else:
        counts = {b: len(tok.encode(p)) for b, p in prompts.items()}
        if len(set(counts.values())) != 1:
            failures.append(f"token counts differ across nonce bases: {counts}")
        token_note = f"PASS ({sorted(set(counts.values()))[0]} tokens at every base)"

    print(f"# check-shapes isl={isl} width={NONCE_WIDTH} bases={bases}")
    print(f"#   prompt chars     : {len(ref)}")
    print(f"#   nonce fields     : {sorted(distinct)}")
    print(f"#   token-count check: {token_note}")
    for f in failures:
        print(f"FAIL: {f}")
    print("# check-shapes OK" if not failures else f"# check-shapes FAILED ({len(failures)})")
    return 0 if not failures else 1


async def main():
    ap = argparse.ArgumentParser()
    # NOT `required=True`, because --check-shapes needs none of them; the
    # explicit check below keeps the file's rule that no measurement knob is
    # ever defaulted silently.
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--label")
    ap.add_argument("--out")
    ap.add_argument("--concs")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--isl", type=int)
    ap.add_argument("--osl", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=true (default false)")
    ap.add_argument("--nonce-base", type=int, default=None,
                    help="per-process prompt-nonce base (default: random). Pin it only "
                         "to reproduce a specific run's prompts; two runs against one "
                         "server must NOT share a base or the second hits the prefix "
                         "cache. Recorded as `nonce_base` in the output JSON.")
    ap.add_argument("--check-shapes", action="store_true",
                    help="self-test: assert the nonce base changes no prompt's token "
                         "count, then exit. Needs no server.")
    ap.add_argument("--tokenizer", default=os.environ.get("W55_TOKENIZER"),
                    help="local tokenizer dir for --check-shapes (offline only)")
    a = ap.parse_args()

    if a.check_shapes:
        return check_shapes(a.isl if a.isl else 1024, a.tokenizer)

    if aiohttp is None:
        ap.error("aiohttp is required to drive a ladder (only --check-shapes runs without it)")

    missing = [f"--{n.replace('_', '-')}" for n in
               ("url", "model", "label", "out", "concs", "reps", "isl", "osl", "warmup")
               if getattr(a, n) is None]
    if missing:
        ap.error("missing required argument(s): " + ", ".join(missing))

    global ENABLE_THINKING
    ENABLE_THINKING = bool(a.enable_thinking)
    nonce_base = set_nonce_base(
        a.nonce_base if a.nonce_base is not None else default_nonce_base()
    )

    concs = [int(x) for x in a.concs.split(",") if x.strip()]
    chat = a.url.rstrip("/") + "/v1/chat/completions"
    with open(__file__, "rb") as _self_src:
        me = hashlib.sha256(_self_src.read()).hexdigest()

    record = {
        "label": a.label, "url": a.url, "model": a.model,
        "isl": a.isl, "osl": a.osl, "reps": a.reps, "warmup": a.warmup,
        "temperature": TEMPERATURE, "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
        # Which prompts this run sent. Random per process unless --nonce-base
        # pinned it; recorded so a receipt can say so, and so two runs against
        # one server can be shown not to have shared a prefix-cache entry.
        "nonce_base": nonce_base,
        "nonce_base_pinned": a.nonce_base is not None,
        "nonce_width": NONCE_WIDTH,
        "driver_sha256": me,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rungs": [],
    }
    print(f"# driver sha256 {me}", flush=True)
    print(f"# nonce base {nonce_base:0{NONCE_WIDTH}d} "
          f"({'pinned' if a.nonce_base is not None else 'random'})", flush=True)

    conn = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(connector=conn) as session:
        for conc in concs:
            for w in range(a.warmup):
                await run_rep(session, chat, a.model, conc, a.isl, a.osl)
            reps = []
            for r in range(a.reps):
                # SM clock sampled INSIDE the rep window, not before it.
                clk = os.popen("nvidia-smi --query-gpu=clocks.sm,power.draw "
                               "--format=csv,noheader,nounits").read().strip()
                rep = await run_rep(session, chat, a.model, conc, a.isl, a.osl)
                rep["rep"] = r
                rep["clock_sample_at_rep_start"] = clk
                reps.append(rep)
                print(f"[{a.label}] C={conc:>3} rep{r}  "
                      f"tok/s={rep['tok_s']:8.2f}  wall={rep['wall_s']:7.2f}s  "
                      f"ctok={rep['completion_tokens']:>7}  ptok/req={rep['prompt_tokens_per_req']}  "
                      f"ttft_p50={rep['ttft_p50_ms']:.0f}ms  err={rep['n_err']}  clk={clk}",
                      flush=True)
            series = [r["tok_s"] for r in reps]
            rung = {
                "concurrency": conc,
                "reps": reps,
                "tok_s_series": series,
                "tok_s_mean": statistics.fmean(series),
                "tok_s_median": statistics.median(series),
                "tok_s_spread_pct": (max(series) - min(series)) / statistics.fmean(series) * 100.0
                                    if statistics.fmean(series) > 0 else 0.0,
                "wall_s_series": [r["wall_s"] for r in reps],
                "wall_s_mean": statistics.fmean([r["wall_s"] for r in reps]),
                "completion_tokens_series": [r["completion_tokens"] for r in reps],
                "completion_tokens_mean": statistics.fmean([r["completion_tokens"] for r in reps]),
                "errors_total": sum(r["n_err"] for r in reps),
            }
            record["rungs"].append(rung)
            print(f"[{a.label}] C={conc:>3} SERIES {['%.2f' % s for s in series]} "
                  f"mean={rung['tok_s_mean']:.2f} spread={rung['tok_s_spread_pct']:.2f}%",
                  flush=True)
            # written after every rung so a crash never loses completed work
            with open(a.out, "w") as f:
                json.dump(record, f, indent=2)
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(a.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"# wrote {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
