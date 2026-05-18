#!/usr/bin/env python3
"""N>=10 statistical long-code degeneration harness for Atlas.

Drives the canonical 3D-three.js-chess stress prompt N times at a FIXED
list of seeds (paired comparison: the same N stochastic trajectories
before/after a change), streams the OpenAI-compatible endpoint, and
scores each sample via analyze.mjs (node+acorn AST).

PCND: every knob (url, model, n, seeds, temp, max_tokens, label) is an
explicit CLI arg with no behavioural default — the gate must not depend
on a hidden constant. The only fixed value is the prompt, which is the
SSOT in prompts/chess3d.txt and must stay frozen for cross-run
comparability.

Gate: completeness_pass >= PASS_GATE of N  -> exit 0, else exit 1.
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
PROMPT = (HERE / "prompts" / "chess3d.txt").read_text().strip()
PASS_GATE_FRAC = 0.8  # >=8/10; see plan Phase 0 gate definition.


def stream_one(url: str, model: str, seed: int, temp: float, max_tokens: int):
    """One streamed completion. Returns (reasoning, content, finish, stats)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temp,
        "seed": seed,  # paired comparison; harmless if server ignores it.
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    reasoning, content, finish = [], [], None
    t0 = time.time()
    first_t = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = obj.get("choices", [{}])[0]
            d = ch.get("delta", {})
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
                first_t = first_t or time.time()
            if d.get("content"):
                content.append(d["content"])
                first_t = first_t or time.time()
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            n_chunks += 1
    return (
        "".join(reasoning),
        "".join(content),
        finish,
        {
            "elapsed_s": round(time.time() - t0, 1),
            "ttft_s": round((first_t - t0), 2) if first_t else None,
            "chunks": n_chunks,
        },
    )


def analyze(sample_path: pathlib.Path) -> dict:
    """Invoke analyze.mjs on a raw sample JSON; return its metrics dict."""
    out = subprocess.run(
        ["node", str(HERE / "analyze.mjs"), str(sample_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"analyze.mjs failed: {out.stderr.strip()}")
    return json.loads(out.stdout)


def agg(vals):
    nums = [v for v in vals if isinstance(v, (int, float))]
    if not nums:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": round(statistics.fmean(nums), 2),
        "std": round(statistics.pstdev(nums), 2) if len(nums) > 1 else 0.0,
        "n": len(nums),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888/v1/chat/completions")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument(
        "--label",
        required=True,
        help="run label, e.g. baseline-mtp-off / phase1a-mtp-on",
    )
    ap.add_argument("--outdir", default=str(HERE / "results"))
    args = ap.parse_args()

    seeds = list(range(1, args.n + 1))  # fixed -> paired across runs
    outdir = pathlib.Path(args.outdir) / args.label
    outdir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in seeds:
        print(f"[{args.label}] seed={seed} ...", flush=True)
        try:
            R, C, fin, st = stream_one(
                args.url, args.model, seed, args.temp, args.max_tokens
            )
        except Exception as e:  # network/timeout -> record as hard failure
            per_seed.append(
                {"seed": seed, "error": str(e), "completeness_pass": False}
            )
            print(f"  seed={seed} ERROR {e}", flush=True)
            continue
        sp = outdir / f"seed{seed}.json"
        sp.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "finish_reason": fin,
                    "reasoning": R,
                    "content": C,
                    "stats": st,
                },
                ensure_ascii=False,
            )
        )
        m = analyze(sp)
        m["seed"] = seed
        m["finish_reason"] = fin
        m["stats"] = st
        per_seed.append(m)
        print(
            f"  seed={seed} pass={m.get('completeness_pass')} "
            f"valid_js_lines={m.get('valid_js_line_count')} "
            f"dup_decl={m.get('duplicate_declaration_count')} "
            f"deg@{m.get('tokens_to_first_degeneration')} fin={fin}",
            flush=True,
        )

    npass = sum(1 for s in per_seed if s.get("completeness_pass"))
    summary = {
        "label": args.label,
        "n": args.n,
        "temp": args.temp,
        "seeds": seeds,
        "completeness_pass_count": npass,
        "gate": f"{npass}/{args.n} (need >= {int(PASS_GATE_FRAC * args.n)})",
        "valid_js_line_count": agg(
            [s.get("valid_js_line_count") for s in per_seed]
        ),
        "duplicate_declaration_count": agg(
            [s.get("duplicate_declaration_count") for s in per_seed]
        ),
        "tokens_to_first_degeneration": agg(
            [s.get("tokens_to_first_degeneration") for s in per_seed]
        ),
        "finish_reasons": {
            fr: sum(1 for s in per_seed if s.get("finish_reason") == fr)
            for fr in {s.get("finish_reason") for s in per_seed}
        },
        "per_seed": per_seed,
    }
    summ_path = pathlib.Path(args.outdir) / f"{args.label}.summary.json"
    summ_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + "=" * 64)
    print(f"LABEL {args.label}  GATE {summary['gate']}")
    print(f"  valid_js_line_count        {summary['valid_js_line_count']}")
    print(f"  duplicate_declaration_count{summary['duplicate_declaration_count']}")
    print(f"  tokens_to_first_degen      {summary['tokens_to_first_degeneration']}")
    print(f"  finish_reasons             {summary['finish_reasons']}")
    print(f"  summary -> {summ_path}")
    print("=" * 64)
    return 0 if npass >= int(PASS_GATE_FRAC * args.n) else 1


if __name__ == "__main__":
    sys.exit(main())
