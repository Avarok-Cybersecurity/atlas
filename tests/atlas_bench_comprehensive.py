#!/usr/bin/env python3
"""Comprehensive TQ+ KV-dtype bench matrix.

Runs each (dtype, env-knob-config) cell with:
  - PPL similarity on a fixed Manhattan-Project WikiText prompt
  - Decode-short tok/s (5 runs of 256-token completion)
  - Prefill tok/s at 2K / 8K / 16K (5 runs each, max_tokens=1)
  - Decode-after-8K tok/s (3 runs, wall-time incl. prefill)

Reports median per metric (per-run min/max stashed in JSON for reviewers
to compute their own variance).

Usage:
  python3 atlas_bench_comprehensive.py \\
    --image atlas-gb10-tqplus \\
    --model-path /home/pidtom/models/qwen3.6-35b-fp8 \\
    --config-name "tqplus-default" \\
    --dtypes fp8,nvfp4,bf16,turbo2,turbo3,turbo4,turbo8 \\
    --out /tmp/bench_tqplus_default.json

  python3 atlas_bench_comprehensive.py \\
    --image atlas-gb10-tqplus \\
    --model-path /home/pidtom/models/qwen3.6-35b-fp8 \\
    --config-name "tqplus-innerq" \\
    --dtypes turbo3 \\
    --env "TURBO_INNERQ=512" --env "TURBO_INNERQ_STRENGTH=0.5" \\
    --out /tmp/bench_tqplus_innerq.json
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request

PORT = 8889
CONTAINER = "atlas-bench"
HOST = "localhost"

# Single Manhattan-Project PPL prompt (matches /tmp/atlas_matrix_no_hp.py and
# the prior baseline JSONs so before/after deltas are apples-to-apples).
PPL_PROMPT = (
    "Continue this WikiText passage exactly, verbatim:\n\n"
    "The Manhattan Project was a research and development undertaking during World War II that "
    "produced the first nuclear weapons. It was led by the United States with the support of the "
    "United Kingdom and Canada. From 1942 to 1946, the project was under the direction of Major "
    "General Leslie Groves of the U.S. Army Corps of Engineers. Nuclear physicist "
)
PPL_REF = "J. Robert Oppenheimer was the director of the Los Alamos Laboratory that designed the actual bombs."


def sh(cmd, timeout=120):
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


def stop_container():
    sh(f"docker stop {CONTAINER} 2>/dev/null; docker rm {CONTAINER} 2>/dev/null; true")


def start_container(image, model_path, kv_dtype, env_kvs, max_seq_len=32768):
    stop_container()
    env_flags = " ".join(f"-e {kv}" for kv in env_kvs) if env_kvs else ""
    cmd = (
        f"docker run -d --name {CONTAINER} --gpus all --ipc=host "
        f"-p {PORT}:8888 -v {model_path}:/model "
        f"-e RUST_LOG=warn {env_flags} "
        f"{image} serve --model-from-path /model "
        f"--port 8888 --bind 0.0.0.0 --max-seq-len {max_seq_len} "
        f"--kv-cache-dtype {kv_dtype} --kv-high-precision-layers 0"
    )
    out = sh(cmd)
    return out.strip()[:80]


def wait_ready(timeout_s=180):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=2) as r:
                json.load(r)
            return True
        except Exception:
            time.sleep(2)
    return False


def call(prompt, max_tok=64, timeout=600):
    body = {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    wall = time.time() - t0
    u = d.get("usage", {})
    return {
        "wall": wall,
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "text": d["choices"][0]["message"]["content"],
        "ttft_ms": d.get("usage", {}).get("time_to_first_token_ms")
        or u.get("time_to_first_token_ms"),
    }


def build_long_prompt(target_chars):
    base = "Explain the history of transformer architectures, attention mechanisms, key-value caching strategies, and modern inference engines in great technical detail. Cover the original 2017 paper, Multi-Query Attention, Grouped-Query Attention, Multi-Latent Attention, sliding window attention, sink tokens, and YaRN scaling. "
    s = base
    while len(s) < target_chars:
        s += base
    return s[:target_chars]


# Target character counts roughly map to token counts at ~4 chars/token
# (English text). The harness reports the actual prompt_tokens from the
# server response — these constants only control the input string length.
CONTEXT_TARGETS = {
    "2k": 2000,
    "8k": 8000,
    "16k": 16000,
}


def median(xs):
    return statistics.median(xs) if xs else 0.0


def bench_dtype(dtype, image, model_path, env_kvs, runs_decode, runs_prefill, contexts):
    cid = start_container(image, model_path, dtype, env_kvs)
    if not cid:
        return {"error": "container_start_failed"}
    if not wait_ready():
        logs = sh(f"docker logs {CONTAINER} 2>&1 | tail -10")
        return {"error": "load_timeout", "logs_tail": logs[-500:]}

    out = {"env": list(env_kvs)}

    # 1. PPL similarity (single run — quality metric, not perf)
    try:
        r = call(PPL_PROMPT, max_tok=64)
        from difflib import SequenceMatcher
        sim = SequenceMatcher(None, r["text"].strip()[: len(PPL_REF)], PPL_REF).ratio()
        out["ppl_sim"] = round(sim, 4)
        out["ppl_text"] = r["text"].strip()[:120]
    except Exception as e:
        out["ppl_error"] = str(e)[:200]

    # 2. Decode short tok/s (5 runs, max_tok=256)
    speeds = []
    for _ in range(runs_decode):
        try:
            r = call("Explain attention.", max_tok=256)
            if r["wall"] > 0 and r["completion_tokens"] > 0:
                speeds.append(r["completion_tokens"] / r["wall"])
        except Exception as e:
            print(f"    decode_short error: {e}", flush=True)
    out["decode_short_tok_s"] = {
        "median": round(median(speeds), 2),
        "min": round(min(speeds), 2) if speeds else 0,
        "max": round(max(speeds), 2) if speeds else 0,
        "n": len(speeds),
    }

    # 3. Prefill tok/s at each context length (max_tok=1)
    out["prefill"] = {}
    for label, chars in contexts.items():
        prompt = build_long_prompt(chars)
        walls = []
        prompt_tokens = 0
        for _ in range(runs_prefill):
            try:
                r = call(prompt, max_tok=1, timeout=120)
                walls.append(r["wall"])
                prompt_tokens = max(prompt_tokens, r["prompt_tokens"])
            except Exception as e:
                print(f"    prefill {label} error: {e}", flush=True)
        if walls:
            med_wall = median(walls)
            out["prefill"][label] = {
                "tok_s_median": round(prompt_tokens / med_wall, 2) if med_wall > 0 else 0,
                "prompt_tokens": prompt_tokens,
                "wall_median_s": round(med_wall, 3),
                "wall_min_s": round(min(walls), 3),
                "wall_max_s": round(max(walls), 3),
                "n": len(walls),
            }

    # 4. Decode-after-8K wall-time (3 runs, max_tok=128)
    prompt_8k = build_long_prompt(8000)
    speeds = []
    for _ in range(3):
        try:
            r = call(prompt_8k, max_tok=128, timeout=180)
            if r["wall"] > 0 and r["completion_tokens"] > 0:
                speeds.append(r["completion_tokens"] / r["wall"])
        except Exception as e:
            print(f"    dec_after_8k error: {e}", flush=True)
    out["decode_after_8k_tok_s"] = {
        "median": round(median(speeds), 2),
        "min": round(min(speeds), 2) if speeds else 0,
        "max": round(max(speeds), 2) if speeds else 0,
        "n": len(speeds),
    }

    return out


def format_table(results, config_name):
    """Render the result dict as a markdown table on stdout."""
    rows = []
    for dt, r in results.items():
        if "error" in r:
            rows.append(f"| {dt} | ERROR: {r['error']} |")
            continue
        ppl = r.get("ppl_sim", "—")
        dec_s = r.get("decode_short_tok_s", {}).get("median", "—")
        p2k = r.get("prefill", {}).get("2k", {}).get("tok_s_median", "—")
        p8k = r.get("prefill", {}).get("8k", {}).get("tok_s_median", "—")
        p16k = r.get("prefill", {}).get("16k", {}).get("tok_s_median", "—")
        dec_a8k = r.get("decode_after_8k_tok_s", {}).get("median", "—")
        rows.append(f"| {dt:<14} | {ppl} | {dec_s} | {p2k} | {p8k} | {p16k} | {dec_a8k} |")
    print(f"\n## {config_name}\n")
    print("| dtype | PPL sim | dec_short | pre_2K | pre_8K | pre_16K | dec_after_8K |")
    print("|-------|--------:|----------:|-------:|-------:|--------:|-------------:|")
    for row in rows:
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config-name", required=True, help="label for this env-knob config")
    ap.add_argument("--dtypes", required=True, help="comma-separated dtype list")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL env var (repeatable)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--runs-decode", type=int, default=5)
    ap.add_argument("--runs-prefill", type=int, default=5)
    ap.add_argument(
        "--contexts",
        default="2k,8k,16k",
        help="comma list of context labels to bench (subset of 2k,8k,16k)",
    )
    args = ap.parse_args()

    contexts = {k: CONTEXT_TARGETS[k] for k in args.contexts.split(",") if k in CONTEXT_TARGETS}
    print(
        f"Config: {args.config_name} | env: {args.env or 'none'} | "
        f"dtypes: {args.dtypes} | contexts: {list(contexts)}",
        flush=True,
    )

    results = {"_meta": {"config_name": args.config_name, "env": args.env, "image": args.image}}
    for dt in args.dtypes.split(","):
        dt = dt.strip()
        print(f"\n=== {dt} ({args.config_name}) ===", flush=True)
        r = bench_dtype(
            dt, args.image, args.model_path, args.env,
            args.runs_decode, args.runs_prefill, contexts,
        )
        results[dt] = r
        print(f"  {dt}: {json.dumps({k: v for k, v in r.items() if k != 'env'}, indent=2)[:400]}", flush=True)

    stop_container()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nartifact: {args.out}", flush=True)
    format_table({k: v for k, v in results.items() if not k.startswith("_")}, args.config_name)


if __name__ == "__main__":
    main()
