#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
GDN HeadParallel golden gate — TP=2 (loopback) vs TP=1 long-sequence parity.

WHY THIS EXISTS
---------------
Gated-DeltaNet (linear_attention / SSM) layers are STATEFUL: the recurrence
carries `h_state` + `conv_state` across every token. A layout / head-slicing /
all-reduce bug in the GDN HeadParallel tensor-parallel path does NOT show up on
a short prompt — the state has to accumulate over hundreds of tokens before a
mis-sliced value head or a missing out_proj all-reduce visibly diverges. So the
gate here is a LONG prompt (>~1k tokens) plus a long greedy generation, compared
between a TP=1 reference and a TP=2 (two ranks, one GPU, NCCL loopback) run.

Correct GDN HeadParallel ⇒ the two runs produce the IDENTICAL greedy token
sequence, and near-identical top-token logprobs (the out_proj all-reduce sums
partial products in a different float order per rank, so expect ~1e-2, not bit
exact).

WHAT IT CHECKS
--------------
  1. Greedy (temperature=0) generated token sequence is identical (HARD gate).
  2. Per-position top-1 logprob max-abs-diff < --logprob-tol (soft numeric gate).

TWO WAYS TO RUN
---------------
(A) Point it at two servers you launched yourself (recommended on a shared box):

      python3 scripts/gdn_tp_golden.py \
          --url-tp1 http://127.0.0.1:8800 \
          --url-tp2 http://127.0.0.1:8801

(B) Let it launch both servers on ONE GB10 (TP=2 = two ranks pinned to the same
    GPU → NCCL loopback). Build the server first:

      export LIBRARY_PATH=/home/ms/nccl/build/lib CUDARC_CUDA_VERSION=12000 \
             ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b \
             ATLAS_TARGET_QUANT=nvfp4
      cargo build --release -p spark-server --bin spark

      python3 scripts/gdn_tp_golden.py --launch \
          --model <HF_ID_or_local_path_of_a_qwen3.5/3.6_SSM_model> \
          --bin target/release/spark

    NOTE (single-GB10 memory): TP shards attention + GDN weights, but with
    ep_size=1 the MoE experts are REPLICATED per rank, so two ranks on one GPU
    roughly double MoE-resident memory. Use a model that fits 2× on the GB10,
    or lower --gpu-mem-util, or run the two ranks on two GPUs. FP8-native SSM
    checkpoints force TP=1 by design (block-scale slicing deferred) — use a BF16
    or NVFP4 SSM checkpoint for the TP=2 side.

EXIT CODE: 0 = parity PASS, 1 = FAIL (token mismatch or logprob drift).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


# A deterministic, self-contained long prompt. Repeated to force a multi-chunk
# prefill (>1k tokens) so the GDN state genuinely accumulates before we diff.
_PARAGRAPH = (
    "The recurrence in a gated delta network threads a hidden state and a "
    "convolution state through every position, so any tensor-parallel head "
    "slicing bug compounds silently over a long context rather than failing "
    "loudly on the first token. Consider the counting sequence carefully: "
    "one, two, three, four, five, six, seven, eight, nine, ten. "
)


def build_prompt(repeat: int) -> str:
    body = _PARAGRAPH * repeat
    return (
        body
        + "\n\nContinue the numeric analysis precisely and deterministically, "
        "step by step, without repeating yourself:\n"
    )


def http_json(url: str, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_healthy(base_url: str, timeout_s: int) -> str:
    """Poll /v1/models until the server answers; return the model id."""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = http_json(f"{base_url}/v1/models", timeout=10)
            return r["data"][0]["id"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2.0)
    raise RuntimeError(f"server at {base_url} not healthy after {timeout_s}s: {last_err}")


def sample(base_url: str, model: str, prompt: str, max_tokens: int):
    """Greedy completion with per-token top-1 logprobs."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "logprobs": 1,  # OpenAI completions: top-1 logprob per generated token
        "stream": False,
    }
    r = http_json(f"{base_url}/v1/completions", payload)
    choice = r["choices"][0]
    text = choice.get("text", "")
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or []
    token_logprobs = lp.get("token_logprobs") or []
    return text, tokens, token_logprobs


def compare(a, b, logprob_tol: float) -> bool:
    text_a, tok_a, lpa = a
    text_b, tok_b, lpb = b

    print("\n=== GDN HeadParallel TP golden gate ===")
    print(f"TP=1 tokens: {len(tok_a)}   TP=2 tokens: {len(tok_b)}")

    ok = True

    # HARD gate 1: identical greedy token sequence.
    if tok_a and tok_b:
        n = min(len(tok_a), len(tok_b))
        first_div = next((i for i in range(n) if tok_a[i] != tok_b[i]), None)
        if len(tok_a) != len(tok_b) or first_div is not None:
            ok = False
            where = first_div if first_div is not None else n
            print(f"  FAIL: greedy token sequences diverge at position {where}")
            print(f"    TP=1[{where}]={tok_a[where] if where < len(tok_a) else '<eos>'!r}")
            print(f"    TP=2[{where}]={tok_b[where] if where < len(tok_b) else '<eos>'!r}")
        else:
            print(f"  PASS: {len(tok_a)} greedy tokens identical")
    else:
        # Fallback when the server does not echo per-token logprobs: compare text.
        if text_a != text_b:
            ok = False
            print("  FAIL: generated text differs (no per-token logprobs available)")
            print(f"    TP=1: {text_a[:160]!r}")
            print(f"    TP=2: {text_b[:160]!r}")
        else:
            print(f"  PASS: generated text identical ({len(text_a)} chars)")

    # SOFT gate 2: numeric closeness of top-1 logprobs.
    if lpa and lpb:
        n = min(len(lpa), len(lpb))
        diffs = [
            abs(x - y)
            for x, y in zip(lpa[:n], lpb[:n])
            if x is not None and y is not None
        ]
        if diffs:
            mx, mean = max(diffs), sum(diffs) / len(diffs)
            status = "PASS" if mx <= logprob_tol else "FAIL"
            if mx > logprob_tol:
                ok = False
            print(
                f"  {status}: top-1 logprob drift  max={mx:.4e}  mean={mean:.4e}  "
                f"(tol={logprob_tol:.2e})"
            )

    print("=== " + ("GOLDEN GATE PASS" if ok else "GOLDEN GATE FAIL") + " ===\n")
    return ok


def spawn_server(bin_path, model, *, rank, world_size, tp_size, port,
                 master_port, gpu_mem_util, log_path):
    cmd = [
        bin_path, "serve", model,
        "--port", str(port),
        "--rank", str(rank),
        "--world-size", str(world_size),
        "--tp-size", str(tp_size),
        "--ep-size", "1",
        "--master-addr", "127.0.0.1",
        "--master-port", str(master_port),
        "--gpu-memory-utilization", str(gpu_mem_util),
    ]
    env = dict(os.environ)
    # Both ranks share the single GB10 → NCCL loopback.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    log = open(log_path, "w")
    print(f"  launch rank{rank}/{world_size} tp={tp_size} port={port} -> {log_path}")
    # setsid so the whole server (+ its threads) is killable as one group.
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True), log


def launch_mode(args):
    procs = []
    try:
        # --- TP=1 reference: single process, world_size=1 ---
        p1, _ = spawn_server(
            args.bin, args.model, rank=0, world_size=1, tp_size=1,
            port=args.port_tp1, master_port=args.master_port,
            gpu_mem_util=args.gpu_mem_util, log_path="/tmp/gdn_tp_golden_tp1.log")
        procs.append(p1)
        model_id = wait_healthy(f"http://127.0.0.1:{args.port_tp1}", args.startup_timeout)
        ref = sample(f"http://127.0.0.1:{args.port_tp1}", model_id,
                     build_prompt(args.prompt_repeat), args.max_tokens)
        # Free the reference server before bringing up two ranks (memory).
        p1.terminate()
        procs.remove(p1)
        try:
            p1.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p1.kill()

        # --- TP=2 loopback: rank0 (HTTP) + rank1 (worker), same GPU ---
        pr0, _ = spawn_server(
            args.bin, args.model, rank=0, world_size=2, tp_size=2,
            port=args.port_tp2, master_port=args.master_port + 1,
            gpu_mem_util=args.gpu_mem_util, log_path="/tmp/gdn_tp_golden_tp2_rank0.log")
        procs.append(pr0)
        pr1, _ = spawn_server(
            args.bin, args.model, rank=1, world_size=2, tp_size=2,
            port=args.port_tp2 + 1, master_port=args.master_port + 1,
            gpu_mem_util=args.gpu_mem_util, log_path="/tmp/gdn_tp_golden_tp2_rank1.log")
        procs.append(pr1)
        model_id2 = wait_healthy(f"http://127.0.0.1:{args.port_tp2}", args.startup_timeout)
        got = sample(f"http://127.0.0.1:{args.port_tp2}", model_id2,
                     build_prompt(args.prompt_repeat), args.max_tokens)

        return compare(ref, got, args.logprob_tol)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()


def url_mode(args):
    m1 = wait_healthy(args.url_tp1, args.startup_timeout)
    m2 = wait_healthy(args.url_tp2, args.startup_timeout)
    prompt = build_prompt(args.prompt_repeat)
    ref = sample(args.url_tp1, m1, prompt, args.max_tokens)
    got = sample(args.url_tp2, m2, prompt, args.max_tokens)
    return compare(ref, got, args.logprob_tol)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launch", action="store_true",
                    help="launch both servers on one GB10 (TP=2 = NCCL loopback)")
    ap.add_argument("--model", help="HF id or local path (required with --launch)")
    ap.add_argument("--bin", default="target/release/spark", help="spark server binary")
    ap.add_argument("--url-tp1", default="http://127.0.0.1:8800")
    ap.add_argument("--url-tp2", default="http://127.0.0.1:8801")
    ap.add_argument("--port-tp1", type=int, default=8800)
    ap.add_argument("--port-tp2", type=int, default=8801)
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    ap.add_argument("--prompt-repeat", type=int, default=90,
                    help="paragraph repeats (~1.5k+ prompt tokens by default)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="greedy generation length (>=250 per house rule)")
    ap.add_argument("--logprob-tol", type=float, default=5e-2)
    ap.add_argument("--startup-timeout", type=int, default=600)
    args = ap.parse_args()

    if args.launch:
        if not args.model:
            ap.error("--launch requires --model")
        ok = launch_mode(args)
    else:
        ok = url_mode(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
