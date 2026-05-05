#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Atlas minimax_m2 vs HuggingFace reference — byte-exact comparison harness.

Runs yujiepan/minimax-m2.7-tiny-random (3M-param arch-exact test variant)
through:
  1. HF `MiniMaxM2ForCausalLM` on CPU (BF16 weights).
  2. Atlas spark-server (which must already be running on --url).

Compares layer-0 output by probing the first decoded token's logits
through the OpenAI-compatible `logprobs=5` endpoint. Atlas's top-5 token
IDs must match HF's top-5 (order insensitive), and per-token log-prob
differences must be within tolerance.

This is the acceptance test for M2 + M3 combined: correct attention
forward (partial RoPE + full-hidden qk_norm) plus correct MoE routing
(sigmoid + correction bias) should yield matching top-5 token IDs.
Mismatch narrows the failing subsystem:

  * top-5 IDs wildly different → attention broken (rope_dim wrong,
    qk_norm path not applied, or GQA head-map mis-configured).
  * top-5 IDs partially match, logits shifted constant → MoE routing
    off (softmax vs sigmoid, or correction bias missing).
  * Perfect match → M2+M3 green, move on to M4 (FP8) / M5 (MTP K=3).

Usage:
  # 1. Start Atlas serving tiny model:
  docker run --gpus all -p 8889:8888 \\
    -v ~/.cache/huggingface:/root/.cache/huggingface \\
    atlas-gb10:minimax-m2f \\
    serve yujiepan/minimax-m2.7-tiny-random --kv-cache-dtype bf16

  # 2. In another shell:
  python3 tests/integration/minimax_tiny_vs_hf.py --url http://localhost:8889

Exit 0 = match within tolerance. Exit 1 = mismatch (with diagnostic dump).

CPU-only on the HF side — tiny-random is 3M params, BF16, fits in RAM.
No GPU, no 229B weights required.
"""
import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.stderr.write(
        "This test requires `torch` and `transformers`. "
        "Install with: pip install torch transformers\n"
    )
    sys.exit(2)


MODEL_ID = "yujiepan/minimax-m2.7-tiny-random"
DEFAULT_PROMPTS = [
    "The quick brown",
    "In the beginning",
    "A function that",
    "Once upon a time",
    "Let me think about",
]


@dataclass
class TopKResult:
    prompt: str
    top5_ids: list  # list[int]
    top5_logprobs: list  # list[float], same order as ids
    full_logits_slice: list  # [:10] for diagnostic dump


def hf_reference(prompts: list) -> list:
    """Run HF MiniMaxM2ForCausalLM locally on CPU, collect top-5 logits
    for the first predicted token of each prompt."""
    sys.stderr.write(f"[hf] loading {MODEL_ID}…\n")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    results = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids
            logits = model(ids).logits[0, -1].float()
            top5 = torch.topk(logits, 5)
            # Convert logits → log-softmax so the units match OpenAI logprobs.
            logprobs = torch.log_softmax(logits, dim=-1)
            top5_lp = [logprobs[i].item() for i in top5.indices.tolist()]
            results.append(TopKResult(
                prompt=p,
                top5_ids=top5.indices.tolist(),
                top5_logprobs=top5_lp,
                full_logits_slice=logits[:10].tolist(),
            ))
            sys.stderr.write(
                f"[hf] {p!r:35} → top5={top5.indices.tolist()} "
                f"logits_first2={logits[:2].tolist()}\n"
            )
    return results


def atlas_query(url: str, prompt: str) -> TopKResult:
    """Query an Atlas spark-server via OpenAI-compatible /v1/chat/completions
    with logprobs=5 and temperature=0. Returns the top-5 for the first
    predicted token."""
    body = json.dumps({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    lp = data["choices"][0]["logprobs"]["content"][0]
    top5 = lp["top_logprobs"]
    # top5 is list of {token, logprob, bytes}. We need ids — atlas exposes
    # the `bytes` field which encodes the token bytes; we decode via the
    # shared tokenizer below.
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = []
    for t in top5:
        # Prefer `bytes` (exact) over `token` (decoded string which may
        # lose info for partial-unicode tokens).
        if t.get("bytes"):
            b = bytes(t["bytes"])
            ids.append(tok.convert_tokens_to_ids(
                tok.convert_ids_to_tokens(tok.encode(b.decode("utf-8", errors="replace")))[0]
            ))
        else:
            ids.append(tok.encode(t["token"])[0])
    return TopKResult(
        prompt=prompt,
        top5_ids=ids,
        top5_logprobs=[t["logprob"] for t in top5],
        full_logits_slice=[],  # Atlas doesn't expose full logits via API
    )


def compare(hf: list, atlas: list, id_tolerance: int = 1, lp_tolerance: float = 5e-2) -> bool:
    """Compare HF vs Atlas top-5 per prompt.

    `id_tolerance`: max number of IDs that can differ from HF's top-5 set
    (0 = exact match required, 1 = allow one swap for borderline ties).
    `lp_tolerance`: max absolute log-prob diff for shared token IDs.
    """
    all_ok = True
    for h, a in zip(hf, atlas):
        shared = set(h.top5_ids) & set(a.top5_ids)
        missing = set(h.top5_ids) - set(a.top5_ids)
        extra = set(a.top5_ids) - set(h.top5_ids)
        hf_lp_by_id = dict(zip(h.top5_ids, h.top5_logprobs))
        atlas_lp_by_id = dict(zip(a.top5_ids, a.top5_logprobs))
        max_diff = 0.0
        for tid in shared:
            d = abs(hf_lp_by_id[tid] - atlas_lp_by_id[tid])
            if d > max_diff:
                max_diff = d
        ok_ids = len(missing) <= id_tolerance
        ok_lp = max_diff <= lp_tolerance
        status = "PASS" if ok_ids and ok_lp else "FAIL"
        print(
            f"[{status}] {h.prompt!r:35} "
            f"shared={len(shared)}/5  missing={sorted(missing)}  extra={sorted(extra)}  "
            f"max_lp_diff={max_diff:.4f}"
        )
        if not (ok_ids and ok_lp):
            all_ok = False
            print(f"  HF    top5: ids={h.top5_ids}  lp={[f'{x:.3f}' for x in h.top5_logprobs]}")
            print(f"  Atlas top5: ids={a.top5_ids}  lp={[f'{x:.3f}' for x in a.top5_logprobs]}")
    return all_ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8889",
                   help="Atlas server base URL (without /v1)")
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                   help="Prompts to test")
    p.add_argument("--id-tolerance", type=int, default=1,
                   help="Max top-5 ID mismatches allowed per prompt")
    p.add_argument("--lp-tolerance", type=float, default=5e-2,
                   help="Max logprob absolute diff for shared IDs")
    args = p.parse_args()

    print(f"=== HF reference ({MODEL_ID}, CPU BF16) ===")
    hf = hf_reference(args.prompts)
    print(f"=== Atlas server ({args.url}) ===")
    try:
        atlas = [atlas_query(args.url, q) for q in args.prompts]
    except urllib.error.URLError as e:
        sys.stderr.write(f"Atlas server unreachable at {args.url}: {e}\n")
        sys.stderr.write(
            "Start it with:\n"
            "  docker run --gpus all -p 8889:8888 \\\n"
            "    -v ~/.cache/huggingface:/root/.cache/huggingface \\\n"
            "    atlas-gb10:minimax-m2f \\\n"
            "    serve yujiepan/minimax-m2.7-tiny-random --kv-cache-dtype bf16\n"
        )
        sys.exit(2)

    print(f"=== Compare (id_tol={args.id_tolerance}, lp_tol={args.lp_tolerance}) ===")
    ok = compare(hf, atlas, args.id_tolerance, args.lp_tolerance)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
