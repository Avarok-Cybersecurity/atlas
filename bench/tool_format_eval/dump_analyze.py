# SPDX-License-Identifier: AGPL-3.0-only
"""Tool-format eval harness — Layer 3b/4: ATLAS_LOGIT_DUMP analysis + Atlas↔vLLM diff.

Consumes the JSONL written by `ATLAS_LOGIT_DUMP=<file>` (see
crates/spark-server/src/scheduler/logit_dump.rs). Each record has:
    step, in_body, chars, raw_topk:[[id,logit]...], bias:[[id,delta]...],
    post_argmax, sampled
`raw_topk` is the model's OWN distribution before Atlas's additive bias stack;
`bias` is the Atlas-only processor delta (vLLM applies none).

Two modes:

  single  — localize each step within ONE Atlas dump:
              * does Atlas's `bias` flip the argmax vs raw?  => PROCESSOR divergence
              * is the sampled token NOT the raw argmax?      => sampler (min_p/penalty)
              * otherwise the pick == model's own argmax       => MODEL distribution
            Tells you whether the wrong token is the model's own top pick or an
            Atlas post-processor artifact — without an external reference.

  diff    — compare an Atlas dump vs a vLLM dump (same teacher-forced tokens):
              per-step top-1 agreement on `raw_topk` + symmetric divergence of
              the shared top-K. raw_topk DIFFERS => model/inference (forward pass)
              divergence; raw_topk MATCHES but argmax flips => Atlas bias. Mirrors
              tests/metal_kv_kld_compare.py's stop-at-first-divergence logic.

Produce the vLLM reference with vLLM's patched sampler (same raw_topk+sampled
shape, per logit_dump.rs) or `prompt_logprobs` over the identical forced prefix.

Usage:
  ATLAS_LOGIT_DUMP=/tmp/atlas.jsonl ./target/release/spark serve ...   # then 1 request
  python3 bench/tool_format_eval/dump_analyze.py single /tmp/atlas.jsonl
  python3 bench/tool_format_eval/dump_analyze.py diff /tmp/atlas.jsonl /tmp/vllm.jsonl
"""
import argparse
import json
import math


def load(path):
    # The dump formats f32 ±inf as bare `inf`/`-inf` (Rust Display), which is not
    # valid JSON. These are Atlas's hard-mask bias deltas (WS mask / think
    # suppression). Map to large finite sentinels so argmax math still reflects
    # "hard-masked" / "force".
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace("-inf", "-1e30").replace("inf", "1e30").replace("nan", "0")
            recs.append(json.loads(line))
    return recs


def raw_argmax(rec):
    tk = rec["raw_topk"]
    return max(tk, key=lambda p: p[1])[0]


def post_bias_argmax(rec):
    bias = {b[0]: b[1] for b in rec.get("bias", [])}
    best, bid = -1e30, None
    for tid, lg in rec["raw_topk"]:
        v = lg + bias.get(tid, 0.0)
        if v > best:
            best, bid = v, tid
    return bid


def single(path):
    recs = load(path)
    n = len(recs)
    model_pick = bias_flip = sampler_pick = 0
    print(f"# single-dump localization — {n} steps from {path}\n")
    print(f"{'step':>4} {'in_body':>7} {'raw_argmax':>10} {'post_bias':>9} "
          f"{'sampled':>7}  verdict")
    for r in recs:
        ra = raw_argmax(r)
        pb = post_bias_argmax(r)
        sm = r["sampled"]
        if pb != ra:
            verdict = "BIAS flips argmax (Atlas processor)"
            bias_flip += 1
        elif sm != ra:
            verdict = "sampler picked non-argmax (min_p/penalty/temp)"
            sampler_pick += 1
        else:
            verdict = "model's own argmax"
            model_pick += 1
        if r["step"] < 24 or pb != ra or sm != ra:
            print(f"{r['step']:>4} {str(r['in_body']):>7} {ra:>10} {pb:>9} {sm:>7}  {verdict}")
    print(f"\nmodel-argmax picks: {model_pick}  | bias-flips: {bias_flip}  "
          f"| sampler picks: {sampler_pick}  (of {n})")
    print("bias-flips>0 => Atlas's additive processor stack is steering tokens (vLLM has none).")
    print("else the wrong tokens are the MODEL's own argmax => forward-pass/quant; diff vs vLLM.")


def topk_map(rec):
    return {p[0]: p[1] for p in rec["raw_topk"]}


def sym_div(a, b):
    """Symmetric mean |logit| gap over shared top-K ids (proxy for divergence)."""
    shared = set(a) & set(b)
    if not shared:
        return float("nan")
    return sum(abs(a[i] - b[i]) for i in shared) / len(shared)


def diff(atlas_path, vllm_path):
    A, V = load(atlas_path), load(vllm_path)
    n = min(len(A), len(V))
    agree = 0
    compared = 0
    print(f"# Atlas↔vLLM raw_topk diff — comparing {n} aligned steps\n")
    print(f"{'step':>4} {'atlas_argmax':>12} {'vllm_argmax':>11} {'top1':>5} {'sharedΔ':>8}")
    for i in range(n):
        aa, va = raw_argmax(A[i]), raw_argmax(V[i])
        same = aa == va
        agree += int(same)
        compared += 1
        d = sym_div(topk_map(A[i]), topk_map(V[i]))
        print(f"{i:>4} {aa:>12} {va:>11} {'=' if same else 'X':>5} {d:>8.3f}")
        if not same:
            print(f"     -> FIRST top-1 divergence at step {i}: "
                  f"raw_topk differs => MODEL/inference (forward-pass) divergence.")
            break
    print(f"\ntop-1 agreement: {agree}/{compared} up to first divergence.")
    print("high agreement => Atlas inference matches vLLM (look at bias/sampler/parsing).")
    print("early divergence => Atlas forward pass (quant kernels/KV) differs from vLLM.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("single"); s.add_argument("dump")
    df = sub.add_parser("diff"); df.add_argument("atlas"); df.add_argument("vllm")
    a = ap.parse_args()
    if a.mode == "single":
        single(a.dump)
    else:
        diff(a.atlas, a.vllm)


if __name__ == "__main__":
    main()
