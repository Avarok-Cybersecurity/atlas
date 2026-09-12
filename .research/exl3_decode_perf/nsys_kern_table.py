#!/usr/bin/env python3
"""Turn an `nsys stats --report cuda_gpu_kern_sum --format csv` file into the two tables the
prefill decision needs: per-kernel share of captured GPU time, and per-FAMILY share (EXL3 MoE
fused tier, EXL3 overflow tier, EXL3 dense K=6, QSA, GDN, cuBLASLt, ...). The family map is
the one the decision thresholds in MEASUREMENT_PLAN.md are written against.

Same pattern as the decode work in EXL3_DECODE_PERF.md (kernel table from cuda_gpu_kern_sum),
with the family roll-up added because the prefill question is "which tier", not "which kernel".

    nsys_kern_table.py <kern_sum.csv> [--api <api_sum.csv>] [--wall-s <captured wall>] [--top N]
"""
import argparse, csv, re, sys

# Ordered: first match wins. Names are the __global__ symbols in kernels/gb10 (cutlass/cuBLASLt
# kernels arrive as mangled template names, hence the loose patterns at the bottom).
FAMILIES = [
    ("EXL3 MoE fused tier (<=128 rows/expert)", r"^exl3_moe_k\d"),
    ("EXL3 MoE overflow tier GEMM (K=4, >128 rows/expert)", r"^exl3_gemm_k4_"),
    ("EXL3 MoE glue (gather/store/reduce/stage/silu)", r"^exl3_moe_(gather|store|scatter|reduce|stage|replicate)|^exl3_silu_mul"),
    ("EXL3 routed decode mgemm (the 1 decode token)", r"^exl3_mgemm_"),
    ("EXL3 dense GEMM (GDN/attn/lm_head, K=6)", r"^exl3_gemm_k\d|^exl3_gemv_"),
    ("EXL3 converters (f16/bf16/f32)", r"^exl3_(f16|bf16|f32)_to|^convert_f32_to_bf16|^f32_to_bf16"),
    ("QSA sparse attention", r"^qsa_"),
    ("dense attention (prefill/paged/rope)", r"inferspark_prefill|prefill_attn|paged_decode_attn|fused_k_norm_rope|PAGED_CONCAT"),
    ("GDN (delta rule, conv, gated norm)", r"gated_delta_rule|causal_conv1d|compute_gdn_gates|^gdn_|gated_rms_norm|deinterleave_qg|^l2_norm"),
    ("mHC glue (hc_*)", r"^hc_|qhc_"),
    ("cuBLASLt / BF16 GEMM (mHC collapse, router)", r"cutlass|cublas|nvjet|Kernel2|splitK|gemm.*bf16|dense_gemm_bf16"),
    ("router / sort / top-k / blend", r"^moe_(sort|count|permute|topk|gate_topk|hash_route|batched_blend|weighted_sum|unpermute|build_tile|zero_expert|silu_mul)"),
    ("NVFP4 shared expert (w4a16)", r"w4a16|^moe_expert_|nvfp4|e2m1"),
    ("PLE / embedding", r"^ple_|batched_embed|^embed_"),
    ("MTP / DFlash draft", r"dflash|mtp"),
    ("memcpy/memset (as kernels)", r"memcpy|memset|Memcpy|Memset"),
]


def family_of(name):
    for fam, pat in FAMILIES:
        if re.search(pat, name):
            return fam
    return "other"


def read_kern_sum(path):
    rows = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                rows.append({
                    "name": r["Name"],
                    "total_ns": float(r["Total Time (ns)"]),
                    "inst": int(float(r["Instances"])),
                    "avg_ns": float(r["Avg (ns)"]),
                    "med_ns": float(r["Med (ns)"]),
                })
            except (KeyError, ValueError) as e:
                print(f"skip row {r}: {e}", file=sys.stderr)
    return rows


def read_api_sum(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append((r["Name"], float(r["Total Time (ns)"]), int(float(r["Num Calls"]))))
            except (KeyError, ValueError):
                pass
    return out


def short(name, n=70):
    name = re.sub(r"\(.*$", "", name)  # drop template arg lists
    name = name.replace("void ", "")
    return name if len(name) <= n else name[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kern_csv")
    ap.add_argument("--api", help="cuda_api_sum csv (optional)")
    ap.add_argument("--wall-s", type=float, default=0.0, help="captured request wall (s) for the busy fraction")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    rows = read_kern_sum(args.kern_csv)
    if not rows:
        sys.exit(f"no kernel rows in {args.kern_csv}")
    total_ns = sum(r["total_ns"] for r in rows)
    total_inst = sum(r["inst"] for r in rows)
    rows.sort(key=lambda r: -r["total_ns"])

    print(f"# Captured GPU kernel time: {total_ns/1e6:.1f} ms over {total_inst} launches ({len(rows)} distinct kernels)")
    if args.wall_s > 0:
        print(f"# Captured request wall: {args.wall_s:.2f} s -> GPU busy fraction {total_ns/1e9/args.wall_s*100:.1f}% "
              f"(the rest is host: launch gaps, D2H syncs, PLE hash/gather, scheduler)")
    print()
    print("## Per family (% of captured GPU kernel time)")
    print()
    print("| family | % GPU | ms | launches | kernels |")
    print("|---|---:|---:|---:|---|")
    fams = {}
    for r in rows:
        f = fams.setdefault(family_of(r["name"]), {"ns": 0.0, "inst": 0, "names": []})
        f["ns"] += r["total_ns"]
        f["inst"] += r["inst"]
        f["names"].append(short(r["name"], 40))
    for fam, f in sorted(fams.items(), key=lambda kv: -kv[1]["ns"]):
        names = ", ".join(f["names"][:4]) + (f" (+{len(f['names'])-4})" if len(f["names"]) > 4 else "")
        print(f"| {fam} | {f['ns']/total_ns*100:5.1f} | {f['ns']/1e6:8.1f} | {f['inst']:6d} | {names} |")
    print()
    print(f"## Top {args.top} kernels")
    print()
    print("| % GPU | ms | inst | avg us | med us | kernel | family |")
    print("|---:|---:|---:|---:|---:|---|---|")
    for r in rows[: args.top]:
        print(f"| {r['total_ns']/total_ns*100:5.1f} | {r['total_ns']/1e6:8.1f} | {r['inst']:6d} | {r['avg_ns']/1e3:8.1f} | "
              f"{r['med_ns']/1e3:8.1f} | `{short(r['name'])}` | {family_of(r['name'])} |")

    if args.api:
        api = read_api_sum(args.api)
        api.sort(key=lambda t: -t[1])
        print()
        print("## Host CUDA API (top 12 by time) — syncs and launch counts")
        print()
        print("| ms | calls | api |")
        print("|---:|---:|---|")
        for name, ns, calls in api[:12]:
            print(f"| {ns/1e6:8.1f} | {calls:7d} | `{name}` |")
        syncs = sum(c for n, _, c in api if re.search(r"Synchronize|MemcpyDtoH|EventQuery", n))
        launches = sum(c for n, _, c in api if "Launch" in n)
        print(f"\nlaunch-class calls: {launches}; sync/copy-class calls: {syncs}")


if __name__ == "__main__":
    main()
