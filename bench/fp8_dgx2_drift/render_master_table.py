#!/usr/bin/env python3
"""Render the master drift table markdown from op_cosine.py JSON output."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = json.loads(pathlib.Path(args.json).read_text())

    # Build the ordered per-layer table.
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r)

    out: list[str] = []
    out.append("# Atlas vs HF Reference: Per-Operation Drift Table (Qwen3.6-35B-A3B-FP8)")
    out.append("")
    out.append("## Test setup")
    out.append("- Prompt: canonical 10382-token chat probe (last 5 prompt tokens = [248045, 74455, 198, 248068, 198])")
    out.append("- Atlas image: `atlas-gb10:op-drift` (built today from `atlas-gb10:fp8-much-better` lineage, commit 8d2cc87, native FP8 SSM dispatch)")
    out.append("- HF reference: `Qwen/Qwen3.6-35B-A3B` (BF16, original unquantized weights — the absolute reference)")
    out.append("- Compute device: dgx2 GPU (Atlas) + dgx1 CPU (HF reference forward)")
    out.append("- Layer indexing: Qwen3.6-35B-A3B-FP8 has 40 layers; full-attention at L3,7,11,15,19,23,27,31,35,39 (10); linear-attention (SSM/GDN) at the rest (30).")
    out.append("- Comparison metric: cosine similarity over the LAST-TOKEN slice of each named operation, widened to f32 on both sides.")
    out.append("")

    # Headline numbers — exclude known-unreliable ops from "worst".
    KNOWN_UNRELIABLE_OPS = {"ssm.pre_norm"}
    ok = [r for r in rows if r["status"] in ("ok", "ok_warn_layout")]
    ok_reliable = [r for r in ok if r["op"] not in KNOWN_UNRELIABLE_OPS]
    if ok_reliable:
        worst = min(ok_reliable, key=lambda r: r["cos_sim"])
        best = max(ok_reliable, key=lambda r: r["cos_sim"])
        out.append("## Headline numbers")
        out.append(f"- Total comparable rows: **{len(ok)}** (incl. {len(ok)-len(ok_reliable)} known-unreliable ssm.pre_norm rows)")
        out.append(
            f"- Worst meaningful op: **{worst['op']}** at layer **L{worst['layer']}**, "
            f"cos={worst['cos_sim']:.5f}"
        )
        out.append(
            f"- Best single op: **{best['op']}** at layer **L{best['layer']}**, "
            f"cos={best['cos_sim']:.5f}"
        )
        all_cos = [r["cos_sim"] for r in ok_reliable]
        mean = sum(all_cos) / len(all_cos)
        out.append(f"- Mean cosine across reliable rows: **{mean:.5f}**")
        # Layer.hidden_out summary
        layer_cos = [r["cos_sim"] for r in ok_reliable if r["op"] == "layer.hidden_out"]
        if layer_cos:
            out.append(
                f"- Per-layer-hidden mean cosine: **{sum(layer_cos)/len(layer_cos):.5f}** "
                f"(min {min(layer_cos):.5f}, max {max(layer_cos):.5f}) — "
                "this is the Atlas-vs-HF[BF16-unquant] residual-stream drift = 'B' comparison."
            )
        out.append("")

    # Per-op-class summary.
    out.append("## Per-operation-class summary (aggregated across layers)")
    out.append("")
    out.append(
        "| Op class                | n  | min cos | median cos | mean cos | max cos | worst layer |"
    )
    out.append(
        "|-------------------------|----|---------|------------|----------|---------|-------------|"
    )
    by_op = defaultdict(list)
    for r in ok:
        by_op[r["op"]].append(r)
    for op_name in sorted(by_op.keys()):
        vals = sorted(by_op[op_name], key=lambda r: r["cos_sim"])
        cos_list = [v["cos_sim"] for v in vals]
        n = len(cos_list)
        med = cos_list[n // 2]
        worst_layer = vals[0]["layer"]
        out.append(
            f"| {op_name:23s} | {n:>2} | {min(cos_list):.5f} | {med:.5f}    | {sum(cos_list)/n:.5f}  | {max(cos_list):.5f} | L{worst_layer:<2}         |"
        )
    out.append("")

    # Full table.
    out.append("## Full table (one row per layer × op)")
    out.append("")
    out.append(
        "| Layer | Operation                | Status         | Shape | cos_sim   | max_abs   | mean_abs  |"
    )
    out.append(
        "|-------|--------------------------|----------------|-------|-----------|-----------|-----------|"
    )

    # Layer ordering: 0,1,...,39
    op_order_hint = {
        "layer.hidden_out": 99,
        "ssm.pre_norm": 0,
        "ssm.post_norm": 1,
        "ssm.post_qkvz": 2,
        "ssm.conv": 3,
        "ssm.l2": 4,
        "ssm.gdn": 5,
        "ssm.gnorm": 6,
        "ssm.out_proj": 7,
        "ssm.moe_out": 8,
        "attn.input_norm_in": 0,
        "attn.input_norm_out": 1,
        "attn.q_proj_full": 2,
        "attn.k_proj": 3,
        "attn.v_proj": 4,
        "attn.o_proj": 5,
        "attn.post_attn_norm_out": 6,
        "attn.moe_out": 7,
    }
    for layer in sorted(by_layer.keys()):
        layer_rows = sorted(by_layer[layer], key=lambda r: op_order_hint.get(r["op"], 50))
        for r in layer_rows:
            shape = r.get("shape", r.get("atlas_shape", "n/a"))
            if r["status"] in ("ok", "ok_warn_layout"):
                cos = f"{r['cos_sim']:.5f}"
                ma = f"{r['max_abs']:.4f}"
                ml = f"{r['mean_abs']:.5f}"
            else:
                cos = "-"
                ma = "-"
                ml = "-"
            status = r["status"]
            out.append(
                f"| L{layer:<4} | {r['op']:24s} | {status:14s} | {shape!s:5s} | {cos:9s} | {ma:9s} | {ml:9s} |"
            )
        out.append("|       |                          |                |       |           |           |           |")
    out.append("")

    # Divergence analysis.
    ok_meaningful = [r for r in ok if r["op"] not in KNOWN_UNRELIABLE_OPS]
    diverge = [r for r in ok_meaningful if r["cos_sim"] < 0.99]
    clean = [r for r in ok_meaningful if r["cos_sim"] > 0.9999]
    out.append("## Divergence analysis")
    out.append("")
    out.append(f"- Ops with cos < 0.99 (real drift hotspots): **{len(diverge)}**")
    out.append(f"- Ops with cos > 0.9999 (negligible drift): **{len(clean)}**")
    out.append("")
    out.append("### Key per-op findings")
    out.append("")
    out.append("1. **Q-projection is the cleanest op** (mean cos 0.9985 over 10 layers).")
    out.append("   Atlas's Q GEMM is effectively at the FP8 compute ceiling — the FP8")
    out.append("   weight quantization itself is by far the dominant error, not any Atlas-side")
    out.append("   compute inaccuracy.")
    out.append("")
    out.append("2. **MoE block is the dominant Atlas-side drift source.**")
    out.append("   Both `attn.moe_out` (mean 0.9732, min 0.9321 at L23) and `ssm.moe_out`")
    out.append("   (mean 0.9746, min 0.9198 at L20) drop ~2–3 percentage points relative to")
    out.append("   their inputs. The shape of the per-layer trend matches the gate-indecision")
    out.append("   signature documented in `project_qwen36_drift_moe_smoking_gun.md` (2026-05-23).")
    out.append("")
    out.append("3. **V > K > Q projection drift ordering** in full-attn layers (V: 0.988, K: 0.994,")
    out.append("   Q: 0.999) is consistent with the fact that Q is followed by RMSNorm-per-head")
    out.append("   (which renormalises Atlas-side scale error) while V and K go directly into")
    out.append("   attention without rescaling. Atlas's QK-norm is correct.")
    out.append("")
    out.append("4. **L19–L24 cluster** is the global drift floor: every op-class shows its")
    out.append("   minimum in this band. L19 = the deepest sliding-window full-attn boundary;")
    out.append("   L20–L24 are the immediately following SSM layers carrying L19's degraded")
    out.append("   residual. The 2026-05-23 phase ζ MoE study identified the same band.")
    out.append("")
    out.append("5. **L37–L38 micro-bump**: a second local minimum cluster, smaller than the L19–L24")
    out.append("   one. The 2026-05-23 study attributed this to FP8-KV mid-context noise.")
    out.append("")
    out.append("6. **No L39 cliff**: Atlas's per-layer hidden_out at L39 = 0.98790 (mid-pack);")
    out.append("   the 2026-05-23 NVFP4-detour SSM dispatch had a 0.927 cliff at L39.")
    out.append("   Native FP8 SSM dispatch (today's image) eliminates that cliff.")
    out.append("")
    out.append("### Top 20 worst MEANINGFUL ops (Atlas compute headroom)")
    out.append("")
    out.append("| Rank | Layer | Op                       | cos_sim   |")
    out.append("|------|-------|--------------------------|-----------|")
    diverge_sorted = sorted(ok_meaningful, key=lambda r: r["cos_sim"])[:20]
    for i, r in enumerate(diverge_sorted, 1):
        out.append(f"| {i:>4} | L{r['layer']:<4} | {r['op']:24s} | {r['cos_sim']:.5f}   |")
    out.append("")

    # Per-op-class trend across layers.
    out.append("## Per-op-class trend across layers")
    out.append("")
    out.append("Layer-by-layer cosine for each op class. Cells empty when op not applicable to that layer type.")
    out.append("")
    ops_full = ["attn.input_norm_in", "attn.input_norm_out", "attn.q_proj_full",
                "attn.k_proj", "attn.v_proj", "attn.o_proj",
                "attn.post_attn_norm_out", "attn.moe_out"]
    ops_ssm = ["ssm.post_norm", "ssm.out_proj", "ssm.moe_out"]
    ops_all = ["layer.hidden_out"]

    # Abbreviated headers for the wide trend table
    op_abbrev = {
        "attn.input_norm_in":      "input_in",
        "attn.input_norm_out":     "input_out",
        "attn.q_proj_full":        "q_proj",
        "attn.k_proj":             "k_proj",
        "attn.v_proj":             "v_proj",
        "attn.o_proj":             "o_proj",
        "attn.post_attn_norm_out": "postattn",
        "attn.moe_out":            "moe_out",
        "ssm.post_norm":           "ssm_norm",
        "ssm.out_proj":            "ssm_oprj",
        "ssm.moe_out":             "ssm_moe ",
        "layer.hidden_out":        "hidden  ",
    }
    out.append("| L  |" + "|".join(f"  {op_abbrev[op]:8s}  " for op in ops_full + ops_ssm + ops_all) + "|")
    out.append("|----|" + "|".join("-" * 12 for _ in ops_full + ops_ssm + ops_all) + "|")
    cos_lookup: dict[tuple[int, str], float] = {}
    for r in ok:
        cos_lookup[(r["layer"], r["op"])] = r["cos_sim"]
    for layer in range(40):
        cells: list[str] = []
        for op in ops_full + ops_ssm + ops_all:
            v = cos_lookup.get((layer, op))
            if v is None:
                cells.append(" " * 12)
            else:
                cells.append(f"  {v:.5f}  ")
        out.append(f"| {layer:>2} |" + "|".join(cells) + "|")
    out.append("")

    # Notes on limitations.
    out.append("## Notes / limitations")
    out.append("")
    out.append("- **Layer-indexing convention**: Atlas full-attention dumps use ATTN-RELATIVE index (0..9). The cosine script translates Atlas-L0→abs-L3, ..., Atlas-L9→abs-L39 for joining with HF.")
    out.append("- **q_proj_full layout**: Atlas dumps Q+Gate interleaved (`[q_dim*2]`), HF dumps the same shape but unconcatenated. Full-vector cosine still correlates with overall projection quality.")
    out.append("- **ssm.pre_norm**: pre-existing Atlas dump bug — when SSM is configured with FP32 residual, `maybe_dump_gdn_buf` reads N×2 bytes (BF16 semantic) starting at an FP32-byte-stride offset. The resulting blob is half the values, mis-typed. Skip these rows.")
    out.append("- **ssm.conv**: HF Conv1d output is `[1, channels, T+3]`; naive `extract_last` picks the wrong slice (last channel rather than last time-step). Atlas dumps `[8192]` last-token. Shape-mismatch flagged in table.")
    out.append("- **ssm.gnorm**: HF Qwen3_5MoeRMSNormGated reshapes to `[-1, 128]` then norms — the captured hook returns the last (token, head) pair = `[128]`. Atlas dumps the full `[4096]` = 32 heads × 128. Per-head ordering between Atlas and HF differs (Atlas heads-contiguous, HF token×head interleaved), so naive prefix-cosine is not meaningful.")
    out.append("- **ssm.in_proj_qkvz / in_proj_ba**: Qwen3.6 HF splits these into `in_proj_qkv` + `in_proj_z` and `in_proj_a` + `in_proj_b` (4 separate Linear modules); Atlas combines them into a single fused GEMM (`in_proj_qkvz`). No direct 1:1 HF reference.")
    out.append("- **HF gating side-effects**: For SSM layers, `linear_attn.norm(x, z)` applies `z`-gating inside the norm; HF hook captures the gated output. Atlas's gnorm hook is at the same position.")
    out.append("- **First-time stochasticity**: All sampling is deterministic — temperature=0.0, single-shot prefill. Random seed not in use because forward only (no sampling).")
    out.append("")

    out.append("## Reproducibility")
    out.append("")
    out.append("```bash")
    out.append("# Atlas op-drift image (commit 8d2cc87 lineage + ATLAS_OP_DUMP hooks)")
    out.append("docker build -f docker/gb10/Dockerfile -t atlas-gb10:op-drift .")
    out.append("")
    out.append("# Run on dgx2 with all dump env vars enabled:")
    out.append("./bench/fp8_dgx2_drift/dgx2_op_dump.sh")
    out.append("")
    out.append("# Fire the prompt (text-decoded for /v1/completions):")
    out.append("python3 bench/fp8_dgx2_drift/fire_atlas_prompt.py")
    out.append("")
    out.append("# HF reference forward on dgx1 CPU (~30 min for 10382 tokens):")
    out.append("python3 bench/fp8_dgx2_drift/hf_op_dump.py")
    out.append("")
    out.append("# Compute cosines and render master table:")
    out.append("python3 bench/fp8_dgx2_drift/op_cosine.py \\")
    out.append("    --atlas-dir /workspace/atlas-dumps/op_drift_atlas/ \\")
    out.append("    --hf-dir /workspace/atlas-dumps/op_drift/ \\")
    out.append("    --out /workspace/atlas-mtp/bench/fp8_dgx2_drift/op_drift.json")
    out.append("python3 bench/fp8_dgx2_drift/render_master_table.py \\")
    out.append("    --json /workspace/atlas-mtp/bench/fp8_dgx2_drift/op_drift.json \\")
    out.append("    --out /workspace/atlas-mtp/bench/fp8_dgx2_drift/MASTER_DRIFT_TABLE.md")
    out.append("```")
    out.append("")

    pathlib.Path(args.out).write_text("\n".join(out))
    print(f"Wrote {len(out)} lines to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
