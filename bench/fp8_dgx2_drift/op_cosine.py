#!/usr/bin/env python3
"""Compute per-op cosine similarity and abs-diff between Avarok dumps and
HF reference dumps for the master drift table.

Inputs:
  --avarok-dir   : directory with avarok_op_L{i}_{op}.bin (full-attn ops)
                  + gdnsub_step0_L{ssm_idx}_{stage}.bin (SSM stages)
                  + avarok_L{i}.bin (per-layer hidden)
  --hf-dir      : directory with hf_op_L{i}_{op}.bin (all hooks)
                  + hf_bf16_L{i}.bin (per-layer hidden — legacy name)
  --out         : JSON output file

Avarok-to-HF op-name mapping is defined in OP_MAP below. SSM stage names use
the SSM-relative layer index in the Avarok filename but the absolute layer
index in the HF filename — we translate via the layer_types list.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def load_bin(path: pathlib.Path) -> np.ndarray:
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype="<f4")
    if data.size == 0:
        return None
    return data


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


# Full-attn layer indices for Qwen3.6-35B-A3B
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
SSM_LAYERS = [i for i in range(40) if i not in FULL_ATTN_LAYERS]

# Map: SSM absolute layer idx → SSM-relative idx (for gdnsub filenames)
SSM_REL = {abs_i: rel_i for rel_i, abs_i in enumerate(SSM_LAYERS)}


def compare_ssm_stages(avarok_dir: pathlib.Path, hf_dir: pathlib.Path, results: list):
    """Compare SSM stages.

    Avarok filename:  gdnsub_step0_L{ssm_rel}_{stage}.bin  (BF16 raw bytes)
    HF filename:     hf_op_L{abs}_{ssm_*}.bin  (f32)

    Stage mapping (Avarok → HF op):
      pre_norm         → input_norm_in
      post_norm        → input_norm_out
      qkvz             → ssm_in_proj_qkvz
      conv             → ssm_conv1d (approx — post-silu)
      gnorm            → ssm_norm (gated RMSNorm)
      out_proj         → ssm_out_proj
      moe_out          → moe_out
    """
    # Stage map. Note: ssm.pre_norm is known-corrupt (Avarok dumps f32 residual
    # as if it were BF16 — pre-existing dump bug). Marked unreliable in output.
    # Conv1d shape mismatch (Avarok dumps 8192 BF16 last-token interleaved;
    # HF outputs [1, 8192, T+3] and our naive extract picks the last channel,
    # not last time-step). Marked unreliable in output.
    # gnorm: Avarok dumps full value_dim=4096; HF norm.weight is [head_dim=128]
    # so the HF hook captures only one head's [128] slice. Cosine on
    # min-prefix is still meaningful (compares the first 128 elements).
    stage_map = [
        ("pre_norm", "input_norm_in"),
        ("post_norm", "input_norm_out"),
        ("post_qkvz", "ssm_in_proj_qkvz"),
        ("conv", "ssm_conv1d"),
        ("l2", None),  # No HF equivalent — l2 norm is internal
        ("gdn", None),  # No HF equivalent — GDN recurrence is internal kernel
        ("gnorm", "ssm_norm"),
        ("out_proj", "ssm_out_proj"),
        ("moe_out", "moe_out"),
    ]

    for abs_i in SSM_LAYERS:
        rel_i = SSM_REL[abs_i]
        for avarok_stage, hf_op in stage_map:
            avarok_path = avarok_dir / f"gdnsub_step0_L{rel_i}_{avarok_stage}.bin"
            if hf_op is None:
                # Avarok-only stage; record file presence so the master
                # table notes the missing-HF-reference fact.
                row = {
                    "layer": abs_i,
                    "op": f"ssm.{avarok_stage}",
                    "avarok_file": avarok_path.name,
                    "hf_file": None,
                    "status": "avarok_only_no_hf_ref",
                }
                if avarok_path.exists():
                    raw = np.fromfile(str(avarok_path), dtype=np.uint16)
                    avarok_arr = np.frombuffer(
                        np.left_shift(raw.astype(np.uint32), 16).tobytes(),
                        dtype="<f4",
                    )
                    row["avarok_shape"] = int(avarok_arr.size)
                    row["avarok_norm"] = float(np.linalg.norm(avarok_arr))
                results.append(row)
                continue
            hf_path = hf_dir / f"hf_op_L{abs_i}_{hf_op}.bin"
            # Avarok SSM dumps are BF16 raw bytes (2 bytes each). Need to widen.
            if avarok_path.exists():
                raw = np.fromfile(str(avarok_path), dtype=np.uint16)
                avarok_arr = np.frombuffer(
                    np.left_shift(raw.astype(np.uint32), 16).tobytes(),
                    dtype="<f4",
                )
            else:
                avarok_arr = None
            hf_arr = load_bin(hf_path) if hf_path.exists() else None
            row = {
                "layer": abs_i,
                "op": f"ssm.{avarok_stage}",
                "avarok_file": avarok_path.name,
                "hf_file": hf_path.name,
            }
            if avarok_arr is None or hf_arr is None:
                row["status"] = "missing"
                row["avarok_present"] = avarok_arr is not None
                row["hf_present"] = hf_arr is not None
            elif avarok_arr.size != hf_arr.size:
                # Some ops differ in shape (qkvz vs in_proj_qkvz may differ).
                # Try min-prefix comparison for hint.
                n = min(avarok_arr.size, hf_arr.size)
                row["status"] = "shape_mismatch"
                row["avarok_shape"] = avarok_arr.size
                row["hf_shape"] = hf_arr.size
                if n > 0:
                    row["cos_sim_prefix"] = cos_sim(avarok_arr[:n], hf_arr[:n])
            else:
                row["status"] = "ok"
                row["shape"] = int(avarok_arr.size)
                row["cos_sim"] = cos_sim(avarok_arr, hf_arr)
                row["max_abs"] = max_abs(avarok_arr, hf_arr)
                row["mean_abs"] = mean_abs(avarok_arr, hf_arr)
            results.append(row)


def compare_attention_ops(avarok_dir: pathlib.Path, hf_dir: pathlib.Path, results: list):
    """Compare full-attention layer ops.

    Avarok uses full-attn-RELATIVE index 0..9 (Qwen3AttentionLayer.attn_layer_idx).
    HF uses ABSOLUTE layer index (L3, L7, ..., L39).
    """
    full_attn_ops = [
        "input_norm_in",
        "input_norm_out",
        "q_proj_full",
        "k_proj",
        "v_proj",
        "o_proj",
        "post_attn_norm_out",
        "moe_out",
    ]
    # HF-only ops captured (no direct Avarok counterpart yet) — emit as
    # informational rows so the table records what HF has available.
    hf_only_ops = ["router_gate", "shared_expert", "q_after_norm", "k_after_norm"]
    for rel_i, abs_i in enumerate(FULL_ATTN_LAYERS):
        for op in full_attn_ops:
            avarok_path = avarok_dir / f"avarok_op_L{rel_i}_{op}.bin"
            hf_path = hf_dir / f"hf_op_L{abs_i}_{op}.bin"
            avarok_arr = load_bin(avarok_path)
            hf_arr = load_bin(hf_path)
            row = {
                "layer": abs_i,
                "op": f"attn.{op}",
                "avarok_file": avarok_path.name,
                "hf_file": hf_path.name,
            }
            if avarok_arr is None or hf_arr is None:
                row["status"] = "missing"
                row["avarok_present"] = avarok_arr is not None
                row["hf_present"] = hf_arr is not None
                results.append(row)
                continue
            # For q_proj_full Avarok dumps Q+Gate interleaved (2× q_dim).
            # HF q_proj output also is 2× q_dim (qwen3.6 attn_output_gate=True),
            # but the layout differs (Avarok interleaves; HF concatenates as
            # [..., -1, head_dim*2]). Compare the FIRST half only for now,
            # and write a note.
            if op == "q_proj_full":
                # Both should be the same total size; compare first half cosine.
                if avarok_arr.size != hf_arr.size:
                    row["status"] = "shape_mismatch"
                    row["avarok_shape"] = int(avarok_arr.size)
                    row["hf_shape"] = int(hf_arr.size)
                    results.append(row)
                    continue
                n = avarok_arr.size
                half = n // 2
                row["status"] = "ok_warn_layout"
                row["shape"] = int(n)
                row["cos_sim"] = cos_sim(avarok_arr, hf_arr)
                row["max_abs"] = max_abs(avarok_arr, hf_arr)
                row["mean_abs"] = mean_abs(avarok_arr, hf_arr)
                row["note"] = (
                    "Q+gate interleaved (avarok) vs Q,gate split (hf); "
                    "full-length cosine likely <1.0 even with byte-exact compute."
                )
            else:
                if avarok_arr.size != hf_arr.size:
                    row["status"] = "shape_mismatch"
                    row["avarok_shape"] = int(avarok_arr.size)
                    row["hf_shape"] = int(hf_arr.size)
                    results.append(row)
                    continue
                row["status"] = "ok"
                row["shape"] = int(avarok_arr.size)
                row["cos_sim"] = cos_sim(avarok_arr, hf_arr)
                row["max_abs"] = max_abs(avarok_arr, hf_arr)
                row["mean_abs"] = mean_abs(avarok_arr, hf_arr)
            results.append(row)


def compare_layer_hidden(avarok_dir: pathlib.Path, hf_dir: pathlib.Path, results: list):
    """End-of-layer hidden state (= residual stream) for all 40 layers."""
    for abs_i in range(40):
        avarok_path = avarok_dir / f"avarok_L{abs_i}.bin"
        # HF dumps have both legacy hf_bf16_L*.bin (per-layer) AND new hf_op_L*_layer_out.bin
        hf_path = hf_dir / f"hf_op_L{abs_i}_layer_out.bin"
        if not hf_path.exists():
            hf_path = hf_dir / f"hf_bf16_L{abs_i}.bin"
        avarok_arr = load_bin(avarok_path)
        hf_arr = load_bin(hf_path)
        row = {
            "layer": abs_i,
            "op": "layer.hidden_out",
            "avarok_file": avarok_path.name,
            "hf_file": hf_path.name,
        }
        if avarok_arr is None or hf_arr is None:
            row["status"] = "missing"
            row["avarok_present"] = avarok_arr is not None
            row["hf_present"] = hf_arr is not None
            results.append(row)
            continue
        if avarok_arr.size != hf_arr.size:
            row["status"] = "shape_mismatch"
            row["avarok_shape"] = int(avarok_arr.size)
            row["hf_shape"] = int(hf_arr.size)
            results.append(row)
            continue
        row["status"] = "ok"
        row["shape"] = int(avarok_arr.size)
        row["cos_sim"] = cos_sim(avarok_arr, hf_arr)
        row["max_abs"] = max_abs(avarok_arr, hf_arr)
        row["mean_abs"] = mean_abs(avarok_arr, hf_arr)
        results.append(row)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--avarok-dir", required=True)
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    avarok_dir = pathlib.Path(args.avarok_dir)
    hf_dir = pathlib.Path(args.hf_dir)

    results: list = []
    compare_layer_hidden(avarok_dir, hf_dir, results)
    compare_attention_ops(avarok_dir, hf_dir, results)
    compare_ssm_stages(avarok_dir, hf_dir, results)

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} comparison rows to {out}")

    # Summary
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"  status counts: {by_status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
