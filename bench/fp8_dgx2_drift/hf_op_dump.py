#!/usr/bin/env python3
"""HF[BF16-unquant] CPU forward on the 10382-token prompt with PER-OPERATION
hooks for the master drift table.

For every transformer layer (40 layers, 10 full-attn at L3,7,11,...,39 and 30
linear-attn / SSM at the rest), register forward hooks on every named
submodule that maps to an Atlas op-dump. Captures the LAST token of each
op's output as f32 to `<dir>/hf_op_L{i}_{op}.bin`.

Atlas op names → HF module mapping (per layer):
  - input_norm_in    : input to `input_layernorm` (= residual stream, captured
                       via the layer's __call__ pre-hook on hidden_states arg)
  - input_norm_out   : output of `input_layernorm`
  - q_proj_full      : output of `self_attn.q_proj`  (full-attn only)
  - k_proj           : output of `self_attn.k_proj`  (full-attn only)
  - v_proj           : output of `self_attn.v_proj`  (full-attn only)
  - o_proj           : output of `self_attn.o_proj`  (full-attn only)
  - q_after_norm     : output of `self_attn.q_norm`  (full-attn only)
  - k_after_norm     : output of `self_attn.k_norm`  (full-attn only)
  - ssm_in_proj_qkvz : output of `linear_attn.in_proj_qkvz`  (ssm only)
  - ssm_in_proj_ba   : output of `linear_attn.in_proj_ba`  (ssm only)
  - ssm_conv1d       : output of `linear_attn.conv1d`  (ssm only)
  - ssm_norm         : output of `linear_attn.norm`  (gated RMSNorm) (ssm only)
  - ssm_out_proj     : output of `linear_attn.out_proj`  (ssm only)
  - post_attn_norm_out : output of `post_attention_layernorm`
  - router_gate      : output of `mlp.gate`  (pre-softmax router logits)
  - shared_expert    : output of `mlp.shared_expert`  (pre-gate)
  - moe_out          : output of `mlp` (final MoE block output)
  - layer_out        : output of full layer (already covered by atlas_L*.bin)

For tensors with leading [batch, seq, ...] shape, captures `t[0, -1, ...]`
flattened.

For tensors with shape [batch*seq, ...] (some HF kernels reshape), captures
`t[-1, ...]`.
"""
from __future__ import annotations

import json
import pathlib
import time
import gc

import numpy as np
import torch
from transformers import AutoModelForCausalLM

SNAP = "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
TOKENS_PATH = pathlib.Path(
    "/workspace/atlas-mtp/bench/fp8_dgx2_drift/atlas_tokens_dgx2.json"
)
OUT_DIR = pathlib.Path("/workspace/atlas-dumps/op_drift")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_f32(path: pathlib.Path, arr: np.ndarray):
    arr_f32 = np.ascontiguousarray(arr.reshape(-1), dtype="<f4")
    path.write_bytes(arr_f32.tobytes())


def extract_last(t: torch.Tensor) -> np.ndarray:
    """Pull the LAST sequence position out of a tensor, regardless of layout."""
    if t is None:
        return None
    t = t.detach()
    if t.is_floating_point() is False:
        t = t.float()
    else:
        t = t.float()
    a = t.cpu().numpy()
    # Common layouts:
    #   [B, S, ...]   → [-1]
    #   [B*S, ...]    → [-1]
    #   [S, ...]      → [-1]  (batch_first=False; rare here)
    if a.ndim >= 2:
        # If shape starts with (1, ...) treat dim 0 as batch and squeeze.
        if a.shape[0] == 1:
            a = a[0]
        # Now first dim should be the sequence axis.
        a = a[-1]
    return a


def make_module_hook(out_dir: pathlib.Path, layer_idx: int, op_name: str):
    def hook(_module, _inp, out):
        # Output may be Tensor or tuple.
        t = out[0] if isinstance(out, tuple) else out
        if not isinstance(t, torch.Tensor):
            return out
        try:
            last = extract_last(t)
            path = out_dir / f"hf_op_L{layer_idx}_{op_name}.bin"
            write_f32(path, last)
        except Exception as e:
            print(f"  WARN: hook L{layer_idx}/{op_name} failed: {e}", flush=True)
        return out

    return hook


def make_layer_input_prehook(out_dir: pathlib.Path, layer_idx: int):
    """Forward-pre-hook on the decoder layer to capture the input hidden state."""

    def prehook(_module, args, kwargs):
        # The layer call signature is forward(hidden_states, ...)
        t = None
        if args:
            t = args[0]
        elif "hidden_states" in kwargs:
            t = kwargs["hidden_states"]
        if isinstance(t, torch.Tensor):
            try:
                last = extract_last(t)
                path = out_dir / f"hf_op_L{layer_idx}_input_norm_in.bin"
                write_f32(path, last)
            except Exception as e:
                print(f"  WARN: prehook L{layer_idx}/input_norm_in failed: {e}", flush=True)
        return args, kwargs

    return prehook


def main() -> None:
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading tokens from {TOKENS_PATH}", flush=True)
    tok_data = json.loads(TOKENS_PATH.read_text())
    all_tokens = tok_data["all_tokens"]
    generated = tok_data.get("generated_tokens", [])
    prompt_len = len(all_tokens) - len(generated)
    prompt_tokens = all_tokens[:prompt_len]
    print(f"  prompt_len={prompt_len}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading BF16 model from {SNAP}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    # Locate the decoder layers. For Qwen3_5MoeForConditionalGeneration the
    # layer list is at model.model.layers (text model is direct).
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    elif hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        raise RuntimeError("Cannot locate decoder layers in model")
    n_layers = len(layers)
    print(f"  found {n_layers} decoder layers", flush=True)

    # Layer type list — use model config to determine which layers are full-attn.
    cfg = model.config
    if hasattr(cfg, "text_config"):
        layer_types = list(cfg.text_config.layer_types)
    else:
        layer_types = list(cfg.layer_types)
    full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    ssm_idx = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    print(
        f"  full-attn layers ({len(full_idx)}): {full_idx[:5]}...{full_idx[-3:]}",
        flush=True,
    )
    print(
        f"  ssm  layers ({len(ssm_idx)}): {ssm_idx[:5]}...{ssm_idx[-3:]}",
        flush=True,
    )

    hooks = []
    print(f"[{time.strftime('%H:%M:%S')}] registering hooks", flush=True)
    for i in range(n_layers):
        layer = layers[i]
        # Layer-input prehook for input_norm_in
        hooks.append(
            layer.register_forward_pre_hook(
                make_layer_input_prehook(OUT_DIR, i), with_kwargs=True
            )
        )
        # input_layernorm output
        hooks.append(
            layer.input_layernorm.register_forward_hook(
                make_module_hook(OUT_DIR, i, "input_norm_out")
            )
        )
        # post_attention_layernorm output
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                make_module_hook(OUT_DIR, i, "post_attn_norm_out")
            )
        )
        # Attention / SSM submodules
        if layer_types[i] == "full_attention":
            sa = layer.self_attn
            hooks.append(sa.q_proj.register_forward_hook(make_module_hook(OUT_DIR, i, "q_proj_full")))
            hooks.append(sa.k_proj.register_forward_hook(make_module_hook(OUT_DIR, i, "k_proj")))
            hooks.append(sa.v_proj.register_forward_hook(make_module_hook(OUT_DIR, i, "v_proj")))
            hooks.append(sa.o_proj.register_forward_hook(make_module_hook(OUT_DIR, i, "o_proj")))
            hooks.append(sa.q_norm.register_forward_hook(make_module_hook(OUT_DIR, i, "q_after_norm")))
            hooks.append(sa.k_norm.register_forward_hook(make_module_hook(OUT_DIR, i, "k_after_norm")))
        else:  # linear_attention / SSM
            la = layer.linear_attn
            # Confirm exact attribute names (defensive)
            if hasattr(la, "in_proj_qkvz"):
                hooks.append(la.in_proj_qkvz.register_forward_hook(
                    make_module_hook(OUT_DIR, i, "ssm_in_proj_qkvz")))
            if hasattr(la, "in_proj_ba"):
                hooks.append(la.in_proj_ba.register_forward_hook(
                    make_module_hook(OUT_DIR, i, "ssm_in_proj_ba")))
            if hasattr(la, "conv1d"):
                hooks.append(la.conv1d.register_forward_hook(
                    make_module_hook(OUT_DIR, i, "ssm_conv1d")))
            if hasattr(la, "norm"):
                hooks.append(la.norm.register_forward_hook(
                    make_module_hook(OUT_DIR, i, "ssm_norm")))
            if hasattr(la, "out_proj"):
                hooks.append(la.out_proj.register_forward_hook(
                    make_module_hook(OUT_DIR, i, "ssm_out_proj")))
        # MoE block
        mlp = layer.mlp
        hooks.append(mlp.gate.register_forward_hook(make_module_hook(OUT_DIR, i, "router_gate")))
        if hasattr(mlp, "shared_expert"):
            hooks.append(mlp.shared_expert.register_forward_hook(
                make_module_hook(OUT_DIR, i, "shared_expert")))
        hooks.append(mlp.register_forward_hook(make_module_hook(OUT_DIR, i, "moe_out")))
        # Whole-layer output (this matches atlas_L*.bin)
        hooks.append(layer.register_forward_hook(make_module_hook(OUT_DIR, i, "layer_out")))

    print(f"  registered {len(hooks)} hooks across {n_layers} layers", flush=True)

    print(
        f"[{time.strftime('%H:%M:%S')}] forward pass — {prompt_len} tokens", flush=True
    )
    t0 = time.time()
    with torch.no_grad():
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        _ = model(input_ids, use_cache=False, output_hidden_states=False)
    dt = time.time() - t0
    print(
        f"  forward done in {dt:.1f}s ({prompt_len / dt:.1f} tok/s)", flush=True
    )

    for h in hooks:
        h.remove()

    files = sorted(OUT_DIR.glob("hf_op_L*.bin"))
    print(f"[{time.strftime('%H:%M:%S')}] wrote {len(files)} dump files to {OUT_DIR}", flush=True)
    # Sanity: print summary of one full-attn and one ssm layer
    print("  full-attn L3 ops:", sorted(p.name for p in OUT_DIR.glob("hf_op_L3_*.bin")), flush=True)
    print("  ssm L0   ops:", sorted(p.name for p in OUT_DIR.glob("hf_op_L0_*.bin")), flush=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] DONE (total {time.time() - t_start:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
