#!/usr/bin/env python3
"""vLLM-FP8 per-layer residual-stream dump for Atlas vs vLLM cosine diff.

Runs the NATIVE FP8 Qwen3.6-35B-A3B-FP8 model offline in vLLM with
enforce_eager=True so torch forward hooks fire during prefill. For each of the
40 decoder layers, captures the LAST prompt-token full post-layer residual
stream (2048 floats) and writes f32 little-endian to vllm_L{i}.bin, matching
hf_forward_gpu.py's format and semantic capture point.

The served arch resolves to Qwen3_5MoeForConditionalGeneration. Its decoder
layer is Qwen3_5DecoderLayer(Qwen3NextDecoderLayer) and its text model is
Qwen3_5Model(Qwen3NextModel), so the forward convention is identical to
qwen3_next: DecoderLayer.forward(hidden_states, residual) -> (hidden_states, residual)
using the fused-add-norm pattern. The TRUE post-layer residual stream that is
the output of layer i (== HF's out[0]) equals hidden_states + residual — exactly
the value the final norm consumes
(qwen3_next.py: `hidden_states, _ = self.norm(hidden_states, residual)`).

vLLM V1 runs EngineCore in a subprocess by default, so hooks registered in the
main process never fire. We set VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-process
worker) and register hooks via llm.apply_model(func) — vLLM's official API for
running code on the model inside the worker. Capture dicts live in this same
process, so hooks fire and write during generate().
"""
from __future__ import annotations

import os

# MUST be set before importing vllm: force in-process engine core so torch
# forward hooks registered here actually fire during prefill.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

OUT_DIR = pathlib.Path("/workspace/atlas-dumps/fp8native_dgx2")
REF_TOKENS = OUT_DIR / "ref_tokens.json"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

# Shared capture state (in-process worker writes here via hook closures).
CAPTURED: dict[int, np.ndarray] = {}
FINAL_NORM: dict[str, np.ndarray] = {}


def write_f32(path: pathlib.Path, arr: np.ndarray, label: str) -> None:
    arr_f32 = np.ascontiguousarray(arr, dtype="<f4")
    path.write_bytes(arr_f32.tobytes())
    print(f"  wrote {path.name:24s} n={arr_f32.size:>7d}  {label}", flush=True)


def _last_row(t: torch.Tensor) -> torch.Tensor:
    """Last prompt-token row. vLLM prefill flattens to [num_tokens, hidden]."""
    if t.dim() == 2:
        return t[-1, :]
    if t.dim() == 3:
        return t[0, -1, :]
    return t.reshape(-1, t.shape[-1])[-1, :]


def find_text_model(model: nn.Module):
    """Locate the nn.Module whose `.layers` is the 40-layer decoder ModuleList.
    Walk through the wrapper (Qwen3_5MoeForConditionalGeneration ->
    .language_model -> .model -> .layers), robust across vLLM versions."""
    candidates = []
    # Common paths first.
    for path in (
        ("language_model", "model"),
        ("model",),
        ("language_model", "model", "model"),
    ):
        obj = model
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(getattr(obj, "layers", None), nn.ModuleList):
            candidates.append((".".join(path), obj))

    if candidates:
        name, obj = candidates[0]
        print(f"[path] text model at model.{name} (layers={len(obj.layers)}) -> {type(obj).__name__}", flush=True)
        return obj

    # Fallback: BFS over submodules for one with a >=30-len ModuleList `.layers`.
    for mod_name, mod in model.named_modules():
        layers = getattr(mod, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) >= 30:
            print(f"[path] text model (BFS) at '{mod_name}' (layers={len(layers)}) -> {type(mod).__name__}", flush=True)
            return mod
    raise RuntimeError("Could not locate text model with 40-layer decoder ModuleList")


def register_hooks(model: nn.Module):
    """Runs INSIDE the worker via apply_model. Registers per-layer + final-norm
    forward hooks. Returns the layer count for the driver to read back."""
    text_model = find_text_model(model)
    layers = text_model.layers
    n_layers = len(layers)

    def make_hook(li):
        def hook(_module, _inp, out):
            # Qwen3_5/Qwen3Next DecoderLayer returns (hidden_states, residual).
            # TRUE post-layer residual stream (== HF out[0]) = hidden_states + residual.
            if isinstance(out, tuple):
                hs = out[0]
                resid = out[1] if len(out) > 1 and out[1] is not None else None
                full = hs + resid if resid is not None else hs
            else:
                full = out
            CAPTURED[li] = _last_row(full).detach().float().cpu().numpy()
            return out

        return hook

    for i in range(n_layers):
        layers[i].register_forward_hook(make_hook(i))

    # Final norm hook (best effort).
    norm_mod = getattr(text_model, "norm", None)
    if norm_mod is not None:
        def norm_hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            FINAL_NORM["norm"] = _last_row(o).detach().float().cpu().numpy()
            return out

        norm_mod.register_forward_hook(norm_hook)
        print("[hook] registered final-norm hook", flush=True)

    # Report quantization to prove FP8 native.
    qcfg = getattr(getattr(model, "config", None), "quantization_config", None)
    print(f"[quant] config.quantization_config = {qcfg}", flush=True)
    print(f"[hook] registered {n_layers} per-layer hooks", flush=True)
    return n_layers


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_data = json.loads(REF_TOKENS.read_text())
    prompt_len = tok_data["prompt_len"]
    all_tokens = tok_data["all_tokens"]
    prompt_tokens = all_tokens[:prompt_len]
    print(f"prompt_len={prompt_len}, n_prompt_tokens={len(prompt_tokens)}, last={prompt_tokens[-1]}", flush=True)
    print(f"[env] VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')}", flush=True)

    from vllm import LLM, SamplingParams

    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM LLM({MODEL}) FP8 native, enforce_eager", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        dtype="auto",
        gpu_memory_utilization=0.85,
        max_model_len=11000,
        enforce_eager=True,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    print(f"  LLM loaded in {time.time() - t0:.1f}s", flush=True)

    # Register hooks inside the (in-process) worker. apply_model returns a list
    # (one per worker); TP=1 so one entry.
    n_layers_list = llm.apply_model(register_hooks)
    n_layers = n_layers_list[0] if isinstance(n_layers_list, list) else n_layers_list
    print(f"[{time.strftime('%H:%M:%S')}] worker hook registration done; n_layers={n_layers}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] generate (prefill {prompt_len} tokens, max_tokens=1)", flush=True)
    t0 = time.time()
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    out = llm.generate(prompts=[{"prompt_token_ids": prompt_tokens}], sampling_params=sp)
    print(f"  generate done in {time.time() - t0:.1f}s", flush=True)

    try:
        comp = out[0].outputs[0]
        print(f"[gen] sampled token_id={comp.token_ids[0]!r} text={comp.text!r}", flush=True)
    except Exception as e:
        print(f"[gen] could not read generation output: {e!r}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] captured {len(CAPTURED)} layers; writing dumps", flush=True)
    norms = []
    for i in range(n_layers):
        if i not in CAPTURED:
            print(f"  !! layer {i} NOT captured", flush=True)
            continue
        arr = CAPTURED[i]
        assert arr.size == 2048, f"layer {i} dim {arr.size} != 2048"
        finite = bool(np.isfinite(arr).all())
        l2 = float(np.linalg.norm(arr))
        norms.append((i, l2, finite))
        write_f32(OUT_DIR / f"vllm_L{i}.bin", arr, f"L{i} resid[last_tok] L2={l2:.2f} finite={finite}")

    print(f"[{time.strftime('%H:%M:%S')}] per-layer L2 norms:", flush=True)
    for i, l2, fin in norms:
        flag = "" if fin else "  <-- NON-FINITE"
        print(f"  L{i:02d}  L2={l2:10.3f}{flag}", flush=True)

    if "norm" in FINAL_NORM:
        fn = FINAL_NORM["norm"]
        write_f32(OUT_DIR / "vllm_final_norm.bin", fn, f"final_norm L2={float(np.linalg.norm(fn)):.2f}")

    # Final logits over vocab (best effort): run lm_head on captured final-norm
    # vector inside the worker (weights live there).
    if "norm" in FINAL_NORM:
        normed_np = FINAL_NORM["norm"]

        def run_lm_head(model: nn.Module):
            lm_head = getattr(model, "lm_head", None)
            if lm_head is None:
                return None
            dev = next(model.parameters()).device
            dt = next(model.parameters()).dtype
            vec = torch.tensor(normed_np, dtype=dt, device=dev).unsqueeze(0)
            with torch.no_grad():
                lg = lm_head(vec)
                lg = (lg[0] if isinstance(lg, tuple) else lg).squeeze().float().cpu().numpy()
            return lg

        try:
            res = llm.apply_model(run_lm_head)
            logits = res[0] if isinstance(res, list) else res
            if logits is not None:
                write_f32(OUT_DIR / "vllm_logits.bin", logits, f"logits vocab={logits.size}")
                top10 = np.argsort(-logits)[:10]
                print(f"[logits] top10 ids={[int(x) for x in top10]} vals={[round(float(logits[x]),3) for x in top10]}", flush=True)
        except Exception as e:
            print(f"[logits] best-effort logits dump skipped: {e!r}", flush=True)

    n_written = len(list(OUT_DIR.glob("vllm_L*.bin")))
    print(f"[{time.strftime('%H:%M:%S')}] DONE — {n_written} vllm_L*.bin files written", flush=True)


if __name__ == "__main__":
    main()
