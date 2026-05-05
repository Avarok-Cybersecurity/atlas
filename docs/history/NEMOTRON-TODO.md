# Nemotron-H Super 120B — Debug TODO

## Current Status

**FIXED** — Model produces coherent output at 22.5 tok/s (BF16 KV cache). Three root causes identified and fixed.

## Fixes Applied

1. **`nemotron_mamba2.rs:260`** — out_proj buffer: `ssm_qkvz()` → `qkv_output()` (write-after-read race fix) ✅
2. **`nemotron_moe.rs:170,270`** — shared_down_out: `attn_output()` → `ssm_deinterleaved()` (M-E prefill aliasing fix) ✅
3. **`model.rs`** — Hidden state norm diagnostic using `readback_bf16()` (was broken with `half::bf16`) ✅
4. **Dockerfile** — Added `jinja-templates/` COPY to runtime stage ✅
5. **`weight_map.rs:2278`** — **ROOT CAUSE**: fc2_latent_proj FP8→BF16 dequant missing for layer 3 (FP8 bytes loaded as BF16, corrupting hidden states from layer 3 onward → all-zero logits) ✅
6. **`weight_map.rs:2176`** — Attention o_proj FP8→BF16 dequant missing for layers 69, 78 (same class of bug as #5) ✅

## Docker Image

`atlas-super-120b:diag` — built and ready for deployment.

## Next Step: Deploy Diagnostics

```bash
sudo docker run --gpus all --ipc=host -p 8888:8888 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-super-120b:diag serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --max-seq-len 4096 --gpu-memory-utilization 0.92 --profile
```

Then send a test request:
```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'
```

Check logs for `DIAG L{i}` lines — they show hidden state norm after each layer during prefill:
- Norm goes to 0 early → SSM state issue
- Norm goes to Inf/NaN → numerical overflow
- Norm stable through 88 layers but bad logits → lm_head or final_norm bug

## Research Findings (False Positives)

All three "bugs" reported by the research agent are false positives for NVFP4:
1. `dense_bf16_as_f32` is exact (BF16 = top 16 bits of FP32)
2. `set_fp8_weights()` only matters for FP8 checkpoints
3. Gate weight F32→BF16 conversion is never triggered (gates are already BF16)

## Verified Correct

- All dimensions (d_inner=8192, d_xbc=10240, in_proj_size=18560)
- Buffer sizes (`.max()` guards cover Nemotron dims)
- Kernel arguments match signatures
- Weight loading for mixed FP8/NVFP4 formats
- Standard RMS norm (no offset-from-1)
- Conv1d with bias + SiLU activation
- Mamba-2 SSM decode + prefill kernels
- LatentMoE fc1/fc2 projections and expert routing

## If Diagnostics Show Early Collapse

Investigate:
- SSM state initialization (should be all zeros)
- Conv1d state initialization (should be all zeros)
- in_proj weight correctness (NVFP4 dequant)
- A_log / dt_bias values (BF16→F32 conversion)

## If Diagnostics Show Late NaN/Inf

Investigate:
- Accumulation overflow in SSM state (128 heads × 64 head_dim × 128 state_size)
- Missing state normalization (Nemotron doesn't use GDN norm like Qwen3)
- MoE weighted sum overflow with top-22 experts
