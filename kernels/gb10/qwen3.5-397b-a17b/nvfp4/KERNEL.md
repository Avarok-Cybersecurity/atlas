# GB10 / Qwen3.5-397B-A17B / NVFP4 — Kernel Context

> AI instruction context for optimizing kernels in this (H, M, Q) target.

## Hardware: GB10 (SM121)

- **Architecture**: SM121 (Blackwell, compute capability 12.1)
- **Memory**: 120 GB LPDDR5X @ 273 GB/s peak bandwidth
- **Practical bandwidth**: ~65-70% of peak (178-191 GB/s) due to memory controller overhead
- **No multi-CTA clusters**: ClusterShape forced to 1x1x1
- **No HBM**: LPDDR5X has higher latency, lower per-pin bandwidth than HBM3e
- **FP4 tensor cores**: SM120_16x8x64_TN_VS (native E2M1 support)
- **Missing PTX**: `cvt.rn.satfinite.e2m1x2.f32` not available on SM121

## Compilation

```bash
nvcc --ptx -arch=sm_121f -O3 --use_fast_math <file>.cu
```

## Model: Qwen3.5-397B-A17B-NVFP4

- **Parameters**: 397B total, ~17B active per token (MoE)
- **Architecture**: Hybrid — 15 full attention + 45 linear attention (Gated DeltaNet)
- **Layer pattern**: 3:1 (linear, linear, linear, full) — `full_attention_interval = 4`, 60 layers
- **Full Attention**: GQA 32:2 (32 query heads, 2 KV heads), head_dim=256
- **Linear Attention**: 16 key heads (dim 128), 64 value heads (dim 128)
- **DeltaNet**: Gated delta rule with causal Conv1d (kernel=4)
- **MoE**: 512 experts, top-10 routing per token + 1 shared expert (intermediate 1024)
- **Hidden dim**: 4096
- **Vocab**: 248,320 tokens
- **MRoPE**: `mrope_interleaved=true`, `mrope_section=[11, 11, 10]` — `parse_config`
  rewrites `model_type` qwen3_5_moe → qwen3_6_moe (interleaved 3-axis RoPE on the
  full-attention layers; text-only serving keeps H/W position IDs at zero)
- **Vision**: ViT tower present in the checkpoint, but the qwen3_6_moe path loads
  via the text-only Qwen35WeightLoader — vision weights are unused for text serving
- **MTP**: 1 full-attention MTP layer in the config, but the NVFP4 checkpoint ships
  no usable MTP head and MTP regresses throughput here — served without speculation
- **Q/Gate interleaving**: HF q_proj output is per-head interleaved
  `[Q_h0, G_h0, Q_h1, G_h1, ...]`; deinterleave before RoPE/norm

## Multi-node: EP=4, TP=1 (mandatory)

The NVFP4 weights (~200 GB) exceed a single GB10's 120 GB, so this target runs
**expert-parallel across all four GB10 nodes** (EP=4). Tensor parallelism is not
an option: `num_key_value_heads = 2` cannot shard across 4 TP ranks. Launch via
`/home/cluster/launch-atlas-ep4.sh` (`--tp-size 1 --ep-size 4 --world-size 4`).

## Quantization: Mixed NVFP4 + BF16 (ModelOpt)

`quantization_config.quant_algo = NVFP4` (ModelOpt export). As with the smaller
Qwen3.5 MoE targets, not all tensors are NVFP4: full-attention and MoE-expert
projections are NVFP4, while linear-attention projections, the LM head, conv1d,
norms, and gates remain BF16 (the quantizer skips them).

## Kernel reuse (why this dir is thin)

This target carries only three `.cu` overrides — `gated_delta_rule.cu`,
`moe_w4a16_grouped_gemm.cu`, `w4a16_gemm.cu` — copied verbatim from
`qwen3.5-35b-a3b/nvfp4/` (the repo stores copies, not symlinks). Everything else
comes from `kernels/gb10/common/`. This is correct because:

- The MoE grouped GEMMs (`moe_w4a16_grouped_gemm`, `w4a16_gemm`) are
  **byte-identical** between the 35B and 122B targets — they tile over runtime
  M/N/K and branch on a runtime `unsigned int num_experts` (no compile-time
  expert count), so 512 experts work unchanged.
- `gated_delta_rule.cu` takes `num_v_heads`/`num_k_heads`/`v_dim`/`k_dim` as
  runtime args; the 35B copy additionally has a register-tiled decode path that
  assumes per-head `K_DIM = v_dim = 128`, which holds here
  (`linear_key_head_dim = linear_value_head_dim = 128`). 64 value heads vs the
  35B's 32 only scale the launch grid (`grid.x = num_v_heads`).

## MoE Dispatch (512 experts, top-10)

- 3D grid `(expert_idx, row_chunk, col_chunk)`; expert GEMV fuses gate_up + SiLU + down.
- Routing uses the **softmax** top-k path (`moe_topk_softmax`,
  `kernels/gb10/common/moe_topk.cu`), whose `__shared__` selection buffers are
  sized `MAX_EXPERTS = 512` — this model's 512 experts fit **exactly** at that
  ceiling. (The single-CTA `moe_gate_topk_fused`, capped at 256, is not used.)
- `norm_topk_prob = true` (Qwen3.5 MoE normalizes top-k expert weights).

## DeltaNet Linear Attention (45 layers)

- 64 value heads × 16 key heads, all dim 128; depthwise causal Conv1d (kernel=4).
- Recurrent FP32 state per layer: 64 heads × 128 × 128 ≈ 4.19 MB; ×45 layers ≈ 189 MB.

## Full Attention (15 layers)

- GQA 32:2, head_dim=256, partial_rotary_factor=0.25, MRoPE-interleaved.
- Paged KV cache (block_size=16), FP8 with cache_stride.
- Q/Gate interleaving requires the deinterleave_qg kernel.
