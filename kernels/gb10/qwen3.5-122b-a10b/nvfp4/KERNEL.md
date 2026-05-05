# Qwen3.5-122B-A10B-NVFP4 on GB10

## Kernel Origin

Inherits all kernels from parent `kernels/gb10/nvfp4/`. Model-specific `.cu` files
in this directory shadow the parent versions for dimension-tuned optimizations.

## Dimension Changes vs 35B

| Parameter | 35B | 122B |
|-----------|-----|------|
| hidden_size | 2048 | 3072 |
| num_attention_heads | 16 | 32 |
| linear_num_value_heads | 32 | 64 |
| moe_intermediate_size | 512 | 1024 |
| shared_expert_intermediate_size | 512 | 1024 |
| num_hidden_layers | 40 | 48 |
| num_experts | 256 | 256 (same) |
| head_dim | 256 | 256 (same) |

## Shadowed Kernels

### `moe_shared_expert_fused_batch2.cu` (MoE GEMV for K=2 verify)

Wider block optimization: BLOCK_SIZE=256 (vs parent's 128), giving 64 threads per
output (2 warps with cross-warp shared memory reduction) instead of 32.

**Rationale**: The 122B's larger dimensions (K=3072 hidden, N=1024 intermediate) mean
more memory bandwidth is needed per MoE layer. Doubling threads per output increases
memory-level parallelism (more outstanding LPDDR5X requests per SM), improving
bandwidth utilization for the GEMV-dominated MoE computation.

**Impact**: 48 MoE layers × 2 kernel launches (gate_up + silu_down) = 96 launches
affected. Each launch uses BLOCK_SIZE=256 instead of 128, with the same grid
dimensions. Expected improvement: 5-15% MoE throughput gain (bandwidth-bound).

**Trade-off**: Fewer concurrent blocks per SM (8 vs 16), offset by better per-block
bandwidth utilization from 2× MLP per output.

## Memory Budget

- Weights: ~76 GB (NVFP4 packed)
- SSM state: ~4.7 GB per slot (36 layers × 64 value_heads × 128 value_dim × 2 arrays × 4 bytes)
- With 4 slots: ~95 GB total → fits in 119.7 GB GB10 GPU
- KV cache: negligible (12 attention layers, 2 KV heads)

## Optimization Notes

- All kernels are dimension-parameterized via kernel args (N, K, etc.)
- head_dim=256 is identical to 35B — attention kernels unchanged
- MoE expert count (256) identical to 35B — topk/permute kernels unchanged
- `moe_gate_topk.cu` uses `extern __shared__` (dynamic sizing) — no shadow needed
