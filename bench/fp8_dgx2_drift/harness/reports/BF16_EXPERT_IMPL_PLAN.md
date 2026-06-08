# BF16 MoE Expert Loading — Implementation Plan

**Goal**: lift Atlas FP8 cargo_valid from 30% (current) to ≥90% (vLLM-parity) on opencode rust-axum harness by eliminating the per-layer 0.989 FP8 MMA-precision ceiling.

**Strategy**: dequant FP8 expert weights to BF16 at load time; run MoE expert GEMM through a new BF16 grouped GEMM kernel. Keep attention + SSM as native FP8 (those kernels are tractable; only experts dominate the 40× layer compounding).

---

## Memory budget (GB10, 119.7 GB)

| Component | FP8 (current) | BF16 experts (proposed) |
|---|---|---|
| Attention + SSM weights | ~3 GB FP8 | ~3 GB FP8 (unchanged) |
| MoE expert weights | ~32 GB FP8 | ~64 GB BF16 |
| Embeddings + norms | ~2 GB BF16 | ~2 GB BF16 |
| KV cache | ~60 GB @ seq 65536 | ~30 GB @ seq 32768 |
| Activations + buffers | ~15 GB | ~15 GB |
| **Total** | **~112 GB** | **~114 GB** at 32k context |

Fits at GPU_MEMORY_UTIL=0.88 with `--max-seq-len 32768`. opencode uses < 32k typically (most runs in harness peak at ~16k).

---

## Code changes (ranked by complexity)

### 1. New kernel: `moe_bf16_grouped_gemm.cu` (~250 LoC)

`kernels/gb10/common/moe_bf16_grouped_gemm.cu` — mirror of `moe_fp8_grouped_gemm.cu` but with BF16 weights and NO dequant step:

```cu
extern "C" __global__ void moe_bf16_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,           // [total_tokens, K] BF16
    const unsigned long long* __restrict__ B_weight_ptrs,  // [E] → [N, K] BF16
    __nv_bfloat16* __restrict__ C,                // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,        // [E+1]
    const int* __restrict__ sorted_token_ids,      // [total_expanded]
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
);
```

- Tile/MMA layout: identical to FP8 v2 (`M_TILE=64, N_TILE=64, K_STEP=16, 4 warps, m16n8k16.f32.bf16.bf16.f32`).
- Per-K-step: load `__nv_bfloat16* B_exp` directly into smem_B (no LUT, no scale). Strip the inner_acc / outer_acc two-level dance — single FP32 accumulator across K.
- Coalesced thread mapping: copy v2's `thread_group / k_offset` layout verbatim.
- ~250 LoC after pruning all FP8-specific paths.

### 2. Build registration

- Add `t{N}__moe_bf16_grouped_gemm.ptx` entry to `kernels/gb10/common/KERNEL.toml`.
- Add tag binding in `crates/atlas-kernels/src/lib.rs` matching the FP8 pattern.

### 3. Rust dispatch wrapper (~80 LoC)

`crates/spark-model/src/layers/ops/bf16_moe.rs` — new file:

```rust
pub fn moe_bf16_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b_weight_ptrs: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()>
```

Identical signature to `fp8_moe_grouped_gemm` minus scale pointers.

### 4. Weight loader path (~150 LoC across 3 files)

**`crates/spark-model/src/weight_map/ssm_qwen35.rs::load_moe_qwen35`** — add `Fp8DequantedToBf16` arm:

```rust
let load_expert = |prefix: &str| -> Result<ExpertWeight> {
    match variant {
        Nvfp4Variant::Fp8DequantedToBf16 => {
            // Dequant FP8 block-scaled → BF16, return as DenseWeight wrapped
            // in QuantizedWeight::Bf16Dense (new variant)
            Ok(ExpertWeight {
                gate_proj: dequant_fp8_to_quantized_bf16(store, &format!("{prefix}.gate_proj"), inter, h, gpu)?,
                up_proj: dequant_fp8_to_quantized_bf16(store, &format!("{prefix}.up_proj"), inter, h, gpu)?,
                down_proj: dequant_fp8_to_quantized_bf16(store, &format!("{prefix}.down_proj"), h, inter, gpu)?,
            })
        }
        // ... existing arms ...
    }
};
```

**`crates/spark-model/src/weight_map/mod.rs`** — extend `QuantizedWeight` enum:

```rust
pub enum QuantizedWeight {
    Nvfp4(Nvfp4Weight),
    Bf16Dense(DenseWeight),  // NEW
}
```

**`crates/spark-model/src/weight_loader/qwen35/load_layers.rs`** — detect new variant:

```rust
let dequant_moe_to_bf16 = std::env::var("ATLAS_FP8_DEQUANT_MOE_TO_BF16").ok().as_deref() == Some("1");
let variant = if native_fp8 && dequant_moe_to_bf16 {
    Nvfp4Variant::Fp8DequantedToBf16
} else { variant };
```

### 5. MoE forward dispatch (~50 LoC)

**`crates/spark-model/src/layers/moe/forward_prefill.rs`** + **`forward.rs`** — add branch:

```rust
match (self.weights.experts[0].gate_proj, ...) {
    QuantizedWeight::Bf16Dense(_) => {
        ops::moe_bf16_grouped_gemm(ctx.gpu, self.bf16_moe_k, ...)
    }
    QuantizedWeight::Nvfp4(_) => {
        // existing NVFP4 path
    }
}
```

### 6. Kernel handle plumbing (~20 LoC)

Add `bf16_moe_k: KernelHandle` field to `MoeLayer` in `layers/moe/mod.rs`. Wire through factory.rs and the kernel registry.

---

## Testing strategy

1. **Cosine micro-bench** (~30 min): run `bench/fp8_dgx2_drift/cosine_run.py` against L20 / L31-L39 ssm.moe_out with the new variant active. Expect ≥ 0.999 per layer (vs current 0.989).
2. **Harness N=10** (~25 min): run `bf16experts` tier against sm1_a2ao_sc1 baseline. Target: ≥ 80% cargo_valid. Statistically significant lift expected.
3. **Performance check**: decode tok/s should remain within ~20% of native FP8 (BF16 GEMM is bandwidth-bound on GB10's LPDDR5X; expert weights stay on-chip in smem during accumulation).

## Risk register

- **Memory pressure at long context**: --max-seq-len must drop to 32768 from 65536. Document in MODEL.toml.
- **EP=2 not implemented for new path**: scope this to single-rank first. EP=2 BF16 expert sharding needs separate work.
- **Shared expert + MTP head**: `shared_expert` and `mtp.layers.0.mlp` share the same loader path; need to extend both. Otherwise MTP verify may fall back to FP8 and reintroduce drift.

## Cumulative effort estimate

- Kernel + Rust wrapper: 6-10 hours
- Weight loader changes: 3-5 hours
- MoE dispatch + factory plumbing: 2-3 hours
- Tests + harness validation: 2-3 hours
- **Total: 1.5-2.5 days of focused work**

## What to verify *before* writing code

1. **One real cosine run** with `dequant_fp8_blockscaled_to_bf16` applied to a single L20 expert + a custom call into `dense_gemm_bf16` for that expert's matmul. If cosine improves >0.999, the kernel-rewrite path is confirmed worth the work. If still ~0.989, the bug is elsewhere and don't ship the kernel.

The micro-bench can use Python (PyTorch + Atlas runtime FFI) — no kernel work required. **Highly recommended as the first step.**
