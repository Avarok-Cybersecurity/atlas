# Primitives: norm / activation / embed / reduce

Four small crates — `atlas-norm`, `atlas-activation`, `atlas-embed`, `atlas-reduce` — that together define the **primitive-op trait surface** every layer composes. They are siblings; this chapter covers them as a group because each one is a single trait with narrow scope.

The common pattern: *this crate owns the trait; `atlas-kernels` owns the PTX; `spark-model` owns the call sites; `spark-runtime` owns the backend impl.*

## `atlas-norm` — normalization

```rust
pub trait Normalize {
    fn rms_norm(
        &self, input_ptr: u64, weight_ptr: u64, output_ptr: u64,
        num_tokens: u32, hidden_size: u32, eps: f32, stream_ptr: u64,
    ) -> Result<()>;

    fn gated_rms_norm(
        &self, input_ptr: u64, gate_ptr: u64, weight_ptr: u64, output_ptr: u64,
        num_tokens: u32, hidden_size: u32, eps: f32, stream_ptr: u64,
    ) -> Result<()>;
}
```

Two ops:

- **RMSNorm** — `y = weight ⊙ x / rms(x)`, the residual-stream normalizer used by every transformer layer in every supported model.
- **Gated RMSNorm** — `y = silu(gate) ⊙ rms_norm(x)`, the Mamba-specific variant fused with the SiLU gate. Used by Nemotron-H and the SSM branches of Qwen3.5 / Qwen3.6.

Both ops hit the kernel benchmark top-10 for speedup — RMSNorm runs 6–9× faster than PyTorch on GB10, RoPE hits 18× (see [Benchmarks](../operations/benchmarks.md)). The CUDA kernels in `kernels/<hw>/<model>/<quant>/rms_norm.cu` use a single-block tree reduction and fused scale multiplication.

## `atlas-activation` — activation functions

```rust
pub trait Activation {
    fn silu_mul(
        &self, gate_ptr: u64, up_ptr: u64, output_ptr: u64,
        num_elements: u32, stream_ptr: u64,
    ) -> Result<()>;

    fn silu_mul_quant(
        &self, gate_ptr: u64, up_ptr: u64, output_ptr: u64, scale_ptr: u64,
        num_elements: u32, group_size: u32, stream_ptr: u64,
    ) -> Result<()>;
}
```

Two fused ops:

- **`silu_mul`** — `output = silu(gate) * up`, the SwiGLU activation used by every MoE expert and the dense FFNs in Qwen3-VL, Gemma-4 (GeGLU variant), and every recent model. Implemented as a single-pass elementwise kernel.
- **`silu_mul_quant`** — the same op fused with NVFP4 quantization. Used between the expert up-projection and the expert down-projection: the up-proj output is never materialised in BF16, it's computed, SwiGLU'd, and NVFP4-quantized in one kernel. Saves an expensive round-trip through memory.

The kernel benchmarks show `silu_mul` at 2.4–4.8× vs PyTorch; the fused-quant variant saves roughly another 30% by avoiding the BF16 intermediate.

## `atlas-embed` — position + token embeddings

Two modules:

- **`token`** — embedding-lookup ops. Batched `embed_ids` that reads from the shared `embed_tokens` weight tensor and writes directly into the residual-stream buffer.
- **`rope`** — Rotary Position Embedding. Critical for every model in the matrix.

RoPE has two variants in use:

- **Standard RoPE** — used by Qwen3, Nemotron-H, Mistral, Gemma-4 full attention.
- **MRoPE** (multi-RoPE) — used by Qwen3.5 / Qwen3.6 / Qwen3-VL for vision models. Splits the head dim into spatial (H, W) and temporal (T) segments and applies RoPE to each independently. The buffer arena carries a `3× positions` scratch instead of `1×`; a bug in early builds where scratch was sized for `1×` caused `cuMemcpyHtoDAsync status 1` at long context — see the [MRoPE scratch fix](https://github.com/Avarok-Cybersecurity/atlas/blob/main/docs/history) notes.

The actual RoPE kernel (`rope.cu`) is one of the top-three-fastest ops in the benchmark suite — 18.2× faster than PyTorch at seq=512 GQA16:2. The trick is precomputing `cos`/`sin` on the Grace CPU via NEON SIMD (`precompute_freqs_cis_simd` for AArch64), so the GPU only does the rotation, not the trig.

## `atlas-reduce` — reductions and MoE gating

```rust
pub trait Reduce {
    fn topk(
        &self, scores_ptr: u64, topk_ids_ptr: u64, topk_weights_ptr: u64,
        num_tokens: u32, num_experts: u32, topk: u32, stream_ptr: u64,
    ) -> Result<()>;

    fn moe_sum(
        &self, expert_outputs_ptr: u64, topk_weights_ptr: u64, output_ptr: u64,
        num_tokens: u32, hidden_size: u32, topk: u32, stream_ptr: u64,
    ) -> Result<()>;

    fn softmax(
        &self, input_ptr: u64, output_ptr: u64,
        num_rows: u32, num_cols: u32, stream_ptr: u64,
    ) -> Result<()>;
}
```

Three ops, all MoE-adjacent:

- **`topk`** — for each token, select the top-k experts by gate score. Supports up to 256 experts (MiniMax-M2.7) and topk up to 10. The kernel uses a warp-level bitonic + selection.
- **`moe_sum`** — weighted sum of expert outputs after dispatch/gather. Used at the end of an MoE block to reduce back to the residual-stream shape.
- **`softmax`** — row-wise softmax. Used on the gate logits before `topk` (for some models) and in the attention kernel (for others — some attention kernels fuse softmax, some call this separately during prefill).

There is also an **`argmax_bf16`** kernel referenced from here but living in `kernels/<hw>/<model>/<quant>/argmax.cu`. It is used by the greedy sampler and is a case where Atlas is dramatically faster than PyTorch — a single-block tree reduction that returns a 4-byte result to the host, instead of PyTorch's approach of computing a 304 KB logits vector on device and D2H-copying the lot. Bandwidth savings compound with `--adaptive-sampling` enabled.

## What these crates are not

- Not the kernel source. That's `kernels/<hw>/<model>/<quant>/`.
- Not the backend impls. Those live in `spark-runtime/src/cuda_backend.rs` (the `AtlasCudaBackend` implements every primitive trait).
- Not mandatory. Some layers don't need all four — a vision-only model barely touches `atlas-reduce`. The trait boundary is there so that when you *do* need one of these, you never have to reach past it.

The pattern — one trait per op family, impl in `spark-runtime`, kernel in the `kernels/` tree — is how every primitive composition in Atlas works. When you're adding a new op family (e.g. convolutions for a ViT variant that needs them), the first move is to add a crate here.
