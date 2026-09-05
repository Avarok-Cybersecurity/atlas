// SPDX-License-Identifier: AGPL-3.0-only

//! Dense SwiGLU FFN component for non-MoE models.
//!
//! Forward: gate = gate_proj(x), up = up_proj(x), out = down_proj(SiLU(gate) * up)
//! 2 fused kernel launches per decode token (dual GEMV + SiLU-fused down GEMV).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::{
    DenseWeight, Fp8Weight, Fp8WeightTransposed, PackedQ2Weight, QuantizedWeight,
};

pub struct DenseFfnWeights {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    /// Transposed ([K/2, N]) copies for the fast `w4a16_gemm_t_m128` prefill
    /// kernel. `None` → prefill falls back to the slow M64xN64 base kernel.
    /// The non-transposed copies above are kept for the decode gemv path.
    pub gate_proj_t: Option<QuantizedWeight>,
    pub up_proj_t: Option<QuantizedWeight>,
    pub down_proj_t: Option<QuantizedWeight>,
}

/// BF16 dense MLP weights — alternative to NVFP4 for precision-sensitive
/// models (Gemma-4-31B). Each is `[N, K]` row-major BF16. When installed
/// on a `DenseFfnLayer` via `set_bf16_weights`, the forward paths
/// dispatch to `dense_gemv_bf16` / `dense_gemm_bf16` instead of the
/// w4a16 NVFP4 kernels. Costs ~3.4 GB extra GPU memory on Gemma-4-31B
/// (3 × hidden×intermediate × 2 bytes) vs NVFP4's 0.5 bytes/weight.
pub struct DenseFfnWeightsBf16 {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// Native block-scaled FP8 dense MLP weights — loaded directly from an FP8
/// checkpoint (no NVFP4 requant). When installed via `set_fp8_weights`, decode
/// dispatches `w8a16_gemv` and prefill `w8a16_gemm` per projection (BF16 act ×
/// FP8 E4M3 weight with 2D block scales), mirroring the SSM/attention FP8 path.
pub struct DenseFfnWeightsFp8 {
    pub gate_proj: Fp8Weight,
    pub up_proj: Fp8Weight,
    pub down_proj: Fp8Weight,
}

/// Native keep-packed ternary Q2_0 dense MLP weights — loaded directly from a
/// PrismML Q2_0 GGUF (`ATLAS_GGUF_NATIVE_Q2=1`) with NO dequant / NVFP4 requant.
/// Each projection is a raw `block_q2_0` buffer (2-bit codes + inline fp16 scale
/// per group). When installed via `set_q2_weights`, decode dispatches
/// `q2_0_gemv` (BF16 act × 2-bit weight, dequant-in-dot-product), mirroring the
/// FP8 path but with the weights ~4× smaller resident.
pub struct DenseFfnWeightsQ2 {
    pub gate_proj: PackedQ2Weight,
    pub up_proj: PackedQ2Weight,
    pub down_proj: PackedQ2Weight,
}

/// Activation function for gated FFN (SiLU for Qwen/Llama, GELU for Gemma-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SiLU,
    GeLU,
}

/// A per-projection int8 W4A8 weight, built lazily from the NVFP4 weight on the
/// first `ATLAS_INT8_PREFILL` prefill (see `DenseFfnLayer::ensure_int8_weight`).
/// `w_i8` is `[N, K]` signed int8; `w_scale` is `[N, K/32]` F32. Cached for the
/// process lifetime in a `OnceLock`, so the requant kernel runs once per weight.
#[derive(Debug, Clone, Copy)]
struct Int8Weight {
    w_i8: DevicePtr,
    w_scale: DevicePtr,
}

/// Q4_K-quantized FFN weight (GGML block_q4_K layout), materialized once at first
/// `ATLAS_FFN_MMQ` prefill and cached for process lifetime in a `OnceLock`.
#[derive(Debug, Clone, Copy)]
struct Q4kWeight {
    w_q4k: DevicePtr,
}

/// block_nvfp4-repacked FFN weight for the `ATLAS_FFN_NVFP4_MMQ` W4A4 prefill arm.
/// Raw bit shuffle of the checkpoint's NVFP4 (same e2m1 codes + e4m3 scale bytes,
/// same total bytes) — materialized once and cached for process lifetime.
#[derive(Debug, Clone, Copy)]
struct Fp4MmqWeight {
    w: DevicePtr,
}

pub struct DenseFfnLayer {
    pub weights: DenseFfnWeights,
    activation: FfnActivation,
    w4a16_gemv: KernelHandle,
    /// Single-warp `w4a16_gemv_sw`. `KernelHandle(0)` on miss → base GEMV.
    w4a16_gemv_sw: KernelHandle,
    w4a16_gemv_dual: KernelHandle,
    w4a16_gemv_silu_input: KernelHandle,
    // LOSSLESS single-warp-per-output decode variants (8 outputs/block, no smem
    // cross-warp reduce). Bit-identical to the 64-thread kernels (proven by the
    // w4a16_gemv_sw microtest). Default ON via `ModelLevers::gemv_sw`;
    // `ATLAS_NO_GEMV_SW=1` restores the 64-thread kernels. KernelHandle(0) on
    // miss → fall back to base kernels.
    w4a16_gemv_dual_sw: KernelHandle,
    w4a16_gemv_silu_input_sw: KernelHandle,
    w4a16_gemv_dual_batch2: KernelHandle,
    w4a16_gemv_dual_batch3: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    /// Narrow `w4a16_gemv_batch{M}` family (M=4..8) for the K=4 verify FFN and
    /// the K=5..8 chain verify. SSOT for the M -> tier decision; individual
    /// tiers are 0-handles when the target did not load them.
    w4a16_batchm: W4a16BatchmTiers,
    w4a16_gemm: KernelHandle,
    // 128x128 2-stage cp.async pipelined w4a16 GEMM — the fast prefill kernel
    // attention/SSM already use. The base `w4a16_gemm` (M64xN64) only hits
    // ~10 TFLOPS at M=8k and was the flat ~155 tok/s dense-FFN prefill
    // bottleneck on Qwen3.6-27B. KernelHandle(0) on miss → scalar-tile fallback.
    w4a16_gemm_t_m128_k: KernelHandle,
    // v2: 8-warp (256-thread) variant of t_m128 — parallel chunk MMAs, 3 CTAs/SM.
    // Preferred over t_m128 for dense-FFN prefill when present. KernelHandle(0) → use t_m128.
    w4a16_gemm_t_m128_v2_k: KernelHandle,
    // LOSSLESS BF16 variant of t_m128: same 128x128 cp.async tiling, but FP4→BF16
    // dequant + BF16 m16n8k16 MMA (FP32 accum) instead of the FP8-E4M3 crush the
    // default NVIDIA t_m128 uses. The FP8 path perturbs generation (measured
    // length-truncations / accuracy risk on Qwen3.6-27B); this kernel keeps prefill
    // outputs bit-for-bit vs the base `w4a16_gemm`. OPT-IN only, gated by
    // ATLAS_BF16_TC_PREFILL (default off → dispatch unchanged). KernelHandle(0) on miss.
    w4a16_gemm_t_m128_bf16_k: KernelHandle,
    // v2 of the LOSSLESS BF16 128x128 prefill kernel: same MMA instruction order
    // (so BIT-IDENTICAL to bf16_k, proven by w4a16_bf16_v2_microtest) but a
    // smaller A-tile smem pad lifts occupancy from 2→3 CTAs/SM (~+50% resident
    // warps), giving a measured ~3-8% faster prefill GEMM on this latency-bound
    // kernel. Preferred over bf16_k when present. KernelHandle(0) on miss → bf16_k.
    w4a16_gemm_t_m128_bf16_v2_k: KernelHandle,
    // FP8 M64 prefill (w4a16_gemm_t): m16n8k32 e4m3 MMA + M_TILE=64. Packed 1-byte
    // operands cut shared-memory load instructions ~4x (the v2 BF16 path is
    // smem-bandwidth-bound, L1/TEX 90% per ncu), and M64's lower register pressure
    // lifts occupancy → measured ~44 TFLOP/s vs ~30 for v2 (~1.47x prefill) on dgx1.
    // LOSSY (FP8 E4M3, cosine ~0.9997) — OPT-IN via ATLAS_FP8_M64_PREFILL, gated on
    // quality. KernelHandle(0) on miss → dispatch unchanged.
    w4a16_gemm_t_k: KernelHandle,
    // int8 W4A8 prefill (ATLAS_INT8_PREFILL): the validated requant→faith2
    // pipeline (cosine 0.999978). `int8_gemm_faith2` is an int8×int8 MMA with
    // per-32 block scales, so BOTH operands must be int8 — unlike the FP8 path
    // (mixed BF16×FP8). At first int8 prefill we requant the NVFP4 gate/up/down
    // weights to int8 once (`requant_w_nvfp4_int8`, cached in the OnceLocks
    // below) and requant the BF16 activations every call (`requant_a_bf16_int8`,
    // into `int8_a_scratch`). KernelHandle(0) on miss → arm never taken.
    int8_faith2_k: KernelHandle,
    // faith5: int32 per-sb accumulation (breaks the MMA→scale dependency chain).
    // Opt-in via ATLAS_INT8_FAITH5=1 (replaces faith2 for int8 prefill GEMMs).
    int8_faith5_k: KernelHandle,
    requant_w_int8_k: KernelHandle,
    requant_a_int8_k: KernelHandle,
    // Lazily-built, process-lifetime int8 weight copies (one per projection),
    // requanted from `self.weights.{gate,up,down}_proj`. Only ever touched when
    // ATLAS_INT8_PREFILL is set → default-off path is byte-identical.
    int8_gate: std::sync::OnceLock<Int8Weight>,
    int8_up: std::sync::OnceLock<Int8Weight>,
    int8_down: std::sync::OnceLock<Int8Weight>,
    // Activation-requant scratch for the int8/NVFP4/Q4_K prefill GEMMs is now
    // shared, arena-owned (BufferArena::ffn_act_{q8,a,scale}), sized once for
    // max_batch_tokens × max(h, inter) — no per-layer allocation.
    // W4A4 native-FP4 prefill (ATLAS_FP4_PREFILL): NVFP4 weights consumed directly
    // (no requant), BF16 activations quantized to NVFP4 each call into ffn_act_a/scale.
    // KernelHandle(0) on miss → arm never taken (default-off byte-identical).
    w4a4_gemm_k: KernelHandle,
    quantize_nvfp4_k: KernelHandle,
    // Q4_K MMQ prefill (ATLAS_FFN_MMQ): vendored llama Q4_K W4A8 GEMM. Weights
    // materialized NVFP4→bf16→Q4_K once (lazy, cached in the OnceLocks); activations
    // quantized to q8_1_mmq each call into ffn_act_q8. KernelHandle(0) → arm skipped.
    q4k_mmq_nc_k: KernelHandle,
    q4k_mmq_wc_k: KernelHandle,
    q4k_quant_act_k: KernelHandle,
    q4k_quant_w_k: KernelHandle,
    dequant_nvfp4_bf16_k: KernelHandle,
    q4k_gate: std::sync::OnceLock<Q4kWeight>,
    q4k_up: std::sync::OnceLock<Q4kWeight>,
    q4k_down: std::sync::OnceLock<Q4kWeight>,
    // NVFP4 W4A4 MMQ prefill (ATLAS_FFN_NVFP4_MMQ): vendored llama Blackwell block-scale
    // FP4 MMA (80 TFLOP/s vs t_m128 ~51 on GB10). Gate/up weights repacked ONCE at load
    // (raw bit shuffle, checkpoint layout → block_nvfp4, zero requantization); activations
    // quantized per call into the shared ffn_act_q8 scratch; the per-tensor scale2 is
    // folded in the scaled SiLU-mul. KernelHandle(0) → arm skipped.
    nvfp4_mmq_nc_k: KernelHandle,
    nvfp4_mmq_wc_k: KernelHandle,
    /// M-sized MMQ tiles for DECODE. The 128 tile issues MMAs for all 128 columns
    /// regardless of m, so at m=16 it discards 112 of them; these size the tile to
    /// the batch. try_kernel: 0-handle -> dispatch keeps the 128 tile.
    nvfp4_mmq16_nc_k: KernelHandle,
    nvfp4_mmq16_wc_k: KernelHandle,
    nvfp4_mmq32_nc_k: KernelHandle,
    nvfp4_mmq32_wc_k: KernelHandle,
    nvfp4_mmq64_nc_k: KernelHandle,
    nvfp4_mmq64_wc_k: KernelHandle,
    nvfp4_quant_act_k: KernelHandle,
    nvfp4_repack_k: KernelHandle,
    nvfp4_silu_scaled_k: KernelHandle,
    nvfp4_silu_quant_k: KernelHandle,
    nvfp4_scale_k: KernelHandle,
    fp4mmq_gate: std::sync::OnceLock<Fp4MmqWeight>,
    fp4mmq_up: std::sync::OnceLock<Fp4MmqWeight>,
    fp4mmq_down: std::sync::OnceLock<Fp4MmqWeight>,
    // Small-M (DFlash verify M=17) routing companion to `w4a16_gemm_t_k`
    // (declared above): deep-K variant. w4a16_m17_bench: `w4a16_gemm_t_k64`
    // wins deep-K down_proj (554 vs 810us at K=17408); the M64-tile
    // `w4a16_gemm_t` beats M128 tiles at M<=64 (283 vs 324us on gate/up).
    // KernelHandle(0) → m128 dispatch.
    w4a16_gemm_t_k64_k: KernelHandle,
    /// SiLU(gate)*up or GELU(gate)*up depending on activation.
    act_mul: KernelHandle,
    /// BF16 dense MLP weights — when `Some`, all forward paths use the
    /// `dense_gemv_bf16` / `dense_gemm_bf16` kernels instead of w4a16
    /// NVFP4. Falls back to the NVFP4 weights when `None`. Set via
    /// `set_bf16_weights`. Used by Gemma-4 dense to avoid the structural
    /// NVFP4 attention drift on greedy code generation (the fib test's
    /// broken-indentation pattern).
    bf16_weights: Option<DenseFfnWeightsBf16>,
    dense_gemv_bf16_k: KernelHandle,
    dense_gemm_bf16_k: KernelHandle,
    // Tensor-core BF16 GEMM (m16n8k16 MMA) for the dense-FFN PREFILL path.
    // The scalar `dense_gemm_bf16` is ~10x too slow on long prefills (it was
    // the flat ~155 tok/s prefill bottleneck on Qwen3.6-27B dense NVFP4).
    // KernelHandle(0) on miss → forward_prefill falls back to the scalar path.
    // Decode (gemv, M=1) is untouched, so TPOT is unaffected.
    dense_gemm_tc_k: KernelHandle,
    /// Native FP8 dense MLP weights — when `Some`, decode/prefill dispatch the
    /// block-scaled FP8 kernels (`w8a16_gemv` / `w8a16_gemm`) instead of w4a16
    /// NVFP4. Set via `set_fp8_weights` for native FP8 checkpoints (Qwythos /
    /// Ornith-FP8). Spec-decode batched paths fall back to dequant — dense
    /// qwen3_5 has no MTP, so they're never reached.
    fp8_weights: Option<DenseFfnWeightsFp8>,
    w8a16_gemv_k: KernelHandle,
    w8a16_gemm_k: KernelHandle,
    // Fused FP8 decode GEMVs (gate+up in one launch / silu+down in one launch),
    // mirroring the NVFP4 w4a16_gemv_dual / w4a16_gemv_silu_input. KernelHandle(0)
    // on miss → fall back to the 3-launch w8a16_gemv path. Module = .cu file stem.
    w8a16_gemv_dual_k: KernelHandle,
    w8a16_gemv_silu_input_k: KernelHandle,
    // Fast transposed FP8 prefill GEMM (128x128 / 8-warp / two-level FP32 fold).
    // Preferred over w8a16_gemm when a transposed FP8 weight copy is present.
    // KernelHandle(0) → fall back to non-transposed w8a16_gemm.
    w8a16_gemm_t_m128_k: KernelHandle,
    /// v0 LoRA overlay for gate/up/down. `set_lora_weights` REJECTS layers
    /// where `fp8_weights`, `bf16_weights` or `q2_weights` are installed (v0
    /// supports the NVFP4 dispatch path only — those branches early-return
    /// before the NVFP4 tail where the deltas land; holo is NVFP4 so it is
    /// unaffected).
    ///
    /// M1 (2026-08-19): the deltas are APPLIED. `apply_lora_gate_up` runs
    /// after the gate/up projection and before `silu_mul`; `apply_lora_down`
    /// runs after the down projection. Every NVFP4 dispatch this layer can
    /// take — decode `forward`, `forward_k2`/`k3`/`km`, `forward_prefill`,
    /// `forward_batched` — calls both, because an adapter that applies on one
    /// path and not another produces a model that contradicts itself between
    /// prefill and decode.
    ///
    /// Until M1 this field was written by `set_lora_weights` and never read,
    /// so an adapter targeting gate/up/down loaded successfully and changed
    /// nothing. On a hybrid like Qwen3.8-27B that is most of the adapter:
    /// community LoRAs for it put 67-78% of their parameter mass in the FFN.
    lora: Option<ops::lora_delta::LoraFfnWeights>,

    /// Native keep-packed ternary Q2_0 dense MLP weights (`ATLAS_GGUF_NATIVE_Q2`).
    /// When installed via `set_q2_weights`, decode dispatches `q2_0_gemv_vec`
    /// (BF16 activation × packed 2-bit weight, dequant-in-dot-product) — the
    /// weights stay 2-bit resident (no NVFP4 requant). Highest-priority forward
    /// branch. Prefill/batched paths for packed-Q2 are a deferred (Tier-2) phase
    /// and currently bail — dense qwen35 has no MTP so k2/k3 are never reached.
    q2_weights: Option<DenseFfnWeightsQ2>,
    q2_0_gemv_k: KernelHandle,
    // Batched (M=1..8) packed-Q2 decode GEMV handle. The kernel + wrapper are
    // built and validated (CPU math test), but the batched-decode call site
    // (spec-decode verify rows) is deferred to the same Tier-2 phase as prefill;
    // dense qwen35 has no MTP so no batched decode reaches the FFN today.
    #[allow(dead_code)]
    q2_0_gemv_batchm_k: KernelHandle,
    // Load-time packed-Q2 → BF16 dequant kernel (`dequant_gguf_bf16` module).
    // Used by packed-Q2 PREFILL: dequant each proj into a TRANSIENT BF16 scratch
    // buffer, run the normal BF16 GEMM, free the scratch — the resident weight
    // stays 2-bit. Decode uses the native `q2_0_gemv` (no dequant). Tier-1 path.
    dequant_q2_0_gn_k: KernelHandle,
    // Native Q2_0 MMQ prefill (Tier-2, `ATLAS_GGUF_NATIVE_Q2_MMQ=1`): keeps the
    // 2-bit weight packed and runs a tensor-core int8 MMA (dequant-in-register)
    // against a q8_1 activation — no BF16 weight scratch, no dequant tax, no race.
    // The q8_1 activation quantizer is SHARED with Q4_K (`q4k_quant_act_k`).
    // KernelHandle(0) when absent → falls back to the transient-dequant path.
    q2_0_mmq_nc_k: KernelHandle,
    q2_0_mmq_wc_k: KernelHandle,
}

/// M-sized MMQ tiles: **ON by default**, disabled by `ATLAS_NO_MMQ_SMALL_TILE=1`.
/// Strict `== "1"` on an `ATLAS_NO_*` name — presence flags here are enabled by `=0`.
fn mmq_small_tile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_MMQ_SMALL_TILE").as_deref() != Ok("1"))
}

/// The m=64 MMQ tile: **ON by default**, disabled by `ATLAS_NO_MMQ_TILE64=1`. Separate
/// from `ATLAS_NO_MMQ_SMALL_TILE` so this arm can be A/B'd without also reverting the
/// already-shipped 16/32 tiles. Strict `== "1"`, matching the sibling above.
fn mmq_tile64_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_MMQ_TILE64").as_deref() != Ok("1"))
}
impl DenseFfnLayer {
    pub fn new(weights: DenseFfnWeights, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_activation(weights, FfnActivation::SiLU, gpu)
    }

    pub fn new_with_activation(
        weights: DenseFfnWeights,
        activation: FfnActivation,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let act_mul = match activation {
            FfnActivation::SiLU => gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            FfnActivation::GeLU => gpu.kernel("gelu", "gelu_mul")?,
        };
        // BF16 path kernels — optional (only loaded if available; gemma4
        // is the only consumer today). `try_kernel` returns
        // `KernelHandle(0)` on miss so we don't break NVFP4-only models
        // that were built without these kernels. Module names per
        // `kernels/gb10/{target}/nvfp4/KERNEL.toml`:
        //   `dense_gemv_bf16 = "gemv"`, `dense_gemm_bf16 = "gemm"`.
        let dense_gemv_bf16_k = super::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let dense_gemm_bf16_k = super::try_kernel(gpu, "gemm", "dense_gemm_bf16");
        let dense_gemm_tc_k = super::try_kernel(gpu, "gemm_tc", "dense_gemm_tc");

        let layer = Self {
            weights,
            activation,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemv_dual: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            w4a16_gemv_silu_input: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?,
            w4a16_gemv_dual_sw: super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_sw"),
            w4a16_gemv_silu_input_sw: super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_silu_input_sw",
            ),
            w4a16_gemv_dual_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_dual_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_batchm: W4a16BatchmTiers::resolve(gpu),
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            w4a16_gemm_t_m128_v2_k: super::w4a16_v2_kernel(gpu),
            w4a16_gemm_t_m128_bf16_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128_bf16"),
            w4a16_gemm_t_m128_bf16_v2_k: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m128_bf16_v2",
            ),
            w4a16_gemm_t_k: super::tgemm_kernel(gpu),
            int8_faith2_k: super::try_kernel(gpu, "w4a16", "int8_gemm_faith2"),
            int8_faith5_k: super::try_kernel(gpu, "w4a16", "int8_gemm_i32acc"),
            requant_w_int8_k: super::try_kernel(gpu, "w4a16", "requant_w_nvfp4_int8"),
            requant_a_int8_k: super::try_kernel(gpu, "w4a16", "requant_a_bf16_int8"),
            int8_gate: std::sync::OnceLock::new(),
            int8_up: std::sync::OnceLock::new(),
            int8_down: std::sync::OnceLock::new(),
            w4a4_gemm_k: super::try_kernel(gpu, "w4a4", "w4a4_gemm"),
            quantize_nvfp4_k: super::try_kernel(gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4"),
            q4k_mmq_nc_k: super::try_kernel(gpu, "q4k_mmq", "atlas_q4k_mmq128_nc"),
            q4k_mmq_wc_k: super::try_kernel(gpu, "q4k_mmq", "atlas_q4k_mmq128_wc"),
            q4k_quant_act_k: super::try_kernel(gpu, "q4k_mmq", "atlas_q8_1_quantize_ds4_bf16"),
            q4k_quant_w_k: super::try_kernel(gpu, "q4k_quantize", "q4k_quantize"),
            dequant_nvfp4_bf16_k: super::try_kernel(
                gpu,
                "dequant_nvfp4_bf16",
                "dequant_nvfp4_to_bf16",
            ),
            q4k_gate: std::sync::OnceLock::new(),
            q4k_up: std::sync::OnceLock::new(),
            q4k_down: std::sync::OnceLock::new(),
            nvfp4_mmq_nc_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_mmq128_nc"),
            nvfp4_mmq_wc_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_mmq128_wc"),
            nvfp4_mmq16_nc_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_mmq16_nc"),
            nvfp4_mmq16_wc_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_mmq16_wc"),
            nvfp4_mmq32_nc_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_mmq32_nc"),
            nvfp4_mmq32_wc_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_mmq32_wc"),
            nvfp4_mmq64_nc_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_mmq64_nc"),
            nvfp4_mmq64_wc_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_mmq64_wc"),
            nvfp4_quant_act_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_quantize_bf16"),
            nvfp4_repack_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_repack"),
            nvfp4_silu_scaled_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_silu_mul_scaled"),
            nvfp4_silu_quant_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_silu_mul_quant"),
            nvfp4_scale_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_scale_bf16"),
            fp4mmq_gate: std::sync::OnceLock::new(),
            fp4mmq_up: std::sync::OnceLock::new(),
            fp4mmq_down: std::sync::OnceLock::new(),
            w4a16_gemm_t_k64_k: super::k64_kernel(gpu).unwrap_or(KernelHandle(0)),
            act_mul,
            bf16_weights: None,
            dense_gemv_bf16_k,
            dense_gemm_bf16_k,
            dense_gemm_tc_k,
            fp8_weights: None,
            w8a16_gemv_k: super::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv"),
            w8a16_gemm_k: super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemv_dual_k: super::try_kernel(gpu, "w8a16_gemv_fused", "w8a16_gemv_dual"),
            w8a16_gemv_silu_input_k: super::try_kernel(
                gpu,
                "w8a16_gemv_fused",
                "w8a16_gemv_silu_input",
            ),
            w8a16_gemm_t_m128_k: super::try_kernel(gpu, "w8a16_gemm_t_m128", "w8a16_gemm_t_m128"),
            lora: None,
            q2_weights: None,
            // Winner of the decode-GEMV bench: candidate B (vectorized code loads
            // + smem A-stage, 1 warp/row × 8 rows/CTA). ~268 GB/s (98% of the
            // 273 GB/s LPDDR5X peak) at gate/up M=1 — 9.5× the original
            // whole-block-strided `q2_0_gemv`. Same `(code-1)*d` FP32 numerics.
            q2_0_gemv_k: super::try_kernel(gpu, "q2_0_gemv_vec", "q2_0_gemv_vec"),
            q2_0_gemv_batchm_k: super::try_kernel(gpu, "q2_0_gemv_vec", "q2_0_gemv_vec_batchm"),
            dequant_q2_0_gn_k: super::try_kernel(
                gpu,
                "dequant_gguf_bf16",
                "dequant_q2_0_gn_to_bf16",
            ),
            // Resolved by `set_q2_weights`, never here: q2_0_mmq ships only
            // in GGUF-serving targets, and an unconditional probe fails the
            // boot audit on every dense-FFN model that never installs
            // packed-Q2 weights.
            q2_0_mmq_nc_k: KernelHandle(0),
            q2_0_mmq_wc_k: KernelHandle(0),
        };
        Ok(layer)
    }

    /// Load-time finalize for the Q4_K MMQ prefill path (`ATLAS_FFN_MMQ`). MUST run at
    /// load, BEFORE the KV cache is sized, so the net FFN footprint is correct when the KV
    /// cache claims free memory. Order is critical: (1) eagerly materialize the Q4_K weights
    /// (+9.63 GB) so they are accounted for now rather than lazily on first prefill (which
    /// would over-subscribe AFTER the KV cache already grabbed the freed `_t` space → decode
    /// OOM-throttle); (2) free the transposed `_proj_t` copies (−9.63 GB, dead under Q4_K
    /// prefill — only the unreachable `Some(wt)` arms read them). Net FFN = baseline; decode
    /// untouched (NVFP4 gemv on the non-`_t` copies). No-op unless Q4_K is active.
    pub fn finalize_q4k_load(
        &mut self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        stream: u64,
    ) -> Result<()> {
        // Packed-Q2 (ATLAS_GGUF_NATIVE_Q2) FFN keeps its NVFP4 source weights
        // NULL — the Q4_K prefill copy is built by dequant-ing those (NULL) NVFP4
        // blocks, so running it here is a null-ptr kernel launch (CUDA 700).
        // Packed-Q2 has its own prefill path (transient dequant), so skip.
        if self.q2_weights.is_some() {
            return Ok(());
        }
        let q4k_active = self.q4k_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && self.q4k_quant_w_k.0 != 0
            && self.dequant_nvfp4_bf16_k.0 != 0
            && std::env::var_os("ATLAS_FFN_MMQ").is_some();
        if !q4k_active {
            return Ok(());
        }
        // (1) eagerly materialize the prefill weights BEFORE freeing `_t`, so the KV cache
        // (sized after load) can't claim the freed space before the weights exist.
        // gate/up: Q4_K (N=inter,K=h). down: HYBRID → int8 faith2 (N=h,K=inter) for accuracy,
        // else Q4_K. ensure_int8_weight reads the non-`_t` NVFP4 down_proj (kept for decode gemv).
        self.ensure_q4k_weight(
            &self.q4k_gate,
            gpu,
            &self.weights.gate_proj,
            inter,
            h,
            stream,
        )?;
        self.ensure_q4k_weight(&self.q4k_up, gpu, &self.weights.up_proj, inter, h, stream)?;
        let down_faith2 = self.int8_faith2_k.0 != 0
            && self.requant_a_int8_k.0 != 0
            && std::env::var_os("ATLAS_FFN_MMQ_DOWN_Q4K").is_none();
        if down_faith2 {
            self.ensure_int8_weight(
                &self.int8_down,
                gpu,
                &self.weights.down_proj,
                h,
                inter,
                stream,
            )?;
        } else {
            self.ensure_q4k_weight(
                &self.q4k_down,
                gpu,
                &self.weights.down_proj,
                h,
                inter,
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        // (2) free the dead transposed copies
        let mut freed = 0usize;
        for wt in [
            &mut self.weights.gate_proj_t,
            &mut self.weights.up_proj_t,
            &mut self.weights.down_proj_t,
        ] {
            if let Some(w) = wt.as_ref()
                && !w.weight.is_null()
            {
                gpu.free(w.weight)?;
                gpu.free(w.weight_scale)?;
                freed += 1;
            }
            *wt = None;
        }
        if freed > 0 {
            // Log-once latch (see `atlas_core::scope`). It holds no model-derived
            // value — the message is rebuilt from the arguments every call — so a
            // stale entry cannot produce a wrong answer, only a suppressed duplicate
            // line after a model swap. Scoping it would thread a logging concern
            // through the call path to prevent one repeated INFO line.
            // Latched on the BACKEND (`OpCache::once`), which exists by load
            // time: `finalize_q4k_load` takes the `gpu` it is loading onto. A
            // static meant only the first model in the process reported the
            // decision.
            if gpu.op_cache().once("log:ffn_mmq_freed_twins") {
                tracing::info!(
                    "[atlas] ATLAS_FFN_MMQ: freed transposed FFN `_t` copies (dead under Q4_K prefill) — Q4_K weights net to ~0 vs NVFP4 baseline"
                );
            }
        }
        Ok(())
    }

    /// Eagerly materialize the block_nvfp4 gate/up copies for the `ATLAS_FFN_NVFP4_MMQ`
    /// W4A4 prefill arm at LOAD time (before KV sizing), then free the now-dead gate/up
    /// transposed `_t` copies so net FFN footprint stays at the NVFP4 baseline. Down is
    /// untouched (hybrid: it stays on the default t_m128 path for accuracy → keeps its
    /// `_t` copy). No-op unless the env + kernels are present.
    pub fn finalize_nvfp4_mmq_load(
        &mut self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        stream: u64,
    ) -> Result<()> {
        // Packed-Q2 (ATLAS_GGUF_NATIVE_Q2) FFN keeps its NVFP4 source weights
        // NULL. This W4A4-MMQ finalize is active by DEFAULT (SiLU + kernels
        // present) and repacks the NVFP4 gate/up — over NULL pointers that's a
        // CUDA-700 illegal access. Packed-Q2 uses its own decode/prefill path.
        if self.q2_weights.is_some() {
            return Ok(());
        }
        let active = self.nvfp4_mmq_nc_k.0 != 0
            && self.nvfp4_quant_act_k.0 != 0
            && self.nvfp4_repack_k.0 != 0
            && self.nvfp4_silu_scaled_k.0 != 0
            && matches!(self.activation, FfnActivation::SiLU)
            && std::env::var_os("ATLAS_NO_FFN_NVFP4_MMQ").is_none();
        if !active {
            return Ok(());
        }
        self.ensure_nvfp4_mmq_weight(
            &self.fp4mmq_gate,
            gpu,
            &self.weights.gate_proj,
            inter,
            h,
            stream,
        )?;
        self.ensure_nvfp4_mmq_weight(
            &self.fp4mmq_up,
            gpu,
            &self.weights.up_proj,
            inter,
            h,
            stream,
        )?;
        let down_mmq = std::env::var_os("ATLAS_NO_FFN_NVFP4_MMQ_DOWN").is_none();
        if down_mmq {
            self.ensure_nvfp4_mmq_weight(
                &self.fp4mmq_down,
                gpu,
                &self.weights.down_proj,
                h,
                inter,
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        // Free the dead transposed copies (prefill for those projections now runs on the
        // MMQ arm; decode reads the non-transposed originals). down_proj_t is freed only
        // when the down A/B gate is on.
        let mut down_t = if down_mmq {
            Some(&mut self.weights.down_proj_t)
        } else {
            None
        };
        let mut freed = 0usize;
        for wt in [&mut self.weights.gate_proj_t, &mut self.weights.up_proj_t]
            .into_iter()
            .chain(down_t.take())
        {
            if let Some(w) = wt.as_ref()
                && !w.weight.is_null()
            {
                gpu.free(w.weight)?;
                gpu.free(w.weight_scale)?;
                freed += 1;
            }
            *wt = None;
        }
        if freed > 0 {
            // Log-once latch (see `atlas_core::scope`). It holds no model-derived
            // value — the message is rebuilt from the arguments every call — so a
            // stale entry cannot produce a wrong answer, only a suppressed duplicate
            // line after a model swap. Scoping it would thread a logging concern
            // through the call path to prevent one repeated INFO line.
            // Latched on the BACKEND (`OpCache::once`), which exists by load
            // time: `finalize_q4k_load` takes the `gpu` it is loading onto. A
            // static meant only the first model in the process reported the
            // decision.
            if gpu.op_cache().once("log:ffn_fp4mmq_freed_twins") {
                tracing::info!(
                    "[atlas] ATLAS_FFN_NVFP4_MMQ: freed gate/up `_t` copies (dead under FP4-MMQ prefill) — block_nvfp4 copies net to ~0 vs NVFP4 baseline"
                );
            }
        }
        Ok(())
    }

    /// Ensure the block_nvfp4 copy of one NVFP4 projection exists (raw repack of the
    /// checkpoint's packed E2M1 `[N, K/2]` + E4M3 `[N, K/16]` scales — zero numerics;
    /// scale2 folded at the SiLU-mul). Cached in `cell` for process lifetime.
    fn ensure_nvfp4_mmq_weight(
        &self,
        cell: &std::sync::OnceLock<Fp4MmqWeight>,
        gpu: &dyn GpuBackend,
        src: &QuantizedWeight,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<Fp4MmqWeight> {
        if let Some(w) = cell.get() {
            return Ok(*w);
        }
        let w = gpu.alloc(ops::nvfp4_mmq_weight_bytes(n, k))?;
        ops::nvfp4_mmq_repack(
            gpu,
            self.nvfp4_repack_k,
            src.weight,
            src.weight_scale,
            w,
            n,
            k,
            stream,
        )?;
        let built = Fp4MmqWeight { w };
        if let Err(dup) = cell.set(built) {
            gpu.synchronize(stream)?;
            let _ = gpu.free(dup.w);
        }
        Ok(*cell.get().expect("fp4mmq weight cell set above"))
    }

    /// Install native block-scaled FP8 dense MLP weights. After this call the
    /// forward paths dispatch `w8a16_gemv` (decode) / `w8a16_gemm` (prefill)
    /// instead of w4a16 NVFP4. Caller must ensure those kernels are present in
    /// the target (they are for the qwen3_5/ornith nvfp4 bundle).
    pub fn set_fp8_weights(&mut self, gate: Fp8Weight, up: Fp8Weight, down: Fp8Weight) {
        self.fp8_weights = Some(DenseFfnWeightsFp8 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }

    /// Install the startup-static LoRA FFN overlay (gate/up/down deltas).
    /// Hard-rejects when FP8/BF16 weight overlays are installed — those
    /// decode branches early-return before the NVFP4 tail where the M1
    /// delta insertions land, so a permissive install would silently skip
    /// deltas. holo is NVFP4, so it is unaffected.
    pub fn set_lora_weights(&mut self, w: ops::lora_delta::LoraFfnWeights) -> Result<()> {
        anyhow::ensure!(
            self.fp8_weights.is_none() && self.bf16_weights.is_none(),
            "LoRA v0 supports only the NVFP4 dense-FFN path (FP8/BF16 weight \
             overlays installed on this layer)"
        );
        // Packed-Q2 has its own gemv/batchm branches that early-return before
        // the NVFP4 tail where the deltas land, exactly like FP8/BF16. Refusing
        // here keeps the invariant the M1 apply relies on: if `self.lora` is
        // Some, EVERY dispatch this layer can take applies it. A silently
        // skipping path is worse than a refused load — it makes the adapter
        // active in prefill and absent in decode, which reads as model
        // weirdness rather than as a missing feature.
        anyhow::ensure!(
            self.q2_weights.is_none(),
            "LoRA v0 supports only the NVFP4 dense-FFN path (packed-Q2 weights \
             installed on this layer)"
        );
        // The decode down delta contracts over silu(gate)*up, which only the
        // split-SiLU path materialises; `forward` pins that path whenever an
        // adapter is installed. Refuse here if the layer cannot take it, so
        // the pin is a guarantee rather than a hope.
        anyhow::ensure!(
            self.activation == FfnActivation::SiLU && self.act_mul.0 != 0 && self.w4a16_gemv.0 != 0,
            "LoRA v0 needs the split-SiLU decode path (SiLU activation + \
             act_mul + w4a16_gemv kernels); this layer resolved activation \
             {:?}, act_mul={}, w4a16_gemv={}",
            self.activation,
            self.act_mul.0,
            self.w4a16_gemv.0,
        );
        self.lora = Some(w);
        Ok(())
    }

    /// M1 gate/up delta: `gate_out += ΔW_gate · x`, `up_out += ΔW_up · x`.
    ///
    /// Call AFTER the gate/up projection and BEFORE `silu_mul` — the deltas
    /// belong to the projections, so they must land while gate/up are still
    /// separate. Both buffers are the arena's dedicated `expert_gate_out` /
    /// `expert_up_out` regions, contiguous with row stride `inter*2`, which is
    /// what `apply_lora_delta`'s contiguity contract requires.
    ///
    /// No-op (and no launches) when the layer carries no adapter, so the
    /// non-LoRA path stays byte-identical.
    fn apply_lora_gate_up(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if ops::lora_delta::lora_no_ffn() {
            return Ok(());
        }
        let Some(ref lw) = self.lora else {
            return Ok(());
        };
        for (pair, base) in [(&lw.gate, gate_out), (&lw.up, up_out)] {
            if let Some(pair) = pair.as_ref() {
                ops::lora_delta::apply_lora_delta(
                    ctx.gpu,
                    &lw.kernels,
                    pair,
                    input,
                    base,
                    m,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// M1 down delta: `output += ΔW_down · act`.
    ///
    /// Call AFTER the down projection. `act` is the SiLU(gate)*up activation —
    /// the same tensor the base down projection contracted over, NOT the
    /// layer input. Every dense path leaves it in the `expert_gate_out`
    /// region (silu_mul writes in place over gate), contiguous at row stride
    /// `inter*2`.
    fn apply_lora_down(
        &self,
        ctx: &ForwardContext,
        act: DevicePtr,
        output: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if ops::lora_delta::lora_no_ffn() {
            return Ok(());
        }
        let Some(ref lw) = self.lora else {
            return Ok(());
        };
        let Some(ref pair) = lw.down else {
            return Ok(());
        };
        ops::lora_delta::apply_lora_delta(
            ctx.gpu,
            &lw.kernels,
            pair,
            act,
            output,
            m,
            ctx.buffers.lora_xa(),
            ctx.buffers.lora_delta(),
            stream,
        )
    }

    /// Install native keep-packed ternary Q2_0 dense MLP weights. After this
    /// call, decode `forward` dispatches `q2_0_gemv` per projection (weights
    /// stay 2-bit resident, no NVFP4 requant) as the highest-priority path.
    /// Caller must ensure the `q2_0_gemv` kernel is present in the target
    /// (checked at forward time; falls through to a clear error otherwise).
    /// Prefill for packed-Q2 is a deferred phase — see `forward_prefill`.
    pub fn set_q2_weights(
        &mut self,
        gate: PackedQ2Weight,
        up: PackedQ2Weight,
        down: PackedQ2Weight,
        gpu: &dyn GpuBackend,
    ) {
        self.q2_weights = Some(DenseFfnWeightsQ2 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
        // Resolved here, not in the constructor: these ship only in
        // GGUF-serving targets and the boot audit fails closed on an
        // unconditional probe everywhere else.
        self.q2_0_mmq_nc_k = super::try_kernel(gpu, "q2_0_mmq", "atlas_q2_0_mmq128_nc");
        self.q2_0_mmq_wc_k = super::try_kernel(gpu, "q2_0_mmq", "atlas_q2_0_mmq128_wc");
    }

    /// Install BF16 dense MLP weights. After this call, the forward paths
    /// dispatch to the BF16 GEMV/GEMM kernels instead of w4a16. The
    /// caller must ensure the BF16 kernels are loaded (see
    /// `dense_gemv_bf16_k` / `dense_gemm_bf16_k` checks). Small-batch
    /// paths reuse `forward_prefill` so they cannot enter NVFP4 kernels
    /// with the null placeholder weights used by BF16-native layers.
    pub fn set_bf16_weights(&mut self, gate: DenseWeight, up: DenseWeight, down: DenseWeight) {
        self.bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }

    /// Ensure the int8 W4A8 copy of one NVFP4 projection weight exists, building
    /// it once via `requant_w_nvfp4_int8` and caching it in `cell`. Reads the
    /// NON-transposed NVFP4 layout (`weight` = packed E2M1 `[N, K/2]`,
    /// `weight_scale` = per-16 E4M3 `[N, K/16]`, `weight_scale_2` = per-tensor
    /// F32) — so it is independent of the `*_proj_t` transposed copies. The
    /// requant launches on `stream`; the subsequent faith2 read is stream-ordered
    /// after it, so no host sync is needed.
    fn ensure_int8_weight(
        &self,
        cell: &std::sync::OnceLock<Int8Weight>,
        gpu: &dyn GpuBackend,
        src: &QuantizedWeight,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<Int8Weight> {
        if let Some(w) = cell.get() {
            return Ok(*w);
        }
        let (nn, kk) = (n as usize, k as usize);
        let w_i8 = gpu.alloc(nn * kk)?; // [N, K] int8
        let w_scale = gpu.alloc(nn * (kk / 32) * 4)?; // [N, K/32] F32
        ops::requant_w_nvfp4_int8(
            gpu,
            self.requant_w_int8_k,
            src.weight,
            src.weight_scale,
            src.weight_scale_2,
            w_i8,
            w_scale,
            n,
            k,
            stream,
        )?;
        let built = Int8Weight { w_i8, w_scale };
        // Lost a race (another thread built first): free our duplicate buffers.
        if let Err(dup) = cell.set(built) {
            let _ = gpu.free(dup.w_i8);
            let _ = gpu.free(dup.w_scale);
        }
        Ok(*cell.get().expect("int8 weight cell set above"))
    }

    /// Lazily materialize a Q4_K FFN weight from the NVFP4 source: dequant NVFP4→bf16
    /// (transient buffer, freed) then quantize bf16→GGML block_q4_K (cached for the
    /// process lifetime). `src` is the non-transposed NVFP4 weight `[n, k]`.
    fn ensure_q4k_weight(
        &self,
        cell: &std::sync::OnceLock<Q4kWeight>,
        gpu: &dyn GpuBackend,
        src: &QuantizedWeight,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<Q4kWeight> {
        if let Some(w) = cell.get() {
            return Ok(*w);
        }
        // transient bf16 [n, k] (freed after quantize); persistent Q4_K bytes.
        let bf16_tmp = gpu.alloc((n as usize) * (k as usize) * 2)?;
        ops::dequant_nvfp4_to_bf16(
            gpu,
            self.dequant_nvfp4_bf16_k,
            src.weight,
            src.weight_scale,
            bf16_tmp,
            src.weight_scale_2,
            n,
            k,
            stream,
        )?;
        let w_q4k = gpu.alloc(ops::q4k_weight_bytes(n, k))?;
        ops::quantize_weight_q4k(gpu, self.q4k_quant_w_k, bf16_tmp, w_q4k, n, k, stream)?;
        // bf16_tmp consumed by the quantize on `stream`; sync before freeing it.
        gpu.synchronize(stream)?;
        let _ = gpu.free(bf16_tmp);
        let built = Q4kWeight { w_q4k };
        if let Err(dup) = cell.set(built) {
            let _ = gpu.free(dup.w_q4k);
        }
        Ok(*cell.get().expect("q4k weight cell set above"))
    }

    /// Single-token decode: 2-3 kernel launches depending on activation.
    /// SiLU: dual GEMV + SiLU-fused down GEMV (2 launches).
    /// GELU: dual GEMV + gelu_mul + down GEMV (3 launches, no fused GELU down kernel).
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Native keep-packed Q2_0 dispatch (highest priority). Per-projection
        // `q2_0_gemv`: BF16 activation × packed 2-bit weight, dequant in the
        // dot-product — weights never expand to BF16/NVFP4. No fused dual/silu
        // kernel yet, so this is gate + up + silu_mul + down (4 launches),
        // mirroring the FP8 non-fused fallback. SiLU only (Ternary-Bonsai is a
        // Qwen-family SwiGLU); GeLU packed-Q2 is a follow-up.
        if let Some(ref q2w) = self.q2_weights {
            if self.q2_0_gemv_k.0 == 0 {
                anyhow::bail!(
                    "q2_0_gemv kernel missing in this target build — packed-Q2 decode \
                     (ATLAS_GGUF_NATIVE_Q2) is unavailable"
                );
            }
            if self.activation != FfnActivation::SiLU {
                anyhow::bail!(
                    "packed-Q2 FFN decode supports SiLU only (got {:?})",
                    self.activation
                );
            }
            let output = ctx.buffers.moe_output();
            ops::q2_0_gemv_vec(
                ctx.gpu,
                self.q2_0_gemv_k,
                input,
                &q2w.gate_proj,
                gate_out,
                stream,
            )?;
            ops::q2_0_gemv_vec(
                ctx.gpu,
                self.q2_0_gemv_k,
                input,
                &q2w.up_proj,
                up_out,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            ops::q2_0_gemv_vec(
                ctx.gpu,
                self.q2_0_gemv_k,
                gate_out,
                &q2w.down_proj,
                output,
                stream,
            )?;
            return Ok(output);
        }

        // FP8 dispatch: prefer the fused FP8 dual-GEMV (gate+up in one launch) +
        // SiLU-fused down GEMV, mirroring the NVFP4 path. Collapses gate+up+
        // silu_mul+down (4 launches) to dual+silu (2). Falls back to the
        // 3-launch per-projection `w8a16_gemv` path when the fused kernels or a
        // non-SiLU activation make the fast path unavailable.
        if let Some(ref fp8w) = self.fp8_weights {
            let output = ctx.buffers.moe_output();
            if self.activation == FfnActivation::SiLU
                && self.w8a16_gemv_dual_k.0 != 0
                && self.w8a16_gemv_silu_input_k.0 != 0
            {
                ops::w8a16_gemv_dual(
                    ctx.gpu,
                    self.w8a16_gemv_dual_k,
                    input,
                    fp8w.gate_proj.weight,
                    fp8w.gate_proj.row_scale,
                    gate_out,
                    fp8w.up_proj.weight,
                    fp8w.up_proj.row_scale,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
                ops::w8a16_gemv_silu_input(
                    ctx.gpu,
                    self.w8a16_gemv_silu_input_k,
                    gate_out,
                    up_out,
                    fp8w.down_proj.weight,
                    fp8w.down_proj.row_scale,
                    output,
                    h,
                    inter,
                    stream,
                )?;
                return Ok(output);
            }
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                input,
                fp8w.gate_proj.weight,
                fp8w.gate_proj.row_scale,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                input,
                fp8w.up_proj.weight,
                fp8w.up_proj.row_scale,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                gate_out,
                fp8w.down_proj.weight,
                fp8w.down_proj.row_scale,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // BF16 dispatch: per-projection GEMV via `dense_gemv_bf16`. We
        // don't have a fused dual-BF16-GEMV kernel today; two sequential
        // launches are still BF16-precision-correct and only ~10% slower
        // than the fused w4a16 path on Gemma-4-31B (the cost is dominated
        // by the bigger BF16 weight reads, not launch overhead).
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // ATLAS_DECODE_FFN_VIA_GEMM=1: route decode's M=1 FFN projections
        // through the SAME transposed-weight GEMM kernels the DFlash verify
        // path uses (`w4a16_prefill_gemm` → w4a16_gemm_t / _t_k64), instead
        // of the dedicated GEMV kernels. Purpose: bit-identical FFN numerics
        // between serial decode and batched verify — the batch-K vs batch-1
        // divergence #218's bisect isolated ("FFN non-associativity") and the
        // root cause of the T=0 spec trajectory flips (2026-07-07 session).
        // Split SiLU staging already matches prefill SiLU numerics (swiglu
        // clamp), so with this arm the whole FFN block is kernel-identical to
        // a verify row. Requires the *_proj_t transposed copies (the NVFP4-MMQ
        // prefill arm FREES them — disable it if the warn below fires).
        // The `OnceLock<bool>` static that lived here is now a field on
        // `layers::ops::ModelLevers` — resolved when the model is built and carried
        // on `ForwardContext`, because a static outlives the model whose flags it
        // encodes.
        if ctx.levers.decode_ffn_via_gemm
            && self.activation == FfnActivation::SiLU
            && self.act_mul.0 != 0
        {
            let wt_alive =
                |w: &Option<QuantizedWeight>| w.as_ref().is_some_and(|w| !w.weight.is_null());
            if wt_alive(&self.weights.gate_proj_t) && wt_alive(&self.weights.up_proj_t) {
                // Log-once latch (see `atlas_core::scope`). It holds no model-derived
                // value — the message is rebuilt from the arguments every call — so a
                // stale entry cannot produce a wrong answer, only a suppressed duplicate
                // line after a model swap. Scoping it would thread a logging concern
                // through the call path to prevent one repeated INFO line.
                if ctx.stats.once("log:decode_ffn_via_gemm") {
                    tracing::info!(
                        "decode FFN via verify GEMM path (ATLAS_DECODE_FFN_VIA_GEMM=1): \
                         gate/up/down through w4a16_prefill_gemm at M=1"
                    );
                }
                self.w4a16_prefill_gemm(
                    ctx,
                    &self.weights.gate_proj,
                    self.weights.gate_proj_t.as_ref(),
                    input,
                    gate_out,
                    1,
                    inter,
                    h,
                    stream,
                )?;
                self.w4a16_prefill_gemm(
                    ctx,
                    &self.weights.up_proj,
                    self.weights.up_proj_t.as_ref(),
                    input,
                    up_out,
                    1,
                    inter,
                    h,
                    stream,
                )?;
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    inter,
                    stream,
                )?;
                let output = ctx.buffers.moe_output();
                self.w4a16_prefill_gemm(
                    ctx,
                    &self.weights.down_proj,
                    self.weights.down_proj_t.as_ref(),
                    gate_out,
                    output,
                    1,
                    h,
                    inter,
                    stream,
                )?;
                return Ok(output);
            }
            // Log-once latch (see `atlas_core::scope`). It holds no model-derived
            // value — the message is rebuilt from the arguments every call — so a
            // stale entry cannot produce a wrong answer, only a suppressed duplicate
            // line after a model swap. Scoping it would thread a logging concern
            // through the call path to prevent one repeated INFO line.
            if ctx.stats.once("log:decode_ffn_no_twins") {
                tracing::warn!(
                    "ATLAS_DECODE_FFN_VIA_GEMM=1 requested but transposed FFN copies \
                     are freed/absent (NVFP4-MMQ prefill arm?) — falling back to GEMV; \
                     the unification experiment is NOT active"
                );
            }
        }

        // Fused gate_proj + up_proj: [1, H] → [1, inter] × 2.
        // Single-warp variant (lossless) when the lever is on and the kernel
        // resolved; otherwise the 64-thread kernel. Dual and silu-input SW
        // are independent — missing silu_input_sw must not skip dual_sw on
        // the default split-SiLU path.
        let use_dual_sw = ops::use_gemv_sw(ctx.levers.gemv_sw, self.w4a16_gemv_dual_sw);
        let use_silu_sw = ops::use_gemv_sw(ctx.levers.gemv_sw, self.w4a16_gemv_silu_input_sw);
        if use_dual_sw {
            ops::w4a16_gemv_dual_sw(
                ctx.gpu,
                self.w4a16_gemv_dual_sw,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
        }

        let output = ctx.buffers.moe_output();
        // Split SiLU+down (DEFAULT; kill-switch ATLAS_NO_DECODE_SPLIT_SILU): the fused
        // silu_input kernel recomputes the SiLU transcendentals per OUTPUT ROW (N/4
        // blocks × redundant __expf) and measures COMPUTE-bound — ncu: SM 57% vs
        // memory 23%, 186 GB/s vs the dual GEMV's 266. Staging silu(gate)*up once
        // (one elementwise launch, CUDA graphs amortize it) lets the down GEMV run
        // memory-bound like the dual. Also aligns decode with the prefill SiLU
        // numerics (swiglu clamp), which the fused kernel lacked.
        //
        // An installed LoRA adapter PINS this path. The fused `silu_input`
        // alternative never materialises silu(gate)*up — it consumes gate and
        // up straight into the down GEMV — and the down delta has to contract
        // over exactly that activation. Reproducing it into `lora_hact` just
        // to feed the delta would compute the SiLU twice for a path that is
        // already the default and already the numerically preferred one, so
        // the adapter pins it instead. `set_lora_weights` refuses an adapter
        // when this path is unavailable on the layer, which makes
        // `lora.is_some()` imply the three conditions above.
        let split_silu = self.activation == FfnActivation::SiLU
            && self.act_mul.0 != 0
            && self.w4a16_gemv.0 != 0
            && (std::env::var_os("ATLAS_NO_DECODE_SPLIT_SILU").is_none() || self.lora.is_some());
        if split_silu {
            self.apply_lora_gate_up(ctx, input, gate_out, up_out, 1, stream)?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                self.w4a16_gemv_sw,
                ctx.levers.gemv_sw,
                gate_out,
                &self.weights.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            self.apply_lora_down(ctx, gate_out, output, 1, stream)?;
            return Ok(output);
        }
        debug_assert!(
            self.lora.is_none(),
            "LoRA installed but decode took the fused silu_input path, which \
             never materialises the activation the down delta contracts over; \
             set_lora_weights is supposed to make this unreachable"
        );
        match self.activation {
            FfnActivation::SiLU => {
                // Fused SiLU(gate)*up + down_proj: [1, inter] → [1, H]
                if use_silu_sw {
                    ops::w4a16_gemv_silu_input_sw(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input_sw,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemv_silu_input(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                }
            }
            FfnActivation::GeLU => {
                // GELU(gate)*up → gate_out, then down_proj GEMV
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    inter,
                    stream,
                )?;
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv,
                    self.w4a16_gemv_sw,
                    ctx.levers.gemv_sw,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    h,
                    inter,
                    stream,
                )?;
            }
        }

        Ok(output)
    }

    /// Packed-Q2 batched decode FFN for `m` concurrent rows (`m >= 2`). Mirrors
    /// the single-token `forward` packed-Q2 arm — per-projection keep-packed
    /// `q2_0_gemv_vec_batchm` (BF16 `[m,·]` activation × 2-bit weight, dequant in
    /// the dot-product, no BF16/NVFP4 expansion), SiLU-mul between gate/up, then
    /// down — but with `m` activation rows staged per weight read. This is the
    /// correctness path for concurrent decode (C>=2): the NVFP4 `forward_k2/k3`
    /// GEMVs read the NULL NVFP4 fallback weights, so packed-Q2 must route here.
    /// The wrapper chunks internally for `m > 8`. SiLU only (Ternary-Bonsai is a
    /// SwiGLU); output lands in `moe_output` as `[m, h]` row-major.
    fn forward_km_q2(
        &self,
        q2w: &DenseFfnWeightsQ2,
        input: DevicePtr,
        ctx: &ForwardContext,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if self.q2_0_gemv_batchm_k.0 == 0 {
            anyhow::bail!(
                "q2_0_gemv_vec_batchm kernel missing in this target build — packed-Q2 \
                 batched decode (ATLAS_GGUF_NATIVE_Q2, C>=2) is unavailable"
            );
        }
        if self.activation != FfnActivation::SiLU {
            anyhow::bail!(
                "packed-Q2 FFN batched decode supports SiLU only (got {:?})",
                self.activation
            );
        }
        let inter = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let batchm = |w: &PackedQ2Weight, inp: DevicePtr, out: DevicePtr| -> Result<()> {
            ops::q2_0_gemv_vec_batchm(ctx.gpu, self.q2_0_gemv_batchm_k, inp, w, out, m, stream)
        };
        batchm(&q2w.gate_proj, input, gate_out)?;
        batchm(&q2w.up_proj, input, up_out)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        batchm(&q2w.down_proj, gate_out, output)?;
        Ok(())
    }

    /// K=2 speculative: batched GEMV for 2 tokens.
    /// 3 launches: dual batch2 (gate+up) + silu_mul + batch2 (down).
    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        // Packed-Q2: NVFP4 fallback weights are NULL, so the NVFP4 batch2 GEMVs
        // below would fault. Route to the keep-packed batchm FFN (m=2).
        if let Some(ref q2w) = self.q2_weights {
            return self.forward_km_q2(q2w, input, ctx, 2, stream);
        }
        if native_small_batch_uses_prefill(self.bf16_weights.is_some(), self.fp8_weights.is_some())
        {
            return self.forward_prefill(input, 2, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 2 tokens
        ops::w4a16_gemv_dual_batch2(
            ctx.gpu,
            self.w4a16_gemv_dual_batch2,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        self.apply_lora_gate_up(ctx, input, gate_out, up_out, 2, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            2 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch2(
            ctx.gpu,
            self.w4a16_gemv_batch2,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;
        self.apply_lora_down(ctx, gate_out, output, 2, stream)?;

        Ok(())
    }

    /// K=3 speculative: batched GEMV for 3 tokens.
    /// 3 launches: dual batch3 (gate+up) + silu_mul + batch3 (down).
    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        // Packed-Q2: route to the keep-packed batchm FFN (m=3); NVFP4 weights null.
        if let Some(ref q2w) = self.q2_weights {
            return self.forward_km_q2(q2w, input, ctx, 3, stream);
        }
        if native_small_batch_uses_prefill(self.bf16_weights.is_some(), self.fp8_weights.is_some())
        {
            return self.forward_prefill(input, 3, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 3 tokens
        ops::w4a16_gemv_dual_batch3(
            ctx.gpu,
            self.w4a16_gemv_dual_batch3,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        self.apply_lora_gate_up(ctx, input, gate_out, up_out, 3, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            3 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch3(
            ctx.gpu,
            self.w4a16_gemv_batch3,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;
        self.apply_lora_down(ctx, gate_out, output, 3, stream)?;

        Ok(())
    }

    /// Batchm-GEMV kernel for `m` verify rows: the narrowest resolved tier in
    /// `w4a16_gemv_batch{4,5,6,7,8}` that covers `m`. 0-handle when out of
    /// range or absent. See `layers::w4a16_gemv_tiers` for the decision and
    /// the `ATLAS_NO_GEMV_EXACT_M_TIERS=1` kill switch.
    fn batchm_kernel(&self, m: u32) -> KernelHandle {
        self.w4a16_batchm.kernel(m)
    }

    /// Whether the M-row batched-GEMV verify path is available for `m` rows
    /// (batchm kernel present AND NVFP4 weights loaded — the batchm GEMV
    /// reads the non-transposed NVFP4 layout).
    pub fn can_forward_km(&self, m: u32) -> bool {
        self.batchm_kernel(m).0 != 0 && !self.weights.gate_proj.weight.is_null()
    }

    /// K=m (m<=8) speculative verify: batched GEMV for m tokens.
    /// 4 launches: batchm gate + batchm up + silu_mul + batchm down — each
    /// projection weight is read ONCE for all m rows at near-peak stream
    /// bandwidth. nsys (2026-07-18, M=4): the `forward_prefill` MMQ arm this
    /// replaces for the K=4 verify cost 54.8 ms/step across the 64-layer
    /// dense FFN stack (~156 GB/s effective at M=4); the batch GEMV family
    /// measures ~290 GB/s on the same shapes (w8a16_gemv_batch4 sibling),
    /// putting this path at the ~31 ms weight-traffic floor. m=5..8 uses
    /// `w4a16_gemv_batch8` (batchm_bench: same weight-streaming bandwidth,
    /// removing the M>4 tile-GEMM cliff for chain-verify K=5..8).
    pub fn forward_km(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let kh = self.batchm_kernel(m);

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kh,
            input,
            &self.weights.gate_proj,
            gate_out,
            m,
            inter,
            h,
            stream,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kh,
            input,
            &self.weights.up_proj,
            up_out,
            m,
            inter,
            h,
            stream,
        )?;
        self.apply_lora_gate_up(ctx, input, gate_out, up_out, m, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kh,
            gate_out,
            &self.weights.down_proj,
            output,
            m,
            h,
            inter,
            stream,
        )?;
        self.apply_lora_down(ctx, gate_out, output, m, stream)?;

        Ok(())
    }

    /// N-token prefill: GEMM for all projections.
    /// W4A16 prefill/verify GEMM dispatch, routed by (M, K) per
    /// w4a16_m17_bench measurements on GB10:
    ///   - M<=64 (DFlash verify M=17): the M64-tile `w4a16_gemm_t` beats the
    ///     M128-tile kernels (283 vs 324us on gate/up — 87% of an M128 tile
    ///     is padding at M=17), and `w4a16_gemm_t_k64` wins deep-K down_proj
    ///     (554 vs 810us at K=17408, where N/128 CTAs can't fill the GPU and
    ///     the halved K-loop matters).
    ///   - M>64 (real prefill): v2 (8-warp) > t_m128 (4-warp), unchanged.
    ///   - No transposed copy: base `w4a16_gemm` (9-12x the bandwidth floor —
    ///     last resort).
    ///
    /// Kill-switch: ATLAS_FFN_SMALLM=0 restores the m128-only dispatch for A/B.
    #[allow(clippy::too_many_arguments)]
    fn w4a16_prefill_gemm(
        &self,
        ctx: &ForwardContext,
        w: &QuantizedWeight,
        wt: Option<&QuantizedWeight>,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        // The `OnceLock<bool>` static that lived here is now a field on
        // `layers::ops::ModelLevers` — resolved when the model is built and carried
        // on `ForwardContext`, because a static outlives the model whose flags it
        // encodes.
        if let Some(wt) = wt {
            if m <= 64 && k.is_multiple_of(32) && ctx.levers.ffn_small_m {
                if k >= crate::layers::w4a16_k64_min_k()
                    && k.is_multiple_of(64)
                    && self.w4a16_gemm_t_k64_k.0 != 0
                {
                    return ops::w4a16_gemm_n128(
                        ctx.gpu,
                        self.w4a16_gemm_t_k64_k,
                        input,
                        wt,
                        output,
                        m,
                        n,
                        k,
                        stream,
                    );
                }
                if self.w4a16_gemm_t_k.0 != 0 {
                    return ops::w4a16_gemm_n128(
                        ctx.gpu,
                        self.w4a16_gemm_t_k,
                        input,
                        wt,
                        output,
                        m,
                        n,
                        k,
                        stream,
                    );
                }
            }
            if self.w4a16_gemm_t_m128_v2_k.0 != 0 {
                return ops::w4a16_gemm_n128_m128_v2(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_v2_k,
                    input,
                    wt,
                    output,
                    m,
                    n,
                    k,
                    stream,
                );
            }
            if self.w4a16_gemm_t_m128_k.0 != 0 {
                return ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    input,
                    wt,
                    output,
                    m,
                    n,
                    k,
                    stream,
                );
            }
        }
        ops::w4a16_gemm(ctx.gpu, self.w4a16_gemm, input, w, output, m, n, k, stream)
    }

    /// Timed wrapper around the dense-FFN prefill.
    ///
    /// ★ THIS PATH HAD NO TIMERS AT ALL, and that hid the largest unexplained
    /// number on the board. Profiling nvidia/Gemma-4-31B-IT-NVFP4 at a
    /// 4096-token prompt: wall 28,180 ms, while EVERY profiled phase across
    /// `ATTN prefill [...]` and `MoE prefill [...]` summed to 3,269.8 ms. 88% of
    /// the prefill was invisible — not attributed to something slow, simply not
    /// instrumented. `forward_prefill` dispatches ~20 quantization arms and none
    /// of them reported elapsed time; only one-shot "which arm was chosen" INFO
    /// lines existed.
    ///
    /// One coarse timer first, deliberately: it answers whether the missing time
    /// is here at all before anyone threads timers through twenty arms. Same
    /// `<AREA> prefill [phase] N=<n>: <us>µs` shape the attention and MoE paths
    /// already emit, so the existing log-summing one-liners pick it up unchanged.
    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !ctx.profile {
            return self.forward_prefill_inner(input, num_tokens, ctx, stream);
        }
        let t0 = std::time::Instant::now();
        let r = self.forward_prefill_inner(input, num_tokens, ctx, stream);
        // Sync so the figure is the kernel's, not the launch queue's — the
        // attention and MoE timers do the same under `ctx.profile`.
        ctx.gpu.synchronize(stream)?;
        tracing::info!(
            "  FFN prefill [dense_total] N={}: {}µs",
            num_tokens,
            t0.elapsed().as_micros()
        );
        r
    }

    fn forward_prefill_inner(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let m = num_tokens as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Native keep-packed Q2_0 prefill (Tier-1): the resident weight stays
        // 2-bit, but prefill has no packed-MMQ kernel yet (that's Tier-2). So we
        // dequant each projection into a TRANSIENT BF16 scratch `[N, K]` via the
        // load-time `dequant_q2_0_gn_to_bf16` kernel, run the normal BF16
        // prefill GEMM (tensor-core when present), then free the scratch. Only a
        // per-matmul scratch is BF16 — the WeightStore blocks stay 2-bit. Decode
        // still uses the native `q2_0_gemv` (no dequant). SiLU only.
        if let Some(ref q2w) = self.q2_weights {
            if self.activation != FfnActivation::SiLU {
                anyhow::bail!(
                    "packed-Q2 FFN prefill supports SiLU only (got {:?})",
                    self.activation
                );
            }

            // Tier-2 native MMQ prefill (ATLAS_GGUF_NATIVE_Q2_MMQ=1): quantize the
            // activation to q8_1 ONCE per projection-input (gate/up share `input`;
            // down re-quantizes `gate_out`), then run the packed 2-bit MMQ GEMM —
            // no BF16 weight dequant, no shared `q2_dequant_scratch`. Requires the
            // MMQ kernel + the shared q8_1 quantizer + group-128 weights.
            let q2_mmq = self.q2_0_mmq_nc_k.0 != 0
                && self.q4k_quant_act_k.0 != 0
                && ops::native_q2_mmq_enabled()
                && q2w.gate_proj.group == 128
                && q2w.up_proj.group == 128
                && q2w.down_proj.group == 128;
            if q2_mmq {
                static Q2MMQ_LOG: std::sync::Once = std::sync::Once::new();
                Q2MMQ_LOG.call_once(|| {
                    eprintln!(
                        "[atlas] ATLAS_GGUF_NATIVE_Q2_MMQ=1: dense-FFN prefill via native packed Q2_0 MMQ (W2A8, keep-packed)"
                    );
                });
                let a_q8 = ctx.buffers.q2_act_q8();
                let mmq = |w: &PackedQ2Weight, out: DevicePtr| -> Result<()> {
                    ops::q2_0_mmq_gemm(
                        ctx.gpu,
                        self.q2_0_mmq_nc_k,
                        self.q2_0_mmq_wc_k,
                        a_q8,
                        w.weight,
                        out,
                        m,
                        w.n,
                        w.k,
                        stream,
                    )
                };
                // gate/up: quantize `input` [m,h] once, feed both.
                ops::quantize_act_q8_1(ctx.gpu, self.q4k_quant_act_k, input, a_q8, m, h, stream)?;
                mmq(&q2w.gate_proj, gate_out)?;
                mmq(&q2w.up_proj, up_out)?;
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
                // down: quantize `gate_out` [m,inter] (same-stream after silu_mul).
                let output = ctx.buffers.moe_output();
                ops::quantize_act_q8_1(
                    ctx.gpu,
                    self.q4k_quant_act_k,
                    gate_out,
                    a_q8,
                    m,
                    inter,
                    stream,
                )?;
                mmq(&q2w.down_proj, output)?;
                return Ok(());
            }

            // Transient-dequant stopgap (Tier-1): requires the load-time dequant kernel.
            if self.dequant_q2_0_gn_k.0 == 0 {
                anyhow::bail!(
                    "dequant_q2_0_gn_to_bf16 kernel missing in this target build — \
                     packed-Q2 (ATLAS_GGUF_NATIVE_Q2) prefill is unavailable"
                );
            }
            let tc = self.dense_gemm_tc_k.0 != 0;
            // Dequant one packed-Q2 projection into the PERSISTENT arena BF16
            // scratch `[n, k]`, run the BF16 GEMM (A=`in` [m,k] → out [m,n]). No
            // per-matmul alloc/sync/free: the arena buffer is sized to the
            // largest packed projection and reused. Gate → up → down run
            // sequentially on `stream`, so each GEMM consumes the scratch before
            // the next projection's dequant overwrites it (same-stream order).
            let scratch = ctx.buffers.q2_dequant_scratch();
            let q2_gemm = |w: &PackedQ2Weight, input: DevicePtr, out: DevicePtr| -> Result<()> {
                let (n, k) = (w.n, w.k);
                debug_assert!(
                    (n as usize) * (k as usize) * 2 <= ctx.buffers.q2_dequant_scratch_bytes(),
                    "packed-Q2 FFN dequant scratch too small for [{n},{k}] BF16"
                );
                ops::dequant_q2_0_gn_to_bf16(
                    ctx.gpu,
                    self.dequant_q2_0_gn_k,
                    w.weight,
                    scratch,
                    n,
                    k,
                    w.group as u32,
                    stream,
                )?;
                let dw = DenseWeight { weight: scratch };
                if tc {
                    ops::dense_gemm_tc(
                        ctx.gpu,
                        self.dense_gemm_tc_k,
                        input,
                        &dw,
                        out,
                        m,
                        n,
                        k,
                        stream,
                    )?;
                } else {
                    ops::dense_gemm(
                        ctx.gpu,
                        self.dense_gemm_bf16_k,
                        input,
                        &dw,
                        out,
                        m,
                        n,
                        k,
                        stream,
                    )?;
                }
                Ok(())
            };
            q2_gemm(&q2w.gate_proj, input, gate_out)?;
            q2_gemm(&q2w.up_proj, input, up_out)?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            q2_gemm(&q2w.down_proj, gate_out, output)?;
            return Ok(());
        }

        // FP8 prefill dispatch: per-projection block-scaled E4M3 weight × BF16
        // act. Prefer the fast transposed `w8a16_gemm_t_m128` (128x128 / 8-warp /
        // two-level FP32 fold) when a transposed FP8 weight copy is available;
        // fall back to the non-transposed `w8a16_gemm`. `DenseFfnWeightsFp8`
        // currently stores only non-transposed weights, so the fallback is taken
        // here today — the m128 preference engages once a `*_proj_t` FP8 copy is
        // installed (the kernel + handle are wired and ship via common/).
        if let Some(ref fp8w) = self.fp8_weights {
            // helper: transposed m128 when a B_t copy + handle are present, else
            // non-transposed w8a16_gemm.
            macro_rules! w8_gemm {
                ($w:expr, $wt:expr, $in:expr, $out:expr, $n:expr, $k:expr) => {
                    match $wt {
                        Some(wt) if self.w8a16_gemm_t_m128_k.0 != 0 => {
                            let wt: Fp8WeightTransposed = wt;
                            ops::w8a16_gemm_n128_m128(
                                ctx.gpu,
                                self.w8a16_gemm_t_m128_k,
                                $in,
                                wt.weight_t,
                                wt.scale_t,
                                $out,
                                m,
                                $n,
                                $k,
                                stream,
                            )?
                        }
                        _ => ops::w8a16_gemm(
                            ctx.gpu,
                            self.w8a16_gemm_k,
                            $in,
                            $w.weight,
                            $w.row_scale,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?,
                    }
                };
            }
            let gate_t: Option<Fp8WeightTransposed> = None;
            let up_t: Option<Fp8WeightTransposed> = None;
            let down_t: Option<Fp8WeightTransposed> = None;
            w8_gemm!(fp8w.gate_proj, gate_t, input, gate_out, inter, h);
            w8_gemm!(fp8w.up_proj, up_t, input, up_out, inter, h);
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            w8_gemm!(fp8w.down_proj, down_t, gate_out, output, h, inter);
            return Ok(());
        }

        // BF16 prefill dispatch. Prefer the tensor-core m16n8k16 MMA kernel
        // (`dense_gemm_tc`, 3-5x+ over scalar) — the scalar `dense_gemm_bf16`
        // was the flat ~155 tok/s prefill bottleneck on Qwen3.6-27B dense
        // NVFP4 (FFN = ~83% of prefill). Falls back to scalar if the TC
        // kernel isn't loaded for this target. Decode (gemv, M=1) is a
        // separate path, so TPOT is unaffected; BF16 MMA preserves coherence.
        if let Some(ref bf16w) = self.bf16_weights {
            let tc = self.dense_gemm_tc_k.0 != 0;
            // helper: cuBLASLt when enabled (the big win at prefill M), else the
            // tensor-core MMA kernel, else scalar. dense_gemm_tc is ~1.4 TFLOP/s
            // on the large dense-FFN shapes (e.g. Laguna layer-0 gate/up/down at
            // N=12288/3072, K=3072) — nsys measured its 3 launches at ~100 ms
            // EACH = 33% of the whole C=1 prefill. cuBLASLt runs the identical
            // BF16×BF16→FP32 GEMM at 90+ TFLOP/s (~65× faster), the same path
            // q/k/v/o and the head-gate already use. Gated on ATLAS_CUBLAS_GEMM.
            macro_rules! ffn_gemm {
                ($a:expr, $b:expr, $c:expr, $n:expr, $k:expr) => {
                    if ctx.dispatch.cublas_gemm {
                        ops::cublas_bf16_proj_dense($a, $b.weight, $c, m, $n, $k, stream)?;
                    } else if tc {
                        ops::dense_gemm_tc(
                            ctx.gpu,
                            self.dense_gemm_tc_k,
                            $a,
                            $b,
                            $c,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    } else {
                        ops::dense_gemm(
                            ctx.gpu,
                            self.dense_gemm_bf16_k,
                            $a,
                            $b,
                            $c,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    }
                };
            }
            ffn_gemm!(input, &bf16w.gate_proj, gate_out, inter, h);
            ffn_gemm!(input, &bf16w.up_proj, up_out, inter, h);
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ffn_gemm!(gate_out, &bf16w.down_proj, output, h, inter);
            return Ok(());
        }

        // Prefill: prefer the 128x128 cp.async-pipelined `w4a16_gemm_t_m128`
        // (the kernel attention/SSM use) over the M64xN64 base `w4a16_gemm`
        // (~10 TFLOPS, the flat ~155 tok/s bottleneck). That kernel needs the
        // TRANSPOSED weight layout, so we use the `*_proj_t` copies built at
        // load (decode keeps the non-transposed weights via gemv → TPOT/
        // coherence unaffected). Falls back to base when no transposed copy /
        // kernel is present.
        // LOSSLESS prefill opt-in: when ATLAS_BF16_TC_PREFILL is set AND the
        // BF16 128x128 kernel is present, route prefill GEMMs through the
        // bit-equivalent BF16 tensor-core path instead of the default FP8-E4M3
        // `t_m128`. The FP8 crush is fast but perturbs generation (measured
        // length-truncations / accuracy risk on Qwen3.6-27B); the BF16 variant
        // keeps the same 128x128 cp.async speed at base-kernel precision.
        // Unset (default) → every arm below is byte-for-byte the prior behavior
        // (PCND: explicit opt-in, no silent default change). Read once per call.
        // Env read only here; the usable gate (`bf16_tc_prefill`) is derived
        // below AFTER v1/v2 selection, from the handle actually launched.
        // Gating on v1's handle while dispatching v2 admitted launches of a
        // kernel this target may not carry.
        let bf16_tc_env = std::env::var_os("ATLAS_BF16_TC_PREFILL").is_some();
        // FP8 M64 fast-prefill opt-in: route prefill GEMMs through the m16n8k32
        // e4m3 M64 kernel (~1.47x vs v2 BF16, smem-relieved). Lossy (cosine 0.9997)
        // → highest priority when set, so it overrides the BF16/FP8 t_m128 arms.
        // PCND: explicit opt-in, default off = byte-for-byte prior behavior.
        let fp8_m64_prefill =
            self.w4a16_gemm_t_k.0 != 0 && std::env::var_os("ATLAS_FP8_M64_PREFILL").is_some();
        // int8 W4A8 fast-prefill opt-in (ATLAS_INT8_PREFILL): route prefill GEMMs
        // through the validated requant→`int8_gemm_faith2` pipeline (cosine
        // 0.999978 vs the host full-precision dequant GEMM). HIGHEST priority when
        // set, so it overrides every other prefill arm. Needs both operands int8:
        // the NVFP4 weights are requanted to int8 once (cached, see
        // `ensure_int8_weight`) and the BF16 activations are requanted every call
        // into the shared scratch (`ensure_int8_scratch`). LOSSY (perf gate, not
        // bit-identical) — the _2.5h IoU gate is the final arbiter.
        // PCND: explicit opt-in, default off = byte-for-byte prior behavior; the
        // arm is a no-op (and no buffers are built) unless the kernels are loaded.
        let int8_prefill =
            self.int8_faith2_k.0 != 0 && std::env::var_os("ATLAS_INT8_PREFILL").is_some();
        if int8_prefill {
            // Log-once latch (see `atlas_core::scope`). It holds no model-derived
            // value — the message is rebuilt from the arguments every call — so a
            // stale entry cannot produce a wrong answer, only a suppressed duplicate
            // line after a model swap. Scoping it would thread a logging concern
            // through the call path to prevent one repeated INFO line.
            if ctx.stats.once("log:ffn_int8_prefill") {
                tracing::info!(
                    "[atlas] ATLAS_INT8_PREFILL=1: dense-FFN prefill via int8_gemm_faith2 (W4A8 requant→int8 MMA, lossy ~0.99998 cosine)"
                );
            }
        }
        // NVFP4 W4A4 MMQ prefill (ATLAS_FFN_NVFP4_MMQ) — vendored llama Blackwell
        // block-scale FP4 MMA, gate/up ONLY (hybrid: down stays on the default t_m128
        // path — SiLU(gate)*up is heavy-tailed and accuracy-critical). SiLU models only
        // (the scale2 fold lives in the scaled SiLU-mul). Mutually exclusive with
        // ATLAS_FFN_MMQ (both use the shared ffn_act_q8 scratch); this arm wins.
        //
        // An installed LoRA adapter turns this arm OFF. The MMQ path leaves
        // gate_out/up_out holding UNSCALED products and folds each
        // projection's `weight_scale_2` later, inside the SiLU-mul kernel. A
        // LoRA delta is a true-valued quantity, so adding it to those buffers
        // would put it through a scale2 multiply that does not belong to it —
        // silently wrong output rather than a failure. Folding scale2 into the
        // delta instead would mean reproducing the quant layout's arithmetic
        // in the adapter path, which is a much larger commitment than the
        // ~8% prefill this arm is worth (measured 831 vs 767 tok/s at 8K).
        // Correctness first; making LoRA and MMQ coexist is its own change.
        let fp4mmq_prefill = self.nvfp4_mmq_nc_k.0 != 0
            && self.nvfp4_quant_act_k.0 != 0
            && self.nvfp4_silu_scaled_k.0 != 0
            && matches!(self.activation, FfnActivation::SiLU)
            && self.lora.is_none()
            && std::env::var_os("ATLAS_NO_FFN_NVFP4_MMQ").is_none();
        if fp4mmq_prefill {
            // Log-once latch (see `atlas_core::scope`). It holds no model-derived
            // value — the message is rebuilt from the arguments every call — so a
            // stale entry cannot produce a wrong answer, only a suppressed duplicate
            // line after a model swap. Scoping it would thread a logging concern
            // through the call path to prevent one repeated INFO line.
            if ctx.stats.once("log:ffn_fp4_mmq_prefill") {
                tracing::info!(
                    "[atlas] ATLAS_FFN_NVFP4_MMQ=1: dense-FFN gate/up prefill via vendored llama NVFP4 W4A4 MMQ (block-scale FP4 MMA, ~80 TFLOP/s vs t_m128 ~51)"
                );
            }
        }
        // Down-projection MMQ arm (DEFAULT ON; kill-switch ATLAS_NO_FFN_NVFP4_MMQ_DOWN=1): route down through
        // the same MMQ arm (t_m128 runs the narrow-N down at only ~34 TFLOP/s in-model).
        // Accuracy note: down W4A4 cosine 0.9961 (random) — better than the previously
        // coherence-validated all-W4A4 config (0.991) — but still the heavy-tailed
        // projection, so it stays a SEPARATE opt-in gate.
        let fp4mmq_down = fp4mmq_prefill
            && self.nvfp4_scale_k.0 != 0
            && std::env::var_os("ATLAS_NO_FFN_NVFP4_MMQ_DOWN").is_none();
        // HYBRID: route the accuracy-critical down_proj OFF Q4_K onto the near-lossless faith2
        // NVFP4 path (W4A8 requant, cos 0.99998). down=SiLU(gate)*up is heavy-tailed; Q4_K
        // superblock scaling clips it (BFCL `multiple` -4.0%; llama promotes only down→Q6_K for
        // this reason). gate/up stay on Q4_K. Default ON when MMQ active; ATLAS_FFN_MMQ_DOWN_Q4K=1
        // = lossy all-Q4_K (A/B only). Defined here (self-fields+env, no q4k_prefill var dep) so the
        // int8 scratch below can size for the hybrid down.
        let down_faith2 = self.q4k_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && self.q4k_quant_w_k.0 != 0
            && self.dequant_nvfp4_bf16_k.0 != 0
            && self.int8_faith2_k.0 != 0
            && self.requant_a_int8_k.0 != 0
            && !fp4mmq_prefill
            && std::env::var_os("ATLAS_FFN_MMQ").is_some()
            && std::env::var_os("ATLAS_FFN_MMQ_DOWN_Q4K").is_none();
        // Pre-allocate (or reuse) the activation-requant scratch once per call,
        // sized to the largest projection K (= max(h, inter)) so the per-GEMM
        // arms never trigger a mid-call grow/sync. NULL when the int8 path is off.
        // Shared, arena-owned activation-requant scratch (sized once for
        // max_batch_tokens × max(h, inter) in BufferSizes::from_config). Replaces
        // the former per-DenseFfnLayer grow-on-demand allocator that leaked
        // ~286MB × 64 layers on the MMQ prefill path.
        let (int8_a_i8, int8_a_scale) = if int8_prefill || down_faith2 {
            (ctx.buffers.ffn_act_a(), ctx.buffers.ffn_act_scale())
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        // W4A4 native-FP4 prefill (ATLAS_FP4_PREFILL) — HIGHEST priority. NVFP4 weights
        // used directly (no requant); BF16 activations quantized to NVFP4 each GEMM into
        // the shared scratch. Native FP4 tensor cores (sm_121a). Lossy (cos ~0.99 vs fp32).
        let fp4_prefill = self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && std::env::var_os("ATLAS_FP4_PREFILL").is_some();
        if fp4_prefill {
            // Log-once latch (see `atlas_core::scope`). It holds no model-derived
            // value — the message is rebuilt from the arguments every call — so a
            // stale entry cannot produce a wrong answer, only a suppressed duplicate
            // line after a model swap. Scoping it would thread a logging concern
            // through the call path to prevent one repeated INFO line.
            if ctx.stats.once("log:ffn_fp4_prefill") {
                tracing::info!(
                    "[atlas] ATLAS_FP4_PREFILL=1: dense-FFN prefill via w4a4_gemm (native FP4 MMA sm_121a, W4A4)"
                );
            }
        }
        // NVFP4 packed [m,K/2] + scale [m,K/16] both fit within the shared int8
        // buffers (a_i8 [m,K] ⊇ packed; a_scale [m,(K/32)*4] ⊇ scale). FP4-prefill
        // is a standalone A/B flag, never co-active with the int8/Q4_K down path.
        let (nvfp4_a_packed, nvfp4_a_scale) = if fp4_prefill {
            (ctx.buffers.ffn_act_a(), ctx.buffers.ffn_act_scale())
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        // Q4_K MMQ prefill (ATLAS_FFN_MMQ) — vendored llama Q4_K W4A8 GEMM. Highest priority
        // when enabled. Lossy (Q4_K weight format ≠ NVFP4); gate via BFCL before relying on it.
        let q4k_prefill = self.q4k_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && self.q4k_quant_w_k.0 != 0
            && self.dequant_nvfp4_bf16_k.0 != 0
            && !fp4mmq_prefill
            && std::env::var_os("ATLAS_FFN_MMQ").is_some();
        if q4k_prefill {
            // Log-once latch (see `atlas_core::scope`). It holds no model-derived
            // value — the message is rebuilt from the arguments every call — so a
            // stale entry cannot produce a wrong answer, only a suppressed duplicate
            // line after a model swap. Scoping it would thread a logging concern
            // through the call path to prevent one repeated INFO line.
            if ctx.stats.once("log:ffn_q4k_prefill") {
                tracing::info!(
                    "[atlas] ATLAS_FFN_MMQ=1: dense-FFN prefill via vendored llama Q4_K MMQ (W4A8, +25%/+10% gate·down vs faith2)"
                );
            }
        }
        let q4k_a = if q4k_prefill {
            ctx.buffers.ffn_act_q8()
        } else {
            DevicePtr::NULL
        };
        // FP4-MMQ y scratch: block_fp4_mmq activations, in the SAME shared arena buffer
        // (fp4_act_scratch_bytes ≤ q8_1_scratch_bytes; mutually exclusive with q4k_prefill).
        let fp4_y = if fp4mmq_prefill {
            ctx.buffers.ffn_act_q8()
        } else {
            DevicePtr::NULL
        };
        // A/B escape hatch (benchmark only): force the proven v1 BF16 kernel even
        // when v2 is loaded, so v1-vs-v2 prefill TTFT can be compared in one
        // binary. Default unset → prefer v2 (the faster, bit-identical variant).
        let use_v2 = self.w4a16_gemm_t_m128_bf16_v2_k.0 != 0
            && std::env::var_os("ATLAS_DISABLE_PREFILL_V2").is_none();
        let bf16_kernel = if use_v2 {
            self.w4a16_gemm_t_m128_bf16_v2_k
        } else {
            self.w4a16_gemm_t_m128_bf16_k
        };
        // Final gate: the flag is honored only when the SELECTED kernel is
        // loaded (v2 when preferred, else v1) — not v1's handle unconditionally.
        let bf16_tc_prefill = bf16_kernel.0 != 0 && bf16_tc_env;

        macro_rules! w4_gemm {
            ($w:expr, $wt:expr, $cell:expr, $qcell:expr, $fp4cell:expr, $allow_fp4:expr, $in:expr, $out:expr, $n:expr, $k:expr, $allow_q4k:expr) => {
                match $wt {
                    // NVFP4 W4A4 MMQ prefill (ATLAS_FFN_NVFP4_MMQ) — HIGHEST priority.
                    // `$allow_fp4` = fp4mmq_prefill for gate/up, fp4mmq_down for down.
                    // Activation pre-quantized into `fp4_y` by the caller; the output is
                    // missing ×scale2, folded downstream (scaled SiLU-mul / scale_bf16).
                    _ if $allow_fp4 => {
                        let _ = $in;
                        let qw =
                            self.ensure_nvfp4_mmq_weight($fp4cell, ctx.gpu, $w, $n, $k, stream)?;
                        // Size the M tile to the batch when the batch is small and
                        // the small-tile entries are present. m must be <= mmq_x or
                        // grid.y>1 re-streams the weights per tile.
                        let (tk_nc, tk_wc, tile) = if m <= 16
                            && self.nvfp4_mmq16_nc_k.0 != 0
                            && mmq_small_tile_enabled()
                        {
                            (self.nvfp4_mmq16_nc_k, self.nvfp4_mmq16_wc_k, 16u32)
                        } else if m <= 32
                            && self.nvfp4_mmq32_nc_k.0 != 0
                            && mmq_small_tile_enabled()
                        {
                            (self.nvfp4_mmq32_nc_k, self.nvfp4_mmq32_wc_k, 32u32)
                        } else if m <= 64
                            && self.nvfp4_mmq64_nc_k.0 != 0
                            && mmq_small_tile_enabled()
                            && mmq_tile64_enabled()
                        {
                            (self.nvfp4_mmq64_nc_k, self.nvfp4_mmq64_wc_k, 64u32)
                        } else {
                            (self.nvfp4_mmq_nc_k, self.nvfp4_mmq_wc_k, 128u32)
                        };
                        ops::nvfp4_mmq_gemm_tiled(
                            ctx.gpu, tk_nc, tk_wc, tile, fp4_y, qw.w, $out, m, $n, $k, stream,
                        )?;
                    }
                    // Q4_K MMQ prefill (ATLAS_FFN_MMQ) — next priority, gated per-GEMM by
                    // `$allow_q4k` (false for down in the hybrid → falls to the faith2 arm).
                    // Activation `$in` is pre-quantized to q8_1 in `q4k_a` by the caller.
                    _ if q4k_prefill && $allow_q4k => {
                        let qw = self.ensure_q4k_weight($qcell, ctx.gpu, $w, $n, $k, stream)?;
                        ops::q4k_mmq_gemm(
                            ctx.gpu,
                            self.q4k_mmq_nc_k,
                            self.q4k_mmq_wc_k,
                            q4k_a,
                            qw.w_q4k,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    }
                    // W4A4 native-FP4 prefill (ATLAS_FP4_PREFILL) — HIGHEST priority.
                    // The activation is PRE-quantized into the NVFP4 scratch by the caller
                    // (`input` once for gate+up which share it; `gate_out` for down) — opt #1,
                    // avoids the redundant re-quant. This arm just runs w4a4_gemm against the
                    // native NVFP4 weight `$w` (no requant). sm_121a FP4 MMA.
                    _ if fp4_prefill => {
                        let _ = $in;
                        ops::w4a4_gemm(
                            ctx.gpu,
                            self.w4a4_gemm_k,
                            nvfp4_a_packed,
                            nvfp4_a_scale,
                            $w,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    }
                    // int8 W4A8 fast prefill (ATLAS_INT8_PREFILL) — next priority.
                    // Independent of `$wt`/the transposed copies: requant reads the
                    // non-transposed NVFP4 `$w` directly. Builds (once) + caches the
                    // int8 weight in `$cell`, then requant_a + faith2 via the shared
                    // scratch. Lossy (cosine ~0.99998). Also the HYBRID down path
                    // (down_faith2 && !$allow_q4k): down falls here instead of Q4_K.
                    _ if int8_prefill || (down_faith2 && !$allow_q4k) => {
                        let iw = self.ensure_int8_weight($cell, ctx.gpu, $w, $n, $k, stream)?;
                        // faith5 (ATLAS_INT8_FAITH5=1): int32 per-sb accumulation
                        // breaks the MMA→scale dependency chain. Same kernel signature
                        // + grid/block as faith2 — just a different KernelHandle.
                        let int8_kernel = if self.int8_faith5_k.0 != 0
                            && std::env::var_os("ATLAS_INT8_FAITH5").is_some()
                        {
                            self.int8_faith5_k
                        } else {
                            self.int8_faith2_k
                        };
                        ops::int8_gemm_faith2_prefill(
                            ctx.gpu,
                            int8_kernel,
                            self.requant_a_int8_k,
                            $in,
                            iw.w_i8,
                            iw.w_scale,
                            int8_a_i8,
                            int8_a_scale,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    }
                    // Lossless opt-in: BF16 128x128 tensor-core prefill (bit-equivalent
                    // to base `w4a16_gemm`). Preferred over the FP8 t_m128/v2 paths only
                    // when ATLAS_BF16_TC_PREFILL is set and the kernel is loaded. Within
                    // the lossless path, prefer the higher-occupancy v2 kernel (3 CTAs/SM,
                    // bit-identical to v1) when it is loaded; else the proven v1 kernel.
                    // Both go through the same launch helper (identical grid/block/args).
                    // FP8 M64 fast prefill (ATLAS_FP8_M64_PREFILL) — highest priority,
                    // M64 grid via the w4a16_gemm_n128 launcher.
                    Some(wt) if fp8_m64_prefill => ops::w4a16_gemm_n128(
                        ctx.gpu,
                        self.w4a16_gemm_t_k,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    // v2's COMPILED signature carries a 9th param, `ldb`
                    // (transposed-B row stride; == N for the FFN twins, which
                    // are built unpadded). It MUST go through the `_ldb`
                    // launcher: the 8-arg helper leaves cuLaunchKernel reading
                    // one-past-the-end of the param array for `ldb` —
                    // CUDA_ERROR_INVALID_VALUE or a host SIGSEGV depending on
                    // the neighboring heap word. v1 takes exactly 8 params and
                    // stays on the 8-arg helper.
                    Some(wt) if bf16_tc_prefill && use_v2 => ops::w4a16_gemm_n128_m128_bf16_ldb(
                        ctx.gpu,
                        bf16_kernel,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        $n,
                        stream,
                    )?,
                    Some(wt) if bf16_tc_prefill => ops::w4a16_gemm_n128_m128_bf16(
                        ctx.gpu,
                        bf16_kernel,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    // Small-M routing (DFlash verify, M<=64): delegate to
                    // `w4a16_prefill_gemm`, which picks `w4a16_gemm_t` /
                    // `w4a16_gemm_t_k64` per the w4a16_m17_bench numbers and
                    // falls back to the same v2/m128 kernels below.
                    // ATLAS_FFN_SMALLM=0 disables. Sits after the opt-in
                    // quant arms so explicit MMQ/int8/FP8 experiments keep
                    // priority.
                    Some(wt) if m <= 64 => {
                        self.w4a16_prefill_gemm(ctx, $w, Some(&wt), $in, $out, m, $n, $k, stream)?
                    }
                    // Prefer v2 (8-warp) > t_m128 (4-warp) > scalar-tile base.
                    Some(wt) if self.w4a16_gemm_t_m128_v2_k.0 != 0 => ops::w4a16_gemm_n128_m128_v2(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128_v2_k,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    Some(wt) if self.w4a16_gemm_t_m128_k.0 != 0 => ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128_k,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    _ => {
                        ops::w4a16_gemm(ctx.gpu, self.w4a16_gemm, $in, $w, $out, m, $n, $k, stream)?
                    }
                }
            };
        }

        // W4A4 opt #1: quantize the gate/up SHARED input `[M, H]` to NVFP4 ONCE
        // (gate and up both read it) instead of per-GEMM. Reused by both arms below.
        if fp4_prefill {
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                input,
                nvfp4_a_packed,
                nvfp4_a_scale,
                m,
                h,
                stream,
            )?;
        }
        // Q4_K opt: quantize the gate/up SHARED input `[M, H]` to q8_1 ONCE (both read it).
        if q4k_prefill {
            ops::quantize_act_q8_1(ctx.gpu, self.q4k_quant_act_k, input, q4k_a, m, h, stream)?;
        }
        // FP4-MMQ: quantize the gate/up SHARED input `[M, H]` to block_fp4_mmq ONCE.
        if fp4mmq_prefill {
            ops::nvfp4_mmq_quantize_act(
                ctx.gpu,
                self.nvfp4_quant_act_k,
                input,
                fp4_y,
                m,
                h,
                stream,
            )?;
        }
        // Per-projection timers. `dense_total` localised 23.7 s of a 28.2 s
        // Gemma-4-31B prefill to this function; these say WHICH of the three
        // projections it is. Roofline for that shape: 3 x 5376 x 21504 x 4012 tok
        // x 60 layers = 167 TFLOP, so 23.7 s is ~7 TFLOP/s against a bf16
        // tensor-core peak two orders higher — a hypothesis these numbers test
        // rather than assume.
        // ★ PER-STEP, NOT CUMULATIVE. The first version of this timer measured
        // elapsed-since-one-start at each of the three call sites, so `up_proj`
        // included `gate_proj` and `down_proj` included both — and the summed
        // "total profiled" then exceeded the wall clock, which is the tell.
        macro_rules! ffn_step {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!(
                        "  FFN prefill [{}] N={}: {}µs",
                        $label,
                        num_tokens,
                        $t0.elapsed().as_micros()
                    );
                    #[allow(unused_assignments)]
                    {
                        $t0 = std::time::Instant::now();
                    }
                }
            };
        }
        #[allow(unused_mut, unused_assignments)]
        let mut t_ffn = std::time::Instant::now();
        // gate_proj GEMM: [M, H] → [M, inter]
        w4_gemm!(
            &self.weights.gate_proj,
            self.weights.gate_proj_t,
            &self.int8_gate,
            &self.q4k_gate,
            &self.fp4mmq_gate,
            fp4mmq_prefill,
            input,
            gate_out,
            inter,
            h,
            true
        );
        ffn_step!("gate_proj", t_ffn);
        // up_proj GEMM: [M, H] → [M, inter]
        w4_gemm!(
            &self.weights.up_proj,
            self.weights.up_proj_t,
            &self.int8_up,
            &self.q4k_up,
            &self.fp4mmq_up,
            fp4mmq_prefill,
            input,
            up_out,
            inter,
            h,
            true
        );
        ffn_step!("up_proj", t_ffn);

        // LoRA gate/up deltas land here: the projections are complete and, with
        // the MMQ arm disabled above, gate_out/up_out hold true-valued BF16 —
        // so the delta adds in the same units it was trained in.
        self.apply_lora_gate_up(ctx, input, gate_out, up_out, m, stream)?;
        // activation(gate) * up for all M tokens (SiLU or GELU)
        let fused_down_quant = fp4mmq_down && self.nvfp4_silu_quant_k.0 != 0;
        if fused_down_quant {
            // Fused SiLU-mul + quantize straight into the down MMQ's y-format: the
            // [M, inter] bf16 intermediate is never written or re-read (that round-trip
            // is why the unfused down arm measured neutral). scale2 folds happen inside,
            // pre-clamp — identical math to the two-step path below.
            ops::nvfp4_silu_mul_quant(
                ctx.gpu,
                self.nvfp4_silu_quant_k,
                gate_out,
                up_out,
                fp4_y,
                self.weights.gate_proj.weight_scale_2,
                self.weights.up_proj.weight_scale_2,
                m,
                inter,
                stream,
            )?;
        } else if fp4mmq_prefill {
            // FP4-MMQ outputs are missing the per-tensor FP32 scale2 (the hardware MMA
            // applies only the per-16 e4m3 scales) — fold it here, before the nonlinearity.
            ops::nvfp4_silu_mul_scaled(
                ctx.gpu,
                self.nvfp4_silu_scaled_k,
                gate_out,
                up_out,
                gate_out,
                self.weights.gate_proj.weight_scale_2,
                self.weights.up_proj.weight_scale_2,
                m * inter,
                stream,
            )?;
        } else {
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
        }

        // W4A4 opt #1: quantize the down input (SiLU(gate)*up, `[M, inter]`) to NVFP4.
        if fp4_prefill {
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                gate_out,
                nvfp4_a_packed,
                nvfp4_a_scale,
                m,
                inter,
                stream,
            )?;
        }
        // Q4_K opt: quantize the down input (SiLU(gate)*up, `[M, inter]`) to q8_1.
        // Skip when the hybrid routes down to faith2 (it does its own int8 requant).
        if q4k_prefill && !down_faith2 {
            ops::quantize_act_q8_1(
                ctx.gpu,
                self.q4k_quant_act_k,
                gate_out,
                q4k_a,
                m,
                inter,
                stream,
            )?;
        }
        // FP4-MMQ down (two-step fallback, only when the fused kernel is absent):
        // quantize the down input (SiLU(gate)*up, `[M, inter]`) to block_fp4_mmq.
        if fp4mmq_down && !fused_down_quant {
            ops::nvfp4_mmq_quantize_act(
                ctx.gpu,
                self.nvfp4_quant_act_k,
                gate_out,
                fp4_y,
                m,
                inter,
                stream,
            )?;
        }
        // down_proj GEMM: [M, inter] → [M, H]
        // ($fp4cell is a placeholder — the FP4-MMQ arm is gated off by `false` below;
        // down stays on the default path in the FP4-MMQ hybrid.)
        let output = ctx.buffers.moe_output();
        w4_gemm!(
            &self.weights.down_proj,
            self.weights.down_proj_t,
            &self.int8_down,
            &self.q4k_down,
            &self.fp4mmq_down,
            fp4mmq_down,
            gate_out,
            output,
            h,
            inter,
            false
        );
        ffn_step!("down_proj", t_ffn);
        // FP4-MMQ down: fold the down-projection's per-tensor scale2 (no SiLU-mul here;
        // the consumer is the residual add).
        if fp4mmq_down {
            ops::nvfp4_scale_bf16(
                ctx.gpu,
                self.nvfp4_scale_k,
                output,
                self.weights.down_proj.weight_scale_2,
                m * h,
                stream,
            )?;
        }
        // AFTER the scale2 fold, not before: that fold scales the base
        // projection's output, and the delta is not part of that product.
        // `gate_out` holds silu(gate)*up — the activation the base down GEMM
        // just contracted over — because the fused silu+quant arm is off
        // whenever an adapter is installed.
        self.apply_lora_down(ctx, gate_out, output, m, stream)?;

        Ok(())
    }

    /// Batched forward (per-token loop). Used by forward_batched in model loop.
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }
}

/// Native BF16/FP8 layers do not own usable NVFP4 fallback weights. Their
/// small-batch path must therefore use the format-aware prefill dispatcher.
fn native_small_batch_uses_prefill(has_bf16: bool, has_fp8: bool) -> bool {
    has_bf16 || has_fp8
}

#[cfg(test)]
#[path = "dense_ffn_mmq_tests.rs"]
mod mmq_tests;

#[cfg(test)]
mod tests {
    use super::native_small_batch_uses_prefill;

    #[test]
    fn native_weight_presence_requires_prefill_dispatch() {
        assert!(native_small_batch_uses_prefill(true, false));
        assert!(native_small_batch_uses_prefill(false, true));
        assert!(native_small_batch_uses_prefill(true, true));
        assert!(!native_small_batch_uses_prefill(false, false));
    }
}
