// SPDX-License-Identifier: AGPL-3.0-only

//! Dense SwiGLU FFN component for non-MoE models.
//!
//! Forward: gate = gate_proj(x), up = up_proj(x), out = down_proj(SiLU(gate) * up)
//! 2 fused kernel launches per decode token (dual GEMV + SiLU-fused down GEMV).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, Fp8WeightTransposed, QuantizedWeight};

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
    w4a16_gemv_dual: KernelHandle,
    w4a16_gemv_silu_input: KernelHandle,
    // LOSSLESS single-warp-per-output decode variants (8 outputs/block, no smem
    // cross-warp reduce). Bit-identical to the 64-thread kernels (proven by the
    // w4a16_gemv_sw microtest). Opt-in via ATLAS_DECODE_OPT (default off →
    // dispatch unchanged). KernelHandle(0) on miss → fall back to base kernels.
    w4a16_gemv_dual_sw: KernelHandle,
    w4a16_gemv_silu_input_sw: KernelHandle,
    decode_opt: bool,
    w4a16_gemv_dual_batch2: KernelHandle,
    w4a16_gemv_dual_batch3: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
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
            w4a16_gemv_dual: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            w4a16_gemv_silu_input: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?,
            w4a16_gemv_dual_sw: super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_sw"),
            w4a16_gemv_silu_input_sw: super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_silu_input_sw",
            ),
            decode_opt: std::env::var_os("ATLAS_DECODE_OPT").is_some(),
            w4a16_gemv_dual_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_dual_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            w4a16_gemm_t_m128_v2_k: super::try_kernel(gpu, "w4a16_v2", "w4a16_gemm_t_m128_v2"),
            w4a16_gemm_t_m128_bf16_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128_bf16"),
            w4a16_gemm_t_m128_bf16_v2_k: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m128_bf16_v2",
            ),
            w4a16_gemm_t_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t"),
            int8_faith2_k: super::try_kernel(gpu, "w4a16", "int8_gemm_faith2"),
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
            nvfp4_quant_act_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_quantize_bf16"),
            nvfp4_repack_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_repack"),
            nvfp4_silu_scaled_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_silu_mul_scaled"),
            nvfp4_silu_quant_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_silu_mul_quant"),
            nvfp4_scale_k: super::try_kernel(gpu, "nvfp4_mmq", "atlas_nvfp4_scale_bf16"),
            fp4mmq_gate: std::sync::OnceLock::new(),
            fp4mmq_up: std::sync::OnceLock::new(),
            fp4mmq_down: std::sync::OnceLock::new(),
            w4a16_gemm_t_k64_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_k64"),
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
            static TWIN_LOG: std::sync::Once = std::sync::Once::new();
            TWIN_LOG.call_once(|| {
                eprintln!(
                    "[atlas] ATLAS_FFN_MMQ: freed transposed FFN `_t` copies (dead under Q4_K prefill) — Q4_K weights net to ~0 vs NVFP4 baseline"
                );
            });
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
            .chain(down_t.take().into_iter())
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
            static FP4_TWIN_LOG: std::sync::Once = std::sync::Once::new();
            FP4_TWIN_LOG.call_once(|| {
                eprintln!(
                    "[atlas] ATLAS_FFN_NVFP4_MMQ: freed gate/up `_t` copies (dead under FP4-MMQ prefill) — block_nvfp4 copies net to ~0 vs NVFP4 baseline"
                );
            });
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

    /// Install BF16 dense MLP weights. After this call, the forward paths
    /// dispatch to the BF16 GEMV/GEMM kernels instead of w4a16. The
    /// caller must ensure the BF16 kernels are loaded (see
    /// `dense_gemv_bf16_k` / `dense_gemm_bf16_k` checks). Spec-decode
    /// batched paths (`forward_k2`, `forward_k3`) are NOT supported on
    /// the BF16 path — Gemma-4 dense has no MTP so they're never called.
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
        fn decode_ffn_via_gemm() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("ATLAS_DECODE_FFN_VIA_GEMM").ok().as_deref() == Some("1")
            })
        }
        if decode_ffn_via_gemm() && self.activation == FfnActivation::SiLU && self.act_mul.0 != 0 {
            let wt_alive =
                |w: &Option<QuantizedWeight>| w.as_ref().is_some_and(|w| !w.weight.is_null());
            if wt_alive(&self.weights.gate_proj_t) && wt_alive(&self.weights.up_proj_t) {
                static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                LOGGED.get_or_init(|| {
                    tracing::info!(
                        "decode FFN via verify GEMM path (ATLAS_DECODE_FFN_VIA_GEMM=1): \
                         gate/up/down through w4a16_prefill_gemm at M=1"
                    );
                });
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
            static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            WARNED.get_or_init(|| {
                tracing::warn!(
                    "ATLAS_DECODE_FFN_VIA_GEMM=1 requested but transposed FFN copies \
                     are freed/absent (NVFP4-MMQ prefill arm?) — falling back to GEMV; \
                     the unification experiment is NOT active"
                );
            });
        }

        // Fused gate_proj + up_proj: [1, H] → [1, inter] × 2.
        // Single-warp variant (lossless) when ATLAS_DECODE_OPT is on and the
        // _sw kernel is present; otherwise the proven 64-thread kernel.
        let use_sw = self.decode_opt
            && self.w4a16_gemv_dual_sw.0 != 0
            && self.w4a16_gemv_silu_input_sw.0 != 0;
        if use_sw {
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
        let split_silu = self.activation == FfnActivation::SiLU
            && self.act_mul.0 != 0
            && self.w4a16_gemv.0 != 0
            && std::env::var_os("ATLAS_NO_DECODE_SPLIT_SILU").is_none();
        if split_silu {
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                gate_out,
                &self.weights.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }
        match self.activation {
            FfnActivation::SiLU => {
                // Fused SiLU(gate)*up + down_proj: [1, inter] → [1, H]
                if use_sw {
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
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv,
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

    /// K=2 speculative: batched GEMV for 2 tokens.
    /// 3 launches: dual batch2 (gate+up) + silu_mul + batch2 (down).
    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
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

        Ok(())
    }

    /// K=3 speculative: batched GEMV for 3 tokens.
    /// 3 launches: dual batch3 (gate+up) + silu_mul + batch3 (down).
    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
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
        fn small_m_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("ATLAS_FFN_SMALLM").ok().as_deref() != Some("0"))
        }
        if let Some(wt) = wt {
            if m <= 64 && k.is_multiple_of(32) && small_m_enabled() {
                if k >= 8192 && k.is_multiple_of(64) && self.w4a16_gemm_t_k64_k.0 != 0 {
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

    pub fn forward_prefill(
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
            // helper: tensor-core GEMM when available, else scalar
            macro_rules! ffn_gemm {
                ($a:expr, $b:expr, $c:expr, $n:expr, $k:expr) => {
                    if tc {
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
        let bf16_tc_prefill = self.w4a16_gemm_t_m128_bf16_k.0 != 0
            && std::env::var_os("ATLAS_BF16_TC_PREFILL").is_some();
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
            static INT8_LOG: std::sync::Once = std::sync::Once::new();
            INT8_LOG.call_once(|| {
                eprintln!(
                    "[atlas] ATLAS_INT8_PREFILL=1: dense-FFN prefill via int8_gemm_faith2 (W4A8 requant→int8 MMA, lossy ~0.99998 cosine)"
                );
            });
        }
        // NVFP4 W4A4 MMQ prefill (ATLAS_FFN_NVFP4_MMQ) — vendored llama Blackwell
        // block-scale FP4 MMA, gate/up ONLY (hybrid: down stays on the default t_m128
        // path — SiLU(gate)*up is heavy-tailed and accuracy-critical). SiLU models only
        // (the scale2 fold lives in the scaled SiLU-mul). Mutually exclusive with
        // ATLAS_FFN_MMQ (both use the shared ffn_act_q8 scratch); this arm wins.
        let fp4mmq_prefill = self.nvfp4_mmq_nc_k.0 != 0
            && self.nvfp4_quant_act_k.0 != 0
            && self.nvfp4_silu_scaled_k.0 != 0
            && matches!(self.activation, FfnActivation::SiLU)
            && std::env::var_os("ATLAS_NO_FFN_NVFP4_MMQ").is_none();
        if fp4mmq_prefill {
            static FP4MMQ_LOG: std::sync::Once = std::sync::Once::new();
            FP4MMQ_LOG.call_once(|| {
                eprintln!(
                    "[atlas] ATLAS_FFN_NVFP4_MMQ=1: dense-FFN gate/up prefill via vendored llama NVFP4 W4A4 MMQ (block-scale FP4 MMA, ~80 TFLOP/s vs t_m128 ~51)"
                );
            });
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
            static FP4_LOG: std::sync::Once = std::sync::Once::new();
            FP4_LOG.call_once(|| {
                eprintln!(
                    "[atlas] ATLAS_FP4_PREFILL=1: dense-FFN prefill via w4a4_gemm (native FP4 MMA sm_121a, W4A4)"
                );
            });
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
            static Q4K_LOG: std::sync::Once = std::sync::Once::new();
            Q4K_LOG.call_once(|| {
                eprintln!(
                    "[atlas] ATLAS_FFN_MMQ=1: dense-FFN prefill via vendored llama Q4_K MMQ (W4A8, +25%/+10% gate·down vs faith2)"
                );
            });
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
                        ops::nvfp4_mmq_gemm(
                            ctx.gpu,
                            self.nvfp4_mmq_nc_k,
                            self.nvfp4_mmq_wc_k,
                            fp4_y,
                            qw.w,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
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
                        ops::int8_gemm_faith2_prefill(
                            ctx.gpu,
                            self.int8_faith2_k,
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
