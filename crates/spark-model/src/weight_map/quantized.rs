// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Runtime tag for the actual quantization format of a weight buffer in
/// GPU memory. Distinct from on-disk format (which `Nvfp4Variant` describes).
/// Used to assert at kernel-call sites that the weight matches what the
/// kernel expects — preventing silent leaks like FP8-block-scaled data
/// being passed through a NVFP4 GEMM, or single-scale FP8 being passed
/// through a kernel that expects per-row scales.
///
/// Phase 2c day-3 follow-up (2026-05-24): introduced after the audit at
/// `bench/phase2c-kv-sweep/CAUSAL-PATHWAY-AUDIT.md` found that block-scaled
/// FP8 weights from disk were being silently stuffed into the `row_scale`
/// field of `Fp8Weight` (which documents itself as per-row F32), causing
/// either crashes (when concat math read past the smaller block-scale
/// tensor) or — if the concat dimension happened to fit — silent precision
/// loss because downstream kernels (`fp8_gemm_n128`) take no scale arg
/// and assume single-scale FP8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuantFormat {
    /// BF16 dense — no quantization. Kernel must consume BF16 inputs.
    Bf16,
    /// FP8 E4M3 weight + per-row F32 dequant scale (`[N]` f32).
    /// Produced by runtime quantization from BF16 (`Fp8DenseWeight`)
    /// or by checkpoints that ship per-row scales.
    /// Consumed by `w8a16_gemv` / `w8a16_gemm`.
    Fp8PerRow,
    /// FP8 E4M3 weight + per-block BF16 dequant scale (`[N/BS, K/BS]` BF16).
    /// Standard Qwen-team FP8 release format (BS=128). NO Atlas kernel
    /// currently consumes this directly for SSM — kernels expect either
    /// dequant-to-BF16-then-NVFP4 (current path) or single-scale FP8.
    /// **Block-scaled FP8 GEMV/GEMM is the missing kernel** (open task).
    Fp8BlockScaled,
    /// FP8 E4M3 weight with a single global scale baked into the kernel
    /// (or implicit). Produced by `bf16_to_fp8` from a BF16 dense.
    /// Consumed by `fp8_gemm_n128` (takes no scale argument).
    Fp8SingleScale,
    /// NVFP4: packed E2M1 nibbles + per-group FP8 block scales + per-tensor
    /// F32 scale. Consumed by `w4a16_gemv`, `w4a16_gemm`, and variants.
    Nvfp4,
    /// Native MXFP4 (OCP micro-scaling): packed E2M1 nibbles + per-block
    /// **E8M0** power-of-2 scales (`GROUP_SIZE=32`), **no** per-tensor global.
    /// This is DeepSeek-V4-Flash's ORIGINAL on-disk routed-expert format. The
    /// bytes are landed device-resident UNCHANGED (transcode-free) — the
    /// scale byte is a biased exponent, effective scale `2^(byte-127)`.
    /// Consumed by the E8M0 variants of the MoE grouped/decode GEMMs
    /// (Phase-K lane); feeding these bytes through an `Nvfp4` kernel (which
    /// reads the scale as FP8-E4M3 per-16 and applies a global) = silent
    /// garbage — assert with `WeightQuantFormat::expect` at the dispatch site.
    Mxfp4E8m0,
    /// Keep-packed PrismML ternary Q2_0 (ggml id 42): raw `block_q2_0` blocks
    /// (fp16 inline scale + 2-bit codes per group), dequantized in-kernel by the
    /// native `q2_0_gemv` decode GEMV. Consumed only by that kernel — feeding
    /// these bytes through any other GEMV/GEMM is silent garbage.
    PackedQ2_0,
    /// Keep-packed EXL3 (QTIP trellis, `ATLAS_EXL3_NATIVE=1`): u16 trellis
    /// codes `[in/16, out/16, 16*K]` + exact-f16 `suh`/`svh` Hadamard sign
    /// vectors, decoded in-kernel by the fused `exl3_matmul` GEMV/GEMM
    /// (cooperative launches; see `layers/ops/exl3_matmul.rs`). Consumed only
    /// by those kernels — the trellis bytes are code STREAMS, not values, so
    /// feeding them through any other GEMV/GEMM is silent garbage. This tag is
    /// also the future `experts_scale_kind` value for a grouped-MoE trellis
    /// arm (the `Mxfp4E8m0` precedent).
    Exl3Trellis,
}

impl WeightQuantFormat {
    /// Assert that `self` matches `expected`; panic with a descriptive
    /// message if not. Used at kernel-call sites to prevent silent leaks
    /// of one quant format into a kernel that expects a different one.
    #[inline]
    #[track_caller]
    pub fn expect(self, expected: WeightQuantFormat, context: &str) {
        if self != expected {
            panic!(
                "WeightQuantFormat mismatch at {context}: kernel expects {expected:?}, \
                 but the weight buffer is tagged {self:?}. This is a silent quant-leak \
                 that would produce wrong outputs without this assertion."
            );
        }
    }
}

/// Keep-packed ternary Q2_0 weight: a single contiguous buffer of raw PrismML
/// `block_q2_0` blocks ([fp16 d][group/4 bytes of 2-bit codes], `value =
/// (code-1)*d`), row-major over `[n, k]`. The scale is INLINE (one fp16 per
/// group of `group` elements) — there is no companion scale tensor, unlike
/// NVFP4/FP8. Consumed by the native `q2_0_gemv` decode kernel, which reads the
/// scale from each block. Built from a `WeightDtype::PackedQ2_0` store tensor
/// under `ATLAS_GGUF_NATIVE_Q2=1`; the buffer is owned by the `WeightStore`, so
/// this struct only borrows the pointer (no free on drop).
#[derive(Debug, Clone, Copy)]
pub struct PackedQ2Weight {
    /// Raw packed `block_q2_0` bytes, `n * (k/group) * (2 + group/4)` long.
    pub weight: DevicePtr,
    /// Output rows (weight is `[n, k]`).
    pub n: u32,
    /// Input columns (contraction dim).
    pub k: u32,
    /// Group size (128 or 64) — elements per block / per inline scale.
    pub group: u16,
}

impl PackedQ2Weight {
    /// True if the backing buffer is NULL (unset placeholder).
    pub fn is_null(&self) -> bool {
        self.weight == DevicePtr::NULL
    }
}

/// NVFP4 quantized weight: packed E2M1 data + FP8 block scales + FP32 per-tensor scale.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedWeight {
    /// Packed E2M1 weights (2 values per byte).
    pub weight: DevicePtr,
    /// Per-group FP8 block scales.
    pub weight_scale: DevicePtr,
    /// Per-tensor FP32 scale factor (extracted from GPU via D2H copy at load time).
    pub weight_scale_2: f32,
    /// Input activation scale (FP32 on device, for FP8 activation path).
    pub input_scale: DevicePtr,
    /// Per-row FP32 scale2 on device (`[N]` floats). When set, the `w4a16_gemv_prs`
    /// kernel reads scale2 per output row instead of the scalar `weight_scale_2`,
    /// eliminating precision loss from per-tensor absmax on outlier rows.
    pub weight_scale_2_vec: DevicePtr,
}

impl QuantizedWeight {
    /// Null weight (all pointers NULL). Used for remote experts under EP.
    pub fn null() -> Self {
        Self {
            weight: DevicePtr::NULL,
            weight_scale: DevicePtr::NULL,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        }
    }

    /// Whether this weight has per-row scale2 (for PRS GEMV dispatch).
    pub fn has_per_row_scale2(&self) -> bool {
        self.weight_scale_2_vec != DevicePtr::NULL
    }

    /// Whether this weight points to NULL (remote expert placeholder).
    pub fn is_null(&self) -> bool {
        self.weight == DevicePtr::NULL
    }

    /// Concatenate two NVFP4 weights by rows: `[N1, K/2]` + `[N2, K/2]` → `[N1+N2, K/2]`.
    ///
    /// Both weights MUST share the same `K` (input dimension) and the same scalar
    /// `weight_scale_2`. The packed weight bytes and FP8 block scales are concatenated
    /// on-GPU via `cuMemcpy`.
    pub fn concat_rows(
        &self,
        other: &QuantizedWeight,
        n1: usize,
        n2: usize,
        k: usize,
        gpu: &dyn GpuBackend,
    ) -> anyhow::Result<QuantizedWeight> {
        // The concatenated weight carries a single scalar scale2 (self's) for
        // ALL rows — a mismatched `other` would silently dequantize its rows
        // with the wrong per-tensor scale. This bit-exact equality only holds
        // for `Nvfp4Variant::Standard` (NVIDIA ModelOpt) checkpoints, whose
        // convention is a single global per-tensor `weight_scale_2` scalar
        // shared across every row of the tensor — so two tensors quantized
        // together by the same run share the identical f32 bit pattern.
        // Other conventions (e.g. compressed-tensors) may carry independent
        // per-tensor scales even for logically concatenable projections.
        anyhow::ensure!(
            self.weight_scale_2 == other.weight_scale_2,
            "concat_rows: weight_scale_2 mismatch (self={}, other={}) — both NVFP4 \
             tensors must share the same per-tensor scale to be concatenated. \
             This is expected for ModelOpt/Standard NVFP4 checkpoints (single \
             global per-tensor scale2); re-quantize with the ModelOpt/Standard \
             quantizer, or report which checkpoint/quantizer produced independent \
             per-tensor scales for these projections",
            self.weight_scale_2,
            other.weight_scale_2,
        );
        const GROUP_SIZE: usize = 16;
        let half_k = k / 2;
        let num_groups = k / GROUP_SIZE;

        let total_n = n1 + n2;
        let packed_size = total_n * half_k;
        let scale_size = total_n * num_groups;

        let new_weight = gpu.alloc(packed_size)?;
        let new_scale = gpu.alloc(scale_size)?;

        gpu.copy_d2d(self.weight, new_weight, n1 * half_k)?;
        gpu.copy_d2d(other.weight, new_weight.offset(n1 * half_k), n2 * half_k)?;

        gpu.copy_d2d(self.weight_scale, new_scale, n1 * num_groups)?;
        gpu.copy_d2d(
            other.weight_scale,
            new_scale.offset(n1 * num_groups),
            n2 * num_groups,
        )?;

        Ok(QuantizedWeight {
            weight: new_weight,
            weight_scale: new_scale,
            weight_scale_2: self.weight_scale_2,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        })
    }

    /// Resolve the `transpose_u8` GPU kernel for the load-time transpose
    /// paths, or `None` to use the host byte-loop fallback. `None` when the
    /// target's kernel set lacks it, or when `ATLAS_HOST_TRANSPOSE=1` forces
    /// the host path (parity/debug kill switch).
    fn host_transpose_kernel(gpu: &dyn GpuBackend) -> Option<spark_runtime::gpu::KernelHandle> {
        if std::env::var("ATLAS_HOST_TRANSPOSE").as_deref() == Ok("1") {
            return None;
        }
        let k = crate::layers::try_kernel(gpu, "transpose_u8", "transpose_u8");
        (k.0 != 0).then_some(k)
    }

    /// Transpose weight layout from [N, K/2] to [K/2, N] for coalesced GEMM reads.
    ///
    /// Also transposes scale from [N, K/GROUP_SIZE] to [K/GROUP_SIZE, N].
    /// Returns a NEW `QuantizedWeight` with freshly allocated GPU buffers,
    /// leaving the original untouched (needed for decode kernels).
    pub fn transpose_for_gemm(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        k: usize,
    ) -> Result<QuantizedWeight> {
        // NVFP4 default: per-16 block scales. Native MXFP4 (E8M0) is per-32 —
        // ARM-2 Phase-K callers use `transpose_for_gemm_gs(.., 32)` for routed
        // experts (the scale tensor is [N, K/32], not [N, K/16]).
        self.transpose_for_gemm_gs(gpu, n, k, 16)
    }

    /// `transpose_for_gemm` with an explicit scale block size. Scale tensor is
    /// `[N, K/group_size]`; the packed-weight transpose is group-size-independent.
    pub fn transpose_for_gemm_gs(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<QuantizedWeight> {
        let half_k = k / 2;
        let num_groups = k / group_size;
        let packed_size = n * half_k;
        let scale_size = n * num_groups;

        // GPU path: two transpose_u8 launches instead of D2H -> host
        // O(N*K) byte loop -> H2D (the cold-load host bounce; ~13.6 GB at
        // 27B). ATLAS_HOST_TRANSPOSE=1 forces the host path (parity/debug);
        // targets without the kernel fall back to it silently.
        if let Some(tk) = Self::host_transpose_kernel(gpu) {
            let new_weight = gpu.alloc(packed_size)?;
            let new_scale = gpu.alloc(scale_size)?;
            crate::layers::ops::transpose_u8(
                gpu,
                tk,
                self.weight,
                new_weight,
                n as u32,
                half_k as u32,
                0,
            )?;
            crate::layers::ops::transpose_u8(
                gpu,
                tk,
                self.weight_scale,
                new_scale,
                n as u32,
                num_groups as u32,
                0,
            )?;
            gpu.synchronize(0)?;
            return Ok(QuantizedWeight {
                weight: new_weight,
                weight_scale: new_scale,
                weight_scale_2: self.weight_scale_2,
                input_scale: self.input_scale,
                weight_scale_2_vec: self.weight_scale_2_vec,
            });
        }

        // Transpose B_packed: [N, K/2] → [K/2, N] into a NEW GPU allocation.
        let mut buf = vec![0u8; packed_size];
        gpu.copy_d2h(self.weight, &mut buf)?;
        let mut t_buf = vec![0u8; packed_size];
        for i in 0..n {
            for j in 0..half_k {
                t_buf[j * n + i] = buf[i * half_k + j];
            }
        }
        let new_weight = gpu.alloc(packed_size)?;
        gpu.copy_h2d(&t_buf, new_weight)?;

        // Transpose B_scale: [N, K/group_size] → [K/group_size, N] into a NEW allocation.
        let mut sbuf = vec![0u8; scale_size];
        gpu.copy_d2h(self.weight_scale, &mut sbuf)?;
        let mut st_buf = vec![0u8; scale_size];
        for i in 0..n {
            for j in 0..num_groups {
                st_buf[j * n + i] = sbuf[i * num_groups + j];
            }
        }
        let new_scale = gpu.alloc(scale_size)?;
        gpu.copy_h2d(&st_buf, new_scale)?;

        Ok(QuantizedWeight {
            weight: new_weight,
            weight_scale: new_scale,
            weight_scale_2: self.weight_scale_2,
            input_scale: self.input_scale,
            weight_scale_2_vec: self.weight_scale_2_vec,
        })
    }

    /// Transpose SEVERAL weights sharing one K and concatenate them along N
    /// into a single `[K/2, N_total]` twin, so three GEMMs become one.
    ///
    /// Motivation (GB10, decode M=16): the attention k/v projections are
    /// N=1024, which against the 128-wide N tile yields **8 CTAs on 48 SMs** —
    /// 40 SMs idle, 23.6 GB/s, 9.75x off the bandwidth floor. Concatenating
    /// q|k|v to N=14336 gives 112 CTAs in ONE launch. Bit-identical: every
    /// output element is the same dot product against the same column, merely
    /// relocated along N.
    ///
    /// REQUIRES all parts to share `weight_scale_2` — the GEMM applies a single
    /// `scale2` to the whole launch. Callers MUST verify this (the values live
    /// on device); `None` is returned if the caller passes an empty list.
    pub fn transpose_concat_for_gemm(
        gpu: &dyn GpuBackend,
        parts: &[(&QuantizedWeight, usize)],
        k: usize,
    ) -> Result<QuantizedWeight> {
        Self::transpose_concat_for_gemm_gs(gpu, parts, k, 16)
    }

    /// `transpose_concat_for_gemm` with an explicit scale block size.
    /// `transpose_concat_for_gemm_gs` with the output ROW STRIDE padded to
    /// `align_up(n_total, align)`, pad columns left zero.
    ///
    /// The transposed layout puts row r at byte offset `r * stride`, and the
    /// tile GEMM reads B with 16-byte `cp.async`, which requires a 16-byte
    /// aligned source. When `n_total` is not a multiple of 16 — lm_head's N is
    /// the VOCAB SIZE, 248077 here, which is ODD — 15 of every 16 rows are
    /// misaligned and the kernel faults with CUDA_ERROR_MISALIGNED_ADDRESS.
    /// Padding the stride is what makes a transposed lm_head legal at all.
    ///
    /// Returns `(weight, stride)`; pass the stride to `w4a16_gemm_n128_ldb`.
    pub fn transpose_concat_for_gemm_padded(
        gpu: &dyn GpuBackend,
        parts: &[(&QuantizedWeight, usize)],
        k: usize,
        group_size: usize,
        align: usize,
    ) -> Result<(QuantizedWeight, usize)> {
        let n_total: usize = parts.iter().map(|(_, n)| *n).sum();
        let stride = n_total.div_ceil(align) * align;
        Self::transpose_impl(gpu, parts, k, group_size, stride).map(|w| (w, stride))
    }

    pub fn transpose_concat_for_gemm_gs(
        gpu: &dyn GpuBackend,
        parts: &[(&QuantizedWeight, usize)],
        k: usize,
        group_size: usize,
    ) -> Result<QuantizedWeight> {
        let n_total: usize = parts.iter().map(|(_, n)| *n).sum();
        Self::transpose_impl(gpu, parts, k, group_size, n_total)
    }

    /// Single implementation for both (SSOT). `stride >= n_total` is the row
    /// pitch of the transposed output; columns `n_total..stride` stay zero.
    fn transpose_impl(
        gpu: &dyn GpuBackend,
        parts: &[(&QuantizedWeight, usize)],
        k: usize,
        group_size: usize,
        stride: usize,
    ) -> Result<QuantizedWeight> {
        let first = parts
            .first()
            .map(|(w, _)| *w)
            .context("transpose_concat_for_gemm: empty parts")?;
        let half_k = k / 2;
        let num_groups = k / group_size;
        let n_total: usize = parts.iter().map(|(_, n)| *n).sum();
        debug_assert!(
            stride >= n_total,
            "transpose_impl: stride {stride} < n_total {n_total}"
        );

        // GPU path: per part, one transpose_u8 launch into a contiguous
        // [half_k, n] temp, then ONE pitched 2D copy into the strided dest
        // column window (cudaMemcpy2DAsync on the CUDA backend). Replaces
        // the D2H -> host O(N*K) byte loop -> H2D cold-load bounce. Pad
        // columns `n_total..stride` are zeroed by the memset up front,
        // matching the host path's zeroed staging vec.
        if let Some(tk) = Self::host_transpose_kernel(gpu) {
            let new_weight = gpu.alloc(stride * half_k)?;
            let new_scale = gpu.alloc(stride * num_groups)?;
            if stride > n_total {
                gpu.memset(new_weight, 0, stride * half_k)?;
                gpu.memset(new_scale, 0, stride * num_groups)?;
            }
            let mut temps: Vec<DevicePtr> = Vec::with_capacity(parts.len() * 2);
            let mut n_off = 0usize;
            for (w, n) in parts {
                let n = *n;
                let t_w = gpu.alloc(n * half_k)?;
                crate::layers::ops::transpose_u8(
                    gpu,
                    tk,
                    w.weight,
                    t_w,
                    n as u32,
                    half_k as u32,
                    0,
                )?;
                gpu.copy_d2d_2d_async(t_w, n, new_weight.offset(n_off), stride, n, half_k, 0)?;
                let t_s = gpu.alloc(n * num_groups)?;
                crate::layers::ops::transpose_u8(
                    gpu,
                    tk,
                    w.weight_scale,
                    t_s,
                    n as u32,
                    num_groups as u32,
                    0,
                )?;
                gpu.copy_d2d_2d_async(t_s, n, new_scale.offset(n_off), stride, n, num_groups, 0)?;
                temps.push(t_w);
                temps.push(t_s);
                n_off += n;
            }
            gpu.synchronize(0)?;
            for t in temps {
                gpu.free(t)?;
            }
            return Ok(QuantizedWeight {
                weight: new_weight,
                weight_scale: new_scale,
                weight_scale_2: first.weight_scale_2,
                input_scale: first.input_scale,
                weight_scale_2_vec: first.weight_scale_2_vec,
            });
        }

        let mut t_buf = vec![0u8; stride * half_k];
        let mut st_buf = vec![0u8; stride * num_groups];
        let mut n_off = 0usize;
        for (w, n) in parts {
            let n = *n;
            let mut buf = vec![0u8; n * half_k];
            gpu.copy_d2h(w.weight, &mut buf)?;
            for i in 0..n {
                for j in 0..half_k {
                    t_buf[j * stride + n_off + i] = buf[i * half_k + j];
                }
            }
            let mut sbuf = vec![0u8; n * num_groups];
            gpu.copy_d2h(w.weight_scale, &mut sbuf)?;
            for i in 0..n {
                for j in 0..num_groups {
                    st_buf[j * stride + n_off + i] = sbuf[i * num_groups + j];
                }
            }
            n_off += n;
        }

        let new_weight = gpu.alloc(t_buf.len())?;
        gpu.copy_h2d(&t_buf, new_weight)?;
        let new_scale = gpu.alloc(st_buf.len())?;
        gpu.copy_h2d(&st_buf, new_scale)?;

        Ok(QuantizedWeight {
            weight: new_weight,
            weight_scale: new_scale,
            weight_scale_2: first.weight_scale_2,
            input_scale: first.input_scale,
            weight_scale_2_vec: first.weight_scale_2_vec,
        })
    }

    /// Pre-dequant NVFP4 → FP8 E4M3 for zero-overhead prefill GEMMs.
    ///
    /// Reads B_packed[N, K/2] + B_scale[N, K/GROUP_SIZE] + scale2 and produces
    /// B_fp8[N, K] on GPU.  The resulting DevicePtr can be used with `fp8_gemm_t`
    /// which eliminates the per-inference dequant phase entirely.
    pub fn predequant_to_fp8(
        &self,
        gpu: &dyn GpuBackend,
        predequant_kernel: spark_runtime::gpu::KernelHandle,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let fp8_buf = gpu.alloc(n * k)?;
        crate::layers::ops::predequant_nvfp4_to_fp8(
            gpu,
            predequant_kernel,
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            fp8_buf,
            n as u32,
            k as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        Ok(fp8_buf)
    }
}

/// BF16 dense weight (no quantization).
#[derive(Debug, Clone, Copy)]
pub struct DenseWeight {
    pub weight: DevicePtr,
}

impl DenseWeight {
    /// Quantize a BF16 weight `[N, K]` to FP8 E4M3 `[N, K]` with per-row
    /// f32 scales. Allocates the FP8 buffer + row_scale buffer on the
    /// GPU, runs the `quantize_bf16_to_fp8` kernel, and returns the
    /// resulting [`Fp8DenseWeight`].
    ///
    /// Called once at model load time. Caller is responsible for any
    /// stream synchronization needed before the returned weight is
    /// consumed by `fp8_gemm_n128` or related kernels.
    ///
    /// Phase G (DFlash drafter FP8 weights). Mirrors
    /// [`QuantizedWeight::predequant_to_fp8`] for the BF16 source path.
    pub fn quantize_to_fp8(
        &self,
        gpu: &dyn GpuBackend,
        quantize_kernel: spark_runtime::gpu::KernelHandle,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<Fp8DenseWeight> {
        let fp8_buf = gpu.alloc(n * k)?;
        let row_scale_buf = gpu.alloc(n * std::mem::size_of::<f32>())?;
        crate::layers::ops::quantize_bf16_to_fp8(
            gpu,
            quantize_kernel,
            self.weight,
            fp8_buf,
            row_scale_buf,
            n as u32,
            k as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        Ok(Fp8DenseWeight {
            weight: fp8_buf,
            row_scale: row_scale_buf,
        })
    }
}

/// FP8 E4M3 dense weight (runtime-quantized from BF16).
///
/// Halves weight bandwidth vs BF16. Per-row f32 scale preserves accuracy.
/// Created at model load time via GPU-side quantization kernel.
#[derive(Debug, Clone, Copy)]
pub struct Fp8DenseWeight {
    /// FP8 E4M3 weight data: [N, K] bytes.
    pub weight: DevicePtr,
    /// Per-row dequant scale: `[N]` f32.
    pub row_scale: DevicePtr,
}

/// FP8 E4M3 checkpoint weight loaded directly from safetensors.
///
/// This struct carries an FP8 weight buffer along with its dequantization
/// scale. The exact scale layout depends on the [`WeightQuantFormat`] tag
/// in `scale_format`:
///   - [`WeightQuantFormat::Fp8PerRow`] — `scale` is `[N]` f32 per-row.
///   - [`WeightQuantFormat::Fp8BlockScaled`] — `scale` is `[N/BS, K/BS]`
///     BF16 per-block (BS = 128 typically, the Qwen FP8 release convention).
///   - [`WeightQuantFormat::Fp8SingleScale`] — `scale` is the NULL DevicePtr;
///     a single global scale is baked into the kernel that consumes this.
///
/// **Always check `scale_format` before reading `scale` as a particular
/// shape.** Prior to the format tag (Phase 2c day-3 follow-up), the
/// `Fp8Weight` struct silently mixed all three layouts in a single
/// field, causing a `cuMemcpyDtoDAsync_v2 INVALID_VALUE` crash when the
/// SSM build path tried to concat per-row F32 scales out of a buffer
/// that actually held per-block BF16 scales (lower memory than expected).
#[derive(Debug, Clone, Copy)]
pub struct Fp8Weight {
    /// [N, K] FP8 E4M3 weight bytes on GPU.
    pub weight: DevicePtr,
    /// Dequantization scale pointer. **Shape and dtype depend on
    /// `scale_format`** — see struct docs.
    pub row_scale: DevicePtr,
    /// Output dimension (rows).
    pub n: u32,
    /// Input dimension (columns).
    pub k: u32,
    /// Tag for the `row_scale` buffer's actual format. Asserted at
    /// kernel call sites via `WeightQuantFormat::expect(...)`.
    pub scale_format: WeightQuantFormat,
}

/// FP8 E4M3 weight with transposed layout for coalesced prefill GEMM.
///
/// B_t: [K, N] — transposed from checkpoint's B[N, K].
/// block_scale_t: [K/128, N/128] — transposed from [N/128, K/128].
/// Enables ~14x faster prefill via w8a16_gemm_t kernel.
#[derive(Debug, Clone, Copy)]
pub struct Fp8WeightTransposed {
    /// [K, N] FP8 E4M3 transposed weight on GPU.
    pub weight_t: DevicePtr,
    /// [K/128, N/128] FP32 transposed block scales on GPU (widened at load).
    pub scale_t: DevicePtr,
    pub n: u32,
    pub k: u32,
}

impl Fp8Weight {
    /// Transpose this FP8 weight for coalesced prefill GEMM.
    /// Allocates new GPU buffers for `B_t[K,N]` (FP8 bytes) and
    /// `scale_t[K/128, N/128]` (FP32; `row_scale` is already FP32).
    pub fn transpose_for_gemm(
        &self,
        gpu: &dyn GpuBackend,
        transpose_k: spark_runtime::gpu::KernelHandle,
        transpose_scale_k: spark_runtime::gpu::KernelHandle,
        stream: u64,
    ) -> anyhow::Result<Fp8WeightTransposed> {
        let n = self.n as usize;
        let k = self.k as usize;

        // Allocate transposed weight: [K, N] bytes
        let weight_t = gpu.alloc(k * n)?;
        crate::layers::ops::transpose_fp8(
            gpu,
            transpose_k,
            self.weight,
            weight_t,
            self.n,
            self.k,
            stream,
        )?;

        // Allocate transposed scale: [K/128, N/128] × 4 bytes (FP32).
        // `row_scale` is now an FP32 block-scale buffer (widened at load), and
        // `transpose_block_scale` is an FP32→FP32 transpose — see
        // `load_fp8_block_scaled_as_fp8weight` / `w8a16_gemm_t.cu`.
        let n_blocks = n.div_ceil(128);
        let k_blocks = k.div_ceil(128);
        let scale_t = gpu.alloc(k_blocks * n_blocks * 4)?;
        crate::layers::ops::transpose_block_scale(
            gpu,
            transpose_scale_k,
            self.row_scale,
            scale_t,
            n_blocks as u32,
            k_blocks as u32,
            stream,
        )?;

        gpu.synchronize(stream)?;

        Ok(Fp8WeightTransposed {
            weight_t,
            scale_t,
            n: self.n,
            k: self.k,
        })
    }
}
