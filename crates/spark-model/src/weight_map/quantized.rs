// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

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
}

impl QuantizedWeight {
    /// Null weight (all pointers NULL). Used for remote experts under EP.
    pub fn null() -> Self {
        Self {
            weight: DevicePtr::NULL,
            weight_scale: DevicePtr::NULL,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
        }
    }

    /// Whether this weight points to NULL (remote expert placeholder).
    pub fn is_null(&self) -> bool {
        self.weight == DevicePtr::NULL
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
        const GROUP_SIZE: usize = 16;
        let half_k = k / 2;

        // Transpose B_packed: [N, K/2] → [K/2, N] into a NEW GPU allocation.
        let packed_size = n * half_k;
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

        // Transpose B_scale: [N, K/GROUP_SIZE] → [K/GROUP_SIZE, N] into a NEW allocation.
        let num_groups = k / GROUP_SIZE;
        let scale_size = n * num_groups;
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
/// Unlike [`Fp8DenseWeight`] (runtime-quantized from BF16 with per-row scales),
/// this struct represents an FP8 weight that was already quantized on disk
/// with per-row f32 scales. Used by the `w8a16_gemv` LUT-based kernel for
/// native FP8 serving without converting to NVFP4.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Weight {
    /// [N, K] FP8 E4M3 weight bytes on GPU.
    pub weight: DevicePtr,
    /// `[N]` f32 per-row dequant scale on GPU.
    pub row_scale: DevicePtr,
    /// Output dimension (rows).
    pub n: u32,
    /// Input dimension (columns).
    pub k: u32,
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
    /// [K/128, N/128] BF16 transposed block scales on GPU.
    pub scale_t: DevicePtr,
    pub n: u32,
    pub k: u32,
}

impl Fp8Weight {
    /// Transpose this FP8 weight for coalesced prefill GEMM.
    /// Allocates new GPU buffers for `B_t[K,N]` and `scale_t[K/128, N/128]`.
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

        // Allocate transposed scale: [K/128, N/128] × 2 bytes (BF16)
        let n_blocks = n.div_ceil(128);
        let k_blocks = k.div_ceil(128);
        let scale_t = gpu.alloc(k_blocks * n_blocks * 2)?;
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
