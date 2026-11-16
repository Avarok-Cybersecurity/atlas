// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/qlinear.zig:35-47 and 340-452 (AffineN and AffineNKernels: the weight view, packing
// and decode). Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0;
// code written before TensorFold 0.6.0 is also available under the MIT License. See
// TENSORFOLD-ATTRIBUTION.md. Changes for Atlas: Zig to Rust; load-time quantization from BF16 with
// an optional clip search; the `W4Linear` contract of the port sprint (`port::w4`).
//! Responsibility: MLX affine weights of 2, 3, 5, 6 and 8 bits a weight (lane L17c of the port
//! sprint, `port/consumer-opt`): the [`AffineNWeight`] that owns one packed projection on the device,
//! its constructors (from a BF16 weight, from MLX tensors on the device or on the host) and, through
//! [`linear`], its `W4Linear` implementation. Width 4 is the 4-bit lane path of the parent module
//! (lane L17a), which dispatches every other width here through [`quantize_from_bf16`].
//!
//! Nothing here runs unless a W4 route row is on (`dense_affine4_from_bf16`, `gdn_affine4_from_bf16`,
//! `attn_affine4_from_bf16`, lane L23a) and the route asked for a width other than 4 (`--w4-bits`);
//! the kernels exist only in the Ada build (`kernels/ada/common/affine_n_ada.cu`), and on every other
//! target the constructors return an error before any lookup.
//!
//! Contract, per the port sprint's 3.7.2: decode takes 1..=16 rows; a row's output bits do not depend
//! on how many rows ran (one 16-row MMA tile, missing rows zero) nor on whether the split-K form ran
//! (slices summed in slice order either way); prefill (`m > 16`) declines, so the caller dequantizes
//! through [`AffineNWeight`]'s `dequant_bf16` and runs its BF16 GEMM. Group size is 64 only.
//!
//! Scratch (`W4Scratch`, rules in `port::w4`): `xs` holds `rows * k / 64` f32 here (the group sums,
//! written every call; `scratch::xs_bytes` reserves more); the split form also needs `ws` of
//! `scratch_bytes()` bytes and the first [`layout::tickets_bytes`] bytes of `tickets`, zero on entry
//! and left zero by the kernel. With a null or short `ws` or a null `tickets`, decode runs the
//! unsplit form, with the same output bits.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::port::w4::{W4Box, W4Linear, W4Opts};

pub mod kernels;
pub mod layout;
mod linear;
pub mod reference;
mod reference_gemv;
#[cfg(test)]
#[path = "widths_tests.rs"]
mod tests;
#[cfg(test)]
mod tests_mock;

pub use layout::{BITS, GROUP, MAX_DECODE_ROWS, supports};

/// Device allocations made by one constructor: freed on drop (after the stream drains) unless kept.
struct Allocs<'a> {
    gpu: &'a dyn GpuBackend,
    stream: u64,
    ptrs: Vec<DevicePtr>,
}

impl<'a> Allocs<'a> {
    fn new(gpu: &'a dyn GpuBackend, stream: u64) -> Self {
        Self {
            gpu,
            stream,
            ptrs: Vec::new(),
        }
    }

    fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        let p = self.gpu.alloc(bytes.max(4))?;
        self.ptrs.push(p);
        Ok(p)
    }

    fn keep(&mut self, p: DevicePtr) {
        self.ptrs.retain(|q| *q != p);
    }
}

impl Drop for Allocs<'_> {
    fn drop(&mut self) {
        if self.ptrs.is_empty() {
            return;
        }
        let _ = self.gpu.synchronize(self.stream);
        for p in self.ptrs.drain(..) {
            let _ = self.gpu.free(p);
        }
    }
}

/// An MLX affine weight on the device in checkpoint layout: `(n, k * bits / 32)` u32 words and
/// `(n, k / 64)` bf16 scales and biases. Produced by [`quantize_mlx_device`]; the caller frees it.
#[derive(Clone, Copy, Debug)]
pub struct MlxDevice {
    pub words: DevicePtr,
    pub scales: DevicePtr,
    pub biases: DevicePtr,
    pub n: usize,
    pub k: usize,
    pub bits: u8,
}

impl MlxDevice {
    /// Free the three buffers.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.words)?;
        gpu.free(self.scales)?;
        gpu.free(self.biases)
    }
}

/// Quantize bf16 rows `src` `[n][k]` on the device into MLX layout at `bits` (group 64), with the
/// clip search of `clip` extra candidates (at most `reference::MAX_CLIP_SEARCH`, the 4-bit
/// quantizer's meaning of `--w4-clip-search`). Bit-exact with `reference::quantize_mlx`. Fails,
/// freeing everything, on a group with a non-finite entry or no representable scale.
pub fn quantize_mlx_device(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    n: usize,
    k: usize,
    bits: u8,
    clip: u8,
    stream: u64,
) -> Result<MlxDevice> {
    layout::check_shape(n, k, bits)?;
    ensure!(
        clip <= reference::MAX_CLIP_SEARCH,
        "affine widths: clip search {clip} exceeds {}",
        reference::MAX_CLIP_SEARCH
    );
    ensure!(
        (n * layout::groups(k)) < u32::MAX as usize,
        "affine widths: [{n}, {k}] has too many groups for the quantizer's flag"
    );
    let h = kernels::Kernels::resolve(gpu, bits)?;
    let kg = layout::groups(k);
    let mut a = Allocs::new(gpu, stream);
    let words = a.alloc(n * layout::mlx_row_words(k, bits) * 4)?;
    let scales = a.alloc(n * kg * 2)?;
    let biases = a.alloc(n * kg * 2)?;
    let bad = a.alloc(4)?;
    gpu.memset_async(words, 0, n * layout::mlx_row_words(k, bits) * 4, stream)?;
    gpu.memset_async(bad, 0xff, 4, stream)?;
    let q = kernels::Quantize {
        src,
        n,
        k,
        bits,
        clip,
        words,
        scales,
        biases,
        bad,
    };
    kernels::launch_quantize(gpu, &h, &q, stream)?;
    gpu.synchronize(stream)?;
    let mut flag = [0u8; 4];
    gpu.copy_d2h(bad, &mut flag)?;
    let first = u32::from_le_bytes(flag);
    if first != u32::MAX {
        let first = first as usize;
        bail!(
            "affine widths: {bits}-bit quantization of a [{n}, {k}] weight failed at row {} group {} \
             (a non-finite entry or no representable scale)",
            first / kg,
            first % kg
        );
    }
    for p in [words, scales, biases] {
        a.keep(p);
    }
    Ok(MlxDevice {
        words,
        scales,
        biases,
        n,
        k,
        bits,
    })
}

/// One packed projection on the device: tiles of `bits` and the transposed `[k / 64][npad]` bf16
/// scale and bias tables, plus the resolved kernel handles. Implements `port::w4::W4Linear`.
///
/// The device buffers are released by [`AffineNWeight::free_device`] (call once, at teardown) or by
/// the backend's teardown sweep; dropping the value alone does not free them (no backend is held).
#[derive(Debug)]
pub struct AffineNWeight {
    w: DevicePtr,
    scales_t: DevicePtr,
    biases_t: DevicePtr,
    n: usize,
    k: usize,
    bits: u8,
    grid: u32,
    kernels: kernels::Kernels,
    freed: AtomicBool,
}

impl AffineNWeight {
    /// Pack an MLX weight already on the device (checkpoint layout) into tiles. The source buffers
    /// stay the caller's; nothing is freed here.
    pub fn from_mlx_device(gpu: &dyn GpuBackend, m: &MlxDevice, stream: u64) -> Result<Self> {
        let (n, k, bits) = (m.n, m.k, m.bits);
        layout::check_shape(n, k, bits)?;
        let h = kernels::Kernels::resolve(gpu, bits)?;
        let sms = gpu.sm_count()?;
        let mut a = Allocs::new(gpu, stream);
        let w = a.alloc(layout::tile_words(n, k, bits) * 4)?;
        let scales_t = a.alloc(layout::table_elems(n, k) * 2)?;
        let biases_t = a.alloc(layout::table_elems(n, k) * 2)?;
        kernels::launch_pack(gpu, &h, m.words, n, k, bits, w, stream)?;
        kernels::launch_transpose(gpu, &h, m.scales, n, k, scales_t, stream)?;
        kernels::launch_transpose(gpu, &h, m.biases, n, k, biases_t, stream)?;
        gpu.synchronize(stream)?;
        for p in [w, scales_t, biases_t] {
            a.keep(p);
        }
        Ok(Self {
            w,
            scales_t,
            biases_t,
            n,
            k,
            bits,
            grid: layout::gemv_grid(n, bits, sms),
            kernels: h,
            freed: AtomicBool::new(false),
        })
    }

    /// Quantize a bf16 weight `src` `[n][k]` and pack it; the MLX intermediate is freed, `src` is not.
    pub fn from_bf16(
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        n: usize,
        k: usize,
        bits: u8,
        clip: u8,
        stream: u64,
    ) -> Result<Self> {
        let m = quantize_mlx_device(gpu, src, n, k, bits, clip, stream)?;
        let packed = Self::from_mlx_device(gpu, &m, stream);
        let freed = m.free(gpu);
        let packed = packed?;
        freed?;
        Ok(packed)
    }

    /// Output features.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Input features.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Bits a weight.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// The tiles (`layout::tile_words` u32 words).
    pub fn tiles_ptr(&self) -> DevicePtr {
        self.w
    }

    /// The transposed scale table (`[k / 64][npad]` bf16).
    pub fn scales_ptr(&self) -> DevicePtr {
        self.scales_t
    }

    /// The transposed bias table (`[k / 64][npad]` bf16).
    pub fn biases_ptr(&self) -> DevicePtr {
        self.biases_t
    }

    /// Release the device buffers. Idempotent; any later decode fails instead of reading freed memory.
    pub fn free_device(&self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.freed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        gpu.free(self.w)?;
        gpu.free(self.scales_t)?;
        gpu.free(self.biases_t)
    }

    fn live(&self) -> Result<()> {
        ensure!(
            !self.freed.load(Ordering::Acquire),
            "affine widths: weight used after free_device"
        );
        Ok(())
    }
}

/// The 3.7.2 constructor for every width but 4 (lane L17a's `quantize_from_bf16` dispatches here):
/// quantize bf16 `src` `[n][k]` at `opts.bits` with `opts.clip_search`, group 64 only. `src` is not
/// freed. `opts.int8_prefill` has no effect: the int8 prompt GEMM serves 4-bit codes only, and this
/// weight's prefill declines (the caller's BF16 fallback runs).
pub fn quantize_from_bf16(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    n: usize,
    k: usize,
    opts: &W4Opts,
    stream: u64,
) -> Result<W4Box> {
    if !layout::supports(opts.bits, opts.group) {
        bail!(
            "affine widths: {}-bit group {} is not served (bits {BITS:?}, group {GROUP} only)",
            opts.bits,
            opts.group
        );
    }
    if opts.int8_prefill {
        tracing::debug!(
            "affine widths: int8 prefill is 4-bit only; {}-bit weights prefill through BF16",
            opts.bits
        );
    }
    let w = AffineNWeight::from_bf16(gpu, src, n, k, opts.bits, opts.clip_search, stream)?;
    Ok(Arc::new(w))
}

/// Pack an MLX checkpoint tensor already on the device (`words` `[n][k * bits / 32]`, `scales` and
/// `biases` `[n][k / 64]` bf16). The sources stay the caller's.
pub fn from_mlx(
    gpu: &dyn GpuBackend,
    words: DevicePtr,
    scales: DevicePtr,
    biases: DevicePtr,
    n: usize,
    k: usize,
    bits: u8,
    stream: u64,
) -> Result<W4Box> {
    let m = MlxDevice {
        words,
        scales,
        biases,
        n,
        k,
        bits,
    };
    Ok(Arc::new(AffineNWeight::from_mlx_device(gpu, &m, stream)?))
}

/// Upload an MLX checkpoint tensor from host memory and pack it; the upload buffers are freed.
pub fn from_host_mlx(
    gpu: &dyn GpuBackend,
    words: &[u32],
    scales: &[u16],
    biases: &[u16],
    n: usize,
    k: usize,
    bits: u8,
    stream: u64,
) -> Result<W4Box> {
    layout::check_shape(n, k, bits)?;
    let kg = layout::groups(k);
    ensure!(
        words.len() == n * layout::mlx_row_words(k, bits)
            && scales.len() == n * kg
            && biases.len() == n * kg,
        "affine widths: MLX tensor sizes do not match [{n}, {k}] at {bits} bits"
    );
    let mut a = Allocs::new(gpu, stream);
    let bytes_w: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes_s: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes_b: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let m = MlxDevice {
        words: a.alloc(bytes_w.len())?,
        scales: a.alloc(bytes_s.len())?,
        biases: a.alloc(bytes_b.len())?,
        n,
        k,
        bits,
    };
    gpu.copy_h2d(&bytes_w, m.words)?;
    gpu.copy_h2d(&bytes_s, m.scales)?;
    gpu.copy_h2d(&bytes_b, m.biases)?;
    Ok(Arc::new(AffineNWeight::from_mlx_device(gpu, &m, stream)?))
}

/// Free the device buffers of a widths `W4Box` (the counterpart of the 4-bit path's
/// `affine::release`). `Ok(false)` when `w` is not an [`AffineNWeight`]; a second call frees nothing.
/// The caller guarantees no launch still reads the weight.
pub fn release(gpu: &dyn GpuBackend, w: &dyn W4Linear) -> Result<bool> {
    let Some(a) = w.as_any().and_then(|a| a.downcast_ref::<AffineNWeight>()) else {
        return Ok(false);
    };
    a.free_device(gpu)?;
    Ok(true)
}

/// True when this target built the widths kernels (Ada); never issues a lookup elsewhere.
pub fn available(gpu: &dyn GpuBackend) -> bool {
    kernels::available(gpu)
}
