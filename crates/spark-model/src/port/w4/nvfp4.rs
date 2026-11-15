// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: [`Nvfp4Linear`], the existing NVFP4 weight (`QuantizedWeight`: packed E2M1 `[N, K/2]`,
//! E4M3 per-16 scales `[N, K/16]`, one F32 tensor scale) and its `w4a16_*` kernels, wrapped as a `W4Linear`,
//! so the 4-bit dispatch of lanes L21/L22 can serve it like any other packed
//! format. Lane L23b. No new kernel and no new weight format: every launch below is an existing launcher in
//! `layers/ops` over a kernel every CUDA target already compiles (`kernels/<hw>/common/w4a16_gemv.cu`,
//! `w4a16_gemm.cu`, `dequant_nvfp4_bf16.cu`; on Ada these are the gb10 files, software E2M1 dequant).
//!
//! | rows | kernel |
//! |---|---|
//! | 1 | `w4a16_gemv` |
//! | 2, 3 | `w4a16_gemv_batch2`, `w4a16_gemv_batch3` |
//! | 4..=8 | the narrowest resolved `w4a16_gemv_batch{4..8}` tier (`W4a16BatchmTiers`) |
//! | 9..=16 | `w4a16_gemv_batch16` (absent: `max_decode_rows` is 8) |
//! | prefill (m > max) | `w4a16_gemm` (64 x 64 tiles over the `[N, K/2]` layout); absent or K not a multiple of 64: declined |
//! | dequant | `dequant_nvfp4_to_bf16` |
//!
//! ROW INVARIANCE, the contract's one hard rule, holds by construction: `w4a16_gemv_batch{2..32}` are thin
//! instantiations of `w4a16_gemv_batchm_impl`, whose per-row accumulation order is identical to `w4a16_gemv`
//! (`w4a16_gemv.cu`, the note above `w4a16_gemv_batchm_impl`; enforced on the card by
//! `examples/w4a16_batch_bitparity_microtest.rs`), and the rt2 width-8 tier is bit-exact against batch8.
//! The load-time self-check (`weight_loader/qwen35_dense/affine4_selfcheck.rs`) re-proves it on the real
//! weights.
//!
//! OWNERSHIP. The wrapper never frees: the loader that built the weight adopts its two buffers into the
//! store's derived ledger (`DerivedStore::adopt`, #736), which releases them with the model.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{W4Box, W4Format, W4Linear, W4Scratch};
use crate::layers::ops;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::QuantizedWeight;

/// Module of the NVFP4 GEMV family.
pub const GEMV_MODULE: &str = "w4a16_gemv";
/// Module of the NVFP4 tile GEMM (`w4a16_gemm`).
pub const GEMM_MODULE: &str = "w4a16";
/// Module of the NVFP4 -> BF16 dequant.
pub const DEQUANT_MODULE: &str = "dequant_nvfp4_bf16";
/// Elements per E4M3 group scale.
pub const GROUP: usize = 16;
/// The prompt GEMM's K tile (`w4a16_gemm` walks K in whole 64-wide steps of four 16-groups).
pub const GEMM_K_TILE: usize = 64;

/// Bytes one NVFP4 `[n, k]` weight occupies: `n*k/2` packed codes plus `n*k/16` group scales (the two
/// buffers `quantize_to_nvfp4` allocates; same arithmetic as `fp8_residency::nvfp4_bytes`).
pub fn device_bytes_of(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / GROUP
}

/// The resolved kernel handles one load shares across every NVFP4 weight. A zero handle is "this build
/// does not carry it".
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4Kernels {
    pub gemv: KernelHandle,
    pub batch2: KernelHandle,
    pub batch3: KernelHandle,
    pub tiers: W4a16BatchmTiers,
    pub batch16: KernelHandle,
    pub gemm: KernelHandle,
    pub dequant: KernelHandle,
}

impl Nvfp4Kernels {
    /// Resolve every handle. A module this build does not carry issues no lookup at all
    /// (`try_target_kernel`), so the boot audit sees no new failed lookup on any target.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        let k = |m: &str, f: &str| crate::layers::try_target_kernel(gpu, m, f);
        Self {
            gemv: k(GEMV_MODULE, "w4a16_gemv"),
            batch2: k(GEMV_MODULE, "w4a16_gemv_batch2"),
            batch3: k(GEMV_MODULE, "w4a16_gemv_batch3"),
            tiers: if gpu.has_module(GEMV_MODULE) {
                W4a16BatchmTiers::resolve(gpu)
            } else {
                W4a16BatchmTiers::default()
            },
            batch16: k(GEMV_MODULE, "w4a16_gemv_batch16"),
            gemm: k(GEMM_MODULE, "w4a16_gemm"),
            dequant: k(DEQUANT_MODULE, "dequant_nvfp4_to_bf16"),
        }
    }

    /// Every kernel the decode window 1..=4 and the dequant fallback need is present.
    pub fn decode_ready(&self) -> bool {
        self.gemv.0 != 0
            && self.batch2.0 != 0
            && self.batch3.0 != 0
            && self.tiers.has_base()
            && self.dequant.0 != 0
    }

    /// The widest row count [`Nvfp4Linear::decode`] serves with these kernels (0 when not ready).
    pub fn max_decode_rows(&self) -> usize {
        if !self.decode_ready() {
            0
        } else if self.batch16.0 != 0 {
            16
        } else if self.tiers.width(8) == Some(8) {
            8
        } else {
            4
        }
    }
}

/// An NVFP4 weight as a `W4Linear`.
#[derive(Debug)]
pub struct Nvfp4Linear {
    q: QuantizedWeight,
    n: usize,
    k: usize,
    kernels: Nvfp4Kernels,
}

/// `port::w4::nvfp4::from_quantized` (3.7.2). The plan's one-argument form cannot be built:
/// `QuantizedWeight` carries no shape and no kernel handles, so both are passed in. Refuses a NULL weight,
/// a K that is not a whole number of 16-groups, a per-row scale2 (these kernels read the scalar
/// `weight_scale_2` only) and a kernel set that cannot serve rows 1..=4.
pub fn from_quantized(
    q: QuantizedWeight,
    n: usize,
    k: usize,
    kernels: Nvfp4Kernels,
) -> Result<W4Box> {
    Ok(std::sync::Arc::new(Nvfp4Linear::new(q, n, k, kernels)?))
}

impl Nvfp4Linear {
    pub fn new(q: QuantizedWeight, n: usize, k: usize, kernels: Nvfp4Kernels) -> Result<Self> {
        ensure!(
            !q.is_null() && !q.weight_scale.is_null(),
            "nvfp4: NULL weight [{n}x{k}]"
        );
        ensure!(n > 0 && k > 0, "nvfp4: empty weight [{n}x{k}]");
        ensure!(
            k.is_multiple_of(GROUP),
            "nvfp4: K {k} is not a multiple of the {GROUP}-element group"
        );
        ensure!(
            n <= u32::MAX as usize && k <= u32::MAX as usize,
            "nvfp4: [{n}x{k}] exceeds the kernels' u32 extents"
        );
        ensure!(
            !q.has_per_row_scale2(),
            "nvfp4: a per-row scale2 weight needs the _prs kernels, which this wrapper does not drive"
        );
        ensure!(
            kernels.decode_ready(),
            "nvfp4: the w4a16 GEMV family or the dequant kernel is missing from this build"
        );
        Ok(Self { q, n, k, kernels })
    }

    /// The wrapped weight (device pointers and the tensor scale).
    pub fn quantized(&self) -> &QuantizedWeight {
        &self.q
    }

    fn nk(&self) -> (u32, u32) {
        (self.n as u32, self.k as u32)
    }
}

impl W4Linear for Nvfp4Linear {
    fn format(&self) -> W4Format {
        W4Format::Nvfp4
    }

    fn n(&self) -> usize {
        self.n
    }

    fn k(&self) -> usize {
        self.k
    }

    fn device_bytes(&self) -> usize {
        device_bytes_of(self.n, self.k)
    }

    fn max_decode_rows(&self) -> usize {
        self.kernels.max_decode_rows()
    }

    /// The GEMV family keeps its reduction in shared memory: no workspace.
    fn scratch_bytes(&self, _max_rows: usize) -> usize {
        0
    }

    fn decode(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        _s: &W4Scratch,
        stream: u64,
    ) -> Result<()> {
        let (n, k) = self.nk();
        let kk = &self.kernels;
        let q = &self.q;
        match rows {
            1 => ops::w4a16_gemv(gpu, kk.gemv, x, q, out, n, k, stream),
            2 => ops::w4a16_gemv_batch2(gpu, kk.batch2, x, q, out, n, k, stream),
            3 => ops::w4a16_gemv_batch3(gpu, kk.batch3, x, q, out, n, k, stream),
            4..=16 if rows <= self.max_decode_rows() => {
                // The narrowest resolved tier, else batch16: every width is the same M-guarded template,
                // so the choice changes time, never bits.
                let tier = if rows <= 8 {
                    kk.tiers.kernel(rows as u32)
                } else {
                    KernelHandle(0)
                };
                let h = if tier.0 != 0 { tier } else { kk.batch16 };
                ensure!(
                    h.0 != 0,
                    "nvfp4: no w4a16_gemv_batch tier covers {rows} rows"
                );
                ops::w4a16_gemv_batchm(gpu, h, x, q, out, rows as u32, n, k, stream)
            }
            _ => bail!(
                "nvfp4: decode of {rows} rows outside 1..={} for [{}x{}]",
                self.max_decode_rows(),
                self.n,
                self.k
            ),
        }
    }

    fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        m: usize,
        out: DevicePtr,
        _s: &W4Scratch,
        stream: u64,
    ) -> Result<bool> {
        ensure!(
            m > self.max_decode_rows(),
            "nvfp4: prefill of {m} rows is inside the decode window 1..={}",
            self.max_decode_rows()
        );
        if self.kernels.gemm.0 == 0 || !self.k.is_multiple_of(GEMM_K_TILE) || m > u32::MAX as usize
        {
            return Ok(false);
        }
        let (n, k) = self.nk();
        ops::w4a16_gemm(
            gpu,
            self.kernels.gemm,
            x,
            &self.q,
            out,
            m as u32,
            n,
            k,
            stream,
        )?;
        Ok(true)
    }

    fn dequant_bf16(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()> {
        let (n, k) = self.nk();
        // The quantizer's scale2 is a direct multiplier (ModelOpt convention), which is the kernel's
        // `combined_global` (`weight_map/fp8_lut.rs::dequant_nvfp4_to_bf16`).
        KernelLaunch::new(gpu, self.kernels.dequant)
            .grid([n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.q.weight)
            .arg_ptr(self.q.weight_scale)
            .arg_ptr(out)
            .arg_f32(self.q.weight_scale_2)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
#[path = "nvfp4_tests.rs"]
mod tests;
