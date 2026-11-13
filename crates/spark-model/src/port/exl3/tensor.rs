// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the route entry of contract 3.7.2, [`from_exl3_tensors`]: one reader tensor
//! (`spark_runtime::weights::exl3::Exl3Tensor`, lane L20a) to a `W4Box`, behind the `weights_exl3` row;
//! and [`TensorBytes`] / [`TensorBytes::parts`], the only place that knows the reader's field names.
//!
//! The reader holds the trellis as stored (`Vec<i16>`, `[K/16, N/16, 16 bits]`), fp16 `suh` / `svh`
//! with packed sign words already expanded (`ScaleForm` records the stored form), and an optional
//! bias. Everything is checked again by [`Exl3Parts::check`] (and a bias refused), so a reader change
//! shows up as a startup error, never as a wrong answer.

use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::exl3::{Codebook as ReaderCodebook, Exl3Tensor};

use super::format::{Exl3Codebook, Exl3Parts, Exl3Scales};
use super::kernels;
use super::w4linear::Exl3Linear;
use crate::port::w4::W4Box;

/// The reader's codebook as this module's.
pub fn codebook_of(c: ReaderCodebook) -> Exl3Codebook {
    match c {
        ReaderCodebook::Inst3 => Exl3Codebook::Inst3,
        ReaderCodebook::Mcg => Exl3Codebook::Mcg,
        ReaderCodebook::Mul1 => Exl3Codebook::Mul1,
    }
}

/// The upload image of one reader tensor: little-endian trellis and scale bytes.
pub struct TensorBytes {
    pub k: usize,
    pub n: usize,
    pub halves: u16,
    pub codebook: Exl3Codebook,
    pub trellis: Vec<u8>,
    pub suh: Vec<u8>,
    pub svh: Vec<u8>,
    pub bias: Option<Vec<u8>>,
}

impl TensorBytes {
    /// The bytes of `t` (the reader's own little-endian views).
    pub fn of(t: &Exl3Tensor) -> Self {
        Self {
            k: t.k,
            n: t.n,
            halves: t.width.halves(),
            codebook: codebook_of(t.codebook),
            trellis: t.trellis_le_bytes(),
            suh: t.suh_le_bytes(),
            svh: t.svh_le_bytes(),
            bias: t.bias_le_bytes(),
        }
    }

    /// A borrowed [`Exl3Parts`] view (the scales are fp16: the reader expanded any sign words).
    pub fn parts(&self) -> Exl3Parts<'_> {
        Exl3Parts {
            k: self.k,
            n: self.n,
            halves: self.halves,
            codebook: self.codebook,
            trellis: &self.trellis,
            su: Exl3Scales::F16(&self.suh),
            sv: Exl3Scales::F16(&self.svh),
            bias: self.bias.as_deref(),
        }
    }
}

/// True when this process serves EXL3 weights: the `weights_exl3` row is on and the build carries the
/// Ada EXL3 modules (checked without a kernel lookup). A loader route asks this before reading tensors.
pub fn route_enabled(gpu: &dyn GpuBackend) -> bool {
    crate::layers::ops::port_rows::weights_exl3::row() && kernels::available(gpu)
}

/// Contract 3.7.2: one EXL3 tensor to a `W4Box` on the Ada kernels. Refused while the `weights_exl3`
/// row is off (every target's declaration; `AVAROK_WEIGHTS_EXL3=1` turns it on), on a build without
/// the Ada EXL3 modules, and for a tensor with a bias, so no target changes behavior until the row's
/// receipt flips it.
pub fn from_exl3_tensors(gpu: &dyn GpuBackend, t: &Exl3Tensor, stream: u64) -> Result<W4Box> {
    ensure!(
        crate::layers::ops::port_rows::weights_exl3::row(),
        "exl3: the weights_exl3 row is off on this target (AVAROK_WEIGHTS_EXL3=1 turns it on)"
    );
    let bytes = TensorBytes::of(t);
    Exl3Linear::upload(gpu, &bytes.parts(), stream)
        .map_err(|e| e.context(format!("exl3: {} [{} -> {}]", t.prefix, t.k, t.n)))
}
