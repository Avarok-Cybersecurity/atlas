// SPDX-License-Identifier: AGPL-3.0-only
// After ExLlamaV3 (turboderp-org/exllamav3), MIT, Copyright (c) 2025 Turboderp, via TensorFold
// zig/src/core/exl3_load.zig:15-24, 52-59 (the group and its parts) and zig/src/core/exl3_layout.zig:69-93
// (a rank's part) (Apache-2.0, ashhart/TensorFold, commit 3885dce; Copyright 2026 TensorFold
// contributors). Changes: one owned host tensor type with validated invariants; packed sign words are
// expanded to fp16 at load; parts are materialized as sub-tensors. See EXL3-ATTRIBUTION.md.

//! Responsibility: [`Exl3Tensor`], one EXL3 linear layer held on the host exactly as the checkpoint
//! stores it (stored-order trellis words, fp16 `suh` / `svh`, optional fp16 bias), with its geometry,
//! the byte views a device upload needs, the strips re-layout, a tensor-parallel part, and the FP64
//! reference decode. This is the type lane L20b's `port::exl3::from_exl3_tensors` consumes.

use anyhow::{Result, bail};

use super::dequant::{self, HADAMARD_DIM};
use super::format::{self, Codebook, TILE_DIM, Width};
use super::layout::{self, Group, Split};

/// How the checkpoint stored the input and output scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleForm {
    /// fp16 `suh` `[K]` and `svh` `[N]`.
    Fp16,
    /// Packed sign words `su` `[K/16]` and `sv` `[N/16]` of older checkpoints, expanded to +1 / -1.
    Signs,
}

/// One EXL3 linear layer `y = x @ W + bias`, `W [K, N]`, on the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exl3Tensor {
    /// The tensor prefix (`model.layers.0.mlp.down_proj`).
    pub prefix: String,
    /// The codebook its states decode with.
    pub codebook: Codebook,
    /// Its bit width.
    pub width: Width,
    /// Inputs (a multiple of 128).
    pub k: usize,
    /// Outputs (a multiple of 128).
    pub n: usize,
    /// Trellis int16 words in stored order `[K/16, N/16, 16 * bits]`, unchanged from the file.
    pub trellis: Vec<i16>,
    /// fp16 input scales `[K]` (bit patterns).
    pub suh: Vec<u16>,
    /// fp16 output scales `[N]` (bit patterns).
    pub svh: Vec<u16>,
    /// fp16 bias `[N]` (bit patterns), if the layer has one.
    pub bias: Option<Vec<u16>>,
    /// How the scales were stored.
    pub scale_form: ScaleForm,
}

fn le_bytes(words: impl Iterator<Item = [u8; 2]>, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    for w in words {
        out.extend_from_slice(&w);
    }
    out
}

impl Exl3Tensor {
    /// A tensor from its parts, with every invariant of the format checked: K and N positive
    /// multiples of 128, the trellis length `(K/16) (N/16) (16 bits)`, `suh` `[K]`, `svh` `[N]`, the
    /// bias `[N]`, and half widths only with `mul1`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prefix: String,
        codebook: Codebook,
        width: Width,
        k: usize,
        n: usize,
        trellis: Vec<i16>,
        suh: Vec<u16>,
        svh: Vec<u16>,
        bias: Option<Vec<u16>>,
        scale_form: ScaleForm,
    ) -> Result<Self> {
        if k == 0 || n == 0 || !k.is_multiple_of(HADAMARD_DIM) || !n.is_multiple_of(HADAMARD_DIM) {
            bail!(
                "EXL3 {prefix}: [{k}, {n}] is not a positive multiple of {HADAMARD_DIM} each way"
            );
        }
        if width.is_half() && codebook != Codebook::Mul1 {
            bail!(
                "EXL3 {prefix}: {} bits is a half width, which only the mul1 codebook has",
                width.bits()
            );
        }
        format::check_trellis(width, &trellis, k / TILE_DIM, n / TILE_DIM)?;
        if suh.len() != k || svh.len() != n {
            bail!(
                "EXL3 {prefix}: suh has {} and svh {} values for [{k}, {n}]",
                suh.len(),
                svh.len()
            );
        }
        if let Some(b) = &bias
            && b.len() != n
        {
            bail!("EXL3 {prefix}: bias has {} values for N = {n}", b.len());
        }
        Ok(Self {
            prefix,
            codebook,
            width,
            k,
            n,
            trellis,
            suh,
            svh,
            bias,
            scale_form,
        })
    }

    /// Tiles along K.
    pub fn tiles_k(&self) -> usize {
        self.k / TILE_DIM
    }

    /// Tiles along N.
    pub fn tiles_n(&self) -> usize {
        self.n / TILE_DIM
    }

    /// int16 words a tile.
    pub fn tile_words(&self) -> usize {
        self.width.tile_words()
    }

    /// Half bits a value (the kernels' K2).
    pub fn k2(&self) -> u32 {
        u32::from(self.width.halves())
    }

    /// Bits a value times 100 (`W4Format::Exl3 { bpw_x100 }`).
    pub fn bpw_x100(&self) -> u16 {
        self.width.bpw_x100()
    }

    /// The layout geometry of the whole tensor.
    pub fn group(&self) -> Group {
        Group {
            k_tiles: self.tiles_k() as u32,
            n_tiles: self.tiles_n() as u32,
            words: self.tile_words() as u32,
        }
    }

    /// The int16 words of tile `(tk, tn)`.
    pub fn tile(&self, tk: usize, tn: usize) -> &[i16] {
        let words = self.tile_words();
        let at = (tk * self.tiles_n() + tn) * words;
        &self.trellis[at..at + words]
    }

    /// Trellis bytes.
    pub fn trellis_bytes(&self) -> usize {
        self.trellis.len() * 2
    }

    /// Bytes a device copy of the layer takes: the trellis, both scale vectors and the bias.
    pub fn device_bytes(&self) -> usize {
        self.trellis_bytes() + 2 * (self.k + self.n) + self.bias.as_ref().map_or(0, |b| 2 * b.len())
    }

    /// The trellis as little-endian bytes, stored order (the upload image).
    pub fn trellis_le_bytes(&self) -> Vec<u8> {
        le_bytes(
            self.trellis.iter().map(|w| w.to_le_bytes()),
            self.trellis.len(),
        )
    }

    /// The trellis in the strips order `[N/128, K/16, 8, words]` as little-endian bytes.
    pub fn strips_le_bytes(&self) -> Result<Vec<u8>> {
        let strips = layout::strips(
            &self.trellis,
            self.tiles_k(),
            self.tiles_n(),
            self.tile_words(),
        )?;
        Ok(le_bytes(
            strips.iter().map(|w| w.to_le_bytes()),
            strips.len(),
        ))
    }

    /// `suh` as little-endian fp16 bytes.
    pub fn suh_le_bytes(&self) -> Vec<u8> {
        le_bytes(self.suh.iter().map(|w| w.to_le_bytes()), self.suh.len())
    }

    /// `svh` as little-endian fp16 bytes.
    pub fn svh_le_bytes(&self) -> Vec<u8> {
        le_bytes(self.svh.iter().map(|w| w.to_le_bytes()), self.svh.len())
    }

    /// The bias as little-endian fp16 bytes.
    pub fn bias_le_bytes(&self) -> Option<Vec<u8>> {
        self.bias
            .as_ref()
            .map(|b| le_bytes(b.iter().map(|w| w.to_le_bytes()), b.len()))
    }

    /// `W_q` `[K, N]` (fp16 bit patterns, rotated domain).
    pub fn unpack_wq(&self) -> Result<Vec<u16>> {
        format::unpack(
            self.width,
            self.codebook,
            &self.trellis,
            self.tiles_k(),
            self.tiles_n(),
        )
    }

    /// The whole `W` `[K, N]` in FP64 (`8 K N` bytes).
    pub fn dequantize_f64(&self) -> Result<Vec<f64>> {
        dequant::dequantize(
            self.width,
            self.codebook,
            &self.trellis,
            self.tiles_k(),
            self.tiles_n(),
            &self.suh,
            &self.svh,
        )
    }

    /// The 128x128 block `(kb, nb)` of `W` in FP64.
    pub fn dequantize_block(&self, kb: usize, nb: usize) -> Result<Vec<f64>> {
        dequant::dequantize_block(
            self.width,
            self.codebook,
            &self.trellis,
            self.tiles_k(),
            self.tiles_n(),
            &self.suh,
            &self.svh,
            kb,
            nb,
        )
    }

    /// `y = x @ W + bias` in FP64 for one row `x` `[K]`.
    pub fn matvec_f64(&self, x: &[f64]) -> Result<Vec<f64>> {
        dequant::matvec(
            self.width,
            self.codebook,
            &self.trellis,
            self.tiles_k(),
            self.tiles_n(),
            &self.suh,
            &self.svh,
            self.bias.as_deref(),
            x,
        )
    }

    /// Rank `rank` of `world`'s part of the layer under `split`, as its own tensor. Columns take N
    /// tiles, `svh` and the bias; rows take K tiles and `suh`, and only rank 0 keeps the bias (the
    /// ranks' partial outputs are added once).
    pub fn part(&self, split: Split, rank: u32, world: u32) -> Result<Self> {
        let p = self.group().part(split, rank, world)?;
        let elem = |bytes: u64| (bytes / 2) as usize;
        let mut trellis = Vec::with_capacity(elem(p.trellis.bytes()));
        for i in 0..p.trellis.count {
            let start = elem(p.trellis.at(i));
            trellis.extend_from_slice(&self.trellis[start..start + elem(p.trellis.len)]);
        }
        let slice = |v: &[u16], s: layout::Span| v[elem(s.offset)..elem(s.offset + s.len)].to_vec();
        let bias = match split {
            Split::Whole => self.bias.clone(),
            Split::Col => self.bias.as_ref().map(|b| slice(b, p.svh)),
            Split::Row => self.bias.clone().filter(|_| rank == 0),
        };
        Self::new(
            self.prefix.clone(),
            self.codebook,
            self.width,
            p.group.k() as usize,
            p.group.n() as usize,
            trellis,
            slice(&self.suh, p.suh),
            slice(&self.svh, p.svh),
            bias,
            self.scale_form,
        )
    }
}
