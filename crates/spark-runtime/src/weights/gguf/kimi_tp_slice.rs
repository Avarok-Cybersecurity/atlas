// SPDX-License-Identifier: AGPL-3.0-only
//! TP byte slicing for K3 GGUF keep-packed tensors (Q8_0 / IQ2_XS / IQ3_XXS).

use anyhow::{Result, bail, ensure};

pub const IQ2_QK: usize = 256;
pub const IQ2_BLOCK: usize = 74;
pub const IQ3_BLOCK: usize = 98;
pub const Q8_QK: usize = 32;
pub const Q8_BLOCK: usize = 34;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackKind {
    Iq2Xs,
    Iq3Xxs,
    Q8_0,
}

impl PackKind {
    pub fn qk(self) -> usize {
        match self {
            Self::Iq2Xs | Self::Iq3Xxs => IQ2_QK,
            Self::Q8_0 => Q8_QK,
        }
    }
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Iq2Xs => IQ2_BLOCK,
            Self::Iq3Xxs => IQ3_BLOCK,
            Self::Q8_0 => Q8_BLOCK,
        }
    }
}

/// Row-parallel (Atlas ColumnParallel): split output rows. Contiguous & block-safe
/// when each row's K is a multiple of the quant block.
pub fn slice_rows_packed(
    raw: &[u8],
    n: usize,
    k: usize,
    rank: usize,
    world: usize,
    kind: PackKind,
) -> Result<(Vec<u8>, usize, usize)> {
    ensure!(world > 0 && rank < world, "bad TP");
    ensure!(n % world == 0, "rows {n} not divisible by TP{world}");
    ensure!(k % kind.qk() == 0, "K={k} not multiple of {}", kind.qk());
    let local_n = n / world;
    let row_bytes = (k / kind.qk()) * kind.block_bytes();
    ensure!(raw.len() >= n * row_bytes, "row-packed raw too small");
    let start = rank * local_n * row_bytes;
    let end = start + local_n * row_bytes;
    Ok((raw[start..end].to_vec(), local_n, k))
}

/// Column-parallel (Atlas RowParallel): split K. For Q8 (qk=32) local K is
/// block-aligned at TP8×7168. For IQ2 (qk=256) local K=384 is NOT aligned —
/// store covering blocks; shape stays mathematical `(n, k_local)`.
pub fn slice_cols_packed(
    raw: &[u8],
    n: usize,
    k: usize,
    rank: usize,
    world: usize,
    kind: PackKind,
) -> Result<(Vec<u8>, usize, usize)> {
    ensure!(world > 0 && rank < world, "bad TP");
    ensure!(k % world == 0, "cols {k} not divisible by TP{world}");
    let k_local = k / world;
    let qk = kind.qk();
    let blk = kind.block_bytes();
    let full_blocks = k / qk;
    let row_bytes_full = full_blocks * blk;
    ensure!(raw.len() >= n * row_bytes_full, "col-packed raw too small");

    if k_local % qk == 0 {
        // Block-aligned column memcpy.
        let local_blocks = k_local / qk;
        let row_bytes = local_blocks * blk;
        let mut out = Vec::with_capacity(n * row_bytes);
        let col0 = rank * local_blocks * blk;
        for row in 0..n {
            let base = row * row_bytes_full + col0;
            out.extend_from_slice(&raw[base..base + row_bytes]);
        }
        return Ok((out, n, k_local));
    }

    // IQ2 cover: store blocks [floor(k0/qk), ceil(k1/qk)).
    ensure!(
        matches!(kind, PackKind::Iq2Xs | PackKind::Iq3Xxs),
        "non-aligned column split only implemented for IQ2_XS/IQ3_XXS (K_local={k_local})"
    );
    let k0 = rank * k_local;
    let k1 = k0 + k_local;
    let b0 = k0 / qk;
    let b1 = k1.div_ceil(qk);
    let n_cover = b1 - b0;
    let row_bytes = n_cover * blk;
    let mut out = Vec::with_capacity(n * row_bytes);
    for row in 0..n {
        let base = row * row_bytes_full + b0 * blk;
        out.extend_from_slice(&raw[base..base + row_bytes]);
    }
    Ok((out, n, k_local))
}

pub fn pack_kind_from_ggml(id: u32) -> Result<PackKind> {
    match id {
        8 => Ok(PackKind::Q8_0),
        17 => Ok(PackKind::Iq2Xs),
        18 => Ok(PackKind::Iq3Xxs),
        other => bail!("kimi TP packed: unsupported ggml type id {other}"),
    }
}
