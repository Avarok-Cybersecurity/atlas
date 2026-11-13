// SPDX-License-Identifier: AGPL-3.0-only
// After ExLlamaV3 (turboderp-org/exllamav3), MIT, Copyright (c) 2025 Turboderp, via TensorFold
// zig/src/core/exl3_format.zig:136-187 (Apache-2.0, ashhart/TensorFold, commit 3885dce; Copyright 2026
// TensorFold contributors). Changes: translated from Zig to Rust; the 128-point rotation runs as a
// fast Walsh-Hadamard butterfly instead of the naive matrix product (same matrix, different FP64
// summation order; the tests compare the two); added the block-local dequantization and the
// factorized FP64 matvec. See EXL3-ATTRIBUTION.md.

//! Responsibility: the FP64 reference arithmetic of an EXL3 layer: the 128-point Hadamard rotations,
//! full and block dequantization of `W`, and `y = x @ W + bias` in its factorized form. This is the
//! CPU oracle the EXL3 kernels (lane L20b) are checked against; nothing here runs on the serve path.
//!
//! The layer (TensorFold `docs/recipes/exl3.md`, "The layer"): with `H` the 128x128 Sylvester Hadamard
//! matrix scaled by `1/sqrt(128)`, applied to each block of 128 inputs or outputs,
//! `W = diag(suh) @ H_K @ W_q @ H_N @ diag(svh)`, so
//! `y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias`. Because `H_K` and `H_N` are block diagonal,
//! the 128x128 block `(kb, nb)` of `W` depends only on the tiles of that block and on
//! `suh[128 kb ..][..128]`, `svh[128 nb ..][..128]` ([`dequantize_block`]).

use anyhow::{Result, bail};

use super::format::{
    Codebook, TILE_DIM, TILE_PLACEMENT, Width, check_trellis, codebook_table, codebook_value,
    tile_states, unpack,
};
use super::fp16;

/// Edge of a Hadamard block.
pub const HADAMARD_DIM: usize = 128;

/// Which axis of a row-major `[rows, cols]` matrix a rotation runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Along the rows index (each column's 128-row blocks): `H @ X`, the input side `H_K`.
    Rows,
    /// Along the columns index (each row's 128-column blocks): `X @ H`, the output side `H_N`.
    Cols,
}

/// One 128-point normalized Walsh-Hadamard transform in place: `v <- H v`, `H[i][j] =
/// (-1)^popcount(i & j) / sqrt(128)` (Sylvester order; `H` is symmetric, so this is also `v H`).
pub fn hadamard_128(v: &mut [f64]) {
    debug_assert_eq!(v.len(), HADAMARD_DIM);
    let mut h = 1;
    while h < HADAMARD_DIM {
        let mut i = 0;
        while i < HADAMARD_DIM {
            for j in i..i + h {
                let a = v[j];
                let b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let inv = 1.0 / (HADAMARD_DIM as f64).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Rotates each 128-value block along `axis` of the row-major `[rows, cols]` FP64 matrix `x`.
pub fn rotate(x: &mut [f64], rows: usize, cols: usize, axis: Axis) -> Result<()> {
    if rows.checked_mul(cols) != Some(x.len()) {
        bail!(
            "EXL3 rotate: {} values for a [{rows}, {cols}] matrix",
            x.len()
        );
    }
    match axis {
        Axis::Cols => {
            if !cols.is_multiple_of(HADAMARD_DIM) {
                bail!("EXL3 rotate: {cols} columns is not a multiple of {HADAMARD_DIM}");
            }
            for block in x.chunks_exact_mut(HADAMARD_DIM) {
                hadamard_128(block);
            }
        }
        Axis::Rows => {
            if !rows.is_multiple_of(HADAMARD_DIM) {
                bail!("EXL3 rotate: {rows} rows is not a multiple of {HADAMARD_DIM}");
            }
            let mut scratch = [0f64; HADAMARD_DIM];
            for base in (0..rows).step_by(HADAMARD_DIM) {
                for c in 0..cols {
                    for (i, s) in scratch.iter_mut().enumerate() {
                        *s = x[(base + i) * cols + c];
                    }
                    hadamard_128(&mut scratch);
                    for (i, s) in scratch.iter().enumerate() {
                        x[(base + i) * cols + c] = *s;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Applies `diag(suh) @ H_K @ wq @ H_N @ diag(svh)` in place to the FP64 matrix `wq` `[k, n]`, in
/// TensorFold's order: rotate rows, scale rows, rotate columns, scale columns.
fn finish(wq: &mut [f64], k: usize, n: usize, suh: &[u16], svh: &[u16]) -> Result<()> {
    rotate(wq, k, n, Axis::Rows)?;
    for (r, row) in wq.chunks_exact_mut(n).enumerate() {
        let s = fp16::to_f64(suh[r]);
        for v in row.iter_mut() {
            *v *= s;
        }
    }
    rotate(wq, k, n, Axis::Cols)?;
    for row in wq.chunks_exact_mut(n) {
        for (v, &s) in row.iter_mut().zip(svh.iter()) {
            *v *= fp16::to_f64(s);
        }
    }
    Ok(())
}

fn check_scales(k: usize, n: usize, suh: &[u16], svh: &[u16]) -> Result<()> {
    if suh.len() != k || svh.len() != n {
        bail!(
            "EXL3 scales: suh has {} and svh {} values for a [{k}, {n}] layer",
            suh.len(),
            svh.len()
        );
    }
    if !k.is_multiple_of(HADAMARD_DIM) || !n.is_multiple_of(HADAMARD_DIM) {
        bail!("EXL3 layer [{k}, {n}]: K and N must be multiples of {HADAMARD_DIM}");
    }
    Ok(())
}

/// The whole of `W` `[K, N]` in FP64, row-major: `diag(suh) @ H_K @ W_q @ H_N @ diag(svh)`. Costs
/// `8 K N` bytes; use [`dequantize_block`] for large layers.
pub fn dequantize(
    w: Width,
    codebook: Codebook,
    trellis: &[i16],
    tiles_k: usize,
    tiles_n: usize,
    suh: &[u16],
    svh: &[u16],
) -> Result<Vec<f64>> {
    let (k, n) = (tiles_k * TILE_DIM, tiles_n * TILE_DIM);
    check_scales(k, n, suh, svh)?;
    let mut wq: Vec<f64> = unpack(w, codebook, trellis, tiles_k, tiles_n)?
        .into_iter()
        .map(fp16::to_f64)
        .collect();
    finish(&mut wq, k, n, suh, svh)?;
    Ok(wq)
}

/// The 128x128 block `(kb, nb)` of `W` in FP64, row-major (rows `128 kb ..`, columns `128 nb ..`).
/// Equal, value for value, to the same block of [`dequantize`].
#[allow(clippy::too_many_arguments)]
pub fn dequantize_block(
    w: Width,
    codebook: Codebook,
    trellis: &[i16],
    tiles_k: usize,
    tiles_n: usize,
    suh: &[u16],
    svh: &[u16],
    kb: usize,
    nb: usize,
) -> Result<Vec<f64>> {
    let (k, n) = (tiles_k * TILE_DIM, tiles_n * TILE_DIM);
    check_trellis(w, trellis, tiles_k, tiles_n)?;
    check_scales(k, n, suh, svh)?;
    let per = HADAMARD_DIM / TILE_DIM;
    if (kb + 1) * HADAMARD_DIM > k || (nb + 1) * HADAMARD_DIM > n {
        bail!("EXL3 block ({kb}, {nb}) is outside a [{k}, {n}] layer");
    }
    let words = w.tile_words();
    let mut wq = vec![0f64; HADAMARD_DIM * HADAMARD_DIM];
    for i in 0..per {
        for j in 0..per {
            let (tk, tn) = (kb * per + i, nb * per + j);
            let at = (tk * tiles_n + tn) * words;
            let states = tile_states(w, &trellis[at..at + words])?;
            for (state, place) in states.iter().zip(TILE_PLACEMENT.iter()) {
                let r = i * TILE_DIM + usize::from(place.row);
                let c = j * TILE_DIM + usize::from(place.col);
                wq[r * HADAMARD_DIM + c] = fp16::to_f64(codebook_value(codebook, *state));
            }
        }
    }
    let rows = kb * HADAMARD_DIM..(kb + 1) * HADAMARD_DIM;
    let cols = nb * HADAMARD_DIM..(nb + 1) * HADAMARD_DIM;
    finish(&mut wq, HADAMARD_DIM, HADAMARD_DIM, &suh[rows], &svh[cols])?;
    Ok(wq)
}

/// `y = x @ W + bias` in FP64 for one input row `x` `[K]`, in the factorized form
/// `((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias`. `W_q` is decoded tile by tile and never stored.
#[allow(clippy::too_many_arguments)]
pub fn matvec(
    w: Width,
    codebook: Codebook,
    trellis: &[i16],
    tiles_k: usize,
    tiles_n: usize,
    suh: &[u16],
    svh: &[u16],
    bias: Option<&[u16]>,
    x: &[f64],
) -> Result<Vec<f64>> {
    let (k, n) = (tiles_k * TILE_DIM, tiles_n * TILE_DIM);
    check_trellis(w, trellis, tiles_k, tiles_n)?;
    check_scales(k, n, suh, svh)?;
    if x.len() != k {
        bail!(
            "EXL3 matvec: x has {} values, the layer has K = {k}",
            x.len()
        );
    }
    if let Some(b) = bias
        && b.len() != n
    {
        bail!(
            "EXL3 matvec: bias has {} values, the layer has N = {n}",
            b.len()
        );
    }
    let mut t: Vec<f64> = x
        .iter()
        .zip(suh.iter())
        .map(|(&xi, &s)| xi * fp16::to_f64(s))
        .collect();
    for block in t.chunks_exact_mut(HADAMARD_DIM) {
        hadamard_128(block);
    }
    let table = codebook_table(codebook);
    let words = w.tile_words();
    let mut z = vec![0f64; n];
    for tk in 0..tiles_k {
        for tn in 0..tiles_n {
            let at = (tk * tiles_n + tn) * words;
            let states = tile_states(w, &trellis[at..at + words])?;
            for (state, place) in states.iter().zip(TILE_PLACEMENT.iter()) {
                let r = tk * TILE_DIM + usize::from(place.row);
                let c = tn * TILE_DIM + usize::from(place.col);
                z[c] += t[r] * table[usize::from(*state)];
            }
        }
    }
    for block in z.chunks_exact_mut(HADAMARD_DIM) {
        hadamard_128(block);
    }
    for (c, v) in z.iter_mut().enumerate() {
        *v *= fp16::to_f64(svh[c]);
        if let Some(b) = bias {
            *v += fp16::to_f64(b[c]);
        }
    }
    Ok(z)
}
