// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08):
// csrc/quantization/marlin/gptq_marlin_repack.cu:14-184 (`gptq_marlin_repack_kernel`, 4-bit),
// csrc/quantization/marlin/awq_marlin_repack.cu:14-140 (`awq_marlin_repack_kernel`, the AWQ un-interleave),
// vllm/model_executor/layers/quantization/utils/marlin_utils.py:509-575 (`get_scale_perms`,
// `marlin_permute_scales`, `marlin_zero_points`) and quant_utils.py:858-910 (`pack_cols`,
// `unpack_cols`). Copyright contributors to the vLLM project, Apache-2.0. Copyright (C) Marlin.2024
// Elias Frantar, Apache-2.0, adapted from https://github.com/IST-DASLab/marlin.
// Changes: the CUDA repack kernels (one thread per 32-bit output word, cp.async staging) become one CPU
// loop over the same (k tile, n tile, warp, lane) indices writing the same word; the Python
// permutations are restated as index loops; the inverses (unrepack, unpermute, unpack) are new and
// serve the round-trip oracle and `dequant_bf16`.

//! Responsibility: the Marlin weight layout on the host: [`MarlinHost`] (tiled codes, permuted BF16
//! scales, packed zero points) built from a [`LogicalInt4`] by [`to_host`], and its exact inverse
//! [`from_host`].
//!
//! Layouts (4-bit only):
//! * codes: `[k / 16][n * 16 / 8]` u32, one 16 x 64 tile (128 words) per `(k tile, n tile)`, in tile
//!   order `k_tile * (n / 64) + n_tile`; inside a tile, word `lane * 4 + warp` holds eight codes of
//!   the tensor-core fragment of (warp, lane), nibble order `0, 2, 4, 6, 1, 3, 5, 7`;
//! * scales: `[groups][n]` BF16 with every run of 64 (grouped) or 32 (one group) permuted for the
//!   fragment that applies them;
//! * zero points: `[groups][n / 8]` u32, the scale permutation then the nibble interleave, packed
//!   eight to a word along N.

use anyhow::{Result, ensure};

use super::MarlinType;
use super::reference::LogicalInt4;

/// The tile edge (`marlin.cuh` `tile_size`).
pub const TILE: usize = 16;
/// Columns of one repack tile (`tile_n_size = tile_k_size * 4`).
pub const TILE_N: usize = 64;
/// Words of one repacked 16 x 64 tile at 4 bits.
pub const TILE_WORDS: usize = TILE * TILE_N / 8;
/// The fragment row offsets of a lane (`tc_offsets`).
const TC_OFFSETS: [usize; 4] = [0, 1, 8, 9];
/// The nibble order of a packed word (`pack_idx`), from FasterTransformer's interleaved conversion.
pub const PACK_IDX: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
/// AWQ's in-word position of column `n % 8` (`undo_pack`, the inverse of the AWQ interleave).
pub const AWQ_UNDO_PACK: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// A Marlin weight on the host, ready to upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarlinHost {
    pub btype: MarlinType,
    pub n: usize,
    pub k: usize,
    pub group: usize,
    /// `[k / 16][2 * n]` repacked codes.
    pub words: Vec<u32>,
    /// `[groups][n]` permuted BF16 scales.
    pub scales: Vec<u16>,
    /// `[groups][n / 8]` packed zero points (`U4` only).
    pub zeros: Option<Vec<u32>>,
}

impl MarlinHost {
    /// Groups along K.
    pub fn groups(&self) -> usize {
        self.k / self.group
    }
}

/// `(k, n)` of every code a repacked word holds, in nibble order: the visiting order of
/// `gptq_marlin_repack_kernel`. `f(word_index, [(k, n); 8])` is called once per output word.
fn for_each_word(k: usize, n: usize, mut f: impl FnMut(usize, [(usize, usize); 8])) {
    let k_tiles = k / TILE;
    let n_tiles = n / TILE_N;
    for k_tile in 0..k_tiles {
        for n_tile in 0..n_tiles {
            let out_offset = (k_tile * n_tiles + n_tile) * TILE_WORDS;
            for warp in 0..4 {
                for lane in 0..32 {
                    let tc_col = lane / 4;
                    let tc_row = (lane % 4) * 2;
                    let cur_n = warp * 16 + tc_col;
                    let mut vals = [(0usize, 0usize); 8];
                    for (i, off) in TC_OFFSETS.iter().enumerate() {
                        let kk = k_tile * TILE + tc_row + off;
                        vals[i] = (kk, n_tile * TILE_N + cur_n);
                        vals[4 + i] = (kk, n_tile * TILE_N + cur_n + 8);
                    }
                    let mut ordered = [(0usize, 0usize); 8];
                    for (slot, &src) in PACK_IDX.iter().enumerate() {
                        ordered[slot] = vals[src];
                    }
                    f(out_offset + lane * 4 + warp, ordered);
                }
            }
        }
    }
}

/// Check the repack preconditions (`gptq_marlin_repack.cu:202-205`).
pub fn check_repack_shape(k: usize, n: usize) -> Result<()> {
    ensure!(
        k > 0 && k.is_multiple_of(TILE),
        "marlin repack: K={k} is not a multiple of {TILE}"
    );
    ensure!(
        n > 0 && n.is_multiple_of(TILE_N),
        "marlin repack: N={n} is not a multiple of {TILE_N}"
    );
    Ok(())
}

/// Repack codes given as a `(k, n) -> code` accessor (`gptq_marlin_repack_kernel`, 4-bit).
pub fn repack_with(k: usize, n: usize, code: impl Fn(usize, usize) -> u32) -> Result<Vec<u32>> {
    check_repack_shape(k, n)?;
    let mut out = vec![0u32; k * n / 8];
    for_each_word(k, n, |w, coords| {
        let mut res = 0u32;
        for (i, &(kk, nn)) in coords.iter().enumerate() {
            res |= (code(kk, nn) & 0xF) << (i * 4);
        }
        out[w] = res;
    });
    Ok(out)
}

/// Repack a `[k][n]` code matrix.
pub fn repack_codes(q: &[u8], k: usize, n: usize) -> Result<Vec<u32>> {
    ensure!(
        q.len() == k * n,
        "marlin repack: {} codes for [{k}x{n}]",
        q.len()
    );
    repack_with(k, n, |kk, nn| u32::from(q[kk * n + nn]))
}

/// Repack GPTQ `qweight` (`[k / 8][n]` u32, eight K-consecutive codes per word, low nibble first).
pub fn repack_gptq(qweight: &[u32], k: usize, n: usize) -> Result<Vec<u32>> {
    ensure!(
        k.is_multiple_of(8) && qweight.len() == k / 8 * n,
        "marlin repack: GPTQ qweight has {} words, [{k}/8 x {n}] expected",
        qweight.len()
    );
    repack_with(k, n, |kk, nn| qweight[(kk / 8) * n + nn] >> ((kk % 8) * 4))
}

/// Repack AWQ `qweight` (`[k][n / 8]` u32, AWQ-interleaved along N; `awq_marlin_repack_kernel`).
pub fn repack_awq(qweight: &[u32], k: usize, n: usize) -> Result<Vec<u32>> {
    ensure!(
        n.is_multiple_of(8) && qweight.len() == k * (n / 8),
        "marlin repack: AWQ qweight has {} words, [{k} x {n}/8] expected",
        qweight.len()
    );
    repack_with(k, n, |kk, nn| {
        qweight[kk * (n / 8) + nn / 8] >> (AWQ_UNDO_PACK[nn % 8] * 4)
    })
}

/// The inverse of [`repack_codes`]: the `[k][n]` code matrix of repacked words.
pub fn unrepack_codes(words: &[u32], k: usize, n: usize) -> Result<Vec<u8>> {
    check_repack_shape(k, n)?;
    ensure!(
        words.len() == k * n / 8,
        "marlin unrepack: {} words for [{k}x{n}]",
        words.len()
    );
    let mut q = vec![0u8; k * n];
    for_each_word(k, n, |w, coords| {
        let word = words[w];
        for (i, &(kk, nn)) in coords.iter().enumerate() {
            q[kk * n + nn] = ((word >> (i * 4)) & 0xF) as u8;
        }
    });
    Ok(q)
}

/// `scale_perm` (64 entries: `i + 8 j`) and `scale_perm_single` (32 entries) of `get_scale_perms`.
pub fn scale_perms() -> ([usize; 64], [usize; 32]) {
    let mut perm = [0usize; 64];
    for i in 0..8 {
        for j in 0..8 {
            perm[i * 8 + j] = i + 8 * j;
        }
    }
    let mut single = [0usize; 32];
    for i in 0..4 {
        for (j, d) in [0usize, 1, 8, 9, 16, 17, 24, 25].iter().enumerate() {
            single[i * 8 + j] = 2 * i + d;
        }
    }
    (perm, single)
}

/// The permutation `marlin_permute_scales` applies to each run: the 64-entry one for a grouped weight
/// (`group < k`), the 32-entry one for a single group.
fn scale_perm_for(k: usize, group: usize) -> Vec<usize> {
    let (perm, single) = scale_perms();
    if group < k {
        perm.to_vec()
    } else {
        single.to_vec()
    }
}

/// `marlin_permute_scales`: `out[run][j] = s[run][perm[j]]` over consecutive runs of the flattened
/// `[groups][n]` array.
pub fn permute_scales(s: &[u16], k: usize, n: usize, group: usize) -> Result<Vec<u16>> {
    let perm = scale_perm_for(k, group);
    let len = perm.len();
    ensure!(
        s.len() == (k / group) * n && s.len().is_multiple_of(len),
        "marlin: {} scales for [{}/{group} x {n}] do not split into runs of {len}",
        s.len(),
        k
    );
    let mut out = vec![0u16; s.len()];
    for (dst, src) in out.chunks_exact_mut(len).zip(s.chunks_exact(len)) {
        for (j, &p) in perm.iter().enumerate() {
            dst[j] = src[p];
        }
    }
    Ok(out)
}

/// The inverse of [`permute_scales`].
pub fn unpermute_scales(s: &[u16], k: usize, n: usize, group: usize) -> Result<Vec<u16>> {
    let perm = scale_perm_for(k, group);
    let len = perm.len();
    ensure!(
        s.len() == (k / group) * n && s.len().is_multiple_of(len),
        "marlin: {} permuted scales for [{}/{group} x {n}]",
        s.len(),
        k
    );
    let mut out = vec![0u16; s.len()];
    for (dst, src) in out.chunks_exact_mut(len).zip(s.chunks_exact(len)) {
        for (j, &p) in perm.iter().enumerate() {
            dst[p] = src[j];
        }
    }
    Ok(out)
}

/// `marlin_zero_points` (4-bit): the 64-entry scale permutation (never the single one: zero points
/// apply on every MMA), the nibble interleave, then `pack_cols` (word `c` holds columns `8c..8c+8`,
/// slot `i` at bits `4i`).
pub fn pack_zero_points(z: &[u8], groups: usize, n: usize) -> Result<Vec<u32>> {
    let (perm, _) = scale_perms();
    ensure!(
        z.len() == groups * n && n.is_multiple_of(64),
        "marlin: {} zero points for [{groups} x {n}]",
        z.len()
    );
    let mut permuted = vec![0u8; z.len()];
    for (dst, src) in permuted.chunks_exact_mut(64).zip(z.chunks_exact(64)) {
        for (j, &p) in perm.iter().enumerate() {
            dst[j] = src[p];
        }
    }
    let mut out = vec![0u32; groups * n / 8];
    for (w, chunk) in out.iter_mut().zip(permuted.chunks_exact(8)) {
        let mut word = 0u32;
        for (i, &src) in PACK_IDX.iter().enumerate() {
            word |= u32::from(chunk[src] & 0xF) << (i * 4);
        }
        *w = word;
    }
    Ok(out)
}

/// The inverse of [`pack_zero_points`].
pub fn unpack_zero_points(words: &[u32], groups: usize, n: usize) -> Result<Vec<u8>> {
    let (perm, _) = scale_perms();
    ensure!(
        words.len() == groups * n / 8 && n.is_multiple_of(64),
        "marlin: {} zero-point words for [{groups} x {n}]",
        words.len()
    );
    let mut permuted = vec![0u8; groups * n];
    for (&word, chunk) in words.iter().zip(permuted.chunks_exact_mut(8)) {
        for (i, &dst) in PACK_IDX.iter().enumerate() {
            chunk[dst] = ((word >> (i * 4)) & 0xF) as u8;
        }
    }
    let mut z = vec![0u8; groups * n];
    for (dst, src) in z.chunks_exact_mut(64).zip(permuted.chunks_exact(64)) {
        for (j, &p) in perm.iter().enumerate() {
            dst[p] = src[j];
        }
    }
    Ok(z)
}

/// Build the upload image of a logical weight.
pub fn to_host(l: &LogicalInt4) -> Result<MarlinHost> {
    l.validate()?;
    let words = repack_codes(&l.q, l.k, l.n)?;
    let scales = permute_scales(&l.scales, l.k, l.n, l.group)?;
    let zeros = match &l.zeros {
        Some(z) => Some(pack_zero_points(z, l.groups(), l.n)?),
        None => None,
    };
    Ok(MarlinHost {
        btype: l.btype(),
        n: l.n,
        k: l.k,
        group: l.group,
        words,
        scales,
        zeros,
    })
}

/// The exact inverse of [`to_host`] (the round-trip oracle and `dequant_bf16`).
pub fn from_host(h: &MarlinHost) -> Result<LogicalInt4> {
    let q = unrepack_codes(&h.words, h.k, h.n)?;
    let scales = unpermute_scales(&h.scales, h.k, h.n, h.group)?;
    let zeros = match (&h.zeros, h.btype) {
        (Some(z), MarlinType::U4) => Some(unpack_zero_points(z, h.groups(), h.n)?),
        (None, MarlinType::U4B8) => None,
        (z, t) => anyhow::bail!(
            "marlin: a {t:?} weight with{} zero points",
            if z.is_some() { "" } else { "out" }
        ),
    };
    Ok(LogicalInt4 {
        n: h.n,
        k: h.k,
        group: h.group,
        q,
        scales,
        zeros,
    })
}
