// SPDX-License-Identifier: AGPL-3.0-only
//! CPU references for the native EXL3 matmul:
//!
//!  * `exact_a_had` — BIT-EXACT replica of the fused kernels' input-rotation
//!    stage (`had_hf_r_128_inner<true,false>`): fp16 suh pre-scale (one
//!    rounding), fp32 4-point butterfly, five XOR-stage adds in fp32 with
//!    sign-on-own-value orientation, fp32 `* 1/sqrt(128)`, ONE terminal
//!    rounding to fp16.
//!  * `truth_matmul` — f64 end-to-end oracle
//!    `y = R_out( R_in(x) @ W_hat )` with W_hat from the parity-proven
//!    `cpu_ref::decode_tile`, for the tolerance-gated legs.

use half::f16;
use spark_runtime::weights::exl3::{Exl3Codebook, cpu_ref};

/// The kernels' rotation scale constant: the f32 nearest 1/sqrt(128)
/// (upstream literal 0.088388347648f = bits 0x3db504f3). Written as bits
/// because the clippy-truncated decimal literal rounds ONE ULP HIGH
/// (0x3db504f4) and would break the bit-exact A_had leg.
pub const RS: f32 = f32::from_bits(0x3db504f3);

pub fn cb_enum(cb: u32) -> Exl3Codebook {
    match cb {
        0 => Exl3Codebook::Inst3,
        1 => Exl3Codebook::Mcg,
        _ => Exl3Codebook::Mul1,
    }
}

/// In-place natural-order Walsh-Hadamard transform of one 128-vector (f64).
/// Stage order is irrelevant at f64 precision (~1e-16 vs gates of 1e-3).
pub fn fwht128(v: &mut [f64]) {
    debug_assert_eq!(v.len(), 128);
    let mut s = 1;
    while s < 128 {
        let mut a = 0;
        while a < 128 {
            for i in a..a + s {
                let x = v[i];
                let y = v[i + s];
                v[i] = x + y;
                v[i + s] = x - y;
            }
            a += 2 * s;
        }
        s <<= 1;
    }
}

/// Decode the raw (pre-rotation) trellis weights to f64 `[k, n]` row-major.
pub fn decode_what_f64(
    trellis: &[u16],
    k_dim: usize,
    n_dim: usize,
    k_bits: u32,
    cb: Exl3Codebook,
) -> Vec<f64> {
    let kt = 16 * k_bits as usize;
    let tiles_n = n_dim / 16;
    assert_eq!(trellis.len(), (k_dim / 16) * tiles_n * kt);
    let mut w = vec![0f64; k_dim * n_dim];
    for tr in 0..k_dim / 16 {
        for tc in 0..tiles_n {
            let base = (tr * tiles_n + tc) * kt;
            let tile = cpu_ref::decode_tile(&trellis[base..base + kt], k_bits, cb);
            for (r, tile_row) in tile.iter().enumerate() {
                for (c, v) in tile_row.iter().enumerate() {
                    w[(tr * 16 + r) * n_dim + tc * 16 + c] = v.to_f64();
                }
            }
        }
    }
    w
}

/// f64 truth for the fused pipeline on `m` rows of fp16 activations:
/// per row `y = (((x .* suh) H128 rs) @ What) H128 (rs * out_scale) .* svh`.
/// `out_scale` folds a routing weight into the output rotation exactly the
/// way the mgemm epilogue does (`scale = rs * B_weights[j]`).
pub fn truth_matmul(
    a_bits: &[u16],
    suh: &[u16],
    svh: &[u16],
    what: &[f64],
    m: usize,
    k: usize,
    n: usize,
    out_scale: f64,
) -> Vec<f64> {
    let rs = RS as f64;
    assert_eq!(a_bits.len(), m * k);
    assert_eq!(what.len(), k * n);
    let suh_f: Vec<f64> = suh.iter().map(|&b| f16::from_bits(b).to_f64()).collect();
    let svh_f: Vec<f64> = svh.iter().map(|&b| f16::from_bits(b).to_f64()).collect();
    let mut y_all = vec![0f64; m * n];
    let mut u = vec![0f64; k];
    for row in 0..m {
        for i in 0..k {
            u[i] = f16::from_bits(a_bits[row * k + i]).to_f64() * suh_f[i];
        }
        for blk in u.chunks_mut(128) {
            fwht128(blk);
            for x in blk {
                *x *= rs;
            }
        }
        let y = &mut y_all[row * n..(row + 1) * n];
        for (i, &ui) in u.iter().enumerate() {
            if ui == 0.0 {
                continue;
            }
            let wrow = &what[i * n..(i + 1) * n];
            for (yj, wj) in y.iter_mut().zip(wrow.iter()) {
                *yj += ui * wj;
            }
        }
        for blk in y.chunks_mut(128) {
            fwht128(blk);
            for x in blk {
                *x *= rs * out_scale;
            }
        }
        for (yj, sj) in y.iter_mut().zip(svh_f.iter()) {
            *yj *= sj;
        }
    }
    y_all
}

/// Idealized materialized-path reference: `y = x @ W` in f64 with `W` given
/// as f64 `[k, n]` (already carrying rotations + scales). Used with a
/// bf16-rounded reconstruct for the calibration backstop.
pub fn truth_dense(a_bits: &[u16], w: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut y_all = vec![0f64; m * n];
    for row in 0..m {
        let y = &mut y_all[row * n..(row + 1) * n];
        for i in 0..k {
            let a = f16::from_bits(a_bits[row * k + i]).to_f64();
            if a == 0.0 {
                continue;
            }
            let wrow = &w[i * n..(i + 1) * n];
            for (yj, wj) in y.iter_mut().zip(wrow.iter()) {
                *yj += a * wj;
            }
        }
    }
    y_all
}

/// `cpu_ref::reconstruct_had_f16` output rounded through bf16, as f64
/// `[k, n]` — the materialized serving path's weights.
pub fn materialized_bf16_f64(
    trellis: &[u16],
    suh: &[u16],
    svh: &[u16],
    k_dim: usize,
    n_dim: usize,
    k_bits: u32,
    cb: Exl3Codebook,
) -> Vec<f64> {
    let w_f16 = cpu_ref::reconstruct_had_f16(trellis, suh, svh, k_dim, n_dim, k_bits, cb);
    w_f16
        .iter()
        .map(|&b| half::bf16::from_f32(f16::from_bits(b).to_f32()).to_f64())
        .collect()
}

/// Bit-exact CPU replica of the input-rotation stage writing A_had:
/// warp `w` covers elements `[w*128, (w+1)*128)` of row-major `[m, k]`,
/// with `suh` re-based at `(w*128) % k`.
pub fn exact_a_had(a_bits: &[u16], suh: &[u16], m: usize, k: usize) -> Vec<u16> {
    assert_eq!(a_bits.len(), m * k);
    assert_eq!(k % 128, 0);
    let mut out = vec![0u16; m * k];
    for w in 0..(m * k / 128) {
        let base = w * 128;
        let sbase = (w * 128) % k;
        let mut h = [0f32; 128];
        for t in 0..32 {
            let mut v = [0f32; 4];
            for (j, vj) in v.iter_mut().enumerate() {
                let x = f16::from_bits(a_bits[base + 4 * t + j]);
                let s = f16::from_bits(suh[sbase + 4 * t + j]);
                // __hmul: exact product in f32, one fp16 rounding.
                *vj = f16::from_f32(x.to_f32() * s.to_f32()).to_f32();
            }
            let s0 = v[0] + v[1];
            let d0 = v[0] - v[1];
            let s1 = v[2] + v[3];
            let d1 = v[2] - v[3];
            h[4 * t] = s0 + s1;
            h[4 * t + 1] = d0 + d1;
            h[4 * t + 2] = s0 - s1;
            h[4 * t + 3] = d0 - d1;
        }
        // shuffle_had_f4x32: five XOR stages; the lane with (t & i) != 0
        // sign-flips its OWN value, then adds the partner's.
        for i in [1usize, 2, 4, 8, 16] {
            let prev = h;
            for t in 0..32 {
                let p = t ^ i;
                for j in 0..4 {
                    let own = if t & i != 0 {
                        -prev[4 * t + j]
                    } else {
                        prev[4 * t + j]
                    };
                    h[4 * t + j] = own + prev[4 * p + j];
                }
            }
        }
        for (idx, hv) in h.iter().enumerate() {
            // __floats2half2_rn(h * r_scale): fp32 multiply, one rounding.
            out[base + idx] = f16::from_f32(hv * RS).to_bits();
        }
    }
    out
}
