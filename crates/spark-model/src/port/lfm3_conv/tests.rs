// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of the LFM3 conv layer that need no GPU: the
//! CPU reference against itself (decode vs prefill, chunked vs one shot,
//! packed sequences), a line-for-line emulation of the CUDA prefill tiling
//! against the reference, and the layer-level invariants on `RefLayer`. The
//! host plumbing (blob, launch ABI, state lifecycle) is in `tests_mock.rs`.

use super::launch;
use super::reference::{self as r, RefLayer};
use super::{CONV_WIDTH, STATE_LEN, state};

/// The prefill lengths the brief names.
const LENS: [usize; 10] = [1, 2, 5, 6, 7, 8, 63, 64, 65, 4096];

fn dim_for(t: usize) -> usize {
    if t >= 1024 { 8 } else { 24 }
}

// ── reference self-consistency ──────────────────────────────────────────────

#[test]
fn bf16_rounding_is_nearest_even() {
    assert_eq!(r::bf16_bits(1.0), 0x3F80);
    assert_eq!(r::bf16_value(0x3F80), 1.0);
    // 1 + 2^-8 is exactly halfway between 1.0 and the next BF16: ties to even (down).
    assert_eq!(r::bf16_bits(1.0 + 1.0 / 256.0), 0x3F80);
    // 1 + 3 * 2^-8 is halfway between 0x3F81 and 0x3F82: ties to even (up).
    assert_eq!(r::bf16_bits(1.0 + 3.0 / 256.0), 0x3F82);
    assert!(r::bf16_value(r::bf16_bits(f32::NAN)).is_nan());
    assert_eq!(r::bf16_bits(-0.0), 0x8000);
}

#[test]
fn decode_step_equals_a_one_token_prefill() {
    for warm in [false, true] {
        let p = r::generate(11 + warm as u64, 1, 32, warm);
        let (mut s1, mut s2) = (p.state.clone(), p.state.clone());
        let a = r::decode_step(&p.bcx, &p.weight, &mut s1, p.dim);
        let b = r::prefill(&p.bcx, &p.weight, &mut s2, 1, p.dim);
        assert_eq!(a, b, "warm={warm}");
        assert_eq!(s1, s2, "warm={warm}");
    }
}

#[test]
fn chunked_prefill_equals_one_shot_for_every_length() {
    for &t in &LENS {
        let dim = dim_for(t);
        let p = r::generate(100 + t as u64, t, dim, true);
        let mut s_one = p.state.clone();
        let y_one = r::prefill(&p.bcx, &p.weight, &mut s_one, t, dim);
        // Splits that put a chunk shorter than the window on either side.
        for split in [1usize, 3, 5, 6, 7, t / 2, t.saturating_sub(2)] {
            if split == 0 || split >= t {
                continue;
            }
            let mut s = p.state.clone();
            let mut y = r::prefill(&p.bcx[..split * 3 * dim], &p.weight, &mut s, split, dim);
            y.extend(r::prefill(
                &p.bcx[split * 3 * dim..],
                &p.weight,
                &mut s,
                t - split,
                dim,
            ));
            assert_eq!(y, y_one, "t={t} split={split}");
            assert_eq!(s, s_one, "t={t} split={split} state");
        }
    }
}

#[test]
fn decode_after_prefill_equals_continuing_the_prefill() {
    for &t in &[1usize, 5, 6, 7, 64] {
        let dim = 16;
        let extra = 9;
        let p = r::generate(300 + t as u64, t + extra, dim, false);
        let mut s_all = p.state.clone();
        let y_all = r::prefill(&p.bcx, &p.weight, &mut s_all, t + extra, dim);
        let mut s = p.state.clone();
        let mut y = r::prefill(&p.bcx[..t * 3 * dim], &p.weight, &mut s, t, dim);
        for i in t..t + extra {
            y.extend(r::decode_step(
                &p.bcx[i * 3 * dim..(i + 1) * 3 * dim],
                &p.weight,
                &mut s,
                dim,
            ));
        }
        assert_eq!(y, y_all, "t={t}");
        assert_eq!(s, s_all, "t={t} state");
    }
}

#[test]
fn a_short_chunk_shifts_the_old_window() {
    // T < 6: the new window is the old one shifted left by T, the chunk's u appended.
    let p = r::generate(7, 2, 4, true);
    let mut s = p.state.clone();
    r::prefill(&p.bcx, &p.weight, &mut s, 2, 4);
    for c in 0..4 {
        assert_eq!(
            s[c * STATE_LEN..c * STATE_LEN + 4],
            p.state[c * STATE_LEN + 2..(c + 1) * STATE_LEN]
        );
        for i in 0..2 {
            let u = r::u_value(p.bcx[i * 12 + c], p.bcx[i * 12 + 8 + c]);
            assert_eq!(s[c * STATE_LEN + 4 + i], r::bf16_bits(u));
        }
    }
}

#[test]
fn packed_sequences_do_not_leak() {
    let dim = 16;
    let lens = [5usize, 0, 70, 6, 1];
    let mut cu = vec![0usize];
    for l in lens {
        cu.push(cu.last().unwrap() + l);
    }
    let total = *cu.last().unwrap();
    let p = r::generate(41, total, dim, false);
    let mut rng = r::Rng::new(42);
    let base: Vec<Vec<u16>> = lens
        .iter()
        .map(|_| rng.bf16_vec(dim * STATE_LEN, 2.0))
        .collect();
    let mut states = base.clone();
    let y = r::prefill_packed(&p.bcx, &p.weight, &cu, &mut states, dim);
    // Perturb sequence 2's tokens and window: every other sequence is unchanged.
    let mut bcx2 = p.bcx.clone();
    for v in &mut bcx2[cu[2] * 3 * dim..cu[3] * 3 * dim] {
        *v ^= 0x0101;
    }
    let mut states2 = base.clone();
    states2[2] = rng.bf16_vec(dim * STATE_LEN, 2.0);
    let y2 = r::prefill_packed(&bcx2, &p.weight, &cu, &mut states2, dim);
    for s in [0usize, 3, 4] {
        assert_eq!(
            y[cu[s] * dim..cu[s + 1] * dim],
            y2[cu[s] * dim..cu[s + 1] * dim],
            "seq {s}"
        );
        assert_eq!(states[s], states2[s], "seq {s} state");
    }
    assert_eq!(
        states[1], base[1],
        "a zero-length sequence keeps its window"
    );
}

// ── emulation of the CUDA prefill (tiles, halo, sequence walk) ──────────────

/// `lfm3_shortconv_prefill_bf16` and `..._state_bf16`, line for line, on the
/// host: grid y walks 64-token tiles, every channel keeps a six-value window
/// filled from the tile's halo (same sequence) or the initial state.
fn emulate_kernels(
    bcx: &[u16],
    weight: &[u16],
    cu: Option<&[i32]>,
    states: &mut [Vec<u16>],
    total: usize,
    dim: usize,
) -> Vec<u16> {
    let tile = launch::TILE_T as usize;
    let mut y = vec![0u16; total * dim];
    let u_at =
        |t: usize, c: usize| r::u_value(bcx[t * 3 * dim + c], bcx[t * 3 * dim + 2 * dim + c]);
    let num_seqs = states.len();
    let bounds = |s: usize| -> (usize, usize) {
        match cu {
            Some(cu) => (cu[s] as usize, cu[s + 1] as usize),
            None => (0, total),
        }
    };
    let fill =
        |states: &[Vec<u16>], s: usize, start: usize, local: usize, c: usize| -> [f32; STATE_LEN] {
            std::array::from_fn(|j| {
                let li = local as isize - STATE_LEN as isize + j as isize;
                if li >= 0 {
                    u_at(start + li as usize, c)
                } else {
                    r::bf16_value(states[s][c * STATE_LEN + (li + STATE_LEN as isize) as usize])
                }
            })
        };
    for c in 0..dim {
        let w: [f32; CONV_WIDTH] =
            std::array::from_fn(|k| r::bf16_value(weight[c * CONV_WIDTH + k]));
        let mut t0 = 0;
        while t0 < total {
            let t1 = (t0 + tile).min(total);
            let mut s = match cu {
                Some(cu) => (0..num_seqs).rev().find(|&s| cu[s] as usize <= t0).unwrap(),
                None => 0,
            };
            let (mut start, mut end) = bounds(s);
            let mut win = fill(states, s, start, t0 - start, c);
            for t in t0..t1 {
                if t >= end {
                    loop {
                        s += 1;
                        (start, end) = bounds(s);
                        if t < end {
                            break;
                        }
                    }
                    win = fill(states, s, start, 0, c);
                }
                let u = u_at(t, c);
                let v = r::conv_value(&w, &win, u);
                y[t * dim + c] = r::bf16_bits(r::bf16_value(bcx[t * 3 * dim + dim + c]) * v);
                win.copy_within(1.., 0);
                win[STATE_LEN - 1] = u;
            }
            t0 = t1;
        }
    }
    // The state kernel, after the y kernel.
    for (s, st) in states.iter_mut().enumerate() {
        let (start, end) = bounds(s);
        let len = end - start;
        for c in 0..dim {
            let old: Vec<u16> = st[c * STATE_LEN..(c + 1) * STATE_LEN].to_vec();
            for j in 0..STATE_LEN {
                let e = len + j;
                st[c * STATE_LEN + j] = if e < STATE_LEN {
                    old[e]
                } else {
                    r::bf16_bits(u_at(start + e - STATE_LEN, c))
                };
            }
        }
    }
    y
}

#[test]
fn kernel_tiling_matches_the_reference_single_sequence() {
    for &t in &LENS {
        let dim = dim_for(t);
        let p = r::generate(500 + t as u64, t, dim, true);
        let mut s_ref = p.state.clone();
        let y_ref = r::prefill(&p.bcx, &p.weight, &mut s_ref, t, dim);
        let mut st = vec![p.state.clone()];
        let y = emulate_kernels(&p.bcx, &p.weight, None, &mut st, t, dim);
        assert_eq!(y, y_ref, "t={t}");
        assert_eq!(st[0], s_ref, "t={t} state");
    }
}

#[test]
fn kernel_tiling_matches_the_reference_packed() {
    let dim = 8;
    // Boundaries inside a tile, on a tile edge, zero-length and shorter-than-window sequences.
    let lens = [3usize, 61, 0, 64, 2, 130, 6];
    let mut cu = vec![0i32];
    for l in lens {
        cu.push(cu.last().unwrap() + l as i32);
    }
    let total = *cu.last().unwrap() as usize;
    let p = r::generate(77, total, dim, false);
    let mut rng = r::Rng::new(78);
    let init: Vec<Vec<u16>> = lens
        .iter()
        .map(|_| rng.bf16_vec(dim * STATE_LEN, 2.0))
        .collect();
    let mut st_ref = init.clone();
    let cu_usize: Vec<usize> = cu.iter().map(|&v| v as usize).collect();
    let y_ref = r::prefill_packed(&p.bcx, &p.weight, &cu_usize, &mut st_ref, dim);
    let mut st = init;
    let y = emulate_kernels(&p.bcx, &p.weight, Some(cu.as_slice()), &mut st, total, dim);
    assert_eq!(y, y_ref);
    assert_eq!(st, st_ref);
}

// ── layer-level invariants on the CPU model ─────────────────────────────────

fn hidden_rows(seed: u64, t: usize, dim: usize) -> Vec<f32> {
    r::Rng::new(seed)
        .bf16_vec(t * dim, 1.5)
        .into_iter()
        .map(r::bf16_value)
        .collect()
}

#[test]
fn layer_decode_tokens_equal_layer_prefill() {
    let l = RefLayer::generate(9, 16, 24);
    let n = 11;
    let h0 = hidden_rows(10, n, 16);
    let mut h_pre = h0.clone();
    let mut s_pre = l.fresh_state();
    l.forward(&mut h_pre, &mut s_pre, n);
    let mut h_dec = h0.clone();
    let mut s_dec = l.fresh_state();
    for i in 0..n {
        l.forward(&mut h_dec[i * 16..(i + 1) * 16], &mut s_dec, 1);
    }
    assert_eq!(h_dec, h_pre);
    assert_eq!(s_dec, s_pre);
}

#[test]
fn layer_interleaved_sequences_keep_separate_windows() {
    let l = RefLayer::generate(12, 16, 24);
    let (a, b) = (hidden_rows(13, 9, 16), hidden_rows(14, 9, 16));
    let (mut ha, mut hb) = (a.clone(), b.clone());
    let (mut sa, mut sb) = (l.fresh_state(), l.fresh_state());
    for i in 0..9 {
        l.forward(&mut ha[i * 16..(i + 1) * 16], &mut sa, 1);
        l.forward(&mut hb[i * 16..(i + 1) * 16], &mut sb, 1);
    }
    let (mut ra, mut rb) = (a, b);
    let (mut ta, mut tb) = (l.fresh_state(), l.fresh_state());
    l.forward(&mut ra, &mut ta, 9);
    l.forward(&mut rb, &mut tb, 9);
    assert_eq!((ha, sa), (ra, ta));
    assert_eq!((hb, sb), (rb, tb));
}

#[test]
fn layer_snapshot_then_restore_at_a_chunk_boundary_reproduces_the_continuation() {
    let l = RefLayer::generate(15, 16, 24);
    let (p, q) = (40usize, 13usize);
    let h0 = hidden_rows(16, p + q, 16);
    let mut h_full = h0.clone();
    let mut s_full = l.fresh_state();
    l.forward(&mut h_full, &mut s_full, p + q);
    // First request: chunk 1, snapshot the window at the boundary.
    let mut h1 = h0[..p * 16].to_vec();
    let mut s1 = l.fresh_state();
    l.forward(&mut h1, &mut s1, p);
    let payload: Vec<u8> = s1.iter().flat_map(|v| v.to_le_bytes()).collect();
    let blob = state::encode_blob(16, &payload).unwrap();
    // A later request restores into a fresh window and continues.
    let restored = state::decode_blob(16, &blob).unwrap();
    let mut s2: Vec<u16> = restored
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut h2 = h0[p * 16..].to_vec();
    l.forward(&mut h2, &mut s2, q);
    assert_eq!(h2, h_full[p * 16..]);
    assert_eq!(s2, s_full);
}
