// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The
// HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors
// to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas. The conv math
// follows `Lfm3ShortConv.forward` and `Lfm3DecoderLayer.forward`.

//! Responsibility: the f32 CPU reference of the LFM3 gated short convolution
//! (the oracle of `kernels/ada/common/lfm3_shortconv_ada.cu`) and of one
//! whole conv layer, plus a deterministic problem generator.
//!
//! The conv reference is written from the DEFINITION, not from the kernel's
//! tiling: the extended input `E = state ++ u[0..T]`, `y[t] = C[t] *
//! conv7(E[t..t+7])`, new state `E[T..T+6]`. It performs the same IEEE f32
//! operations in the same order as the kernel (`u = bf16(B * x)`, products
//! and sums un-fused, taps ascending from 0.0, BF16 narrowing by round to
//! nearest even), so the GPU must match it bit for bit.
//!
//! The layer reference (`RefLayer`) is f32 throughout with BF16 rounding at
//! the points where the device path stores BF16; it is the CPU model the
//! layer-level invariants are proven on (decode equals prefill, chunked equals
//! one shot, sequences keep separate windows, snapshot then restore).

use super::{CONV_WIDTH, STATE_LEN};

/// BF16 bits of `v`, round to nearest even (CUDA `__float2bfloat16_rn`).
pub fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) as u16) | 0x0040;
    }
    let lsb = (b >> 16) & 1;
    ((b.wrapping_add(0x7FFF + lsb)) >> 16) as u16
}

/// The f32 value of BF16 bits.
pub fn bf16_value(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// `v` rounded to BF16 and widened back.
pub fn round_bf16(v: f32) -> f32 {
    bf16_value(bf16_bits(v))
}

/// `u = bf16(B * x)` for one channel.
pub fn u_value(b: u16, x: u16) -> f32 {
    round_bf16(bf16_value(b) * bf16_value(x))
}

/// `bf16(sum_k w[k] * window[k])` over six older inputs then `u`, taps ascending.
pub fn conv_value(w: &[f32; CONV_WIDTH], older: &[f32; STATE_LEN], u: f32) -> f32 {
    let mut acc = 0.0f32;
    for k in 0..STATE_LEN {
        acc += w[k] * older[k];
    }
    acc += w[STATE_LEN] * u;
    round_bf16(acc)
}

fn taps(weight: &[u16], c: usize) -> [f32; CONV_WIDTH] {
    std::array::from_fn(|k| bf16_value(weight[c * CONV_WIDTH + k]))
}

/// Prefill of `t` tokens of ONE sequence. `bcx` is `[t, 3 * dim]` BF16 bits
/// (`B | C | x`), `weight` `[dim, 7]`, `state` `[dim, 6]` (read, then
/// replaced by the window after the chunk). Returns y `[t, dim]` BF16 bits.
pub fn prefill(bcx: &[u16], weight: &[u16], state: &mut [u16], t: usize, dim: usize) -> Vec<u16> {
    assert_eq!(bcx.len(), t * 3 * dim, "bcx shape");
    assert_eq!(weight.len(), dim * CONV_WIDTH, "weight shape");
    assert_eq!(state.len(), dim * STATE_LEN, "state shape");
    let mut y = vec![0u16; t * dim];
    for c in 0..dim {
        let w = taps(weight, c);
        // E = old window ++ u[0..t].
        let mut e: Vec<f32> = (0..STATE_LEN)
            .map(|j| bf16_value(state[c * STATE_LEN + j]))
            .collect();
        e.extend((0..t).map(|i| u_value(bcx[i * 3 * dim + c], bcx[i * 3 * dim + 2 * dim + c])));
        for i in 0..t {
            let older: [f32; STATE_LEN] = std::array::from_fn(|k| e[i + k]);
            let v = conv_value(&w, &older, e[i + STATE_LEN]);
            let cg = bf16_value(bcx[i * 3 * dim + dim + c]);
            y[i * dim + c] = bf16_bits(cg * v);
        }
        for j in 0..STATE_LEN {
            state[c * STATE_LEN + j] = bf16_bits(e[t + j]);
        }
    }
    y
}

/// One decode token: the explicit shift form (`s' = [s1..s5, u]`), written
/// separately from [`prefill`] so the two definitions check each other.
pub fn decode_step(bcx_row: &[u16], weight: &[u16], state: &mut [u16], dim: usize) -> Vec<u16> {
    assert_eq!(bcx_row.len(), 3 * dim, "bcx row shape");
    let mut y = vec![0u16; dim];
    for c in 0..dim {
        let w = taps(weight, c);
        let s = &mut state[c * STATE_LEN..(c + 1) * STATE_LEN];
        let older: [f32; STATE_LEN] = std::array::from_fn(|j| bf16_value(s[j]));
        let u = u_value(bcx_row[c], bcx_row[2 * dim + c]);
        let v = conv_value(&w, &older, u);
        y[c] = bf16_bits(bf16_value(bcx_row[dim + c]) * v);
        s.copy_within(1.., 0);
        s[STATE_LEN - 1] = bf16_bits(u);
    }
    y
}

/// Packed prefill: sequence `s` owns rows `cu[s]..cu[s + 1]` of `bcx` and the
/// window `states[s]`. Each sequence is computed by [`prefill`] alone, so no
/// value can cross a boundary by construction (the property the kernel's
/// packed form must reproduce).
pub fn prefill_packed(
    bcx: &[u16],
    weight: &[u16],
    cu: &[usize],
    states: &mut [Vec<u16>],
    dim: usize,
) -> Vec<u16> {
    assert_eq!(cu.len(), states.len() + 1, "cu has num_seqs + 1 entries");
    let total = *cu.last().expect("cu is non-empty");
    let mut y = vec![0u16; total * dim];
    for (s, st) in states.iter_mut().enumerate() {
        let (a, b) = (cu[s], cu[s + 1]);
        let ys = prefill(&bcx[a * 3 * dim..b * 3 * dim], weight, st, b - a, dim);
        y[a * dim..b * dim].copy_from_slice(&ys);
    }
    y
}

/// xorshift64* stream, the generator of every fixture here.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[-scale, scale)`.
    pub fn uniform(&mut self, scale: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (unit * 2.0 - 1.0) * scale
    }
    /// `n` BF16 values uniform in `[-scale, scale)`.
    pub fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| bf16_bits(self.uniform(scale))).collect()
    }
}

/// A conv problem: BCx `[t, 3 * dim]`, weight `[dim, 7]`, initial window `[dim, 6]`.
#[derive(Clone, Debug)]
pub struct ConvProblem {
    pub t: usize,
    pub dim: usize,
    pub bcx: Vec<u16>,
    pub weight: Vec<u16>,
    pub state: Vec<u16>,
}

/// Deterministic problem. `warm` gives a non-zero initial window (a chunk
/// that continues a sequence); otherwise the window is zero (a fresh one).
pub fn generate(seed: u64, t: usize, dim: usize, warm: bool) -> ConvProblem {
    let mut r = Rng::new(seed);
    let bcx = r.bf16_vec(t * 3 * dim, 2.0);
    let weight = r.bf16_vec(dim * CONV_WIDTH, 0.6);
    let state = if warm {
        r.bf16_vec(dim * STATE_LEN, 2.0)
    } else {
        vec![0u16; dim * STATE_LEN]
    };
    ConvProblem {
        t,
        dim,
        bcx,
        weight,
        state,
    }
}

/// f32 CPU model of a whole conv layer at a small width, weights BF16-valued.
/// FFN: `down(silu(gate x) * up x)` (`w2(silu(w1 x) * w3 x)`).
#[derive(Clone, Debug)]
pub struct RefLayer {
    pub dim: usize,
    pub inter: usize,
    pub eps: f32,
    pub operator_norm: Vec<f32>,
    pub in_proj: Vec<f32>,
    pub conv: Vec<u16>,
    pub out_proj: Vec<f32>,
    pub post_operator_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub post_ffn_norm: Vec<f32>,
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, g)| round_bf16(v * r * g))
        .collect()
}

fn matvec(w: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    let k = x.len();
    (0..n)
        .map(|i| {
            round_bf16(
                w[i * k..(i + 1) * k]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum(),
            )
        })
        .collect()
}

impl RefLayer {
    pub fn generate(seed: u64, dim: usize, inter: usize) -> Self {
        let mut r = Rng::new(seed);
        let mut vals = |n: usize, s: f32| -> Vec<f32> {
            r.bf16_vec(n, s).into_iter().map(bf16_value).collect()
        };
        let operator_norm = vals(dim, 1.0)
            .into_iter()
            .map(|v| 1.0 + 0.25 * v)
            .map(round_bf16)
            .collect();
        let in_proj = vals(3 * dim * dim, 0.5);
        let out_proj = vals(dim * dim, 0.5);
        let post_operator_norm = vals(dim, 0.5);
        let ffn_norm = vals(dim, 1.0)
            .into_iter()
            .map(|v| 1.0 + 0.25 * v)
            .map(round_bf16)
            .collect();
        let gate = vals(inter * dim, 0.5);
        let up = vals(inter * dim, 0.5);
        let down = vals(dim * inter, 0.5);
        let post_ffn_norm = vals(dim, 0.5);
        let conv = Rng::new(seed ^ 0xC0).bf16_vec(dim * CONV_WIDTH, 0.6);
        Self {
            dim,
            inter,
            eps: 1e-5,
            operator_norm,
            in_proj,
            conv,
            out_proj,
            post_operator_norm,
            ffn_norm,
            gate,
            up,
            down,
            post_ffn_norm,
        }
    }

    /// A fresh window for one sequence.
    pub fn fresh_state(&self) -> Vec<u16> {
        vec![0u16; self.dim * STATE_LEN]
    }

    /// `t` tokens of one sequence (`hidden` is `[t, dim]`, updated in place),
    /// continuing from and advancing `state`.
    pub fn forward(&self, hidden: &mut [f32], state: &mut [u16], t: usize) {
        let d = self.dim;
        let mut bcx = vec![0u16; t * 3 * d];
        for i in 0..t {
            let a = rms_norm(&hidden[i * d..(i + 1) * d], &self.operator_norm, self.eps);
            for (j, v) in matvec(&self.in_proj, &a, 3 * d).into_iter().enumerate() {
                bcx[i * 3 * d + j] = bf16_bits(v);
            }
        }
        let y = prefill(&bcx, &self.conv, state, t, d);
        for i in 0..t {
            let yi: Vec<f32> = y[i * d..(i + 1) * d]
                .iter()
                .map(|b| bf16_value(*b))
                .collect();
            let o = rms_norm(
                &matvec(&self.out_proj, &yi, d),
                &self.post_operator_norm,
                self.eps,
            );
            let h = &mut hidden[i * d..(i + 1) * d];
            for (hv, ov) in h.iter_mut().zip(&o) {
                *hv = round_bf16(*hv + ov);
            }
            let f_in = rms_norm(h, &self.ffn_norm, self.eps);
            let g = matvec(&self.gate, &f_in, self.inter);
            let u = matvec(&self.up, &f_in, self.inter);
            let act: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(gv, uv)| round_bf16(gv / (1.0 + (-gv).exp()) * uv))
                .collect();
            let f = rms_norm(&matvec(&self.down, &act, d), &self.post_ffn_norm, self.eps);
            for (hv, fv) in h.iter_mut().zip(&f) {
                *hv = round_bf16(*hv + fv);
            }
        }
    }
}
