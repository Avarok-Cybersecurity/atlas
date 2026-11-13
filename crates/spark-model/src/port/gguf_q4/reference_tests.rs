// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the prefill side of the keep-packed GGUF oracle (port lane L19): the MMQ
//! activation mirror in its D4 and DS4 layouts, the Q4_K sub-block mins against spark-runtime's decoder,
//! the Q4_K partial-sum correction, and the prefill bound.

use super::launch::Q4Type;
use super::reference::{
    MmqLayout, bf16_to_f32, bound, dequant_weight, grade_with, mmq_layout, prefill_bound,
    q4_k_mins, q4_k_sum_correction, quantize_q8_1_mmq_row,
};

/// xorshift64*: deterministic bytes.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

/// Random Q4_K bytes of `[n, k]` with small positive `d` and `dmin`.
fn random_q4_k(rng: &mut Rng, n: usize, k: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * (k / 256) * 144);
    for _ in 0..n * (k / 256) {
        let start = v.len();
        v.extend((0..144).map(|_| (rng.next() >> 56) as u8));
        let d = half::f16::from_f32(0.001 + 0.01 * rng.unit().abs()).to_le_bytes();
        let dmin = half::f16::from_f32(0.001 + 0.004 * rng.unit().abs()).to_le_bytes();
        v[start..start + 2].copy_from_slice(&d);
        v[start + 2..start + 4].copy_from_slice(&dmin);
    }
    v
}

#[test]
fn the_mmq_mirror_keeps_d_in_f32_for_d4_and_f16_for_ds4() {
    let mut row = vec![0u16; 64];
    row[0] = bf16(0.3);
    row[5] = bf16(-0.1);
    let d4 = quantize_q8_1_mmq_row(&row, MmqLayout::D4);
    let ds4 = quantize_q8_1_mmq_row(&row, MmqLayout::Ds4);
    let amax = bf16_to_f32(bf16(0.3));
    let d = 1.0f32 / (127.0f32 / amax);
    assert_eq!(d4[0].d, d);
    assert_eq!(ds4[0].d, half::f16::from_f32(d).to_f32());
    assert_eq!(d4[0].qs, ds4[0].qs, "the codes do not depend on the layout");
    assert_eq!(d4[0].qs[0], 127);
    // The partial sum is of the original values, stored as f16.
    let sum = bf16_to_f32(bf16(0.3)) + bf16_to_f32(bf16(-0.1));
    assert_eq!(ds4[0].s, half::f16::from_f32(sum).to_f32());
    // An all-zero block: NaN codes become 0 and the scale is 0, as on the GPU.
    assert_eq!((d4[1].qs, d4[1].d), ([0i8; 32], 0.0));
    assert_eq!(mmq_layout(Q4Type::Q4K), MmqLayout::Ds4);
    assert_eq!(mmq_layout(Q4Type::Iq4Xs), MmqLayout::D4);
    assert_eq!(mmq_layout(Q4Type::Q6K), MmqLayout::D4);
}

#[test]
fn q4_k_mins_are_the_decoders_constant_part() {
    let mut rng = Rng(0x0004_000b);
    let (n, k) = (3, 512);
    let raw = random_q4_k(&mut rng, n, k);
    let mins = q4_k_mins(&raw, n, k);
    assert_eq!(mins.len(), n * k / 32);
    // With every code zero the decoder returns `-min` for the whole sub-block.
    let mut zeroed = raw.clone();
    for blk in zeroed.chunks_exact_mut(144) {
        blk[16..].fill(0);
    }
    let w = dequant_weight(Q4Type::Q4K, &zeroed, n, k).unwrap();
    for (i, v) in w.iter().enumerate() {
        assert_eq!(*v, -mins[i / 32], "value {i}");
    }
}

#[test]
fn the_sum_correction_is_min_times_the_partial_sum_gap() {
    let mut rng = Rng(0x0004_000c);
    let (n, k) = (3, 512);
    let mins = q4_k_mins(&random_q4_k(&mut rng, n, k), n, k);
    let xb: Vec<u16> = (0..k).map(|_| bf16(rng.unit())).collect();
    let rows = vec![quantize_q8_1_mmq_row(&xb, MmqLayout::Ds4)];
    let (corr, slack) = q4_k_sum_correction(&mins, n, k, &rows);
    // Zero when the stored partial sum equals d * sum(q): a power-of-two scale makes that exact in f32.
    let mut exact = rows.clone();
    for b in &mut exact[0] {
        b.d = 1.0 / 128.0;
        b.s = b.d * b.qs.iter().map(|&q| f32::from(q)).sum::<f32>();
    }
    let (zero, _) = q4_k_sum_correction(&mins, n, k, &exact);
    for j in 0..n {
        let want: f64 = (0..k / 32)
            .map(|b| {
                let q8 = &rows[0][b];
                let sq: i32 = q8.qs.iter().map(|&q| i32::from(q)).sum();
                f64::from(mins[j * 16 + b]) * (f64::from(q8.d) * f64::from(sq) - f64::from(q8.s))
            })
            .sum();
        assert_eq!(corr[j], want);
        assert_eq!(zero[j], 0.0, "row {j}");
        assert!(slack[j] > 0.0);
    }
}

#[test]
fn the_prefill_bound_is_looser_and_grade_with_uses_it() {
    assert!(prefill_bound(1.0, 10.0, 512) > bound(1.0, 10.0, 512));
    // An error between the two bounds passes the prefill grade and fails the decode grade.
    let (want, mag, k) = (1.0f64, 100.0f64, 512usize);
    let err = (bound(want, mag, k) + prefill_bound(want, mag, k)) / 2.0;
    let got = [(want + err) as f32];
    assert_eq!(grade_with(&got, &[want], &[mag], k, prefill_bound).0, 0);
    assert_eq!(grade_with(&got, &[want], &[mag], k, bound).0, 1);
}
