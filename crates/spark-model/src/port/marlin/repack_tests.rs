// SPDX-License-Identifier: AGPL-3.0-only
// The oracle is translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// vllm/model_executor/layers/quantization/utils/marlin_utils_test.py:27-89 (`marlin_permute_weights`,
// `marlin_weights`, `get_weight_perm`), Copyright contributors to the vLLM project, Apache-2.0.
// Changes: NumPy/torch reshapes restated as index loops in Rust, 4-bit only.

//! Responsibility: CPU tests of the Marlin layout: the repack (a translation of the CUDA repack
//! kernel) against vLLM's independent permutation formulation, every inverse as a round trip, the
//! GPTQ and AWQ parsers, and their refusals.
#![allow(clippy::useless_vec)]

use super::MarlinType;
use super::checkpoint::{
    self, AwqWeights, GptqWeights, GptqZeroFormat, ScaleDtype, awq_pack, gptq_pack,
    gptq_pack_zeros, le_bytes,
};
use super::reference::{LogicalInt4, f32_to_bf16};
use super::repack::{self, from_host, to_host};

/// A deterministic xorshift stream.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub(super) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub(super) fn code(&mut self) -> u8 {
        (self.next() % 16) as u8
    }

    /// A uniform value in `[-1, 1)`.
    pub(super) fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

pub(super) fn random_logical(n: usize, k: usize, group: usize, zp: bool, seed: u64) -> LogicalInt4 {
    let mut r = Rng::new(seed);
    let q = (0..k * n).map(|_| r.code()).collect();
    let groups = k / group;
    let scales = (0..groups * n)
        .map(|_| f32_to_bf16(0.001 + 0.05 * (r.unit() + 1.0)))
        .collect();
    let zeros = zp.then(|| (0..groups * n).map(|_| r.code()).collect());
    LogicalInt4 {
        n,
        k,
        group,
        q,
        scales,
        zeros,
    }
}

/// `get_weight_perm(4)`: 1024 indices, the interleave applied.
fn get_weight_perm() -> Vec<usize> {
    let mut perm = Vec::with_capacity(1024);
    for i in 0..32 {
        let mut perm1 = Vec::with_capacity(8);
        let col = i / 4;
        for block in [0usize, 1] {
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1.push(16 * row + col + 8 * block);
            }
        }
        for j in 0..4 {
            perm.extend(perm1.iter().map(|p| p + 256 * j));
        }
    }
    let mut out = Vec::with_capacity(1024);
    for chunk in perm.chunks_exact(8) {
        out.extend(repack::PACK_IDX.iter().map(|&i| chunk[i]));
    }
    out
}

/// `marlin_weights(q_w, k, n, 4, perm)`: tile permute, run permute, pack eight along the row.
fn oracle_marlin_weights(q: &[u8], k: usize, n: usize) -> Vec<u32> {
    let perm = get_weight_perm();
    let row_len = n * 16;
    // (k/16, 16, n/16, 16) -> permute (0, 2, 1, 3) -> (k/16, n*16).
    let mut tiled = vec![0u8; k * n];
    for kt in 0..k / 16 {
        for r in 0..16 {
            for nb in 0..n / 16 {
                for c in 0..16 {
                    tiled[kt * row_len + nb * 256 + r * 16 + c] =
                        q[(kt * 16 + r) * n + nb * 16 + c];
                }
            }
        }
    }
    let mut permuted = vec![0u8; k * n];
    for (dst, src) in permuted
        .chunks_exact_mut(1024)
        .zip(tiled.chunks_exact(1024))
    {
        for (j, &p) in perm.iter().enumerate() {
            dst[j] = src[p];
        }
    }
    // q_packed[:, c] |= q[:, 8c + i] << 4i over each (k/16) row.
    let mut out = vec![0u32; k * n / 8];
    for (w, chunk) in out.iter_mut().zip(permuted.chunks_exact(8)) {
        for (i, &v) in chunk.iter().enumerate() {
            *w |= u32::from(v) << (4 * i);
        }
    }
    out
}

const SHAPES: [(usize, usize); 4] = [(16, 64), (32, 128), (128, 256), (64, 192)];

#[test]
fn the_kernel_translation_equals_vllms_permutation_oracle() {
    for (seed, &(k, n)) in SHAPES.iter().enumerate() {
        let mut r = Rng::new(seed as u64 + 11);
        let q: Vec<u8> = (0..k * n).map(|_| r.code()).collect();
        let ours = repack::repack_codes(&q, k, n).unwrap();
        assert_eq!(ours, oracle_marlin_weights(&q, k, n), "[{k}x{n}]");
        // Planted control: one moved code must change the image.
        let mut moved = q.clone();
        moved.swap(0, 1);
        if moved != q {
            assert_ne!(
                repack::repack_codes(&moved, k, n).unwrap(),
                ours,
                "control [{k}x{n}]"
            );
        }
    }
}

#[test]
fn unrepack_inverts_repack_and_the_gptq_and_awq_kernels_agree() {
    for (seed, &(k, n)) in SHAPES.iter().enumerate() {
        let mut r = Rng::new(seed as u64 + 101);
        let q: Vec<u8> = (0..k * n).map(|_| r.code()).collect();
        let words = repack::repack_codes(&q, k, n).unwrap();
        assert_eq!(
            repack::unrepack_codes(&words, k, n).unwrap(),
            q,
            "[{k}x{n}]"
        );
        assert_eq!(
            repack::repack_gptq(&gptq_pack(&q, k, n), k, n).unwrap(),
            words
        );
        assert_eq!(
            repack::repack_awq(&awq_pack(&q, k, n), k, n).unwrap(),
            words
        );
    }
    assert!(
        repack::repack_codes(&[0u8; 8 * 64], 8, 64).is_err(),
        "K must be a multiple of 16"
    );
    assert!(
        repack::repack_codes(&[0u8; 16 * 32], 16, 32).is_err(),
        "N must be a multiple of 64"
    );
}

#[test]
fn scale_permutation_matches_the_formula_and_inverts() {
    let (perm, single) = repack::scale_perms();
    assert_eq!(&perm[..10], &[0, 8, 16, 24, 32, 40, 48, 56, 1, 9]);
    assert_eq!(&single[..10], &[0, 1, 8, 9, 16, 17, 24, 25, 2, 3]);
    // Grouped: [4 groups][128] scales, runs of 64.
    let s: Vec<u16> = (0..4 * 128).map(|i| i as u16).collect();
    let p = repack::permute_scales(&s, 512, 128, 128).unwrap();
    for run in 0..s.len() / 64 {
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(p[run * 64 + i * 8 + j], s[run * 64 + i + 8 * j]);
            }
        }
    }
    assert_eq!(repack::unpermute_scales(&p, 512, 128, 128).unwrap(), s);
    // One group: runs of 32 with the single permutation.
    let s1: Vec<u16> = (0..128).map(|i| 1000 + i as u16).collect();
    let p1 = repack::permute_scales(&s1, 128, 128, 128).unwrap();
    assert_eq!(p1[..8], [1000, 1001, 1008, 1009, 1016, 1017, 1024, 1025]);
    assert_eq!(repack::unpermute_scales(&p1, 128, 128, 128).unwrap(), s1);
}

#[test]
fn zero_points_pack_permuted_and_interleaved() {
    let (groups, n) = (3, 128);
    let mut r = Rng::new(7);
    let z: Vec<u8> = (0..groups * n).map(|_| r.code()).collect();
    let packed = repack::pack_zero_points(&z, groups, n).unwrap();
    assert_eq!(packed.len(), groups * n / 8);
    assert_eq!(repack::unpack_zero_points(&packed, groups, n).unwrap(), z);
    // Spot value: run 0 slot 0 is column 0; slot 1 of word 0 holds permuted entry 2 = column 16.
    let mut one = vec![0u8; groups * n];
    one[16] = 0xA;
    let w = repack::pack_zero_points(&one, groups, n).unwrap();
    assert_eq!(w[0], 0xA << 4);
}

#[test]
fn host_image_round_trips_for_both_types() {
    for zp in [false, true] {
        let l = random_logical(256, 512, 128, zp, 3);
        let h = to_host(&l).unwrap();
        assert_eq!(h.btype, if zp { MarlinType::U4 } else { MarlinType::U4B8 });
        assert_eq!(h.words.len(), 512 * 256 / 8);
        assert_eq!(h.zeros.as_ref().map(Vec::len), zp.then_some(4 * 256 / 8));
        assert_eq!(from_host(&h).unwrap(), l);
    }
    let mut bad = random_logical(64, 128, 128, false, 1);
    bad.q[5] = 16;
    assert!(to_host(&bad).is_err(), "codes are 4-bit");
}

#[test]
fn gptq_parses_codes_scales_and_both_zero_formats() {
    let (n, k, group) = (128, 256, 128);
    let l = random_logical(n, k, group, true, 5);
    let qweight = le_bytes(&gptq_pack(&l.q, k, n));
    let scales: Vec<u8> = l.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    for format in [GptqZeroFormat::V1, GptqZeroFormat::V2] {
        let zeros = le_bytes(&gptq_pack_zeros(
            l.zeros.as_ref().unwrap(),
            k / group,
            n,
            format,
        ));
        let w = GptqWeights {
            qweight: &qweight,
            scales: &scales,
            scale_dtype: ScaleDtype::Bf16,
            qzeros: Some(&zeros),
            g_idx: None,
            n,
            k,
            group,
            zero_format: format,
        };
        assert_eq!(checkpoint::gptq_to_logical(&w).unwrap(), l, "{format:?}");
    }
    // Symmetric: stored 7 in v1 means 8, which collapses to kU4B8.
    let sym_zeros = le_bytes(&gptq_pack_zeros(
        &vec![8u8; 2 * n],
        2,
        n,
        GptqZeroFormat::V1,
    ));
    assert!(
        sym_zeros
            .chunks_exact(4)
            .all(|c| c == [0x77, 0x77, 0x77, 0x77])
    );
    let w = GptqWeights {
        qweight: &qweight,
        scales: &scales,
        scale_dtype: ScaleDtype::Bf16,
        qzeros: Some(&sym_zeros),
        g_idx: None,
        n,
        k,
        group,
        zero_format: GptqZeroFormat::V1,
    };
    let sym = checkpoint::gptq_to_logical(&w).unwrap();
    assert_eq!(sym.zeros, None);
    assert_eq!(sym.btype(), MarlinType::U4B8);
    assert_eq!(sym.q, l.q);
}

#[test]
fn gptq_refuses_act_order_and_bad_sizes() {
    let (n, k, group) = (64, 256, 128);
    let l = random_logical(n, k, group, false, 9);
    let qweight = le_bytes(&gptq_pack(&l.q, k, n));
    let scales: Vec<u8> = l.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let trivial: Vec<u32> = (0..k as u32).map(|i| i / group as u32).collect();
    let trivial = le_bytes(&trivial);
    let mut w = GptqWeights {
        qweight: &qweight,
        scales: &scales,
        scale_dtype: ScaleDtype::Bf16,
        qzeros: None,
        g_idx: Some(&trivial),
        n,
        k,
        group,
        zero_format: GptqZeroFormat::V1,
    };
    assert_eq!(checkpoint::gptq_to_logical(&w).unwrap(), l);
    let mut desc: Vec<u32> = (0..k as u32).map(|i| i / group as u32).collect();
    desc.swap(0, k - 1);
    let desc = le_bytes(&desc);
    w.g_idx = Some(&desc);
    let e = checkpoint::gptq_to_logical(&w).unwrap_err().to_string();
    assert!(e.contains("act-order"), "{e}");
    w.g_idx = None;
    w.qweight = &qweight[..qweight.len() - 4];
    assert!(checkpoint::gptq_to_logical(&w).is_err());
}

#[test]
fn awq_parses_the_interleaved_layout() {
    let (n, k, group) = (128, 256, 128);
    let l = random_logical(n, k, group, true, 13);
    let qweight = le_bytes(&awq_pack(&l.q, k, n));
    let qzeros = le_bytes(&awq_pack(l.zeros.as_ref().unwrap(), k / group, n));
    let scales: Vec<u8> = l.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let w = AwqWeights {
        qweight: &qweight,
        qzeros: &qzeros,
        scales: &scales,
        scale_dtype: ScaleDtype::Bf16,
        n,
        k,
        group,
    };
    assert_eq!(checkpoint::awq_to_logical(&w).unwrap(), l);
    // AWQ's documented interleave: column 1 sits in nibble 4 of word 0.
    let mut single = vec![0u8; n];
    single[1] = 0x3;
    assert_eq!(awq_pack(&single, 1, n)[0], 0x3 << 16);
}

#[test]
fn fp16_scales_convert_and_non_finite_scales_are_refused() {
    let one_f16 = 0x3C00u16.to_le_bytes();
    let s = checkpoint::scales_bf16(&one_f16, 1, ScaleDtype::F16).unwrap();
    assert_eq!(s, [0x3F80]);
    let inf = 0x7C00u16.to_le_bytes();
    assert!(checkpoint::scales_bf16(&inf, 1, ScaleDtype::F16).is_err());
    assert!(checkpoint::scales_bf16(&[0u8; 3], 1, ScaleDtype::Bf16).is_err());
}
