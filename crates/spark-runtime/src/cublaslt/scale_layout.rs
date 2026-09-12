// SPDX-License-Identifier: AGPL-3.0-only

//! SSOT for the cuBLASLt block-scaling factor layouts — what the library
//! documents, mirrored as index math the CUDA adapter kernel and the CPU tests
//! share.
//!
//! WHY THIS FILE EXISTS. Measured on 1xH100 (2026-09-11 07:15Z,
//! `native_fp8_ffn_w8a8_microtest`, tip `5f78270dc`, cuBLASLt 13.1): the
//! [`super::fp8_gemm_act_weight_t_blkscaled`] arm ran at 1140 TFLOP/s but
//! disagreed with the in-tree `fp8_gemm_t_blockscaled` on the SAME quantized
//! inputs — rel_rms 1.1e-2 / cosine 0.99994 at M=64, 7.7e-2-8.8e-2 / cosine
//! 0.996 at M=1193, ~33000 BF16 ULP, 84-94% of elements unequal. Two correct
//! W8A8 implementations over identical FP8 bytes with FP32 accumulation agree
//! to ~1e-3, and the error grew with M, so the defect was a scale-tensor
//! layout, not arithmetic. It was: the activation (VEC128) scale tensor was
//! handed over in the quantizer's `[M, K/128]` order, and cuBLASLt reads that
//! operand's scales MN-major.
//!
//! THE DOCUMENTED RULES (cuBLAS 13.4 manual, "128-element 1D and 128x128 2D
//! Block Scaling For FP8 Data Types" and its "Scaling factors layouts"
//! subsection):
//!
//! * Supported mode pairs are VEC128/VEC128, VEC128/BLK128x128 and
//!   BLK128x128/VEC128; BLK128x128 on BOTH A and B is listed unsupported. The
//!   A=BLK128x128 (weight) + B=VEC128 (activation) pairing Atlas uses is
//!   therefore legal as written — the pairing was never the bug.
//! * Scaling-factor start addresses must be 16 B aligned, and the matmul's M
//!   and N "must be multiples of 4" — which is what the caller's ceil16(M) pad
//!   satisfies for the token dimension.
//! * VEC128_32F: the factors are "M-major for A with shape M x L" and
//!   "N-major for B with shape N x L", where `L = ceil(K/128)` and major means
//!   that dimension is contiguous. So for B the token index is contiguous and
//!   the K-group index strides by the (padded) token count — the TRANSPOSE of
//!   the `[M, K/128]` the quantizer writes.
//! * BLK128x128_32F: the factors are K-major, "the stride between the
//!   consecutive columns must be a multiple of 4", shape `L4 x ceil(M/128)`
//!   for A (`L4 x ceil(N/128)` for B) with `L4` = L rounded up to a multiple
//!   of 4. K-major with `ceil(M/128)` columns IS the checkpoint's row-major
//!   `[N/128, K/128]` weight-scale grid whenever L is already a multiple of 4
//!   — see [`blk128x128_stride_ok`]. The weight side needed no change.
//!
//! Atlas maps `out[M,N] = act[M,K] @ weight[N,K]ᵀ` onto cuBLASLt as
//! `D[N,M] = opT(weightᶜ[K,N]) · opN(actᶜ[K,M])`, so the library's M is the
//! weight's N and the library's N is the token count. Read the doc quotes
//! above with that substitution: the VEC128 "N-major" operand is the
//! activation, and its contiguous dimension is tokens.

/// Number of 128-wide K groups a K extent carries (`L` in the cuBLAS docs).
#[must_use]
pub fn k_groups(k: usize) -> usize {
    k.div_ceil(128)
}

/// Offset of the VEC128 scale for `(token, k_group)` in the layout cuBLASLt
/// documents for the B operand: shape `N x L`, N-major, N = `m_pad` tokens.
///
/// This is the SSOT the CUDA adapter `fp8_act_scale_to_kmajor` mirrors.
#[must_use]
pub const fn vec128_b_index(m_pad: usize, token: usize, k_group: usize) -> usize {
    k_group * m_pad + token
}

/// Offset of the same `(token, k_group)` scale in the layout
/// `per_token_group_quant_fp8` writes: row-major `[M, K/128]`, K-group
/// contiguous. The in-tree `fp8_gemm_t_blockscaled` indexes this one.
#[must_use]
pub const fn rowmajor_index(l: usize, token: usize, k_group: usize) -> usize {
    token * l + k_group
}

/// FP32 element count of the VEC128 B-scale tensor cuBLASLt reads.
#[must_use]
pub fn vec128_b_elems(m_pad: usize, k: usize) -> usize {
    m_pad * k_groups(k)
}

/// Whether a checkpoint's row-major `[N/128, K/128]` weight-scale grid already
/// satisfies the BLK128x128 column-stride rule ("must be a multiple of 4"),
/// i.e. whether `L = ceil(K/128)` needs no padding to `L4`.
///
/// True for every shape Atlas serves today (K=5120 → L=40, K=17408 → L=136),
/// which is why the weight scales pass through untouched. A K that breaks it
/// would need a padded copy, so the dispatch gate checks this rather than
/// assuming it.
#[must_use]
pub fn blk128x128_stride_ok(k: usize) -> bool {
    k_groups(k).is_multiple_of(4)
}

/// CPU reference for the `fp8_act_scale_to_kmajor` CUDA kernel: read the
/// quantizer's row-major `[m, l]` scales, write cuBLASLt's `[l, m_pad]`, with
/// the `m..m_pad` pad slots zeroed (their FP8 activation bytes are zeroed too,
/// so the phantom rows contribute a defined zero).
///
/// Exists so the layout can be pinned by a unit test on any host — the GPU
/// kernel is one line of index math and this is that line, in Rust.
#[must_use]
pub fn act_scale_rowmajor_to_kmajor(src: &[f32], m: usize, m_pad: usize, l: usize) -> Vec<f32> {
    debug_assert!(m_pad >= m, "pad cannot shrink the token extent");
    debug_assert!(src.len() >= m * l, "source holds fewer than m x l scales");
    let mut dst = vec![0.0f32; vec128_b_elems(m_pad, l * 128)];
    for token in 0..m {
        for kg in 0..l {
            dst[vec128_b_index(m_pad, token, kg)] = src[rowmajor_index(l, token, kg)];
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known values, hand-placed. `src` is `[M=3, L=2]` row-major, so reading
    /// it as-is gives `[1, 2, 3, 4, 5, 6]`; the VEC128 B layout wants tokens
    /// contiguous within each K group.
    #[test]
    fn kmajor_adapter_transposes_and_zero_fills_the_pad() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dst = act_scale_rowmajor_to_kmajor(&src, 3, 4, 2);
        // group 0: tokens 0..3 then the pad token; group 1: the same.
        assert_eq!(dst, vec![1.0, 3.0, 5.0, 0.0, 2.0, 4.0, 6.0, 0.0]);
    }

    #[test]
    fn adapter_is_identity_free_only_when_one_dimension_is_trivial() {
        // A single K group or a single token means the two orders coincide —
        // useful as a sanity anchor, and the reason a 1-token decode never
        // showed the bug.
        let src = [7.0f32, 8.0, 9.0];
        assert_eq!(
            act_scale_rowmajor_to_kmajor(&src, 3, 3, 1),
            vec![7.0, 8.0, 9.0]
        );
        assert_eq!(
            act_scale_rowmajor_to_kmajor(&src, 1, 1, 3),
            vec![7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn the_two_readings_disagree_at_the_measured_shape() {
        // WHY: this is the H100 failure, reduced. Same buffer, two readings;
        // at M=64/K=5120 the offsets coincide only where 63*kg == 39*token, so
        // all but 4 of the 2560 scales land on the wrong element. A permutation
        // of plausible per-token scale values is "wrong but not garbage" —
        // cosine 0.99994, not 0, which is what made this look like precision.
        let (m_pad, l) = (64usize, 40usize);
        let same = (0..m_pad)
            .flat_map(|token| (0..l).map(move |kg| (token, kg)))
            .filter(|(token, kg)| {
                vec128_b_index(m_pad, *token, *kg) == rowmajor_index(l, *token, *kg)
            })
            .count();
        assert_eq!(same, 4, "only (0,0), (21,13), (42,26), (63,39) coincide");
        assert_eq!(m_pad * l - same, 2556);
    }

    #[test]
    fn documented_extents_match_the_shapes_we_serve() {
        // Qwen3.8-27B dense FFN: gate/up K=5120, down K=17408.
        assert_eq!(k_groups(5120), 40);
        assert_eq!(k_groups(17408), 136);
        assert!(blk128x128_stride_ok(5120));
        assert!(blk128x128_stride_ok(17408));
        // A K whose group count is not a multiple of 4 would need the L4 pad.
        assert!(!blk128x128_stride_ok(128 * 5));
        assert_eq!(vec128_b_elems(1200, 5120), 1200 * 40);
    }
}
