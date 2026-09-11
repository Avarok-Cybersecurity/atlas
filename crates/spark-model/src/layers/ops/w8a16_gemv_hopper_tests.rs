// SPDX-License-Identifier: AGPL-3.0-only

//! HOST SIMULATION of the Hopper-tuned `w8a16_gemv`'s index and pipeline math
//! against the gb10 kernel's reduction order (#928).
//!
//! `kernels/hopper/common/w8a16_gemv.cu` is a target override: same entry
//! point, same `ceil(N/4)` x 256 launch, and — the claim this file exists to
//! test — the same arithmetic in the same order. It changes two things, and
//! only one of them is checkable without a GPU:
//!
//!  * the DEQUANT INSTRUCTION (a shared-memory `E4M3_LUT` gather becomes
//!    `cvt.rn.f16x2.e4m3x2`). That is a hardware-semantics claim; the receipt
//!    is `examples/native_fp8_gemv_hopper_microtest.rs`, which runs both
//!    kernels on a device and asserts `unequal=0`. Nothing here touches it —
//!    [`decode`] below stands in for BOTH paths precisely so that a difference
//!    reported by this file can only be an ORDERING difference.
//!  * the LOOP SHAPE: [`UNROLL`] chunk loads are issued before the first is
//!    consumed, so a warp has that many outstanding weight requests instead of
//!    one (`w8a16_gemv_hopper.cuh`, DIAGNOSIS 2). Reordering LOADS is free;
//!    reordering ACCUMULATION is not, because every batch oracle in the tree
//!    compares against this exact FP32 chain. That is what is pinned here.
//!
//! [`hopper_row`] is a transcription of the override's loop — the `UNROLL`-wide
//! prefetch body plus the one-at-a-time tail — and [`reference_row`] is the
//! gb10 loop. They are asserted to emit the same chunk sequence and the same
//! FP32 accumulator, at K values that exercise every relevant residue: an exact
//! multiple of `UNROLL * LANES` chunks, a tail shorter than one unroll group,
//! and a K with a partial chunk group in the middle of a scale block.
//!
//! Both assertions are shown to have teeth: [`rotated_row`] is the same code
//! with the unroll group consumed in a different order — the one mistake the
//! rewrite could actually make — and it is required to DIFFER.

use half::bf16;

/// `w8a16_gemv.cu`'s `threads_per_out` (256 / `N_PER_BLOCK`).
const LANES: usize = 64;
/// `w8a16_gemv_hopper.cuh`'s `HOPPER_GEMV_UNROLL`.
const UNROLL: usize = 4;
/// K values one lane consumes per chunk (`K_PER_CHUNK`).
const K_PER_CHUNK: usize = 16;
/// Chunks per 128-wide FP8 scale block (`CHUNKS_PER_SCALE`).
const CHUNKS_PER_SCALE: usize = 8;

/// Stand-in for the E4M3 decode, identical on both sides. The kernels differ
/// in the INSTRUCTION that produces this value, not in the value — see the
/// module note.
fn decode(byte: u8) -> f32 {
    // Sign-symmetric, exactly representable, and spread over three decades, so
    // an accumulation-order difference shows up as unequal bits rather than
    // cancelling.
    let mag = f32::from(byte & 0x7F) * 0.015_625 + 0.001_953_125;
    if byte & 0x80 != 0 { -mag } else { mag }
}

/// One lane's chunk sequence under the gb10 loop: `k16 = lane; k16 < n; k16 += 64`.
fn reference_chunks(lane: usize, chunk_count: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut k16 = lane;
    while k16 < chunk_count {
        out.push(k16);
        k16 += LANES;
    }
    out
}

/// The override's chunk sequence: an `UNROLL`-wide prefetch body whose group is
/// consumed in issue order, then the same one-at-a-time tail.
fn hopper_chunks(lane: usize, chunk_count: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut k16 = lane;
    while k16 + (UNROLL - 1) * LANES < chunk_count {
        for u in 0..UNROLL {
            out.push(k16 + u * LANES);
        }
        k16 += UNROLL * LANES;
    }
    while k16 < chunk_count {
        out.push(k16);
        k16 += LANES;
    }
    out
}

/// The same body with each unroll group consumed last-to-first — a LOAD
/// reordering promoted to an ACCUMULATION reordering, which is the mistake
/// the rewrite could make and the thing the assertions must be able to see.
fn rotated_chunks(lane: usize, chunk_count: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut k16 = lane;
    while k16 + (UNROLL - 1) * LANES < chunk_count {
        for u in (0..UNROLL).rev() {
            out.push(k16 + u * LANES);
        }
        k16 += UNROLL * LANES;
    }
    while k16 < chunk_count {
        out.push(k16);
        k16 += LANES;
    }
    out
}

/// One lane's FP32 accumulator over a given chunk sequence, in the kernels'
/// per-chunk operand order: bytes 0..3 of each 4-byte group against
/// activations 0..3, `w = decode(byte) * scale`, then `acc += a * w`.
///
/// `--fmad=false` is tree-wide (`kernels/gb10/common/KERNEL.toml`), so the
/// multiply and the add are separate FP32 roundings on the device too, and
/// plain `+`/`*` here is the faithful model — `mul_add` would NOT be.
fn lane_acc(chunks: &[usize], weights: &[u8], act: &[f32], scales: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for &k16 in chunks {
        let scale = scales[k16 / CHUNKS_PER_SCALE];
        for i in 0..K_PER_CHUNK {
            let w = decode(weights[k16 * K_PER_CHUNK + i]) * scale;
            acc += act[k16 * K_PER_CHUNK + i] * w;
        }
    }
    acc
}

/// The kernels' two-stage reduction: a 5-step `shfl.down` butterfly inside each
/// 32-lane warp, then the two warps' lane-0 partials added through smem, then
/// ONE round to BF16.
fn reduce(partials: &[f32; LANES]) -> bf16 {
    let warp = |base: usize| {
        let mut v = [0.0_f32; 32];
        v.copy_from_slice(&partials[base..base + 32]);
        let mut off = 16;
        while off > 0 {
            for i in 0..32 - off {
                v[i] += v[i + off];
            }
            off >>= 1;
        }
        v[0]
    };
    bf16::from_f32(warp(0) + warp(32))
}

/// One output row, for a chunk-sequence rule.
fn row(
    seq: fn(usize, usize) -> Vec<usize>,
    chunk_count: usize,
    weights: &[u8],
    act: &[f32],
    scales: &[f32],
) -> bf16 {
    let mut partials = [0.0_f32; LANES];
    for (lane, p) in partials.iter_mut().enumerate() {
        *p = lane_acc(&seq(lane, chunk_count), weights, act, scales);
    }
    reduce(&partials)
}

fn reference_row(n: usize, w: &[u8], a: &[f32], s: &[f32]) -> bf16 {
    row(reference_chunks, n, w, a, s)
}
fn hopper_row(n: usize, w: &[u8], a: &[f32], s: &[f32]) -> bf16 {
    row(hopper_chunks, n, w, a, s)
}
fn rotated_row(n: usize, w: &[u8], a: &[f32], s: &[f32]) -> bf16 {
    row(rotated_chunks, n, w, a, s)
}

/// K values that exercise the loop's residues. 16,384 chunks is 64 whole
/// unroll groups per lane; 5,120 (K=81,920) leaves a 16-chunk tail; 328
/// (K=5,248) is shorter than two unroll groups and straddles a scale block.
const CHUNK_COUNTS: [usize; 4] = [16_384, 5_120, 328, 256];

fn fixture(chunk_count: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let n = chunk_count * K_PER_CHUNK;
    let mut state = 0x0928_2026_5a5a_0011_u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as u32
    };
    let weights: Vec<u8> = (0..n).map(|_| (next() & 0xFF) as u8).collect();
    let act: Vec<f32> = (0..n)
        .map(|_| bf16::from_f32((next() % 2049) as f32 / 1024.0 - 1.0).to_f32())
        .collect();
    let scales: Vec<f32> = (0..chunk_count.div_ceil(CHUNKS_PER_SCALE))
        .map(|_| (next() % 16 + 1) as f32 / 1024.0)
        .collect();
    (weights, act, scales)
}

#[test]
fn the_unrolled_loop_visits_the_same_chunks_in_the_same_order() {
    for count in CHUNK_COUNTS {
        for lane in [0, 1, 31, 32, 63] {
            assert_eq!(
                hopper_chunks(lane, count),
                reference_chunks(lane, count),
                "lane {lane}, {count} chunks"
            );
        }
    }
}

#[test]
fn every_chunk_is_visited_exactly_once_across_the_64_lanes() {
    for count in CHUNK_COUNTS {
        let mut seen: Vec<usize> = (0..LANES).flat_map(|l| hopper_chunks(l, count)).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..count).collect::<Vec<_>>(), "{count} chunks");
    }
}

#[test]
fn the_scale_index_is_the_gb10_expression() {
    // The kernel folds `(k16 * K_PER_CHUNK) / FP8_BLOCK` to `k16 / 8`.
    for k16 in 0..4096_usize {
        assert_eq!(k16 / CHUNKS_PER_SCALE, (k16 * K_PER_CHUNK) / 128);
    }
}

#[test]
fn the_unrolled_accumulator_is_bit_identical_to_the_gb10_order() {
    for count in CHUNK_COUNTS {
        let (w, a, s) = fixture(count);
        assert_eq!(
            hopper_row(count, &w, &a, &s).to_bits(),
            reference_row(count, &w, &a, &s).to_bits(),
            "{count} chunks (K={})",
            count * K_PER_CHUNK
        );
    }
}

/// ...and the comparison is not vacuous: consuming the prefetched group in a
/// different order changes the chunk sequence AND the FP32 result, at every K
/// long enough to contain a whole unroll group.
#[test]
fn reordering_the_unroll_group_is_detected() {
    let mut differing = 0;
    for count in CHUNK_COUNTS {
        assert_ne!(
            rotated_chunks(0, count),
            reference_chunks(0, count),
            "{count} chunks: the mutation did not even change the sequence"
        );
        let (w, a, s) = fixture(count);
        if rotated_row(count, &w, &a, &s).to_bits() != reference_row(count, &w, &a, &s).to_bits() {
            differing += 1;
        }
    }
    assert_eq!(
        differing,
        CHUNK_COUNTS.len(),
        "an accumulation reordering went unnoticed at some K"
    );
}
