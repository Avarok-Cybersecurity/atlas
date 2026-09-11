// SPDX-License-Identifier: AGPL-3.0-only

//! HOST SIMULATION of the `w8a16_gemm_m16` two-halves rung at round 6's
//! `gate/up M=32` geometry — the test that settles what that red cell was.
//!
//! ROUND 6 (1xH100, 2026-09-11, `native_fp8_ffn_m16_tc_microtest`): eleven of
//! twelve cells green, and `gate/up N=17408 K=5120 M=32` RED — `max_ulp 28`,
//! `over_budget 5` of 557,056 elements, `sign_flips 0`, `rel_rms 4.237e-5`,
//! `max_abs 0.500`. `down M=32` was green, as was every M <= 16 cell. M=32 is
//! the two-halves rung (`first = m.div_ceil(2)` = 16 + 16), so the first
//! suspicion was a row or pitch offset on the second half.
//!
//! IT IS NOT. Two independent things are pinned here, and the second is the
//! one that explains the cell:
//!
//! 1. **The offsets are right.** [`halves_route`] applies exactly the byte
//!    offsets `DenseFfnLayer::w8a16_m16_tc_proj` applies (`first * k` on the
//!    input, `first * n` on the output), and the rows it produces are
//!    BIT-IDENTICAL to evaluating each row on its own. Both a wrong input
//!    offset and a wrong output offset are shown to break that, so the
//!    assertion is not vacuous.
//!
//! 2. **The red cell is the oracle's metric.** A full-geometry host run of the
//!    microtest's own data generator (K=5120, N=17408, M=32, seed
//!    `0x927_16_7C_2026`) reproduces the signature with no offset arithmetic
//!    anywhere: 5 elements over the 2-ULP budget, none of them in rows 0..15,
//!    `sign_flips` 0. Every one is an output that cancelled to |ref| between
//!    5.7e-6 and 1.6e-4 against a reference RMS of 39.1 — 1e-7..4e-6 of the
//!    matrix scale — with absolute errors of 3.8e-6..1.05e-5. M=32 trips it and
//!    M=16 does not because M=32 samples twice as many outputs; gate/up trips it
//!    and down does not because gate/up has 3.4x the columns. The cases below
//!    pin the round-6 numbers through [`within_m16_tc_budget`] in BOTH
//!    directions: the cancelled elements pass, and an error of structural size
//!    does not.
//!
//! The simulation is faithful to the two reduction orders it compares:
//! [`gemv_reference`] is `w8a16_gemv`'s (64 lanes striding by 64 over 16-wide
//! K chunks, then the shfl butterfly and the two-warp add) and [`mma_row`] is
//! `w8a16_gemm_m16`'s (four m16n8k16 sub-steps per 64-K step, two K-steps per
//! 128-wide scale block, the block scale folded ONCE onto an FP32 outer
//! accumulator). The MMA's internal 16-product order is unspecified hardware, so
//! the sequential model here is one plausible realisation — which is the point:
//! the tail does not depend on picking the right one.
//!
//! N is the sampled width, not 17408: what governs the cancellation is the
//! REDUCTION DEPTH, and K is the real 5120 (40 whole 128-wide scale blocks).
//! Running the real N here would be a 2.9-GMAC unit test.

use super::{M16_TC_ACC_FLOOR, M16TcPlan, bf16_ord, m16_tc_plan, within_m16_tc_budget};
use half::bf16;

/// gate/up's real reduction depth — 40 whole 128-wide FP8 scale blocks.
const K: usize = 5120;
/// Sampled output width (whole scale blocks). See the module note on N.
const N: usize = 128;
/// The rung under test.
const M: usize = 32;
/// The microtest's seed, so this simulation draws the same alphabet. Written
/// `0x0927_167C_2026` rather than the oracle's `0x927_16_7C_2026` only because
/// clippy wants equal-sized groups; it is the same value, and the mnemonic is
/// still #927 / M=16 / TC / 2026.
const SEED: u64 = 0x0927_167C_2026;
const FP8_BLOCK: usize = 128;

/// The microtest's generator, bit for bit.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

/// E4M3 -> f32, matching `E4M3_LUT` (0x7F/0xFF, the format's only NaNs, decode
/// to +-0 — and the generator never draws them, exactly as the oracle's does
/// not).
fn e4m3(byte: u8) -> f32 {
    if byte & 0x7F == 0x7F {
        return 0.0;
    }
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0 };
    let exp = i32::from((byte >> 3) & 0xF);
    let mant = f32::from(byte & 0x7) / 8.0;
    if exp == 0 {
        sign * mant * 2.0_f32.powi(-6)
    } else {
        sign * (1.0 + mant) * 2.0_f32.powi(exp - 7)
    }
}

struct Fixture {
    /// `[N, K]` FP8 E4M3, already dequantized (the LUT is exact in f32).
    weight: Vec<f32>,
    /// `[M, K]` activations as the BF16 values the kernel actually sees.
    acts: Vec<f32>,
    /// `[N/128, K/128]` FP32 block scales.
    scales: Vec<f32>,
}

impl Fixture {
    /// Draws in the microtest's ORDER (weights, then activations, then scales)
    /// so the fixture is the same distribution the H100 measured.
    fn new() -> Self {
        let mut rng = Rng(SEED);
        let weight = (0..N * K)
            .map(|_| {
                let x = rng.next();
                e4m3(((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128))
            })
            .collect();
        let acts = (0..M * K)
            .map(|_| bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0).to_f32())
            .collect();
        let scales = (0..(N / FP8_BLOCK) * (K / FP8_BLOCK))
            .map(|_| ((rng.next() % 16 + 1) as f32) / 1024.0)
            .collect();
        Self {
            weight,
            acts,
            scales,
        }
    }

    fn scale(&self, col: usize, k_block: usize) -> f32 {
        self.scales[(col / FP8_BLOCK) * (K / FP8_BLOCK) + k_block]
    }
}

/// `w8a16_gemv`'s reduction: 64 threads per output, each walking 16-wide K
/// chunks with a stride of 64, then a 32-lane shfl butterfly per warp and one
/// add across the two warps. The block scale multiplies each WEIGHT, per
/// element, which is the other half of why this and the MMA differ.
fn gemv_reference(f: &Fixture, row: &[f32], col: usize) -> u16 {
    let chunks = K / 16;
    let mut lanes = [0.0_f32; 64];
    for (lane, acc) in lanes.iter_mut().enumerate() {
        let mut chunk = lane;
        while chunk < chunks {
            for i in 0..16 {
                let k = chunk * 16 + i;
                *acc += row[k] * (f.weight[col * K + k] * f.scale(col, k / FP8_BLOCK));
            }
            chunk += 64;
        }
    }
    let mut warps = [0.0_f32; 2];
    for (w, out) in warps.iter_mut().enumerate() {
        let mut v = [0.0_f32; 32];
        v.copy_from_slice(&lanes[w * 32..(w + 1) * 32]);
        let mut off = 16;
        while off > 0 {
            for l in 0..off {
                v[l] += v[l + off];
            }
            off >>= 1;
        }
        *out = v[0];
    }
    bf16::from_f32(warps[0] + warps[1]).to_bits()
}

/// `w8a16_gemm_m16`'s reduction: eight 16-wide MMA sub-steps per 128-K scale
/// block into an UNSCALED FP32 inner accumulator, folded onto the outer one
/// once per block with that block's scale.
fn mma_row(f: &Fixture, row: &[f32], col: usize) -> u16 {
    let mut outer = 0.0_f32;
    for block in 0..K / FP8_BLOCK {
        let mut inner = 0.0_f32;
        for sub in 0..FP8_BLOCK / 16 {
            let base = block * FP8_BLOCK + sub * 16;
            for i in 0..16 {
                inner += row[base + i] * f.weight[col * K + base + i];
            }
        }
        outer += inner * f.scale(col, block);
    }
    bf16::from_f32(outer).to_bits()
}

/// One `w8a16_gemm_m16` launch: rows `[a_off, a_off + rows)` of the activation
/// buffer into rows `[c_off, c_off + rows)` of the output, in ELEMENTS — the
/// same two quantities `w8a16_m16_tc_proj` turns into byte offsets.
fn launch(f: &Fixture, out: &mut [u16], rows: usize, a_off: usize, c_off: usize) {
    assert!(rows <= 16, "the kernel's M tile is 16 rows");
    for r in 0..rows {
        let row = &f.acts[(a_off + r) * K..(a_off + r + 1) * K];
        for col in 0..N {
            out[(c_off + r) * N + col] = mma_row(f, row, col);
        }
    }
}

/// The route `w8a16_m16_tc_proj` runs for `m` rows, with the offsets it uses.
/// `a_skew` / `c_skew` corrupt the second half's input / output offset — the
/// negative controls, 0 for the real route.
fn halves_route(f: &Fixture, m: usize, a_skew: isize, c_skew: isize) -> Vec<u16> {
    let mut out = vec![0_u16; M * N];
    match m16_tc_plan(m as u32, K as u32, true, true).expect("5..=32 is this tier's band") {
        M16TcPlan::Single => launch(f, &mut out, m, 0, 0),
        M16TcPlan::Halves { first } => {
            let first = first as usize;
            launch(f, &mut out, first, 0, 0);
            launch(
                f,
                &mut out,
                m - first,
                (first as isize + a_skew) as usize,
                (first as isize + c_skew) as usize,
            );
        }
    }
    out
}

/// RMS of a BF16 block — the scale [`within_m16_tc_budget`]'s absolute floor is
/// expressed in.
fn rms(block: &[u16]) -> f64 {
    let sum: f64 = block
        .iter()
        .map(|b| {
            let v = f64::from(bf16::from_bits(*b).to_f32());
            v * v
        })
        .sum();
    (sum / block.len() as f64).sqrt()
}

/// THE OFFSET PIN. At M=32 the arm launches the 16-row kernel twice on
/// contiguous row halves; every row must come out exactly as it would from a
/// launch of its own. Bit-identical, not within a tolerance — the split changes
/// no row's arithmetic, so there is nothing for it to be within.
#[test]
fn the_two_halves_rung_reproduces_every_row_of_a_per_row_launch() {
    let f = Fixture::new();
    let split = halves_route(&f, M, 0, 0);
    let mut direct = vec![0_u16; M * N];
    for r in 0..M {
        launch(&f, &mut direct, 1, r, r);
    }
    for r in 0..M {
        assert_eq!(
            &split[r * N..(r + 1) * N],
            &direct[r * N..(r + 1) * N],
            "row {r} differs between the two-halves route and a per-row launch"
        );
    }
}

/// ...and the pin above is not vacuous: BOTH of the offsets the suspicion was
/// about are shown to be load-bearing. A wrong input offset feeds the second
/// half the wrong activations; a wrong output offset writes it to the wrong
/// rows. Either one moves the result.
#[test]
fn a_wrong_second_half_offset_is_caught() {
    let f = Fixture::new();
    let good = halves_route(&f, M, 0, 0);
    for (label, a_skew, c_skew) in [
        ("input offset short by one row", -1_isize, 0_isize),
        ("input offset dropped entirely", -16, 0),
        ("output offset short by one row", 0, -1),
    ] {
        assert_ne!(
            good,
            halves_route(&f, M, a_skew, c_skew),
            "{label}: the offset pin would not have caught this"
        );
    }
}

/// Which activation row each launch reads into which output row, for a given
/// `m` — the plan's offset arithmetic with the MACs taken out, so the whole
/// 5..=32 band is affordable in one test.
fn row_map(m: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut record = |rows: usize, a_off: usize, c_off: usize| {
        assert!(
            rows <= 16,
            "m={m}: a launch exceeded the kernel's 16-row M tile"
        );
        for r in 0..rows {
            pairs.push((a_off + r, c_off + r));
        }
    };
    match m16_tc_plan(m as u32, K as u32, true, true).expect("5..=32 is this tier's band") {
        M16TcPlan::Single => record(m, 0, 0),
        M16TcPlan::Halves { first } => {
            let first = first as usize;
            record(first, 0, 0);
            record(m - first, first, first);
        }
    }
    pairs
}

/// Every row count the tier claims maps activation row `r` onto output row `r`,
/// exactly once, for every `r < m` and no `r >= m` — across BOTH rungs, not
/// just the M=32 one the H100 flagged.
#[test]
fn every_row_count_in_the_band_maps_each_row_to_itself_exactly_once() {
    for m in 5..=M {
        let pairs = row_map(m);
        assert_eq!(pairs.len(), m, "m={m}: {} rows covered", pairs.len());
        for (i, (a, c)) in pairs.iter().enumerate() {
            assert_eq!(
                (*a, *c),
                (i, i),
                "m={m}: launch slot {i} reads/writes the wrong row"
            );
        }
    }
}

/// THE METRIC PIN — round 6's five red elements, as measured, put through the
/// criterion the oracle now uses.
///
/// `reference` / `actual` are the worst pair the full-geometry host run of this
/// same simulation produced (|ref| 5.722e-6, absolute error 3.8e-6) at the
/// reference RMS the run measured (39.13). In ordinal BF16 ULP that is a
/// three-figure distance and an automatic FAIL; in absolute terms it is the
/// FP32 accumulator's own floor at K=5120, and the fixed budget never had
/// anything to say about it.
#[test]
fn the_m32_red_cell_is_a_cancelled_output_not_a_row_offset() {
    const RMS: f64 = 39.13;
    let reference = bf16::from_f32(-5.722e-6).to_bits();
    let actual = bf16::from_f32(-9.537e-6).to_bits();
    assert!(
        (bf16_ord(actual) - bf16_ord(reference)).abs() > 2,
        "the round-6 pair must be the one the ordinal budget rejects"
    );
    assert!(
        within_m16_tc_budget(actual, reference, RMS),
        "an output cancelled to 1.5e-7 of the matrix RMS has no relative accuracy to grade"
    );
    // ...and so are the other two sites the full-geometry run flagged.
    for (r, a) in [(-1.163_48e-4, -1.058_58e-4), (-1.564_03e-4, -1.506_81e-4)] {
        let (rb, ab) = (bf16::from_f32(r).to_bits(), bf16::from_f32(a).to_bits());
        assert!(
            within_m16_tc_budget(ab, rb, RMS),
            "ref={r} actual={a} is inside the accumulation floor"
        );
    }
}

/// ...and the floor has NOT been widened into a rubber stamp. An error of the
/// size a real row or pitch defect produces — and even one at the BF16 quantum
/// of the same matrix, which round 6 reported as `max_abs 0.500` — is still
/// refused.
#[test]
fn the_accumulation_floor_still_rejects_a_structural_error() {
    const RMS: f64 = 39.13;
    for (label, r, a) in [
        ("a misplaced output", 64.0_f32, 71.5_f32),
        ("three BF16 quanta at the top of the range", 96.0, 97.5),
        (
            "a cancelled output moved by a matrix-scale error",
            1.0e-5,
            0.5,
        ),
    ] {
        let (rb, ab) = (bf16::from_f32(r).to_bits(), bf16::from_f32(a).to_bits());
        assert!(
            !within_m16_tc_budget(ab, rb, RMS),
            "{label}: ref={r} actual={a} must still fail the budget"
        );
    }
    // Round 6's `max_abs 0.500` is ONE quantum at magnitude 64..128 and has
    // always been inside the 2-ULP budget — the floor does not touch it, in
    // either direction.
    let (r, a) = (
        bf16::from_f32(96.0).to_bits(),
        bf16::from_f32(96.5).to_bits(),
    );
    assert_eq!(
        (bf16_ord(a) - bf16_ord(r)).abs(),
        1,
        "max_abs 0.500 is 1 ULP"
    );
    assert!(within_m16_tc_budget(a, r, RMS));
    // The floor is 2^-20 of the RMS: state the number the constant buys, so a
    // future widening has to argue with an assertion instead of a comment.
    assert!(
        (M16_TC_ACC_FLOOR * RMS - 3.731e-5).abs() < 1e-8,
        "the absolute floor at the round-6 RMS is ~3.7e-5"
    );
}

/// End to end on the sampled geometry: the two-halves route agrees with the
/// scalar `w8a16_gemv` under the oracle's criterion, on every row of both
/// halves. This is the assertion the H100 cell is supposed to be making.
#[test]
fn the_two_halves_rung_meets_the_oracle_budget_against_the_scalar_gemv() {
    let f = Fixture::new();
    let actual = halves_route(&f, M, 0, 0);
    let mut reference = vec![0_u16; M * N];
    for r in 0..M {
        let row = &f.acts[r * K..(r + 1) * K];
        for col in 0..N {
            reference[r * N + col] = gemv_reference(&f, row, col);
        }
    }
    let scale = rms(&reference);
    assert!(scale > 1.0, "the fixture must produce a real matrix scale");
    let mut worst = (0_i32, 0_usize, 0_usize);
    for r in 0..M {
        for col in 0..N {
            let (a, b) = (actual[r * N + col], reference[r * N + col]);
            let ulp = (bf16_ord(a) - bf16_ord(b)).abs();
            if ulp > worst.0 {
                worst = (ulp, r, col);
            }
            assert!(
                within_m16_tc_budget(a, b, scale),
                "row {r} col {col}: reference {} actual {} ({ulp} ordinal ULP) is outside \
                 both the 2-ULP budget and the accumulation floor",
                bf16::from_bits(b),
                bf16::from_bits(a),
            );
        }
    }
    // The worst element is reported rather than asserted small: which element
    // wins is a property of the draw, and pinning it would make the test a
    // record of this fixture instead of of the contract.
    println!(
        "worst ordinal ULP {} at row {} col {} (reference RMS {scale:.3})",
        worst.0, worst.1, worst.2
    );
}
