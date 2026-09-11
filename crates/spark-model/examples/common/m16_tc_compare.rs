// SPDX-License-Identifier: AGPL-3.0-only

//! Host half of `native_fp8_ffn_m16_tc_microtest.rs`, split out to keep that
//! example under the 500-LoC cap. Everything here runs on the CPU: the shape
//! table and the checkpoint draws, the loader's `_t` transpose, the guard and
//! gap audits, the per-case verdict and the line that reports it, and the two
//! known-bad controls that prove the comparison still bites. The example keeps
//! the GPU half — kernel handles, uploads, launches and timing.
//!
//! ⚠ THE ACCEPTANCE PREDICATE ITSELF IS NOT HERE AND MUST NOT BE COPIED HERE.
//! `within_m16_tc_budget` / `compare_m16_tc_block` live in the library, at
//! `layers::dense_ffn::m16_tc::oracle`, for the reason the example's header
//! gives: the microtest and the host simulation that pins the same contract
//! grade the tier with ONE predicate so the two cannot drift. A second copy in
//! `examples/` would be exactly the drift that arrangement exists to prevent.
//! This module only calls it.
//!
//! ⚠ THE ABSOLUTE FLOOR IS ROUND 6'S FIX AND IT IS NOT A LOOSENING. Round 6
//! failed `gate/up M=32` on `max_ulp 28`, 5 of 557,056 elements, `sign_flips 0`,
//! `rel_rms 4.2e-5`, and that was read as a possible row/pitch defect in the
//! two-halves rung. It was the metric: a host simulation of the same geometry
//! reproduces all five with no offset arithmetic at all, and every one is an
//! output that cancelled to |ref| 5.7e-6..1.6e-4 against a reference RMS of
//! 39.1 — 1e-7..4e-6 of the matrix scale, where an ordinal ULP has nothing left
//! to measure and one FP32 accumulation rounding spans hundreds of them. The
//! floor is 13,500x below the BF16 quantum at the top of the same matrix, so a
//! real misplacement (errors of order the RMS) still fails. Derivation:
//! `layers::dense_ffn::m16_tc::M16_TC_ACC_FLOOR`; both directions are pinned in
//! `dense_ffn_m16_tc_m32_tests.rs`.
//!
//! Sign flips below |scalar| < 0.05 are still counted separately rather than
//! graded: one ULP across zero is a full sign change. Nothing is silently
//! dropped — every rejected element is printed with its (row, col), which round
//! 6 could not do and which is why diagnosing it needed a second run.
//!
//! Test-only harness code; no serving path runs any of it.

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::dense_ffn::m16_tc::oracle::{
    M16_TC_MAX_ULP, M16TcDiff, compare_m16_tc_block,
};

/// Qwen/Qwen3.8-27B hidden size.
pub(crate) const H: usize = 5120;
/// Qwen/Qwen3.8-27B FFN intermediate size.
pub(crate) const INTER: usize = 17408;
pub(crate) const MAX_M: usize = 32;
pub(crate) const GUARD: usize = 64;
/// Extra columns/elements between rows for the strided-sibling leg.
pub(crate) const A_PAD: usize = 8;
pub(crate) const C_PAD: usize = 64;
/// Block-level relative-RMS gate. The per-element budget lives in the lib
/// (`M16_TC_MAX_ULP` + `M16_TC_ACC_FLOOR`), so neither half of this microtest
/// can grade the tier differently from the host simulation that pins the same
/// contract.
pub(crate) const REL_RMS_GATE: f64 = 1e-3;

pub(crate) struct Shape {
    pub(crate) name: &'static str,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

pub(crate) const SHAPES: [Shape; 2] = [
    Shape {
        name: "gate/up",
        n: INTER,
        k: H,
    },
    Shape {
        name: "down",
        n: H,
        k: INTER,
    },
];

pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    pub(crate) fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

/// One shape's host-drawn checkpoint: the FP8 weight, the BF16 activations for
/// all [`MAX_M`] rows, the per-128x128 block scales, and the same activations
/// re-laid at the padded pitch the strided leg reads.
pub(crate) struct Inputs {
    pub(crate) weights: Vec<u8>,
    pub(crate) acts: Vec<u8>,
    pub(crate) scales: Vec<u8>,
    pub(crate) acts_strided: Vec<u8>,
}

pub(crate) fn draw_inputs(rng: &mut Rng, n: usize, k: usize) -> Inputs {
    // E4M3 byte draws skip 0x7F/0xFF (NaN), as the batch4 oracle does —
    // and as `w8a16_gemm_m16.cu`'s DEQUANT note requires: those are the two
    // bytes on which the hardware `cvt` and the `E4M3_LUT` disagree, and a
    // block-scaled checkpoint cannot contain them.
    let weights: Vec<u8> = (0..n * k)
        .map(|_| {
            let x = rng.next();
            ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
        })
        .collect();
    let acts: Vec<u8> = (0..MAX_M * k)
        .flat_map(|_| {
            bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let scales: Vec<u8> = (0..(n / 128) * (k / 128))
        .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 1024.0).to_le_bytes())
        .collect();
    // Strided A: the same rows, re-laid at a `k + A_PAD` pitch.
    let a_pitch = k + A_PAD;
    let mut acts_strided = vec![0x3c_u8; MAX_M * a_pitch * 2];
    for row in 0..MAX_M {
        let src = row * k * 2;
        let dst = row * a_pitch * 2;
        acts_strided[dst..dst + k * 2].copy_from_slice(&acts[src..src + k * 2]);
    }
    Inputs {
        weights,
        acts,
        scales,
        acts_strided,
    }
}

/// `[N, K]` FP8 -> `[K, N]`, and `[N/128, K/128]` FP32 -> `[K/128, N/128]` —
/// the loader's `_t` copy, built on the host so the pre-#927 arm can be timed.
pub(crate) fn transpose(weights: &[u8], scales: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let mut wt = vec![0_u8; n * k];
    for row in 0..n {
        for col in 0..k {
            wt[col * n + row] = weights[row * k + col];
        }
    }
    let (nb, kb) = (n / 128, k / 128);
    let mut st = vec![0_u8; nb * kb * 4];
    for bn in 0..nb {
        for bk in 0..kb {
            let src = (bn * kb + bk) * 4;
            let dst = (bk * nb + bn) * 4;
            st[dst..dst + 4].copy_from_slice(&scales[src..src + 4]);
        }
    }
    (wt, st)
}

/// Print every element the criterion rejected, with its coordinates, magnitude
/// and how far below the block RMS it sits — the three numbers that separate a
/// cancellation tail from a defect.
pub(crate) fn report_outliers(label: &str, d: &M16TcDiff) {
    for o in &d.over_budget {
        let relative = if d.rms > 0.0 {
            f64::from(o.reference).abs() / d.rms
        } else {
            f64::NAN
        };
        println!(
            "  OVER_BUDGET {label} (m={}, n={}) reference={:+.9e} actual={:+.9e} \
             ulp={} |ref|/rms={relative:.3e} budget={M16_TC_MAX_ULP} ULP or the floor",
            o.row, o.col, o.reference, o.actual, o.ulp
        );
    }
}

/// Nothing outside `[M, N]`: the leading/trailing sentinel and every row past M
/// must be untouched, on BOTH instantiations.
pub(crate) fn guards_intact(
    observed: &[u8],
    observed_n64: &[u8],
    sentinel: &[u8],
    bytes: usize,
) -> bool {
    observed[..GUARD] == sentinel[..GUARD]
        && observed[GUARD + bytes..] == sentinel[GUARD + bytes..]
        && observed_n64[..GUARD] == sentinel[..GUARD]
        && observed_n64[GUARD + bytes..] == sentinel[GUARD + bytes..]
}

pub(crate) struct StridedCheck {
    pub(crate) gaps_intact: bool,
    pub(crate) max_ulp: i32,
}

/// The strided leg's audit: each row's `[n, c_pitch)` gap and every row past
/// `m` must still be sentinel, and the used extent must match the scalar within
/// the same budget — every row of it, halves included.
pub(crate) fn check_strided(
    strided: &[u8],
    sentinel_s: &[u8],
    baseline: &[u8],
    m: usize,
    n: usize,
    c_pitch: usize,
) -> StridedCheck {
    let mut gaps_intact = strided[..GUARD] == sentinel_s[..GUARD];
    let mut max_ulp = 0_i32;
    for row in 0..MAX_M {
        let base = GUARD + row * c_pitch * 2;
        let gap = &strided[base + n * 2..base + c_pitch * 2];
        gaps_intact &= gap.iter().all(|b| *b == 0x5a);
        if row < m {
            let sd = compare_m16_tc_block(
                &strided[base..base + n * 2],
                &baseline[GUARD + row * n * 2..GUARD + (row + 1) * n * 2],
                n,
            );
            max_ulp = max_ulp.max(sd.max_ulp);
            if !sd.over_budget.is_empty() {
                gaps_intact = false;
            }
        } else {
            gaps_intact &= strided[base..base + n * 2].iter().all(|b| *b == 0x5a);
        }
    }
    StridedCheck {
        gaps_intact,
        max_ulp,
    }
}

/// The four arms' wall times for one shape/M case, in ms per rep.
pub(crate) struct Timings {
    pub(crate) tc_ms: f64,
    pub(crate) n64_ms: f64,
    pub(crate) b16_ms: f64,
    pub(crate) tile_ms: f64,
}

/// Everything one shape/M case's verdict and report line read.
pub(crate) struct Case<'a> {
    pub(crate) name: &'a str,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) passes: usize,
    pub(crate) d: &'a M16TcDiff,
    pub(crate) d64: &'a M16TcDiff,
    pub(crate) strided: &'a StridedCheck,
    pub(crate) guards_intact: bool,
    pub(crate) timings: &'a Timings,
    pub(crate) weight_gb: f64,
}

/// The case-level verdict: both instantiations inside the per-element budget
/// and the block rel-RMS gate, with every guard and gap intact.
pub(crate) fn case_passed(c: &Case) -> bool {
    c.d.over_budget.is_empty()
        && c.d64.over_budget.is_empty()
        && c.d.rel_rms <= REL_RMS_GATE
        && c.d64.rel_rms <= REL_RMS_GATE
        && c.guards_intact
        && c.strided.gaps_intact
}

/// Print the case's line, then NAME the rejected elements. `over_budget 5` with
/// no coordinates is what made round 6's red cell take a second machine to
/// diagnose; `over_ulp_only` above it is the count the OLD gate would have
/// rejected, so a cancellation tail is legible without re-running.
///
/// Returns the verdict so the caller can count failures.
pub(crate) fn report_case(c: &Case) -> bool {
    let ok = case_passed(c);
    let (d, d64, t) = (c.d, c.d64, c.timings);
    let gbs = |ms: f64| c.weight_gb * c.passes as f64 / (ms / 1e3);
    println!(
        "{name:<8} M={m:<3} N={n} K={k} passes={passes} rms={rms:.3} \
         max_ulp={ulp} over_budget={ob} over_ulp_only={ou} sign_flips={sf} \
         max_abs={ma:.9} rel_rms={rr:.3e} strided_max_ulp={su} \
         n64_over_budget={ob64} n64_max_ulp={ulp64} guards={g} gaps={gp} | \
         m16_tc {tc_ms:.3}ms ({tcg:.1} GB/s) \
         vs n64 {n64_ms:.3}ms ({n64g:.1} GB/s) = {sp0:.2}x \
         vs batch16 {b16_ms:.3}ms ({b16g:.1} GB/s) = {sp1:.2}x \
         vs t_m128 {tile_ms:.3}ms ({tileg:.1} GB/s) = {sp2:.2}x  {verdict}",
        name = c.name,
        m = c.m,
        n = c.n,
        k = c.k,
        passes = c.passes,
        rms = d.rms,
        ulp = d.max_ulp,
        ob = d.over_budget.len(),
        ou = d.over_ulp_only,
        sf = d.sign_flips,
        ma = d.max_abs,
        rr = d.rel_rms,
        su = c.strided.max_ulp,
        ob64 = d64.over_budget.len(),
        ulp64 = d64.max_ulp,
        g = if c.guards_intact { "ok" } else { "CLOBBERED" },
        gp = if c.strided.gaps_intact {
            "ok"
        } else {
            "CLOBBERED"
        },
        tc_ms = t.tc_ms,
        n64_ms = t.n64_ms,
        b16_ms = t.b16_ms,
        tile_ms = t.tile_ms,
        tcg = gbs(t.tc_ms),
        n64g = gbs(t.n64_ms),
        b16g = gbs(t.b16_ms),
        tileg = c.weight_gb / (t.tile_ms / 1e3),
        sp0 = t.n64_ms / t.tc_ms,
        sp1 = t.b16_ms / t.tc_ms,
        sp2 = t.tile_ms / t.tc_ms,
        verdict = if ok { "PASS" } else { "FAIL" },
    );
    report_outliers("m16_tc", d);
    report_outliers("n64", d64);
    ok
}

/// Oracle self-checks, once per shape: the comparison the loop runs MUST refuse
/// a known-bad block, so a green report cannot mean "the comparison was
/// vacuous". TWO controls, because round 6's fix widened the criterion and a
/// widened criterion has to prove it still bites — one at three ULP on a LARGE
/// value (which the absolute floor must not rescue), one that MOVES A WHOLE ROW
/// (the row/pitch defect the M=32 cell was suspected of, which is what the gate
/// is really for).
///
/// `good` is the scalar baseline buffer, sentinel guards included.
pub(crate) fn assert_oracle_bites(good: &[u8], name: &str, n: usize) -> Result<()> {
    let rows = &good[GUARD..GUARD + MAX_M * n * 2];
    let mut bad = good.to_vec();
    // Find an element well outside the sign-flip band and push it 3 ULP.
    let idx = (GUARD..GUARD + n * 2)
        .step_by(2)
        .find(|i| {
            bf16::from_bits(u16::from_le_bytes([good[*i], good[*i + 1]]))
                .to_f32()
                .abs()
                > 1.0
        })
        .expect("baseline has a value above 1.0");
    let bits = u16::from_le_bytes([good[idx], good[idx + 1]]);
    bad[idx..idx + 2].copy_from_slice(&(bits.wrapping_add(3)).to_le_bytes());
    let caught = !compare_m16_tc_block(&bad[GUARD..GUARD + MAX_M * n * 2], rows, n)
        .over_budget
        .is_empty();
    println!("KNOWN_BAD {name} three-ULP mutation on a |value| > 1: refused={caught}");
    ensure!(
        caught,
        "comparison oracle admitted a three-ULP mutation above the accumulation floor"
    );
    // Row 17 served row 16's outputs — exactly the failure a wrong second-half
    // offset produces.
    let mut shifted = good.to_vec();
    let (src, dst) = (GUARD + 16 * n * 2, GUARD + 17 * n * 2);
    let row16 = good[src..src + n * 2].to_vec();
    shifted[dst..dst + n * 2].copy_from_slice(&row16);
    let caught_row = !compare_m16_tc_block(&shifted[GUARD..GUARD + MAX_M * n * 2], rows, n)
        .over_budget
        .is_empty();
    println!("KNOWN_BAD {name} second-half row offset (row 17 <- row 16): refused={caught_row}");
    ensure!(
        caught_row,
        "comparison oracle admitted a misplaced output row — the absolute floor is too wide"
    );
    Ok(())
}
