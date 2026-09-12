// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle for the TENSOR-CORE 5..=16-row BF16 LM-head arm
//! (`ATLAS_LM_HEAD_M16_TC`, #927/#928) — `dense_gemm_m16_bf16`.
//!
//! Runs the REAL head shape — `[N = 248077, K = 5120]` BF16, 2.54 GB — at
//! M in {5, 8, 13, 16} and compares against the scalar `dense_gemv_bf16` that
//! each row's M=1 decode runs.
//!
//! WHY THIS SHAPE AND NOT A SAMPLED ONE. nsys round 7 (1xH100, 2026-09-11,
//! Qwen/Qwen3.8-27B-FP8 with `--lm-head-dtype bf16` and
//! `ATLAS_LM_HEAD_BATCHM_MAX=16`, decode batch 16) puts `dense_gemv_bf16_batchm`
//! at **3,571 µs in ONE launch = 8.19% of the 43.6 ms step** — ~710 GB/s, where
//! the SAME kernel reads the SAME 2.54 GB at 3.2 TB/s-class for a single row
//! (798 µs at C=1). The thing under test is what happens to a kernel that is
//! FP32-FMA-bound at 16 rows, and only the real N and K reproduce it: N is
//! 7,753 CTAs of weight streaming and K is the reduction depth the ULP budget
//! is about.
//!
//! 🔴 N IS THE UNPADDED VOCAB, 248,077 — ODD, AND THAT IS THE POINT.
//! The padded 248,192 stride belongs to the NVFP4 TRANSPOSED twin
//! (`model/impl_a1.rs`), whose tile GEMM reads B with 16-byte `cp.async` down
//! the K axis. The BF16 head is `[N, K]` row-major: every weight ROW starts at
//! `n * 5120` BF16 = a multiple of 10,240 B, so it is 16-byte aligned for any
//! N, and the head projects at N = `config.vocab_size` with no padding at all.
//! What the odd N does cost is a permanently PARTIAL last CTA, which is why the
//! guard checks below are not ceremony.
//!
//! 🔴 ROUND 9 RE-SCALED THE ACCEPTANCE FLOOR — THIS FILE IS WHY.
//! Round 9 (1xH100, 2026-09-11) MET the speed target (M=16 best arm 0.914 ms /
//! 2,779 GB/s, 3.67x `dense_gemv_bf16_batchm`'s 3.486 ms) and passed the
//! block-level `rel_rms` gate with a 9x margin (1.107e-4 against 1e-3), but
//! FAILED the per-element gate at every M: `over_budget` 10/16/25/37 at
//! M=5/8/13/16, `max_ulp` 32/100/100/100, with `n64` rejecting the IDENTICAL
//! set. That count is linear in M (~2 elements per row out of 248,077
//! columns), which is a per-element statistical tail and not a boundary defect;
//! every rejected element was a logit cancelled to 4.9e-6..2.6e-4 of the block
//! RMS, the worst being `reference=-1.173019409e-4` at 100 ordinal ULP against
//! a block RMS of 23.80. The floor was the problem: round 6's fixed `2^-20 *
//! rms` was fitted to `w8a16_gemm_m16`, whose reduction folds a 128-wide FP8
//! scale block onto an outer accumulator; THIS kernel has no block scale, so
//! its accumulator is ONE uninterrupted 320-step FP32 chain at K=5120 and the
//! fixed constant landed under its noise. The floor is now
//! `8 * u32 * sqrt(K) * row_rms` — see
//! `layers::dense_ffn::m16_tc::m16_tc_acc_floor` and the host pins in
//! `layers/ops/dense_gemm_m16_bf16_floor_tests.rs`.
//!
//! 🔴 THE PASS CONDITION IS A TOLERANCE, NOT BIT-EQUALITY.
//! `dense_gemv_bf16_batchm` reduces each output in ONE FP32 accumulator in
//! strict K order and is byte-identical to M serial `dense_gemv_bf16` calls; an
//! m16n8k16 MMA reduces 16 K-products in the tensor core's own order first, so
//! this kernel is not. The contract is ONE predicate, shared with
//! `w8a16_gemm_m16` and with the host simulation
//! (`layers/ops/dense_gemm_m16_bf16_tests.rs`) so the three cannot drift:
//! `layers::dense_ffn::m16_tc::within_m16_tc_budget` — within 2 ordinal BF16
//! ULP, OR an absolute error under `8 * u32 * sqrt(K)` of the reference ROW's
//! RMS — plus `rel_rms <= 1e-3` over the block. At the LM head that seam is
//! token-visible, which is why the arm defaults OFF.
//!
//! It also pins the guards the wrapper promises (nothing written outside
//! `[M, N]`, on either CTA width) and times the arm against
//! `dense_gemv_bf16_batchm`, the tier it would displace.
//!
//! TARGET (the number this file exists to settle): **<= 1.3 ms at M=16 on the
//! real head = >= 1,950 GB/s**, against the 3.571 ms / ~710 GB/s nsys measured.
//!
//! TIMING METHOD: `synchronize` + host `Instant` over 20 reps, the house pattern
//! (`examples/native_fp8_ffn_m16_tc_microtest.rs`).
//!
//! RESOURCES: ~2.6 GB of HOST RAM for the generated weight and ~2.6 GB of
//! device memory for it, plus four 7.9 MB output buffers. Generation is ~1.3e9
//! LCG draws; expect tens of seconds before the first line prints.
//!
//! Run on the H100:
//!     cargo run --release --example native_bf16_lm_head_m16_microtest \
//!       --features cuda,gpu-examples
//!
//! ★ REQUIRES A KERNEL SET THAT CARRIES `dense_gemm_m16_bf16.cu`. Since the
//! 2026-09-11 arch separation that source is a HOPPER-TUNED file
//! (`kernels/hopper/common`, declared in that target's `[kernels] overrides`)
//! and `kernels/gb10` does not carry it, so this oracle is for a binary built
//! with `ATLAS_TARGET_HW=hopper` (or `b200`, which symlinks the same source).
//! On a GB10 build the `gpu.kernel(...)` lookup below fails by name, which is
//! the honest answer: the tier this grades does not exist there.

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::dense_ffn::m16_tc::oracle::{
    M16_TC_MAX_ULP, M16TcDiff, compare_m16_tc_block, m16_tc_acc_floor,
};
use spark_model::layers::ops;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

/// Qwen3.8-27B hidden size — the head's reduction depth.
const K: usize = 5120;
/// The UNPADDED vocab. See the module note on why it is not 248,192.
const N: usize = 248_077;
const MAX_M: usize = 16;
/// The two CTA widths, from the launcher's own SSOT — 7,753 resp. 3,877 CTAs
/// at this N, both ending on the SAME 13-column tail because 248,064 is a
/// multiple of both.
const N_TILE: usize = ops::DENSE_GEMM_M16_BF16_N_TILE as usize;
const N_TILE_WIDE: usize = ops::DENSE_GEMM_M16_BF16_N_TILE_WIDE as usize;
const GUARD: usize = 64;
const REPS: u32 = 20;
const WARMUP: u32 = 3;
/// Block-level relative-RMS gate; the per-element budget lives in the lib.
const REL_RMS_GATE: f64 = 1e-3;
/// The round-7 baseline this arm is measured against, in ms at M=16.
const NSYS_BATCHM_MS: f64 = 3.571;
/// The tier's acceptance target at M=16, in ms and GB/s.
const TARGET_MS: f64 = 1.3;
const TARGET_GBS: f64 = 1_950.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// A BF16-representable value in [-1, 1), the alphabet the FFN oracle uses.
    fn bf16_bits(&mut self) -> u16 {
        bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0).to_bits()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// Fill `dst` with BF16 draws in place — `collect()` on 1.3e9 elements would
/// reallocate its way through several gigabytes for the same bytes.
fn fill_bf16(rng: &mut Rng, dst: &mut [u8]) {
    for slot in dst.chunks_exact_mut(2) {
        slot.copy_from_slice(&rng.bf16_bits().to_le_bytes());
    }
}

/// Print every element the criterion rejected, with its coordinates, how far
/// below its own ROW's RMS it sits and the floor it missed — the numbers that
/// separate a cancellation tail from a defect, and the ones round 9's diagnosis
/// ran on. Capped: at 248,077 columns a real defect would otherwise print
/// millions of lines. The CTA a column belongs to is printed too, because the
/// one defect class the statistics cannot exclude on their own is the last,
/// always-partial CTA (248,064 is a multiple of both tile widths, so the tail
/// is the same 13 columns on both arms).
fn report_outliers(label: &str, d: &M16TcDiff, n_tile: usize) {
    for o in d.over_budget.iter().take(16) {
        let relative = if o.row_rms > 0.0 {
            f64::from(o.reference).abs() / o.row_rms
        } else {
            f64::NAN
        };
        println!(
            "  OVER_BUDGET {label} (m={}, n={}) reference={:+.9e} actual={:+.9e} \
             ulp={} |ref|/row_rms={relative:.3e} row_rms={:.4} floor={:.6e} \
             cta={} of {} (tail={}) budget={M16_TC_MAX_ULP} ULP or the floor",
            o.row,
            o.col,
            o.reference,
            o.actual,
            o.ulp,
            o.row_rms,
            m16_tc_acc_floor(K, o.row_rms),
            o.col / n_tile,
            N.div_ceil(n_tile),
            o.col >= (N / n_tile) * n_tile,
        );
    }
    if d.over_budget.len() > 16 {
        println!("  … and {} more", d.over_budget.len() - 16);
    }
}

fn time_ms(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / f64::from(REPS))
}

/// One arm of the comparison: a name, its launcher and its handle. Both
/// instantiations take the same arguments, so the arm is data, not a branch.
struct Arm {
    label: &'static str,
    gemm: ops::DenseM16Bf16Gemm,
    kernel: KernelHandle,
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let scalar = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let batchm = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
    let tc = gpu.kernel("dense_gemm_m16_bf16", "dense_gemm_m16_bf16")?;
    let tc_n64 = gpu.kernel("dense_gemm_m16_bf16", "dense_gemm_m16_bf16_n64")?;

    let mut rng = Rng(0x0927_167C_2026);
    println!(
        "generating [{N}, {K}] BF16 weights ({:.2} GB) — this takes a moment",
        (N * K * 2) as f64 / 1e9
    );
    let mut weights = vec![0_u8; N * K * 2];
    fill_bf16(&mut rng, &mut weights);
    let mut acts = vec![0_u8; MAX_M * K * 2];
    fill_bf16(&mut rng, &mut acts);

    let weight = DenseWeight {
        weight: upload(&gpu, &weights)?,
    };
    let input = upload(&gpu, &acts)?;
    drop(weights);

    let out_bytes = MAX_M * N * 2;
    let sentinel = vec![0x5a_u8; out_bytes + 2 * GUARD];
    let scalar_base = upload(&gpu, &sentinel)?;
    let tc_base = upload(&gpu, &sentinel)?;
    let n64_base = upload(&gpu, &sentinel)?;
    let batchm_base = upload(&gpu, &sentinel)?;
    let (scalar_out, tc_out, n64_out, batchm_out) = (
        scalar_base.offset(GUARD),
        tc_base.offset(GUARD),
        n64_base.offset(GUARD),
        batchm_base.offset(GUARD),
    );
    let weight_gb = (N * K * 2) as f64 / 1e9;

    let arms = [
        Arm {
            label: "m16_tc",
            gemm: ops::dense_gemm_m16_bf16,
            kernel: tc,
        },
        Arm {
            label: "n64",
            gemm: ops::dense_gemm_m16_bf16_n64,
            kernel: tc_n64,
        },
    ];
    let mut failures = 0_usize;

    for m in [5_usize, 8, 13, 16] {
        gpu.copy_h2d(&sentinel, scalar_base)?;
        gpu.copy_h2d(&sentinel, tc_base)?;
        gpu.copy_h2d(&sentinel, n64_base)?;
        // The scalar per-row baseline — exactly what an M=1 decode runs.
        for row in 0..m {
            ops::dense_gemv(
                &gpu,
                scalar,
                input.offset(row * K * 2),
                &weight,
                scalar_out.offset(row * N * 2),
                N as u32,
                K as u32,
                0,
            )?;
        }
        let run = |arm: &Arm, out: DevicePtr| {
            (arm.gemm)(
                &gpu, arm.kernel, input, &weight, out, m as u32, N as u32, K as u32, K as u32,
                N as u32, 0,
            )
        };
        run(&arms[0], tc_out)?;
        run(&arms[1], n64_out)?;
        gpu.synchronize(0)?;

        let mut baseline = vec![0_u8; sentinel.len()];
        let mut observed = vec![0_u8; sentinel.len()];
        let mut observed_n64 = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut baseline)?;
        gpu.copy_d2h(tc_base, &mut observed)?;
        gpu.copy_d2h(n64_base, &mut observed_n64)?;
        let bytes = m * N * 2;
        let d = compare_m16_tc_block(
            &observed[GUARD..GUARD + bytes],
            &baseline[GUARD..GUARD + bytes],
            N,
            K,
        );
        let d64 = compare_m16_tc_block(
            &observed_n64[GUARD..GUARD + bytes],
            &baseline[GUARD..GUARD + bytes],
            N,
            K,
        );

        // GUARDS. Nothing outside [M, N]: the leading/trailing sentinel and
        // every row past M must be untouched, on BOTH instantiations. With
        // N = 248,077 the last CTA is ALWAYS partial, so the trailing sentinel
        // is the column mask's only witness.
        let guards_intact = observed[..GUARD] == sentinel[..GUARD]
            && observed[GUARD + bytes..] == sentinel[GUARD + bytes..]
            && observed_n64[..GUARD] == sentinel[..GUARD]
            && observed_n64[GUARD + bytes..] == sentinel[GUARD + bytes..];

        let tc_ms = time_ms(&gpu, || run(&arms[0], tc_out))?;
        let n64_ms = time_ms(&gpu, || run(&arms[1], n64_out))?;
        let batchm_ms = time_ms(&gpu, || {
            ops::dense_gemv_batchm(
                &gpu, batchm, input, &weight, batchm_out, m as u32, N as u32, K as u32, N as u32, 0,
            )
        })?;
        let gbs = |ms: f64| weight_gb / (ms / 1e3);
        let ok = d.over_budget.is_empty()
            && d64.over_budget.is_empty()
            && d.rel_rms <= REL_RMS_GATE
            && d64.rel_rms <= REL_RMS_GATE
            && guards_intact;
        println!(
            "lm_head M={m:<3} N={N} K={K} rms={rms:.3} max_ulp={ulp} over_budget={ob} \
             over_ulp_only={ou} sign_flips={sf} max_abs={ma:.9} rel_rms={rr:.3e} \
             n64_over_budget={ob64} n64_max_ulp={ulp64} guards={g} | \
             m16_tc {tc_ms:.3}ms ({tcg:.1} GB/s) \
             vs n64 {n64_ms:.3}ms ({n64g:.1} GB/s) = {sp0:.2}x \
             vs batchm {batchm_ms:.3}ms ({bmg:.1} GB/s) = {sp1:.2}x  {verdict}",
            rms = d.rms,
            ulp = d.max_ulp,
            ob = d.over_budget.len(),
            ou = d.over_ulp_only,
            sf = d.sign_flips,
            ma = d.max_abs,
            rr = d.rel_rms,
            ob64 = d64.over_budget.len(),
            ulp64 = d64.max_ulp,
            g = if guards_intact { "ok" } else { "CLOBBERED" },
            tcg = gbs(tc_ms),
            n64g = gbs(n64_ms),
            bmg = gbs(batchm_ms),
            sp0 = n64_ms / tc_ms,
            sp1 = batchm_ms / tc_ms,
            verdict = if ok { "PASS" } else { "FAIL" },
        );
        report_outliers(arms[0].label, &d, N_TILE);
        report_outliers(arms[1].label, &d64, N_TILE_WIDE);
        if m == MAX_M {
            // The acceptance line. Reported, NOT asserted: the numerics gate
            // is this file's pass/fail, and a perf target that fails the build
            // on a busy box would be a flaky test rather than a receipt.
            let best = tc_ms.min(n64_ms);
            println!(
                "TARGET M=16: {TARGET_MS} ms / {TARGET_GBS} GB/s — best arm {best:.3} ms \
                 ({:.1} GB/s), {:.2}x the round-7 nsys baseline of {NSYS_BATCHM_MS} ms: {}",
                gbs(best),
                NSYS_BATCHM_MS / best,
                if best <= TARGET_MS { "MET" } else { "MISSED" },
            );
        }
        if !ok {
            failures += 1;
        }
    }

    // Oracle self-checks: the comparison MUST refuse a known-bad block, so a
    // green report cannot mean "the comparison was vacuous". One at three ULP
    // on a LARGE value (which the absolute floor must not rescue), one that
    // MOVES A WHOLE ROW (the row/pitch defect the gate is really for).
    let mut good = vec![0_u8; sentinel.len()];
    gpu.copy_d2h(scalar_base, &mut good)?;
    let rows = &good[GUARD..GUARD + MAX_M * N * 2];
    let mut bad = good.clone();
    let idx = (GUARD..GUARD + N * 2)
        .step_by(2)
        .find(|i| {
            bf16::from_bits(u16::from_le_bytes([good[*i], good[*i + 1]]))
                .to_f32()
                .abs()
                > 1.0
        })
        .expect("baseline has a value above 1.0");
    let bits = u16::from_le_bytes([good[idx], good[idx + 1]]);
    bad[idx..idx + 2].copy_from_slice(&bits.wrapping_add(3).to_le_bytes());
    let caught = !compare_m16_tc_block(&bad[GUARD..GUARD + MAX_M * N * 2], rows, N, K)
        .over_budget
        .is_empty();
    println!("KNOWN_BAD three-ULP mutation on a |value| > 1: refused={caught}");
    ensure!(
        caught,
        "comparison oracle admitted a three-ULP mutation above the accumulation floor"
    );
    let mut shifted = good.clone();
    let (src, dst) = (GUARD + 8 * N * 2, GUARD + 9 * N * 2);
    let row8 = good[src..src + N * 2].to_vec();
    shifted[dst..dst + N * 2].copy_from_slice(&row8);
    let caught_row = !compare_m16_tc_block(&shifted[GUARD..GUARD + MAX_M * N * 2], rows, N, K)
        .over_budget
        .is_empty();
    println!("KNOWN_BAD misplaced output row (row 9 <- row 8): refused={caught_row}");
    ensure!(
        caught_row,
        "comparison oracle admitted a misplaced output row — the absolute floor is too wide"
    );
    // 🔴 THE PARTIAL-TAIL CONTROL — round 9's defect class (b). N=248,077 is
    // 7,753 CTAs at N_TILE=32 with a 13-column tail, and 248,064 is a multiple
    // of 32 AND 64, so BOTH arms end on the SAME 13 columns: "the two arms
    // agree, therefore it is not a tile-edge bug" does not cover the LAST CTA.
    // The metric has to be shown to catch a tail defect, so here it is served
    // the tail columns from 13 columns to their left and must refuse.
    let tail = N - (N / N_TILE) * N_TILE;
    let mut wrapped = good.clone();
    let (src, dst) = (GUARD + (N - 2 * tail) * 2, GUARD + (N - tail) * 2);
    let moved = good[src..src + tail * 2].to_vec();
    wrapped[dst..dst + tail * 2].copy_from_slice(&moved);
    let caught_tail = !compare_m16_tc_block(&wrapped[GUARD..GUARD + MAX_M * N * 2], rows, N, K)
        .over_budget
        .is_empty();
    println!(
        "KNOWN_BAD partial-tail store (last {tail} columns of the 7,753rd CTA shifted): \
         refused={caught_tail}"
    );
    ensure!(
        caught_tail,
        "comparison oracle admitted a shifted partial-CTA tail — the round-9 tail \
         hypothesis would have been unfalsifiable"
    );

    ensure!(
        failures == 0,
        "{failures} M cases exceeded the {M16_TC_MAX_ULP}-ULP / accumulation-floor criterion \
         or the {REL_RMS_GATE} rel_rms budget, or broke a guard"
    );
    println!(
        "ALL PASS: real Qwen3.8-27B BF16 lm_head [{N}, {K}], dense_gemm_m16_bf16 AND \
         dense_gemm_m16_bf16_n64 at M 5/8/13/16 within {M16_TC_MAX_ULP} BF16 ULP (or the \
         accumulation floor) and {REL_RMS_GATE} rel_rms of the scalar dense_gemv_bf16, \
         [M,N] bounds intact"
    );
    Ok(())
}
