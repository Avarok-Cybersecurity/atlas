// SPDX-License-Identifier: AGPL-3.0-only

//! HOST SIMULATION of the ARITHMETIC of the two Hopper GDN prefill remnant
//! twins (#928) — the half of their contract that a shape check cannot reach.
//!
//! Neither twin can run on any device this repository's CI or its GB10 boxes
//! own (they are `kernels/hopper` sources and need sm_90a), so the claim "two
//! bf16 limbs put the operand error two orders inside the bf16 output floor"
//! would otherwise ship with no evidence at all until an H100 appears. It is
//! decidable on a CPU: reproduce each algorithm in the precision the kernel
//! uses, run BOTH it and the parent against an f64 reference on the same
//! fixture, and require the twin to land on the parent's own floor.
//!
//! WHAT IS AND IS NOT MODELLED. Operand rounding (bf16 limbs), accumulator
//! precision (f32), the order of the three limb products, the blocked solve's
//! block order and the explicit 16x16 diagonal inverse are all exact. The
//! 16-wide reduction tree INSIDE one `mma.sync.m16n8k16` is hardware-defined
//! and is modelled as an ascending f32 sum; that is a second-order difference
//! against operand rounding, and it is the one thing here a device would
//! settle differently. Read these numbers as "the algorithm's error budget",
//! not as a substitute for `native_gdn_prefill_remnants_microtest`.
//!
//! The fixture is the sibling microtests' recipe: a fixed LCG, gates in
//! [0.80, 0.999], beta in [0, 1], key/value in [-0.5, 0.5] stored bf16. It is
//! HARSHER than production, where GDN keys are L2-normalised and the Gram is
//! bounded by 1; here `<k_l, k_i>` has an rms near 0.94, so `(I + L)` is worse
//! conditioned than the real thing and the explicit diagonal inverse is being
//! asked a harder question than it will be asked in serve.

use half::bf16;

const KD: usize = 128;
const VD: usize = 128;
const C: usize = 64;
const BLK: usize = 16;
const NB: usize = C / BLK;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
/// hi = bf16(x), lo = bf16(x - hi) — `gdnh_split` in `gdn_prefill_hopper.cuh`.
fn limbs(x: f32) -> (f32, f32) {
    let h = bf(x);
    (h, bf(x - h))
}

fn rel_rms(got: &[f32], want: &[f64]) -> f64 {
    let (mut se, mut sr) = (0.0f64, 0.0f64);
    for (a, b) in got.iter().zip(want.iter()) {
        se += (*a as f64 - b) * (*a as f64 - b);
        sr += b * b;
    }
    if sr > 0.0 { (se / sr).sqrt() } else { 0.0 }
}

/// One `mma.sync` pass: `acc[m][n] += SUM_k a[m][k] * b[k][n]`, f32
/// accumulate, operands taken from the chosen bf16 limb of each.
fn mma_pass(
    acc: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    al: bool,
    bl: bool,
) {
    for i in 0..m {
        for j in 0..n {
            let mut s = acc[i * n + j];
            for t in 0..k {
                let (a_hi, a_lo) = limbs(a[i * k + t]);
                let (b_hi, b_lo) = limbs(b[t * n + j]);
                s += if al { a_lo } else { a_hi } * if bl { b_lo } else { b_hi };
            }
            acc[i * n + j] = s;
        }
    }
}

/// The twins' three-product limb scheme: Ah.Bh + Ah.Bl + Al.Bh, in that order.
fn mma_x2(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    mma_pass(acc, a, b, m, k, n, false, false);
    mma_pass(acc, a, b, m, k, n, false, true);
    mma_pass(acc, a, b, m, k, n, true, false);
}

// ── the fixture: one chunk's L, and the two right-hand sides ───────────────

struct Chunk {
    l: Vec<f32>,   // [C][C] f32, strict lower; the value both kernels build
    rhs: Vec<f32>, // [C][VD] f32, beta_i * V
    uc: Vec<f32>,  // [C][VD], already bf16-valued
    kq: Vec<f32>,  // [C][C] f32, exp(gc_i - gc_l) * <q_i, k_l>, masked l <= i
}

fn fixture(seed: u64) -> Chunk {
    let mut r = Lcg(seed);
    let key: Vec<f32> = (0..C * KD).map(|_| bf(r.r(-0.5, 0.5) as f32)).collect();
    let query: Vec<f32> = (0..C * KD).map(|_| bf(r.r(-0.5, 0.5) as f32)).collect();
    let val: Vec<f32> = (0..C * VD).map(|_| bf(r.r(-0.5, 0.5) as f32)).collect();
    let beta: Vec<f32> = (0..C).map(|_| r.r(0.0, 1.0) as f32).collect();
    let mut gc = vec![0.0f32; C];
    let mut a = 0.0f32;
    for g in gc.iter_mut() {
        a += r.r(0.80, 0.999).ln() as f32;
        *g = a;
    }
    // Gram in f32, ascending k — the same value the MMA produces on both paths.
    let gram = |x: &[f32], y: &[f32], i: usize, l: usize| -> f32 {
        let mut s = 0.0f32;
        for t in 0..KD {
            s += x[i * KD + t] * y[l * KD + t];
        }
        s
    };
    let mut l = vec![0.0f32; C * C];
    let mut kq = vec![0.0f32; C * C];
    for i in 0..C {
        for j in 0..C {
            if j < i {
                l[i * C + j] = beta[i] * (gc[i] - gc[j]).exp() * gram(&key, &key, i, j);
            }
            if j <= i {
                kq[i * C + j] = (gc[i] - gc[j]).exp() * gram(&query, &key, i, j);
            }
        }
    }
    let mut rhs = vec![0.0f32; C * VD];
    for i in 0..C {
        for v in 0..VD {
            rhs[i * VD + v] = beta[i] * val[i * VD + v];
        }
    }
    Chunk {
        l,
        rhs,
        uc: val,
        kq,
    }
}

// ── the three solvers ──────────────────────────────────────────────────────

/// The ORACLE: `(I + L) x = b` by plain forward substitution in f64.
fn reference_solve(ch: &Chunk) -> Vec<f64> {
    let mut x = vec![0.0f64; C * VD];
    for v in 0..VD {
        for i in 0..C {
            let mut s = ch.rhs[i * VD + v] as f64;
            for j in 0..i {
                s -= ch.l[i * C + j] as f64 * x[j * VD + v];
            }
            x[i * VD + v] = s;
        }
    }
    x
}

/// The PARENT: right-looking blocked forward substitution, RL_BLK = 16, one
/// thread per column, f32 throughout, bf16 on store. Transcribed from
/// `gated_delta_rule_recompute_wu`'s two solve passes.
fn parent_solve(ch: &Chunk) -> Vec<f32> {
    let mut out = vec![0.0f32; C * VD];
    for v in 0..VD {
        let mut acc: Vec<f32> = (0..C).map(|i| ch.rhs[i * VD + v]).collect();
        for jb in (0..C).step_by(BLK) {
            let mut xb = [0.0f32; BLK];
            for r in 0..BLK {
                let mut x = acc[jb + r];
                for q in 0..r {
                    x -= ch.l[(jb + r) * C + jb + q] * xb[q];
                }
                xb[r] = x;
                acc[jb + r] = x;
                out[(jb + r) * VD + v] = bf(x);
            }
            for i in jb + BLK..C {
                let mut a = acc[i];
                for (q, xq) in xb.iter().enumerate() {
                    a -= ch.l[i * C + jb + q] * xq;
                }
                acc[i] = a;
            }
        }
    }
    out
}

/// The TWIN: blocked solve with an explicit f32 16x16 diagonal inverse applied
/// by MMA, and two bf16 limbs on every MMA operand. Transcribed from
/// `gdn_recompute_wu_hopper.cu` steps (2) and (3).
fn twin_solve(ch: &Chunk) -> Vec<f32> {
    // (2) T_jj = (I + L_jj)^-1, exact f32 forward substitution per column.
    let mut t = vec![0.0f32; NB * BLK * BLK];
    for j in 0..NB {
        for c in 0..BLK {
            t[j * BLK * BLK + c * BLK + c] = 1.0;
            for r in c + 1..BLK {
                let mut s = 0.0f32;
                for m in c..r {
                    s -= ch.l[(j * BLK + r) * C + j * BLK + m] * t[j * BLK * BLK + m * BLK + c];
                }
                t[j * BLK * BLK + r * BLK + c] = s;
            }
        }
    }
    // -L for the off-diagonal updates: an MMA only accumulates.
    let neg: Vec<f32> = ch.l.iter().map(|x| -x).collect();
    // (3) the solve. Every warp holds the same [64][n] panel shape; the column
    // split across warps is a partition, so one panel of VD columns is the
    // whole computation.
    let mut x = ch.rhs.clone();
    for j in 0..NB {
        let bj: Vec<f32> = x[j * BLK * VD..(j + 1) * BLK * VD].to_vec();
        let mut xj = vec![0.0f32; BLK * VD];
        mma_x2(
            &mut xj,
            &t[j * BLK * BLK..(j + 1) * BLK * BLK],
            &bj,
            BLK,
            BLK,
            VD,
        );
        x[j * BLK * VD..(j + 1) * BLK * VD].copy_from_slice(&xj);
        for i in j + 1..NB {
            let mut blk = vec![0.0f32; BLK * BLK];
            for r in 0..BLK {
                blk[r * BLK..(r + 1) * BLK]
                    .copy_from_slice(&neg[(i * BLK + r) * C + j * BLK..][..BLK]);
            }
            let mut acc: Vec<f32> = x[i * BLK * VD..(i + 1) * BLK * VD].to_vec();
            mma_x2(&mut acc, &blk, &xj, BLK, BLK, VD);
            x[i * BLK * VD..(i + 1) * BLK * VD].copy_from_slice(&acc);
        }
    }
    x.iter().map(|v| bf(*v)).collect()
}

// ── 1. the blocked solve lands on the parent's own bf16 floor ──────────────

#[test]
fn the_blocked_solve_matches_the_parent_within_the_bf16_storage_floor() {
    for seed in [0x0928_A11A_u64, 0x0928_B22B, 0x0928_C33C] {
        let ch = fixture(seed);
        let want = reference_solve(&ch);
        let par = rel_rms(&parent_solve(&ch), &want);
        let twin = rel_rms(&twin_solve(&ch), &want);
        // Printed, not just asserted: `cargo test -- --nocapture` is where the
        // error budget of this algorithm is READ, and a gate that only says
        // PASS cannot be compared against the H100 microtest when it lands.
        println!("wu solve seed {seed:#x}: parent {par:.4e}  blocked twin {twin:.4e}");
        // The parent's own deviation IS the bf16 storage floor of this output:
        // it accumulates in f32 and only the store rounds. The twin is required
        // to land on that floor, with the same 1.25x headroom the tensor-core
        // state spine's microtest uses for its bf16 tensors.
        assert!(
            twin <= 1.25 * par,
            "seed {seed:#x}: blocked solve rel_rms {twin:e} against the parent's \
             {par:e}; two bf16 limbs must keep the solve on the storage floor"
        );
        // ...and the floor itself must be a bf16 floor, not something larger
        // that both arms happen to share.
        assert!(
            par < 5e-3,
            "seed {seed:#x}: parent rel_rms {par:e} is not a bf16 floor"
        );
    }
}

/// NEGATIVE CONTROL. The test above is only evidence if ONE limb fails it —
/// otherwise it says nothing about why the kernel pays for three products.
/// Same solve, `Ah.Bh` only.
#[test]
fn a_single_bf16_limb_does_not_meet_the_contract() {
    let ch = fixture(0x0928_A11A);
    let want = reference_solve(&ch);
    let par = rel_rms(&parent_solve(&ch), &want);
    let mut t = vec![0.0f32; NB * BLK * BLK];
    for j in 0..NB {
        for c in 0..BLK {
            t[j * BLK * BLK + c * BLK + c] = 1.0;
            for r in c + 1..BLK {
                let mut s = 0.0f32;
                for m in c..r {
                    s -= ch.l[(j * BLK + r) * C + j * BLK + m] * t[j * BLK * BLK + m * BLK + c];
                }
                t[j * BLK * BLK + r * BLK + c] = s;
            }
        }
    }
    let neg: Vec<f32> = ch.l.iter().map(|x| -x).collect();
    let mut x = ch.rhs.clone();
    for j in 0..NB {
        let bj: Vec<f32> = x[j * BLK * VD..(j + 1) * BLK * VD].to_vec();
        let mut xj = vec![0.0f32; BLK * VD];
        mma_pass(
            &mut xj,
            &t[j * BLK * BLK..(j + 1) * BLK * BLK],
            &bj,
            BLK,
            BLK,
            VD,
            false,
            false,
        );
        x[j * BLK * VD..(j + 1) * BLK * VD].copy_from_slice(&xj);
        for i in j + 1..NB {
            let mut blk = vec![0.0f32; BLK * BLK];
            for r in 0..BLK {
                blk[r * BLK..(r + 1) * BLK]
                    .copy_from_slice(&neg[(i * BLK + r) * C + j * BLK..][..BLK]);
            }
            let mut acc: Vec<f32> = x[i * BLK * VD..(i + 1) * BLK * VD].to_vec();
            mma_pass(&mut acc, &blk, &xj, BLK, BLK, VD, false, false);
            x[i * BLK * VD..(i + 1) * BLK * VD].copy_from_slice(&acc);
        }
    }
    let one: Vec<f32> = x.iter().map(|v| bf(*v)).collect();
    let got = rel_rms(&one, &want);
    println!("wu solve, ONE bf16 limb: {got:.4e} against the parent's {par:.4e}");
    assert!(
        got > 1.25 * par,
        "one bf16 limb measured {got:e} against the parent's {par:e}; if a \
         single limb were enough, the kernel's second and third products are \
         MMA issue spent on nothing and the header's claim is wrong"
    );
}

// ── 2. the masked triangular product ───────────────────────────────────────

/// `fwd_o`'s remnant, three ways: the parent's dependent f32 chain over
/// `l <= i`, the twin's masked square on two bf16 limbs of `kq~`, and f64.
#[test]
fn the_masked_triangular_product_matches_the_parent() {
    let ch = fixture(0x0928_F0F0);
    let mut want = vec![0.0f64; C * VD];
    let mut par = vec![0.0f32; C * VD];
    for i in 0..C {
        for v in 0..VD {
            let mut sw = 0.0f64;
            let mut sp = 0.0f32;
            for l in 0..=i {
                sw += ch.kq[i * C + l] as f64 * ch.uc[l * VD + v] as f64;
                sp += ch.kq[i * C + l] * ch.uc[l * VD + v];
            }
            want[i * VD + v] = sw;
            // The parent stores bf16; so does the twin. Round both.
            par[i * VD + v] = bf(sp);
        }
    }
    // The twin: `kq~` is already masked to zero above `i`, so the MMA over the
    // full 64-wide square is the same sum. Two limbs on `kq~`; `uc` is bf16 in
    // memory on both paths and is exact here.
    let mut acc = vec![0.0f32; C * VD];
    mma_pass(&mut acc, &ch.kq, &ch.uc, C, C, VD, false, false);
    mma_pass(&mut acc, &ch.kq, &ch.uc, C, C, VD, true, false);
    let twin: Vec<f32> = acc.iter().map(|v| bf(*v)).collect();

    let rp = rel_rms(&par, &want);
    let rt = rel_rms(&twin, &want);
    println!("fwd_o tril: parent {rp:.4e}  masked-square twin {rt:.4e}");
    assert!(
        rt <= 1.25 * rp,
        "masked square rel_rms {rt:e} against the parent's {rp:e}"
    );
    assert!(rp < 5e-3, "parent rel_rms {rp:e} is not a bf16 floor");
}

/// NEGATIVE CONTROL for the mask. Dropping it — letting the MMA see the raw
/// Gram above the diagonal, which is exactly what the parent's shared `kq`
/// buffer holds there — must be visible, or the twin's extra fold work is
/// unjustified and a future edit could remove it silently.
#[test]
fn an_unmasked_gram_breaks_the_triangular_product() {
    let ch = fixture(0x0928_F0F0);
    let mut want = vec![0.0f64; C * VD];
    for i in 0..C {
        for v in 0..VD {
            let mut s = 0.0f64;
            for l in 0..=i {
                s += ch.kq[i * C + l] as f64 * ch.uc[l * VD + v] as f64;
            }
            want[i * VD + v] = s;
        }
    }
    // A plausible slip: the fold writes every (i, l) it owns without the
    // `l <= i` test, so the strictly-upper half carries the decayed Gram.
    let mut bad = ch.kq.clone();
    let mut r = Lcg(0xBAD_0928);
    for i in 0..C {
        for l in i + 1..C {
            bad[i * C + l] = r.r(-1.0, 1.0) as f32;
        }
    }
    let mut acc = vec![0.0f32; C * VD];
    mma_pass(&mut acc, &bad, &ch.uc, C, C, VD, false, false);
    mma_pass(&mut acc, &bad, &ch.uc, C, C, VD, true, false);
    let got: Vec<f32> = acc.iter().map(|v| bf(*v)).collect();
    assert!(
        rel_rms(&got, &want) > 0.1,
        "an unmasked upper triangle must move the result"
    );
}
