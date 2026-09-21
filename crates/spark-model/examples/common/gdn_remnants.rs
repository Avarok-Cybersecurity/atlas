// SPDX-License-Identifier: AGPL-3.0-only
//! Fixture, guarded allocation and f64 references shared by
//! `native_gdn_prefill_remnants_microtest` (#928).
//!
//! Split out of that example only because Atlas caps a Rust source at 500 LoC;
//! it is one oracle, and nothing else includes this file. Lives under
//! `examples/common/` so cargo does not pick it up as an example target of its
//! own — `examples/*.rs` is auto-discovered, `examples/common/*.rs` is not.

use anyhow::Result;
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub const KD: usize = 128;
pub const VD: usize = 128;
pub const NK: usize = 16;
pub const NV: usize = 48;
pub const C: usize = 64;
/// Heads the f64 reference recomputes. Every chunk and head is independent
/// here, so 2 of 48 is a complete check of the math at 1/24 of the CPU cost.
pub const REF_HEADS: usize = 2;
/// Sentinel tail on every kernel output, in bytes.
pub const GUARD: usize = 512;
pub const GUARD_BYTE: u8 = 0xA5;

pub struct Lcg(pub u64);
impl Lcg {
    pub fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    pub fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

pub fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
/// An output buffer with a sentinel tail. The kernels below write from C
/// fragments, not from `for i < ce` loops, so "did it stay inside the tensor"
/// is a live question and not a formality.
pub fn alloc_guarded(g: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    let p = g.alloc(bytes + GUARD)?;
    g.copy_h2d(&vec![GUARD_BYTE; bytes + GUARD], p)?;
    Ok(p)
}
pub fn guard_intact(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<bool> {
    let mut tail = vec![0u8; GUARD];
    g.copy_d2h(DevicePtr(p.0 + bytes as u64), &mut tail)?;
    Ok(tail.iter().all(|b| *b == GUARD_BYTE))
}
pub fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
pub fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
/// max_abs, rel_rms = ||a-r||/||r||. Reference in f64.
pub fn metrics(a: &[f32], r: &[f64]) -> (f64, f64) {
    let (mut mx, mut se, mut sr) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(r.iter()) {
        let d = (*x as f64 - y).abs();
        mx = mx.max(d);
        se += d * d;
        sr += y * y;
    }
    (mx, if sr > 0.0 { (se / sr).sqrt() } else { 0.0 })
}

pub struct Case {
    pub t: usize,
    pub nt: usize,
    pub query: Vec<bf16>,
    pub key: Vec<bf16>,
    pub val: Vec<bf16>,
    pub gate: Vec<f32>,
    pub beta: Vec<f32>,
    pub h0: Vec<f32>,
}

/// The sibling GDN microtests' fixture recipe: fixed LCG, gates in
/// [0.80, 0.999], beta in [0, 1].
pub fn gen_case(t: usize) -> Case {
    let mut r = Lcg(0x9D8E_2026 ^ (t as u64));
    let bf = |r: &mut Lcg| bf16::from_f64(r.r(-0.5, 0.5));
    Case {
        t,
        nt: t.div_ceil(C),
        query: (0..t * NK * KD).map(|_| bf(&mut r)).collect(),
        key: (0..t * NK * KD).map(|_| bf(&mut r)).collect(),
        val: (0..t * NV * VD).map(|_| bf(&mut r)).collect(),
        gate: (0..t * NV).map(|_| r.r(0.80, 0.999) as f32).collect(),
        beta: (0..t * NV).map(|_| r.r(0.0, 1.0) as f32).collect(),
        h0: (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect(),
    }
}

/// f64 reference for the WY pass on heads [0, REF_HEADS): gc scan, Gram, L,
/// then plain forward substitution. Laid out in the kernels' index order.
pub fn ref_wu(c: &Case) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let hr = NV / NK;
    let (mut w, mut u, mut gc) = (
        vec![0.0; c.nt * REF_HEADS * C * KD],
        vec![0.0; c.nt * REF_HEADS * C * VD],
        vec![0.0; c.nt * REF_HEADS * C],
    );
    for vh in 0..REF_HEADS {
        let kh = vh / hr;
        for ch in 0..c.nt {
            let (cs, rb) = (ch * C, ch * REF_HEADS + vh);
            let ce = (c.t - cs).min(C);
            let mut g = vec![0.0f64; C];
            let mut a = 0.0f64;
            for (i, gi) in g.iter_mut().enumerate().take(ce) {
                a += (c.gate[(cs + i) * NV + vh] as f64).max(1e-30).ln();
                *gi = a;
                gc[rb * C + i] = a;
            }
            let kv = |i: usize, d: usize| c.key[(cs + i) * NK * KD + kh * KD + d].to_f64();
            let mut l = vec![0.0f64; C * C];
            for i in 0..ce {
                for j in 0..i {
                    let gram: f64 = (0..KD).map(|d| kv(i, d) * kv(j, d)).sum();
                    l[i * C + j] = c.beta[(cs + i) * NV + vh] as f64 * (g[i] - g[j]).exp() * gram;
                }
            }
            for (n, (out, cols)) in [(0usize, VD), (1, KD)].iter().enumerate() {
                let _ = out;
                for col in 0..*cols {
                    for i in 0..ce {
                        let b = c.beta[(cs + i) * NV + vh] as f64;
                        let mut s = if n == 0 {
                            b * c.val[(cs + i) * NV * VD + vh * VD + col].to_f64()
                        } else {
                            b * g[i].exp() * kv(i, col)
                        };
                        for j in 0..i {
                            s -= l[i * C + j]
                                * if n == 0 {
                                    u[rb * C * VD + j * VD + col]
                                } else {
                                    w[rb * C * KD + j * KD + col]
                                };
                        }
                        if n == 0 {
                            u[rb * C * VD + i * VD + col] = s;
                        } else {
                            w[rb * C * KD + i * KD + col] = s;
                        }
                    }
                }
            }
        }
    }
    (w, u, gc)
}

/// f64 reference for the output pass, given the spine's own `S_c` and `uc`
/// (bf16 on both arms, so they are inputs here, not things being scored).
pub fn ref_fwd_o(c: &Case, sc: &[f32], uc: &[f32], gc: &[f32]) -> Vec<f64> {
    let hr = NV / NK;
    let inv = 1.0 / (KD as f64).sqrt();
    let mut o = vec![0.0f64; c.t * REF_HEADS * VD];
    for vh in 0..REF_HEADS {
        let kh = vh / hr;
        for ch in 0..c.nt {
            let (cs, base) = (ch * C, ch * NV + vh);
            let ce = (c.t - cs).min(C);
            let g = |i: usize| gc[base * C + i] as f64;
            let qk = |i: usize, l: usize| -> f64 {
                (0..KD)
                    .map(|d| {
                        c.query[(cs + i) * NK * KD + kh * KD + d].to_f64()
                            * c.key[(cs + l) * NK * KD + kh * KD + d].to_f64()
                    })
                    .sum()
            };
            for i in 0..ce {
                let kqi: Vec<f64> = (0..=i).map(|l| (g(i) - g(l)).exp() * qk(i, l)).collect();
                for v in 0..VD {
                    let mut s: f64 = (0..KD)
                        .map(|d| {
                            c.query[(cs + i) * NK * KD + kh * KD + d].to_f64()
                                * sc[base * KD * VD + d * VD + v] as f64
                        })
                        .sum();
                    s *= g(i).exp();
                    for (l, kq) in kqi.iter().enumerate() {
                        s += kq * uc[base * C * VD + l * VD + v] as f64;
                    }
                    o[((cs + i) * REF_HEADS + vh) * VD + v] = s * inv;
                }
            }
        }
    }
    o
}

/// Gather the reference heads out of a full-NV output laid out [row][NV][per].
///
/// `rows` is the OUTER dimension ONLY — the chunk count for `W`/`U`/`gc`, the
/// token count for `O`. This function applies `NV` itself, so a caller that
/// pre-multiplies (`nt * NV`, `t * NV`) applies it twice and walks off the end.
/// That is not hypothetical: it is exactly how
/// `native_gdn_prefill_remnants_microtest` panicked on its first-ever hardware
/// run (H100 round 13, 2026-09-11) — `take(&w, c.nt * NV, C * KD)` at T=256
/// indexed element 1 581 056 of a 1 572 864-element `W`, 0.06 s in, before the
/// first comparison. The length check below turns that class of slip into a
/// named failure at the call site instead of a slice-range panic 200 lines away,
/// and [`selfcheck_take`] runs it on every invocation of the example.
pub fn take(full: &[f32], rows: usize, per: usize) -> Vec<f32> {
    assert_eq!(
        full.len(),
        rows * NV * per,
        "take(rows={rows}, per={per}) gathers {REF_HEADS} of NV={NV} heads out \
         of a [rows][NV][per] buffer and therefore wants {} elements, but was \
         handed {}. `rows` is the OUTER dimension only — pass `c.nt` (or `t`), \
         never `c.nt * NV`: NV is applied here",
        rows * NV * per,
        full.len(),
    );
    let mut out = Vec::with_capacity(rows * REF_HEADS * per);
    for r in 0..rows {
        for vh in 0..REF_HEADS {
            let b = (r * NV + vh) * per;
            out.extend_from_slice(&full[b..b + per]);
        }
    }
    out
}

/// Every `take` the example performs, at the example's OWN geometry — run from
/// `main` before a single byte is allocated on the device.
///
/// It is a self-check and not only a `#[cfg(test)]` module because an
/// `examples/` target in this workspace has no test harness: `cargo test -p
/// spark-model` never compiles this file, so a unit test here alone would be
/// documentation that nothing executes. The unit tests below exist as well and
/// call this same function, so the two cannot describe different geometries.
///
/// Two halves. The GATHER is value-checked at T=256 — the shape round 13 died
/// on — with every element stamped by the `(row, head, offset)` it belongs to,
/// so a wrong-but-in-bounds `rows` is named rather than merely surviving. The
/// LENGTH relation is then checked at all three T against the buffer
/// expressions `main` actually allocates, which costs nothing and is what makes
/// "the exact geometry" true for T=4593 as well.
pub fn selfcheck_take() {
    // (what, rows, per) exactly as the example calls `take`: W, U and gc are
    // keyed by CHUNK, O by TOKEN. Nothing here pre-multiplies by NV.
    let shapes = |t: usize| {
        let nt = t.div_ceil(C);
        [
            ("W", nt, C * KD),
            ("U", nt, C * VD),
            ("gc", nt, C),
            ("O", t, VD),
        ]
    };

    for (what, rows, per) in shapes(256) {
        let full: Vec<f32> = (0..rows * NV * per).map(|i| i as f32).collect();
        let got = take(&full, rows, per);
        assert_eq!(got.len(), rows * REF_HEADS * per, "{what}: gathered length");
        for r in 0..rows {
            for vh in 0..REF_HEADS {
                for (e, x) in got[(r * REF_HEADS + vh) * per..][..per].iter().enumerate() {
                    assert_eq!(
                        *x,
                        ((r * NV + vh) * per + e) as f32,
                        "{what}: row {r} head {vh} element {e} came from the wrong stride",
                    );
                }
            }
        }
    }

    for &t in &[256usize, 1193, 4593] {
        let nt = t.div_ceil(C);
        // The four buffer lengths `main` allocates, spelled independently here.
        let (wb, ub, gcb, outs) = (nt * NV * C * KD, nt * NV * C * VD, nt * NV * C, t * NV * VD);
        for ((what, rows, per), len) in shapes(t).into_iter().zip([wb, ub, gcb, outs]) {
            assert_eq!(
                rows * NV * per,
                len,
                "T={t} {what}: take's (rows, per) must describe the buffer main \
                 allocates",
            );
        }
    }
}

/// THE KNOWN_BAD CONTROL, as one function of an arm's output and its f64
/// reference: perturb ONE reference element and report `(perturbed, clean)`
/// `max_abs`. The caller refuses the run unless `perturbed > clean`.
///
/// # Why the injection is priced in the clean extreme
///
/// H100 round 14 (`h100-round14-report.md`, §2.1 and anomaly 1). The control
/// added `0.1 * rms(reference)` to element 0 and required `max_abs` — a maximum
/// over `T * VD` elements — to move. Clean `max_abs` grows with T (1.895e-3 at
/// T=256, 7.668e-3 at T=1193, **2.774e-2** at T=4593) because a larger tensor
/// holds a larger extreme, while `0.1 * rms` does not keep pace. At T=4593 the
/// injection was already smaller than the worst element that was there anyway,
/// `max` could not change, and the example exited 1 with every numerics gate
/// green. It is arch-independent: it fails the same way on GB10, at a large
/// enough T.
///
/// So the injection is scaled to the clean extreme AND lands on the element
/// that holds it, pointing away from the arm's value. The perturbed deviation
/// there is `|d_j| + 3 * max_abs_clean + 0.1 * rms`, which exceeds
/// `max_abs_clean` by construction at every T and on every arch. The `rms`
/// term is a floor for a bit-exact arm, whose extreme is 0 and where a purely
/// multiplicative injection would be 0 too.
pub fn known_bad_probe(actual: &[f32], reference: &[f64]) -> (f64, f64) {
    assert_eq!(
        actual.len(),
        reference.len(),
        "the KNOWN_BAD control scores an arm against ITS OWN reference; unequal \
         lengths mean `metrics` silently truncated one of the two and the \
         control would be measuring a prefix",
    );
    assert!(
        !reference.is_empty(),
        "an empty reference cannot be perturbed"
    );
    let (clean, _) = metrics(actual, reference);
    // The element that HOLDS the clean extreme, and the signed deviation there.
    let (mut j, mut dj) = (0usize, 0.0f64);
    for (i, (x, y)) in actual.iter().zip(reference.iter()).enumerate() {
        let d = *x as f64 - y;
        if d.abs() > dj.abs() {
            (j, dj) = (i, d);
        }
    }
    let rms = (reference.iter().map(|x| x * x).sum::<f64>() / reference.len() as f64).sqrt();
    let mag = 3.0 * clean + 0.1 * rms;
    assert!(
        mag > 0.0,
        "an all-zero reference against a bit-exact arm leaves nothing to \
         perturb, and a control that cannot trip is not a control",
    );
    let mut bad = reference.to_vec();
    // AWAY from the arm's value, so the deviation at j adds rather than cancels.
    bad[j] -= mag * if dj < 0.0 { -1.0 } else { 1.0 };
    let (perturbed, _) = metrics(actual, &bad);
    (perturbed, clean)
}

/// Synthetic `(reference, actual)` at the example's own `O` geometry whose
/// clean `max_abs` is `clean_max`, held by ONE element that is deliberately not
/// element 0 — finding it is half of what [`known_bad_probe`] fixes.
///
/// The reference is drawn on [-0.35, 0.35] so `0.1 * rms` lands at ~2.0e-2,
/// between round 14's T=1193 and T=4593 extremes: that is what reproduces the
/// receipt's own trip pattern below.
fn known_bad_fixture(t: usize, clean_max: f64) -> (Vec<f64>, Vec<f32>) {
    let n = t * REF_HEADS * VD;
    let mut r = Lcg(0x4B4E_4F57 ^ (t as u64));
    let reference: Vec<f64> = (0..n).map(|_| r.r(-0.35, 0.35)).collect();
    // f64 -> f32 alone deviates by ~2e-8 here, orders under every extreme
    // below, so the extreme is the one planted.
    let mut actual: Vec<f32> = reference.iter().map(|x| *x as f32).collect();
    let ex = n / 2 + 7;
    actual[ex] = (reference[ex] + clean_max) as f32;
    (reference, actual)
}

/// [`known_bad_probe`] run on synthetic data at the three T the example walks
/// and at the clean extremes round 14 MEASURED there — from `main`, before the
/// device is touched, for the same reason [`selfcheck_take`] is: the standing
/// gate is `cargo test -p spark-model --lib`, which never reaches an
/// `examples/` target, so a unit test alone would be a check the gates do not
/// run. The unit test below calls this same function, so the two cannot
/// disagree.
///
/// Both directions are pinned. The shipped rule must TRIP at all three T, and
/// round 14's rule (`reference[0] += 0.1 * rms`) must be shown NOT to at the
/// largest — the defect itself, kept executable so it cannot come back unseen.
pub fn selfcheck_known_bad() {
    for (t, clean_max) in [(256usize, 1.895e-3f64), (1193, 7.668e-3), (4593, 2.774e-2)] {
        let (reference, actual) = known_bad_fixture(t, clean_max);
        let (perturbed, clean) = known_bad_probe(&actual, &reference);
        assert!(
            (clean / clean_max - 1.0).abs() < 1e-4,
            "T={t}: the fixture must hold round 14's clean extreme {clean_max:.3e}, \
             got {clean:.3e}",
        );
        assert!(
            perturbed > clean,
            "T={t}: the KNOWN_BAD control must trip against a clean extreme of \
             {clean:.3e}, got {perturbed:.3e} — round 14's failure, at T=4593",
        );
        // Round 14's rule, spelled out, on the same fixture: it trips only
        // while `0.1 * rms` still exceeds the extreme already present.
        let rms = (reference.iter().map(|x| x * x).sum::<f64>() / reference.len() as f64).sqrt();
        let mut old = reference.clone();
        old[0] += 0.1 * rms;
        let (old_perturbed, _) = metrics(&actual, &old);
        assert_eq!(
            old_perturbed > clean,
            t != 4593,
            "T={t}: round 14's `0.1 * rms` injection ({:.3e}) against a clean \
             extreme of {clean:.3e} — the scaling defect this replaces",
            0.1 * rms,
        );
    }
}

pub fn report(tag: &str, a: &[f32], r: &[f64]) -> f64 {
    let (mx, rel) = metrics(a, r);
    println!("    {tag:<22} max_abs={mx:.6e}  rel_rms={rel:.4e}");
    rel
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The KNOWN_BAD control, at the three T and the three clean extremes round
    /// 14 measured. Mirrors [`selfcheck_known_bad`]; see its docs.
    #[test]
    fn the_known_bad_control_trips_at_every_t() {
        selfcheck_known_bad();
    }

    /// The example's own geometry, gathered and value-checked. Mirrors
    /// [`selfcheck_take`]; see its docs for why the self-check exists too.
    #[test]
    fn take_gathers_the_example_geometry() {
        selfcheck_take();
    }

    /// THE REGRESSION. `rows = nt * NV` is the argument round 13 passed; it must
    /// be refused BY NAME rather than read off the end of the buffer.
    #[test]
    #[should_panic(expected = "`rows` is the OUTER dimension only")]
    fn a_pre_multiplied_rows_is_refused() {
        let (t, per) = (256usize, C * KD);
        let nt = t.div_ceil(C);
        let full = vec![0.0f32; nt * NV * per];
        let _ = take(&full, nt * NV, per);
    }

    /// …and the two numbers in round 13's panic are this geometry, so the
    /// diagnosis in `GDN-PREFILL-ATTRIBUTION.md` cannot drift from the code:
    /// `W` at T=256 is nt*NV*C*KD = 4*48*64*128 = 1 572 864 elements, and the
    /// doubled-NV walk's first out-of-range read is `b + per` at r = nt,
    /// vh = 0 — (4*48)*8192 + 8192 = 1 581 056.
    #[test]
    fn the_round_thirteen_panic_indices_are_this_geometry() {
        let (t, per) = (256usize, C * KD);
        let nt = t.div_ceil(C);
        assert_eq!(nt * NV * per, 1_572_864);
        assert_eq!((nt * NV) * per + per, 1_581_056);
    }
}
