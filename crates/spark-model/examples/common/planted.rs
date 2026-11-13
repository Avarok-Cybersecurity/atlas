// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the planted-control gate harness shared by the L07 examples,
//! plus the small BF16 byte helpers the gates use.
//!
//! Method credit: TensorFold's kernel contract (`docs/kernel-contract.md`,
//! gates 1 to 5): every gate has a deliberately broken variant that must fail,
//! and "a run where a control passes is a failed run". The implementation here
//! is Atlas-native.
//!
//! A gate is a closure `check(variant) -> Verdict`:
//! * `check(None)` runs the check on the real data and must pass;
//! * `check(Some(control))` runs the SAME check on a deliberately broken
//!   variant (a flipped ULP, a dropped group, an L2-resident ring) and must
//!   FAIL, which proves the check can see that class of defect.
//!
//! A gate with no control, or whose control passes, is VACUOUS and counts as a
//! failure, because it would also pass on broken kernels. [`Gates::finish`]
//! returns an error (so the example exits non-zero) when any gate failed.
#![allow(dead_code)]

use anyhow::{Result, bail};

/// The outcome of one check run.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub ok: bool,
    pub detail: String,
}

impl Verdict {
    pub fn new(ok: bool, detail: impl Into<String>) -> Self {
        Self {
            ok,
            detail: detail.into(),
        }
    }

    pub fn pass(detail: impl Into<String>) -> Self {
        Self::new(true, detail)
    }

    pub fn fail(detail: impl Into<String>) -> Self {
        Self::new(false, detail)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    /// The real check failed.
    Fail,
    /// No control, a control that passed, or a control that could not run.
    Vacuous,
    Skip,
}

#[derive(Clone, Debug)]
pub struct GateResult {
    pub name: String,
    pub status: Status,
    pub detail: String,
}

/// Judge one gate from its real verdict and its control verdicts.
pub fn judge(name: &str, real: &Verdict, controls: &[(String, Verdict)]) -> GateResult {
    let mk = |status, detail: String| GateResult {
        name: name.to_string(),
        status,
        detail,
    };
    if !real.ok {
        return mk(Status::Fail, format!("real check failed: {}", real.detail));
    }
    if controls.is_empty() {
        return mk(
            Status::Vacuous,
            "no planted control was supplied, so a pass proves nothing".to_string(),
        );
    }
    let missed: Vec<&str> = controls
        .iter()
        .filter(|(_, v)| v.ok)
        .map(|(n, _)| n.as_str())
        .collect();
    if !missed.is_empty() {
        return mk(
            Status::Vacuous,
            format!(
                "planted control(s) {} PASSED, so this gate cannot see that defect",
                missed.join(", ")
            ),
        );
    }
    mk(
        Status::Pass,
        format!(
            "{}; controls detected {}/{}",
            real.detail,
            controls.len(),
            controls.len()
        ),
    )
}

/// Collects gate results and prints one line per gate.
pub struct Gates {
    title: String,
    results: Vec<GateResult>,
}

impl Gates {
    pub fn new(title: &str) -> Self {
        println!("== gates: {title}");
        Self {
            title: title.to_string(),
            results: Vec::new(),
        }
    }

    /// Run `check` on the real data and on every named control, judge, print.
    pub fn gate(
        &mut self,
        name: &str,
        controls: &[&str],
        mut check: impl FnMut(Option<&str>) -> Result<Verdict>,
    ) -> Status {
        let result = match check(None) {
            Err(e) => GateResult {
                name: name.to_string(),
                status: Status::Fail,
                detail: format!("real check errored: {e:#}"),
            },
            Ok(real) => {
                let mut cs = Vec::new();
                let mut ctl_error = None;
                for c in controls {
                    match check(Some(*c)) {
                        Ok(v) => cs.push(((*c).to_string(), v)),
                        Err(e) => {
                            ctl_error = Some(format!("control {c} could not run: {e:#}"));
                            break;
                        }
                    }
                }
                match ctl_error {
                    Some(d) => GateResult {
                        name: name.to_string(),
                        status: Status::Vacuous,
                        detail: d,
                    },
                    None => judge(name, &real, &cs),
                }
            }
        };
        self.record(result)
    }

    /// Record a gate that does not apply here (a kernel is absent, a backend
    /// lacks a capability). A skip is not a pass and is not a failure.
    pub fn skip(&mut self, name: &str, why: &str) {
        self.record(GateResult {
            name: name.to_string(),
            status: Status::Skip,
            detail: why.to_string(),
        });
    }

    fn record(&mut self, r: GateResult) -> Status {
        let tag = match r.status {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
            Status::Vacuous => "FAIL(vacuous)",
            Status::Skip => "SKIP",
        };
        println!("  [{tag}] {}: {}", r.name, r.detail);
        let s = r.status;
        self.results.push(r);
        s
    }

    /// Gates that failed or were vacuous.
    pub fn failures(&self) -> usize {
        self.results
            .iter()
            .filter(|r| matches!(r.status, Status::Fail | Status::Vacuous))
            .count()
    }

    pub fn passes(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.status == Status::Pass)
            .count()
    }

    /// Print the summary line; an error when any gate failed.
    pub fn finish(self) -> Result<()> {
        let f = self.failures();
        println!(
            "== {}: {} gate(s) passed, {} failed, {} skipped",
            self.title,
            self.passes(),
            f,
            self.results.len() - self.passes() - f
        );
        if f > 0 {
            bail!("{f} gate(s) failed in {}", self.title);
        }
        Ok(())
    }
}

/// How two BF16 byte buffers differ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf16Diff {
    pub n_diff: usize,
    pub worst_abs: f32,
    pub first: Option<usize>,
}

fn bf16_at(b: &[u8], i: usize) -> f32 {
    half::bf16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32()
}

/// Element-wise comparison of two BF16 buffers. A length mismatch counts every
/// element of the longer buffer past the shorter as different.
pub fn diff_bf16_bytes(a: &[u8], b: &[u8]) -> Bf16Diff {
    let (na, nb) = (a.len() / 2, b.len() / 2);
    let common = na.min(nb);
    let mut d = Bf16Diff {
        n_diff: na.max(nb) - common,
        worst_abs: 0.0,
        first: (na != nb).then_some(common),
    };
    for i in 0..common {
        if a[2 * i..2 * i + 2] != b[2 * i..2 * i + 2] {
            d.n_diff += 1;
            d.first.get_or_insert(i);
            let delta = (bf16_at(a, i) - bf16_at(b, i)).abs();
            if delta.is_nan() || delta > d.worst_abs {
                d.worst_abs = delta;
            }
        }
    }
    d
}

/// A byte-equality verdict for two BF16 buffers.
pub fn verdict_bytes_equal(label: &str, a: &[u8], b: &[u8]) -> Verdict {
    let d = diff_bf16_bytes(a, b);
    if d.n_diff == 0 {
        Verdict::pass(format!("{label}: {} elements byte-identical", a.len() / 2))
    } else {
        Verdict::fail(format!(
            "{label}: {} of {} elements differ (first {:?}, max |delta| {})",
            d.n_diff,
            a.len().max(b.len()) / 2,
            d.first,
            d.worst_abs
        ))
    }
}

/// Flip the lowest mantissa bit of BF16 element `elem` (a 1-ULP perturbation).
pub fn flip_one_ulp_bf16(bytes: &mut [u8], elem: usize) {
    bytes[2 * elem] ^= 1;
}

/// True when every BF16 element is finite.
pub fn all_finite_bf16(bytes: &[u8]) -> bool {
    (0..bytes.len() / 2).all(|i| bf16_at(bytes, i).is_finite())
}

/// Count of BF16 elements that are not +0 or -0.
pub fn count_nonzero_bf16(bytes: &[u8]) -> usize {
    (0..bytes.len() / 2)
        .filter(|&i| bf16_at(bytes, i) != 0.0)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(n: &str, ok: bool) -> (String, Verdict) {
        (n.to_string(), Verdict::new(ok, "x"))
    }

    #[test]
    fn a_detected_control_passes_the_gate() {
        let r = judge(
            "g",
            &Verdict::pass("ok"),
            &[named("a", false), named("b", false)],
        );
        assert_eq!(r.status, Status::Pass);
        assert!(r.detail.contains("2/2"), "{}", r.detail);
    }

    #[test]
    fn a_passing_control_makes_the_gate_vacuous() {
        let r = judge(
            "g",
            &Verdict::pass("ok"),
            &[named("a", false), named("b", true)],
        );
        assert_eq!(r.status, Status::Vacuous);
        assert!(r.detail.contains("b PASSED"), "{}", r.detail);
    }

    #[test]
    fn no_control_is_vacuous_and_a_real_failure_is_a_failure() {
        assert_eq!(
            judge("g", &Verdict::pass("ok"), &[]).status,
            Status::Vacuous
        );
        assert_eq!(
            judge("g", &Verdict::fail("bad"), &[named("a", false)]).status,
            Status::Fail
        );
    }

    #[test]
    fn gate_runs_the_same_check_on_each_variant() {
        let mut g = Gates::new("unit");
        // The check passes only on the real data: a detecting gate.
        let s = g.gate("equal", &["flip"], |v| {
            Ok(Verdict::new(v.is_none(), "compare"))
        });
        assert_eq!(s, Status::Pass);
        // A check that passes on everything cannot detect its control.
        let s = g.gate("blind", &["flip"], |_| Ok(Verdict::pass("always")));
        assert_eq!(s, Status::Vacuous);
        // An erroring control is vacuous, an erroring real check is a failure.
        let s = g.gate("ctl-err", &["flip"], |v| match v {
            None => Ok(Verdict::pass("ok")),
            Some(_) => bail!("cannot build the control"),
        });
        assert_eq!(s, Status::Vacuous);
        let s = g.gate("real-err", &["flip"], |_| bail!("launch failed"));
        assert_eq!(s, Status::Fail);
        g.skip("absent", "kernel not in this target");
        assert_eq!((g.passes(), g.failures()), (1, 3));
        assert!(g.finish().is_err());
    }

    #[test]
    fn finish_is_ok_when_everything_passed() {
        let mut g = Gates::new("unit-ok");
        g.gate("a", &["c"], |v| Ok(Verdict::new(v.is_none(), "")));
        g.skip("b", "n/a");
        assert!(g.finish().is_ok());
    }

    #[test]
    fn one_ulp_flip_is_seen_by_the_byte_compare() {
        let base: Vec<u8> = (0..8u16)
            .flat_map(|i| {
                half::bf16::from_f32(1.0 + f32::from(i))
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let mut b = base.clone();
        assert_eq!(diff_bf16_bytes(&base, &b).n_diff, 0);
        flip_one_ulp_bf16(&mut b, 3);
        let d = diff_bf16_bytes(&base, &b);
        assert_eq!((d.n_diff, d.first), (1, Some(3)));
        assert!(d.worst_abs > 0.0);
        assert!(!verdict_bytes_equal("t", &base, &b).ok);
        assert!(verdict_bytes_equal("t", &base, &base).ok);
    }

    #[test]
    fn length_mismatch_counts_as_different() {
        let a = vec![0u8; 8];
        let b = vec![0u8; 4];
        let d = diff_bf16_bytes(&a, &b);
        assert_eq!((d.n_diff, d.first), (2, Some(2)));
    }

    #[test]
    fn finite_and_nonzero_counts() {
        let mut v: Vec<u8> = Vec::new();
        for x in [0.0f32, 1.0, -2.0, 0.0] {
            v.extend(half::bf16::from_f32(x).to_bits().to_le_bytes());
        }
        assert!(all_finite_bf16(&v));
        assert_eq!(count_nonzero_bf16(&v), 2);
        v.extend(half::bf16::from_f32(f32::NAN).to_bits().to_le_bytes());
        assert!(!all_finite_bf16(&v));
    }
}
