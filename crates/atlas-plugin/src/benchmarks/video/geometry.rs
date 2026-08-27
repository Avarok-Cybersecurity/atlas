// SPDX-License-Identifier: AGPL-3.0-only

//! How many tokens a clip should cost, and why it is asserted as a RATIO.
//!
//! A still image's token count is fully determined by its pixels, so the image
//! ladder can assert absolute numbers. A clip's is not: it depends on the
//! server's `--video-fps`, which the benchmark does not set and should not
//! assume. Two ways out, and only one of them is honest.
//!
//! The dishonest one is to hard-code the default rate. It passes on a
//! default-configured server and fails on a correct one that was tuned, which
//! trains people to ignore the benchmark.
//!
//! The honest one is to assert what is TRUE AT EVERY RATE: a clip of twice the
//! duration has twice the temporal groups. That relation holds for any fps, is
//! violated the moment sampling or grouping breaks, and needs no knowledge of
//! how the server was started.

/// Merged tokens one temporal group of a `w x h` clip occupies.
///
/// Identical arithmetic to a still image's — a group IS a still, spatially —
/// so this deliberately mirrors `vision::geometry::expected_vision_tokens`
/// rather than reimplementing it differently.
pub fn tokens_per_group(w: u32, h: u32, patch: u32, merge: u32) -> u32 {
    crate::benchmarks::vision::geometry::expected_vision_tokens(w, h, patch, merge)
}

/// What the geometry leg concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ratio {
    /// Temporal groups inferred for the SHORTEST clip.
    pub unit_groups: usize,
    /// Template overhead implied, in tokens.
    pub overhead: usize,
}

/// Check that temporal groups scale with duration, from three clips at 1x, 2x
/// and 4x — and infer the group count and template overhead as a by-product.
///
/// ★ WHY THREE AND NOT TWO. Two measurements cannot test this. Given totals
/// `t1` and `t2` and a tokens-per-group figure, `groups = (t2 - t1) / plane`
/// always satisfies a 2:1 relation, because the implied overhead absorbs
/// whatever is left over — a sampler emitting 3 and 2 groups reads as "1 and 2
/// groups with a bigger template", indistinguishably. Two points always fit a
/// line. The first version of this function did exactly that and its own test
/// caught it.
///
/// With a third duration the system is over-determined, and proportionality
/// becomes a claim that can fail:
///
/// ```text
///   t4 - t2  ==  2 * (t2 - t1)
/// ```
///
/// That form is independent of BOTH the template overhead and the
/// tokens-per-group figure, so it needs no calibration and holds whatever
/// `--video-fps` the server was started with — which is the property the whole
/// leg is built around.
///
/// `plane` is used only to turn the verified differences into a reportable
/// group count; the pass/fail decision does not depend on it.
pub fn check_proportional(t1: usize, t2: usize, t4: usize, plane: usize) -> Option<Ratio> {
    let d21 = t2.checked_sub(t1)?;
    let d42 = t4.checked_sub(t2)?;
    if d21 == 0 || d42 != d21.checked_mul(2)? {
        return None;
    }
    if plane == 0 || d21 % plane != 0 {
        return None;
    }
    let unit_groups = d21 / plane;
    // The overhead the shortest clip implies must also explain the other two,
    // or the three points are not one line and the inference is meaningless.
    let step = unit_groups.checked_mul(plane)?;
    let overhead = t1.checked_sub(step)?;
    Some(Ratio {
        unit_groups,
        overhead,
    })
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;
