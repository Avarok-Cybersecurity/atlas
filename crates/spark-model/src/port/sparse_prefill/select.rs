// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after XAttention (antidiagonal block scoring for block-sparse prefill), described in its paper;
// independent implementation, no source copied (the reference repository has no detected license).

//! Responsibility: the block-selection rule of one (head, query block), stated twice: [`select_f32`] is
//! the kernel's exact procedure (forced blocks, then a binary search on the FP32 bit pattern of the
//! cut), [`select_greedy`] is the mathematical definition it implements (largest mass first until the
//! kept mass reaches the threshold, ties at the cut kept). Plus the comparison the GPU oracle uses when
//! two selections come from masses that differ by rounding.
//!
//! Candidates are the key blocks `0..=jd` of a query block whose diagonal block is `jd`; `mass[j]` is
//! the share of the block's strided softmax that falls in key block `j` (they sum to 1).

/// The FP32 bit pattern of +infinity: a cut no finite mass reaches.
pub const INF_BITS: u32 = 0x7F80_0000;

/// Key block `j` is always kept: the sink block 0, the diagonal block `jd` and its neighbor `jd - 1`.
pub fn forced(j: usize, jd: usize) -> bool {
    j == 0 || j == jd || j + 1 == jd
}

/// The kernel's selection from FP32 masses (`mass.len()` = candidates). Sums run in index order here
/// and in a fixed parallel order in the kernel, so the two agree up to FP32 rounding of the partial
/// sums; on dyadic masses (exact sums) they agree bit for bit.
pub fn select_f32(mass: &[f32], threshold: f32) -> Vec<bool> {
    let n = mass.len();
    if n == 0 {
        return Vec::new();
    }
    if threshold.is_nan() || threshold >= 1.0 {
        return vec![true; n];
    }
    let cut = cut_bits(mass, threshold);
    let jd = n - 1;
    (0..n)
        .map(|j| forced(j, jd) || mass[j].to_bits() >= cut)
        .collect()
}

/// The cut the kernel searches for: every non-forced block whose mass has bits `>= cut` is kept.
/// `INF_BITS` = only the forced blocks; `0` = every block.
pub fn cut_bits(mass: &[f32], threshold: f32) -> u32 {
    let n = mass.len();
    if n == 0 {
        return 0;
    }
    let jd = n - 1;
    let mut mf = mass[0];
    if jd >= 1 {
        mf += mass[jd];
    }
    if jd >= 2 {
        mf += mass[jd - 1];
    }
    let need = threshold - mf;
    if need <= 0.0 {
        return INF_BITS;
    }
    let kept_from = |bits: u32| -> f32 {
        (0..n)
            .filter(|&j| !forced(j, jd) && mass[j].to_bits() >= bits)
            .map(|j| mass[j])
            .sum()
    };
    if kept_from(0) < need {
        return 0;
    }
    let (mut lo, mut hi) = (0u32, INF_BITS);
    while hi - lo > 1 {
        let mid = lo + ((hi - lo) >> 1);
        if kept_from(mid) >= need {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The definition: forced blocks, then the others in decreasing mass until the kept mass reaches
/// `threshold`; every block with the same mass as the last one taken is kept too. FP64 sums.
pub fn select_greedy(mass: &[f64], threshold: f64) -> Vec<bool> {
    let n = mass.len();
    if n == 0 {
        return Vec::new();
    }
    if threshold.is_nan() || threshold >= 1.0 {
        return vec![true; n];
    }
    let jd = n - 1;
    let mut keep: Vec<bool> = (0..n).map(|j| forced(j, jd)).collect();
    let mut kept: f64 = (0..n).filter(|&j| keep[j]).map(|j| mass[j]).sum();
    let mut order: Vec<usize> = (0..n).filter(|&j| !keep[j]).collect();
    order.sort_by(|&a, &b| mass[b].total_cmp(&mass[a]).then(a.cmp(&b)));
    let mut last: Option<f64> = None;
    for &j in &order {
        if kept >= threshold {
            break;
        }
        keep[j] = true;
        kept += mass[j];
        last = Some(mass[j]);
    }
    if let Some(v) = last {
        for &j in &order {
            if mass[j] == v {
                keep[j] = true;
            }
        }
    }
    keep
}

/// The smallest mass among the kept non-forced blocks (the effective cut), if any was kept.
pub fn cut_value(mass: &[f64], keep: &[bool]) -> Option<f64> {
    let jd = mass.len().checked_sub(1)?;
    (0..mass.len())
        .filter(|&j| keep[j] && !forced(j, jd))
        .map(|j| mass[j])
        .min_by(|a, b| a.total_cmp(b))
}

/// The largest mass among the non-forced blocks `keep` leaves out (the next block the rule would take).
pub fn next_value(mass: &[f64], keep: &[bool]) -> Option<f64> {
    let jd = mass.len().checked_sub(1)?;
    (0..mass.len())
        .filter(|&j| !keep[j] && !forced(j, jd))
        .map(|j| mass[j])
        .max_by(|a, b| a.total_cmp(b))
}

/// How a selection `b` differs from the reference selection `a` (made from `mass`):
/// `(differing blocks, differing blocks whose mass is farther than `band` from the reference's
/// boundary)`. The boundary is the reference's last kept mass and its next candidate's mass; two
/// selections from masses that differ by rounding may disagree only there, so the second count must be
/// zero.
pub fn mismatches(a: &[bool], b: &[bool], mass: &[f64], band: f64) -> (usize, usize) {
    let edges = [cut_value(mass, a), next_value(mass, a)];
    let near = |m: f64| edges.iter().flatten().any(|c| (m - c).abs() <= band);
    let mut total = 0;
    let mut outside = 0;
    for ((x, y), m) in a.iter().zip(b).zip(mass) {
        if x != y {
            total += 1;
            if !near(*m) {
                outside += 1;
            }
        }
    }
    (total, outside)
}
