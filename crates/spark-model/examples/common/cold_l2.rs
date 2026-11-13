// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: DRAM-true microbenchmark plumbing for the L07 examples.
//! Rotating distinct weight copies, an L2 flush, rotated and interleaved arm
//! order, the median and spread statistics, and the tiny argument helpers the
//! examples share.
//!
//! Why this exists. A GEMV microtest that reads one weight 20 times measures
//! the 32 MB L2, not the 288 GB/s GDDR6: the guide
//! (`docs/porting/consumer-nvidia-ada.md`, FP8 microtest section) records
//! 927 GB/s for the 4B down projection, 3.2x the card's DRAM limit. The fix is
//! the method of TensorFold (`QWEN27_DISTINCT`, "a matrix called again sits in
//! the system cache and misleads") and the vLLM kernel-microbenchmark skill
//! (rotating buffers larger than L2): cycle through enough distinct copies that
//! the set read between two uses of the same bytes exceeds twice the L2, and
//! refuse to report a number when it does not.
//!
//! Interleaving. Cinference and TensorFold both rotate the arm order across
//! rounds (five rounds, three warm-up calls, twenty timed calls per sample,
//! CUDA events). GEMV rows on this card moved by up to 45% between runs of the
//! same binary, so a fixed A-then-B order bakes thermal and clock drift into
//! the comparison. [`rotated_order`] rotates the order each round and
//! [`bench_arms`] reports the median over rounds plus the spread.
//!
//! An example that includes this file also declares
//! `#[path = "common/profiler.rs"] mod profiler;` (this file calls
//! `crate::profiler`).
#![allow(dead_code)]

use crate::profiler::{GpuTimer, ProfilerRange};
use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::cell::Cell;

/// A ring is cold when the bytes read between two uses of the same copy exceed
/// this many times the L2 size.
pub const COLD_FACTOR: usize = 2;
/// Used when the driver does not report an L2 size (the 4060 Ti has 32 MiB).
pub const FALLBACK_L2_BYTES: usize = 32 << 20;
/// Rotate at least this many copies when memory allows (TensorFold, vLLM).
pub const MIN_RING_COPIES: usize = 3;
/// Default cap on one ring's total bytes: 1.5 GiB.
pub const DEFAULT_RING_CAP: usize = 1536 << 20;

/// How many distinct copies of one launch's weight bytes to rotate through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingPlan {
    pub copies: usize,
    pub per_copy_bytes: usize,
    pub l2_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingError {
    /// A copy of zero bytes.
    EmptyCopy,
    /// Even the minimum cold ring does not fit under the cap.
    TooBig { need_total: usize, cap: usize },
}

impl std::fmt::Display for RingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCopy => write!(f, "a ring copy of zero bytes"),
            Self::TooBig { need_total, cap } => write!(
                f,
                "a cold ring needs {} MiB but the cap is {} MiB",
                need_total >> 20,
                cap >> 20
            ),
        }
    }
}

impl std::error::Error for RingError {}

impl RingPlan {
    pub fn total_bytes(&self) -> usize {
        self.copies * self.per_copy_bytes
    }

    /// True when the ring holds more than [`COLD_FACTOR`] times the L2.
    pub fn is_cold(&self) -> bool {
        self.total_bytes() > COLD_FACTOR * self.l2_bytes
    }
}

/// Copies needed so that `copies * per_copy > COLD_FACTOR * l2`.
pub fn min_cold_copies(per_copy_bytes: usize, l2_bytes: usize) -> usize {
    (COLD_FACTOR * l2_bytes) / per_copy_bytes.max(1) + 1
}

/// Plan a cold ring: at least [`MIN_RING_COPIES`] copies when the cap allows,
/// otherwise the smallest ring that is still cold, otherwise an error.
pub fn plan_ring(
    per_copy_bytes: usize,
    l2_bytes: usize,
    cap_total_bytes: usize,
) -> Result<RingPlan, RingError> {
    if per_copy_bytes == 0 {
        return Err(RingError::EmptyCopy);
    }
    let need = min_cold_copies(per_copy_bytes, l2_bytes);
    let want = need.max(MIN_RING_COPIES);
    let copies = if want * per_copy_bytes <= cap_total_bytes {
        want
    } else if need * per_copy_bytes <= cap_total_bytes {
        need
    } else {
        return Err(RingError::TooBig {
            need_total: need * per_copy_bytes,
            cap: cap_total_bytes,
        });
    };
    Ok(RingPlan {
        copies,
        per_copy_bytes,
        l2_bytes,
    })
}

/// The deliberately L2-resident plan: one copy, used for the HOT contrast rows.
pub fn plan_hot(per_copy_bytes: usize, l2_bytes: usize) -> RingPlan {
    RingPlan {
        copies: 1,
        per_copy_bytes,
        l2_bytes,
    }
}

/// Refuse to report when the plan is not cold.
pub fn guard_cold(plan: &RingPlan) -> Result<()> {
    if plan.is_cold() {
        return Ok(());
    }
    bail!(
        "cold-L2 refused: the ring holds {} MiB, not more than {}x the {} MiB L2, \
         so the timing would be served from cache",
        plan.total_bytes() >> 20,
        COLD_FACTOR,
        plan.l2_bytes >> 20
    )
}

/// The label every printed row carries. A hot plan is allowed only when the
/// caller says so, and is then marked as not a bandwidth number.
pub fn temperature_label(plan: &RingPlan, allow_hot: bool) -> Result<&'static str> {
    if plan.is_cold() {
        Ok("cold")
    } else if allow_hot {
        Ok("HOT(L2-resident, not a bandwidth number)")
    } else {
        guard_cold(plan).map(|()| "cold")
    }
}

/// A fixed set of items handed out round-robin. Interior mutability lets many
/// arm closures share one ring by reference.
pub struct Ring<T: Copy> {
    items: Vec<T>,
    next: Cell<usize>,
}

impl<T: Copy> Ring<T> {
    pub fn new(items: Vec<T>) -> Self {
        assert!(!items.is_empty(), "a ring needs at least one item");
        Self {
            items,
            next: Cell::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn all(&self) -> &[T] {
        &self.items
    }

    /// The next item, wrapping.
    pub fn pick(&self) -> T {
        let i = self.next.get();
        self.next.set((i + 1) % self.items.len());
        self.items[i]
    }
}

/// Build a ring by calling `make(copy_index)` once per planned copy.
pub fn build_ring<T: Copy>(
    plan: &RingPlan,
    mut make: impl FnMut(usize) -> Result<T>,
) -> Result<Ring<T>> {
    let items = (0..plan.copies)
        .map(&mut make)
        .collect::<Result<Vec<T>>>()?;
    Ok(Ring::new(items))
}

/// Writes four times the L2 size so the next launch starts with an empty cache.
pub struct L2Flusher {
    buf: DevicePtr,
    bytes: usize,
}

impl L2Flusher {
    pub fn new(gpu: &dyn GpuBackend, l2_bytes: usize) -> Result<Self> {
        let bytes = 4 * l2_bytes;
        Ok(Self {
            buf: gpu.alloc(bytes)?,
            bytes,
        })
    }

    pub fn flush(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        gpu.memset_async(self.buf, 0xA5, self.bytes, stream)
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.buf)
    }
}

/// Median (mean of the two middle values for an even count); NaN when empty.
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// `(max - min) / median` in percent.
pub fn spread_pct(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for x in v {
        lo = lo.min(*x);
        hi = hi.max(*x);
    }
    (hi - lo) / median(v) * 100.0
}

/// GB/s for `bytes` moved in `us` microseconds.
pub fn gbs(bytes: f64, us: f64) -> f64 {
    bytes / (us * 1e-6) / 1e9
}

/// Arm order for one round: the order rotated left by `round`.
pub fn rotated_order(n_arms: usize, round: usize) -> Vec<usize> {
    (0..n_arms).map(|i| (round + i) % n_arms).collect()
}

/// One thing being compared: a name and a closure that enqueues one call.
pub struct Arm<'a> {
    pub name: String,
    run: Box<dyn FnMut() -> Result<()> + 'a>,
}

impl<'a> Arm<'a> {
    pub fn new(name: impl Into<String>, f: impl FnMut() -> Result<()> + 'a) -> Self {
        Self {
            name: name.into(),
            run: Box::new(f),
        }
    }

    pub fn call(&mut self) -> Result<()> {
        (self.run)()
    }
}

/// Rounds, warm-ups and timed calls per sample.
#[derive(Clone, Copy, Debug)]
pub struct BenchCfg {
    pub rounds: usize,
    pub warmups: usize,
    pub calls: usize,
    pub stream: u64,
    /// Wrap only the timed calls of each sample in a profiler range.
    pub profile: bool,
}

impl Default for BenchCfg {
    fn default() -> Self {
        Self {
            rounds: 5,
            warmups: 3,
            calls: 20,
            stream: 0,
            profile: false,
        }
    }
}

/// Per-arm result: one sample per round, in microseconds per call.
#[derive(Clone, Debug)]
pub struct ArmResult {
    pub name: String,
    pub samples_us: Vec<f64>,
    pub median_us: f64,
    pub spread_pct: f64,
}

/// Time every arm `cfg.rounds` times, rotating the order each round. Each
/// sample is `cfg.warmups` untimed calls, a stream sync, then `cfg.calls` calls
/// between two CUDA events.
pub fn bench_arms(
    gpu: &dyn GpuBackend,
    arms: &mut [Arm<'_>],
    cfg: &BenchCfg,
) -> Result<Vec<ArmResult>> {
    if arms.is_empty() {
        return Ok(Vec::new());
    }
    let timer = GpuTimer::new()?;
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(cfg.rounds); arms.len()];
    for round in 0..cfg.rounds {
        for idx in rotated_order(arms.len(), round) {
            let arm = &mut arms[idx];
            for _ in 0..cfg.warmups {
                arm.call()?;
            }
            gpu.synchronize(cfg.stream)?;
            let range = ProfilerRange::begin(cfg.profile)?;
            timer.start(cfg.stream)?;
            for _ in 0..cfg.calls {
                arm.call()?;
            }
            let ms = timer.stop(cfg.stream)?;
            drop(range);
            samples[idx].push(ms * 1e3 / cfg.calls as f64);
        }
    }
    Ok(arms
        .iter()
        .zip(samples)
        .map(|(a, s)| ArmResult {
            name: a.name.clone(),
            median_us: median(&s),
            spread_pct: spread_pct(&s),
            samples_us: s,
        })
        .collect())
}

/// One profiled call with a cold cache: warm once, flush the L2, then a single
/// call inside the profiler range. For `ncu`/`nsys` single-kernel captures.
pub fn profile_single_call(
    gpu: &dyn GpuBackend,
    flusher: &L2Flusher,
    arm: &mut Arm<'_>,
    stream: u64,
) -> Result<()> {
    arm.call()?;
    gpu.synchronize(stream)?;
    flusher.flush(gpu, stream)?;
    let range = ProfilerRange::begin(true)?;
    arm.call()?;
    gpu.synchronize(stream)?;
    drop(range);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const L2: usize = 32 << 20;

    #[test]
    fn small_weight_gets_many_copies() {
        // 4B down projection in FP8 with scales is about 23.7 MB.
        let p = plan_ring(23_700_000, L2, DEFAULT_RING_CAP).unwrap();
        assert!(p.copies >= MIN_RING_COPIES);
        assert!(p.is_cold());
        assert!(p.total_bytes() > 2 * L2);
    }

    #[test]
    fn tiny_weight_needs_more_than_the_minimum() {
        let p = plan_ring(3_700_000, L2, DEFAULT_RING_CAP).unwrap();
        assert!(p.copies > MIN_RING_COPIES);
        assert!(p.is_cold());
    }

    #[test]
    fn large_weight_falls_back_to_the_smallest_cold_ring() {
        // 1.27 GB lm_head: one copy is already more than twice the L2.
        let p = plan_ring(1_272_000_000, L2, 2 << 30).unwrap();
        assert_eq!(p.copies, 1);
        assert!(p.is_cold());
    }

    #[test]
    fn cap_too_small_is_an_error_not_a_hot_ring() {
        let e = plan_ring(23_700_000, L2, 40 << 20).unwrap_err();
        assert!(matches!(e, RingError::TooBig { .. }), "{e}");
        assert_eq!(plan_ring(0, L2, 1 << 30).unwrap_err(), RingError::EmptyCopy);
    }

    #[test]
    fn boundary_is_strictly_more_than_twice_l2() {
        // Exactly 2x is not cold; one more copy is.
        let half = L2;
        let p = RingPlan {
            copies: 2,
            per_copy_bytes: half,
            l2_bytes: L2,
        };
        assert!(!p.is_cold());
        assert_eq!(min_cold_copies(half, L2), 3);
    }

    #[test]
    fn hot_plan_is_refused_unless_allowed() {
        let hot = plan_hot(8 << 20, L2);
        assert!(guard_cold(&hot).is_err());
        assert!(temperature_label(&hot, false).is_err());
        assert!(temperature_label(&hot, true).unwrap().starts_with("HOT"));
        let cold = plan_ring(23_700_000, L2, DEFAULT_RING_CAP).unwrap();
        assert_eq!(temperature_label(&cold, false).unwrap(), "cold");
    }

    #[test]
    fn ring_hands_out_round_robin() {
        let r = Ring::new(vec![10u32, 20, 30]);
        let got: Vec<u32> = (0..7).map(|_| r.pick()).collect();
        assert_eq!(got, vec![10, 20, 30, 10, 20, 30, 10]);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn rotation_gives_every_arm_every_position() {
        let n = 4;
        let mut seen = vec![vec![0usize; n]; n];
        for round in 0..n {
            for (pos, arm) in rotated_order(n, round).into_iter().enumerate() {
                seen[arm][pos] += 1;
            }
        }
        assert!(seen.iter().flatten().all(|c| *c == 1));
        assert_eq!(rotated_order(2, 0), vec![0, 1]);
        assert_eq!(rotated_order(2, 1), vec![1, 0]);
        assert!(rotated_order(0, 3).is_empty());
    }

    #[test]
    fn median_and_spread() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&[]).is_nan());
        let s = spread_pct(&[9.0, 10.0, 11.0]);
        assert!((s - 20.0).abs() < 1e-9, "{s}");
    }

    #[test]
    fn gbs_is_bytes_over_time() {
        // 288 MB in 1 ms is 288 GB/s.
        assert!((gbs(288e6, 1000.0) - 288.0).abs() < 1e-9);
    }

    #[test]
    fn default_cfg_matches_the_method() {
        let c = BenchCfg::default();
        assert_eq!((c.rounds, c.warmups, c.calls), (5, 3, 20));
    }
}
