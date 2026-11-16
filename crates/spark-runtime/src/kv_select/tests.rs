// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the scorer's CPU checks: the f64 oracle's invariants (rows sum to one,
//! planted needle), the f32 device model against the oracle at 100, 7,813 and 40,000 view
//! blocks (the last with chunked merges and row tiles), the tiling plan's scratch bound,
//! and the launcher on the mock backend (no allocation, the planned launch count, a clean
//! refusal where the module is absent). Shared fixtures for the sibling test modules.

use super::emulate::{HostQuery, HostSketch, score_blocks_f32};
use super::oracle::{QueryF64, SketchF64, score_blocks_f64};
use super::*;
use crate::gpu::mock::MockGpuBackend;

/// Deterministic xorshift64* generator.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [0, 1).
    pub(super) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Roughly normal (sum of four uniforms, unit variance).
    pub(super) fn normal(&mut self) -> f64 {
        ((0..4).map(|_| self.unit()).sum::<f64>() - 2.0) * 3f64.sqrt()
    }
    pub(super) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

fn bf16_bits(x: f64) -> u16 {
    half::bf16::from_f64(x).to_bits()
}

fn bf(b: u16) -> f64 {
    half::bf16::from_bits(b).to_f64()
}

/// A random sequence: `n_pages` pages scattered over a pool twice as large, counts 1..=16
/// with a few unsketched pages and a partial tail, and `rows` query rows.
pub(super) struct Fixture {
    pub num_blocks: usize,
    pub layers: usize,
    pub kv_heads: usize,
    pub q_heads: usize,
    pub head_dim: usize,
    pub rows: usize,
    pub means: Vec<u16>,
    pub counts: Vec<u8>,
    pub pages: Vec<u32>,
    pub query: Vec<u16>,
}

impl Fixture {
    pub(super) fn new(
        seed: u64,
        n_pages: usize,
        dims: (usize, usize, usize, usize),
        rows: usize,
    ) -> Self {
        let (layers, q_heads, kv_heads, head_dim) = dims;
        let mut rng = Rng::new(seed);
        let num_blocks = 2 * n_pages + 3;
        let mut pool: Vec<u32> = (0..num_blocks as u32).collect();
        for i in (1..pool.len()).rev() {
            pool.swap(i, rng.below(i + 1));
        }
        let pages = pool[..n_pages].to_vec();
        let per = layers * kv_heads * head_dim;
        let means = (0..num_blocks * per)
            .map(|_| bf16_bits(0.5 * rng.normal()))
            .collect();
        let mut counts = vec![0u8; num_blocks];
        for (i, &p) in pages.iter().enumerate() {
            counts[p as usize] = if i + 1 == n_pages {
                5
            } else if rng.below(37) == 0 {
                0
            } else {
                1 + rng.below(16) as u8
            };
        }
        let query = (0..layers * rows * q_heads * head_dim)
            .map(|_| bf16_bits(rng.normal()))
            .collect();
        Self {
            num_blocks,
            layers,
            kv_heads,
            q_heads,
            head_dim,
            rows,
            means,
            counts,
            pages,
            query,
        }
    }

    pub(super) fn host(&self) -> (HostSketch<'_>, HostQuery<'_>) {
        (
            HostSketch {
                num_blocks: self.num_blocks,
                layers: self.layers,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
                means: &self.means,
                counts: &self.counts,
                pages: &self.pages,
            },
            HostQuery {
                rows: self.rows,
                rows_cap: self.rows,
                layers: self.layers,
                q_heads: self.q_heads,
                head_dim: self.head_dim,
                data: &self.query,
            },
        )
    }

    /// The same values in f64, pages in sequence order.
    pub(super) fn f64(&self) -> (SketchF64, QueryF64) {
        let per = self.layers * self.kv_heads * self.head_dim;
        let means = self
            .pages
            .iter()
            .flat_map(|&p| {
                self.means[p as usize * per..(p as usize + 1) * per]
                    .iter()
                    .map(|&b| bf(b))
            })
            .collect();
        (
            SketchF64 {
                n_pages: self.pages.len(),
                layers: self.layers,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
                means,
                counts: self
                    .pages
                    .iter()
                    .map(|&p| u32::from(self.counts[p as usize]))
                    .collect(),
            },
            QueryF64 {
                rows: self.rows,
                layers: self.layers,
                q_heads: self.q_heads,
                head_dim: self.head_dim,
                data: self.query.iter().map(|&b| bf(b)).collect(),
            },
        )
    }
}

/// (L1 relative error, worst per-block relative error over blocks above 1e-3 of the mean).
pub(super) fn rel_err(got: &[f32], want: &[f64]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let l1: f64 = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - w).abs())
        .sum();
    let total: f64 = want.iter().map(|w| w.abs()).sum();
    let floor = 1e-3 * total / want.len().max(1) as f64;
    let worst = got
        .iter()
        .zip(want)
        .filter(|&(_, &w)| w.abs() > floor)
        .map(|(&g, &w)| (f64::from(g) - w).abs() / w.abs())
        .fold(0.0, f64::max);
    (l1 / total.max(f64::MIN_POSITIVE), worst)
}

fn check_against_oracle(fx: &Fixture, ppb: usize, scratch: usize) {
    let (hs, hq) = fx.host();
    let (os, oq) = fx.f64();
    for gqa in [GqaAgg::Mean, GqaAgg::Max] {
        let want = score_blocks_f64(&os, &oq, ppb, gqa);
        let got = score_blocks_f32(&hs, &hq, ppb, gqa, scratch).expect("emulate");
        let (l1, worst) = rel_err(&got, &want);
        assert!(l1 <= 1e-5, "{gqa:?} L1 relative {l1:e}");
        assert!(worst <= 1e-4, "{gqa:?} worst per-block relative {worst:e}");
        let sum: f64 = want.iter().sum();
        assert!(
            (sum - fx.rows as f64).abs() < 1e-9 * fx.rows as f64,
            "{gqa:?} rows sum {sum}"
        );
    }
}

#[test]
fn f32_model_matches_the_oracle_at_100_blocks() {
    let fx = Fixture::new(1, 800, (2, 8, 2, 64), 5);
    check_against_oracle(&fx, DEFAULT_PAGES_PER_BLOCK, DEFAULT_SCRATCH_BYTES);
}

#[test]
fn f32_model_matches_the_oracle_at_7813_blocks() {
    let fx = Fixture::new(2, 7813 * 8, (1, 4, 2, 32), 2);
    check_against_oracle(&fx, DEFAULT_PAGES_PER_BLOCK, DEFAULT_SCRATCH_BYTES);
}

#[test]
fn chunked_merges_and_row_tiles_match_the_oracle_at_40000_blocks() {
    let fx = Fixture::new(3, 40_000, (1, 2, 1, 16), 3);
    let dims = ScoreDims {
        n_pages: 40_000,
        pages_per_block: 1,
        rows: 3,
        layers: 1,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 16,
    };
    let plan = ScorePlan::new(dims, 1 << 20).expect("plan");
    assert!(plan.n_chunks() > 1 && plan.n_row_tiles() > 1, "{plan:?}");
    check_against_oracle(&fx, 1, 1 << 20);
}

#[test]
fn scores_of_every_row_sum_to_one() {
    let fx = Fixture::new(4, 77, (3, 6, 3, 16), 7);
    let (os, oq) = fx.f64();
    for gqa in [GqaAgg::Mean, GqaAgg::Max] {
        let s = score_blocks_f64(&os, &oq, 4, gqa);
        let total: f64 = s.iter().sum();
        assert!((total - 7.0).abs() < 1e-12, "{gqa:?}: {total}");
        assert!(s.iter().all(|v| *v >= 0.0));
    }
}

#[test]
fn a_planted_needle_is_picked_at_the_smallest_budget() {
    let mut fx = Fixture::new(5, 64 * 8, (2, 4, 2, 32), 4);
    // Every page of block 21 carries the direction of the query rows' mean, scaled up.
    let (layers, kvh, hd, qh) = (fx.layers, fx.kv_heads, fx.head_dim, fx.q_heads);
    let group = qh / kvh;
    for p in 21 * 8..22 * 8 {
        let page = fx.pages[p] as usize;
        fx.counts[page] = 16;
        for l in 0..layers {
            for g in 0..kvh {
                let mut dir = vec![0.0f64; hd];
                for r in 0..fx.rows {
                    for h in g * group..(g + 1) * group {
                        let at = ((l * fx.rows + r) * qh + h) * hd;
                        for (o, &b) in dir.iter_mut().zip(&fx.query[at..at + hd]) {
                            *o += bf(b);
                        }
                    }
                }
                let norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
                let at = ((page * layers + l) * kvh + g) * hd;
                for (e, v) in dir.iter().enumerate() {
                    fx.means[at + e] = bf16_bits(4.0 * v / norm);
                }
            }
        }
    }
    let (hs, hq) = fx.host();
    for gqa in [GqaAgg::Mean, GqaAgg::Max] {
        let scores = score_blocks_f32(&hs, &hq, 8, gqa, DEFAULT_SCRATCH_BYTES).expect("emulate");
        let best = (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        assert_eq!(best, 21, "{gqa:?}");
        for (sink, recent) in [(0, 0), (1, 1), (2, 3)] {
            let cfg = PickCfg {
                budget_blocks: sink + recent.max(1) + 1,
                sink_blocks: sink,
                recent_blocks: recent,
                mandatory: Vec::new(),
            };
            let picked = pick(&scores, scores.len(), &cfg);
            assert!(
                picked.contains(&21),
                "{gqa:?} sink {sink} recent {recent}: {picked:?}"
            );
            assert!(recall_mass(&scores, &picked) > 0.0);
        }
    }
}

#[test]
fn the_default_plan_fits_one_million_tokens_and_512_rows_in_64_mib() {
    let dims = ScoreDims {
        n_pages: 62_500,
        pages_per_block: 8,
        rows: 512,
        layers: 8,
        q_heads: 16,
        kv_heads: 4,
        head_dim: 256,
    };
    let plan = ScorePlan::new(dims, DEFAULT_SCRATCH_BYTES).expect("plan");
    assert_eq!(plan.n_blocks(), 7_813);
    assert_eq!(plan.n_chunks(), 1, "one merge per layer at 1M tokens");
    assert!(plan.rows_tile >= 64, "{plan:?}");
    assert!(plan.layout.total <= DEFAULT_SCRATCH_BYTES);
    assert_eq!(plan.cta_rows * dims.group(), 32);
}

#[test]
fn no_plan_exceeds_its_scratch() {
    let mut rng = Rng::new(9);
    for _ in 0..400 {
        let kv = 1 + rng.below(4);
        let dims = ScoreDims {
            n_pages: 1 + rng.below(200_000),
            pages_per_block: 1 + rng.below(16),
            rows: 1 + rng.below(600),
            layers: 1 + rng.below(8),
            q_heads: kv * (1 + rng.below(8)),
            kv_heads: kv,
            head_dim: 16 * (1 + rng.below(16)),
        };
        let scratch = 4096 + rng.below(80 << 20);
        if let Ok(plan) = ScorePlan::new(dims, scratch) {
            assert!(plan.layout.total <= scratch, "{plan:?} > {scratch}");
            assert!(plan.rows_tile >= 1 && plan.chunk_blocks >= 1);
            let (s, n) = plan.chunk(plan.n_chunks() - 1);
            assert_eq!(s + n, plan.n_blocks());
            let (r, m) = plan.row_tile(plan.n_row_tiles() - 1);
            assert_eq!(r + m, dims.rows);
        }
    }
    let dims = ScoreDims {
        n_pages: 1 << 20,
        pages_per_block: 1,
        rows: 4,
        layers: 1,
        q_heads: 16,
        kv_heads: 4,
        head_dim: 256,
    };
    assert!(
        ScorePlan::new(dims, 1 << 20).is_err(),
        "one row of 1M logits cannot fit 1 MiB"
    );
}

/// A sketch or query view over plain device pointers, for the mock tests.
pub(super) struct Views {
    pub pages: Vec<u32>,
    pub num_blocks: usize,
    pub dims: (usize, usize, usize, usize),
    pub rows: usize,
    pub means: DevicePtr,
    pub counts: DevicePtr,
    pub q: DevicePtr,
}

impl SketchView for Views {
    fn pages(&self) -> &[u32] {
        &self.pages
    }
    fn num_blocks(&self) -> usize {
        self.num_blocks
    }
    fn layers(&self) -> usize {
        self.dims.0
    }
    fn kv_heads(&self) -> usize {
        self.dims.2
    }
    fn head_dim(&self) -> usize {
        self.dims.3
    }
    fn page_tokens(&self) -> usize {
        16
    }
    fn means(&self) -> DevicePtr {
        self.means
    }
    fn counts(&self) -> DevicePtr {
        self.counts
    }
}

struct Query<'a>(&'a Views);

impl QueryView for Query<'_> {
    fn rows(&self) -> usize {
        self.0.rows
    }
    fn layers(&self) -> usize {
        self.0.dims.0
    }
    fn q_heads(&self) -> usize {
        self.0.dims.1
    }
    fn head_dim(&self) -> usize {
        self.0.dims.3
    }
    fn data(&self) -> DevicePtr {
        self.0.q
    }
}

fn mock_views(gpu: &MockGpuBackend, n_pages: usize, rows: usize) -> Views {
    let dims = (2, 8, 2, 64);
    Views {
        pages: (0..n_pages as u32).rev().collect(),
        num_blocks: n_pages,
        dims,
        rows,
        means: gpu.alloc(n_pages * dims.0 * dims.2 * dims.3 * 2).unwrap(),
        counts: gpu.alloc(n_pages).unwrap(),
        q: gpu.alloc(dims.0 * rows * dims.1 * dims.3 * 2).unwrap(),
    }
}

#[test]
fn the_launcher_issues_the_planned_launches_and_never_allocates() {
    let gpu = MockGpuBackend::new();
    let v = mock_views(&gpu, 1000, 9);
    let scratch = ScoreScratch::new(&gpu, 1 << 20).unwrap();
    let out = gpu.alloc(125 * 4).unwrap();
    for gqa in [GqaAgg::Mean, GqaAgg::Max] {
        let before = (gpu.live_alloc_count(), gpu.launches_snapshot().len());
        score_blocks(&gpu, &v, &Query(&v), &scratch.cfg(8, gqa), out, 0).unwrap();
        let plan =
            ScorePlan::new(super::score::dims_of(&v, &Query(&v), 8).unwrap(), 1 << 20).unwrap();
        assert_eq!(gpu.live_alloc_count(), before.0, "score_blocks allocated");
        assert_eq!(
            gpu.launches_snapshot().len() - before.1,
            plan.launches(gqa == GqaAgg::Max),
            "{gqa:?}"
        );
    }
    // The page list landed at the head of the scratch, little-endian i32.
    let head = gpu.read_alloc(scratch.ptr()).unwrap();
    assert_eq!(&head[..8], &[231, 3, 0, 0, 230, 3, 0, 0]);
    assert!(
        gpu.kernel_lookups_snapshot()
            .iter()
            .all(|(m, _)| m == KERNEL_MODULE)
    );
}

#[test]
fn an_image_without_the_module_is_refused_without_a_launch_or_lookup() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(KERNEL_MODULE);
    assert!(!kernels_available(&gpu));
    let v = mock_views(&gpu, 64, 2);
    let scratch = ScoreScratch::new(&gpu, 1 << 20).unwrap();
    let out = gpu.alloc(64).unwrap();
    let err =
        score_blocks(&gpu, &v, &Query(&v), &scratch.cfg(8, GqaAgg::Mean), out, 0).unwrap_err();
    assert!(err.to_string().contains(KERNEL_MODULE), "{err}");
    assert!(gpu.launches_snapshot().is_empty());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
}

#[test]
fn mismatched_views_are_refused() {
    let gpu = MockGpuBackend::new();
    let mut v = mock_views(&gpu, 64, 2);
    let scratch = ScoreScratch::new(&gpu, 1 << 20).unwrap();
    let out = gpu.alloc(64).unwrap();
    v.pages.push(64);
    assert!(score_blocks(&gpu, &v, &Query(&v), &scratch.cfg(8, GqaAgg::Mean), out, 0).is_err());
    v.pages.pop();
    v.dims.1 = 7;
    assert!(score_blocks(&gpu, &v, &Query(&v), &scratch.cfg(8, GqaAgg::Mean), out, 0).is_err());
    assert!(gpu.launches_snapshot().is_empty());
}

#[test]
fn gqa_names_round_trip() {
    for g in [GqaAgg::Mean, GqaAgg::Max] {
        assert_eq!(GqaAgg::parse(g.name()), Some(g));
    }
    assert_eq!(GqaAgg::parse(" MAX "), Some(GqaAgg::Max));
    assert_eq!(GqaAgg::parse("sum"), None);
    assert_eq!(n_blocks(17, 8), 3);
    assert_eq!(n_blocks(16, 8), 2);
    assert_eq!(n_blocks(5, 0), 0);
}
