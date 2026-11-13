// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: kv_view_microtest gates C1 to C3 and the scorer timing: random sketches
//! scattered over a page pool, scored on the card by `kv_select::score_blocks`, compared
//! with the f64 oracle in both GQA modes, run twice for determinism, a planted needle
//! picked, and the 1M-token / 512-row timing on the Qwen3.5-4B shape.

use std::time::Instant;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::sketch::{DEFAULT_MAX_JOBS, SketchBuffers, SketchConfig};
use spark_runtime::kv_select::capture::CapturedRows;
use spark_runtime::kv_select::oracle::{QueryF64, SketchF64, score_blocks_f64};
use spark_runtime::kv_select::{
    DEFAULT_PAGES_PER_BLOCK, DEFAULT_SCRATCH_BYTES, GqaAgg, PickCfg, ScoreScratch, n_blocks, pick,
    recall_mass, score_blocks,
};

use crate::{Args, Gates, Rng, as_f32, bf16_bits, bf16_f64, download, u16_bytes, upload};

const PPB: usize = DEFAULT_PAGES_PER_BLOCK;

/// A random sequence over a page pool: `(layers, q_heads, kv_heads, head_dim)`.
struct Fixture {
    dims: (usize, usize, usize, usize),
    rows: usize,
    num_blocks: usize,
    means: Vec<u16>,
    counts: Vec<u8>,
    pages: Vec<u32>,
    query: Vec<u16>,
}

impl Fixture {
    /// `fast` draws the means uniformly (one draw each) for the 1 GB timing sketch.
    fn new(
        rng: &mut Rng,
        n_pages: usize,
        dims: (usize, usize, usize, usize),
        rows: usize,
        fast: bool,
    ) -> Self {
        let (l, qh, kvh, d) = dims;
        let num_blocks = n_pages + 7;
        let mut pool: Vec<u32> = (0..num_blocks as u32).collect();
        for i in (1..pool.len()).rev() {
            pool.swap(i, rng.below(i + 1));
        }
        let pages = pool[..n_pages].to_vec();
        let means = (0..num_blocks * l * kvh * d)
            .map(|_| {
                bf16_bits(if fast {
                    rng.unit() - 0.5
                } else {
                    0.5 * rng.normal()
                })
            })
            .collect();
        let mut counts = vec![0u8; num_blocks];
        for (i, &p) in pages.iter().enumerate() {
            counts[p as usize] = match (i + 1 == n_pages, rng.below(37)) {
                (true, _) => 5,
                (false, 0) => 0,
                _ => 1 + rng.below(16) as u8,
            };
        }
        let query = (0..l * rows * qh * d)
            .map(|_| bf16_bits(rng.normal()))
            .collect();
        Self {
            dims,
            rows,
            num_blocks,
            means,
            counts,
            pages,
            query,
        }
    }

    fn oracle(&self) -> (SketchF64, QueryF64) {
        let (l, qh, kvh, d) = self.dims;
        let per = l * kvh * d;
        let means = self
            .pages
            .iter()
            .flat_map(|&p| {
                self.means[p as usize * per..(p as usize + 1) * per]
                    .iter()
                    .map(|&b| bf16_f64(b))
            })
            .collect();
        let counts = self
            .pages
            .iter()
            .map(|&p| u32::from(self.counts[p as usize]))
            .collect();
        let s = SketchF64 {
            n_pages: self.pages.len(),
            layers: l,
            kv_heads: kvh,
            head_dim: d,
            means,
            counts,
        };
        let data = self.query.iter().map(|&b| bf16_f64(b)).collect();
        (
            s,
            QueryF64 {
                rows: self.rows,
                layers: l,
                q_heads: qh,
                head_dim: d,
                data,
            },
        )
    }

    /// Point view block `b`'s pages at the mean direction of the query rows, scaled up.
    fn plant(&mut self, b: usize) {
        let (l, qh, kvh, d) = self.dims;
        let group = qh / kvh;
        for p in b * PPB..(b + 1) * PPB {
            let page = self.pages[p] as usize;
            self.counts[page] = 16;
            for layer in 0..l {
                for g in 0..kvh {
                    let mut dir = vec![0.0f64; d];
                    for r in 0..self.rows {
                        for h in g * group..(g + 1) * group {
                            let at = ((layer * self.rows + r) * qh + h) * d;
                            for (o, &x) in dir.iter_mut().zip(&self.query[at..at + d]) {
                                *o += bf16_f64(x);
                            }
                        }
                    }
                    let norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let at = ((page * l + layer) * kvh + g) * d;
                    for (e, v) in dir.iter().enumerate() {
                        self.means[at + e] = bf16_bits(8.0 * v / norm);
                    }
                }
            }
        }
    }
}

/// A fixture resident on the card.
struct Dev {
    sk: SketchBuffers,
    q: DevicePtr,
    out: DevicePtr,
    scratch: ScoreScratch,
    rows: CapturedRows,
    nb: usize,
}

impl Dev {
    fn new(g: &dyn GpuBackend, fx: &Fixture) -> Result<Self> {
        let (l, qh, kvh, d) = fx.dims;
        let cfg = SketchConfig {
            num_blocks: fx.num_blocks,
            attn_layers: l,
            kv_heads: kvh,
            head_dim: d,
            block_size: 16,
            rotary_dim: 0,
            rope_theta: 1.0e7,
            max_jobs: DEFAULT_MAX_JOBS,
        };
        let sk = SketchBuffers::new(g, cfg)?;
        g.copy_h2d(&u16_bytes(&fx.means), sk.means_ptr())?;
        g.copy_h2d(&fx.counts, sk.counts_ptr())?;
        let q = upload(g, &u16_bytes(&fx.query))?;
        let nb = n_blocks(fx.pages.len(), PPB);
        let out = g.alloc(nb * 4)?;
        let scratch = ScoreScratch::new(g, DEFAULT_SCRATCH_BYTES)?;
        let rows = CapturedRows {
            data: q,
            rows: fx.rows,
            rows_cap: fx.rows,
            layers: l,
            q_heads: qh,
            head_dim: d,
        };
        Ok(Self {
            sk,
            q,
            out,
            scratch,
            rows,
            nb,
        })
    }

    fn score(&self, g: &dyn GpuBackend, pages: &[u32], gqa: GqaAgg) -> Result<Vec<f32>> {
        let view = self.sk.view(pages);
        score_blocks(
            g,
            &view,
            &self.rows,
            &self.scratch.cfg(PPB, gqa),
            self.out,
            g.default_stream(),
        )?;
        Ok(as_f32(&download(g, self.out, self.nb * 4)?))
    }

    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        g.synchronize(g.default_stream())?;
        self.sk.free(g)?;
        self.scratch.free(g)?;
        g.free(self.q)?;
        g.free(self.out)
    }
}

/// (L1 relative error, worst per-block relative error over blocks above 1e-3 of the mean).
fn rel_err(got: &[f32], want: &[f64]) -> (f64, f64) {
    let l1: f64 = got
        .iter()
        .zip(want)
        .map(|(&a, &w)| (f64::from(a) - w).abs())
        .sum();
    let total: f64 = want.iter().map(|w| w.abs()).sum();
    let floor = 1e-3 * total / want.len().max(1) as f64;
    let worst = got
        .iter()
        .zip(want)
        .filter(|&(_, &w)| w.abs() > floor)
        .map(|(&a, &w)| (f64::from(a) - w).abs() / w.abs())
        .fold(0.0, f64::max);
    (l1 / total.max(f64::MIN_POSITIVE), worst)
}

fn oracle_case(
    g: &dyn GpuBackend,
    rng: &mut Rng,
    nb: usize,
    dims: (usize, usize, usize, usize),
    rows: usize,
    gates: &mut Gates,
) -> Result<()> {
    let fx = Fixture::new(rng, nb * PPB, dims, rows, false);
    let dev = Dev::new(g, &fx)?;
    let (os, oq) = fx.oracle();
    for gqa in [GqaAgg::Mean, GqaAgg::Max] {
        let got = dev.score(g, &fx.pages, gqa)?;
        let want = score_blocks_f64(&os, &oq, PPB, gqa);
        let (l1, worst) = rel_err(&got, &want);
        gates.check(
            &format!("C1 scorer {nb} blocks {}", gqa.name()),
            l1 <= 1e-5 && worst <= 1e-4,
            &format!(
                "L1 relative {l1:.2e}, worst per-block {worst:.2e}, dims {dims:?}, rows {rows}"
            ),
        );
    }
    dev.free(g)
}

fn needle_and_determinism(g: &dyn GpuBackend, rng: &mut Rng, gates: &mut Gates) -> Result<()> {
    let mut fx = Fixture::new(rng, 100 * PPB, (8, 16, 4, 256), 16, false);
    fx.plant(37);
    let dev = Dev::new(g, &fx)?;
    for gqa in [GqaAgg::Mean, GqaAgg::Max] {
        let a = dev.score(g, &fx.pages, gqa)?;
        let b = dev.score(g, &fx.pages, gqa)?;
        let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
        gates.check(
            &format!("C2 deterministic {}", gqa.name()),
            same,
            "two runs, equal bits",
        );
        let cfg = PickCfg {
            budget_blocks: 3,
            sink_blocks: 1,
            recent_blocks: 1,
            mandatory: Vec::new(),
        };
        let picked = pick(&a, a.len(), &cfg);
        gates.check(
            &format!("C3 planted needle {}", gqa.name()),
            picked.contains(&37),
            &format!(
                "picked {picked:?}, recall mass {:.3}",
                recall_mass(&a, &picked)
            ),
        );
    }
    dev.free(g)
}

/// info: 1M tokens (62,500 pages, 7,813 view blocks), 512 rows, 8 layers x 16/4 heads x 256.
fn timing(g: &dyn GpuBackend, rng: &mut Rng, reps: usize) -> Result<()> {
    let fx = Fixture::new(rng, 62_500, (8, 16, 4, 256), 512, true);
    let dev = Dev::new(g, &fx)?;
    let mut ms = Vec::with_capacity(reps);
    for _ in 0..reps + 1 {
        g.synchronize(g.default_stream())?;
        let t = Instant::now();
        let view = dev.sk.view(&fx.pages);
        score_blocks(
            g,
            &view,
            &dev.rows,
            &dev.scratch.cfg(PPB, GqaAgg::Mean),
            dev.out,
            g.default_stream(),
        )?;
        g.synchronize(g.default_stream())?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ms.remove(0);
    ms.sort_by(f64::total_cmp);
    println!(
        "info scorer 1M tokens 512 rows: median {:.2} ms, min {:.2} ms over {reps} runs (one warm-up dropped)",
        ms[ms.len() / 2],
        ms[0]
    );
    dev.free(g)
}

pub fn run(g: &dyn GpuBackend, a: &Args, gates: &mut Gates) -> Result<()> {
    let mut rng = Rng(0x5eed_0131);
    oracle_case(g, &mut rng, 100, (8, 16, 4, 256), 64, gates)?;
    needle_and_determinism(g, &mut rng, gates)?;
    if !a.quick {
        oracle_case(g, &mut rng, 7_813, (2, 16, 4, 256), 8, gates)?;
        oracle_case(g, &mut rng, 40_000, (1, 4, 2, 64), 16, gates)?;
    }
    timing(g, &mut rng, if a.quick { 1 } else { a.reps })
}
