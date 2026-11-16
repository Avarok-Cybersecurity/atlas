// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of decode-time zone selection: the total order and clamps of the
//! reference, the softmax and GQA semantics (planted clusters, empty clusters, group sums), the
//! agreement with L33's oracle (`oracle_eval`), the graph-safety of the launches on the mock
//! backend (fixed shapes and pointers), the refusals, the absent-module path, the scratch budget
//! against `RetroCfg`, and the kernel source constants.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::reference::{self, Input};
use super::{
    MAX_RG, MAX_SELECTED, TILE_C, scratch_bytes, scratch_layout, select_zones, select_zones_rows,
};
use crate::layers::qwen3_attention::sparse_seam::KvDtypeTag;
use crate::port::retro::build::reference::bf16;
use crate::port::retro::config::{RetroCfg, RetroParams};
use crate::port::retro::oracle::HeadIndex;
use crate::port::retro::oracle_eval;
use crate::port::retro::state::{IndexLayout, LayerIndex, ZoneLayout, ZoneLists};

const D: usize = 256;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uni(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

/// The total order, brute force: sort (key desc, id asc), drop zeros, split.
fn brute(keys: &[u32], kr: usize, ks: usize) -> (Vec<u32>, Vec<u32>) {
    let mut v: Vec<(u32, u32)> = keys
        .iter()
        .enumerate()
        .filter(|(_, k)| **k > 0)
        .map(|(i, &k)| (k, i as u32))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let ks = ks.min(v.len());
    let kr = kr.min(ks);
    (
        v[..kr].iter().map(|x| x.1).collect(),
        v[kr..ks].iter().map(|x| x.1).collect(),
    )
}

#[test]
fn pick_follows_the_total_order_with_ties_and_holes() {
    let mut r = Rng(3);
    for trial in 0..200 {
        let n = 1 + (r.next() % 600) as usize;
        // Few distinct keys, so ties are everywhere; a quarter of the clusters are empty (0).
        let keys: Vec<u32> = (0..n)
            .map(|_| {
                if r.next().is_multiple_of(4) {
                    0
                } else {
                    1 + (r.next() % 7) as u32
                }
            })
            .collect();
        let kr = (r.next() % 40) as u32;
        let ks = kr + (r.next() % 200) as u32;
        let got = reference::pick(&keys, kr, ks);
        assert_eq!(got, brute(&keys, kr as usize, ks as usize), "trial {trial}");
    }
}

#[test]
fn targets_clamp_like_the_kernel() {
    assert_eq!(reference::targets(147, 1901, 8192, 302, 1927), (147, 2048));
    assert_eq!(reference::targets(147, 1901, 100, 302, 1927), (100, 100));
    // A retrieval cap below nprobe hands the clipped clusters to the estimation zone, which its
    // own cap then bounds.
    assert_eq!(reference::targets(147, 1901, 8192, 100, 1927), (100, 2027));
    assert_eq!(reference::targets(147, 1901, 8192, 302, 50), (147, 197));
    assert_eq!(
        reference::targets(300, 5000, 20_000, 600, 6000),
        (300, MAX_SELECTED)
    );
    assert_eq!(reference::targets(0, 0, 0, 4, 4), (0, 0));
}

struct Data {
    cent: Vec<f32>,
    size: Vec<i32>,
    q: Vec<f32>,
}

/// Random BF16 centroids and queries; cluster 5 of head 0 is aligned with query head 0 and so is
/// cluster 9, which is empty; head 1's cluster 3 is aligned with only its last query head.
fn data(kv_heads: usize, group: usize, cap: usize, rows: usize, seed: u64) -> Data {
    let mut r = Rng(seed);
    let mut cent: Vec<f32> = (0..kv_heads * cap * D)
        .map(|_| bf16(0.3 * r.uni()))
        .collect();
    let mut size: Vec<i32> = (0..kv_heads * cap)
        .map(|_| 1 + (r.next() % 30) as i32)
        .collect();
    let q: Vec<f32> = (0..rows * kv_heads * group * D)
        .map(|_| bf16(r.uni()))
        .collect();
    for d in 0..D {
        cent[5 * D + d] = bf16(q[d] * 2.0);
        cent[9 * D + d] = bf16(q[d] * 2.0);
        if kv_heads > 1 {
            let qh = 2 * group - 1;
            cent[(cap + 3) * D + d] = bf16(q[qh * D + d] * 2.0);
        }
    }
    size[9] = 0;
    Data { cent, size, q }
}

fn input(
    dt: &Data,
    kv_heads: usize,
    group: usize,
    cap: usize,
    nc: usize,
    rows: usize,
) -> Input<'_> {
    let p = RetroParams::default();
    let nprobe = crate::port::retro::config::nprobe(nc as u32, p.retrieval_budget);
    Input {
        centroids: &dt.cent,
        cluster_size: &dt.size,
        cap,
        nc,
        kv_heads,
        group,
        rows,
        q: &dt.q,
        scale: 1.0 / 16.0,
        nprobe,
        es: crate::port::retro::config::estimation_clusters(nc as u32, p.estimation_budget, nprobe),
        r_cap: 2 * nprobe,
        e_cap: nc as u32,
    }
}

#[test]
fn planted_clusters_lead_and_empty_clusters_never_appear() {
    let dt = data(2, 4, 512, 1, 17);
    let inp = input(&dt, 2, 4, 512, 500, 1);
    let z = reference::select(&inp);
    assert_eq!(z[0].retrieval.first(), Some(&5));
    assert_eq!(
        z[1].retrieval.first(),
        Some(&3),
        "a cluster one query head of the group wants"
    );
    for zh in &z {
        let ranked: Vec<u32> = zh.retrieval.iter().chain(&zh.estimation).copied().collect();
        assert!(ranked.windows(2).all(|w| {
            let (a, b) = (zh.keys[w[0] as usize], zh.keys[w[1] as usize]);
            a > b || (a == b && w[0] < w[1])
        }));
        assert!(ranked.iter().all(|&c| (c as usize) < 500));
        assert_eq!(zh.retrieval.len() as u32, inp.nprobe);
        assert_eq!(zh.estimation.len() as u32, inp.es);
    }
    assert!(!z[0].retrieval.contains(&9) && !z[0].estimation.contains(&9));
    assert_eq!(z[0].keys[9], 0);
}

#[test]
fn group_softmax_over_live_clusters_sums_to_the_group_size() {
    let dt = data(1, 4, 256, 2, 5);
    let inp = input(&dt, 1, 4, 256, 256, 2);
    let total: f64 = reference::probabilities(&inp, 0)
        .iter()
        .map(|p| p.unwrap_or(0.0))
        .sum();
    // Every (row, query head) softmax sums to 1 over the non-empty clusters.
    assert!((total - 8.0).abs() < 1e-9, "{total}");
}

#[test]
fn logit_is_exact_on_small_integers() {
    let q: Vec<f32> = (0..D).map(|i| (i % 5) as f32).collect();
    let c: Vec<f32> = (0..D).map(|i| (i % 3) as f32 - 1.0).collect();
    let exact: f32 = q.iter().zip(&c).map(|(a, b)| a * b).sum();
    assert_eq!(reference::logit(&q, &c, 0.0625), exact * 0.0625);
}

/// Position by position the same cluster, or two whose oracle scores are within 1e-6 relative.
fn rank_equal(a: &[u32], b: &[u32], score: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(&x, &y)| {
            let (sx, sy) = (score[x as usize], score[y as usize]);
            x == y || (sx - sy).abs() <= 1e-6 * sx.max(sy)
        })
}

#[test]
fn reference_agrees_with_the_l33_oracle() {
    for seed in 0..6u64 {
        let (h, group, cap, nc) = (2usize, 4usize, 320usize, 300usize);
        let dt = data(h, group, cap, 1, 100 + seed);
        let inp = input(&dt, h, group, cap, nc, 1);
        let mine = reference::select(&inp);
        for (hh, zh) in mine.iter().enumerate() {
            let idx = HeadIndex {
                head_dim: D,
                centroids: dt.cent[hh * cap * D..(hh * cap + nc) * D].to_vec(),
                value_sum: vec![0.0; nc * D],
                sizes: dt.size[hh * cap..hh * cap + nc]
                    .iter()
                    .map(|&s| s as u32)
                    .collect(),
                lists: vec![Vec::new(); nc],
            };
            let qg = &dt.q[hh * group * D..(hh + 1) * group * D];
            let score = oracle_eval::score_clusters(&idx, nc, qg, group, 1.0 / 16.0);
            let theirs = oracle_eval::select(&score, inp.nprobe as usize, inp.es as usize);
            assert!(
                rank_equal(&zh.retrieval, &theirs.retrieval, &score),
                "seed {seed} head {hh} retrieval"
            );
            assert!(
                rank_equal(&zh.estimation, &theirs.estimation, &score),
                "seed {seed} head {hh} estimation"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Mock backend
// ---------------------------------------------------------------------------------------------

struct Mock {
    gpu: MockGpuBackend,
    cfg: RetroCfg,
    idx: LayerIndex,
    zl: ZoneLists,
}

fn mock(kv_heads: u32, num_q_heads: u32, max_context: u32, scratch: usize) -> Mock {
    let gpu = MockGpuBackend::new();
    let cfg = RetroCfg::new(
        RetroParams::default(),
        kv_heads,
        num_q_heads,
        256,
        0.0625,
        KvDtypeTag::Bf16,
        max_context,
        16,
    )
    .expect("cfg");
    let layout = IndexLayout {
        kv_heads,
        head_dim: 256,
        cap_centroids: cfg.cap_centroids,
        cap_tokens: max_context,
    };
    let idx = LayerIndex::at(gpu.alloc(layout.total_bytes()).expect("index"), layout);
    let zlay = ZoneLayout {
        kv_heads,
        retrieval_cap: cfg.retrieval_cap,
        es_cap: cfg.es_cap,
    };
    let scores = gpu
        .alloc(kv_heads as usize * cfg.cap_centroids as usize * 4)
        .expect("scores");
    let sc = gpu.alloc(scratch.max(4)).expect("scratch");
    let zl = ZoneLists::at(
        gpu.alloc(zlay.bytes()).expect("zones"),
        zlay,
        scores,
        sc,
        scratch,
    );
    Mock { gpu, cfg, idx, zl }
}

#[test]
fn mock_select_is_three_fixed_launches_and_replays_identically() {
    let probe = RetroCfg::new(
        RetroParams::default(),
        4,
        16,
        256,
        0.0625,
        KvDtypeTag::Bf16,
        32_768,
        16,
    )
    .expect("cfg");
    let mut m = mock(4, 16, 32_768, probe.select_scratch_bytes());
    let q = m.gpu.alloc(16 * D * 2).expect("q");
    select_zones(&m.gpu, &m.idx, q, &m.cfg, &mut m.zl, 7).expect("first");
    let lookups = m.gpu.kernel_lookups_snapshot().len();
    select_zones(&m.gpu, &m.idx, q, &m.cfg, &mut m.zl, 7).expect("second");
    let l = m.gpu.launches_snapshot();
    assert_eq!(l.len(), 6);
    let tiles = m.cfg.cap_centroids.div_ceil(TILE_C);
    assert_eq!(
        (l[0].grid, l[0].block, l[0].stream),
        ([tiles, 4, 1], [256, 1, 1], 7)
    );
    assert_eq!((l[1].grid, l[1].block), ([tiles, 4, 1], [256, 1, 1]));
    assert_eq!((l[2].grid, l[2].block), ([4, 1, 1], [1024, 1, 1]));
    // Same shapes and the same argument bytes on every call: a captured graph replays unchanged.
    for i in 0..3 {
        assert_eq!((l[i].grid, &l[i].args), (l[i + 3].grid, &l[i + 3].args));
    }
    assert_eq!(m.gpu.kernel_lookups_snapshot().len(), 2 * lookups);
}

#[test]
fn mock_select_refuses_bad_shapes() {
    let probe = RetroCfg::new(
        RetroParams::default(),
        4,
        16,
        256,
        0.0625,
        KvDtypeTag::Bf16,
        16_384,
        16,
    )
    .expect("cfg");
    let two_rows = scratch_bytes(&probe, 2);
    let mut m = mock(4, 16, 16_384, two_rows);
    let q = m.gpu.alloc(16 * D * 2 * 9).expect("q");
    let cfg = m.cfg.clone();
    // Group 4: 9 rows are 36 query heads, over the kernel's 32.
    assert!(select_zones_rows(&m.gpu, &m.idx, q, 9, 16 * 256, &cfg, &mut m.zl, 0).is_err());
    // 4 rows need more scratch than a 2-row allocation.
    assert!(select_zones_rows(&m.gpu, &m.idx, q, 4, 16 * 256, &cfg, &mut m.zl, 0).is_err());
    assert!(select_zones_rows(&m.gpu, &m.idx, q, 2, 16 * 256, &cfg, &mut m.zl, 0).is_ok());
    // A short row stride.
    assert!(select_zones_rows(&m.gpu, &m.idx, q, 2, 100, &cfg, &mut m.zl, 0).is_err());
    // A config for another head count.
    let other = RetroCfg::new(
        RetroParams::default(),
        2,
        8,
        256,
        0.0625,
        KvDtypeTag::Bf16,
        16_384,
        16,
    )
    .expect("cfg");
    assert!(select_zones(&m.gpu, &m.idx, q, &other, &mut m.zl, 0).is_err());
}

#[test]
fn absent_module_disables_without_lookups() {
    let mut m = mock(2, 8, 16_384, 1 << 16);
    m.gpu.mark_module_absent(super::MODULE);
    assert!(!super::available(&m.gpu));
    let q = m.gpu.alloc(8 * D * 2).expect("q");
    let cfg = m.cfg.clone();
    let err = select_zones(&m.gpu, &m.idx, q, &cfg, &mut m.zl, 0).expect_err("absent");
    assert!(err.to_string().contains("not built"), "{err}");
    assert!(m.gpu.kernel_lookups_snapshot().is_empty());
}

#[test]
fn retrieval_cap_constant_is_the_hosts() {
    assert_eq!(
        super::MAX_RETRIEVAL,
        crate::port::retro::config::MAX_RETRIEVAL_CAP
    );
    const { assert!(super::MAX_RETRIEVAL <= MAX_SELECTED) };
    for ctx in [16_384u32, 131_072, 262_144, 1 << 20] {
        let cfg = RetroCfg::new(
            RetroParams::default(),
            4,
            16,
            256,
            0.0625,
            KvDtypeTag::Bf16,
            ctx,
            16,
        )
        .expect("cfg");
        assert!(cfg.retrieval_cap <= super::MAX_RETRIEVAL, "ctx {ctx}");
    }
}

#[test]
fn scratch_fits_the_hosts_budget() {
    for (h, nq) in [(4u32, 16u32), (2, 8), (8, 64)] {
        for ctx in [16_384u32, 65_536, 131_072, 262_144] {
            let cfg = RetroCfg::new(
                RetroParams::default(),
                h,
                nq,
                256,
                0.0625,
                KvDtypeTag::Bf16,
                ctx,
                16,
            )
            .expect("cfg");
            assert!(
                scratch_bytes(&cfg, 1) <= cfg.select_scratch_bytes(),
                "h {h} ctx {ctx}"
            );
        }
    }
    let l = scratch_layout(4, 4, 8320);
    assert_eq!(l.tiles_cap, 65);
    assert!(
        l.tsum >= 4 * 4 * 65 * 4
            && l.tsum.is_multiple_of(256)
            && l.total >= l.tsum + 4 * 4 * 65 * 4
    );
}

#[test]
fn kernel_constants_match_and_static_smem_fits() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/ada/common/retro_select_ada.cu"
    );
    let src = std::fs::read_to_string(path).expect("retro_select_ada.cu");
    let define = |name: &str| -> u32 {
        let line = src
            .lines()
            .find(|l| l.starts_with(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("#define {name} missing"));
        line.split_whitespace()
            .nth(2)
            .and_then(|v| v.parse().ok())
            .expect("numeric define")
    };
    assert_eq!(define("RS_D"), 256);
    assert_eq!(define("RS_TILE_C"), TILE_C);
    assert_eq!(define("RS_MAX_RG"), MAX_RG);
    assert_eq!(define("RS_MAX_SELECTED"), MAX_SELECTED);
    use crate::port::retro::state::header;
    assert_eq!(define("RS_HDR_NC") as usize, header::N_CENTROIDS);
    assert_eq!(define("RS_HDR_NPROBE") as usize, header::NPROBE);
    assert_eq!(define("RS_HDR_ES") as usize, header::ES);
    let score = MAX_RG * 256 * 2 + MAX_RG * TILE_C * 4 + 2 * MAX_RG * 4;
    let pick = 256 * 4 + 2 * 4 + 2 * 32 * 4 + 4 + MAX_SELECTED * 8;
    assert!(score <= 48 * 1024 && pick <= 48 * 1024, "{score} {pick}");
    for e in ["retro_sel_stats(", "retro_sel_prob(", "retro_sel_pick("] {
        assert!(src.contains(e), "{e}");
    }
}
