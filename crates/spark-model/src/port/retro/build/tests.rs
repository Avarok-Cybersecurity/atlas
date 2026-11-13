// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of the RetroInfer index build: the job normalization and scratch
//! budget against `RetroCfg`, the initialization against L33's oracle, the CPU mirror's
//! invariants (cluster recovery, CSR, value sums, determinism) and its agreement with L33's
//! `oracle::build_head`, the launch sequence on the mock backend (pending build, offsets, block
//! table paths, refusals, absent module). The kernel source constants are in
//! `kernel_source_tests.rs`.

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::geometry::{self, Job, TILES};
use super::{launch, reference};
use crate::layers::qwen3_attention::sparse_seam::{KvDtypeTag, KvLayerView, SeqBatch};
use crate::port::retro::config::{self, RetroCfg, RetroParams};
use crate::port::retro::oracle::{self, HeadKv, KmeansSpec};
use crate::port::retro::state::{IndexLayout, LayerIndex, PendingBuild};

const D: usize = 256;

/// A small deterministic generator (SplitMix64), so the tests need no RNG crate.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    fn uni(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

#[test]
fn jobs_normalize_like_the_oracle() {
    let j = Job::new(4, 1031, 64, 2, 10, 0, 0).expect("job");
    assert_eq!(
        (j.n_centroids, j.n_segments, j.seg_tokens, j.seg_centroids),
        (64, 2, 515, 32)
    );
    assert_eq!(j.end_pos(), 1035);
    let j = Job::new(0, 5, 8, 3, 10, 0, 0).expect("job");
    assert_eq!((j.n_centroids, j.n_segments), (5, 3));
    let j = Job::new(0, 100, 8, 20, 0, 0, 0).expect("job");
    assert_eq!((j.n_segments, j.iters), (8, 1));
    assert!(Job::new(0, 0, 8, 1, 10, 0, 0).is_err());
    assert!(Job::new(0, 10, 0, 1, 10, 0, 0).is_err());
    for n in [1u32, 255, 256, 4096, 131_072, 1 << 20] {
        let j = Job::new(0, n, 1, 1, 1, 0, 0).expect("job");
        assert!(
            j.n_tiles <= TILES
                && j.tile_tokens.is_multiple_of(32)
                && j.n_tiles * j.tile_tokens >= n
        );
    }
}

#[test]
fn scratch_fits_the_hosts_budget() {
    let p = RetroParams::default();
    for (h, ctx) in [(4u32, 16_384u32), (4, 131_072), (2, 262_144), (8, 65_536)] {
        let cfg = RetroCfg::new(p.clone(), h, 4 * h, 256, 0.0625, KvDtypeTag::Bf16, ctx, 16)
            .expect("cfg");
        let tokens = ctx - p.sink - p.window;
        let segs = config::n_segments(tokens, p.segment_tokens);
        let nc = config::n_centroids(tokens, p.avg_cluster, segs);
        for (t, c) in [(tokens, nc), (p.update_interval, p.update_centroids())] {
            let mine = super::scratch_bytes(h, t, c, ctx.div_ceil(16) + 1).expect("bytes");
            assert!(
                mine <= cfg.build_scratch_bytes(t, c),
                "h {h} ctx {ctx} tokens {t}"
            );
        }
    }
}

#[test]
fn init_matches_the_oracle_positions() {
    for (n, nc) in [(131_072usize, 8192usize), (1031, 64), (1000, 1000), (7, 3)] {
        let theirs = oracle::init_positions(n, nc);
        for (i, &t) in theirs.iter().enumerate() {
            assert_eq!(
                geometry::init_index(i as u32, n as u32, nc as u32) as usize,
                t,
                "n {n} nc {nc} i {i}"
            );
        }
    }
}

/// Keys drawn around `k` separated centers (label per token) and random values.
fn gaussian(n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    keys_around(n, k, seed, |_, r| (r.next() % k as u64) as usize)
}

/// Keys around `k` separated centers with contiguous labels: token `t` belongs to center
/// `t * k / n`. With `k` equal to the job's centroid count, each planted cluster holds exactly
/// one uniform-stride initial centroid (`init_index(i)` falls inside block `i`), so the partition
/// is decided by the data with margins far above BF16 rounding, and both builds must find it.
fn planted(n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    keys_around(n, k, seed, |t, _| t * k / n)
}

/// Keys around `k` centers, the label of token `t` from `label(t, rng)`, and random values.
fn keys_around(
    n: usize,
    k: usize,
    seed: u64,
    label: impl Fn(usize, &mut Rng) -> usize,
) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut r = Rng(seed);
    let centers: Vec<f32> = (0..k * D).map(|_| 3.0 * r.uni()).collect();
    let mut keys = vec![0.0f32; n * D];
    let mut vals = vec![0.0f32; n * D];
    let mut labels = vec![0usize; n];
    for t in 0..n {
        let l = label(t, &mut r);
        labels[t] = l;
        for d in 0..D {
            keys[t * D + d] = reference::bf16(1.0 + centers[l * D + d] + 0.05 * r.uni());
            vals[t * D + d] = reference::bf16(r.uni());
        }
    }
    (keys, vals, labels)
}

fn check_csr(h: &reference::HeadIndex, g: &Job) {
    let total: i32 = h.cluster_size.iter().sum();
    assert_eq!(total as u32, g.n_tokens);
    let mut seen = vec![false; g.n_tokens as usize];
    for c in 0..g.n_centroids as usize {
        let s = h.cluster_start[c] as usize;
        let m = &h.cluster_tokens[s..s + h.cluster_size[c] as usize];
        assert!(
            m.windows(2).all(|w| w[0] < w[1]),
            "cluster {c} members not ascending"
        );
        for &p in m {
            let t = p as u32 - g.first_pos;
            assert!(t < g.n_tokens && !seen[t as usize]);
            seen[t as usize] = true;
            assert_eq!(h.assign[t as usize], c as i32);
        }
    }
}

/// 1,031 tokens, 64 clusters in 2 segments of 515 (one tail token joins only the final pass).
fn small_job() -> Job {
    Job::new(4, 1031, 64, 2, 10, 0, 0).expect("job")
}

#[test]
fn reference_recovers_gaussian_clusters_with_exact_sums() {
    let g = small_job();
    let (keys, vals, labels) = gaussian(g.n_tokens as usize, 4, 7);
    let h = reference::build_head(&keys, &vals, &g);
    check_csr(&h, &g);
    for c in 0..g.n_centroids as usize {
        let s = h.cluster_start[c] as usize;
        let m = &h.cluster_tokens[s..s + h.cluster_size[c] as usize];
        let lbl = |p: i32| labels[(p as u32 - g.first_pos) as usize];
        // Purity: a learned cluster never mixes two planted clusters.
        assert!(m.iter().all(|&p| lbl(p) == lbl(m[0])));
        // Value sums: BF16 of the FP32 sum in member order; empty clusters hold 0.
        for d in 0..D {
            let mut acc = 0.0f32;
            for &p in m {
                acc += vals[(p as u32 - g.first_pos) as usize * D + d];
            }
            let want = if m.is_empty() {
                0.0
            } else {
                reference::bf16(acc)
            };
            assert_eq!(h.value_sum[c * D + d].to_bits(), want.to_bits());
        }
    }
}

#[test]
fn reference_agrees_with_the_l33_oracle_on_gaussian_keys() {
    let g = small_job();
    // One planted center per centroid. With fewer centers than centroids (say 4 for 64), every
    // center is split into sub-clusters by its isotropic noise alone, and which token lands in
    // which sub-cluster turns on differences far below one BF16 step of the centered keys: the
    // BF16 build and the f64 oracle then legitimately part ways (about 9 percent agreement),
    // which says nothing about either. Here the partition is the data's.
    let (keys, vals, labels) = planted(g.n_tokens as usize, g.n_centroids as usize, 9);
    let mine = reference::build_head(&keys, &vals, &g);
    // The oracle indexes positions [first, end) of a head's whole KV; give it the span at 0.
    let kv = HeadKv {
        k: &keys,
        v: &vals,
        head_dim: D,
    };
    let spec = KmeansSpec {
        centroids: 64,
        segments: 2,
        iters: 10,
    };
    let theirs = oracle::build_head(kv, 0, g.n_tokens as usize, spec);
    // The data is well posed: the oracle's clusters never mix two planted centers.
    for (c, l) in theirs.lists.iter().enumerate() {
        assert!(
            l.iter()
                .all(|&t| labels[t as usize] == labels[l[0] as usize]),
            "oracle cluster {c} mixes planted centers"
        );
    }
    let mut agree = 0usize;
    let mut worst = 0.0f32;
    for c in 0..g.n_centroids as usize {
        let s = mine.cluster_start[c] as usize;
        let m: Vec<u32> = mine.cluster_tokens[s..s + mine.cluster_size[c] as usize]
            .iter()
            .map(|&p| p as u32 - g.first_pos)
            .collect();
        agree += m.iter().filter(|t| theirs.lists[c].contains(t)).count();
        if m == theirs.lists[c] && !m.is_empty() {
            // Max-abs error relative to the centroid's max-abs value (BF16 storage and centering).
            let (x, y) = (
                &mine.centroids[c * D..(c + 1) * D],
                &theirs.centroids[c * D..(c + 1) * D],
            );
            let err = x
                .iter()
                .zip(y)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mag = y.iter().map(|b| b.abs()).fold(0.0f32, f32::max).max(1e-6);
            worst = worst.max(err / mag);
        }
    }
    let frac = agree as f64 / f64::from(g.n_tokens);
    assert!(frac >= 0.99, "assignment agreement {frac}");
    assert!(worst <= 1e-2, "centroid relative error {worst}");
}

#[test]
fn reference_is_deterministic_and_total_on_uniform_keys() {
    let g = Job::new(4, 1024, 64, 1, 10, 0, 0).expect("job");
    let mut r = Rng(11);
    let keys: Vec<f32> = (0..1024 * D).map(|_| reference::bf16(r.uni())).collect();
    let vals: Vec<f32> = (0..1024 * D).map(|_| reference::bf16(r.uni())).collect();
    let a = reference::build_head(&keys, &vals, &g);
    let b = reference::build_head(&keys, &vals, &g);
    assert_eq!(a, b);
    check_csr(&a, &g);
}

#[test]
fn csr_is_stable_and_skips_unassigned() {
    let a = [2, -1, 0, 2, 2, 0, -1, 1];
    let (start, size, members) = reference::csr(&a, 4, 100);
    assert_eq!(size, vec![2, 1, 3, 0]);
    assert_eq!(start, vec![0, 2, 3, 6]);
    assert_eq!(members, vec![102, 105, 107, 100, 103, 104]);
}

#[test]
fn butterfly_sums_exact_values() {
    let mut p = [0.0f32; 32];
    for (i, v) in p.iter_mut().enumerate() {
        *v = i as f32;
    }
    assert_eq!(reference::butterfly_sum(p), 496.0);
}

// ---------------------------------------------------------------------------------------------
// Mock backend
// ---------------------------------------------------------------------------------------------

struct Fixture {
    gpu: MockGpuBackend,
    cfg: RetroCfg,
    kv: KvLayerView,
    table: Vec<u32>,
    scratch: DevicePtr,
    scratch_bytes: usize,
}

/// A 2-KV-head BF16 layer of `context` tokens on the mock backend, and its (zeroed) index.
fn fixture(context: u32) -> (Fixture, LayerIndex) {
    let gpu = MockGpuBackend::new();
    let p = RetroParams::default();
    let cfg = RetroCfg::new(p, 2, 8, 256, 0.0625, KvDtypeTag::Bf16, context, 16).expect("cfg");
    let cap_centroids = cfg.cap_centroids;
    let layout = IndexLayout {
        kv_heads: 2,
        head_dim: 256,
        cap_centroids,
        cap_tokens: context,
    };
    let idx = LayerIndex::at(gpu.alloc(layout.total_bytes()).expect("index"), layout);
    let blocks = context.div_ceil(16);
    let pool_bytes = blocks as usize * 16 * 2 * D * 2;
    let kv = KvLayerView {
        k_pool: gpu.alloc(pool_bytes).expect("k"),
        v_pool: gpu.alloc(pool_bytes).expect("v"),
        dtype: KvDtypeTag::Bf16,
        block_size: 16,
        kv_heads: 2,
        head_dim: 256,
        kv_scale: 1.0,
        v_scale: 1.0,
        block_stride_bytes: 16 * 2 * 256 * 2,
        cache_stride: 16 * 2 * 256,
    };
    let scratch_bytes = cfg.build_scratch_bytes(context, context / 8);
    let scratch = gpu.alloc(scratch_bytes).expect("scratch");
    let table = (0..blocks).rev().collect();
    (
        Fixture {
            gpu,
            cfg,
            kv,
            table,
            scratch,
            scratch_bytes,
        },
        idx,
    )
}

/// One build through the public entry point over a one-sequence batch of `lens`.
fn go(
    f: &Fixture,
    idx: &mut LayerIndex,
    lens: &[u32],
    tables: &[&[u32]],
    dev: DevicePtr,
) -> anyhow::Result<()> {
    super::build_layer_index(&f.gpu, &f.kv, &batch(lens, tables, dev), &f.cfg, idx, 0)
}

fn pending(f: &Fixture, first: u32, tokens: u32, centroids: u32, co: u32, to: u32) -> PendingBuild {
    PendingBuild {
        first_pos: first,
        tokens,
        segments: config::n_segments(tokens, 8192),
        centroids,
        centroid_offset: co,
        token_offset: to,
        iters: 10,
        scratch: f.scratch,
        scratch_bytes: f.scratch_bytes,
    }
}

fn batch<'a>(lens: &'a [u32], tables: &'a [&'a [u32]], device_table: DevicePtr) -> SeqBatch<'a> {
    SeqBatch {
        n: lens.len(),
        block_table: device_table,
        seq_lens: DevicePtr::NULL,
        max_blocks_per_seq: 0,
        host_seq_lens: lens,
        seq_ids: &[],
        rows_per_seq: 1,
        q_row_stride: 0,
        q_rows: 0,
        first_pos: &[],
        new_tokens: &[],
        host_block_tables: tables,
        block_table_ptrs: DevicePtr::NULL,
    }
}

#[test]
fn mock_build_issues_the_pass_sequence_and_leaves_the_header_alone() {
    let ctx = 4096 + 68;
    let (f, mut idx) = fixture(ctx);
    idx.pending = Some(pending(&f, 4, 4096, 256, 0, 0));
    go(&f, &mut idx, &[ctx], &[f.table.as_slice()], DevicePtr::NULL).expect("build");
    let l = f.gpu.launches_snapshot();
    assert_eq!(l.len(), launch::launches_per_build(10));
    assert_eq!(l[0].grid, [64, 2, 1]); // mean partial
    assert_eq!(l[2].grid, [32, 2, 1]); // init: 256 centroids / 8 warps
    assert_eq!(l[3].grid, [32, 2, 1]); // assign: 4096 tokens / 128
    assert_eq!(l[5].block, [1024, 1, 1]); // scan
    assert_eq!(l[6].block, [32, 1, 1]); // scatter
    let lookups = f.gpu.kernel_lookups_snapshot();
    assert_eq!(lookups.len(), 8);
    assert!(lookups.iter().all(|(m, _)| m == launch::MODULE));
    // The header is the host's: the build never writes it.
    let region = f.gpu.read_alloc(idx.base).expect("index region");
    assert!(region[..32].iter().all(|&b| b == 0));
    // No device table: the host table went to the scratch's block-table section.
    let job = Job::from_pending(idx.pending.as_ref().expect("pending")).expect("job");
    let at = geometry::scratch_layout(2, &job, ctx.div_ceil(16)).block_table;
    let sc = f.gpu.read_alloc(f.scratch).expect("scratch");
    assert_eq!(
        u32::from_le_bytes([sc[at], sc[at + 1], sc[at + 2], sc[at + 3]]),
        f.table[0]
    );
}

#[test]
fn mock_update_appends_at_the_offsets_and_uses_the_device_table() {
    let ctx = 8192 + 68;
    let (f, mut idx) = fixture(ctx);
    let table = f.gpu.alloc(ctx.div_ceil(16) as usize * 4).expect("table");
    idx.pending = Some(pending(&f, 6000, 1024, 64, 448, 5000));
    go(&f, &mut idx, &[ctx], &[f.table.as_slice()], table).expect("update");
    let l = f.gpu.launches_snapshot();
    use spark_runtime::gpu::mock::MockArg;
    // init's centroid pointer is the index's, 448 rows in; the pool args carry the device table.
    assert_eq!(
        l[2].args[8],
        MockArg::Buffer(idx.centroids.offset(448 * 256 * 2))
    );
    assert_eq!(l[2].args[1], MockArg::Buffer(table));
    // The final scan writes the CSR starts at the index's offset 448 with base 5000.
    let scan = &l[l.len() - 3];
    assert_eq!(
        scan.args[2],
        MockArg::Buffer(idx.list_offsets.offset(448 * 4))
    );
    assert_eq!(scan.args[8], MockArg::Bytes(5000i32.to_le_bytes().to_vec()));
}

#[test]
fn build_refuses_bad_calls() {
    let ctx = 1024 + 68;
    let (f, mut idx) = fixture(ctx);
    let one: [&[u32]; 1] = [&f.table];
    let null = DevicePtr::NULL;
    assert!(
        go(&f, &mut idx, &[ctx], &one, null).is_err(),
        "no pending build"
    );
    let cap = idx.layout.cap_centroids;
    idx.pending = Some(pending(&f, 4, 1024, 64, cap - 10, 0));
    assert!(
        go(&f, &mut idx, &[ctx], &one, null).is_err(),
        "past the capacity"
    );
    idx.pending = Some(pending(&f, 100, 1024, 64, 0, 0));
    assert!(
        go(&f, &mut idx, &[ctx], &one, null).is_err(),
        "past the sequence"
    );
    let mut p = pending(&f, 4, 1024, 64, 0, 0);
    p.scratch_bytes = 1024;
    idx.pending = Some(p);
    assert!(
        go(&f, &mut idx, &[ctx], &one, null).is_err(),
        "scratch too small"
    );
    idx.pending = Some(pending(&f, 4, 1024, 64, 0, 0));
    let two = [f.table.as_slice(), f.table.as_slice()];
    assert!(
        go(&f, &mut idx, &[ctx, ctx], &two, null).is_err(),
        "two sequences"
    );
    assert!(
        go(&f, &mut idx, &[ctx], &[], null).is_err(),
        "no block table at all"
    );
    assert!(
        f.gpu.launches_snapshot().is_empty(),
        "every refusal happens before a launch"
    );
}

#[test]
fn absent_module_disables_without_lookups() {
    let ctx = 1024 + 68;
    let (f, mut idx) = fixture(ctx);
    f.gpu.mark_module_absent(launch::MODULE);
    assert!(!super::available(&f.gpu));
    idx.pending = Some(pending(&f, 4, 1024, 64, 0, 0));
    let err =
        go(&f, &mut idx, &[ctx], &[f.table.as_slice()], DevicePtr::NULL).expect_err("no module");
    assert!(err.to_string().contains("not built"), "{err}");
    assert!(f.gpu.kernel_lookups_snapshot().is_empty());
    assert!(f.gpu.launches_snapshot().is_empty());
}
