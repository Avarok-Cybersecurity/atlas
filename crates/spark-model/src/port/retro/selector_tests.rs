// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the retro selector and its device
//! helpers: the plan decisions (graph capture, prefill, short contexts, the
//! reservation), shadow mode end to end (device K/V read through the block
//! table, the worker's oracle, the published means) against a direct oracle
//! run, and the device operations that do not need the L34 kernels (header
//! writes, index copies, fingerprints, row uploads).

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::config::{RetroCfg, RetroParams};
use super::device::Device;
use super::metrics::{Fallback, Metrics};
use super::oracle::{HeadKv, OracleLayer};
use super::oracle_eval::measure_layer;
use super::oracle_tests::clustered;
use super::selector::RetroSelector;
use super::state::{IndexCounts, IndexLayout};
use super::update::initial_end;
use crate::layers::qwen3_attention::sparse_seam::{
    KvDtypeTag, KvLayerView, KvSelector, SeamCtx, SeamPhase, SeqBatch, SparsePlan,
};

const BS: u32 = 16;

fn metrics() -> &'static Metrics {
    Box::leak(Box::new(Metrics::default()))
}

fn params() -> RetroParams {
    RetroParams {
        segment_tokens: 1024,
        min_context: 1024,
        update_interval: 256,
        index_cache_mb: 64,
        shadow_every: 1,
        ..RetroParams::default()
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn round_bf16(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|&x| half::bf16::from_f32(x).to_f32())
        .collect()
}

/// A paged BF16 pool holding `heads` K and V sequences of `n` tokens each, at
/// blocks `table`.
struct Pool {
    kv: KvLayerView,
    table: Vec<u32>,
}

fn pool(gpu: &MockGpuBackend, k: &[Vec<f32>], v: &[Vec<f32>], hd: usize, n: usize) -> Pool {
    let heads = k.len();
    let blocks = n.div_ceil(BS as usize);
    let block_bytes = BS as usize * heads * hd * 2;
    let k_pool = gpu.alloc(blocks * block_bytes).unwrap();
    let v_pool = gpu.alloc(blocks * block_bytes).unwrap();
    // Reverse block order, so a wrong table walk shows up.
    let table: Vec<u32> = (0..blocks as u32).rev().collect();
    for (pool, src) in [(k_pool, k), (v_pool, v)] {
        for (lb, &blk) in table.iter().enumerate() {
            let mut img = vec![0f32; BS as usize * heads * hd];
            for slot in 0..BS as usize {
                let pos = lb * BS as usize + slot;
                if pos >= n {
                    break;
                }
                for h in 0..heads {
                    img[(slot * heads + h) * hd..(slot * heads + h + 1) * hd]
                        .copy_from_slice(&src[h][pos * hd..(pos + 1) * hd]);
                }
            }
            gpu.copy_h2d(&bf16_bytes(&img), pool.offset(blk as usize * block_bytes))
                .unwrap();
        }
    }
    Pool {
        kv: KvLayerView {
            k_pool,
            v_pool,
            dtype: KvDtypeTag::Bf16,
            block_size: BS,
            kv_heads: heads as u32,
            head_dim: hd as u32,
            kv_scale: 1.0,
            v_scale: 1.0,
            block_stride_bytes: block_bytes as u64,
            cache_stride: 0,
        },
        table,
    }
}

fn ctx<'a>(
    gpu: &'a MockGpuBackend,
    kv: KvLayerView,
    nq: u32,
    phase: SeamPhase,
    in_graph: bool,
) -> SeamCtx<'a> {
    SeamCtx {
        gpu,
        attn_layer: 0,
        kv,
        stream: 0,
        in_graph,
        phase,
        num_q_heads: nq,
        inv_sqrt_d: 1.0 / (kv.head_dim as f32).sqrt(),
    }
}

/// Run the decode hooks of one single-sequence step.
fn decode_step(
    sel: &RetroSelector,
    c: &SeamCtx,
    dev_meta: DevicePtr,
    table: &[u32],
    len: u32,
    q: DevicePtr,
) -> SparsePlan {
    let lens = [len];
    let first = [len - 1];
    let one = [1u32];
    let tables = [table];
    let s = SeqBatch {
        n: 1,
        block_table: dev_meta,
        seq_lens: dev_meta.offset(4096),
        max_blocks_per_seq: table.len() as u32,
        host_seq_lens: &lens,
        seq_ids: &[],
        rows_per_seq: 1,
        q_row_stride: c.num_q_heads * c.kv.head_dim,
        q_rows: 1,
        first_pos: &first,
        new_tokens: &one,
        host_block_tables: &tables,
        block_table_ptrs: DevicePtr::NULL,
    };
    sel.on_kv_written(c, &s, &first, &one).unwrap();
    sel.plan(c, &s, q).unwrap()
}

#[test]
fn reservation_follows_the_output_switch() {
    let on = RetroSelector::with_metrics(params(), true, metrics());
    assert_eq!(on.reserved_bytes(), 64 << 20);
    assert_eq!(on.name(), "retro");
    let shadow_only = RetroSelector::with_metrics(params(), false, metrics());
    assert_eq!(
        shadow_only.reserved_bytes(),
        0,
        "shadow mode allocates nothing on the device"
    );
    assert!(
        !super::api::vetoes_decode_graphs(),
        "nothing registered in tests"
    );
}

#[test]
fn captures_and_short_contexts_plan_dense() {
    let gpu = MockGpuBackend::new();
    let m = metrics();
    let sel = RetroSelector::with_metrics(params(), true, m);
    let hd = 256;
    let (k, v, _) = clustered(600, hd, 8, 16, 1.0, 0.1, 1);
    let p = pool(&gpu, &[k], &[v], hd, 600);
    let meta = gpu.alloc(8192).unwrap();
    let q = gpu.alloc(2 * hd * 2).unwrap();
    let c = ctx(&gpu, p.kv, 2, SeamPhase::Decode, true);
    assert!(matches!(
        decode_step(&sel, &c, meta, &p.table, 600, q),
        SparsePlan::Dense
    ));
    assert_eq!(
        m.fallback(Fallback::Graph),
        1,
        "a capture gets the exact dense tables"
    );
    let c = ctx(&gpu, p.kv, 2, SeamPhase::Decode, false);
    assert!(matches!(
        decode_step(&sel, &c, meta, &p.table, 600, q),
        SparsePlan::Dense
    ));
    // The mock reports every kernel module present, so the reason is the
    // short context (600 tokens is below --retro-min-context); should the L34
    // availability probes ever answer false on the mock, the reason is
    // `kernel` instead, and the plan is dense either way.
    let short = m.fallback(Fallback::ShortContext);
    assert_eq!(short + m.fallback(Fallback::Kernel), 1);
    assert_eq!(m.decodes(), 0);
}

#[test]
fn an_absent_kernel_module_turns_retro_off_and_decode_stays_dense() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent("retro_attn_ada");
    let m = metrics();
    let sel = RetroSelector::with_metrics(params(), true, m);
    let hd = 256;
    let (k, v, _) = clustered(4096, hd, 8, 16, 1.0, 0.1, 2);
    let p = pool(&gpu, &[k], &[v], hd, 4096);
    let meta = gpu.alloc(8192).unwrap();
    let q = gpu.alloc(2 * hd * 2).unwrap();
    let c = ctx(&gpu, p.kv, 2, SeamPhase::Decode, false);
    for _ in 0..3 {
        assert!(matches!(
            decode_step(&sel, &c, meta, &p.table, 4096, q),
            SparsePlan::Dense
        ));
    }
    assert_eq!(
        m.fallback(Fallback::Kernel),
        3,
        "every call counted, none failed"
    );
    assert_eq!(m.decodes(), 0);
    assert_eq!(
        gpu.launches_snapshot().len(),
        0,
        "nothing launched, nothing built"
    );
}

#[test]
fn shadow_mode_measures_what_the_oracle_measures() {
    let gpu = MockGpuBackend::new();
    let m = metrics();
    let pp = params();
    let sel = RetroSelector::with_metrics(
        RetroParams {
            shadow: true,
            ..pp.clone()
        },
        false,
        m,
    );
    let (hd, n, heads, nq) = (32usize, 2048u32, 2usize, 4u32);
    let (k0, v0, dirs) = clustered(n as usize, hd, 128, 16, 4.0, 0.1, 31);
    let (k1, v1, dirs1) = clustered(n as usize, hd, 128, 16, 4.0, 0.1, 32);
    let (k, v) = (
        vec![round_bf16(&k0), round_bf16(&k1)],
        vec![round_bf16(&v0), round_bf16(&v1)],
    );
    let p = pool(&gpu, &k, &v, hd, n as usize);
    // Query heads 0-1 read KV head 0, heads 2-3 read KV head 1; each aims at a
    // cluster of its own KV head.
    let aim = |h: usize| {
        if h < 2 {
            &dirs[(5 * h + 3) % 128]
        } else {
            &dirs1[(5 * h + 3) % 128]
        }
    };
    let qv: Vec<f32> = round_bf16(
        &(0..nq as usize)
            .flat_map(|h| aim(h).iter().map(|x| 16.0 * x).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    );
    let q = gpu.alloc(qv.len() * 2).unwrap();
    gpu.copy_h2d(&bf16_bytes(&qv), q).unwrap();
    let meta = gpu.alloc(8192).unwrap();
    let c = ctx(&gpu, p.kv, nq, SeamPhase::Decode, false);
    assert!(
        matches!(
            decode_step(&sel, &c, meta, &p.table, n, q),
            SparsePlan::Dense
        ),
        "shadow never plans"
    );
    let t0 = std::time::Instant::now();
    let got = loop {
        if let Some(a) = m.shadow(0) {
            break a;
        }
        assert!(
            t0.elapsed().as_secs() < 120,
            "shadow worker produced nothing"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let hk: Vec<HeadKv<'_>> = k
        .iter()
        .zip(&v)
        .map(|(k, v)| HeadKv { k, v, head_dim: hd })
        .collect();
    let l = OracleLayer::build(&hk, &pp, initial_end(n, &pp, BS).unwrap());
    let want = measure_layer(
        &hk,
        &l.heads,
        &l.counts,
        &qv,
        (nq as usize) / heads,
        n,
        c.inv_sqrt_d,
        None,
    );
    assert_eq!(got.samples, 1);
    assert!(
        (got.mass_total - want.mass_total).abs() < 1e-9,
        "{got:?} vs {want:?}"
    );
    assert!((got.out_rel_err - want.out_rel_err).abs() < 1e-9);
    assert!((got.jaccard - want.jaccard).abs() < 1e-9);
    assert!(
        want.mass_total > 0.9,
        "aimed queries are recalled: {want:?}"
    );
}

fn device(gpu: &MockGpuBackend) -> Device {
    let cfg = RetroCfg::new(params(), 2, 4, 32, 0.17, KvDtypeTag::Bf16, 4096, BS).unwrap();
    let mut d = Device::new(gpu, cfg).unwrap();
    d.ensure_bufs(256).unwrap();
    d
}

#[test]
fn headers_and_index_copies_land_where_the_kernels_read() {
    let gpu = MockGpuBackend::new();
    let mut d = device(&gpu);
    let small = IndexLayout {
        kv_heads: 2,
        head_dim: 32,
        cap_centroids: 16,
        cap_tokens: 128,
    };
    let (mut a, _) = d.alloc_index(small).unwrap();
    // Fill a's sections with a pattern per head.
    let c = IndexCounts {
        n_centroids: 10,
        nprobe: 1,
        es: 3,
        indexed_end: 100,
        sink: 4,
        listed: 96,
    };
    let fill = |base: DevicePtr, stride: usize, used: usize, h: u8| {
        gpu.copy_h2d(&vec![h + 1; used], base.offset(stride * h as usize))
            .unwrap()
    };
    for h in 0..2u8 {
        fill(a.centroids, 16 * 32 * 2, 10 * 32 * 2, h);
        fill(a.value_sum, 16 * 32 * 2, 10 * 32 * 2, h);
        fill(a.cluster_size, 16 * 4, 10 * 4, h);
        fill(a.list_offsets, 17 * 4, 11 * 4, h);
        fill(a.list_tokens, 128 * 4, 96 * 4, h);
    }
    a.counts = c;
    d.write_header(&gpu, &a, 0).unwrap();
    let mut hb = vec![0u8; 32];
    gpu.copy_d2h(a.header, &mut hb).unwrap();
    assert_eq!(IndexCounts::from_bytes(&hb), c);
    let big = IndexLayout {
        kv_heads: 2,
        head_dim: 32,
        cap_centroids: 64,
        cap_tokens: 512,
    };
    let (mut b, _) = d.alloc_index(big).unwrap();
    d.copy_index(&gpu, &a, &mut b, 0).unwrap();
    assert_eq!(b.counts, c);
    let read = |p: DevicePtr, n: usize| {
        let mut x = vec![0u8; n];
        gpu.copy_d2h(p, &mut x).unwrap();
        x
    };
    for h in 0..2usize {
        let want = h as u8 + 1;
        assert!(
            read(b.centroids.offset(h * 64 * 32 * 2), 10 * 32 * 2)
                .iter()
                .all(|&x| x == want),
            "head {h} centroids"
        );
        assert!(
            read(b.list_offsets.offset(h * 65 * 4), 11 * 4)
                .iter()
                .all(|&x| x == want)
        );
        assert!(
            read(b.list_tokens.offset(h * 512 * 4), 96 * 4)
                .iter()
                .all(|&x| x == want)
        );
    }
    gpu.copy_d2h(b.header, &mut hb).unwrap();
    assert_eq!(IndexCounts::from_bytes(&hb), c);
    assert!(
        d.copy_index(&gpu, &b, &mut a, 0).is_ok(),
        "same counts fit the smaller region"
    );
    let tiny = IndexLayout {
        kv_heads: 2,
        head_dim: 32,
        cap_centroids: 4,
        cap_tokens: 16,
    };
    let (mut t, _) = d.alloc_index(tiny).unwrap();
    assert!(
        d.copy_index(&gpu, &b, &mut t, 0).is_err(),
        "a region too small is refused"
    );
}

#[test]
fn fingerprints_read_through_the_block_table() {
    let gpu = MockGpuBackend::new();
    let d = device(&gpu);
    let (k, v, _) = clustered(100, 32, 4, 8, 1.0, 0.3, 41);
    let (k2, v2, _) = clustered(100, 32, 4, 8, 1.0, 0.3, 42);
    let p = pool(&gpu, &[k.clone(), k2], &[v, v2], 32, 100);
    let f = d.read_fingerprint(&gpu, &p.kv, &p.table, 37, 0).unwrap();
    assert_eq!(f.pos, 37);
    assert_eq!(
        f.bytes,
        bf16_bytes(&k[37 * 32..38 * 32]),
        "head 0's K at position 37"
    );
    let g = d.read_fingerprint(&gpu, &p.kv, &p.table, 38, 0).unwrap();
    assert_ne!(f, g);
    assert!(d.read_fingerprint(&gpu, &p.kv, &p.table, 4000, 0).is_err());
    d.upload_row(&gpu, &p.table, 100, 0).unwrap();
    let b = d.bufs.as_ref().unwrap();
    let mut t = vec![0u8; p.table.len() * 4];
    gpu.copy_d2h(b.table, &mut t).unwrap();
    let back: Vec<u32> = t
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(back, p.table);
    assert!(
        d.upload_row(&gpu, &[0u32; 300], 100, 0).is_err(),
        "longer than the buffer"
    );
}
