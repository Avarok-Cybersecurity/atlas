// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the sparse-attention seam (port lane
//! L29): the identity selector's fixed buffers, the paged decode launched
//! through an identity View against the dense launch (the CPU stand-in for the
//! GPU identity gate), and layer eligibility. Also the small mock rig the
//! site tests (`sparse_seam_site_tests.rs`) share.

use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use super::identity::{ARENA_BYTES, IdentitySelector};
use super::metrics::Counters;
use super::route::{self, DecodeRoute, RouteEnv, Tables, ViewLedger};
use super::tests::kv_view;
use super::{KvDtypeTag, KvSelector, SeamCtx, SeamDims, SeamPhase, SeqBatch, SparsePlan};
use crate::layer::{AttnMetadataDev, ForwardContext, MoeLoraRoute};
use crate::layers::FfnComponent;
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight};

const H: usize = 128;
pub(super) const NQ: u32 = 2;
pub(super) const NKV: u32 = 1;
pub(super) const HD: u32 = 64;
pub(super) const BS: u32 = 16;
pub(super) const DIMS: SeamDims = SeamDims {
    nq: NQ,
    nkv: NKV,
    hd: HD,
};

/// Everything a forward context borrows, on the mock backend.
pub(super) struct Rig {
    pub(super) gpu: MockGpuBackend,
    config: ModelConfig,
    buffers: BufferArena,
    dispatch: GemmDispatch,
    derived: DerivedWeights,
    levers: ModelLevers,
    stats: ModelStats,
}

impl Rig {
    pub(super) fn new() -> Self {
        let gpu = MockGpuBackend::new();
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.hidden_size = H;
        config.intermediate_size = 128;
        config.num_attention_heads = NQ as usize;
        config.num_key_value_heads = NKV as usize;
        config.head_dim = HD as usize;
        config.num_experts = 1;
        config.num_experts_per_tok = 1;
        config.moe_intermediate_size = 128;
        config.vocab_size = 128;
        let buffers = BufferArena::new(&config, 160, 256, 16, 8, &gpu).unwrap();
        Self {
            gpu,
            config,
            buffers,
            dispatch: GemmDispatch::defaults(),
            derived: DerivedWeights::new(),
            levers: ModelLevers::defaults(),
            stats: ModelStats::new(),
        }
    }

    pub(super) fn ctx(&self, graph_capture: bool) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: &self.gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture,
            decode_step: true,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    }

    fn dense(&self, n: usize, k: usize) -> DenseWeight {
        DenseWeight {
            weight: self.gpu.alloc(n * k * 2).unwrap(),
        }
    }

    /// A plain full-attention layer over `dtype` KV, FP8 scales 0.5 / 2.0.
    pub(super) fn layer(&self, dtype: KvCacheDtype) -> Qwen3AttentionLayer {
        let norm = self.dense(1, H);
        let attn = AttentionWeights {
            q_proj: self.dense((NQ * HD) as usize, H),
            k_proj: self.dense((NKV * HD) as usize, H),
            v_proj: self.dense((NKV * HD) as usize, H),
            o_proj: QuantizedWeight::null(),
            q_norm: self.dense(1, HD as usize),
            k_norm: self.dense(1, HD as usize),
            q_norm_full: None,
            k_norm_full: None,
            k_scale: 0.5,
            v_scale: 2.0,
        };
        Qwen3AttentionLayer::new(
            norm,
            attn,
            norm,
            FfnComponent::None,
            0,
            None,
            None,
            None,
            &self.gpu,
            dtype,
            0,
            &self.config,
        )
        .unwrap()
    }

    /// One attention layer of 8 blocks; `hss` caps blocks per sequence the
    /// way `--high-speed-swap` does.
    pub(super) fn kv(&self, dtype: KvCacheDtype, hss: bool) -> PagedKvCache {
        let cfg = KvCacheConfig {
            block_size: BS as usize,
            num_kv_heads: NKV as usize,
            head_dim: HD as usize,
            num_layers: 1,
            dtype,
            layer_dtypes: Vec::new(),
            layer_dims: Vec::new(),
            cache_blocks_per_seq: hss.then_some(4),
        };
        PagedKvCache::new(cfg, 8, &self.gpu).unwrap()
    }

    pub(super) fn upload(&self, v: &[i32]) -> DevicePtr {
        let p = self.gpu.alloc(v.len() * 4).unwrap();
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.gpu.copy_h2d(&bytes, p).unwrap();
        p
    }

    pub(super) fn meta(&self, bt: DevicePtr, sl: DevicePtr, mb: u32, n: u32) -> AttnMetadataDev {
        AttnMetadataDev {
            positions: DevicePtr::NULL,
            positions_h: DevicePtr::NULL,
            positions_w: DevicePtr::NULL,
            slot: DevicePtr::NULL,
            seq_len: sl,
            block_table: bt,
            max_blocks_per_seq: mb,
            num_seqs: n,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        }
    }
}

fn seam_ctx(gpu: &MockGpuBackend, in_graph: bool) -> SeamCtx<'_> {
    SeamCtx {
        gpu,
        attn_layer: 0,
        kv: kv_view(),
        stream: 0,
        in_graph,
        phase: SeamPhase::Decode,
        num_q_heads: NQ,
        inv_sqrt_d: 0.125,
    }
}

fn seq_batch(bt: DevicePtr, sl: DevicePtr, mb: u32, lens: &[u32]) -> SeqBatch<'_> {
    SeqBatch {
        n: lens.len(),
        block_table: bt,
        seq_lens: sl,
        max_blocks_per_seq: mb,
        host_seq_lens: lens,
        seq_ids: &[],
        rows_per_seq: 1,
        q_row_stride: NQ * HD,
        q_rows: lens.len(),
        first_pos: &[],
        new_tokens: &[],
        host_block_tables: &[],
        block_table_ptrs: DevicePtr::NULL,
    }
}

fn le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn the_identity_view_copies_the_real_tables_into_fixed_buffers() {
    let rig = Rig::new();
    let table = [3, 4, 0, 0, 7, 8, 9, 0];
    let (bt, sl) = (rig.upload(&table), rig.upload(&[33, 50]));
    let sel = IdentitySelector::new();
    let lens = [33u32, 50];
    let s = seq_batch(bt, sl, 4, &lens);
    sel.on_kv_written(&seam_ctx(&rig.gpu, false), &s, &[], &[])
        .unwrap();
    let plan = sel.plan(&seam_ctx(&rig.gpu, false), &s, DevicePtr::NULL);
    let SparsePlan::View {
        block_table,
        seq_lens,
        max_blocks_per_seq,
    } = plan.unwrap()
    else {
        panic!("identity must plan a View once its arena exists");
    };
    assert_eq!(max_blocks_per_seq, 4);
    let arena = rig
        .gpu
        .read_alloc(block_table)
        .expect("the table starts the arena");
    assert_eq!(arena.len(), ARENA_BYTES);
    assert_eq!(&arena[..32], le(&table).as_slice());
    let off = (seq_lens.0 - block_table.0) as usize;
    assert_eq!(&arena[off..off + 8], le(&[33, 50]).as_slice());
    // Captured again: the same buffers (the graph rule).
    let again = sel.plan(&seam_ctx(&rig.gpu, true), &s, DevicePtr::NULL);
    let SparsePlan::View {
        block_table: b2,
        seq_lens: s2,
        ..
    } = again.unwrap()
    else {
        panic!("a capture after the arena exists plans a View");
    };
    assert_eq!((b2, s2), (block_table, seq_lens));
}

#[test]
fn the_identity_selector_plans_dense_before_its_arena_or_beyond_it() {
    let rig = Rig::new();
    let (bt, sl) = (rig.upload(&[1, 2]), rig.upload(&[20]));
    let sel = IdentitySelector::new();
    let lens = [20u32];
    let s = seq_batch(bt, sl, 2, &lens);
    let plan = sel.plan(&seam_ctx(&rig.gpu, true), &s, DevicePtr::NULL);
    assert!(
        matches!(plan.unwrap(), SparsePlan::Dense),
        "never allocates while capturing"
    );
    // One row of ARENA_BYTES / 4 columns: the table alone fills the arena, so
    // the lengths after it cannot fit and nothing may be copied.
    let beyond = u32::try_from(ARENA_BYTES / 4).unwrap();
    assert!(IdentitySelector::view_bytes(1, beyond as usize) > ARENA_BYTES);
    let huge = seq_batch(bt, sl, beyond, &lens);
    let plan = sel.plan(&seam_ctx(&rig.gpu, false), &huge, DevicePtr::NULL);
    assert!(
        matches!(plan.unwrap(), SparsePlan::Dense),
        "a view beyond {ARENA_BYTES} bytes is Dense"
    );
    assert_eq!(sel.reserved_bytes(), ARENA_BYTES as u64);
}

type LaunchKey = (u64, [u32; 3], [u32; 3], Vec<MockArg>);

/// Each launch's identity, with every `from` pointer argument replaced by its `to`.
fn remap(launches: &[MockLaunch], map: &[(DevicePtr, DevicePtr)]) -> Vec<LaunchKey> {
    let swap = |p: &DevicePtr| {
        map.iter()
            .find(|(from, _)| from == p)
            .map_or(*p, |(_, to)| *to)
    };
    launches
        .iter()
        .map(|l| {
            let args = l
                .args
                .iter()
                .map(|a| match a {
                    MockArg::Buffer(p) => MockArg::Buffer(swap(p)),
                    other => other.clone(),
                })
                .collect();
            (l.func, l.grid, l.block, args)
        })
        .collect()
}

/// The query, output and workspace buffers a paged decode reads and writes.
/// Allocated once per test and shared by every run it compares, so the
/// launches can differ only in the table arguments.
#[derive(Clone, Copy)]
struct DecodeIo {
    q: DevicePtr,
    out: DevicePtr,
    ws: DevicePtr,
}

impl DecodeIo {
    fn new(rig: &Rig) -> Self {
        Self {
            q: rig.gpu.alloc(4096).unwrap(),
            out: rig.gpu.alloc(4096).unwrap(),
            ws: rig.gpu.alloc(1 << 16).unwrap(),
        }
    }
}

/// Launch the paged decode of two rows over `t`; the launches it made.
fn decode_launches(
    rig: &Rig,
    layer: &Qwen3AttentionLayer,
    kv: &PagedKvCache,
    io: DecodeIo,
    t: Tables,
) -> Vec<MockLaunch> {
    let before = rig.gpu.launches_snapshot().len();
    layer
        .run_paged_decode(
            &rig.gpu,
            io.q,
            kv,
            io.out,
            t.block_table,
            t.seq_lens,
            t.max_blocks_per_seq,
            2,
            NQ,
            NKV,
            HD,
            BS,
            0.125,
            NQ * HD,
            io.ws,
            16,
            0,
        )
        .unwrap();
    rig.gpu.launches_snapshot()[before..].to_vec()
}

#[test]
fn paged_decode_through_an_identity_view_launches_exactly_the_dense_attention() {
    let rig = Rig::new();
    let layer = rig.layer(KvCacheDtype::Bf16);
    let kv = rig.kv(KvCacheDtype::Bf16, false);
    let (bt, sl) = (rig.upload(&[0, 1, 2, 3, 4, 5, 0, 0]), rig.upload(&[50, 70]));
    let dense = Tables {
        block_table: bt,
        seq_lens: sl,
        max_blocks_per_seq: 4,
    };
    let io = DecodeIo::new(&rig);
    let dense_run = decode_launches(&rig, &layer, &kv, io, dense);
    assert!(!dense_run.is_empty());

    let sel = IdentitySelector::new();
    let lens = [50u32, 70];
    let s = seq_batch(bt, sl, 4, &lens);
    let c = SeamCtx {
        kv: layer.seam_kv_view(&kv, DIMS).unwrap(),
        ..seam_ctx(&rig.gpu, false)
    };
    sel.on_kv_written(&c, &s, &[], &[]).unwrap();
    let (counters, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &counters,
        ledger: &ledger,
    };
    let routed = route::decode(&sel, &c, &s, DevicePtr::NULL, DevicePtr::NULL, &env);
    let DecodeRoute::Run(view) = routed.unwrap() else {
        panic!("identity never computes attention itself");
    };
    assert_ne!(
        view.block_table, bt,
        "the view lives in selector-owned buffers"
    );
    let seam_run = decode_launches(&rig, &layer, &kv, io, view);
    let read_view = seam_run
        .iter()
        .any(|l| l.args.contains(&MockArg::Buffer(view.block_table)));
    assert!(read_view, "the attention read the view table");
    let map = [(view.block_table, bt), (view.seq_lens, sl)];
    assert_eq!(remap(&seam_run, &map), remap(&dense_run, &[]));
}

#[test]
fn a_view_changes_only_the_table_arguments_of_the_dense_launch() {
    let rig = Rig::new();
    let layer = rig.layer(KvCacheDtype::Bf16);
    let kv = rig.kv(KvCacheDtype::Bf16, false);
    let (bt, sl) = (rig.upload(&[0, 1, 2, 3, 4, 5, 6, 7]), rig.upload(&[64, 64]));
    let (vb, vs) = (rig.upload(&[0, 3, 0, 0, 4, 7, 0, 0]), rig.upload(&[32, 32]));
    let real = Tables {
        block_table: bt,
        seq_lens: sl,
        max_blocks_per_seq: 4,
    };
    let compact = Tables {
        block_table: vb,
        seq_lens: vs,
        max_blocks_per_seq: 4,
    };
    let io = DecodeIo::new(&rig);
    let dense_run = decode_launches(&rig, &layer, &kv, io, real);
    let view_run = decode_launches(&rig, &layer, &kv, io, compact);
    assert_eq!(
        remap(&view_run, &[(vb, bt), (vs, sl)]),
        remap(&dense_run, &[])
    );
}

#[test]
fn the_seam_serves_bf16_and_fp8_full_attention_only() {
    let rig = Rig::new();
    let kv = rig.kv(KvCacheDtype::Bf16, false);
    let v = rig
        .layer(KvCacheDtype::Bf16)
        .seam_kv_view(&kv, DIMS)
        .unwrap();
    assert_eq!(
        (v.dtype, v.block_size, v.kv_heads, v.head_dim),
        (KvDtypeTag::Bf16, BS, NKV, HD)
    );
    assert_eq!((v.kv_scale, v.v_scale), (1.0, 1.0));
    assert_eq!(v.k_pool, kv.k_pool_ptr(0));
    let fp8_kv = rig.kv(KvCacheDtype::Fp8, false);
    let f = rig
        .layer(KvCacheDtype::Fp8)
        .seam_kv_view(&fp8_kv, DIMS)
        .unwrap();
    assert_eq!(
        (f.dtype, f.kv_scale, f.v_scale),
        (KvDtypeTag::Fp8, 0.5, 2.0)
    );
    let mut sliding = rig.layer(KvCacheDtype::Bf16);
    sliding.sliding_window = Some(128);
    assert!(sliding.seam_kv_view(&kv, DIMS).is_none());
    let mut turbo = rig.layer(KvCacheDtype::Bf16);
    turbo.kv_dtype = KvCacheDtype::Turbo4;
    assert!(turbo.seam_kv_view(&kv, DIMS).is_none());
    let hss = rig.kv(KvCacheDtype::Bf16, true);
    let plain = rig.layer(KvCacheDtype::Bf16);
    assert!(plain.seam_kv_view(&hss, DIMS).is_none());
}
