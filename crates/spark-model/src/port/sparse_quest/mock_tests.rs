// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the Quest selector's (port lane L30) mock-backend tests:
//! the launch sequence of a verify step, the graph rule (a captured step reuses
//! the eager step's View buffers), the chunk-0 prefill upload, and every
//! refusal (capture before allocation, an absent module, a foreign K pool, a
//! View larger than the arena), each of which must plan `Dense`.

use std::sync::{Arc, Mutex};

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::page_summary::{LayerGeom, SummaryLayout};

use super::{QuestConfig, QuestSelector, Refusal, budget};
use crate::layers::qwen3_attention::sparse_seam::{
    KvDtypeTag, KvLayerView, KvSelector, SeamCtx, SeamPhase, SeqBatch, SparsePlan,
};

/// The tests share the process-wide summary geometry; run them one at a time.
pub(super) static SERIAL: Mutex<()> = Mutex::new(());

const KVH: u32 = 2;
const HD: u32 = 64;

fn serial() -> std::sync::MutexGuard<'static, ()> {
    let g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let layers = vec![
        LayerGeom {
            kv_heads: KVH,
            head_dim: HD,
        };
        2
    ];
    budget::publish(SummaryLayout::new(64, 16, layers).unwrap());
    g
}

fn kv(pool: DevicePtr) -> KvLayerView {
    KvLayerView {
        k_pool: pool,
        v_pool: pool.offset(1 << 20),
        dtype: KvDtypeTag::Bf16,
        block_size: 16,
        kv_heads: KVH,
        head_dim: HD,
        kv_scale: 1.0,
        v_scale: 1.0,
        block_stride_bytes: 16 * 2 * 64 * 2,
        cache_stride: 16 * 2 * 64,
    }
}

fn ctx<'a>(
    gpu: &'a dyn GpuBackend,
    pool: DevicePtr,
    phase: SeamPhase,
    in_graph: bool,
) -> SeamCtx<'a> {
    SeamCtx {
        gpu,
        attn_layer: 0,
        kv: kv(pool),
        stream: 7,
        in_graph,
        phase,
        num_q_heads: 4,
        inv_sqrt_d: 0.125,
    }
}

struct Rows {
    bt: DevicePtr,
    lens: DevicePtr,
    host_lens: Vec<u32>,
    first: Vec<u32>,
    new: Vec<u32>,
}

fn rows(gpu: &MockGpuBackend, n: usize, r: usize) -> Rows {
    Rows {
        bt: gpu.alloc(n * 64 * 4).unwrap(),
        lens: gpu.alloc(n * 4).unwrap(),
        host_lens: (0..n as u32).map(|i| 40_000 + i % r as u32).collect(),
        first: vec![40_000; n / r],
        new: vec![r as u32; n / r],
    }
}

fn batch<'a>(x: &'a Rows, n: usize, r: usize, mbps: u32) -> SeqBatch<'a> {
    SeqBatch {
        n,
        block_table: x.bt,
        seq_lens: x.lens,
        max_blocks_per_seq: mbps,
        host_seq_lens: &x.host_lens,
        seq_ids: &[],
        rows_per_seq: r,
        q_row_stride: 4 * HD,
        q_rows: n,
        first_pos: &x.first,
        new_tokens: &x.new,
        host_block_tables: &[],
        block_table_ptrs: DevicePtr::NULL,
    }
}

fn view(p: SparsePlan) -> (DevicePtr, DevicePtr, u32) {
    match p {
        SparsePlan::View {
            block_table,
            seq_lens,
            max_blocks_per_seq,
        } => (block_table, seq_lens, max_blocks_per_seq),
        SparsePlan::Dense => panic!("expected a View, got Dense"),
        SparsePlan::Custom(_) => panic!("expected a View, got Custom"),
    }
}

#[test]
fn decode_and_verify_plan_views_over_fixed_buffers() {
    let _g = serial();
    let gpu = MockGpuBackend::new();
    let pool = gpu.alloc(2 << 20).unwrap();
    let sel = Arc::new(QuestSelector::new(QuestConfig::default()));
    let x = rows(&gpu, 8, 4);
    let q = gpu.alloc(8 * 4 * 64 * 2).unwrap();
    // Eager verify step: the update, then score and compaction.
    let c = ctx(&gpu, pool, SeamPhase::Verify, false);
    sel.on_kv_written(&c, &batch(&x, 8, 4, 64), &x.first, &x.new)
        .unwrap();
    let v1 = view(sel.plan(&c, &batch(&x, 8, 4, 64), q).unwrap());
    assert_eq!(v1.2, 64, "the View keeps the dense row stride");
    // The same shape captured: the same buffers (graph rule).
    let cg = ctx(&gpu, pool, SeamPhase::Verify, true);
    sel.on_kv_written(&cg, &batch(&x, 8, 4, 64), &x.first, &x.new)
        .unwrap();
    let v2 = view(sel.plan(&cg, &batch(&x, 8, 4, 64), q).unwrap());
    assert_eq!(v1, v2);
    let launches = gpu.launches_snapshot();
    let update: Vec<_> = launches.iter().filter(|l| l.block[0] == 128).collect();
    let score: Vec<_> = launches.iter().filter(|l| l.block[0] == 256).collect();
    let compact: Vec<_> = launches.iter().filter(|l| l.block[0] == 1024).collect();
    assert_eq!(update.len(), 2);
    assert_eq!(
        update[0].grid,
        [2, KVH, 2],
        "4 verify rows touch at most 2 pages; 2 groups"
    );
    assert_eq!(score.len(), 2);
    assert_eq!(
        score[0].grid,
        [1, 2, 1],
        "64 blocks = 16 pages: one CTA of 32 pages per group"
    );
    assert_eq!(compact.len(), 2);
    assert_eq!(compact[0].grid, [2, 1, 1]);
    assert!(launches.iter().all(|l| l.stream == 7));
    assert!(!sel.refused(Refusal::CaptureBeforeAlloc));
}

#[test]
fn a_capture_before_any_eager_hook_plans_dense() {
    let _g = serial();
    let gpu = MockGpuBackend::new();
    let pool = gpu.alloc(2 << 20).unwrap();
    let sel = QuestSelector::new(QuestConfig::default());
    let x = rows(&gpu, 1, 1);
    let c = ctx(&gpu, pool, SeamPhase::Decode, true);
    assert!(matches!(
        sel.plan(&c, &batch(&x, 1, 1, 64), x.bt).unwrap(),
        SparsePlan::Dense
    ));
    assert!(sel.refused(Refusal::CaptureBeforeAlloc));
    assert!(gpu.launches_snapshot().is_empty());
}

#[test]
fn an_absent_module_or_a_foreign_pool_stays_dense() {
    let _g = serial();
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(super::launch::MODULE);
    let pool = gpu.alloc(2 << 20).unwrap();
    let sel = QuestSelector::new(QuestConfig::default());
    let x = rows(&gpu, 1, 1);
    let c = ctx(&gpu, pool, SeamPhase::Decode, false);
    assert!(matches!(
        sel.plan(&c, &batch(&x, 1, 1, 64), x.bt).unwrap(),
        SparsePlan::Dense
    ));
    assert!(sel.refused(Refusal::NoModule));
    assert!(gpu.launches_snapshot().is_empty());
    assert!(
        !gpu.kernel_lookups_snapshot()
            .iter()
            .any(|(m, _)| m == super::launch::MODULE),
        "an absent module issues no lookup"
    );

    let gpu = MockGpuBackend::new();
    let pool = gpu.alloc(2 << 20).unwrap();
    let other = gpu.alloc(2 << 20).unwrap();
    let sel = QuestSelector::new(QuestConfig::default());
    let x = rows(&gpu, 1, 1);
    let q = gpu.alloc(4 * 64 * 2).unwrap();
    let own = sel.plan(
        &ctx(&gpu, pool, SeamPhase::Decode, false),
        &batch(&x, 1, 1, 64),
        q,
    );
    view(own.unwrap());
    let foreign = sel
        .plan(
            &ctx(&gpu, other, SeamPhase::Decode, false),
            &batch(&x, 1, 1, 64),
            q,
        )
        .unwrap();
    assert!(matches!(foreign, SparsePlan::Dense));
    assert!(sel.refused(Refusal::ForeignPool));
}

#[test]
fn a_view_larger_than_the_arena_stays_dense() {
    let _g = serial();
    let gpu = MockGpuBackend::new();
    let pool = gpu.alloc(2 << 20).unwrap();
    let sel = QuestSelector::new(QuestConfig::default());
    let x = rows(&gpu, 1, 1);
    let c = ctx(&gpu, pool, SeamPhase::Decode, false);
    let huge = (super::arena::TABLE_BYTES / 4 + 1) as u32;
    assert!(matches!(
        sel.plan(&c, &batch(&x, 1, 1, huge), x.bt).unwrap(),
        SparsePlan::Dense
    ));
    assert!(sel.refused(Refusal::ViewTooLarge));
}

#[test]
fn a_chunk0_prefill_uploads_its_host_table_and_folds_its_range() {
    let _g = serial();
    let gpu = MockGpuBackend::new();
    let pool = gpu.alloc(2 << 20).unwrap();
    let sel = QuestSelector::new(QuestConfig::default());
    let table: Vec<u32> = (0..20).collect();
    let tables = [table.as_slice()];
    let (first, new, ext) = ([0u32], [300u32], [300u32]);
    let s = SeqBatch {
        n: 1,
        block_table: DevicePtr::NULL,
        seq_lens: DevicePtr::NULL,
        max_blocks_per_seq: 0,
        host_seq_lens: &ext,
        seq_ids: &[],
        rows_per_seq: 1,
        q_row_stride: 4 * HD,
        q_rows: 300,
        first_pos: &first,
        new_tokens: &new,
        host_block_tables: &tables,
        block_table_ptrs: DevicePtr::NULL,
    };
    let c = ctx(&gpu, pool, SeamPhase::Prefill, false);
    sel.on_kv_written(&c, &s, &first, &new).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    // 300 tokens from position 0 touch at most (299 / 64) + 2 = 6 page CTAs.
    assert_eq!(l[0].grid, [6, KVH, 1]);
    let plan = sel.plan(&c, &s, DevicePtr::NULL).unwrap();
    assert!(matches!(plan, SparsePlan::Dense), "prefill attends densely");
}

#[test]
fn the_selector_reserves_its_view_arena() {
    let sel = QuestSelector::new(QuestConfig::default());
    assert_eq!(sel.name(), "quest");
    assert_eq!(sel.reserved_bytes(), super::arena::VIEW_ARENA_BYTES as u64);
}
