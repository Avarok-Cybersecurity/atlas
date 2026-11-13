// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the window desk's Quest scorer (port lane L30, request R3 of lane L36):
//! the bound against the kernels' CPU reference, the cross-layer ranking and its tie rule, the
//! expansion of ranked pages into blocks, the host store's accounting and eviction, and, on the
//! mock backend, the whole path: demote copies the page summaries, the decode query is captured
//! into the fixed buffers and read back, and `rank` answers with the planted page first.

use std::sync::Arc;

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::page_summary::{LayerGeom, SummaryLayout};

use super::mock_tests::SERIAL;
use super::reference::{self, f16_rd, f16_ru};
use super::scorer_rank::{self as rk, LayerQuery, PageRecord};
use super::scorer_store::HostStore;
use super::{QuestConfig, QuestScorer, QuestSelector, budget};
use crate::layers::qwen3_attention::sparse_seam::{
    KvDtypeTag, KvLayerView, SeamCtx, SeamPhase, SeqBatch,
};
use crate::port::kv_window::selector::{PageScorer, ScoreRequest};

const KVH: u32 = 1;
const HD: u32 = 32;
const NQ: u32 = 2;

/// One layer's record of one page: kmin is -1 everywhere, kmax is 1 except channel 0.
fn record(kmax0: f32) -> PageRecord {
    let mut v: Vec<u16> = vec![f16_rd(-1.0); HD as usize];
    v.extend((0..HD).map(|c| f16_ru(if c == 0 { kmax0 } else { 1.0 })));
    Arc::from(v)
}

fn geom() -> Vec<LayerGeom> {
    vec![LayerGeom {
        kv_heads: KVH,
        head_dim: HD,
    }]
}

fn query_channel0(num_q_heads: usize) -> LayerQuery {
    let mut row = vec![0.0f32; num_q_heads * HD as usize];
    for h in 0..num_q_heads {
        row[h * HD as usize] = 1.0;
    }
    LayerQuery {
        layer: 0,
        num_q_heads,
        rows: vec![row],
    }
}

#[test]
fn the_bound_matches_the_kernel_reference() {
    let q: Vec<f32> = (0..64)
        .map(|i| ((i * 37 % 19) as f32 - 9.0) / 4.0)
        .collect();
    let kmin: Vec<u16> = (0..64)
        .map(|i| f16_rd(-((i % 7) as f32) / 3.0 - 0.1))
        .collect();
    let kmax: Vec<u16> = (0..64)
        .map(|i| f16_ru(((i * 5 % 11) as f32) / 3.0 + 0.1))
        .collect();
    let got = f64::from(rk::bound(&q, &kmin, &kmax));
    let want = reference::bound(&q, &kmin, &kmax);
    assert!(
        (got - want).abs() <= 1e-3 * want.abs() + 1e-4,
        "{got} vs {want}"
    );
}

#[test]
fn layers_rank_the_page_with_the_largest_bound_first() {
    let layers = geom();
    // Bounds with q = channel 0 are the pages' kmax0: page 1 = 5, page 2 = 2, page 3 = 1.
    let pages = vec![(3u32, record(1.0)), (1, record(5.0)), (2, record(2.0))];
    let ranked = rk::rank_pages(&layers, &[query_channel0(2)], &pages, false);
    assert_eq!(ranked, vec![1, 2, 3]);
    // The sum over query heads keeps the order (every head sees the same query here).
    assert_eq!(
        rk::rank_pages(&layers, &[query_channel0(2)], &pages, true),
        vec![1, 2, 3]
    );
}

#[test]
fn equal_scores_go_to_the_newer_page_and_unscorable_pages_are_dropped() {
    let layers = geom();
    let pages = vec![(4u32, record(2.0)), (9, record(2.0))];
    assert_eq!(
        rk::rank_pages(&layers, &[query_channel0(2)], &pages, false),
        vec![9, 4]
    );
    // A query of a layer the geometry does not have, or a row too short for its heads, scores nothing.
    let mut stray = query_channel0(2);
    stray.layer = 5;
    assert!(rk::rank_pages(&layers, &[stray], &pages, false).is_empty());
    let short = LayerQuery {
        layer: 0,
        num_q_heads: 2,
        rows: vec![vec![1.0; 8]],
    };
    assert!(rk::rank_pages(&layers, &[short], &pages, false).is_empty());
    assert!(rk::rank_pages(&layers, &[], &pages, false).is_empty());
}

#[test]
fn blocks_expand_ranked_pages_and_stop_at_k() {
    let cands = [0u32, 1, 2, 3, 8, 9, 10, 11, 12];
    assert_eq!(
        rk::blocks_of(&[2, 0], &cands, 4, 6),
        vec![8, 9, 10, 11, 0, 1]
    );
    assert_eq!(
        rk::blocks_of(&[2, 0], &cands, 4, 100),
        vec![8, 9, 10, 11, 0, 1, 2, 3]
    );
    assert_eq!(rk::blocks_of(&[2, 0], &cands, 4, 0), Vec::<u32>::new());
    // A page with a single candidate block (block 12 is page 3) still ranks.
    assert_eq!(rk::blocks_of(&[3], &cands, 4, 8), vec![12]);
    assert!(rk::blocks_of(&[7], &cands, 4, 8).is_empty());
}

#[test]
fn the_host_store_counts_bytes_evicts_the_oldest_sequence_and_resets_per_build() {
    let mut s = HostStore::new(100);
    s.reset_for(1);
    s.put(1, 0, Some(Arc::from(vec![0u16; 30])));
    s.put(2, 0, Some(Arc::from(vec![0u16; 30])));
    assert_eq!(s.bytes(), 120);
    s.evict(2);
    assert_eq!(s.bytes(), 60);
    assert!(
        s.records(1, &[0]).is_empty(),
        "the older sequence was dropped"
    );
    assert_eq!(s.records(2, &[0, 5]).len(), 1);
    // Replacing a record keeps the count; forgetting one releases it.
    s.put(2, 0, Some(Arc::from(vec![0u16; 10])));
    assert_eq!(s.bytes(), 20);
    s.put(2, 0, None);
    assert_eq!(s.bytes(), 0);
    s.put(2, 1, Some(Arc::from(vec![0u16; 10])));
    s.reset_for(2);
    assert_eq!(s.bytes(), 0);
    assert!(s.records(2, &[1]).is_empty());
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
        block_stride_bytes: 16 * 32 * 2,
        cache_stride: 16 * 32,
    }
}

fn ctx<'a>(gpu: &'a dyn GpuBackend, pool: DevicePtr, in_graph: bool) -> SeamCtx<'a> {
    SeamCtx {
        gpu,
        attn_layer: 0,
        kv: kv(pool),
        stream: 7,
        in_graph,
        phase: SeamPhase::Decode,
        num_q_heads: NQ,
        inv_sqrt_d: 0.125,
    }
}

fn request<'a>(block0: u32, candidates: &'a [u32]) -> ScoreRequest<'a> {
    ScoreRequest {
        block0,
        table: &[],
        kv_tokens: 40_000,
        block_size: 16,
        candidates,
        stream: 7,
    }
}

#[test]
fn demote_copies_summaries_decode_captures_the_query_and_rank_orders_the_pages() {
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    budget::publish(SummaryLayout::new(64, 16, geom()).unwrap());
    let gpu = MockGpuBackend::new();
    let pool = gpu.alloc(2 << 20).unwrap();
    let sel = Arc::new(QuestSelector::new(QuestConfig::default()));
    let scorer = QuestScorer::new(sel.clone());
    let block0 = 3u32;

    let bt = gpu.alloc(64 * 4).unwrap();
    let lens = gpu.alloc(4).unwrap();
    let q = gpu.alloc((NQ * HD * 2) as usize).unwrap();
    gpu.copy_h2d(&(block0 as i32).to_le_bytes(), bt).unwrap();
    gpu.copy_h2d(&40_000i32.to_le_bytes(), lens).unwrap();
    // Channel 0 of every query head is 1.0 (BF16 0x3F80), the rest 0.
    let mut qraw = vec![0u8; (NQ * HD * 2) as usize];
    for h in 0..NQ as usize {
        qraw[h * HD as usize * 2] = 0x80;
        qraw[h * HD as usize * 2 + 1] = 0x3F;
    }
    gpu.copy_h2d(&qraw, q).unwrap();
    let (first, new, ext) = ([39_999u32], [1u32], [40_000u32]);
    let s = SeqBatch {
        n: 1,
        block_table: bt,
        seq_lens: lens,
        max_blocks_per_seq: 64,
        host_seq_lens: &ext,
        seq_ids: &[],
        rows_per_seq: 1,
        q_row_stride: NQ * HD,
        q_rows: 1,
        first_pos: &first,
        new_tokens: &new,
        host_block_tables: &[],
        block_table_ptrs: DevicePtr::NULL,
    };

    // A captured step before any eager hook has nothing to write into: no query, no error.
    let cg = ctx(&gpu, pool, true);
    scorer.on_query(&cg, &s, None, q).unwrap();
    assert!(scorer.read_queries(&gpu, 7, block0).unwrap().is_empty());

    // The first eager hook allocates the summary arena (through the selector) and the query buffers.
    let c = ctx(&gpu, pool, false);
    scorer.on_kv_written(&c, &s, &first, &new).unwrap();
    scorer.on_query(&c, &s, None, q).unwrap();
    // A pre-RoPE query (a prefill site) is ignored, whatever it holds.
    scorer.on_query(&c, &s, Some(q), q).unwrap();
    let rows = scorer.read_queries(&gpu, 7, block0).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].num_q_heads, NQ as usize);
    assert_eq!(rows[0].rows.len(), 1);
    assert_eq!(rows[0].rows[0][0], 1.0);
    assert_eq!(rows[0].rows[0][1], 0.0);
    assert!(
        scorer.read_queries(&gpu, 7, 99).unwrap().is_empty(),
        "another sequence has none"
    );

    // The arena holds the pages' summaries as the kernels would have left them: leading blocks
    // 20, 24 and 28 carry kmax0 = 1, 5 and 2, every mask complete.
    let (base, layout) = sel
        .with_summaries(&gpu, |a, _| Ok((a.summary.base, a.summary.layout.clone())))
        .unwrap()
        .expect("the arena exists after the first eager hook");
    for (phys, kmax0) in [(20u32, 1.0f32), (24, 5.0), (28, 2.0)] {
        let mut vals = Vec::new();
        for h in record(kmax0).iter() {
            vals.extend_from_slice(&h.to_le_bytes());
        }
        gpu.copy_h2d(&vals, base.offset(layout.vals_offset(0, phys, 0) as usize))
            .unwrap();
        let mut meta = vec![0u8; 32];
        meta[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        gpu.copy_h2d(&meta, base.offset(layout.meta_offset(0, phys, 0) as usize))
            .unwrap();
    }
    // Logical blocks 4..16 leave the GPU; blocks 4, 8 and 12 lead their pages.
    let demoted: Vec<(u32, u32)> = (4..16u32).map(|i| (i, 20 + (i - 4))).collect();
    scorer.on_demote(&gpu, 7, block0, &demoted).unwrap();
    // An unfolded page (empty mask) is not saved: block 32 leads page 16 of another sequence.
    scorer.on_demote(&gpu, 7, 40, &[(16, 32)]).unwrap();

    let cands: Vec<u32> = (4..16).collect();
    let all = scorer.rank(&gpu, &request(block0, &cands), 12).unwrap();
    assert_eq!(all, vec![8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]);
    let top = scorer.rank(&gpu, &request(block0, &cands), 4).unwrap();
    assert_eq!(top, vec![8, 9, 10, 11]);
    // Only the candidates are ever returned.
    let some = scorer
        .rank(&gpu, &request(block0, &[4, 5, 12]), 12)
        .unwrap();
    assert_eq!(some, vec![12, 4, 5]);

    // No summary for the sequence, or a summary but no captured query: nothing, not an error.
    assert!(
        scorer
            .rank(&gpu, &request(99, &cands), 8)
            .unwrap()
            .is_empty()
    );
    assert!(
        scorer
            .rank(&gpu, &request(40, &[16, 17]), 8)
            .unwrap()
            .is_empty()
    );
    scorer.on_demote(&gpu, 7, 7, &demoted).unwrap();
    assert!(
        scorer
            .rank(&gpu, &request(7, &cands), 8)
            .unwrap()
            .is_empty(),
        "no query for block0 7"
    );

    // A later step with another batch replaces the captured rows: here a different sequence.
    gpu.copy_h2d(&9i32.to_le_bytes(), bt).unwrap();
    scorer.on_query(&c, &s, None, q).unwrap();
    assert!(scorer.read_queries(&gpu, 7, block0).unwrap().is_empty());
    assert_eq!(scorer.read_queries(&gpu, 7, 9).unwrap().len(), 1);
}

#[test]
fn the_scorer_reserves_the_view_arena_and_the_query_buffers() {
    let scorer = QuestScorer::new(Arc::new(QuestSelector::new(QuestConfig::default())));
    assert_eq!(scorer.name(), "quest");
    assert_eq!(
        scorer.reserved_bytes(),
        (super::arena::VIEW_ARENA_BYTES + super::scorer::QUERY_ARENA_BYTES) as u64
    );
}
