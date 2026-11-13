// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the bounded view's wiring (port lane
//! L32) with a fake core: a turn's selection (needle picked, budget kept,
//! descriptor on the device), the identity gate, continuation, edit and retry
//! handling, history commit, the decode plan's fixed buffers and kernel
//! arguments, the prefill view of a continuing conversation, release on free,
//! and the zero-cost state while nothing is armed.

use std::ops::Range;
use std::sync::Mutex;

use anyhow::Result;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::config::{CliKnobs, KvViewConfig};
use super::core_api::{Geom, PickParams, ViewCore};
use super::metrics::{Counters, SkipReason};
use super::selector::KvViewSelector;
use super::view::{build_row, pages_of};
use super::{CallStream, ModelFacts, TurnInfo, TurnKind, ViewOutcome, ViewSpec};
use crate::layers::qwen3_attention::sparse_seam::{
    KvDtypeTag, KvLayerView, KvSelector, PrefillPlan, SeamCtx, SeamPhase, SeqBatch, SparsePlan,
};

const BS: u32 = 16;

/// Scores the fake core hands out (one per 128-token block). Shared by every
/// test; each one sets the same needle scores, so parallel tests agree.
pub(super) static SCORES: Mutex<Vec<f32>> = Mutex::new(Vec::new());

struct FakeCore;

impl ViewCore for FakeCore {
    fn record_written(&self, _: &SeamCtx, _: &SeqBatch, _: &[u32], _: &[u32]) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }
    fn arm(&self, _slot: u64, span: Range<u32>) -> Option<Range<u32>> {
        Some(span.start.max(span.end.saturating_sub(64))..span.end)
    }
    fn disarm(&self, _slot: u64) {}
    fn capture(
        &self,
        _: &SeamCtx,
        _: &SeqBatch,
        _: Option<DevicePtr>,
        _: DevicePtr,
    ) -> Result<usize> {
        Ok(0)
    }
    fn score(
        &self,
        gpu: &dyn GpuBackend,
        pages: &[u32],
        _: u64,
        out: DevicePtr,
        _: u64,
    ) -> Result<bool> {
        let n = pages.len().div_ceil(8);
        let s = SCORES.lock().unwrap().clone();
        let bytes: Vec<u8> = (0..n)
            .flat_map(|b| s.get(b).copied().unwrap_or(0.0).to_le_bytes())
            .collect();
        gpu.copy_h2d(&bytes, out)?;
        Ok(true)
    }
    fn pick(&self, scores: &[f32], p: &PickParams) -> Vec<u32> {
        let cfg = spark_runtime::kv_select::PickCfg {
            budget_blocks: p.budget_blocks as usize,
            sink_blocks: p.sink_blocks as usize,
            recent_blocks: p.recent_blocks as usize,
            mandatory: p.mandatory.clone(),
        };
        spark_runtime::kv_select::pick(scores, scores.len(), &cfg)
    }
    fn recall_mass(&self, scores: &[f32], picked: &[u32]) -> f64 {
        spark_runtime::kv_select::recall_mass(scores, picked)
    }
    fn device_bytes(&self) -> u64 {
        0
    }
}

fn fake(_: &dyn GpuBackend, _: Geom, _: u32, _: usize) -> Result<Box<dyn ViewCore>> {
    Ok(Box::new(FakeCore))
}

fn geom() -> Geom {
    Geom {
        block_size: BS,
        num_blocks: 4096,
        attn_layers: 2,
        kv_heads: 1,
        head_dim: 64,
        q_heads: 2,
        rotary_dim: 64,
        rope_theta: 1.0e6,
        max_blocks_per_seq: 1024,
    }
}

pub(super) fn selector(budget: u32, lambda: Option<&str>) -> KvViewSelector {
    let cli = CliKnobs {
        budget_tokens: Some(budget),
        sink_tokens: Some(128),
        recent_tokens: Some(256),
        query_tokens: Some(64),
    };
    let cfg = KvViewConfig::from_parts(cli, None, lambda).unwrap();
    KvViewSelector::with_core(cfg, fake, Box::leak(Box::new(Counters::new())))
}

fn facts(gpu: &MockGpuBackend) -> ModelFacts<'_> {
    ModelFacts {
        gpu,
        geom: geom(),
        heterogeneous: false,
    }
}

/// A prompt of `len` tokens whose token at `i` is `i`, with `edit` changed.
pub(super) fn prompt(len: u32, edit: Option<u32>) -> Vec<u32> {
    (0..len)
        .map(|i| if Some(i) == edit { 999_999 } else { i })
        .collect()
}

fn turn(kind: TurnKind, len: u32) -> TurnInfo {
    TurnInfo {
        kind,
        query: len - 200..len - 8,
        match_len: len - 6,
    }
}

pub(super) fn table(pages: u32, base: u32) -> Vec<u32> {
    (0..pages).map(|i| base + i).collect()
}

/// Begin a one-stream call and select at its end.
pub(super) fn run_turn(
    sel: &KvViewSelector,
    gpu: &MockGpuBackend,
    slot: u64,
    p: &[u32],
    kind: TurnKind,
    tbl: &[u32],
) -> ViewOutcome {
    let len = p.len() as u32;
    let st = CallStream {
        slot,
        session: 77,
        prompt: p,
        chunk_start: 0,
        chunk_end: len,
        turn: turn(kind, len),
    };
    sel.begin_call(&facts(gpu), std::slice::from_ref(&st))
        .unwrap();
    sel.end_call();
    let spec = ViewSpec {
        prompt: p,
        block_table: tbl,
        seq_len: len,
        prefill_stream: 7,
    };
    sel.set_kv_view(gpu, slot, Some(spec)).unwrap()
}

/// 32 blocks of scores with a needle at block 5.
pub(super) fn needle_scores() -> Vec<f32> {
    let mut s = vec![0.01f32; 32];
    s[5] = 5.0;
    s
}

#[test]
fn a_turn_selects_the_needle_within_the_budget() {
    *SCORES.lock().unwrap() = needle_scores();
    let gpu = MockGpuBackend::new();
    let sel = selector(1024, None);
    let p = prompt(4096, None);
    let tbl = table(256, 1000);
    let out = run_turn(&sel, &gpu, 3, &p, TurnKind::NewUser, &tbl);
    let ViewOutcome::Selected {
        context_tokens,
        view_tokens,
        recall_mass,
    } = out
    else {
        panic!("{out:?}");
    };
    assert_eq!(context_tokens, 4096);
    assert!(view_tokens <= 1024 + 128, "{view_tokens}");
    assert!(recall_mass > 0.5, "{recall_mass}");
    let inner = sel.inner_for_test();
    let live: Vec<_> = inner.live.iter().flatten().collect();
    assert_eq!(live.len(), 1);
    let d = &live[0].desc;
    // The needle block's pages (40..48), the sink block's (0..8) are selected.
    assert!(d.shape.sel_pages.contains(&40) && d.shape.sel_pages.contains(&47));
    assert!(d.shape.sel_pages.contains(&0));
    // The view of the real row is the selection plus the tail.
    let (pages, l) = build_row(&tbl, 4096, BS, &[Some(d)]);
    assert_eq!(l, view_tokens);
    assert_eq!(pages.len() as u32, pages_of(l, BS));
    // Sink 1 + recent 2 (the question's block is the last) + the needle +
    // four ties toward the newer blocks = the 8-block budget, 1,024 tokens.
    assert_eq!(view_tokens, 1024);
    assert!(
        inner.convs.contains_key(&77),
        "the conversation remembers the selection"
    );
}

#[test]
fn identity_gate_a_context_within_the_budget_gets_no_view() {
    *SCORES.lock().unwrap() = needle_scores();
    let gpu = MockGpuBackend::new();
    let sel = selector(8192, None);
    let p = prompt(4096, None);
    let tbl = table(256, 1000);
    let out = run_turn(&sel, &gpu, 1, &p, TurnKind::NewUser, &tbl);
    assert_eq!(out, ViewOutcome::Skipped(SkipReason::ShortContext));
    let inner = sel.inner_for_test();
    assert!(inner.live.iter().all(Option::is_none));
    // With no view the device tables equal the real ones.
    let (pages, l) = build_row(&tbl, 4096, BS, &[]);
    assert_eq!((pages, l), (tbl.clone(), 4096));
}

#[test]
fn continuation_keeps_the_selection_and_an_edit_drops_it() {
    *SCORES.lock().unwrap() = needle_scores();
    let gpu = MockGpuBackend::new();
    let sel = selector(1024, None);
    let first = prompt(4096, None);
    let out = run_turn(&sel, &gpu, 3, &first, TurnKind::NewUser, &table(256, 1000));
    assert!(matches!(out, ViewOutcome::Selected { .. }), "{out:?}");
    let shape = sel.inner_for_test().convs[&77].shape.clone();
    // The next request repeats the prefix and appends a tool result.
    let second = prompt(4400, None);
    let out = run_turn(
        &sel,
        &gpu,
        4,
        &second,
        TurnKind::Continuation,
        &table(275, 5000),
    );
    assert!(matches!(out, ViewOutcome::Kept { .. }), "{out:?}");
    let inner = sel.inner_for_test();
    let kept = inner.live.iter().flatten().find(|l| l.owner == 4).unwrap();
    assert_eq!(kept.desc.shape, shape);
    assert_eq!(kept.desc.a0, 5000, "bound to the new sequence's blocks");
    drop(inner);
    // An edited prefix starts over.
    let edited = prompt(4400, Some(10));
    let out = run_turn(
        &sel,
        &gpu,
        5,
        &edited,
        TurnKind::Continuation,
        &table(275, 9000),
    );
    assert!(matches!(out, ViewOutcome::Selected { .. }), "{out:?}");
}

#[test]
fn history_commits_only_when_the_conversation_moves_on() {
    *SCORES.lock().unwrap() = needle_scores();
    let gpu = MockGpuBackend::new();
    let sel = selector(1024, Some("0.5"));
    let p = prompt(4096, None);
    run_turn(&sel, &gpu, 3, &p, TurnKind::NewUser, &table(256, 1000));
    assert_eq!(sel.inner_for_test().convs[&77].history.rounds, 0);
    assert!(sel.inner_for_test().convs[&77].pending.is_some());
    // A retry of the same round does not commit it.
    run_turn(&sel, &gpu, 3, &p, TurnKind::NewUser, &table(256, 1000));
    assert_eq!(sel.inner_for_test().convs[&77].history.rounds, 0);
    // The next question does.
    let next = prompt(4600, None);
    run_turn(&sel, &gpu, 3, &next, TurnKind::NewUser, &table(288, 1000));
    assert_eq!(sel.inner_for_test().convs[&77].history.rounds, 1);
}

pub(super) fn ctx(gpu: &MockGpuBackend, phase: SeamPhase, in_graph: bool) -> SeamCtx<'_> {
    SeamCtx {
        gpu,
        attn_layer: 0,
        kv: KvLayerView {
            k_pool: DevicePtr(0x5000),
            v_pool: DevicePtr(0x6000),
            dtype: KvDtypeTag::Bf16,
            block_size: BS,
            kv_heads: 1,
            head_dim: 64,
            kv_scale: 1.0,
            v_scale: 1.0,
            block_stride_bytes: 4096,
            cache_stride: 0,
        },
        stream: 0,
        in_graph,
        phase,
        num_q_heads: 2,
        inv_sqrt_d: 0.125,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batch<'a>(
    n: usize,
    table: DevicePtr,
    lens: DevicePtr,
    host_lens: &'a [u32],
    first: &'a [u32],
    new: &'a [u32],
    tables: &'a [&'a [u32]],
    ids: &'a [u64],
) -> SeqBatch<'a> {
    SeqBatch {
        n,
        block_table: table,
        seq_lens: lens,
        max_blocks_per_seq: 1024,
        host_seq_lens: host_lens,
        seq_ids: ids,
        rows_per_seq: 1,
        q_row_stride: 128,
        q_rows: n,
        first_pos: first,
        new_tokens: new,
        host_block_tables: tables,
        block_table_ptrs: DevicePtr::NULL,
    }
}

#[test]
fn decode_plans_are_views_over_fixed_buffers() {
    *SCORES.lock().unwrap() = needle_scores();
    let gpu = MockGpuBackend::new();
    let sel = selector(1024, None);
    // Before any prefill call nothing is allocated: plans are dense.
    let real = gpu.alloc(8 * 1024 * 4).unwrap();
    let lens = gpu.alloc(8 * 4).unwrap();
    let hl = [4097u32];
    let (f, n) = ([4096u32], [1u32]);
    let s = batch(1, real, lens, &hl, &f, &n, &[], &[]);
    assert!(matches!(
        sel.plan(&ctx(&gpu, SeamPhase::Decode, false), &s, DevicePtr::NULL)
            .unwrap(),
        SparsePlan::Dense
    ));
    run_turn(
        &sel,
        &gpu,
        3,
        &prompt(4096, None),
        TurnKind::NewUser,
        &table(256, 1000),
    );
    let launches = gpu.launches_snapshot().len();
    let mut seen = Vec::new();
    for in_graph in [true, false, true] {
        let plan = sel
            .plan(&ctx(&gpu, SeamPhase::Decode, in_graph), &s, DevicePtr::NULL)
            .unwrap();
        let SparsePlan::View {
            block_table,
            seq_lens,
            max_blocks_per_seq,
        } = plan
        else {
            panic!("expected a view");
        };
        assert_eq!(max_blocks_per_seq, 1024);
        seen.push((block_table, seq_lens));
    }
    assert!(
        seen.windows(2).all(|w| w[0] == w[1]),
        "the same buffers on every step"
    );
    let all = gpu.launches_snapshot();
    assert_eq!(all.len(), launches + 3, "one table build per plan");
    let last = all.last().unwrap();
    assert_eq!(last.grid, [1, 1, 1]);
    assert_eq!(last.args[0], MockArg::Buffer(real));
    assert_eq!(last.args[1], MockArg::Buffer(lens));
    assert_eq!(last.args[10], MockArg::Buffer(seen[0].0));
    assert_eq!(last.args[11], MockArg::Buffer(seen[0].1));
}

#[test]
fn a_continuing_conversation_prefills_against_its_view() {
    *SCORES.lock().unwrap() = needle_scores();
    let gpu = MockGpuBackend::new();
    let sel = selector(1024, None);
    run_turn(
        &sel,
        &gpu,
        3,
        &prompt(4096, None),
        TurnKind::NewUser,
        &table(256, 1000),
    );
    let shape = sel.inner_for_test().convs[&77].shape.clone();
    // Turn two arrives with 2,000 new tokens; its second chunk starts at 5,096.
    let p = prompt(6096, None);
    let st = CallStream {
        slot: 6,
        session: 77,
        prompt: &p,
        chunk_start: 5096,
        chunk_end: 6096,
        turn: turn(TurnKind::Continuation, 6096),
    };
    sel.begin_call(&facts(&gpu), std::slice::from_ref(&st))
        .unwrap();
    let row = table(pages_of(6096, BS), 20_000);
    let tables: [&[u32]; 1] = [&row];
    let (hl, f, n) = ([6096u32], [5096u32], [1000u32]);
    let dev_table = gpu.alloc(4096).unwrap();
    let s = batch(1, dev_table, dev_table, &hl, &f, &n, &tables, &[6]);
    let plan = sel
        .plan_prefill(&ctx(&gpu, SeamPhase::Prefill, false), &s, DevicePtr::NULL)
        .unwrap();
    let PrefillPlan::View {
        block_table,
        kv_len,
    } = plan
    else {
        panic!("expected a prefill view");
    };
    assert_eq!(kv_len, 6096 - shape.drop_tokens(BS));
    let (pages, l) = shape.compose(&row, 6096, BS).unwrap();
    assert_eq!(l, kv_len);
    let mut up = vec![0u8; pages.len() * 4];
    gpu.copy_d2h(block_table, &mut up).unwrap();
    let got: Vec<u32> = up
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect();
    assert_eq!(got, pages);
    // A chunk that starts before the tail page stays dense.
    let early = [shape.tail_page * BS - 1];
    let s2 = batch(1, dev_table, dev_table, &hl, &early, &n, &tables, &[6]);
    assert!(matches!(
        sel.plan_prefill(&ctx(&gpu, SeamPhase::Prefill, false), &s2, DevicePtr::NULL)
            .unwrap(),
        PrefillPlan::Dense
    ));
    sel.end_call();
}
