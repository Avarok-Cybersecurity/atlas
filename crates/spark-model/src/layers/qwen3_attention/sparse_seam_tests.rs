// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracles of the sparse-attention seam's decision half
//! (port lane L29): the registry, the step publication, plan routing and the
//! graph-rule checks. Nothing here touches the serve's global selector slot;
//! each test builds its own registry, counters and ledger. The recording
//! selector defined here is shared with the other seam test modules.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::metrics::{Counters, PlanKind};
use super::registry::{Registry, StepInfo, current_step, push_step};
use super::route::{self, DecodeRoute, RouteEnv, Tables, ViewLedger};
use super::{
    CustomAttention, KvDtypeTag, KvLayerView, KvSelector, PrefillPlan, SeamCtx, SeamPhase,
    SeqBatch, SparsePlan,
};

pub(super) type PlanFn = Box<dyn Fn() -> SparsePlan + Send + Sync>;
pub(super) type PrefillFn = Box<dyn Fn() -> PrefillPlan + Send + Sync>;

/// A selector whose plans are scripted and whose hook calls are recorded,
/// with the fields of the `SeqBatch` each call saw.
pub(super) struct Recorder {
    pub(super) name: &'static str,
    pub(super) plan: PlanFn,
    pub(super) prefill: PrefillFn,
    pub(super) reserved: u64,
    /// The per-KV-page sidecar bytes `reserved_bytes_per_kv_block` reports.
    pub(super) per_block: u64,
    /// The pool geometry `on_pool_geometry` last received.
    pub(super) pool: Mutex<Option<super::PoolGeometry>>,
    pub(super) log: Mutex<Vec<String>>,
}

impl Recorder {
    pub(super) fn new(name: &'static str, plan: PlanFn) -> Self {
        Self {
            name,
            plan,
            prefill: Box::new(|| PrefillPlan::Dense),
            reserved: 0,
            per_block: 0,
            pool: Mutex::new(None),
            log: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn dense(name: &'static str) -> Self {
        Self::new(name, Box::new(|| SparsePlan::Dense))
    }

    pub(super) fn calls(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }

    fn note(&self, hook: &str, c: &SeamCtx, s: &SeqBatch, extra: String) {
        self.log.lock().unwrap().push(format!(
            "{hook} phase={:?} graph={} n={} rows_per_seq={} lens={:?} ids={:?} first={:?} \
             new={:?} tables={}{extra}",
            c.phase,
            c.in_graph,
            s.n,
            s.rows_per_seq,
            s.host_seq_lens,
            s.seq_ids,
            s.first_pos,
            s.new_tokens,
            s.host_block_tables.len(),
        ));
    }
}

impl KvSelector for Recorder {
    fn name(&self) -> &'static str {
        self.name
    }

    fn on_kv_written(&self, c: &SeamCtx, s: &SeqBatch, first: &[u32], n_new: &[u32]) -> Result<()> {
        self.note("kv", c, s, format!(" written={first:?}/{n_new:?}"));
        Ok(())
    }

    fn on_query(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        pre: Option<DevicePtr>,
        _q: DevicePtr,
    ) -> Result<()> {
        self.note("query", c, s, format!(" pre_rope={}", pre.is_some()));
        Ok(())
    }

    fn plan(&self, c: &SeamCtx, s: &SeqBatch, _q: DevicePtr) -> Result<SparsePlan> {
        self.note("plan", c, s, String::new());
        Ok((self.plan)())
    }

    fn on_prefill_layer_done(&self, c: &SeamCtx, s: &SeqBatch, last: bool) -> Result<()> {
        self.note("done", c, s, format!(" last={last}"));
        Ok(())
    }

    fn reserved_bytes(&self) -> u64 {
        self.reserved
    }

    fn on_pool_geometry(&self, pool: &super::PoolGeometry) {
        *self.pool.lock().unwrap() = Some(pool.clone());
    }

    fn reserved_bytes_per_kv_block(&self) -> u64 {
        self.per_block
    }

    fn plan_prefill(&self, c: &SeamCtx, s: &SeqBatch, _q: DevicePtr) -> Result<PrefillPlan> {
        self.note("plan_prefill", c, s, String::new());
        Ok((self.prefill)())
    }
}

struct Custom {
    safe: bool,
    runs: AtomicUsize,
}

impl CustomAttention for Custom {
    fn run(&self, _c: &SeamCtx, _s: &SeqBatch, _q: DevicePtr, _out: DevicePtr) -> Result<()> {
        self.runs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn graph_safe(&self) -> bool {
        self.safe
    }
}

pub(super) fn kv_view() -> KvLayerView {
    KvLayerView {
        k_pool: DevicePtr(0x10_0000),
        v_pool: DevicePtr(0x20_0000),
        dtype: KvDtypeTag::Bf16,
        block_size: 16,
        kv_heads: 1,
        head_dim: 64,
        kv_scale: 1.0,
        v_scale: 1.0,
        block_stride_bytes: 4096,
        cache_stride: 64,
    }
}

fn ctx(gpu: &MockGpuBackend, in_graph: bool) -> SeamCtx<'_> {
    SeamCtx {
        gpu,
        attn_layer: 3,
        kv: kv_view(),
        stream: 7,
        in_graph,
        phase: SeamPhase::Decode,
        num_q_heads: 2,
        inv_sqrt_d: 0.125,
    }
}

const BT: DevicePtr = DevicePtr(0xB000);
const SL: DevicePtr = DevicePtr(0x5000);
const Q: DevicePtr = DevicePtr(1);
const OUT: DevicePtr = DevicePtr(2);

fn batch(lens: &[u32]) -> SeqBatch<'_> {
    SeqBatch {
        n: lens.len(),
        block_table: BT,
        seq_lens: SL,
        max_blocks_per_seq: 8,
        host_seq_lens: lens,
        seq_ids: &[],
        rows_per_seq: 1,
        q_row_stride: 128,
        q_rows: lens.len(),
        first_pos: &[],
        new_tokens: &[],
        host_block_tables: &[],
        block_table_ptrs: DevicePtr::NULL,
    }
}

fn dense_tables() -> Tables {
    Tables {
        block_table: BT,
        seq_lens: SL,
        max_blocks_per_seq: 8,
    }
}

fn view(block_table: u64, max_blocks_per_seq: u32) -> SparsePlan {
    SparsePlan::View {
        block_table: DevicePtr(block_table),
        seq_lens: DevicePtr(0xA8),
        max_blocks_per_seq,
    }
}

fn custom(safe: bool) -> (Arc<Custom>, Recorder) {
    let custom = Arc::new(Custom {
        safe,
        runs: AtomicUsize::new(0),
    });
    let handle: Arc<dyn CustomAttention> = custom.clone();
    let sel = Recorder::new("t", Box::new(move || SparsePlan::Custom(handle.clone())));
    (custom, sel)
}

#[test]
fn the_registry_holds_one_selector_and_names_both_on_a_second() {
    let reg = Registry::new();
    assert!(reg.get().is_none());
    reg.register(Arc::new(Recorder::dense("quest"))).unwrap();
    let err = reg
        .register(Arc::new(Recorder::dense("kv_view")))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("`kv_view`") && err.contains("`quest`"),
        "{err}"
    );
    assert_eq!(reg.get().map(|s| s.name()), Some("quest"));
}

#[test]
fn a_step_publication_nests_and_restores_on_drop() {
    assert_eq!(current_step(), None);
    let outer = push_step(StepInfo {
        seq_ids: vec![1, 2],
        last_chunk: false,
    });
    {
        let _inner = push_step(StepInfo {
            seq_ids: vec![9],
            last_chunk: true,
        });
        assert_eq!(current_step().unwrap().seq_ids, vec![9]);
    }
    assert_eq!(current_step().unwrap().seq_ids, vec![1, 2]);
    drop(outer);
    assert_eq!(current_step(), None);
}

#[test]
fn a_dense_plan_keeps_the_real_tables() {
    let gpu = MockGpuBackend::new();
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &c,
        ledger: &ledger,
    };
    let sel = Recorder::dense("t");
    let r = route::decode(&sel, &ctx(&gpu, false), &batch(&[5]), Q, OUT, &env).unwrap();
    assert_eq!(r, DecodeRoute::Run(dense_tables()));
    assert_eq!(c.plans(PlanKind::Dense), 1);
}

#[test]
fn a_view_plan_substitutes_the_selector_tables() {
    let gpu = MockGpuBackend::new();
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &c,
        ledger: &ledger,
    };
    let sel = Recorder::new("t", Box::new(|| view(0xA0, 2)));
    let r = route::decode(&sel, &ctx(&gpu, true), &batch(&[5, 9]), Q, OUT, &env).unwrap();
    let want = Tables {
        block_table: DevicePtr(0xA0),
        seq_lens: DevicePtr(0xA8),
        max_blocks_per_seq: 2,
    };
    assert_eq!(r, DecodeRoute::Run(want));
    assert_eq!(c.plans(PlanKind::View), 1);
}

#[test]
fn an_empty_view_is_an_error_naming_the_selector() {
    let gpu = MockGpuBackend::new();
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &c,
        ledger: &ledger,
    };
    let plans: [PlanFn; 2] = [Box::new(|| view(0, 2)), Box::new(|| view(0xA0, 0))];
    for plan in plans {
        let sel = Recorder::new("broken", plan);
        let err = route::decode(&sel, &ctx(&gpu, false), &batch(&[5]), Q, OUT, &env)
            .unwrap_err()
            .to_string();
        assert!(err.contains("`broken`"), "{err}");
    }
}

#[test]
fn a_custom_plan_runs_eagerly_and_replaces_the_dense_attention() {
    let gpu = MockGpuBackend::new();
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &c,
        ledger: &ledger,
    };
    let (custom, sel) = custom(false);
    let r = route::decode(&sel, &ctx(&gpu, false), &batch(&[5]), Q, OUT, &env).unwrap();
    assert_eq!(r, DecodeRoute::Done);
    assert_eq!(custom.runs.load(Ordering::Relaxed), 1);
    assert_eq!(c.graph_bypass(), 0);
}

#[test]
fn a_custom_plan_that_is_not_graph_safe_is_bypassed_while_capturing() {
    let gpu = MockGpuBackend::new();
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &c,
        ledger: &ledger,
    };
    let (custom, sel) = custom(false);
    let r = route::decode(&sel, &ctx(&gpu, true), &batch(&[5]), Q, OUT, &env).unwrap();
    assert_eq!(r, DecodeRoute::Run(dense_tables()));
    assert_eq!(custom.runs.load(Ordering::Relaxed), 0);
    assert_eq!((c.graph_bypass(), c.plans(PlanKind::Custom)), (1, 1));
}

#[test]
fn a_graph_safe_custom_plan_runs_while_capturing() {
    let gpu = MockGpuBackend::new();
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &c,
        ledger: &ledger,
    };
    let (custom, sel) = custom(true);
    let r = route::decode(&sel, &ctx(&gpu, true), &batch(&[5]), Q, OUT, &env).unwrap();
    assert_eq!(r, DecodeRoute::Done);
    assert_eq!(
        (custom.runs.load(Ordering::Relaxed), c.graph_bypass()),
        (1, 0)
    );
}

#[test]
fn captured_views_must_keep_their_pointers_per_layer_and_row_count() {
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let a = Tables {
        block_table: DevicePtr(0x100),
        seq_lens: DevicePtr(0x200),
        max_blocks_per_seq: 4,
    };
    let b = Tables {
        block_table: DevicePtr(0x300),
        ..a
    };
    assert!(ledger.observe("t", 0, 8, a, &c));
    assert!(
        ledger.observe("t", 0, 8, a, &c),
        "the same capture again is stable"
    );
    assert!(
        ledger.observe("t", 1, 8, b, &c),
        "another layer is another shape"
    );
    assert!(
        ledger.observe("t", 0, 4, b, &c),
        "another row count is another shape"
    );
    assert!(
        !ledger.observe("t", 0, 8, b, &c),
        "a moved table is flagged"
    );
    assert_eq!(c.view_ptr_changes(), 1);
}

#[test]
fn eager_views_never_reach_the_ledger() {
    let gpu = MockGpuBackend::new();
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &c,
        ledger: &ledger,
    };
    let next = Arc::new(AtomicUsize::new(0x1000));
    let bump = next.clone();
    let sel = Recorder::new(
        "t",
        Box::new(move || view(bump.fetch_add(0x100, Ordering::Relaxed) as u64, 2)),
    );
    for _ in 0..3 {
        route::decode(&sel, &ctx(&gpu, false), &batch(&[5]), Q, OUT, &env).unwrap();
    }
    assert_eq!(c.view_ptr_changes(), 0, "eager steps may use any buffers");
    for _ in 0..2 {
        route::decode(&sel, &ctx(&gpu, true), &batch(&[5]), Q, OUT, &env).unwrap();
    }
    assert_eq!(c.view_ptr_changes(), 1);
    assert_eq!(next.load(Ordering::Relaxed), 0x1000 + 5 * 0x100);
}

#[test]
fn a_prefill_view_puts_the_chunk_at_the_end_of_its_extent() {
    let gpu = MockGpuBackend::new();
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &c,
        ledger: &ledger,
    };
    let mut sel = Recorder::dense("t");
    sel.prefill = Box::new(|| PrefillPlan::View {
        block_table: DevicePtr(0xC0),
        kv_len: 600,
    });
    let v = route::prefill(&sel, &ctx(&gpu, false), &batch(&[4096]), Q, 256, 4096, &env)
        .unwrap()
        .unwrap();
    assert_eq!(
        (v.block_table, v.kv_len, v.seq_len_start),
        (DevicePtr(0xC0), 600, 344)
    );
    assert_eq!(c.plans(PlanKind::PrefillView), 1);
    let dense = Recorder::dense("d");
    let none = route::prefill(
        &dense,
        &ctx(&gpu, false),
        &batch(&[4096]),
        Q,
        256,
        4096,
        &env,
    );
    assert!(none.unwrap().is_none());
    assert_eq!(c.plans(PlanKind::PrefillDense), 1);
}

#[test]
fn a_prefill_view_must_cover_the_chunk_and_fit_the_dense_extent() {
    let gpu = MockGpuBackend::new();
    let (c, ledger) = (Counters::default(), ViewLedger::default());
    let env = RouteEnv {
        counters: &c,
        ledger: &ledger,
    };
    for kv_len in [100u32, 5000] {
        let mut sel = Recorder::dense("short");
        sel.prefill = Box::new(move || PrefillPlan::View {
            block_table: DevicePtr(0xC0),
            kv_len,
        });
        let r = route::prefill(&sel, &ctx(&gpu, false), &batch(&[4096]), Q, 256, 4096, &env);
        assert!(r.is_err(), "extent {kv_len} must be refused");
    }
}
