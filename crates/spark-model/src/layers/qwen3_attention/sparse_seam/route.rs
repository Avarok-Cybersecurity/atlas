// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the seam's pure decisions, independent of the layer: what
//! a plan means for the paged attention call, the graph-rule checks, how the
//! rows of a batched step group into logical sequences, and how published ids
//! map onto them. The call sites in `sites.rs` only gather inputs and apply
//! these results.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::metrics::{Counters, PlanKind};
use super::{KvSelector, PAD_SEQ_ID, PrefillPlan, SeamCtx, SeqBatch, SparsePlan};

/// The table triple the paged-decode attention reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tables {
    pub(crate) block_table: DevicePtr,
    pub(crate) seq_lens: DevicePtr,
    pub(crate) max_blocks_per_seq: u32,
}

/// What a decode or verify site does after the plan.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DecodeRoute {
    /// Run the paged-decode attention over these tables.
    Run(Tables),
    /// The selector's custom attention already wrote the output.
    Done,
}

/// What a single-stream paged prefill chunk attends to instead of the dense table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefillView {
    pub(crate) block_table: DevicePtr,
    pub(crate) kv_len: u32,
    /// The chunk's first query position inside the compacted extent.
    pub(crate) seq_len_start: u32,
}

/// The seam's process state a route decision updates. Tests build their own.
pub(crate) struct RouteEnv<'a> {
    pub(crate) counters: &'a Counters,
    pub(crate) ledger: &'a ViewLedger,
}

impl RouteEnv<'static> {
    pub(crate) fn global() -> Self {
        Self {
            counters: &super::metrics::GLOBAL,
            ledger: &GLOBAL_LEDGER,
        }
    }
}

/// Ask the selector for this decode or verify step's plan and turn it into the
/// attention call's inputs.
pub(crate) fn decode(
    sel: &dyn KvSelector,
    c: &SeamCtx,
    s: &SeqBatch,
    q: DevicePtr,
    out: DevicePtr,
    env: &RouteEnv,
) -> Result<DecodeRoute> {
    let dense = Tables {
        block_table: s.block_table,
        seq_lens: s.seq_lens,
        max_blocks_per_seq: s.max_blocks_per_seq,
    };
    match sel.plan(c, s, q)? {
        SparsePlan::Dense => {
            env.counters.count_plan(PlanKind::Dense);
            Ok(DecodeRoute::Run(dense))
        }
        SparsePlan::View {
            block_table,
            seq_lens,
            max_blocks_per_seq,
        } => {
            anyhow::ensure!(
                !block_table.is_null() && !seq_lens.is_null() && max_blocks_per_seq > 0,
                "sparse attention: selector `{}` returned an empty View (block_table {block_table}, \
                 seq_lens {seq_lens}, max_blocks_per_seq {max_blocks_per_seq}) on attention layer {}",
                sel.name(),
                c.attn_layer,
            );
            let view = Tables {
                block_table,
                seq_lens,
                max_blocks_per_seq,
            };
            if c.in_graph {
                env.ledger
                    .observe(sel.name(), c.attn_layer, s.n, view, env.counters);
            }
            env.counters.count_plan(PlanKind::View);
            Ok(DecodeRoute::Run(view))
        }
        SparsePlan::Custom(custom) => {
            env.counters.count_plan(PlanKind::Custom);
            if c.in_graph && !custom.graph_safe() {
                env.counters.count_graph_bypass();
                tracing::debug!(
                    "sparse attention: `{}` custom plan is not graph-safe; capturing the dense \
                     tables for attention layer {} ({} rows)",
                    sel.name(),
                    c.attn_layer,
                    s.n,
                );
                return Ok(DecodeRoute::Run(dense));
            }
            custom.run(c, s, q, out)?;
            Ok(DecodeRoute::Done)
        }
    }
}

/// Ask the selector for a paged prefill chunk's plan. `None` keeps the dense
/// table. A view must keep every query token: its extent covers the chunk and
/// is no longer than the dense extent.
pub(crate) fn prefill(
    sel: &dyn KvSelector,
    c: &SeamCtx,
    s: &SeqBatch,
    q: DevicePtr,
    num_tokens: u32,
    dense_kv_len: u32,
    env: &RouteEnv,
) -> Result<Option<PrefillView>> {
    match sel.plan_prefill(c, s, q)? {
        PrefillPlan::Dense => {
            env.counters.count_plan(PlanKind::PrefillDense);
            Ok(None)
        }
        PrefillPlan::View {
            block_table,
            kv_len,
        } => {
            anyhow::ensure!(
                !block_table.is_null() && kv_len >= num_tokens && kv_len <= dense_kv_len,
                "sparse attention: selector `{}` returned a prefill View of extent {kv_len} \
                 (table {block_table}) for a {num_tokens}-token chunk whose dense extent is \
                 {dense_kv_len}, on attention layer {}",
                sel.name(),
                c.attn_layer,
            );
            env.counters.count_plan(PlanKind::PrefillView);
            Ok(Some(PrefillView {
                block_table,
                kv_len,
                seq_len_start: kv_len - num_tokens,
            }))
        }
    }
}

/// The graph-rule check for captured View plans: one capture per (layer, rows)
/// shape records its table pointers, and a later capture of the same shape
/// with different pointers is counted and warned about, because a graph still
/// replaying the old capture reads buffers the selector may no longer refresh.
#[derive(Default)]
pub(crate) struct ViewLedger {
    seen: Mutex<HashMap<(usize, usize), Tables>>,
}

pub(crate) static GLOBAL_LEDGER: std::sync::LazyLock<ViewLedger> =
    std::sync::LazyLock::new(ViewLedger::default);

impl ViewLedger {
    /// Record a captured View; false (and a counted change) when it differs
    /// from the earlier capture of the same shape.
    pub(crate) fn observe(
        &self,
        selector: &str,
        attn_layer: usize,
        rows: usize,
        view: Tables,
        counters: &Counters,
    ) -> bool {
        let mut seen = self
            .seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match seen.insert((attn_layer, rows), view) {
            Some(prev) if prev != view => {
                counters.count_view_ptr_change();
                tracing::warn!(
                    "sparse attention: `{selector}` captured attention layer {attn_layer} at {rows} \
                     rows with View {view:?}, but an earlier capture of the same shape used {prev:?}; \
                     selectors must back View plans with fixed buffers (graph rule, master plan 3.7.6)"
                );
                false
            }
            _ => true,
        }
    }
}

/// How the rows of a batched decode or verify step group into sequences.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowGroups {
    pub(crate) rows_per_seq: usize,
    /// Per row: attention extent with this step's token (0 = padding).
    pub(crate) host_seq_lens: Vec<u32>,
    /// Per group: first row index.
    pub(crate) first_row: Vec<usize>,
    /// Per group: absolute position of its first new token (0 for padding).
    pub(crate) first_pos: Vec<u32>,
    /// Per group: new tokens (`rows_per_seq`, or 0 for padding).
    pub(crate) new_tokens: Vec<u32>,
    /// Per group: true when the group is padding.
    pub(crate) padding: Vec<bool>,
}

/// Group batched rows. `seq_lens[i]` is row i's pre-append length (the
/// position of the token it adds; 0 marks a padding row) and `tables[i]` its
/// host block table. Consecutive real rows with the same table whose lengths
/// step by one are one sequence (speculative verify rows); two live sequences
/// never share a whole table because each owns its writable tail block. When
/// every group has the same size `g` and every padding run is a multiple of
/// `g`, rows group by `g`; otherwise each row is its own sequence.
pub(crate) fn group_rows(seq_lens: &[usize], tables: &[Vec<u32>]) -> RowGroups {
    let n = seq_lens.len().min(tables.len());
    let host_seq_lens: Vec<u32> = (0..n)
        .map(|i| {
            if seq_lens[i] == 0 {
                0
            } else {
                u32::try_from(seq_lens[i] + 1).unwrap_or(u32::MAX)
            }
        })
        .collect();
    // Runs: (start, len, padding).
    let mut runs: Vec<(usize, usize, bool)> = Vec::new();
    let mut i = 0;
    while i < n {
        let pad = seq_lens[i] == 0;
        let continues = |j: usize| {
            if pad {
                seq_lens[j] == 0
            } else {
                seq_lens[j] == seq_lens[j - 1] + 1 && tables[j] == tables[i]
            }
        };
        let mut j = i + 1;
        while j < n && continues(j) {
            j += 1;
        }
        runs.push((i, j - i, pad));
        i = j;
    }
    let real_sizes: Vec<usize> = runs.iter().filter(|r| !r.2).map(|r| r.1).collect();
    let g = match real_sizes.first() {
        Some(&g) if real_sizes.iter().all(|&s| s == g) => g,
        _ => 1,
    };
    let g = if runs.iter().all(|r| !r.2 || r.1.is_multiple_of(g)) {
        g
    } else {
        1
    };
    let mut out = RowGroups {
        rows_per_seq: g,
        host_seq_lens,
        first_row: Vec::new(),
        first_pos: Vec::new(),
        new_tokens: Vec::new(),
        padding: Vec::new(),
    };
    for (start, len, pad) in runs {
        // `g` divides every run when g > 1; with g == 1 every row is a group.
        let mut r = start;
        while r < start + len {
            out.first_row.push(r);
            out.padding.push(pad);
            out.first_pos.push(if pad {
                0
            } else {
                u32::try_from(seq_lens[r]).unwrap_or(u32::MAX)
            });
            out.new_tokens.push(if pad {
                0
            } else {
                u32::try_from(g).unwrap_or(u32::MAX)
            });
            r += g;
        }
    }
    out
}

/// Map published ids (padding excluded, in order) onto groups. Empty when the
/// publication does not match the step's real groups.
pub(crate) fn assign_ids(published: &[u64], padding: &[bool]) -> Vec<u64> {
    let real = padding.iter().filter(|p| !**p).count();
    if published.is_empty() || published.len() != real {
        return Vec::new();
    }
    let mut next = published.iter();
    padding
        .iter()
        .map(|&pad| {
            if pad {
                PAD_SEQ_ID
            } else {
                *next.next().expect("counted above")
            }
        })
        .collect()
}
