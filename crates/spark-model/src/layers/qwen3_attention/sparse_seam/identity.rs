// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the seam's identity selector, the diagnostic that proves
//! the seam itself is exact. Enabled by `AVAROK_SPARSE_SEAM_IDENTITY=1`
//! (default off on every target; it is a test instrument, not a lever, so it
//! has no `[defaults]` row).
//!
//! It selects everything: every decode and verify plan is a `View` whose
//! table is a copy of the real block table and lengths, made on the stream
//! into selector-owned fixed device buffers, exactly the shape a real
//! selector uses (the graph rule). Every paged prefill chunk
//! gets a `PrefillPlan::View` of the full table. Greedy output must therefore
//! be byte-identical to a serve without it; any difference is a seam bug. It
//! also exercises the registration, the KV-budget charge and the metrics.
//!
//! The buffers are one 16 MiB arena, allocated on the first eager hook after
//! the model build (never while capturing) and charged to the KV budget at
//! registration. A shape that does not fit the arena, or a capture that comes
//! before any eager hook, plans `Dense`, which is numerically the same thing.
//! The arena is keyed by the backend it was allocated on: a model swap builds
//! a new backend, whose sweep frees the old arena, and the next eager hook on
//! the new backend allocates a fresh one.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{KvSelector, PrefillPlan, SeamCtx, SeqBatch, SparsePlan};

/// The opt-in environment variable.
pub const ENV: &str = "AVAROK_SPARSE_SEAM_IDENTITY";

/// Device bytes the identity selector reserves and allocates.
pub const ARENA_BYTES: usize = 16 << 20;

/// Offsets inside the arena are kept 256-byte aligned.
const ALIGN: usize = 256;

/// The env toggle grammar: unset, empty, `0`, `false`, `off` and `no` are off.
pub fn enabled_by(raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        None | Some("") => false,
        Some(v) => !matches!(
            v.to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
    }
}

/// Register the identity selector when `AVAROK_SPARSE_SEAM_IDENTITY` is on
/// (one `port_apply` line). Off: nothing happens.
pub fn register_from_env() -> Result<()> {
    if !enabled_by(std::env::var(ENV).ok().as_deref()) {
        return Ok(());
    }
    tracing::warn!(
        "{ENV} is on: the sparse-attention seam runs its identity selector (a diagnostic; \
         output must equal a serve without it)"
    );
    super::register(std::sync::Arc::new(IdentitySelector::new()))
}

struct Arena {
    /// Address of the backend the arena was allocated on.
    owner: usize,
    base: DevicePtr,
}

/// The selector itself. Public so tests and microbenchmarks can drive it.
pub struct IdentitySelector {
    arena: Mutex<Option<Arena>>,
    warned_fit: AtomicBool,
}

impl Default for IdentitySelector {
    fn default() -> Self {
        Self::new()
    }
}

fn backend_key(gpu: &dyn GpuBackend) -> usize {
    std::ptr::from_ref(gpu).cast::<()>() as usize
}

fn align(v: usize) -> usize {
    v.div_ceil(ALIGN) * ALIGN
}

impl IdentitySelector {
    pub fn new() -> Self {
        Self {
            arena: Mutex::new(None),
            warned_fit: AtomicBool::new(false),
        }
    }

    /// The arena on `gpu`, allocated now when `may_alloc` (never while capturing).
    fn arena(&self, gpu: &dyn GpuBackend, may_alloc: bool) -> Result<Option<DevicePtr>> {
        let key = backend_key(gpu);
        let mut slot = self
            .arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(a) = slot.as_ref()
            && a.owner == key
        {
            return Ok(Some(a.base));
        }
        if !may_alloc {
            return Ok(None);
        }
        let base = gpu.alloc(ARENA_BYTES)?;
        *slot = Some(Arena { owner: key, base });
        Ok(Some(base))
    }

    /// Bytes a View of `rows` rows and `max_blocks` columns occupies.
    pub fn view_bytes(rows: usize, max_blocks: usize) -> usize {
        align(rows * max_blocks * 4) + align(rows * 4)
    }
}

impl KvSelector for IdentitySelector {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn reserved_bytes(&self) -> u64 {
        ARENA_BYTES as u64
    }

    fn on_kv_written(
        &self,
        c: &SeamCtx,
        _s: &SeqBatch,
        _first_pos: &[u32],
        _n_new: &[u32],
    ) -> Result<()> {
        // The first eager hook allocates, so the first decode capture finds it.
        self.arena(c.gpu, !c.in_graph).map(|_| ())
    }

    fn plan(&self, c: &SeamCtx, s: &SeqBatch, _q: DevicePtr) -> Result<SparsePlan> {
        if s.block_table.is_null() || s.seq_lens.is_null() || s.max_blocks_per_seq == 0 {
            return Ok(SparsePlan::Dense);
        }
        let mb = s.max_blocks_per_seq as usize;
        if Self::view_bytes(s.n, mb) > ARENA_BYTES {
            if !self.warned_fit.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "sparse attention identity: a {}-row x {mb}-block view does not fit the {} MiB \
                     arena; planning Dense for that shape",
                    s.n,
                    ARENA_BYTES >> 20,
                );
            }
            return Ok(SparsePlan::Dense);
        }
        let Some(base) = self.arena(c.gpu, !c.in_graph)? else {
            return Ok(SparsePlan::Dense);
        };
        let table_bytes = s.n * mb * 4;
        let block_table = base;
        let seq_lens = base.offset(align(table_bytes));
        c.gpu
            .copy_d2d_async(s.block_table, block_table, table_bytes, c.stream)?;
        c.gpu
            .copy_d2d_async(s.seq_lens, seq_lens, s.n * 4, c.stream)?;
        if !c.in_graph {
            let bs = c.kv.block_size.max(1);
            let pages: u64 = s
                .host_seq_lens
                .iter()
                .map(|&l| u64::from(l.div_ceil(bs)))
                .sum();
            super::record_selected_pages(pages);
        }
        Ok(SparsePlan::View {
            block_table,
            seq_lens,
            max_blocks_per_seq: s.max_blocks_per_seq,
        })
    }

    fn plan_prefill(&self, _c: &SeamCtx, s: &SeqBatch, _q: DevicePtr) -> Result<PrefillPlan> {
        match (s.n, s.host_seq_lens.first()) {
            (1, Some(&kv_len)) if !s.block_table.is_null() => Ok(PrefillPlan::View {
                block_table: s.block_table,
                kv_len,
            }),
            _ => Ok(PrefillPlan::Dense),
        }
    }
}
