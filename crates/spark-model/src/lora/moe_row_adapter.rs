// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 MoE-LoRA per-request routing primitives (pure, GPU-free).
//!
//! Two host-side building blocks that keep the MoE fold correct in a mixed
//! batch without leaking a host D2H into the fold hot path:
//!
//! - [`resolve_moe_lora_route`] — the request-granularity fold decision used by
//!   `TransformerModel::moe_lora_route`. Base / non-active requests SKIP (pay
//!   nothing); the active-adapter request FOLDs; anything the single-active
//!   phase-1 fold cannot serve REFUSES loudly. This is the correctness core of
//!   the zero-overhead-when-a-request-opts-out invariant, unit-tested here.
//! - [`build_moe_row_adapter_host`] — the `[total_tokens]` per-packed-row
//!   adapter map (`< 0` = base) the device-side grouped fold (`lora-solid.md`
//!   Incr 1/3) will consume as a fixed-address kernel arg. It is built the same
//!   way the attention `seq_slot` buffer is (`slot_math::build_seq_slot_host`),
//!   keyed off `cu_seqlens_host` (the packing SSOT = Σ proc_count), NOT
//!   `b * chunk_len`, so varlen + partial-prefix-cache-hit batches stay aligned.
//!   The device consumption is the documented follow-up; the builder is landed +
//!   tested now so the row map is a solved, verified primitive.

use crate::layer::MoeLoraRoute;

/// SOLID Incr-4 host-side pre-lookup guard for the BATCHED decode entries
/// (`decode_batch_compute_main`, `mixed_forward`): a batch containing a row
/// routed to a NON-active adapter (`Refuse`) cannot be served by the
/// single-active fold — [`build_moe_row_adapter_decode`] defensively maps such
/// rows to base, so proceeding would SILENTLY serve base weights for an
/// adapter-routed request. Call BEFORE any graph lookup/capture so captured
/// `padded_n` graphs stay route-agnostic. Pure (no `self`, no GPU) so the
/// decision is unit-testable without hardware.
pub fn ensure_decode_route_servable(route: MoeLoraRoute, path: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !matches!(route, MoeLoraRoute::Refuse),
        "MoE LoRA {path}: a sequence routes to a non-active adapter under single-active \
         phase-1; refusing rather than mis-folding the active adapter. One adapter per batch."
    );
    Ok(())
}

/// Resolve the Feature-1 MoE-LoRA fold decision for a single-request pass.
///
/// `adapter_slot` is the request's `SequenceState.adapter_slot` (`< 0` = no
/// adapter / base); `active` is the installed pool's active slot index
/// (`< 0` when no MoE adapter is installed); `has_moe_lora` is whether this
/// layer actually installed an expert/router delta.
///
/// - no MoE delta installed ⇒ `Fold` (the fold hook no-ops on `self.lora ==
///   None`, so the value is inert — kept `Fold` so nothing changes when off).
/// - base request (`adapter_slot < 0`) ⇒ `Skip` — base tokens pay nothing.
/// - request owns the active adapter (`adapter_slot == active`) ⇒ `Fold`.
/// - request routes to a different, non-installed adapter ⇒ `Refuse` — phase-1
///   installs one active MoE adapter; folding the wrong one is silently wrong,
///   so refuse loudly instead.
pub fn resolve_moe_lora_route(adapter_slot: i32, active: i32, has_moe_lora: bool) -> MoeLoraRoute {
    if !has_moe_lora {
        return MoeLoraRoute::Fold;
    }
    if adapter_slot < 0 {
        return MoeLoraRoute::Skip;
    }
    if adapter_slot == active {
        MoeLoraRoute::Fold
    } else {
        MoeLoraRoute::Refuse
    }
}

/// Build the `[total_tokens]` per-packed-row adapter map for the device-side
/// grouped fold. `cu_seqlens_host` is the `[batch + 1]` prefix-sum of per-stream
/// token counts (the packing SSOT); `adapter_slots[b]` is stream `b`'s
/// `adapter_slot`. Each stream's slot is broadcast across its
/// `[cu_seqlens_host[b], cu_seqlens_host[b + 1])` row span. A base stream
/// (`adapter_slot < 0`) writes `-1` (device kernel skips those rows); a stream
/// deferring to the active adapter would resolve `< 0 → active` at the call site
/// before this, so a genuine base row is a real `-1` here (distinct base
/// sentinel — see `lora-solid.md` §6).
///
/// Returns `None` on a malformed `cu_seqlens_host` (empty, or a non-monotonic
/// boundary) rather than panicking — the caller then declines the device fold.
pub fn build_moe_row_adapter_host(
    cu_seqlens_host: &[i32],
    adapter_slots: &[i32],
) -> Option<Vec<i32>> {
    if cu_seqlens_host.len() < 2 {
        return None;
    }
    let batch = cu_seqlens_host.len() - 1;
    if adapter_slots.len() != batch {
        return None;
    }
    if cu_seqlens_host[0] != 0 {
        return None;
    }
    // Validate the WHOLE prefix-sum is non-decreasing and non-negative BEFORE
    // writing any row, so a boundary that exceeds the declared total (e.g.
    // `[0, 4, 2]`) is rejected rather than overflowing the map.
    for b in 0..batch {
        let start = cu_seqlens_host[b];
        let end = cu_seqlens_host[b + 1];
        if start < 0 || end < start {
            return None; // negative or non-monotonic boundary
        }
    }
    let total = cu_seqlens_host[batch];
    let mut map = vec![-1i32; total as usize];
    for b in 0..batch {
        let start = cu_seqlens_host[b];
        let end = cu_seqlens_host[b + 1];
        let slot = adapter_slots[b];
        for row in start..end {
            map[row as usize] = slot;
        }
    }
    Some(map)
}

/// SOLID Incr-4 (batched decode fold): build the per-row `[padded_n]` i32
/// adapter map the device-side MoE gather-BGMV fold reads. One token per
/// sequence at decode, so a per-row map IS a per-seq map (no `top_k` expansion —
/// the kernel indexes `row_adapter[row / top_k]`).
///
/// Per row `i`, `resolve_moe_lora_route(adapter_slots[i], active, has_moe_lora)`:
///   - `Fold` (owns the installed active adapter) ⇒ write `active` (`>= 0` when
///     an adapter is resident — the kernel only tests the SIGN, and the single
///     active adapter's per-expert tables are folded). When no adapter is
///     installed (`has_moe_lora == false`) `active` is `-1`, so the row
///     correctly skips.
///   - `Skip` (base / non-active) ⇒ `-1` (device kernel skips the row — MoE's
///     `< 0 = base` semantics, NOT the attention `seq_slot` `-1 → active`).
///   - `Refuse` (non-active adapter present) ⇒ `-1` DEFENSIVELY. A `Refuse`
///     batch is bailed host-side before this map is ever uploaded
///     (`stamp_decode_moe_batch` + the `decode_batch_compute_main` pre-lookup
///     guard); mapping to base here means a leaked `Refuse` row folds NOTHING
///     rather than mis-folding the wrong adapter.
///   - pad rows (`i >= adapter_slots.len()`) ⇒ `-1` (skip).
///
/// Pure + GPU-free so the routing is unit-testable without hardware; the device
/// upload is a thin wrapper (`upload_moe_row_adapter`).
pub fn build_moe_row_adapter_decode(
    adapter_slots: &[i32],
    padded_n: usize,
    active: i32,
    has_moe_lora: bool,
) -> Vec<i32> {
    (0..padded_n)
        .map(|i| match adapter_slots.get(i).copied() {
            Some(slot) => match resolve_moe_lora_route(slot, active, has_moe_lora) {
                MoeLoraRoute::Fold => active,
                MoeLoraRoute::Skip | MoeLoraRoute::Refuse => -1,
            },
            None => -1, // pad row
        })
        .collect()
}

#[cfg(test)]
#[path = "moe_row_adapter_tests.rs"]
mod tests;
