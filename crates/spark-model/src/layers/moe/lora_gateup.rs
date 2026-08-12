// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 MoE LoRA gate/up-proj fold hooks (prefill + decode).
//!
//! Split out of `moe/lora.rs` to stay under the 500 LoC cap. Both hooks REUSE
//! the down-fold launchers verbatim — only the args differ:
//!   * per-proj route (`gate_route`/`up_route`, `k_in=hidden`, `n_out=moe_inter`
//!     — the TRANSPOSE of down),
//!   * `x = expert_input` (token-major, NOT the post-SiLU sorted/slot activation
//!     the down fold uses), so the shrink kernel gathers its x-row (`x_gather=1`):
//!     prefill `x[sorted_token_ids[r]]`, decode `x[row / top_k]`,
//!   * `base_out = expert_gate_out` / `expert_up_out`.
//!
//! ORDERING (correctness-critical): both hooks fold BEFORE the `silu_mul` /
//! fused silu+down consumes gate/up in place — folding after would be lost, and
//! folding before makes the decode down-fold's later `silu(gate)*up` recompute
//! automatically see the folded gate/up (numerically consistent end-to-end with
//! prefill). Gate then up reuse `l.xa` serially on the same stream (each shrink
//! precedes its expand). No-op when the layer installs no gate/up pair (both
//! routes `None`) or the request opts out (`moe_route_gate` Skip); `Refuse` bails.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::MoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MoeLayer {
    /// Fold the routed-expert gate/up_proj LoRA deltas onto the SORTED
    /// `expert_gate_out` / `expert_up_out` (`[total_expanded, moe_inter]` BF16),
    /// BEFORE `silu_mul` overwrites `expert_gate_out` with `silu(gate)*up`. `x` =
    /// the token-major `expert_input` (`[num_tokens, hidden]`), gathered per sorted
    /// row via `sorted_token_ids` inside the shrink kernel (`x_gather=1`). No-op
    /// unless a gate/up delta is installed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_expert_lora_prefill_gateup(
        &self,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        te: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else {
            return Ok(());
        };
        if l.gate_route.is_none() && l.up_route.is_none() {
            return Ok(()); // no gate/up pairs: down-only or router-only adapter
        }
        if !self.moe_route_gate(ctx, "expert-gateup")? {
            return Ok(());
        }
        // Chunk `te` into <= cap-row windows (xa is `[cap, max_rank]`, LOCAL-row
        // indexed). Gate then up reuse `l.xa` serially per window: gate's expand
        // fully precedes up's shrink on the ordered stream, so a per-window
        // gate→up pair preserves the same serial-reuse discipline as the
        // unchunked path.
        let cap = l.cap;
        let mut off = 0u32;
        while off < te {
            let end = off.saturating_add(cap).min(te);
            if let Some(ref gate) = l.gate_route {
                ops::moe_lora_grouped_down(
                    ctx.gpu,
                    &l.kernels,
                    gate,
                    expert_input,
                    expert_gate_out,
                    expert_offsets,
                    sorted_token_ids,
                    DevicePtr::NULL,
                    l.xa,
                    off,
                    end,
                    1, // x_gather=1: gather token-major expert_input via sorted_token_ids
                    stream,
                )?;
            }
            if let Some(ref up) = l.up_route {
                ops::moe_lora_grouped_down(
                    ctx.gpu,
                    &l.kernels,
                    up,
                    expert_input,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    DevicePtr::NULL,
                    l.xa,
                    off,
                    end,
                    1,
                    stream,
                )?;
            }
            off = end;
        }
        Ok(())
    }

    /// Decode analogue: fold the routed-expert gate/up_proj LoRA deltas onto the
    /// slot-major `expert_gate_out` / `expert_up_out` (`[n_slots, moe_inter]` BF16)
    /// IN PLACE, BEFORE the fused silu+down consumes them. `x = expert_input`
    /// (`[1, hidden]` BF16, already live — NO silu recompute, unlike the down
    /// decode fold); the shrink kernel reads the owning token `row / top_k`
    /// (`x_gather=1`), which for a single-token decode (`n_slots == top_k`) is row
    /// 0 for every slot. `indices_dev` is the same `[n_slots]` u32 expert-id array
    /// the fused expert GEMV routed on. No-op unless a gate/up delta is installed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_expert_lora_decode_gateup(
        &self,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_input: DevicePtr,
        indices_dev: DevicePtr,
        n_slots: u32,
        top_k: u32,
        row_adapter: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else {
            return Ok(());
        };
        if l.gate_route.is_none() && l.up_route.is_none() {
            return Ok(());
        }
        // SOLID Incr-4: a non-NULL per-row map supersedes the request gate — the
        // batched fold launches unconditionally and skips base rows device-side
        // (route-agnostic capture). Single-seq (NULL map) still uses the gate.
        if row_adapter == DevicePtr::NULL && !self.moe_route_gate(ctx, "expert-gateup-decode")? {
            return Ok(());
        }
        anyhow::ensure!(
            n_slots <= l.cap,
            "MoE expert LoRA decode gate/up-fold: n_slots ({n_slots}) exceeds LoRA scratch cap \
             ({}); raise ATLAS_LORA_EXPERT_MAX_TOKENS to >= num_tokens*top_k.",
            l.cap
        );
        if let Some(ref gate) = l.gate_route {
            ops::moe_lora_gather_bgmv(
                ctx.gpu,
                &l.kernels,
                gate,
                expert_input,
                expert_gate_out,
                indices_dev,
                row_adapter,
                l.xa,
                n_slots,
                top_k,
                1, // x_gather=1: x-row = token = row/top_k (shared expert_input row)
                stream,
            )?;
        }
        if let Some(ref up) = l.up_route {
            ops::moe_lora_gather_bgmv(
                ctx.gpu,
                &l.kernels,
                up,
                expert_input,
                expert_up_out,
                indices_dev,
                row_adapter,
                l.xa,
                n_slots,
                top_k,
                1,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "lora_gateup_tests.rs"]
mod tests;
