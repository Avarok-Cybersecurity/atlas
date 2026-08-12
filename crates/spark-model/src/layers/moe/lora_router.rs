// SPDX-License-Identifier: AGPL-3.0-only

//! Router (`mlp.gate`) LoRA fold for [`MoeLayer`] — split out of `lora.rs` for the
//! LoC budget. Covers the degenerate 1-"expert" route build plus the prefill/
//! single-seq fold (`apply_router_lora_prefill`, dense `apply_lora_delta` on the
//! gate logits) and the batched/concurrent decode fold (`apply_router_lora_batched`,
//! reusing `moe_lora_gather_bgmv` with a constant zero index + `top_k=1`).
//!
//! The routed-expert (gate/up/down) folds live in `lora.rs` + `lora_gateup.rs`; the
//! shared per-request gate `moe_route_gate` stays in `lora.rs`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::MoeLayer;
use crate::layer::MoeLoraRoute;
use crate::layers::ops;
use crate::layers::ops::lora_delta::LoraPair;
use crate::layers::ops::moe_lora_grouped::{MoeExpertRoute, pack_expert_tables};
use crate::lora::apply_router_lora;

/// PURE (unit-tested): map the router `LoraPair` to the single
/// `(expert_id, a_addr, b_addr, scale)` entry `pack_expert_tables` packs into the
/// degenerate 1-"expert" router route. Expert id is always `0` (the router owns
/// the only table slot); the A/B device addresses and scale come straight off the
/// pair. Extracted so the degenerate-table shape is verifiable without a GPU.
pub(super) fn router_expert_entry(rp: &LoraPair) -> (u16, u64, u64, f32) {
    (0, rp.a.weight.0, rp.b.weight.0, rp.scale)
}

impl MoeLayer {
    /// Build the degenerate 1-entry expert route (expert 0 == the router pair) so
    /// the batched decode router fold reuses `moe_lora_gather_bgmv` verbatim.
    /// `k_in=hidden`, `n_out=num_experts`, `max_rank=router rank`.
    pub(super) fn build_router_route(
        rp: &LoraPair,
        gpu: &dyn GpuBackend,
    ) -> Result<MoeExpertRoute> {
        let tables = pack_expert_tables(&[router_expert_entry(rp)])
            .expect("single entry => pack_expert_tables returns Some");
        let up = |vals: &[u8]| -> Result<DevicePtr> {
            let d = gpu.alloc(vals.len())?;
            gpu.copy_h2d(vals, d)?;
            Ok(d)
        };
        let a_bytes: Vec<u8> = tables.a.iter().flat_map(|p| p.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = tables.b.iter().flat_map(|p| p.to_le_bytes()).collect();
        let s_bytes: Vec<u8> = tables.scale.iter().flat_map(|s| s.to_le_bytes()).collect();
        Ok(MoeExpertRoute {
            a_table: up(&a_bytes)?,
            b_table: up(&b_bytes)?,
            scale_table: up(&s_bytes)?,
            n_experts: tables.n_experts, // == 1
            k_in: rp.k_in,               // hidden
            n_out: rp.n_out,             // num_experts
            max_rank: rp.max_rank,
        })
    }

    /// Fold the router LoRA delta onto `gate_logits` (`[n, num_experts]`) in
    /// place, BEFORE top-k. No-op when the layer has no router delta or the
    /// request opts out (base / non-active). Called from `forward_prefill` right
    /// after the base gate GEMM AND from the single-seq decode `forward` right
    /// after the gate GEMV (n=1) — this hook is n-generic. Graph-safe (pure
    /// `apply_lora_delta` kernels, no host D2H) so it needs no capture guard.
    pub(crate) fn apply_router_lora_prefill(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        n: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else {
            return Ok(());
        };
        let Some(ref rp) = l.router else {
            return Ok(());
        };
        if !self.moe_route_gate(ctx, "router")? {
            return Ok(());
        }
        apply_router_lora(
            ctx.gpu,
            &l.kernels,
            rp,
            router_in,
            gate_logits,
            n,
            l.cap,
            l.xa,
            l.delta,
            stream,
        )
    }

    /// Batched/concurrent decode router fold: fold the router delta onto the
    /// whole-batch `gate_logits` (`[n, num_experts]` BF16) in place, before top-k,
    /// via the degenerate 1-"expert" gather (`router_zero_indices` all 0, `top_k=1`
    /// so `row/top_k == row`, `x_gather=0`). Per-row base-skip via `row_adapter`
    /// (device `<0` predicate); NULL map falls back to the single-active
    /// `moe_route_gate`. Refuses only the FP32-gate path (no BF16-ULP oracle) and
    /// only when the batch actually folds — a base/`Skip` batch must stay
    /// byte-identical. Mixed multi-adapter (`Refuse`) is bailed upstream,
    /// pre-capture. Graph-safe: install-time gate + fixed-address metadata.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_router_lora_batched(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        n: u32,
        row_adapter: DevicePtr,
        fp32_gate: bool,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else {
            return Ok(());
        };
        let Some(ref rr) = l.router_route else {
            return Ok(()); // router-less adapter: nothing to fold on the logits
        };
        // Refuse the FP32-gate path ONLY when this batch actually folds. The BF16
        // gather kernel has no BF16-ULP oracle on the FP32 `gate_logits_f32` buffer,
        // but a base/`Skip` batch folds nothing (the device self-skips an all-base
        // per-row map; the NULL-map `moe_route_gate` returns false below), so it
        // must NOT trip this bail — otherwise a pure-base decode with a router
        // adapter merely RESIDENT + FP32 gate would wrongly error (regression vs the
        // old `reject_batched_router_lora` `Skip` carve-out). Mixed multi-adapter
        // (`Refuse`) is already bailed upstream, pre-capture.
        let folds = !matches!(ctx.moe_lora_route, MoeLoraRoute::Skip);
        anyhow::ensure!(
            !(fp32_gate && folds),
            "MoE LoRA batched router fold requires BF16 gate_logits; the FP32-gate path \
             (ATLAS_FP32_GATE / fp32 routing) has no BF16-ULP oracle against the single-stream \
             router fold. Unset the FP32-gate flag to serve a router-adapted adapter \
             concurrently, or route router adapters single-stream."
        );
        // NULL-map single-active fallback — identical to
        // `apply_expert_lora_decode_down` (in `lora.rs`): consult the
        // request-granularity `moe_route_gate` ONLY when there is no per-row map
        // (its `Skip`/`Refuse` host branch would otherwise bake one route into the
        // captured graph). With a per-row map present, the launch is unconditional
        // and route-agnostic — base rows self-skip device-side.
        if row_adapter == DevicePtr::NULL && !self.moe_route_gate(ctx, "router-decode")? {
            return Ok(());
        }
        anyhow::ensure!(
            n <= l.cap,
            "MoE LoRA batched router fold: n ({n}) exceeds LoRA scratch cap ({}); raise \
             ATLAS_LORA_EXPERT_MAX_TOKENS to >= num_tokens.",
            l.cap
        );
        // Degenerate expert gather: `router_zero_indices` (all 0) routes every row
        // to `router_route`'s single entry; `top_k=1` makes `row/top_k == row` so
        // `row_adapter[row]` is this token's base/adapt sign; `x_gather=0` reads the
        // per-token `router_in` row directly (with top_k=1, x_gather 0/1 coincide).
        ops::moe_lora_gather_bgmv(
            ctx.gpu,
            &l.kernels,
            rr,
            router_in,             // x [n, hidden]
            gate_logits,           // base_out [n, num_experts] BF16, folded in place
            l.router_zero_indices, // indices [n] all 0 -> fake-expert 0
            row_adapter,           // [num_tokens] i32 (<0 skip) or NULL
            l.xa,                  // shrink scratch
            n,                     // n_slots
            1,                     // top_k=1 -> row/top_k == row
            0,                     // x_gather=0: x row == token
            stream,
        )
    }
}
