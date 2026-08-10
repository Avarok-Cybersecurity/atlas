// SPDX-License-Identifier: AGPL-3.0-only

//! Token-overlay forward hooks (Feature 2). `apply_embed_overlay` replaces the
//! embedding rows of overridden vocab ids right after the gather and BEFORE
//! `scale_embeddings`; `apply_lmhead_overlay` recomputes the logit columns of
//! overridden ids right after the base projection and BEFORE softcap.
//!
//! ZERO-OVERHEAD WHEN OFF: both hooks early-return on `self.overlays.is_none()`
//! (no overlay adapter installed) — one Option check, byte-identical to a build
//! with no overlay. When installed but a request/row opts out, the per-row skip
//! is a predicated kernel early-return (`s<0`, `slot_map[id]<0`, `j>=n`), never
//! a host branch.
//!
//! ROUTING: identity flows through the per-step device `seq_slot` buffer (fixed
//! address, contents re-uploaded each step) plus a uniform `active` fallback —
//! exactly the attention BGMV contract, so the launches are CUDA-graph safe.
//! This pass wires the UNIFORM-active routing (single-request / batched-per-seq
//! paths pass `seq_slot = NULL` + the resolved active slot); the mixed-decode
//! `seq_slot`-per-row batch (`decode_b2.rs`) lands with the decode MoE routing
//! (SOLID Increment 4).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::layers::ops::token_overlay;

impl TransformerModel {
    /// The uniformly-active adapter slot for the overlay hooks: the installed
    /// pool's `active` index, or `-1` (base / no delta) when no pool is resident.
    /// A `-1` return makes every hook a no-op, so a non-overlay LoRA run and a
    /// no-LoRA run are both byte-identical.
    /// Stamp the per-forward overlay route from a single request's `adapter_slot`
    /// (called at each single-sequence `Model` entry). Cheap unconditional store;
    /// the hooks still early-return when no overlay is installed.
    pub(super) fn stamp_overlay_route(&self, adapter_slot: i32) {
        self.overlay_route_slot
            .store(adapter_slot, std::sync::atomic::Ordering::Relaxed);
    }

    /// Stamp from a batch: the shared `adapter_slot` when every sequence agrees,
    /// else `i32::MIN` (mixed ⇒ the overlay hooks skip; per-token `seq_slot`
    /// routing is SOLID Incr-4). An empty batch resets to `-1`.
    pub(super) fn stamp_overlay_route_batch(&self, seqs: &[&mut crate::traits::SequenceState]) {
        let slot = match seqs.split_first() {
            Some((first, rest)) if rest.iter().all(|s| s.adapter_slot == first.adapter_slot) => {
                first.adapter_slot
            }
            Some(_) => i32::MIN,
            None => -1,
        };
        self.overlay_route_slot
            .store(slot, std::sync::atomic::Ordering::Relaxed);
    }

    pub(super) fn overlay_active_slot(&self) -> i32 {
        let Some(l) = self.lora.as_ref() else {
            return -1;
        };
        let req = self
            .overlay_route_slot
            .load(std::sync::atomic::Ordering::Relaxed);
        if req == i32::MIN {
            // Mixed-adapter decode batch: per-token seq_slot routing is deferred
            // (SOLID Incr-4). Skip rather than uniform-apply one adapter's overlay.
            return -1;
        }
        // Resolve the REQUEST's adapter_slot (not the pool's `active`): a request
        // that selects a non-active pool slot must get that slot's overlay. `-1`
        // (defer-to-active) resolves to `active`; out-of-range ⇒ None ⇒ skip.
        l.routed_prefill_slot(req).map(|s| s as i32).unwrap_or(-1)
    }

    /// Overlay the embedding rows of overridden vocab ids in place on `out`.
    /// `ids_dev` is the device `u32[num_tokens]` token-id buffer for these rows
    /// (the same one the gather read). `seq_slot` = `DevicePtr(0)` selects the
    /// uniform `active` route; a real per-row `seq_slot` device buffer routes a
    /// mixed batch. Inserted AFTER the embed gather, BEFORE `scale_embeddings`.
    pub(super) fn apply_embed_overlay(
        &self,
        ids_dev: DevicePtr,
        seq_slot: DevicePtr,
        out: DevicePtr,
        num_tokens: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(set) = self.overlays.as_ref() else {
            return Ok(()); // feature off — one Option check, no launch.
        };
        if self.overlay_kernels.embed_overlay.0 == 0 || num_tokens == 0 {
            return Ok(());
        }
        let active = self.overlay_active_slot();
        if seq_slot.is_null() && active < 0 {
            return Ok(()); // uniform base request — nothing to override.
        }
        token_overlay::embed_overlay_routed(
            self.gpu.as_ref(),
            self.overlay_kernels.embed_overlay,
            ids_dev,
            seq_slot,
            active,
            set.embed_slot_map_table,
            set.embed_rows_table,
            set.embed_n_table,
            out,
            num_tokens,
            self.config.hidden_size as u32,
            set.vocab,
            stream,
        )
    }

    /// Overlay the logit columns of overridden vocab ids in place on `logits`
    /// (`[m, vocab]`). Inserted AFTER the base lm_head projection, BEFORE
    /// softcap. `is_fp32` selects the f32-logits kernel (single-token decode
    /// with `use_fp32_logits`); otherwise the bf16-logits kernel.
    pub(super) fn apply_lmhead_overlay(
        &self,
        hidden: DevicePtr,
        seq_slot: DevicePtr,
        logits: DevicePtr,
        m: u32,
        is_fp32: bool,
        stream: u64,
    ) -> Result<()> {
        let Some(set) = self.overlays.as_ref() else {
            return Ok(());
        };
        if set.max_n_override == 0 || m == 0 {
            return Ok(()); // no slot overrides lm_head (embed-only correction).
        }
        let kernel = if is_fp32 {
            self.overlay_kernels.lmhead_overlay_f32
        } else {
            self.overlay_kernels.lmhead_overlay_bf16
        };
        if kernel.0 == 0 {
            return Ok(());
        }
        let active = self.overlay_active_slot();
        if seq_slot.is_null() && active < 0 {
            return Ok(());
        }
        token_overlay::lmhead_overlay_routed(
            self.gpu.as_ref(),
            kernel,
            hidden,
            seq_slot,
            active,
            set.lmhead_rows_table,
            set.lmhead_ids_table,
            set.n_override_table,
            logits,
            m,
            set.max_n_override,
            self.config.hidden_size as u32,
            self.config.vocab_size as u32,
            stream,
        )
    }
}
