// SPDX-License-Identifier: AGPL-3.0-only

//! Model-level LoRA adapter lifecycle: startup install (`set_lora_weights` +
//! the per-layer install walk), per-request slot resolution (Tasks #24/#25),
//! runtime rotation (`rotate_lora_to`), RDMA/disk slot swap, and the
//! rotate/swap decode-graph invalidation drain. Split from `impl_b3.rs`
//! (500-LoC cap).

use anyhow::{Context, Result};

use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    /// Install a startup-static LoRA adapter (post-construction, mirroring
    /// [`Self::set_dflash_proposer`]). Walks the model layers by GLOBAL
    /// index — `LoraWeights.layers` is indexed the same way — and copies
    /// each adapted layer's K/V/O (+ optional gate/up/down) pairs into the
    /// `Qwen3AttentionLayer` (which routes FFN pairs into its dense FFN
    /// component). M0: layers only STORE the adapter; base output is
    /// unchanged until the M1 compute insertions read it.
    /// Task #24: stable adapter_id for a per-request pool-slot selector. Returns
    /// the base sentinel `0` when no LoRA pool is resident (byte-identical base),
    /// else the NAME-derived id of the resolved slot (`-1 -> active`). Resolved
    /// here at prefill time because `LoraWeights.active` can rotate between HTTP
    /// request resolution and prefill.
    pub fn adapter_id_for_slot(&self, slot: i32) -> u64 {
        match self.lora.as_ref() {
            Some(lw) => lw.adapter_id_for_slot(slot),
            None => 0,
        }
    }

    /// Task #25: acquire a per-slot ref for a sequence beginning to use its
    /// adapter (called at prefill, resolving `-1 -> active` exactly like
    /// [`Self::adapter_id_for_slot`]). Returns the RESOLVED pool index the ref
    /// was taken on — the caller stores it and releases EXACTLY that index at
    /// terminal free, so an intervening rotate changing `active` cannot make
    /// release hit a different counter. `-1` ("nothing acquired") when no LoRA
    /// pool is resident or the slot is out of range → byte-identical no-op base.
    pub fn acquire_adapter_slot(&self, slot: i32) -> i32 {
        match self.lora.as_ref() {
            Some(lw) => lw.acquire_slot(slot),
            None => -1,
        }
    }

    /// Task #25: release a ref acquired by [`Self::acquire_adapter_slot`], by the
    /// RESOLVED index it returned. `-1` and no-pool are no-ops (base path).
    pub fn release_adapter_slot(&self, resolved: i32) {
        if let Some(lw) = self.lora.as_ref() {
            lw.release_slot(resolved);
        }
    }

    /// Feature-1: resolve the MoE-LoRA fold decision for a single-request pass
    /// from that request's `adapter_slot`. See [`crate::layer::MoeLoraRoute`].
    /// Base / non-active requests skip (base tokens pay nothing); only the
    /// installed active adapter's request folds; a request for a different,
    /// non-installed adapter refuses loudly. No pool resident ⇒ `Fold` (inert:
    /// the fold hook no-ops on the layer's `self.lora == None`).
    pub(crate) fn moe_lora_route(&self, adapter_slot: i32) -> crate::layer::MoeLoraRoute {
        let (active, has) = match self.lora.as_ref() {
            Some(lw) => (lw.active as i32, true),
            None => (-1, false),
        };
        crate::lora::resolve_moe_lora_route(adapter_slot, active, has)
    }

    /// Read the per-decode MoE route stamped at the `Model` entry (decode/verify
    /// `ForwardContext`s use this instead of a hardcoded `Fold`).
    pub(crate) fn decode_moe_route(&self) -> crate::layer::MoeLoraRoute {
        match self
            .decode_moe_route
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => crate::layer::MoeLoraRoute::Skip,
            2 => crate::layer::MoeLoraRoute::Refuse,
            _ => crate::layer::MoeLoraRoute::Fold,
        }
    }

    fn store_decode_moe_route(&self, route: crate::layer::MoeLoraRoute) {
        let v = match route {
            crate::layer::MoeLoraRoute::Skip => 0,
            crate::layer::MoeLoraRoute::Fold => 1,
            crate::layer::MoeLoraRoute::Refuse => 2,
        };
        self.decode_moe_route
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Stamp the per-decode MoE route from a single request's `adapter_slot`.
    pub(crate) fn stamp_decode_moe_single(&self, adapter_slot: i32) {
        self.store_decode_moe_route(self.moe_lora_route(adapter_slot));
    }

    /// Stamp from a decode batch — 3-way reduction over the rows' per-seq routes
    /// (SOLID Incr-4). Any row routing to a NON-active adapter (`Refuse`) stamps
    /// `Refuse`: the single-active phase-1 fold cannot honor a second adapter's
    /// identity (the device gather-BGMV checks only the SIGN of the per-row map,
    /// so a non-active slot would silently fold the ACTIVE adapter's tables), so
    /// `decode_batch_compute_main` bails host-side before graph lookup. Else any
    /// adapter-owning row stamps `Fold` (the batched per-row map skips base rows
    /// individually, so base seqs in a mixed batch still decode clean). Only when
    /// EVERY row is base (or an empty batch) does it stamp `Skip` (nothing to
    /// fold; base decode stays byte-identical).
    pub(crate) fn stamp_decode_moe_batch(&self, seqs: &[&mut crate::traits::SequenceState]) {
        use crate::layer::MoeLoraRoute;
        let mut any_fold = false;
        for s in seqs.iter() {
            match self.moe_lora_route(s.adapter_slot) {
                MoeLoraRoute::Refuse => {
                    self.store_decode_moe_route(MoeLoraRoute::Refuse);
                    return;
                }
                MoeLoraRoute::Fold => any_fold = true,
                MoeLoraRoute::Skip => {}
            }
        }
        self.store_decode_moe_route(if any_fold {
            MoeLoraRoute::Fold
        } else {
            MoeLoraRoute::Skip
        });
    }

    // The batched/mixed-decode Refuse guard is the PURE
    // `crate::lora::ensure_decode_route_servable` (unit-tested there), called
    // by both batched decode entries before graph lookup.

    pub fn set_lora_weights(&mut self, mut lora: Option<crate::lora::LoraWeights>) -> Result<()> {
        if let Some(ref lw) = lora {
            // eager-on-rotate: ONLY the global rotate/swap re-point path forces
            // eager decode. A multi-adapter pool no longer implies eager —
            // per-request routing (M2) is graph-safe by construction (the
            // per-seq slot buffer is per-step-uploaded to a stable address, the
            // pool tables are load-time-fixed), so decode graphs STAY captured
            // under routing. Equating slots.len()>1 with eager here would throw
            // away the entire point of batched routing.
            self.lora_rotatable = self.levers.lora_rotate || crate::lora::lora_peer_env().is_some();
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            // Clone the active slot's pairs (small; LoraPair is Copy) so the
            // install walk can hold a shared borrow while it &mut-borrows
            // `self.layers`. Clone the (Copy) pool table pointers + scale table
            // too so the routed batched-decode path can read them per layer.
            let active = lw.active_layers().to_vec();
            let tables = lw.tables.clone();
            let scale_table = lw.scale_table;
            let installed = self.install_lora_layers(&active, kernels, &tables, scale_table)?;
            // Task #27: `slots` is pre-sized to max_loras with empty cache
            // placeholders; report only the filled (named) adapters.
            let resident: Vec<String> = lw
                .adapter_names()
                .into_iter()
                .filter(|n| !n.is_empty())
                .collect();
            tracing::info!(
                "LoRA: {} adapter(s) resident [{}], active '{}' installed on \
                 {installed} layers (r={}, max_rank={}, max_loras={}, \
                 pool={:.1} MiB, rotatable={})",
                resident.len(),
                resident.join(", "),
                lw.name,
                lw.adapter_config.r,
                lw.max_rank,
                lw.max_loras,
                lw.pool_bytes as f64 / (1024.0 * 1024.0),
                self.lora_rotatable,
            );
        }
        // Feature-2 Stage 2: build the token-overlay tables from the Stage-1 raw
        // uploads now that the served embed/lm_head tables exist (they did NOT
        // at loader time — the two-stage build closes that ordering gap). Only
        // an adapter that actually shipped overlay tensors reaches the builder;
        // a run with no overlay leaves `self.overlays == None` (byte-identical).
        if let Some(ref mut lw) = lora {
            let raws = std::mem::take(&mut lw.overlay_raw);
            if raws.iter().any(|r| r.is_some()) {
                self.build_token_overlays(lw.max_loras, &raws)?;
            }
        }
        self.lora = lora;
        Ok(())
    }

    /// Stage 2 of the token-overlay build: row-diff each staged adapter's raw
    /// overlay tensors against the served embed/lm_head tables, compact the
    /// override rows, and materialize the `[max_loras]` device tables. Sets
    /// `self.overlays` only when some slot actually overrides a row.
    fn build_token_overlays(
        &mut self,
        max_loras: usize,
        raws: &[Option<crate::lora::OverlayRawSlot>],
    ) -> Result<()> {
        // Tie: lm_head aliases embed (shared buffer OR a quantized head derived
        // from embed) ⇒ the logit recompute reuses the embed override rows.
        let tied = self.lm_head_weight.weight.0 == self.embed_tokens.weight.0
            || self.lm_head_nvfp4.is_some()
            || self.lm_head_fp8.is_some();
        let vocab = self.config.vocab_size;
        let h = self.config.hidden_size;
        let served_embed = self.embed_tokens.weight;
        let served_lmhead = self.lm_head_weight.weight;
        let stream = self.gpu.default_stream();
        let mut overlays: Vec<Option<crate::lora::EmbedOverlay>> =
            (0..max_loras).map(|_| None).collect();
        for (k, raw) in raws.iter().enumerate() {
            if let Some(slot) = raw
                && k < max_loras
            {
                overlays[k] = crate::lora::build_overlay(
                    self.gpu.as_ref(),
                    &self.overlay_kernels,
                    slot,
                    served_embed,
                    served_lmhead,
                    vocab,
                    h,
                    tied,
                    stream,
                )?;
            }
        }
        let set =
            crate::lora::TokenOverlaySet::from_slots(self.gpu.as_ref(), overlays, max_loras, tied)?;
        if set.any_active() {
            tracing::info!(
                "LoRA overlay: token-overlay tables built (max_n_override={}, tied={})",
                set.max_n_override,
                tied,
            );
            self.overlays = Some(set);
        }
        Ok(())
    }

    /// Install one slot's per-layer pairs onto the layer structs (the shared
    /// walk used by both initial install and runtime rotation). `layers` is
    /// GLOBAL-layer-indexed. Returns the number of layers installed.
    pub(super) fn install_lora_layers(
        &mut self,
        layers: &[Option<crate::lora::LoraLayerWeights>],
        kernels: ops::lora_delta::LoraKernels,
        tables: &std::collections::BTreeMap<
            (usize, crate::lora::LoraModule),
            (spark_runtime::gpu::DevicePtr, spark_runtime::gpu::DevicePtr),
        >,
        scale_table: spark_runtime::gpu::DevicePtr,
    ) -> Result<usize> {
        use crate::lora::LoraModule;
        // Build the per-module routing table from the frozen pool tables + the
        // active-slot pair dims (k_in/n_out/max_rank identical across slots, so
        // the active pair supplies them). `None` when the module has no table
        // (base-only) — the bgmv apply site then no-ops for that module.
        let mk_route = |layer_idx: usize,
                        module: LoraModule,
                        pair: &Option<ops::lora_delta::LoraPair>|
         -> Option<ops::lora_delta::LoraRoute> {
            let p = pair.as_ref()?;
            let (a_table, b_table) = *tables.get(&(layer_idx, module))?;
            Some(ops::lora_delta::LoraRoute {
                a_table,
                b_table,
                scale_table,
                k_in: p.k_in,
                n_out: p.n_out,
                max_rank: p.max_rank,
            })
        };
        let mut installed = 0usize;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let Some(layer_weights) = layers.get(idx).and_then(|o| o.as_ref()) else {
                continue;
            };
            let has_moe = layer_weights.router.is_some()
                || layer_weights
                    .experts
                    .as_ref()
                    .is_some_and(|e| !e.is_empty());
            let has_attn_ffn = layer_weights.q_proj.is_some()
                || layer_weights.k_proj.is_some()
                || layer_weights.v_proj.is_some()
                || layer_weights.o_proj.is_some()
                || layer_weights.gate_proj.is_some()
                || layer_weights.up_proj.is_some()
                || layer_weights.down_proj.is_some();
            let any = layer
                .as_any_mut()
                .ok_or_else(|| anyhow::anyhow!("LoRA: adapted layer {idx} is not downcastable"))?;
            // Full-attention layer: attention + dense-FFN + MoE. Linear-attention
            // (GDN/SSM) layer: MoE ONLY — its attention projections are rejected at
            // classify, but its MoE FFN exists on every layer, so a real all-layer
            // MoE adapter routes its router/expert deltas here too.
            if let Some(attn) = any.downcast_mut::<crate::layers::Qwen3AttentionLayer>() {
                let attn_weights = ops::lora_delta::LoraAttnWeights {
                    // #30: the global layer index (from `self.layers.enumerate()`) —
                    // the key the request slot's GLOBAL-layer-indexed pairs use.
                    layer_idx: idx,
                    q: layer_weights.q_proj,
                    k: layer_weights.k_proj,
                    v: layer_weights.v_proj,
                    o: layer_weights.o_proj,
                    kernels,
                    q_route: mk_route(idx, LoraModule::QProj, &layer_weights.q_proj),
                    k_route: mk_route(idx, LoraModule::KProj, &layer_weights.k_proj),
                    v_route: mk_route(idx, LoraModule::VProj, &layer_weights.v_proj),
                    o_route: mk_route(idx, LoraModule::OProj, &layer_weights.o_proj),
                };
                let ffn_weights = if layer_weights.gate_proj.is_some()
                    || layer_weights.up_proj.is_some()
                    || layer_weights.down_proj.is_some()
                {
                    Some(ops::lora_delta::LoraFfnWeights {
                        gate: layer_weights.gate_proj,
                        up: layer_weights.up_proj,
                        down: layer_weights.down_proj,
                        kernels,
                    })
                } else {
                    None
                };
                attn.set_lora_weights(attn_weights, ffn_weights)?;
                if has_moe {
                    attn.set_moe_lora_weights(
                        layer_weights.router,
                        layer_weights.experts.clone().unwrap_or_default(),
                        kernels,
                        self.gpu.as_ref(),
                    )?;
                }
            } else if let Some(ssm) = any.downcast_mut::<crate::layers::Qwen3SsmLayer>() {
                if has_attn_ffn {
                    anyhow::bail!(
                        "LoRA: attention/dense-FFN delta on linear-attention layer {idx} — \
                         classify should have rejected this (only MoE router/experts are \
                         valid on GDN layers)"
                    );
                }
                if has_moe {
                    ssm.set_moe_lora_weights(
                        layer_weights.router,
                        layer_weights.experts.clone().unwrap_or_default(),
                        kernels,
                        self.gpu.as_ref(),
                    )?;
                }
            } else {
                anyhow::bail!(
                    "LoRA: adapted layer {idx} is neither a Qwen3AttentionLayer nor a \
                     Qwen3SsmLayer (loader/adapter layer-type mismatch)"
                );
            }
            installed += 1;
        }
        Ok(installed)
    }

    /// #28: drain + DESTROY every graph cache that bakes the installed-active
    /// LoRA pair pointers (decode, batched decode, K=2/3/4 verify, K=γ verify,
    /// fused decode+verify) on a rotate/swap. `GraphHandle` has no `Drop`, so a
    /// bare `.clear()` would LEAK the CUDA graphs. This drain is the rotation
    /// invalidation guard in this port (the compound `(slot, active_id)` graph
    /// re-key of the reference branch is deferred), so it MUST cover every
    /// cache or a stale replay would decode with swapped pool bytes. Runs at
    /// scheduler quiescence on the CUDA-bound model thread (like
    /// `free_sequence`'s destroys).
    pub(super) fn destroy_lora_decode_graphs(&self) {
        let drain = |name: &str, graphs: Vec<spark_runtime::gpu::GraphHandle>| {
            for g in graphs {
                if g.0 != 0
                    && let Err(e) = self.gpu.destroy_graph(g)
                {
                    tracing::warn!("LoRA graph clear: destroy {name}: {e:#}");
                }
            }
        };
        drain(
            "decode_graph",
            self.decode_graph.lock().drain().map(|(_, g)| g).collect(),
        );
        drain(
            "batch_decode_graph",
            self.batch_decode_graphs
                .lock()
                .0
                .drain()
                .map(|(_, (g, _))| g)
                .collect(),
        );
        drain(
            "verify2_graph",
            self.verify2_graph.lock().drain().map(|(_, g)| g).collect(),
        );
        drain(
            "verify3_graph",
            self.verify3_graph.lock().drain().map(|(_, g)| g).collect(),
        );
        drain(
            "verify4_graph",
            self.verify4_graph.lock().drain().map(|(_, g)| g).collect(),
        );
        drain(
            "verify_kgamma_graph",
            self.verify_kgamma_graph
                .lock()
                .drain()
                .map(|(_, (g, _))| g)
                .collect(),
        );
        drain(
            "fused_graph",
            self.fused_graph
                .lock()
                .drain()
                .map(|(_, (g, _))| g)
                .collect(),
        );
    }

    /// Runtime adapter rotation (eager-on-rotate). Selects the resident
    /// adapter named `name` as ACTIVE: re-points every layer's LoraPair (a/b
    /// DevicePtr + rank/scale) to that slot's sub-region, then clears the
    /// decode-graph caches defensively (empty under forced eager). MUST be
    /// called at a scheduler QUIESCENT point (no in-flight decode reading the
    /// old slot). Graph-safety rests on `lora_rotatable` forcing eager decode
    /// — this method never re-captures a graph.
    pub fn rotate_lora_to(&mut self, name: &str) -> Result<()> {
        let slot = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA rotation: no adapter loaded"))?;
            lw.slot_of(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "LoRA rotation: adapter '{name}' is not resident (have [{}])",
                    lw.adapter_names().join(", ")
                )
            })?
        };
        if !self.lora_rotatable {
            // A single startup adapter with no rotation env is baked into the
            // decode graph; re-pointing would be replayed stale. Refuse rather
            // than silently mis-serve.
            anyhow::bail!(
                "LoRA rotation not armed (single adapter, ATLAS_LORA_ROTATE unset); \
                 set ATLAS_LORA_ROTATE=1 (forces eager decode) to rotate at runtime"
            );
        }
        // #25 safety: rotation RE-INSTALLS the new slot's pairs onto the layer
        // structs, so any in-flight sequence still decoding on the OLD active
        // adapter (via the installed pair) would replay with the wrong delta.
        // Refuse while the current active slot has in-flight sequences — rotate
        // only at a scheduler-quiescent point (matches this method's contract).
        {
            let lw = self.lora.as_ref().unwrap();
            let cur = lw.active;
            if lw.slot_ref_count(cur) > 0 {
                anyhow::bail!(
                    "LoRA rotation refused: active slot {cur} has in-flight \
                     sequences (ref_count>0); rotate at a quiescent point"
                );
            }
        }
        // Re-point onto the new active slot.
        let (layers, active_name, r, tables, scale_table) = {
            let lw = self.lora.as_mut().unwrap();
            lw.active = slot;
            lw.name = lw.slots[slot].name.clone();
            lw.adapter_config = lw.slots[slot].adapter_config.clone();
            (
                lw.slots[slot].layers.clone(),
                lw.name.clone(),
                lw.adapter_config.r,
                lw.tables.clone(),
                lw.scale_table,
            )
        };
        let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
        let installed = self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
        // Defensive: drop any captured decode graphs so a stale-pointer replay
        // is impossible even if `lora_rotatable` were ever mis-derived. Under
        // forced eager these are already empty.
        self.destroy_lora_decode_graphs();
        tracing::info!(
            "LoRA rotation → slot {slot} '{active_name}' (r={r}) re-installed on \
             {installed} layers"
        );
        Ok(())
    }
}
