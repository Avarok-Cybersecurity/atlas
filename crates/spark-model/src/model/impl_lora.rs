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

    pub fn set_lora_weights(&mut self, lora: Option<crate::lora::LoraWeights>) -> Result<()> {
        if let Some(ref lw) = lora {
            // eager-on-rotate: ONLY the global rotate/swap re-point path forces
            // eager decode. A multi-adapter pool no longer implies eager —
            // per-request routing (M2) is graph-safe by construction (the
            // per-seq slot buffer is per-step-uploaded to a stable address, the
            // pool tables are load-time-fixed), so decode graphs STAY captured
            // under routing. Equating slots.len()>1 with eager here would throw
            // away the entire point of batched routing.
            self.lora_rotatable =
                crate::lora::lora_rotate_env() || crate::lora::lora_peer_env().is_some();
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
        self.lora = lora;
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
            let attn = layer
                .as_any_mut()
                .and_then(|a| a.downcast_mut::<crate::layers::Qwen3AttentionLayer>())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "LoRA: adapted layer {idx} is not a Qwen3AttentionLayer \
                         (loader/adapter layer-type mismatch)"
                    )
                })?;
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
                .drain()
                .map(|(_, g)| g)
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
                .map(|(_, g)| g)
                .collect(),
        );
        drain(
            "fused_graph",
            self.fused_graph.lock().drain().map(|(_, g)| g).collect(),
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
