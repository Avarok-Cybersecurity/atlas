// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime LoRA slot swap: RDMA peer swap/promote (Tasks #26/#27) and the
//! disk swap sibling. Split from `impl_lora.rs` (500-LoC cap); the install
//! walk / rotation live there.

use anyhow::{Context, Result};

use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    /// RDMA-swap the adapter named `adapter_name` (staged on `$ATLAS_LORA_PEER`
    /// at `adapter_id`) INTO pool `slot`, in place, then make it that slot's
    /// resident adapter. Byte-identical to a disk pack (the loader does the same
    /// F16/F32→BF16 convert + B row-repack). MUST be called at a scheduler
    /// QUIESCENT point (no in-flight decode reading `slot`). Re-zeroes the slot
    /// sub-region first (a reused slot may hold the prior adapter's bytes), then
    /// rebuilds the slot's `LoraLayerWeights` with the NEW adapter's r/scale —
    /// re-installing if the swapped slot is currently active. Requires rotation
    /// armed (`ATLAS_LORA_ROTATE`/`$ATLAS_LORA_PEER`) so decode is eager.
    #[cfg(feature = "cuda")]
    pub fn swap_lora_slot_from_peer(
        &mut self,
        peer_addr: &str,
        adapter_id: &str,
        adapter_name: &str,
        slot: usize,
        peft: atlas_core::config::PeftAdapterConfig,
    ) -> Result<()> {
        use crate::lora::rdma_stage;

        let (pool, max_rank, max_loras) = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA RDMA swap: no adapter pool loaded"))?;
            (lw.pool, lw.max_rank, lw.max_loras)
        };
        if !self.lora_rotatable {
            anyhow::bail!(
                "LoRA RDMA swap needs rotation armed (set $ATLAS_LORA_PEER or \
                 ATLAS_LORA_ROTATE=1 so decode runs eager)"
            );
        }
        if slot >= max_loras {
            anyhow::bail!("LoRA RDMA swap: slot {slot} >= max_loras {max_loras}");
        }
        // Task #25 busy-slot refusal: bail BEFORE the destructive memset/stage
        // below so a refused swap leaves the slot's bytes + identity untouched.
        // Replacing an adapter while sequences are mid-decode on it would corrupt
        // their KV and replay a captured graph over swapped pool bytes.
        {
            let busy = self.lora.as_ref().unwrap().slot_ref_count(slot);
            if busy > 0 {
                anyhow::bail!(
                    "LoRA RDMA swap REFUSED: slot {slot} has {busy} in-flight \
                     sequence(s) (ref_count>0); cannot replace an adapter mid-decode"
                );
            }
        }

        // 1) Fetch manifest + build landing targets (classify + slot offsets).
        let manifest = rdma_stage::fetch_adapter_manifest(peer_addr, adapter_id)?;
        let targets =
            rdma_stage::build_land_targets(&manifest, &self.config, pool, slot, max_rank)?;

        // 2) Re-zero the slot sub-region (in-place reload of a dirty slot),
        //    then RDMA-land the adapter's A/B into it.
        let slot_bytes = rdma_stage::slot_bytes(&self.config, max_rank);
        let slot_base = DevicePtr(pool.0 + (slot * slot_bytes) as u64);
        self.gpu.memset(slot_base, 0, slot_bytes)?;
        let loader =
            spark_storage::RdmaLoraLoader::new(peer_addr.to_string(), adapter_id.to_string());
        loader.stage_into_slot(self.gpu.as_ref(), &targets)?;

        // 3) Rebuild the slot's per-layer pairs (new r/scale), stamp the slot.
        let layers =
            rdma_stage::rebuild_slot_layers(&targets, &self.config, &peft, pool, slot, max_rank)?;
        // Task #26: refresh this slot's a/b pointer tables + scale table from the
        // freshly-staged adapter's actual coverage BEFORE `peft`/`layers` are moved
        // into the slot stamp — a promoted adapter with different module coverage
        // than the evicted one would otherwise keep a stale bgmv route entry for the
        // reused cache slot (missed / wrong-scaled delta). Same fix as the disk swap.
        self.lora.as_ref().unwrap().refresh_slot_tables(
            slot,
            &layers,
            peft.scaling(),
            self.gpu.as_ref(),
        )?;
        {
            let lw = self.lora.as_mut().unwrap();
            let s = lw
                .slots
                .get_mut(slot)
                .ok_or_else(|| anyhow::anyhow!("LoRA RDMA swap: slot {slot} not resident"))?;
            s.name = adapter_name.to_string();
            s.adapter_config = peft;
            s.layers = layers;
            // Task #25: contents changed → bump generation so this re-staged slot
            // yields a FRESH adapter_id (a later same-name request misses the
            // stale prior KV). Pure rotate does NOT reach here.
            s.generation = s.generation.wrapping_add(1);
        }

        // 4) If the swapped slot is active, re-install onto the layer structs.
        let active = self.lora.as_ref().unwrap().active;
        if active == slot {
            let installed_layers = self.lora.as_ref().unwrap().slots[slot].layers.clone();
            let tables = self.lora.as_ref().unwrap().tables.clone();
            let scale_table = self.lora.as_ref().unwrap().scale_table;
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&installed_layers, kernels, &tables, scale_table)?;
            self.lora.as_mut().unwrap().name = adapter_name.to_string();
            self.destroy_lora_decode_graphs();
        }
        tracing::info!(
            "LoRA RDMA swap: '{adapter_name}' landed in slot {slot} \
             ({} targets, active_slot={active})",
            targets.len()
        );
        Ok(())
    }

    /// Task #27 (demand-driven promotion): promote the adapter `adapter_name`
    /// (staged on `peer_addr` at `adapter_id`) from the peer into a CACHE-region
    /// pool slot and make it ACTIVE, returning `(slot, evicted_name)`. Runs on
    /// the scheduler thread at a QUIESCENT point (the only place per-slot
    /// `ref_count` is authoritative). Victim policy (pure `select_victim_slot`):
    /// a never-filled placeholder first, else the LRU idle (`ref_count == 0`)
    /// cache slot, else `POOL_FULL` (retryable — a busy slot is NEVER evicted).
    /// The underlying [`Self::swap_lora_slot_from_peer`] re-checks `ref_count>0`
    /// and bails as a backstop, and bumps the slot generation so #24 KV stays
    /// correct. Making the promoted slot active mirrors the rotate/load control
    /// plane so the delta actually applies under batch-1 (the per-slot bgmv route
    /// tables are still dormant — compute reads the installed active adapter).
    #[cfg(feature = "cuda")]
    pub fn promote_lora_slot_from_peer(
        &mut self,
        peer_addr: &str,
        adapter_id: &str,
        adapter_name: &str,
        peft: atlas_core::config::PeftAdapterConfig,
    ) -> Result<(usize, Option<String>)> {
        // 1) Snapshot the cache region + pick a victim (pure policy).
        let (slot, evicted) = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA promote: no adapter pool loaded"))?;
            let views = lw.cache_slot_views();
            let slot = crate::lora::select_victim_slot(&views).map_err(|e| match e {
                crate::lora::VictimError::PoolFull => anyhow::anyhow!(
                    "POOL_FULL: all {} cache slot(s) are busy (ref_count>0); retry",
                    views.len()
                ),
            })?;
            // The name being replaced (if the victim already held an adapter) so
            // the caller can drop the stale name->slot overlay entry.
            let evicted = lw
                .slots
                .get(slot)
                .map(|s| s.name.clone())
                .filter(|n| !n.is_empty());
            (slot, evicted)
        };

        // 2) RDMA-stage into the victim slot (re-checks ref_count>0, bumps gen).
        self.swap_lora_slot_from_peer(peer_addr, adapter_id, adapter_name, slot, peft)?;

        // 3) Make the promoted slot ACTIVE so its delta applies (batch-1 honest).
        //    `swap_lora_slot_from_peer` already re-installed if the victim WAS the
        //    active slot; otherwise re-point the installed pairs onto it here.
        let already_active = self.lora.as_ref().unwrap().active == slot;
        if !already_active {
            let (layers, tables, scale_table) = {
                let lw = self.lora.as_mut().unwrap();
                lw.active = slot;
                lw.name = lw.slots[slot].name.clone();
                lw.adapter_config = lw.slots[slot].adapter_config.clone();
                (
                    lw.slots[slot].layers.clone(),
                    lw.tables.clone(),
                    lw.scale_table,
                )
            };
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
            self.destroy_lora_decode_graphs();
        }
        // Stamp the freshly-promoted slot as most-recently-used so a back-to-back
        // promote of a DIFFERENT cold adapter picks an older victim, not this one,
        // before the request that triggered this promote has acquired its ref.
        self.lora.as_ref().unwrap().touch_slot(slot);
        tracing::info!(
            "LoRA promote: '{adapter_name}' hot in cache slot {slot} \
             (evicted={:?}), now active",
            evicted
        );
        Ok((slot, evicted))
    }

    /// Demand-driven DISK promotion (no RDMA/peer): load the adapter at
    /// `adapter_dir` (named `name`) into a CACHE-region pool slot (LRU victim)
    /// and make it ACTIVE, returning `(slot, evicted_name)`. Local-disk analog
    /// of [`Self::promote_lora_slot_from_peer`] — same victim policy (pure
    /// `select_victim_slot`: never-filled placeholder first, else the LRU idle
    /// (`ref_count == 0`) cache slot, else `POOL_FULL`, retryable — a busy slot
    /// is NEVER evicted) and the same make-active control plane, but the inner
    /// swap reads the adapter from disk instead of the peer.
    /// [`Self::swap_lora_slot_from_disk`] re-parses the dir's
    /// `adapter_config.json` (so no `peft` arg), re-checks `ref_count>0` as a
    /// backstop, and bumps the slot generation so #24 KV stays correct. Runs on
    /// the scheduler thread at a QUIESCENT point; requires rotation armed
    /// (`ATLAS_LORA_ROTATE=1`) — the inner swap enforces it.
    pub fn promote_lora_slot_from_disk(
        &mut self,
        adapter_dir: &std::path::Path,
        name: &str,
    ) -> Result<(usize, Option<String>)> {
        // 1) Snapshot the cache region + pick a victim (pure policy).
        let (slot, evicted) = {
            let lw = self
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LoRA disk promote: no adapter pool loaded"))?;
            let views = lw.cache_slot_views();
            let slot = crate::lora::select_victim_slot(&views).map_err(|e| match e {
                crate::lora::VictimError::PoolFull => anyhow::anyhow!(
                    "POOL_FULL: all {} cache slot(s) are busy (ref_count>0); retry",
                    views.len()
                ),
            })?;
            // The name being replaced (if the victim already held an adapter) so
            // the caller can drop the stale name->slot overlay entry.
            let evicted = lw
                .slots
                .get(slot)
                .map(|s| s.name.clone())
                .filter(|n| !n.is_empty());
            (slot, evicted)
        };

        // 2) Disk-load into the victim slot (re-checks ref_count>0, bumps gen).
        self.swap_lora_slot_from_disk(adapter_dir, name, slot)?;

        // 3) Make the promoted slot ACTIVE so its delta applies (batch-1 honest).
        //    `swap_lora_slot_from_disk` already re-installed if the victim WAS the
        //    active slot; otherwise re-point the installed pairs onto it here.
        let already_active = self.lora.as_ref().unwrap().active == slot;
        if !already_active {
            let (layers, tables, scale_table) = {
                let lw = self.lora.as_mut().unwrap();
                lw.active = slot;
                lw.name = lw.slots[slot].name.clone();
                lw.adapter_config = lw.slots[slot].adapter_config.clone();
                (
                    lw.slots[slot].layers.clone(),
                    lw.tables.clone(),
                    lw.scale_table,
                )
            };
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
            self.destroy_lora_decode_graphs();
        }
        // Stamp the freshly-promoted slot as most-recently-used so a back-to-back
        // promote of a DIFFERENT cold adapter picks an older victim, not this one,
        // before the request that triggered this promote has acquired its ref.
        self.lora.as_ref().unwrap().touch_slot(slot);
        tracing::info!(
            "LoRA disk-promote: '{name}' hot in cache slot {slot} \
             (evicted={:?}), now active",
            evicted
        );
        Ok((slot, evicted))
    }

    /// Disk-swap the adapter at `adapter_dir` INTO pool `slot`, in place, then
    /// make it that slot's resident adapter (re-installing onto the layer structs
    /// if the slot is currently active). The local-disk analog of
    /// [`Self::swap_lora_slot_from_peer`] — same audit + pack + re-point, no RDMA.
    /// This is the pool-size-1 dynamic-load path: load a DIFFERENT adapter into
    /// the single slot at runtime (per-request weight change). MUST be called at
    /// a scheduler QUIESCENT point (no in-flight decode reading `slot`) and needs
    /// rotation armed (`ATLAS_LORA_ROTATE=1`/`$ATLAS_LORA_PEER`) so decode is
    /// eager and no captured graph replays the swapped slot's stale pointers.
    pub fn swap_lora_slot_from_disk(
        &mut self,
        adapter_dir: &std::path::Path,
        name: &str,
        slot: usize,
    ) -> Result<()> {
        if !self.lora_rotatable {
            anyhow::bail!(
                "LoRA disk swap needs rotation armed (set ATLAS_LORA_ROTATE=1 so \
                 decode runs eager); a single startup adapter with no rotation env \
                 is baked into the decode graph and a re-point would replay stale"
            );
        }
        // Task #25 busy-slot refusal: fail fast (before the disk load + pack)
        // when the target slot has in-flight sequences. `pack_store_into_slot`
        // re-checks under `&mut lw` right before the destructive memset (the
        // authoritative gate); this early check just avoids the wasted load.
        if let Some(lw) = self.lora.as_ref() {
            let busy = lw.slot_ref_count(slot);
            if busy > 0 {
                anyhow::bail!(
                    "LoRA disk swap REFUSED: slot {slot} has {busy} in-flight \
                     sequence(s) (ref_count>0); cannot replace an adapter mid-decode"
                );
            }
        }
        // Parse the adapter's own PEFT config (scaling read per adapter, never
        // defaulted) — the same hard-fail parser the startup path uses.
        let cfg_path = adapter_dir.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let peft = atlas_core::config::parse_peft_adapter_config(&raw)
            .with_context(|| format!("parse {}", cfg_path.display()))?;
        // Load the adapter's A/B into a device WeightStore (host F16/F32→BF16),
        // then pack it into the slot (same layout as a startup pack).
        let store = spark_runtime::weights::adapter::load_adapter_safetensors(
            adapter_dir,
            self.gpu.as_ref(),
            0,
        )
        .context("load LoRA adapter weights for disk swap")?;
        let layers = {
            let lw = self
                .lora
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("LoRA disk swap: no adapter pool loaded"))?;
            crate::lora::pack_store_into_slot(
                lw,
                slot,
                name,
                &store,
                &peft,
                &self.config,
                self.gpu.as_ref(),
            )?
        };
        // If the swapped slot is the active one, re-install onto the layer structs
        // so subsequent requests apply the new adapter's delta.
        let active = self.lora.as_ref().unwrap().active;
        if active == slot {
            let tables = self.lora.as_ref().unwrap().tables.clone();
            let scale_table = self.lora.as_ref().unwrap().scale_table;
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            self.install_lora_layers(&layers, kernels, &tables, scale_table)?;
            self.lora.as_mut().unwrap().name = name.to_string();
            self.destroy_lora_decode_graphs();
        }
        tracing::info!(
            "LoRA disk swap: '{name}' packed into slot {slot} (r={}, active_slot={active})",
            peft.r
        );
        Ok(())
    }
}
