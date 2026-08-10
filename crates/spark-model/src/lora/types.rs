// SPDX-License-Identifier: AGPL-3.0-only

//! LoRA adapter type surface: the module/AB enums, per-layer + per-slot weight
//! structs, the loaded [`LoraWeights`] set with its ref-count/LRU/generation
//! bookkeeping, and the pure victim-selection view types. Split out of the
//! former monolithic `lora/mod.rs` (SDD seam: TYPES) — visibility unchanged.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use anyhow::Result;
use atlas_core::config::PeftAdapterConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::*;
use crate::layers::ops::lora_delta::LoraPair;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoraModule {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

impl LoraModule {
    pub const ALL: [LoraModule; 7] = [
        Self::QProj,
        Self::KProj,
        Self::VProj,
        Self::OProj,
        Self::GateProj,
        Self::UpProj,
        Self::DownProj,
    ];

    /// PEFT suffix name (target_modules vocabulary).
    pub fn peft_name(&self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
        }
    }

    /// (out_dim, in_dim) of the base projection. Holo-3.1-0.8B (verified
    /// against the checkpoint header): k/v `[512,1024]`, o `[1024,2048]`,
    /// gate/up `[3584,1024]`, down `[1024,3584]`.
    ///
    /// q_proj: on a gated-attention model (`attn_gated`) the raw projection
    /// emits the interleaved `[Q|gate]` at width `2·q_heads·head_dim` (the
    /// FULL width the PEFT `lora_B` was trained against — verified `[8192,16]`
    /// on holo-3.1-35b); ungated q is `q_heads·head_dim`.
    pub fn dims(&self, cfg: &atlas_core::config::ModelConfig) -> (usize, usize) {
        let h = cfg.hidden_size;
        match self {
            Self::QProj => (
                (if cfg.attn_gated { 2 } else { 1 }) * cfg.num_attention_heads * cfg.head_dim,
                h,
            ),
            Self::KProj | Self::VProj => (cfg.num_key_value_heads * cfg.head_dim, h),
            Self::OProj => (h, cfg.num_attention_heads * cfg.head_dim),
            Self::GateProj | Self::UpProj => (cfg.intermediate_size, h),
            Self::DownProj => (h, cfg.intermediate_size),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdapterAb {
    A = 0,
    B = 1,
}

/// One full-attention layer's adapted modules. `None` = module not adapted.
/// Pairs are the CANONICAL [`LoraPair`] from `layers::ops::lora_delta`
/// (Copy — installed by copy into the layer structs at model build).
///
/// `Clone` (LoraPair is Copy) so a slot's layers can be re-installed onto the
/// layer structs on a runtime rotation (`set_active_lora`).
#[derive(Clone)]
pub struct LoraLayerWeights {
    pub layer_idx: usize,
    pub q_proj: Option<LoraPair>,
    pub k_proj: Option<LoraPair>,
    pub v_proj: Option<LoraPair>,
    pub o_proj: Option<LoraPair>,
    pub gate_proj: Option<LoraPair>,
    pub up_proj: Option<LoraPair>,
    pub down_proj: Option<LoraPair>,
}

/// One packed pool slot: a resident adapter's own name/config + its per-layer
/// pairs (a/b DevicePtrs into that slot's byte sub-region of the shared pool).
/// `layers` is GLOBAL-layer-indexed (len = num_hidden_layers), the same index
/// the install walk uses.
#[derive(Clone)]
pub struct AdapterSlot {
    pub name: String,
    pub adapter_config: PeftAdapterConfig,
    pub layers: Vec<Option<LoraLayerWeights>>,
    /// Task #25 (slot generation): monotonic counter bumped every time this
    /// slot's CONTENTS are replaced (disk/RDMA swap-into-slot). Folded into the
    /// adapter identity ([`adapter_id_hash`]) so re-staging DIFFERENT weights
    /// under the SAME adapter name yields a FRESH id — a later request then
    /// misses the stale (previous-generation) prefix/KV instead of warm-hitting
    /// it. Init 0; gen 0 is a strict no-op in the fold so a slot's FIRST-load id
    /// (and the base sentinel) stay byte-identical to the pre-#25 (#24) value. A
    /// pure rotate (same weights re-pointed) does NOT bump.
    pub generation: u64,
}

/// One adapter to pack, for the multi-adapter entry point. `store` is the
/// adapter's on-device BF16 `WeightStore` (host F16/F32→BF16 already done by
/// `spark_runtime::weights::adapter::load_adapter_safetensors`).
pub struct LoraAdapterInput<'a> {
    pub name: String,
    pub store: &'a WeightStore,
    pub peft: PeftAdapterConfig,
}

/// The loaded adapter set: one fixed-address rank-padded pool holding up to
/// `max_loras` equal-size slots, one [`AdapterSlot`] per resident adapter, and
/// per-module `[max_loras]` device u64 pointer tables (the frozen M2 BGMV
/// contract — filled index k for each packed slot, NULL for the rest).
///
/// Single-adapter runs pack exactly one slot (`slots.len() == 1`, `active == 0`)
/// — byte-identical to the pre-multi-adapter path. `name`/`adapter_config` mirror
/// the ACTIVE slot for logs/status; the install walk reads [`Self::active_layers`].
pub struct LoraWeights {
    /// Name of the ACTIVE adapter (mirrors `slots[active].name`).
    pub name: String,
    /// Config of the ACTIVE adapter (mirrors `slots[active].adapter_config`).
    pub adapter_config: PeftAdapterConfig,
    pub max_rank: usize,
    pub max_loras: usize,
    /// One fixed-address allocation holding every padded A/B for every slot.
    pub pool: DevicePtr,
    pub pool_bytes: usize,
    /// The resident adapters, slot-indexed (`slots[k]` lives at pool byte
    /// offset `k * pool_slot_bytes`). `len() <= max_loras`.
    pub slots: Vec<AdapterSlot>,
    /// Index into `slots` of the currently-active adapter (0 at load).
    pub active: usize,
    /// key = (global_layer_idx, module) → (a_table, b_table); each table is
    /// a device `[max_loras]` u64 array, NULL (0) = base-only slot.
    pub tables: BTreeMap<(usize, LoraModule), (DevicePtr, DevicePtr)>,
    /// The parallel `[max_loras]` device f32 SCALE table the bgmv reads,
    /// indexed by slot: `scale_table[k]` = `slots[k].adapter_config.scaling()`
    /// (alpha/r, or alpha/√r under rsLoRA — the same per-adapter scale that
    /// rides each [`LoraPair`]), 0.0 for unpacked slots. Scale is per-ADAPTER
    /// (not per-module), so ONE table suffices. Built once at pool pack time
    /// alongside the a/b tables (load-time-fixed → graph-safe kernel arg).
    pub scale_table: DevicePtr,
    /// Task #25 (slot ref_count): per-slot in-flight-sequence count, one
    /// [`AtomicUsize`] per pool index (`len() == max_loras`, stable across
    /// swaps). A sequence acquires (`+1`) its resolved slot at prefill and
    /// releases (`-1`) at terminal free; a swap/rotate INTO a slot with
    /// `ref_count > 0` is REFUSED (you cannot replace an adapter mid-decode —
    /// it would corrupt in-flight KV and replay a captured graph over swapped
    /// pool bytes). Kept as a parallel Vec here (not on [`AdapterSlot`], which
    /// derives Clone and is cloned during install — `AtomicUsize` is not Clone);
    /// [`LoraWeights`] is deliberately non-Clone and already `Send + Sync`.
    /// Interior-mutable through `&self` (acquire/release run on the prefill/free
    /// `&self` paths); swaps read it under `&mut self` at a quiescent point.
    pub ref_counts: Vec<AtomicUsize>,
    /// Task #27 (demand-driven promotion): the PINNED/CACHE boundary. Slots
    /// `[0, pinned)` are the startup `--lora-adapter` set — advertised by
    /// `/v1/models`, resolved by the position-based `resolve_adapter_slot`, and
    /// NEVER an eviction victim. Slots `[pinned, max_loras)` are the promotion
    /// HOT CACHE (empty placeholders at load): a demand-promoted adapter lands
    /// in one of these. `pinned == slots-populated-at-load`.
    pub pinned: usize,
    /// Task #27: per-slot last-used LRU tick, one [`AtomicU64`] per pool index
    /// (`len() == max_loras`, parallel to `ref_counts`). Bumped in
    /// [`Self::acquire_slot`] on the RESOLVED index so victim selection ages the
    /// TRUE slot a request used (including `-1 -> active`). A cache slot with the
    /// smallest `last_used` among the `ref_count == 0` idle slots is the LRU
    /// eviction victim. Interior-mutable through `&self` like `ref_counts`.
    pub last_used: Vec<AtomicU64>,
    /// Task #27: monotonic source for `last_used` ticks (never wraps in
    /// practice). Bumped once per acquire.
    pub lru_tick: AtomicU64,
}

/// Task #27: a per-slot snapshot for the pure victim-selection policy. Taken on
/// the model thread at a scheduler-quiescent point (the only place `ref_count`
/// is authoritative), then handed to `select_victim_slot`.
#[derive(Clone, Copy, Debug)]
pub struct SlotView {
    /// `true` if this slot currently holds a (non-placeholder) adapter.
    pub filled: bool,
    /// In-flight sequence count (`0` == idle == evictable).
    pub ref_count: usize,
    /// LRU tick (larger = more recently used).
    pub last_used: u64,
}

/// Why a promotion cannot find a victim slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VictimError {
    /// Every cache slot is busy (`ref_count > 0`) — a RETRYABLE condition, never
    /// an eviction of an in-flight adapter.
    PoolFull,
}

impl LoraLayerWeights {
    /// Select this layer's [`LoraPair`] for `module` (`None` = module not
    /// adapted). The single source of truth for the (layer, module) → pair map,
    /// shared by [`select_routed_pair`] and [`LoraWeights::refresh_slot_tables`].
    pub fn module_pair(&self, module: LoraModule) -> Option<&LoraPair> {
        match module {
            LoraModule::QProj => self.q_proj.as_ref(),
            LoraModule::KProj => self.k_proj.as_ref(),
            LoraModule::VProj => self.v_proj.as_ref(),
            LoraModule::OProj => self.o_proj.as_ref(),
            LoraModule::GateProj => self.gate_proj.as_ref(),
            LoraModule::UpProj => self.up_proj.as_ref(),
            LoraModule::DownProj => self.down_proj.as_ref(),
        }
    }
}

impl LoraWeights {
    /// The active slot's per-layer pairs (GLOBAL-layer-indexed) — what the
    /// install walk copies onto the layer structs.
    pub fn active_layers(&self) -> &[Option<LoraLayerWeights>] {
        &self.slots[self.active].layers
    }

    /// #30 (routed-prefill precision): resolve a request's `adapter_slot`
    /// (`>= 0` → that slot, `-1` → active) and return `Some(resolved)` ONLY when it
    /// routes to a NON-active, in-range slot — the SINGLE source of truth for
    /// "this prefill must apply the request slot's pair via the dense path". Kept
    /// in exact lockstep with [`crate::model`]'s `upload_seq_slot_uniform`
    /// (`resolved == active` → `DevicePtr(0)` → installed-pair path). Returns
    /// `None` for an active/base request (byte-identical) and for out-of-range
    /// slots (bad request → installed active pair, never a panic).
    pub fn routed_prefill_slot(&self, adapter_slot: i32) -> Option<usize> {
        routed_prefill_slot_of(adapter_slot, self.active, self.slots.len())
    }

    /// Resolve an adapter NAME to its slot index (for runtime rotation).
    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.slots.iter().position(|s| s.name == name)
    }

    /// All resident adapter names in slot order (for `/v1/models`).
    pub fn adapter_names(&self) -> Vec<String> {
        self.slots.iter().map(|s| s.name.clone()).collect()
    }

    /// Stable adapter_id (Task #24) for a pool slot request selector. `slot`
    /// follows the `SequenceState.adapter_slot` convention: `>= 0` selects that
    /// resident slot, `-1` means "defer to the installed active adapter" (so a
    /// default request keys under whatever adapter is actually active — matching
    /// `build_seq_slot_host`'s `-1 -> active` resolution). The id is the NAME
    /// hash, resolved at prefill time (active may rotate between HTTP resolve and
    /// prefill). Out-of-range slots fall back to the base sentinel `0`.
    pub fn adapter_id_for_slot(&self, slot: i32) -> u64 {
        let resolved = if slot >= 0 {
            slot as usize
        } else {
            self.active
        };
        match self.slots.get(resolved) {
            Some(s) => adapter_id_hash(&s.name, s.generation),
            None => 0,
        }
    }

    /// Task #25: resolve `slot` (`>= 0` → that slot, `-1` → active) to a concrete
    /// pool index and `+1` its ref_count, returning the RESOLVED index so the
    /// caller can release EXACTLY that index later (immune to an intervening
    /// rotate changing `active`). Returns `-1` — "nothing acquired" — when the
    /// resolved index is out of range (bad request slot); the active slot is
    /// always in range so `-1 -> active` never no-ops here for a loaded pool.
    pub fn acquire_slot(&self, slot: i32) -> i32 {
        let resolved = if slot >= 0 {
            slot as usize
        } else {
            self.active
        };
        match self.ref_counts.get(resolved) {
            Some(rc) => {
                rc.fetch_add(1, Ordering::AcqRel);
                // Task #27: stamp the RESOLVED slot as most-recently-used so the
                // LRU victim policy ages the slot a request actually touched
                // (including `-1 -> active`). Ticks are strictly increasing.
                if let Some(lu) = self.last_used.get(resolved) {
                    let t = self.lru_tick.fetch_add(1, Ordering::Relaxed) + 1;
                    lu.store(t, Ordering::Relaxed);
                }
                resolved as i32
            }
            None => -1,
        }
    }

    /// Task #27: stamp `slot` as most-recently-used WITHOUT taking a ref. Called
    /// right after a promote so a freshly-staged (ref_count==0) slot is NOT the
    /// immediate LRU victim of a back-to-back promote before its own request has
    /// acquired — otherwise two distinct cold adapters promoted in quick
    /// succession would collide on the same slot (the second evicting the first).
    pub fn touch_slot(&self, slot: usize) {
        if let Some(lu) = self.last_used.get(slot) {
            let t = self.lru_tick.fetch_add(1, Ordering::Relaxed) + 1;
            lu.store(t, Ordering::Relaxed);
        }
    }

    /// Task #27: current LRU tick of pool `slot` (larger = more recently
    /// acquired). Out-of-range → 0 (never used).
    pub fn slot_last_used(&self, slot: usize) -> u64 {
        self.last_used
            .get(slot)
            .map(|lu| lu.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Task #26: refresh `slot`'s cell in the `[max_loras]` a/b pointer tables +
    /// the per-slot scale table from `layers` (the just-staged adapter's actual
    /// per-module coverage). A re-staged adapter whose module coverage DIFFERS
    /// from the evicted one would otherwise keep a STALE table entry: the
    /// bgmv-routed path would SKIP a module the new adapter adds (`a_table[slot]`
    /// stale-NULL → missed delta), keep applying an evicted module (stale non-NULL
    /// → wrong delta), or use the wrong per-slot scale. Shared by BOTH the disk
    /// swap (`pack_store_into_slot`) and the RDMA swap (`swap_lora_slot_from_peer`).
    /// Only the `[slot]` cell of each fixed-address device array is rewritten.
    pub fn refresh_slot_tables(
        &self,
        slot: usize,
        layers: &[Option<LoraLayerWeights>],
        scale: f32,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        for ((layer, module), (a_dev, b_dev)) in &self.tables {
            let pair = layers
                .get(*layer)
                .and_then(|o| o.as_ref())
                .and_then(|lw| match module {
                    LoraModule::QProj => lw.q_proj.as_ref(),
                    LoraModule::KProj => lw.k_proj.as_ref(),
                    LoraModule::VProj => lw.v_proj.as_ref(),
                    LoraModule::OProj => lw.o_proj.as_ref(),
                    LoraModule::GateProj => lw.gate_proj.as_ref(),
                    LoraModule::UpProj => lw.up_proj.as_ref(),
                    LoraModule::DownProj => lw.down_proj.as_ref(),
                });
            let (a_ptr, b_ptr) = pair.map(|p| (p.a.weight.0, p.b.weight.0)).unwrap_or((0, 0));
            gpu.copy_h2d(&a_ptr.to_le_bytes(), DevicePtr(a_dev.0 + (slot * 8) as u64))?;
            gpu.copy_h2d(&b_ptr.to_le_bytes(), DevicePtr(b_dev.0 + (slot * 8) as u64))?;
        }
        if self.scale_table.0 != 0 {
            gpu.copy_h2d(
                &scale.to_le_bytes(),
                DevicePtr(self.scale_table.0 + (slot * 4) as u64),
            )?;
        }
        Ok(())
    }

    /// Task #27: snapshot the CACHE region `[pinned, max_loras)` as
    /// `(slot_index, SlotView)` for `select_victim_slot`. `filled` = the slot
    /// holds a non-placeholder adapter (non-empty name). Read on the model
    /// thread at a quiescent point.
    pub fn cache_slot_views(&self) -> Vec<(usize, SlotView)> {
        (self.pinned..self.max_loras)
            .map(|k| {
                let filled = self.slots.get(k).is_some_and(|s| !s.name.is_empty());
                (
                    k,
                    SlotView {
                        filled,
                        ref_count: self.slot_ref_count(k),
                        last_used: self.slot_last_used(k),
                    },
                )
            })
            .collect()
    }

    /// Task #25: release a ref previously taken by [`Self::acquire_slot`], by the
    /// RESOLVED index it returned. `-1` (nothing acquired) is a no-op. Saturating
    /// so a stray double-release can never wrap the counter below 0.
    pub fn release_slot(&self, resolved: i32) {
        if resolved < 0 {
            return;
        }
        if let Some(rc) = self.ref_counts.get(resolved as usize) {
            let _ = rc.fetch_update(Ordering::Release, Ordering::Acquire, |v| {
                Some(v.saturating_sub(1))
            });
        }
    }

    /// Task #25: current in-flight ref_count of pool `slot` (the exact read the
    /// swap busy-slot gate branches on). Out-of-range → 0.
    pub fn slot_ref_count(&self, slot: usize) -> usize {
        self.ref_counts
            .get(slot)
            .map(|rc| rc.load(Ordering::Acquire))
            .unwrap_or(0)
    }
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
