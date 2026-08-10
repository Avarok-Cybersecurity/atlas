// SPDX-License-Identifier: AGPL-3.0-only

//! LoRA adapter loading: family allow-list check, per-adapter classify/shape
//! audit, the pool-slot pack loop, and the multi/single/disk-swap entry points.
//! Split out of the former monolithic `lora/mod.rs` (SDD seam: LOADING) —
//! visibility unchanged.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use anyhow::{Result, bail};
use atlas_core::config::{ModelConfig, PeftAdapterConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::*;
use crate::layers::ops::lora_delta::LoraPair;
use crate::weight_map::DenseWeight;

/// The v0 family allow-list, checked once per load. v0 is validated on the
/// Qwen3.5-family attention trunk — qwen3_5 DENSE (holo-3.1-0.8b), holo3_1_moe
/// (holo-3.1-35b-a3b, MoE), and qwen3_6_moe (Qwen3.6-35B-A3B, MoE). All route to
/// `Qwen35WeightLoader`, so their full-attention layers are `Qwen3AttentionLayer`
/// — what the install walk downcasts to (attention q/k/v/o; MoE expert MLPs +
/// SSM layers stay rejected by `classify_key`). Other families stay
/// rejected (no validated mapping). NOTE: `qwen3_5_moe` on disk is rewritten to
/// `qwen3_6_moe` at parse time (dispatch.rs, MRoPE MoE), so we gate the
/// post-dispatch name — the on-disk `qwen3_5_moe` never reaches here.
fn check_family(cfg: &ModelConfig) -> Result<()> {
    if !(cfg.is_qwen35_dense()
        || cfg.model_type == "holo3_1_moe"
        || cfg.model_type == "qwen3_6_moe")
    {
        bail!(
            "REJECT[unvalidated-family]: LoRA v0 is validated on qwen3_5 dense \
             (holo-3.1-0.8b), holo3_1_moe (holo-3.1-35b-a3b), and qwen3_6_moe \
             (Qwen3.6-35B-A3B) only; model_type='{}', num_experts={}",
            cfg.model_type,
            cfg.num_experts
        );
    }
    Ok(())
}

/// Classify + audit one adapter's tensors (unconsumed key = fatal; pair
/// completeness; A=[r,in]/B=[out,r] shapes; every `target_modules` entry
/// matched). Returns the (layer, module) → [a_key, b_key] map used to pack.
fn audit_adapter(
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    max_lora_rank: usize,
) -> Result<BTreeMap<(usize, LoraModule), [Option<String>; 2]>> {
    validate_peft_config(peft, max_lora_rank)?;

    // 1) classify EVERY adapter tensor — any unclassifiable/unsupported key
    //    is a hard error, which IS the "unconsumed adapter tensors fatal"
    //    audit direction.
    let mut found: BTreeMap<(usize, LoraModule), [Option<String>; 2]> = BTreeMap::new();
    for name in adapter_store.names() {
        let (layer, module, ab) = classify_key(name, cfg)?;
        let entry = found.entry((layer, module)).or_default();
        let slot = &mut entry[ab as usize];
        if slot.is_some() {
            bail!(
                "REJECT[duplicate-tensor]: two tensors map to layer {layer} \
                 {module:?} lora_{ab:?}"
            );
        }
        *slot = Some(name.to_string());
    }
    if found.is_empty() {
        bail!("REJECT[empty-adapter]: no lora_A/lora_B tensors in adapter");
    }

    // 2) pair completeness + shape audit. PEFT shapes: A=[r, in], B=[out, r].
    for ((layer, module), pair) in &found {
        let [Some(a_key), Some(b_key)] = pair else {
            bail!(
                "REJECT[unpaired-tensor]: layer {layer} {module:?} has only \
                 one of lora_A/lora_B"
            );
        };
        let (out_dim, in_dim) = module.dims(cfg);
        let a = adapter_store.get(a_key)?; // hard-fail get
        let b = adapter_store.get(b_key)?;
        if a.shape != vec![peft.r, in_dim] {
            bail!(
                "REJECT[shape-mismatch]: '{a_key}' is {:?}, expected [{}, {}] \
                 (r, in_dim)",
                a.shape,
                peft.r,
                in_dim
            );
        }
        if b.shape != vec![out_dim, peft.r] {
            bail!(
                "REJECT[shape-mismatch]: '{b_key}' is {:?}, expected [{}, {}] \
                 (out_dim, r)",
                b.shape,
                out_dim,
                peft.r
            );
        }
    }

    // 3) other audit direction: every target_modules entry matched ≥1 pair.
    for t in &peft.target_modules {
        let last = t.rsplit('.').next().unwrap_or(t);
        if !found.keys().any(|(_, m)| m.peft_name() == last) {
            bail!(
                "REJECT[unmatched-target]: target_modules entry '{t}' matched \
                 no adapter tensor on any full-attention layer"
            );
        }
    }
    Ok(found)
}

/// Pack one already-audited adapter into pool `slot` (byte sub-region at base
/// `slot * pool_slot_bytes`). The intra-slot walk (layer asc ×
/// [`LoraModule::ALL`] × A-then-B, A contiguous, B row-repacked stride r →
/// max_rank) is IDENTICAL for every slot — slot 0 is byte-for-byte the
/// pre-multi-adapter path. Returns this slot's GLOBAL-layer-indexed pairs and,
/// per (layer, module), the packed (a_ptr, b_ptr) as raw u64 ((0,0) where the
/// adapter omits the module) for the post-pass pointer-table build.
#[allow(clippy::type_complexity)]
fn pack_slot(
    slot: usize,
    name: &str,
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    found: &BTreeMap<(usize, LoraModule), [Option<String>; 2]>,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    pool: DevicePtr,
    max_lora_rank: usize,
) -> Result<(
    Vec<Option<LoraLayerWeights>>,
    BTreeMap<(usize, LoraModule), (u64, u64)>,
)> {
    let scale = peft.scaling();
    let slot_bytes = pool_slot_bytes(cfg, max_lora_rank);
    let mut layers: Vec<Option<LoraLayerWeights>> =
        (0..cfg.num_hidden_layers).map(|_| None).collect();
    let mut slot_ptrs: BTreeMap<(usize, LoraModule), (u64, u64)> = BTreeMap::new();
    let mut off = slot * slot_bytes; // slot base offset
    for layer_idx in full_attention_layers(cfg) {
        let mut lw = LoraLayerWeights {
            layer_idx,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            gate_proj: None,
            up_proj: None,
            down_proj: None,
        };
        let mut any = false;
        for module in LoraModule::ALL {
            let (out_dim, in_dim) = module.dims(cfg);
            let a_off = off;
            let b_off = off + max_lora_rank * in_dim * BF16_BYTES;
            off = b_off + out_dim * max_lora_rank * BF16_BYTES;
            let a_ptr = DevicePtr(pool.0 + a_off as u64);
            let b_ptr = DevicePtr(pool.0 + b_off as u64);

            let mut this = (0u64, 0u64); // NULL = base-only
            if let Some([Some(a_key), Some(b_key)]) = found.get(&(layer_idx, module)) {
                // A: contiguous [r, in] → head of the padded [max_rank, in] region.
                let a_t = adapter_store.get(a_key)?;
                let mut a_host = vec![0u8; peft.r * in_dim * BF16_BYTES];
                gpu.copy_d2h(a_t.ptr, &mut a_host)?;
                gpu.copy_h2d(&a_host, a_ptr)?;
                // B: [out, r] → row-stride pad to [out, max_rank].
                let b_t = adapter_store.get(b_key)?;
                let mut b_src = vec![0u8; out_dim * peft.r * BF16_BYTES];
                gpu.copy_d2h(b_t.ptr, &mut b_src)?;
                let mut b_host = vec![0u8; out_dim * max_lora_rank * BF16_BYTES];
                for row in 0..out_dim {
                    let d = row * max_lora_rank * BF16_BYTES;
                    let s = row * peft.r * BF16_BYTES;
                    b_host[d..d + peft.r * BF16_BYTES]
                        .copy_from_slice(&b_src[s..s + peft.r * BF16_BYTES]);
                }
                gpu.copy_h2d(&b_host, b_ptr)?;

                let pair = LoraPair {
                    a: DenseWeight { weight: a_ptr },
                    b: DenseWeight { weight: b_ptr },
                    rank: peft.r as u32,
                    k_in: in_dim as u32,
                    n_out: out_dim as u32,
                    scale,
                    // Kernel contraction dim: B's packed row stride (and A's
                    // padded row count) — see LoraPair docs in lora_delta.rs.
                    max_rank: max_lora_rank as u32,
                };
                tracing::info!(
                    "LoRA: slot {slot} '{name}' layer {layer_idx} {module:?} r={} \
                     scale={:.6} A=[{},{}] B=[{},{}] (padded to max_rank={})",
                    peft.r,
                    scale,
                    peft.r,
                    in_dim,
                    out_dim,
                    peft.r,
                    max_lora_rank
                );
                match module {
                    LoraModule::QProj => lw.q_proj = Some(pair),
                    LoraModule::KProj => lw.k_proj = Some(pair),
                    LoraModule::VProj => lw.v_proj = Some(pair),
                    LoraModule::OProj => lw.o_proj = Some(pair),
                    LoraModule::GateProj => lw.gate_proj = Some(pair),
                    LoraModule::UpProj => lw.up_proj = Some(pair),
                    LoraModule::DownProj => lw.down_proj = Some(pair),
                }
                this = (a_ptr.0, b_ptr.0);
                any = true;
            }
            slot_ptrs.insert((layer_idx, module), this);
        }
        if any {
            layers[layer_idx] = Some(lw);
        }
    }
    debug_assert_eq!(off, (slot + 1) * slot_bytes); // one slot filled exactly
    Ok((layers, slot_ptrs))
}

/// Model-agnostic MULTI-adapter PEFT load: audit every adapter, VRAM-preflight
/// the N-slot pool, pack each adapter into its slot (0..N-1), and build the
/// per-module `[max_loras]` pointer tables (index k filled per packed slot,
/// rest NULL). One resident adapter is byte-identical to the single-adapter
/// path (slot 0, `off` starts at 0).
///
/// Called (via the `ModelWeightLoader::load_lora_adapters` hook) from
/// `build_model` BEFORE `BufferArena::new` and the free-memory snapshot, so
/// the pool bytes land in `used_so_far` and the KV budget shrinks
/// automatically. Do NOT move the call later.
pub fn load_lora_adapters_multi(
    adapters: &[LoraAdapterInput<'_>],
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_loras: usize,
    max_lora_rank: usize,
) -> Result<LoraWeights> {
    check_family(cfg)?;
    if adapters.is_empty() {
        bail!("REJECT[no-adapters]: load_lora_adapters_multi called with an empty set");
    }
    if adapters.len() > max_loras {
        bail!(
            "REJECT[too-many-adapters]: {} --lora-adapter given but --max-loras={} \
             (pool has {} slots); raise --max-loras or stage the extras on an \
             $ATLAS_LORA_PEER for on-demand RDMA swap",
            adapters.len(),
            max_loras,
            max_loras
        );
    }

    // Audit every adapter up front (each gets its own classify/shape/target
    // audit + rank<=max_lora_rank check) before touching VRAM.
    let mut audited: Vec<BTreeMap<(usize, LoraModule), [Option<String>; 2]>> =
        Vec::with_capacity(adapters.len());
    for a in adapters {
        audited.push(audit_adapter(a.store, &a.peft, cfg, max_lora_rank)?);
    }

    // VRAM preflight, then one fixed-address pool alloc for ALL slots, zeroed
    // once (pad rows/cols and unpacked slots stay 0 = padded-K correctness).
    let pool_bytes = pool_slot_bytes(cfg, max_lora_rank) * max_loras;
    let free = gpu.free_memory()?;
    if pool_bytes * 2 > free {
        bail!(
            "OOM pre-flight (LoRA pool): {:.1} MiB pool ({} slots × padded A/B) \
             would leave < 1× headroom of {:.1} MiB free; every pool byte comes \
             directly out of the KV-cache budget on GB10 unified memory",
            pool_bytes as f64 / (1024.0 * 1024.0),
            max_loras,
            free as f64 / (1024.0 * 1024.0),
        );
    }
    let pool = gpu.alloc(pool_bytes)?;
    gpu.memset(pool, 0, pool_bytes)?;

    // Pack each adapter into its slot; accumulate per-(layer,module) [max_loras]
    // pointer arrays for the post-pass table build.
    let mut slots: Vec<AdapterSlot> = Vec::with_capacity(adapters.len());
    let mut a_tabs: BTreeMap<(usize, LoraModule), Vec<u64>> = BTreeMap::new();
    let mut b_tabs: BTreeMap<(usize, LoraModule), Vec<u64>> = BTreeMap::new();
    for (k, a) in adapters.iter().enumerate() {
        let (layers, slot_ptrs) = pack_slot(
            k,
            &a.name,
            a.store,
            &a.peft,
            &audited[k],
            cfg,
            gpu,
            pool,
            max_lora_rank,
        )?;
        for ((layer, module), (a_ptr, b_ptr)) in slot_ptrs {
            a_tabs
                .entry((layer, module))
                .or_insert_with(|| vec![0u64; max_loras])[k] = a_ptr;
            b_tabs
                .entry((layer, module))
                .or_insert_with(|| vec![0u64; max_loras])[k] = b_ptr;
        }
        slots.push(AdapterSlot {
            name: a.name.clone(),
            adapter_config: a.peft.clone(),
            layers,
            generation: 0, // first load: gen 0 keeps ids byte-identical to #24
        });
    }

    // Task #27: the pinned/cache boundary is the startup adapter count; the
    // remaining pool indices `[pinned, max_loras)` are the promotion HOT CACHE.
    // Pre-size `slots` to `max_loras` with EMPTY placeholders so a demand-promote
    // (`swap_lora_slot_from_peer`) can `slots.get_mut(cache_slot)` a never-filled
    // index (it would otherwise bail "slot not resident"). The placeholder's pool
    // byte-region is already allocated + zeroed above; its empty name is never
    // matched by the resolver nor advertised, and it contributes nothing to the
    // a/b/scale tables — so resident-only serving is byte-identical. `pinned == 0`
    // is impossible here (the caller rejects an empty adapter set).
    let pinned = slots.len();
    let num_layers = cfg.num_hidden_layers;
    while slots.len() < max_loras {
        slots.push(AdapterSlot {
            name: String::new(),
            adapter_config: PeftAdapterConfig {
                r: 1,
                lora_alpha: 0.0,
                target_modules: Vec::new(),
                use_rslora: false,
                layers_to_transform: None,
            },
            layers: vec![None; num_layers],
            generation: 0,
        });
    }

    // Post-pass: materialize the per-module [max_loras] u64 pointer tables (the
    // frozen M2 BGMV contract; currently dormant — no compute site reads them).
    // build_ptr_table pattern (nemotron_moe.rs:414): pack le bytes → alloc → h2d.
    let mk = |tab: &[u64]| -> Result<DevicePtr> {
        let bytes: Vec<u8> = tab.iter().flat_map(|p| p.to_le_bytes()).collect();
        let d = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, d)?;
        Ok(d)
    };
    let mut tables = BTreeMap::new();
    for (key, a_tab) in &a_tabs {
        let b_tab = &b_tabs[key];
        tables.insert(*key, (mk(a_tab)?, mk(b_tab)?));
    }

    // Parallel [max_loras] f32 scale table (per-slot scale, 0.0 for unpacked
    // slots) — the bgmv fold reads scale_table[seq_slot] in fp32. Same
    // load-time-fixed pattern as the a/b tables.
    let scale_vals = scale_table_values(adapters, max_loras);
    let scale_bytes: Vec<u8> = scale_vals.iter().flat_map(|s| s.to_le_bytes()).collect();
    let scale_table = gpu.alloc(scale_bytes.len())?;
    gpu.copy_h2d(&scale_bytes, scale_table)?;

    Ok(LoraWeights {
        name: slots[0].name.clone(),
        adapter_config: slots[0].adapter_config.clone(),
        max_rank: max_lora_rank,
        max_loras,
        pool,
        pool_bytes,
        slots,
        active: 0,
        tables,
        scale_table,
        // One counter per pool index, stable across swaps (sized to max_loras,
        // not slots.len(), so a later swap-into an empty slot has a counter).
        ref_counts: (0..max_loras).map(|_| AtomicUsize::new(0)).collect(),
        pinned,
        last_used: (0..max_loras).map(|_| AtomicU64::new(0)).collect(),
        lru_tick: AtomicU64::new(0),
    })
}

/// Runtime disk swap: audit + pack an already-loaded adapter `store` into an
/// EXISTING pool `slot` of `lw`, in place, and stamp that slot's
/// name/config/layers. Byte-identical to a startup pack of the same adapter into
/// that slot — same audit, A-contiguous copy, and B row-repack via `pack_slot`.
/// The slot sub-region is re-zeroed first (a reused slot still holds the prior
/// adapter's bytes, and pad rows/cols must stay 0 for padded-K correctness).
/// Returns the rebuilt per-layer pairs so the caller can re-install them if the
/// slot is currently active. Like the startup pack, the intermediate `store`'s
/// device copies leak (small, one-off per swap). Used for the pool-size-1
/// dynamic-load demo (load a different adapter into the single slot at runtime).
pub fn pack_store_into_slot(
    lw: &mut LoraWeights,
    slot: usize,
    name: &str,
    store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Vec<Option<LoraLayerWeights>>> {
    if slot >= lw.max_loras {
        bail!(
            "LoRA disk swap: slot {slot} >= max_loras {} (pool has {} slots)",
            lw.max_loras,
            lw.max_loras
        );
    }
    // Task #25 busy-slot refusal: bail BEFORE any destructive op (memset/pack)
    // so a refused swap leaves the slot's bytes + identity untouched. Replacing
    // an adapter while sequences are mid-decode on it would corrupt their KV and
    // replay a captured graph over swapped pool bytes.
    let busy = lw.slot_ref_count(slot);
    if busy > 0 {
        bail!(
            "LoRA disk swap REFUSED: slot {slot} has {busy} in-flight sequence(s) \
             (ref_count>0); cannot replace an adapter mid-decode"
        );
    }
    validate_peft_config(peft, lw.max_rank)?;
    let found = audit_adapter(store, peft, cfg, lw.max_rank)?;
    let slot_bytes = pool_slot_bytes(cfg, lw.max_rank);
    gpu.memset(
        DevicePtr(lw.pool.0 + (slot * slot_bytes) as u64),
        0,
        slot_bytes,
    )?;
    let (layers, _slot_ptrs) = pack_slot(
        slot,
        name,
        store,
        peft,
        &found,
        cfg,
        gpu,
        lw.pool,
        lw.max_rank,
    )?;
    lw.slots[slot].name = name.to_string();
    lw.slots[slot].adapter_config = peft.clone();
    lw.slots[slot].layers = layers.clone();
    // Task #26: refresh this slot's a/b pointer tables + scale table from the
    // new adapter's actual coverage (see refresh_slot_tables) so a re-staged slot
    // with different module coverage doesn't leave a stale/NULL bgmv route entry.
    lw.refresh_slot_tables(slot, &layers, peft.scaling(), gpu)?;
    // Task #25: contents changed → bump generation so this re-staged slot yields
    // a FRESH adapter_id and a later request misses the stale prior KV. (Covers
    // the disk swap and any future caller of this shared helper.)
    lw.slots[slot].generation = lw.slots[slot].generation.wrapping_add(1);
    Ok(layers)
}

/// Single-adapter convenience wrapper (packs slot 0 only) — byte-identical to
/// the pre-multi-adapter path. Kept for the unit tests and any single-adapter
/// caller. The `name` is stamped onto the sole slot.
pub fn load_lora_adapters_generic(
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_loras: usize,
    max_lora_rank: usize,
) -> Result<LoraWeights> {
    let inputs = [LoraAdapterInput {
        name: String::new(),
        store: adapter_store,
        peft: peft.clone(),
    }];
    load_lora_adapters_multi(&inputs, cfg, gpu, max_loras, max_lora_rank)
}
