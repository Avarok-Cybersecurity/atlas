// SPDX-License-Identifier: AGPL-3.0-only

//! RDMA LoRA staging (spark-model half): turn a peer-staged adapter's manifest
//! into a set of pool-slot LANDING TARGETS (the only place `classify_key` +
//! the per-slot offset math live), then drive `spark_storage::RdmaLoraLoader`
//! to RDMA-load the adapter's A/B straight into a resident slot for fast
//! rotation. Landing is byte-identical to the disk pack (the loader does the
//! same F16/F32→BF16 convert + B row-repack).
//!
//! Gated behind `$ATLAS_LORA_PEER` at the call site; when unset the disk
//! rotation path is unchanged.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use atlas_core::config::{ModelConfig, PeftAdapterConfig};
use spark_runtime::gpu::DevicePtr;
use spark_storage::weight_peer::WeightManifest;
use spark_storage::{LoraAbKind, LoraLandTarget};

use super::{
    AdapterAb, LoraLayerWeights, LoraModule, LoraTarget, classify_key, module_slot_offsets,
    pool_slot_bytes, slot_base_offset,
};
use crate::layers::ops::lora_delta::LoraPair;
use crate::weight_map::DenseWeight;

/// Build the landing targets for one adapter's manifest into pool `slot`. Each
/// `lora_A/lora_B` tensor is classified to (layer, module, A|B) and mapped to
/// its byte sub-region `pool + slot*slot_bytes + a_off|b_off`. The adapter's
/// real rank r is read from the tensor shape (A=`[r,in]`, B=`[out,r]`). Rejections
/// from `classify_key` (GDN / wrong-layer / non-PEFT key) fire here
/// too — never a silent skip.
pub fn build_land_targets(
    manifest: &WeightManifest,
    cfg: &ModelConfig,
    pool: DevicePtr,
    slot: usize,
    max_rank: usize,
) -> Result<Vec<LoraLandTarget>> {
    let base = pool.0 + slot_base_offset(slot, cfg, max_rank) as u64;
    let mut targets = Vec::with_capacity(manifest.tensors.len());
    let mut pairs: BTreeMap<(usize, LoraModule), [Option<usize>; 2]> = BTreeMap::new();
    for rec in &manifest.tensors {
        let (layer, target, ab) = classify_key(&rec.name, cfg)?;
        // RDMA slot-swap stages ONLY the equal-size attention/dense pool. Router
        // and routed-expert LoRA (Feature-1) live in a separate expert pool with
        // its own offset math and are not RDMA-swappable in P1 — reject by name.
        let module = match target {
            LoraTarget::Attn(m) => m,
            LoraTarget::Router | LoraTarget::Expert { .. } => bail!(
                "lora rdma: '{}' is a router/expert delta (Feature-1); RDMA \
                 slot-swap stages the attention pool only",
                rec.name
            ),
        };
        let (a_off, b_off) = module_slot_offsets(cfg, max_rank, layer, module)
            .ok_or_else(|| anyhow!("lora rdma: layer {layer} not a full-attention slot layer"))?;
        let (out_dim, in_dim) = module.dims(cfg);
        // Audit the complete on-wire geometry before deriving r. The landing
        // transforms trust these dimensions when copying into a fixed-size
        // pool sub-region, so accepting an extra/missing/wrong dimension here
        // can otherwise become a panic or a mispacked adapter later.
        let rank = match ab {
            AdapterAb::A if rec.shape.len() == 2 && rec.shape[1] == in_dim as u64 => {
                rec.shape[0] as usize
            }
            AdapterAb::B if rec.shape.len() == 2 && rec.shape[0] == out_dim as u64 => {
                rec.shape[1] as usize
            }
            AdapterAb::A => bail!(
                "REJECT[shape-mismatch]: '{}' is {:?}, expected [r, {}]",
                rec.name,
                rec.shape,
                in_dim
            ),
            AdapterAb::B => bail!(
                "REJECT[shape-mismatch]: '{}' is {:?}, expected [{}, r]",
                rec.name,
                rec.shape,
                out_dim
            ),
        };
        if rank == 0 {
            bail!("REJECT[shape-mismatch]: '{}' has zero rank", rec.name);
        }
        if rank > max_rank {
            bail!(
                "lora rdma: adapter rank {rank} for {} exceeds pool max_rank {max_rank}",
                rec.name
            );
        }
        let (kind, off) = match ab {
            AdapterAb::A => (LoraAbKind::A, a_off),
            AdapterAb::B => (LoraAbKind::B, b_off),
        };
        let pair = pairs.entry((layer, module)).or_default();
        let cell = &mut pair[ab as usize];
        if cell.is_some() {
            bail!("REJECT[duplicate-tensor]: two tensors map to layer {layer} {module:?} {ab:?}");
        }
        *cell = Some(rank);
        targets.push(LoraLandTarget {
            tensor_name: rec.name.clone(),
            kind,
            dst: base + off as u64,
            out_dim,
            in_dim,
            rank,
            max_rank,
        });
    }
    if targets.is_empty() {
        bail!("lora rdma: adapter manifest has no lora_A/lora_B tensors");
    }
    for ((layer, module), pair) in pairs {
        let [Some(a_rank), Some(b_rank)] = pair else {
            bail!(
                "REJECT[unpaired-tensor]: layer {layer} {module:?} has only one of lora_A/lora_B"
            );
        };
        if a_rank != b_rank {
            bail!(
                "REJECT[rank-mismatch]: layer {layer} {module:?} has A rank {a_rank}, B rank {b_rank}"
            );
        }
    }
    Ok(targets)
}

/// Rebuild a slot's per-layer [`LoraLayerWeights`] after an in-place RDMA
/// reload — the A/B bytes changed AND the adapter's r/scale may differ, so the
/// `LoraPair`s (which bake rank + scale) must be rebuilt, not just re-pointed.
/// Pointers are deterministic (`pool + slot*slot_bytes + off`); this does NOT
/// touch the GPU. Modules present are those with a target of the matching kind.
pub fn rebuild_slot_layers(
    targets: &[LoraLandTarget],
    cfg: &ModelConfig,
    peft: &PeftAdapterConfig,
    pool: DevicePtr,
    slot: usize,
    max_rank: usize,
) -> Result<Vec<Option<LoraLayerWeights>>> {
    let scale = peft.scaling();
    let base = pool.0 + slot_base_offset(slot, cfg, max_rank) as u64;
    let mut layers: Vec<Option<LoraLayerWeights>> =
        (0..cfg.num_hidden_layers).map(|_| None).collect();
    // Group targets by (layer, module): we need both A and B present to build a
    // pair. Re-derive from classify (targets carry only geometry, not keys' layer).
    // Simpler: walk the pool layout and, for each (layer, module), find whether a
    // target lands there (by matching dst).
    // Same walk as `pool_slot_bytes` / `pack_slot` / `module_slot_offsets`:
    // every layer, applicable modules only. Walking full-attention layers x
    // ALL would ask for the offset of a module that layer cannot carry (e.g.
    // the GDN `out_proj` on an attention layer), which now correctly has none.
    for rec_layer in 0..cfg.num_hidden_layers {
        let mut lw = LoraLayerWeights::empty(rec_layer);
        let mut any = false;
        for module in LoraModule::ALL {
            if !module.applies_to_layer(cfg, rec_layer) {
                continue;
            }
            let (a_off, b_off) = module_slot_offsets(cfg, max_rank, rec_layer, module)
                .expect("applicable module has a slot offset");
            let a_dst = base + a_off as u64;
            let b_dst = base + b_off as u64;
            let a_t = targets
                .iter()
                .find(|t| t.kind == LoraAbKind::A && t.dst == a_dst);
            let b_t = targets
                .iter()
                .find(|t| t.kind == LoraAbKind::B && t.dst == b_dst);
            if let (Some(a), Some(b)) = (a_t, b_t) {
                let (out_dim, in_dim) = module.dims(cfg);
                if a.rank != b.rank || a.rank != peft.r {
                    bail!(
                        "REJECT[rank-mismatch]: layer {rec_layer} {module:?} has target ranks A={} B={}, config r={}",
                        a.rank,
                        b.rank,
                        peft.r
                    );
                }
                if a.max_rank != max_rank
                    || b.max_rank != max_rank
                    || a.out_dim != out_dim
                    || b.out_dim != out_dim
                    || a.in_dim != in_dim
                    || b.in_dim != in_dim
                {
                    bail!(
                        "REJECT[landing-geometry]: layer {rec_layer} {module:?} target geometry does not match the pool layout"
                    );
                }
                let pair = LoraPair {
                    a: DenseWeight {
                        weight: DevicePtr(a_dst),
                    },
                    b: DenseWeight {
                        weight: DevicePtr(b_dst),
                    },
                    rank: a.rank as u32,
                    k_in: in_dim as u32,
                    n_out: out_dim as u32,
                    scale,
                    max_rank: max_rank as u32,
                };
                match module {
                    LoraModule::QProj => lw.q_proj = Some(pair),
                    LoraModule::KProj => lw.k_proj = Some(pair),
                    LoraModule::VProj => lw.v_proj = Some(pair),
                    LoraModule::OProj => lw.o_proj = Some(pair),
                    LoraModule::GateProj => lw.gate_proj = Some(pair),
                    LoraModule::UpProj => lw.up_proj = Some(pair),
                    LoraModule::DownProj => lw.down_proj = Some(pair),
                    LoraModule::OutProj => lw.out_proj = Some(pair),
                }
                any = true;
            }
        }
        if any {
            layers[rec_layer] = Some(lw);
        }
    }
    Ok(layers)
}

/// The per-slot byte length (re-exported so the swap path can re-zero exactly
/// one slot's sub-region before an in-place reload).
pub fn slot_bytes(cfg: &ModelConfig, max_rank: usize) -> usize {
    pool_slot_bytes(cfg, max_rank)
}

/// Fetch a peer-staged adapter's manifest over the `weight_peer` control
/// channel (connect → request → read manifest, then drop the connection).
/// Needed to build landing targets before the loader's own verbs handshake.
#[cfg(feature = "cuda")]
pub fn fetch_adapter_manifest(peer_addr: &str, adapter_id: &str) -> Result<WeightManifest> {
    use std::net::TcpStream;

    use anyhow::Context;
    use spark_storage::weight_peer::{read_weight_manifest, write_model_request};

    let mut stream =
        TcpStream::connect(peer_addr).with_context(|| format!("connect lora peer {peer_addr}"))?;
    stream.set_nodelay(true).ok();
    write_model_request(&mut stream, adapter_id).context("send adapter request")?;
    let manifest = read_weight_manifest(&mut stream).context("read adapter manifest")?;
    // Drop the connection without a transport handshake; the loader reconnects
    // for the actual one-sided read.
    let _ = std::io::Write::write_all(&mut stream, &[]);
    Ok(manifest)
}

#[cfg(test)]
#[path = "rdma_stage_tests.rs"]
mod tests;
