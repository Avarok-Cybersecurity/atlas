// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 (MoE expert + router LoRA) audit + pack, split out of `loading.rs`
//! to keep that file under the 500-LoC cap.
//!
//! Routed-expert and router deltas do NOT share the equal-size attention pool
//! (`pool_slot_bytes` + the S-LoRA BGMV route tables) — they land in a SEPARATE
//! expert pool sized from the AUDITED key set (real adapters adapt a subset of
//! the up-to-512 experts, so sizing from `num_experts × num_layers` maxima would
//! be catastrophic — 2.34 GiB/slot @ r16 on Holo-35B if fully dense). The
//! per-expert `LoraPair`s point into this pool and are applied by the
//! correctness-first `apply_lora_delta` side-path (`expert_apply.rs`), leaving
//! the base NVFP4/FP8 grouped GEMM byte-identical.
//!
//! Phase-1 is single-active-adapter (installed-pair path, no per-request BGMV
//! routing over experts) — the 2-D `(slot, expert)` route tables + grouped BGMV
//! kernel are phase 2.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use atlas_core::config::{ModelConfig, PeftAdapterConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::*;
use crate::layers::ops::lora_delta::LoraPair;
use crate::weight_map::DenseWeight;

/// Router audit map: global layer → `[a_key, b_key]`.
pub(crate) type RouterMap = BTreeMap<usize, [Option<String>; 2]>;
/// Expert audit map: `(global layer, expert, proj)` → `[a_key, b_key]`.
pub(crate) type ExpertMap = BTreeMap<(usize, u16, ExpertProj), [Option<String>; 2]>;

/// True when the adapter carries any routed-expert or router delta.
pub(crate) fn present(router: &RouterMap, experts: &ExpertMap) -> bool {
    !router.is_empty() || !experts.is_empty()
}

/// Validate the collected router/expert maps: the `ATLAS_LORA_EXPERTS` master
/// gate, the expert-rank cap, pair completeness, and A=[r,in]/B=[out,r] shapes.
/// Every failure is a NAMED reject (never a silent skip).
pub(crate) fn validate(
    cfg: &ModelConfig,
    peft: &PeftAdapterConfig,
    router: &RouterMap,
    experts: &ExpertMap,
) -> Result<()> {
    if !present(router, experts) {
        return Ok(());
    }
    if !lora_experts_env() {
        bail!(
            "REJECT[expert-lora-disabled]: adapter targets {} router + {} expert \
             projection(s), but MoE expert/router LoRA is off. Set ATLAS_LORA_EXPERTS=1 \
             to opt into the correctness-first (single-active, host-synced) expert path.",
            router.len(),
            experts.len()
        );
    }
    let cap = max_lora_expert_rank();
    if peft.r > cap {
        bail!(
            "REJECT[expert-rank-exceeds-cap]: r={} > ATLAS_LORA_EXPERT_RANK={} \
             (the expert pool grows ~num_experts×num_layers faster than attention; \
             raise the cap only with the VRAM headroom for it)",
            peft.r,
            cap
        );
    }
    for (layer, pair) in router {
        let (out, inp) = router_dims(cfg);
        audit_pair_shape(peft, pair, "router", *layer, None, out, inp)?;
    }
    for ((layer, n, proj), pair) in experts {
        let (out, inp) = proj.dims(cfg, *layer);
        audit_pair_shape(peft, pair, proj.peft_name(), *layer, Some(*n), out, inp)?;
    }
    Ok(())
}

fn audit_pair_shape(
    peft: &PeftAdapterConfig,
    _pair: &[Option<String>; 2],
    _module: &str,
    _layer: usize,
    _expert: Option<u16>,
    _out: usize,
    _inp: usize,
) -> Result<()> {
    // Shape checking against the WeightStore is done in `validate_shapes` (needs
    // the store). This split keeps `validate` store-free for the unit tests; the
    // caller runs `validate_shapes` with the store.
    let _ = peft;
    Ok(())
}

/// Store-backed shape audit: pair completeness + A=[r,in] / B=[out,r].
pub(crate) fn validate_shapes(
    store: &WeightStore,
    cfg: &ModelConfig,
    peft: &PeftAdapterConfig,
    router: &RouterMap,
    experts: &ExpertMap,
) -> Result<()> {
    for (layer, pair) in router {
        let (out, inp) = router_dims(cfg);
        check_shape(
            store,
            peft,
            pair,
            &format!("router(layer {layer})"),
            out,
            inp,
        )?;
    }
    for ((layer, n, proj), pair) in experts {
        let (out, inp) = proj.dims(cfg, *layer);
        check_shape(
            store,
            peft,
            pair,
            &format!("expert {n} {:?}(layer {layer})", proj),
            out,
            inp,
        )?;
    }
    Ok(())
}

fn check_shape(
    store: &WeightStore,
    peft: &PeftAdapterConfig,
    pair: &[Option<String>; 2],
    what: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    let [Some(a_key), Some(b_key)] = pair else {
        bail!("REJECT[unpaired-tensor]: {what} has only one of lora_A/lora_B");
    };
    let a = store.get(a_key)?;
    let b = store.get(b_key)?;
    if a.shape != vec![peft.r, in_dim] {
        bail!(
            "REJECT[shape-mismatch]: '{a_key}' is {:?}, expected [{}, {}] (r, in_dim)",
            a.shape,
            peft.r,
            in_dim
        );
    }
    if b.shape != vec![out_dim, peft.r] {
        bail!(
            "REJECT[shape-mismatch]: '{b_key}' is {:?}, expected [{}, {}] (out_dim, r)",
            b.shape,
            out_dim,
            peft.r
        );
    }
    Ok(())
}

/// Derived pool stride for the expert/router pack: `max_rank` rounded UP to a
/// multiple of 8 BF16 elements (16 bytes = one uint4, the vectorized load
/// width of `moe_lora_gather_bgmv` / `moe_lora_grouped_down`). The packed B
/// row stride and padded-A row count are THIS value, never the raw rank — at
/// e.g. r=12 a raw stride is 24 bytes and every second B row starts on a
/// non-16B-aligned address, which is undefined for the kernels' uint4 loads.
///
/// SSOT: the LOGICAL rank stays `peft.r` (`LoraPair.rank`); this is the one
/// derivation point for the PADDED stride (`LoraPair.max_rank`), shared by the
/// sizing estimator (`expert_router_bytes`) and the pack loop below so the two
/// agree byte-for-byte. Pad rows/cols stay zero (pool pre-zeroed + zero-filled
/// repack), so the padded contraction is bit-identical to the true-rank one.
pub(crate) fn packed_stride(max_rank: usize) -> usize {
    max_rank.div_ceil(8) * 8
}

/// Sizing key-lists for [`expert_router_bytes`] over one adapter's audit.
pub(crate) fn key_lists(
    router: &RouterMap,
    experts: &ExpertMap,
) -> (Vec<(usize, ExpertProj)>, Vec<usize>) {
    let ek = experts.keys().map(|(l, _, p)| (*l, *p)).collect();
    let rl = router.keys().copied().collect();
    (ek, rl)
}

/// Copy one padded A/B pair from the store into `(a_ptr, b_ptr)` and build the
/// [`LoraPair`]. Mirrors the attention `pack_slot` copy (A contiguous [r,in];
/// B row-repacked stride r → `stride`; pad rows/cols stay zero for padded-K
/// correctness — the caller must pre-zero the pool). `stride` is the DERIVED
/// [`packed_stride`] (uint4-aligned), never the raw rank — the logical rank
/// travels separately as `LoraPair.rank = peft.r`.
#[allow(clippy::too_many_arguments)]
fn pack_pair(
    store: &WeightStore,
    a_key: &str,
    b_key: &str,
    peft: &PeftAdapterConfig,
    out_dim: usize,
    in_dim: usize,
    stride: usize,
    gpu: &dyn GpuBackend,
    a_ptr: DevicePtr,
    b_ptr: DevicePtr,
) -> Result<LoraPair> {
    const BF16_BYTES: usize = 2;
    let a_t = store.get(a_key)?;
    let mut a_host = vec![0u8; peft.r * in_dim * BF16_BYTES];
    gpu.copy_d2h(a_t.ptr, &mut a_host)?;
    gpu.copy_h2d(&a_host, a_ptr)?;

    let b_t = store.get(b_key)?;
    let mut b_src = vec![0u8; out_dim * peft.r * BF16_BYTES];
    gpu.copy_d2h(b_t.ptr, &mut b_src)?;
    let mut b_host = vec![0u8; out_dim * stride * BF16_BYTES];
    for row in 0..out_dim {
        let d = row * stride * BF16_BYTES;
        let s = row * peft.r * BF16_BYTES;
        b_host[d..d + peft.r * BF16_BYTES].copy_from_slice(&b_src[s..s + peft.r * BF16_BYTES]);
    }
    gpu.copy_h2d(&b_host, b_ptr)?;

    Ok(LoraPair {
        a: DenseWeight { weight: a_ptr },
        b: DenseWeight { weight: b_ptr },
        rank: peft.r as u32,
        k_in: in_dim as u32,
        n_out: out_dim as u32,
        scale: peft.scaling(),
        max_rank: stride as u32,
    })
}

/// Pack this adapter's router + expert pairs into the shared expert pool at
/// running byte offset `*off`, filling `layers[l].router` / `layers[l].experts`
/// (creating the layer entry if attention left it `None`). Returns the number of
/// (router + expert) pairs packed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_into(
    layers: &mut [Option<LoraLayerWeights>],
    store: &WeightStore,
    peft: &PeftAdapterConfig,
    router: &RouterMap,
    experts: &ExpertMap,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    pool: DevicePtr,
    max_rank: usize,
    off: &mut usize,
) -> Result<usize> {
    const BF16_BYTES: usize = 2;
    // ONE derivation of the uint4-aligned pool stride for this whole pack —
    // the same [`packed_stride`] the sizing side (`expert_router_bytes`) uses,
    // so offsets can never outrun the pool the caller sized.
    let stride = packed_stride(max_rank);
    let mut packed = 0usize;
    let ensure = |layers: &mut [Option<LoraLayerWeights>], l: usize| {
        if layers[l].is_none() {
            layers[l] = Some(LoraLayerWeights::empty(l));
        }
    };

    for (layer, pair) in router {
        let [Some(a_key), Some(b_key)] = pair else {
            continue;
        };
        let (out_dim, in_dim) = router_dims(cfg);
        let a_ptr = DevicePtr(pool.0 + *off as u64);
        let b_ptr = DevicePtr(pool.0 + (*off + stride * in_dim * BF16_BYTES) as u64);
        *off += (stride * in_dim + out_dim * stride) * BF16_BYTES;
        let lp = pack_pair(
            store, a_key, b_key, peft, out_dim, in_dim, stride, gpu, a_ptr, b_ptr,
        )?;
        ensure(layers, *layer);
        layers[*layer].as_mut().unwrap().router = Some(lp);
        packed += 1;
    }

    for ((layer, n, proj), pair) in experts {
        let [Some(a_key), Some(b_key)] = pair else {
            continue;
        };
        let (out_dim, in_dim) = proj.dims(cfg, *layer);
        let a_ptr = DevicePtr(pool.0 + *off as u64);
        let b_ptr = DevicePtr(pool.0 + (*off + stride * in_dim * BF16_BYTES) as u64);
        *off += (stride * in_dim + out_dim * stride) * BF16_BYTES;
        let lp = pack_pair(
            store, a_key, b_key, peft, out_dim, in_dim, stride, gpu, a_ptr, b_ptr,
        )?;
        ensure(layers, *layer);
        let el = layers[*layer]
            .as_mut()
            .unwrap()
            .experts
            .get_or_insert_with(ExpertLoraLayer::default);
        el.pairs.insert((*n, *proj), lp);
        packed += 1;
    }
    Ok(packed)
}

#[cfg(test)]
#[path = "expert_pack_tests.rs"]
mod tests;
