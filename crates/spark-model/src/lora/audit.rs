// SPDX-License-Identifier: AGPL-3.0-only

//! Per-adapter classify + shape audit (attention/dense + Feature-1 router /
//! routed-expert), split out of `loading.rs` for the 500-LoC cap. Produces the
//! [`AuditedAdapter`] the pack loops consume; every unsupported/unpaired/mis-
//! shaped tensor is a NAMED hard reject (never a silent skip).

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use atlas_core::config::{ModelConfig, PeftAdapterConfig};
use spark_runtime::weights::WeightStore;

use super::*;

/// One adapter's classified tensor coverage: attention/dense pairs (the
/// equal-size pool), plus Feature-1 router + routed-expert pairs (a separate
/// pool). Attention lands byte-identically to the pre-Feature-1 path.
pub(crate) struct AuditedAdapter {
    pub attn: BTreeMap<(usize, LoraModule), [Option<String>; 2]>,
    pub router: expert_pack::RouterMap,
    pub experts: expert_pack::ExpertMap,
    /// Feature-2: classified token-overlay tensor coverage (embed/lm_head
    /// `trainable_tokens` / `modules_to_save`). Consumed by Stage-1
    /// `stage_overlay_raw`. `lora_embedding_*` is still a Tier-2 load reject.
    pub overlay: OverlayTensors,
}

/// Set the (layer, ab) cell of a dedup'd `[a_key, b_key]` audit entry, hard-
/// erroring on a duplicate tensor for the same slot.
fn set_ab(slot: &mut Option<String>, name: &str, what: &str) -> Result<()> {
    if slot.is_some() {
        bail!("REJECT[duplicate-tensor]: two tensors map to {what}");
    }
    *slot = Some(name.to_string());
    Ok(())
}

/// Classify + audit one adapter's tensors (unconsumed key = fatal; pair
/// completeness; A=[r,in]/B=[out,r] shapes; every `target_modules` entry
/// matched). Returns the per-target audit maps used to pack.
pub(crate) fn audit_adapter(
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    max_lora_rank: usize,
) -> Result<AuditedAdapter> {
    validate_peft_config(peft, max_lora_rank)?;

    // 1) classify EVERY adapter tensor — any unclassifiable/unsupported key
    //    is a hard error, which IS the "unconsumed adapter tensors fatal"
    //    audit direction.
    let mut found: BTreeMap<(usize, LoraModule), [Option<String>; 2]> = BTreeMap::new();
    let mut router: expert_pack::RouterMap = BTreeMap::new();
    let mut experts: expert_pack::ExpertMap = BTreeMap::new();
    let mut overlay = OverlayTensors::default();
    for name in adapter_store.names() {
        // Feature 2: token-overlay tensors (`…token_adapter.*`, bare
        // `modules_to_save` `.weight`, `lora_embedding_*`) are intercepted
        // ABOVE `classify_key` so its lora_A/lora_B suffix gate never
        // mis-rejects them. Collected here, applied by the token-overlay path.
        if let Some(t) = classify_overlay_key(name) {
            overlay.insert(t, name)?;
            continue;
        }
        let (layer, target, ab) = classify_key(name, cfg)?;
        match target {
            LoraTarget::Attn(module) => {
                let entry = found.entry((layer, module)).or_default();
                set_ab(
                    &mut entry[ab as usize],
                    name,
                    &format!("layer {layer} {module:?}"),
                )?;
            }
            LoraTarget::Router => {
                let entry = router.entry(layer).or_default();
                set_ab(
                    &mut entry[ab as usize],
                    name,
                    &format!("layer {layer} router"),
                )?;
            }
            LoraTarget::Expert { n, proj } => {
                let entry = experts.entry((layer, n, proj)).or_default();
                set_ab(
                    &mut entry[ab as usize],
                    name,
                    &format!("layer {layer} expert {n} {proj:?}"),
                )?;
            }
        }
    }
    // Feature-2 FLIP: overlay tensors are now LOADED (Stage-1 upload +
    // Stage-2 build), not blanket-rejected. Only the classic low-rank embedding
    // LoRA (`lora_embedding_A/B`) remains a Tier-2 NAMED reject.
    reject_pending_overlay(&overlay)?;
    if found.is_empty() && !expert_pack::present(&router, &experts) && overlay.is_empty() {
        bail!("REJECT[empty-adapter]: no lora_A/lora_B or overlay tensors in adapter");
    }

    // 2) attention pair completeness + shape audit. PEFT: A=[r, in], B=[out, r].
    for ((layer, module), pair) in &found {
        let [Some(a_key), Some(b_key)] = pair else {
            bail!(
                "REJECT[unpaired-tensor]: layer {layer} {module:?} has only one of lora_A/lora_B"
            );
        };
        let (out_dim, in_dim) = module.dims(cfg);
        let a = adapter_store.get(a_key)?; // hard-fail get
        let b = adapter_store.get(b_key)?;
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
    }

    // Feature-1 router/expert audit: master gate + rank cap (flag-gated so an
    // expert adapter is a NAMED reject unless ATLAS_LORA_EXPERTS=1) + shapes.
    expert_pack::validate(cfg, peft, &router, &experts)?;
    if expert_pack::present(&router, &experts) {
        expert_pack::validate_shapes(adapter_store, cfg, peft, &router, &experts)?;
    }

    // 3) other audit direction: every target_modules entry matched ≥1 pair.
    for t in &peft.target_modules {
        let last = t.rsplit('.').next().unwrap_or(t);
        let matched = found.keys().any(|(_, m)| m.peft_name() == last)
            || (last == "gate" && !router.is_empty())
            || experts.keys().any(|(_, _, p)| p.peft_name() == last);
        if !matched {
            bail!(
                "REJECT[unmatched-target]: target_modules entry '{t}' matched \
                 no adapter tensor on any full-attention layer"
            );
        }
    }
    Ok(AuditedAdapter {
        attn: found,
        router,
        experts,
        overlay,
    })
}
