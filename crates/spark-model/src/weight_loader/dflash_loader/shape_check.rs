// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: check a DFlash drafter checkpoint's tensor shapes against
//! its own `config.json` before any weight is loaded into the head.
//!
//! The head sizes every buffer from the config (`num_attention_heads`,
//! `head_dim`, `intermediate_size`, the tap count) and then runs GEMMs over the
//! checkpoint's tensors. A checkpoint that disagrees with its config does not
//! fail at load: it reads past an allocation (the NVFP4 Lightning bring-up's
//! CUDA 700 was exactly this) or produces silent garbage drafts. This module
//! turns that into a startup error that names the tensor.
//!
//! The expected shapes are the 69 tensors of `z-lab/Qwen3.5-4B-DFlash`
//! (3 + 11 per layer x 6 layers, read from its safetensors header on
//! 2026-11-12) generalized over the config: `fc`, `hidden_norm`, `norm`, and
//! per layer the two norms, the q/k/v/o projections, the q/k head norms and
//! the SwiGLU trio. Optional DFlash2 and DSpark heads are not checked here
//! (their loaders already require their own tensors).
//!
//! Packed NVFP4 tensors (U8 `[n, k/2]`) are skipped: their shapes are not the
//! logical ones, and `dense_bf16_or_nvfp4` validates them on dequantization.
//!
//! Strictness: see [`is_strict`]. Older drafters (no `num_target_layers`) warn
//! instead of failing, so the targets that already serve them are unchanged.

use anyhow::{Result, bail};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::DflashConfig;

/// What the check needs to know about one stored tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TensorShape {
    pub shape: Vec<usize>,
    /// NVFP4-packed U8: logical shape differs, skip.
    pub packed: bool,
}

/// Target hidden size implied by `fc`: `fc` is `[draft_hidden, taps * target_hidden]`.
pub(super) fn derive_target_hidden(cfg: &DflashConfig, fc: &[usize]) -> Result<usize> {
    let taps = cfg.tap_count();
    if taps == 0 {
        bail!("DFlash drafter declares no target_layer_ids (cannot size fc)");
    }
    if fc.len() != 2 || fc[0] != cfg.hidden_size {
        bail!(
            "DFlash drafter fc.weight has shape {fc:?}, expected [{}, taps * target_hidden]",
            cfg.hidden_size
        );
    }
    if !fc[1].is_multiple_of(taps) || fc[1] == 0 {
        bail!(
            "DFlash drafter fc.weight has {} input columns, which {taps} taps do not divide",
            fc[1]
        );
    }
    Ok(fc[1] / taps)
}

/// The (bare tensor name, shape) pairs a drafter with this config must ship,
/// for a target of hidden size `target_hidden`. Names carry no `model.` prefix.
pub(super) fn expected_shapes(
    cfg: &DflashConfig,
    target_hidden: usize,
) -> Vec<(String, Vec<usize>)> {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let i = cfg.intermediate_size;
    let mut v = vec![
        (
            "fc.weight".to_string(),
            vec![h, cfg.fc_in_features(target_hidden)],
        ),
        ("hidden_norm.weight".to_string(), vec![h]),
        ("norm.weight".to_string(), vec![h]),
    ];
    for l in 0..cfg.num_hidden_layers {
        let p = format!("layers.{l}");
        v.push((format!("{p}.input_layernorm.weight"), vec![h]));
        v.push((format!("{p}.post_attention_layernorm.weight"), vec![h]));
        v.push((format!("{p}.self_attn.q_proj.weight"), vec![q, h]));
        v.push((format!("{p}.self_attn.k_proj.weight"), vec![kv, h]));
        v.push((format!("{p}.self_attn.v_proj.weight"), vec![kv, h]));
        v.push((format!("{p}.self_attn.o_proj.weight"), vec![h, q]));
        v.push((format!("{p}.self_attn.q_norm.weight"), vec![cfg.head_dim]));
        v.push((format!("{p}.self_attn.k_norm.weight"), vec![cfg.head_dim]));
        v.push((format!("{p}.mlp.gate_proj.weight"), vec![i, h]));
        v.push((format!("{p}.mlp.up_proj.weight"), vec![i, h]));
        v.push((format!("{p}.mlp.down_proj.weight"), vec![h, i]));
    }
    v
}

/// BF16 bytes of the base tensors above (`expected_shapes`), i.e. the weight
/// footprint the preflight arithmetic should find on disk. 1,268,851,712 for
/// the 4B drafter (its safetensors file is that plus a 7,369-byte header).
pub fn bf16_weight_bytes(cfg: &DflashConfig, target_hidden: usize) -> u64 {
    expected_shapes(cfg, target_hidden)
        .iter()
        .map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>() * 2)
        .sum()
}

/// Pure core: `lookup` maps a FULL tensor name (prefix included) to its shape.
/// Returns the number of tensors compared. Errors list up to eight mismatches.
pub(super) fn check_with(
    cfg: &DflashConfig,
    prefix: &str,
    lookup: &dyn Fn(&str) -> Option<TensorShape>,
) -> Result<usize> {
    let fc_name = format!("{prefix}fc.weight");
    let Some(fc) = lookup(&fc_name) else {
        bail!("DFlash drafter store has no {fc_name}");
    };
    let target_hidden = if fc.packed {
        // Packed fc: logical in-features are twice the stored columns.
        let logical = [
            fc.shape.first().copied().unwrap_or(0),
            fc.shape.get(1).copied().unwrap_or(0) * 2,
        ];
        derive_target_hidden(cfg, &logical)?
    } else {
        derive_target_hidden(cfg, &fc.shape)?
    };
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for (name, want) in expected_shapes(cfg, target_hidden) {
        let full = format!("{prefix}{name}");
        let Some(got) = lookup(&full) else {
            bad.push(format!("{full}: missing"));
            continue;
        };
        if got.packed {
            continue;
        }
        checked += 1;
        if got.shape != want {
            bad.push(format!("{full}: shape {:?}, expected {want:?}", got.shape));
        }
    }
    if !bad.is_empty() {
        let shown = bad.iter().take(8).cloned().collect::<Vec<_>>().join("; ");
        bail!(
            "DFlash drafter checkpoint disagrees with its config.json ({} problem(s), hidden={} \
             heads={}/{} head_dim={} intermediate={} layers={} taps={}): {shown}",
            bad.len(),
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
            cfg.tap_count(),
        );
    }
    Ok(checked)
}

/// A mismatch is an ERROR for drafters that state `num_target_layers` (the
/// z-lab format of the Qwen3.5 generation, which only the Ada 4B wires) or
/// when `AVAROK_DFLASH_SHAPE_CHECK=1`; for every older drafter, which the GB10
/// and Hopper targets already serve, it is a WARNING with the same message, so
/// no working serve on those targets starts failing. `AVAROK_DFLASH_SHAPE_CHECK=0`
/// downgrades any mismatch to a warning.
fn is_strict(cfg: &DflashConfig) -> bool {
    match std::env::var("AVAROK_DFLASH_SHAPE_CHECK").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => cfg.num_target_layers.is_some(),
    }
}

/// Store adaptor used by `load_dflash_weights`.
pub(super) fn check_store(store: &WeightStore, cfg: &DflashConfig, prefix: &str) -> Result<()> {
    let lookup = |name: &str| {
        store.get(name).ok().map(|t| TensorShape {
            shape: t.shape.clone(),
            packed: t.dtype == WeightDtype::UInt8,
        })
    };
    match check_with(cfg, prefix, &lookup) {
        Ok(n) => {
            tracing::info!(
                "DFlash drafter shape check: {n} tensors match config (taps={}, sliding layers \
                 {}/{} window {:?})",
                cfg.tap_count(),
                cfg.sliding_layer_count(),
                cfg.num_hidden_layers,
                cfg.sliding_window,
            );
            Ok(())
        }
        Err(e) if !is_strict(cfg) => {
            tracing::warn!("DFlash drafter shape check (warning only): {e:#}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse_dflash_config;
    use super::*;
    use std::collections::HashMap;

    /// Verbatim `config.json` of `z-lab/Qwen3.5-4B-DFlash` (Apache-2.0).
    const CFG_4B: &str = include_str!("fixtures/qwen3_5_4b_dflash_config.json");

    fn store_for(cfg: &DflashConfig, target_hidden: usize) -> HashMap<String, TensorShape> {
        expected_shapes(cfg, target_hidden)
            .into_iter()
            .map(|(n, s)| {
                (
                    n,
                    TensorShape {
                        shape: s,
                        packed: false,
                    },
                )
            })
            .collect()
    }

    fn run(cfg: &DflashConfig, m: &HashMap<String, TensorShape>) -> Result<usize> {
        check_with(cfg, "", &|n| m.get(n).cloned())
    }

    #[test]
    fn the_real_4b_inventory_is_69_tensors_and_1_27_gb() {
        let cfg = parse_dflash_config(CFG_4B).unwrap();
        let shapes = expected_shapes(&cfg, 2560);
        assert_eq!(shapes.len(), 3 + 11 * 6);
        // Spot checks against the safetensors header read on 2026-11-12.
        let get = |n: &str| shapes.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(get("fc.weight"), vec![2560, 20480]);
        assert_eq!(get("layers.0.self_attn.q_proj.weight"), vec![4096, 2560]);
        assert_eq!(get("layers.5.self_attn.k_proj.weight"), vec![1024, 2560]);
        assert_eq!(get("layers.3.self_attn.o_proj.weight"), vec![2560, 4096]);
        assert_eq!(get("layers.2.mlp.gate_proj.weight"), vec![9216, 2560]);
        assert_eq!(get("layers.2.mlp.down_proj.weight"), vec![2560, 9216]);
        assert_eq!(get("layers.1.self_attn.q_norm.weight"), vec![128]);
        // model.safetensors is 1,268,859,081 bytes; minus the 7,369-byte header.
        assert_eq!(bf16_weight_bytes(&cfg, 2560), 1_268_851_712);
    }

    #[test]
    fn a_matching_checkpoint_passes_and_counts_every_tensor() {
        let cfg = parse_dflash_config(CFG_4B).unwrap();
        let m = store_for(&cfg, 2560);
        assert_eq!(run(&cfg, &m).unwrap(), 69);
    }

    #[test]
    fn the_target_hidden_is_derived_from_fc() {
        let cfg = parse_dflash_config(CFG_4B).unwrap();
        assert_eq!(derive_target_hidden(&cfg, &[2560, 20480]).unwrap(), 2560);
        // A Qwen3.6-style fc (5 taps) does not divide into this 8-tap config.
        assert!(derive_target_hidden(&cfg, &[2560, 5 * 5120 + 1]).is_err());
        assert!(derive_target_hidden(&cfg, &[2048, 20480]).is_err());
    }

    #[test]
    fn a_wrong_projection_shape_names_the_tensor() {
        let cfg = parse_dflash_config(CFG_4B).unwrap();
        let mut m = store_for(&cfg, 2560);
        m.get_mut("layers.4.mlp.up_proj.weight").unwrap().shape = vec![6144, 2560];
        let err = format!("{:#}", run(&cfg, &m).unwrap_err());
        assert!(err.contains("layers.4.mlp.up_proj.weight"), "{err}");
        assert!(err.contains("1 problem"), "{err}");
    }

    #[test]
    fn a_missing_tensor_is_reported_not_skipped() {
        let cfg = parse_dflash_config(CFG_4B).unwrap();
        let mut m = store_for(&cfg, 2560);
        m.remove("layers.5.self_attn.v_proj.weight");
        let err = format!("{:#}", run(&cfg, &m).unwrap_err());
        assert!(
            err.contains("layers.5.self_attn.v_proj.weight: missing"),
            "{err}"
        );
    }

    #[test]
    fn packed_nvfp4_tensors_are_skipped_not_misread() {
        let cfg = parse_dflash_config(CFG_4B).unwrap();
        let mut m = store_for(&cfg, 2560);
        // Stored as U8 [n, k/2]: the logical check must not fire on it.
        let q = m.get_mut("layers.0.self_attn.q_proj.weight").unwrap();
        *q = TensorShape {
            shape: vec![4096, 1280],
            packed: true,
        };
        assert_eq!(run(&cfg, &m).unwrap(), 68);
    }

    #[test]
    fn the_model_prefixed_layout_is_checked_too() {
        let cfg = parse_dflash_config(CFG_4B).unwrap();
        let m: HashMap<String, TensorShape> = store_for(&cfg, 2560)
            .into_iter()
            .map(|(k, v)| (format!("model.{k}"), v))
            .collect();
        assert_eq!(
            check_with(&cfg, "model.", &|n| m.get(n).cloned()).unwrap(),
            69
        );
    }

    /// Only the z-lab Qwen3.5 format (it states `num_target_layers`) fails the
    /// serve on a mismatch; the older drafters the GB10 and Hopper targets
    /// already serve only warn. Skipped when the override is set externally.
    #[test]
    fn only_the_new_format_is_strict_by_default() {
        if std::env::var_os("AVAROK_DFLASH_SHAPE_CHECK").is_some() {
            return;
        }
        let new = parse_dflash_config(CFG_4B).unwrap();
        assert!(is_strict(&new));
        let old = parse_dflash_config(
            r#"{"hidden_size":2048,"num_hidden_layers":8,"intermediate_size":6144,
                "num_attention_heads":32,"num_key_value_heads":4,"head_dim":128,
                "vocab_size":248320}"#,
        )
        .unwrap();
        assert!(!is_strict(&old));
    }

    #[test]
    fn the_qwen3_6_35b_drafter_shape_still_passes() {
        // The shipped Qwen3.6-35B-A3B drafter (loader_tests SHIPPED_CONFIG):
        // 8 layers, hidden 2048, 32/4 heads, head_dim 128, ffn 6144, 5 taps.
        let json = r#"{"hidden_size":2048,"num_hidden_layers":8,"intermediate_size":6144,
            "num_attention_heads":32,"num_key_value_heads":4,"head_dim":128,"vocab_size":248320,
            "dflash_config":{"mask_token_id":248070,"target_layer_ids":[1,10,19,28,37]}}"#;
        let cfg = parse_dflash_config(json).unwrap();
        let m = store_for(&cfg, 2048);
        assert_eq!(run(&cfg, &m).unwrap(), 3 + 11 * 8);
        assert_eq!(m["fc.weight"].shape, vec![2048, 10240]);
    }
}
