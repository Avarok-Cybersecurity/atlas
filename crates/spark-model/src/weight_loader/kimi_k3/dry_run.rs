// SPDX-License-Identifier: AGPL-3.0-only

//! Safetensors `index.json` dry-run: cover every production **text** tensor
//! class without downloading shards. Lists unique shard names and refuses
//! missing required keys.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::classes::{ClassKind, TEXT_CLASSES, classify};

/// Result of walking a K3 `weight_map` (no tensors loaded).
#[derive(Debug, Clone)]
pub struct KimiK3DryRun {
    pub shards: Vec<String>,
    pub present_text_classes: Vec<String>,
    pub missing_text_classes: Vec<String>,
    /// Vision/mm classes seen while `language_model_only` was set.
    pub ignored_vision_classes: Vec<String>,
    pub unknown_keys: Vec<String>,
}

/// Parse a HuggingFace safetensors `index.json` and dry-run the weight map.
pub fn dry_run_index_json(index_json: &str, language_model_only: bool) -> Result<KimiK3DryRun> {
    let raw: Value = serde_json::from_str(index_json).context("invalid safetensors index.json")?;
    let map = raw
        .get("weight_map")
        .and_then(Value::as_object)
        .context("safetensors index.json missing weight_map")?;
    let weight_map: BTreeMap<String, String> = map
        .iter()
        .map(|(k, v)| {
            let shard = v
                .as_str()
                .with_context(|| format!("weight_map[{k}] is not a string"))?
                .to_string();
            Ok((k.clone(), shard))
        })
        .collect::<Result<_>>()?;
    dry_run_weight_map(&weight_map, language_model_only)
}

pub fn dry_run_weight_map(
    weight_map: &BTreeMap<String, String>,
    language_model_only: bool,
) -> Result<KimiK3DryRun> {
    let mut shards = BTreeSet::new();
    let mut present_text = BTreeSet::new();
    let mut ignored_vision = BTreeSet::new();
    let mut unknown_keys = Vec::new();

    for (name, shard) in weight_map {
        shards.insert(shard.clone());
        match classify(name) {
            Some((class, ClassKind::Text)) => {
                present_text.insert(class.to_string());
            }
            Some((class, ClassKind::Vision)) => {
                if language_model_only {
                    ignored_vision.insert(class.to_string());
                }
            }
            None => unknown_keys.push(name.clone()),
        }
    }

    let missing_text_classes: Vec<String> = TEXT_CLASSES
        .iter()
        .filter(|c| !present_text.contains(**c))
        .map(|c| (*c).to_string())
        .collect();

    if !missing_text_classes.is_empty() {
        bail!(
            "K3 weight map missing required text classes ({}): {}",
            missing_text_classes.len(),
            missing_text_classes.join(", ")
        );
    }
    if !unknown_keys.is_empty() {
        bail!(
            "K3 weight map has unrecognised keys ({}): {}",
            unknown_keys.len(),
            unknown_keys
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    Ok(KimiK3DryRun {
        shards: shards.into_iter().collect(),
        present_text_classes: present_text.into_iter().collect(),
        missing_text_classes,
        ignored_vision_classes: ignored_vision.into_iter().collect(),
        unknown_keys,
    })
}

/// Official moonshotai/Kimi-K3 is 96 shards. Twin maps are not.
pub fn require_shard_count(report: &KimiK3DryRun, n: usize) -> Result<()> {
    if report.shards.len() != n {
        bail!(
            "K3 weight map expected {n} shards, got {}",
            report.shards.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::classes::example_key;
    use super::*;
    use std::collections::BTreeMap;

    /// Synthetic 96-shard map: one key per text class, padded so every shard appears.
    fn synthetic_96_shard_map() -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        let n_shards = 96usize;
        for (i, class) in TEXT_CLASSES.iter().enumerate() {
            let shard = format!("model-{:05}-of-00096.safetensors", (i % n_shards) + 1);
            map.insert(example_key(class), shard);
        }
        let pad_class = TEXT_CLASSES[0];
        for i in TEXT_CLASSES.len()..n_shards {
            let shard = format!("model-{:05}-of-00096.safetensors", i + 1);
            let key = pad_class
                .replacen('*', "0", 1)
                .replacen('*', &i.to_string(), 1);
            map.insert(key, shard);
        }
        map
    }

    #[test]
    fn kimi_k3_weight_map_dry_run_lists_96_shards() {
        let map = synthetic_96_shard_map();
        let report = dry_run_weight_map(&map, true).expect("complete text map");
        assert_eq!(report.shards.len(), 96);
        assert_eq!(report.present_text_classes.len(), TEXT_CLASSES.len());
        assert!(report.missing_text_classes.is_empty());
        assert!(report.unknown_keys.is_empty());
        assert_eq!(report.shards[0], "model-00001-of-00096.safetensors");
        assert_eq!(report.shards[95], "model-00096-of-00096.safetensors");
    }

    #[test]
    fn kimi_k3_weight_map_dry_run_refuses_missing_text_class() {
        let mut map = synthetic_96_shard_map();
        let drop = example_key("language_model.model.output_attn_res_proj.weight");
        map.remove(&drop);
        let err = dry_run_weight_map(&map, true).unwrap_err().to_string();
        assert!(
            err.contains("missing required text classes"),
            "unexpected error: {err}"
        );
        assert!(err.contains("output_attn_res_proj"), "{err}");
    }

    #[test]
    fn kimi_k3_weight_map_dry_run_ignores_vision_when_language_model_only() {
        let mut map = synthetic_96_shard_map();
        map.insert(
            "vision_tower.patch_embed.proj.weight".into(),
            "model-00001-of-00096.safetensors".into(),
        );
        let report = dry_run_weight_map(&map, true).expect("vision ignored");
        assert!(
            report
                .ignored_vision_classes
                .iter()
                .any(|c| c == "vision_tower.patch_embed.proj.weight")
        );
    }

    #[test]
    fn kimi_k3_weight_map_dry_run_index_json() {
        let map = synthetic_96_shard_map();
        let index = serde_json::json!({ "weight_map": map });
        let report = dry_run_index_json(&index.to_string(), true).unwrap();
        assert_eq!(report.shards.len(), 96);
        require_shard_count(&report, 96).unwrap();
    }

    #[test]
    fn kimi_k3_weight_map_dry_run_refuses_95_shards() {
        let mut map = synthetic_96_shard_map();
        map.retain(|_, shard| !shard.starts_with("model-00096-of-"));
        let report = dry_run_weight_map(&map, true).expect("text classes still present");
        let err = require_shard_count(&report, 96).unwrap_err().to_string();
        assert!(err.contains("expected 96 shards"), "{err}");
        assert!(err.contains("got 95"), "{err}");
    }
}
