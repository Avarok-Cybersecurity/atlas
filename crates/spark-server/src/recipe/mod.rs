// SPDX-License-Identifier: AGPL-3.0-only

//! Serving recipes from `Avarok-Cybersecurity/atlas-recipes`.
//!
//! A recipe is a validated `spark serve` configuration for one checkpoint: the
//! model id, the flags, and the measured rationale for them. This module reads
//! one and turns it into argv; `schema` owns the key→flag mapping.

pub mod fetch;
mod fetch_github;
pub mod schema;
pub mod yaml;

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;

use yaml::Yaml;

/// One recipe, flattened to what the UI and the launcher need.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    /// `family/stem`, from the path — the stable id the UI selects by.
    pub id: String,
    pub version: String,
    pub model: String,
    /// `atlas` or `vllm`. Absent on v1 recipes, which predate the key.
    pub runtime: Option<String>,
    pub container: String,
    /// Ranks this recipe requires. **Not a `defaults:` key** — the three EP=2
    /// recipes carry `ep_size: 2` in `defaults` and `min_nodes: 2` up here, so
    /// a launcher reading only `defaults` builds a config the validator
    /// rejects with "--ep-size 2 exceeds --world-size 1".
    pub min_nodes: usize,
    pub description: String,
    pub maintainer: String,
    pub category: String,
    pub model_params: String,
    pub quantization: String,
    pub kv_dtype: String,
    /// When this recipe was last published or updated, from
    /// `metadata.updated`. Empty when the recipe does not carry one — which
    /// today is all of them, since the key is new. Deliberately a plain string
    /// rather than a parsed date: it is displayed, never compared, and a
    /// recipe with a malformed date should still load and serve.
    pub updated: String,
    /// The `defaults:` block verbatim.
    pub defaults: BTreeMap<String, String>,
}

impl Recipe {
    /// Read one recipe. `id` is `family/stem`, derived from its path.
    pub fn parse(id: impl Into<String>, text: &str) -> Result<Self> {
        let id = id.into();
        let doc = yaml::parse(text).with_context(|| format!("reading recipe {id}"))?;
        let map = doc.as_map().expect("parse guarantees a mapping");

        let scalar = |key: &str| -> Option<String> {
            map.get(key).and_then(Yaml::as_str).map(str::to_string)
        };
        let required = |key: &str| -> Result<String> {
            scalar(key).with_context(|| format!("{id}: missing required key {key:?}"))
        };

        let meta = map.get("metadata").and_then(Yaml::as_map);
        let meta_str = |key: &str| -> String {
            meta.and_then(|m| m.get(key))
                .and_then(Yaml::as_str)
                .unwrap_or_default()
                .to_string()
        };

        let defaults_block = map
            .get("defaults")
            .and_then(Yaml::as_map)
            .with_context(|| format!("{id}: `defaults:` must be a mapping"))?;
        let mut defaults = BTreeMap::new();
        for (key, value) in defaults_block {
            let Some(text) = value.as_str() else {
                bail!("{id}: defaults.{key} is not a scalar");
            };
            defaults.insert(key.clone(), text.to_string());
        }

        // `min_nodes` is absent on single-node recipes; 1 is the format's
        // meaning for "not stated", not an invented default.
        let min_nodes = match scalar("min_nodes") {
            Some(n) => n
                .parse()
                .with_context(|| format!("{id}: min_nodes {n:?} is not a number"))?,
            None => 1,
        };

        Ok(Self {
            version: required("recipe_version")?,
            model: required("model")?,
            runtime: scalar("runtime"),
            container: required("container")?,
            min_nodes,
            // v1 carries a top-level `description`; v2 puts it in metadata.
            description: match meta_str("description") {
                d if d.is_empty() => scalar("description").unwrap_or_default(),
                d => d,
            },
            maintainer: meta_str("maintainer"),
            category: meta_str("category"),
            model_params: meta_str("model_params"),
            quantization: meta_str("quantization"),
            kv_dtype: meta_str("kv_dtype"),
            updated: meta_str("updated"),
            defaults,
            id,
        })
    }

    /// Whether this recipe drives Atlas. A `vllm` recipe is listed but cannot
    /// be launched from here.
    pub fn is_atlas(&self) -> bool {
        self.runtime.as_deref() == Some("atlas")
    }

    /// The full `spark serve` argv, with `overrides` replacing recipe values.
    ///
    /// Overrides are merged into `defaults` before rendering rather than
    /// appended after, so an edited value cannot end up specified twice.
    pub fn argv(&self, overrides: &BTreeMap<String, String>) -> Result<Vec<String>> {
        if !self.is_atlas() {
            bail!(
                "{} is a {} recipe — only `runtime: atlas` recipes can be served from here",
                self.id,
                self.runtime.as_deref().unwrap_or("non-atlas")
            );
        }
        let mut merged = self.defaults.clone();
        for (key, value) in overrides {
            // A key the recipe does not list is an ADDITION, not an error. The
            // case that forces this: exercising fp8 KV needs BOTH
            // `kv_cache_dtype` (which every recipe pins) and
            // `fp8_kv_calibration_tokens` (which none of them mention) — and a
            // setting is absent from `defaults:` precisely because the recipe
            // does not use it, which is exactly when you need to add it.
            //
            // Refusing here was never the typo shield it looked like: `argv` is
            // rendered and handed straight back through clap by `serve_args`,
            // and clap rejects an unknown flag by name with suggestions. So the
            // shield is still up and is the SSOT one — this check was a second,
            // staler copy of it that also blocked the legitimate case.
            //
            // What clap CANNOT catch is a key `flag_for` drops (`NOT_FLAGS`):
            // that renders to nothing, parses fine, and silently serves the
            // unmodified config. So that one is refused here, where it is
            // visible.
            if !merged.contains_key(key) && schema::flag_for(key).is_none() {
                let known: Vec<&str> = merged.keys().map(String::as_str).collect();
                bail!(
                    "{}: {key:?} is not a serve flag, so setting it would change nothing. \
                     This recipe's settings: {}",
                    self.id,
                    known.join(", ")
                );
            }
            merged.insert(key.clone(), value.clone());
        }

        let mut argv = vec!["spark".to_string(), "serve".to_string(), self.model.clone()];
        for (key, value) in &merged {
            if let Some(mut rendered) = schema::argv_for(key, value) {
                argv.append(&mut rendered);
            }
        }
        // The world size lives outside `defaults:`; without it the EP recipes
        // build a config the validator rejects.
        if self.min_nodes > 1 {
            argv.push("--world-size".into());
            argv.push(self.min_nodes.to_string());
        }
        Ok(argv)
    }

    /// Render, parse and validate — the round trip clap stays the SSOT of.
    pub fn serve_args(
        &self,
        overrides: &BTreeMap<String, String>,
    ) -> Result<crate::cli::ServeArgs> {
        use clap::Parser as _;
        let argv = self.argv(overrides)?;
        let cli = crate::cli::Cli::try_parse_from(&argv)
            .with_context(|| format!("{}: recipe produced an invalid command line", self.id))?;
        let crate::cli::Command::Serve(args) = cli.command else {
            bail!("{}: recipe did not produce a serve command", self.id);
        };
        crate::cli::validate_serve_args(&args).map_err(|e| anyhow::anyhow!("{}: {e}", self.id))?;
        Ok(args)
    }
}

#[cfg(test)]
#[path = "recipe_tests.rs"]
mod tests;
