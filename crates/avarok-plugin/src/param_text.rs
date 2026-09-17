// SPDX-License-Identifier: AGPL-3.0-only

//! Text ↔ [`ParamValues`]: the conversions a run record and a `--param` flag
//! need, which the editing path in `params.rs` did not.
//!
//! A stored run has to describe its whole configuration or it cannot be
//! reproduced, so values go to disk as text rather than as a typed enum. That
//! is not a shortcut — it is the shape that survives schema drift. An
//! externally-tagged `{"osl":{"Int":128}}` breaks the moment a `ParamValue`
//! variant is renamed, and it cannot be re-checked against a spec whose bounds
//! have since tightened. Text goes back through [`crate::params::ParamKind::parse`], the same
//! domain check the edit box uses, so an old value that is no longer legal is
//! *reported* rather than silently accepted.
//!
//! The round-trip is already load-bearing elsewhere: `ParamValues::validate_against`
//! renders every value with `to_edit_string` and re-parses it, and a registry
//! test asserts that holds for every benchmark in the suite.
//!
//! Kept out of `params.rs` because that file is at its size cap; a second
//! inherent `impl` block in a sibling module is the same type either way.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::params::{ParamSpec, ParamValue, ParamValues};

impl ParamValues {
    /// Every `(key, value)` pair, in deterministic key order.
    ///
    /// The map is a `BTreeMap`, so two records built from the same values
    /// serialize byte-identically — worth having when the files are diffed.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ParamValue)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The whole configuration as text, for persistence.
    pub fn to_strings(&self) -> BTreeMap<String, String> {
        self.iter()
            .map(|(k, v)| (k.to_string(), v.to_edit_string()))
            .collect()
    }

    /// Schema defaults with `overrides` applied on top.
    ///
    /// Every value is routed through [`ParamKind::parse`], so an out-of-domain
    /// override fails here rather than mid-run. An **unknown key is an error**
    /// naming the valid ones: a silently-ignored `--param` typo produces a run
    /// that measures something other than what was asked for, which is worse
    /// than a hard stop because the number still looks plausible.
    ///
    /// [`ParamKind::parse`]: crate::params::ParamKind::parse
    pub fn from_overrides<'a>(
        specs: &[ParamSpec],
        overrides: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self> {
        let mut values = Self::defaults(specs);
        for (key, raw) in overrides {
            let Some(spec) = specs.iter().find(|s| s.key == key) else {
                let known: Vec<&str> = specs.iter().map(|s| s.key).collect();
                bail!(
                    "unknown parameter {key:?} — this benchmark takes: {}",
                    known.join(", ")
                );
            };
            let value = spec
                .kind
                .parse(raw)
                .map_err(|e| anyhow::anyhow!("{}: {e}", spec.label))?;
            values.set(key, value);
        }
        Ok(values)
    }
}

#[cfg(test)]
#[path = "param_text_tests.rs"]
mod tests;
