// SPDX-License-Identifier: AGPL-3.0-only

//! Parameter schema — what a benchmark exposes for editing BEFORE it runs.
//!
//! A benchmark declares [`ParamSpec`]s; the TUI renders them, the user edits
//! them as text, and the edited [`ParamValues`] come back through
//! `Benchmark::configure` before the first `next()`.
//!
//! Two SSOT rules this file exists to enforce:
//!   * **Defaults live in the spec, nowhere else.** [`ParamValues::defaults`]
//!     derives the starting values from the schema, so a benchmark cannot
//!     disagree with what the pane shows.
//!   * **Text ↔ value parsing lives in [`ParamKind`].** The renderer never
//!     parses; it hands a string to [`ParamKind::parse`] and shows the error.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};

/// The editing affordance and the validation domain for one parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamKind {
    Bool,
    Int {
        min: i64,
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
    },
    /// Free text (URLs, model ids, paths).
    Text,
    /// One of a fixed set. The value is carried as [`ParamValue::Text`].
    Choice(&'static [&'static str]),
    /// Comma-separated integers — concurrency levels, prompt lengths.
    IntList {
        min: i64,
        max: i64,
    },
}

/// A concrete parameter value.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    IntList(Vec<i64>),
}

impl ParamValue {
    /// Render for the edit box. The inverse of [`ParamKind::parse`].
    pub fn to_edit_string(&self) -> String {
        match self {
            ParamValue::Bool(b) => b.to_string(),
            ParamValue::Int(i) => i.to_string(),
            // `{}` on f64 already drops a trailing `.0`-only fraction cleanly
            // enough for editing, and round-trips through `parse`.
            ParamValue::Float(v) => format!("{v}"),
            ParamValue::Text(s) => s.clone(),
            ParamValue::IntList(v) => v.iter().map(i64::to_string).collect::<Vec<_>>().join(", "),
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            ParamValue::Bool(_) => "bool",
            ParamValue::Int(_) => "int",
            ParamValue::Float(_) => "float",
            ParamValue::Text(_) => "text",
            ParamValue::IntList(_) => "int-list",
        }
    }
}

impl ParamKind {
    /// Parse an edited string into a value of this kind, enforcing the domain.
    /// The error text is shown verbatim under the field, so it names the bound
    /// that was violated rather than saying "invalid".
    pub fn parse(&self, raw: &str) -> Result<ParamValue> {
        let s = raw.trim();
        match self {
            ParamKind::Bool => match s.to_ascii_lowercase().as_str() {
                "true" | "yes" | "on" | "1" => Ok(ParamValue::Bool(true)),
                "false" | "no" | "off" | "0" => Ok(ParamValue::Bool(false)),
                _ => bail!("expected true or false, got {s:?}"),
            },
            ParamKind::Int { min, max } => {
                let v: i64 = s.parse().map_err(|_| anyhow!("not an integer: {s:?}"))?;
                if v < *min || v > *max {
                    bail!("must be between {min} and {max}, got {v}");
                }
                Ok(ParamValue::Int(v))
            }
            ParamKind::Float { min, max } => {
                let v: f64 = s.parse().map_err(|_| anyhow!("not a number: {s:?}"))?;
                if !v.is_finite() {
                    bail!("must be finite, got {s:?}");
                }
                if v < *min || v > *max {
                    bail!("must be between {min} and {max}, got {v}");
                }
                Ok(ParamValue::Float(v))
            }
            ParamKind::Text => {
                if s.is_empty() {
                    bail!("must not be empty");
                }
                Ok(ParamValue::Text(s.to_string()))
            }
            ParamKind::Choice(options) => options
                .iter()
                .find(|o| o.eq_ignore_ascii_case(s))
                .map(|o| ParamValue::Text((*o).to_string()))
                .ok_or_else(|| anyhow!("must be one of: {}", options.join(", "))),
            ParamKind::IntList { min, max } => {
                let mut out = Vec::new();
                for part in s.split(',') {
                    let p = part.trim();
                    if p.is_empty() {
                        continue;
                    }
                    let v: i64 = p.parse().map_err(|_| anyhow!("not an integer: {p:?}"))?;
                    if v < *min || v > *max {
                        bail!("every entry must be between {min} and {max}, got {v}");
                    }
                    out.push(v);
                }
                if out.is_empty() {
                    bail!("must list at least one integer");
                }
                Ok(ParamValue::IntList(out))
            }
        }
    }

    /// Short hint rendered beside the field, e.g. `1–4096` or `a, b, c`.
    pub fn domain_hint(&self) -> String {
        match self {
            ParamKind::Bool => "true / false".into(),
            ParamKind::Int { min, max } | ParamKind::IntList { min, max } => {
                format!("{min}–{max}")
            }
            ParamKind::Float { min, max } => format!("{min}–{max}"),
            ParamKind::Text => "text".into(),
            ParamKind::Choice(options) => options.join(" / "),
        }
    }
}

/// One editable parameter.
#[derive(Clone, Debug)]
pub struct ParamSpec {
    pub key: &'static str,
    pub label: &'static str,
    /// One line shown under the field. Say what the value DOES, not its type.
    pub help: &'static str,
    pub kind: ParamKind,
    pub default: ParamValue,
}

impl ParamSpec {
    pub fn new(
        key: &'static str,
        label: &'static str,
        help: &'static str,
        kind: ParamKind,
        default: ParamValue,
    ) -> Self {
        Self {
            key,
            label,
            help,
            kind,
            default,
        }
    }
}

/// The edited values handed to `Benchmark::configure`.
///
/// Accessors return `Err` on a missing or mistyped key rather than falling back
/// to a default: a benchmark that reads a key its schema never declared is a
/// bug, and silently substituting a value would hide it behind a plausible
/// number.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamValues(pub(crate) BTreeMap<String, ParamValue>);

impl ParamValues {
    /// Starting values for a schema. The spec is the only source of defaults.
    pub fn defaults(specs: &[ParamSpec]) -> Self {
        Self(
            specs
                .iter()
                .map(|s| (s.key.to_string(), s.default.clone()))
                .collect(),
        )
    }

    pub fn set(&mut self, key: impl Into<String>, value: ParamValue) {
        self.0.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&ParamValue> {
        self.0.get(key)
    }

    fn require(&self, key: &str) -> Result<&ParamValue> {
        self.0
            .get(key)
            .ok_or_else(|| anyhow!("parameter {key:?} was never set"))
    }

    fn mistyped(key: &str, want: &str, got: &ParamValue) -> anyhow::Error {
        anyhow!("parameter {key:?} is {}, expected {want}", got.type_name())
    }

    pub fn bool(&self, key: &str) -> Result<bool> {
        match self.require(key)? {
            ParamValue::Bool(b) => Ok(*b),
            other => Err(Self::mistyped(key, "bool", other)),
        }
    }

    pub fn int(&self, key: &str) -> Result<i64> {
        match self.require(key)? {
            ParamValue::Int(i) => Ok(*i),
            other => Err(Self::mistyped(key, "int", other)),
        }
    }

    /// Convenience for the many count-like parameters.
    pub fn usize(&self, key: &str) -> Result<usize> {
        let v = self.int(key)?;
        usize::try_from(v).map_err(|_| anyhow!("parameter {key:?} must not be negative, got {v}"))
    }

    pub fn float(&self, key: &str) -> Result<f64> {
        match self.require(key)? {
            ParamValue::Float(v) => Ok(*v),
            // An integer literal typed into a float field is unambiguous.
            ParamValue::Int(i) => Ok(*i as f64),
            other => Err(Self::mistyped(key, "float", other)),
        }
    }

    pub fn text(&self, key: &str) -> Result<&str> {
        match self.require(key)? {
            ParamValue::Text(s) => Ok(s.as_str()),
            other => Err(Self::mistyped(key, "text", other)),
        }
    }

    pub fn int_list(&self, key: &str) -> Result<&[i64]> {
        match self.require(key)? {
            ParamValue::IntList(v) => Ok(v.as_slice()),
            other => Err(Self::mistyped(key, "int-list", other)),
        }
    }

    /// Every value must parse-check against its spec. Called by `configure`
    /// implementations before touching any of them, so a bad field is reported
    /// against the field rather than as a mid-run failure.
    pub fn validate_against(&self, specs: &[ParamSpec]) -> Result<()> {
        for key in self.0.keys() {
            if !specs.iter().any(|spec| spec.key == key) {
                let known = specs.iter().map(|spec| spec.key).collect::<Vec<_>>();
                bail!(
                    "unknown parameter {key:?} — this benchmark takes: {}",
                    known.join(", ")
                );
            }
        }
        for spec in specs {
            let value = self.require(spec.key)?;
            // Round-trip through the kind: the domain check lives in exactly
            // one place (`ParamKind::parse`) for both editing and validation.
            spec.kind
                .parse(&value.to_edit_string())
                .map_err(|e| anyhow!("{}: {e}", spec.label))?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "params_tests.rs"]
mod tests;
