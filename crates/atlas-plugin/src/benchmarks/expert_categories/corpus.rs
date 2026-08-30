// SPDX-License-Identifier: AGPL-3.0-only

//! The category prompt corpus, compiled into the binary.
//!
//! `include_str!` rather than a provisioned artifact: nothing is downloaded,
//! so the provisioning machinery BFCL needs would buy nothing, and shipping
//! the rows inside the binary means a run cannot silently measure a
//! different corpus than the one this commit describes. It is an asset file
//! rather than Rust consts only because 320 rows of consts would need four
//! files to stay under the 500-line cap.
//!
//! **File order is the draw.** A draw of N takes the FIRST N rows of each
//! category in file order, so re-sorting the file changes which prompts are
//! measured without changing anything a reader would notice. This is the
//! BFCL lesson (`bfcl/dataset.rs`) applied to a corpus we own: the loader
//! preserves order, and the tests pin both the head of each category and a
//! hash of the whole file.

use anyhow::{Context, Result, bail};

const CORPUS: &str = include_str!("../../../assets/expert-categories/corpus.jsonl");

/// One prompt row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub category: String,
    pub id: String,
    pub prompt: String,
}

/// Parse the whole corpus in file order.
pub fn load() -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for (i, line) in CORPUS.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("corpus line {} is not valid JSON", i + 1))?;
        let field = |name: &str| -> Result<String> {
            v.get(name)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("corpus line {}: missing `{name}`", i + 1))
        };
        rows.push(Row {
            category: field("category")?,
            id: field("id")?,
            prompt: field("prompt")?,
        });
    }
    if rows.is_empty() {
        bail!("the compiled-in corpus is empty");
    }
    Ok(rows)
}

/// Category names in file order (first appearance).
pub fn categories(rows: &[Row]) -> Vec<String> {
    let mut seen = Vec::new();
    for r in rows {
        if !seen.iter().any(|c| c == &r.category) {
            seen.push(r.category.clone());
        }
    }
    seen
}

/// Take the first `per_category` rows of each requested category, in file
/// order. `wanted` empty = every category.
///
/// Ordering is load-bearing twice over: it decides WHICH prompts a run of
/// fewer than 32 measures, and it keeps two runs of the same size
/// comparable.
pub fn draw(rows: &[Row], per_category: usize, wanted: &[String]) -> Result<Vec<Row>> {
    if !wanted.is_empty() {
        let known = categories(rows);
        for w in wanted {
            if !known.contains(w) {
                bail!(
                    "unknown category '{w}'. The corpus has: {}",
                    known.join(", ")
                );
            }
        }
    }
    let mut taken: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    let mut out = Vec::new();
    for r in rows {
        if !wanted.is_empty() && !wanted.contains(&r.category) {
            continue;
        }
        let n = taken.entry(r.category.as_str()).or_insert(0);
        if *n < per_category {
            *n += 1;
            out.push(r.clone());
        }
    }
    Ok(out)
}

/// Parse the `categories` parameter: `"all"` or a comma-separated list.
pub fn parse_selection(spec: impl AsRef<str>) -> Vec<String> {
    let spec = spec.as_ref();
    let spec = spec.trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("all") {
        return Vec::new();
    }
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
#[path = "corpus_tests.rs"]
mod tests;
