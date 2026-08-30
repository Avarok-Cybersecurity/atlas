// SPDX-License-Identifier: AGPL-3.0-only

//! The category prompt corpus, compiled into the binary.
//!
//! One JSON document with a manifest and a flat row list, rather than JSONL:
//! the manifest lets a report name the corpus it measured, and a standardized
//! score needs its corpus to be a citable object rather than a bag of lines.
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

const CORPUS: &str = include_str!("../../../assets/expert-categories/corpus.json");

/// One prompt row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub category: String,
    pub id: String,
    pub prompt: String,
}

/// The corpus document's own declaration of what it contains.
///
/// Carried so a run can state the corpus it measured rather than inferring
/// it: `categories` here is the authoritative order, and a mismatch against
/// the rows is a corpus defect the loader refuses rather than papers over.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub name: String,
    pub categories: Vec<String>,
    pub prompts_per_category: usize,
}

/// Parse the whole corpus in file order.
pub fn load() -> Result<Vec<Row>> {
    Ok(load_with_manifest()?.1)
}

/// Parse the corpus and its manifest.
pub fn load_with_manifest() -> Result<(Manifest, Vec<Row>)> {
    let doc: serde_json::Value =
        serde_json::from_str(CORPUS).context("the compiled-in corpus is not valid JSON")?;
    let str_field = |v: &serde_json::Value, name: &str| -> Result<String> {
        v.get(name)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("corpus: missing `{name}`"))
    };
    let manifest = Manifest {
        name: str_field(&doc, "name")?,
        categories: doc
            .get("categories")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("corpus: missing `categories`"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow::anyhow!("corpus: a category name is not a string"))
            })
            .collect::<Result<Vec<_>>>()?,
        prompts_per_category: doc
            .get("prompts_per_category")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("corpus: missing `prompts_per_category`"))?
            as usize,
    };

    let rows_val = doc
        .get("rows")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("corpus: missing `rows`"))?;
    let mut rows = Vec::with_capacity(rows_val.len());
    for (i, v) in rows_val.iter().enumerate() {
        rows.push(Row {
            category: str_field(v, "category").with_context(|| format!("corpus row {i}"))?,
            id: str_field(v, "id").with_context(|| format!("corpus row {i}"))?,
            prompt: str_field(v, "prompt").with_context(|| format!("corpus row {i}"))?,
        });
    }
    if rows.is_empty() {
        bail!("the compiled-in corpus is empty");
    }
    // The manifest is what a report quotes; if it disagreed with the rows the
    // report would describe a corpus that was not measured.
    let found = categories(&rows);
    if found != manifest.categories {
        bail!(
            "corpus manifest lists {:?} but the rows contain {:?}",
            manifest.categories,
            found
        );
    }
    Ok((manifest, rows))
}

/// Content fingerprint of the corpus: SHA-256 over `category\x01id\x01prompt`
/// rows joined by `\x02`, in file order.
///
/// Over CONTENT, not file bytes, so reformatting the JSON does not change the
/// corpus's identity while editing a single prompt does. That identity is
/// part of an EAS score's meaning — the category taxonomy sets the ceiling a
/// model can reach, so a score from another corpus is a different
/// measurement, not a better or worse one.
pub fn content_hash(rows: &[Row]) -> String {
    use sha2::{Digest, Sha256};
    let joined = rows
        .iter()
        .map(|r| format!("{}\u{1}{}\u{1}{}", r.category, r.id, r.prompt))
        .collect::<Vec<_>>()
        .join("\u{2}");
    let mut h = Sha256::new();
    h.update(joined.as_bytes());
    format!("{:x}", h.finalize())
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
