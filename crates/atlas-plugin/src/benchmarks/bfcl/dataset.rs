// SPDX-License-Identifier: AGPL-3.0-only

//! Reading the materialized BFCL table and applying the draw to it.
//!
//! `dataset.jsonl` is written subset-by-subset in `SINGLE_TURN_SUBSETS` order,
//! and within a subset in file order. That ordering IS the sample selection:
//! the draw takes the FIRST `n` rows of each subset, so anything that reorders
//! rows on the way in silently changes which samples are scored.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

use super::draw::{self, DrawSpec};

/// One materialized BFCL sample.
#[derive(Clone, Debug, Deserialize)]
pub struct Sample {
    pub subset: String,
    pub sample_id: String,
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
}

/// Load the table, keeping only what the draw selects.
///
/// Streaming per line rather than reading the file into one `Vec` first: the
/// full table is ~3.6k samples with whole tool schemas attached, and a 62 %
/// draw has no reason to hold the other 38 % in memory.
pub fn load(path: &Path, spec: &DrawSpec) -> Result<Vec<Sample>> {
    let text = std::fs::read_to_string(path).with_context(|| {
        format!(
            "reading {} — delete ~/.atlas/artifacts/bfcl to re-provision",
            path.display()
        )
    })?;
    let totals = totals_of(&text)?;
    let plan: BTreeMap<String, usize> = draw::plan(spec, &totals).into_iter().collect();

    let mut taken: BTreeMap<String, usize> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let sample: Sample = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: malformed sample", path.display(), i + 1))?;
        let Some(limit) = plan.get(&sample.subset) else {
            continue;
        };
        let count = taken.entry(sample.subset.clone()).or_insert(0);
        if *count >= *limit {
            continue;
        }
        *count += 1;
        out.push(sample);
    }
    if out.is_empty() {
        bail!("the draw selected no samples — check the categories and percentages");
    }
    // Sorted by subset then original order, matching the reference's
    // groupby-concat. Scoring is order-independent, but the progress readout
    // and any per-subset partial results are not.
    out.sort_by(|a, b| a.subset.cmp(&b.subset));
    Ok(out)
}

/// Per-subset row counts, without materializing the samples.
pub fn totals(path: &Path) -> Result<BTreeMap<String, usize>> {
    totals_of(&std::fs::read_to_string(path)?)
}

fn totals_of(text: &str) -> Result<BTreeMap<String, usize>> {
    #[derive(Deserialize)]
    struct SubsetOnly {
        subset: String,
    }
    let mut totals = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        // The counting pass sees a corrupt line before the parsing pass does,
        // so it has to name the line too — otherwise the error surfaces as a
        // bare serde message with no way to find the row.
        let row: SubsetOnly = serde_json::from_str(line)
            .with_context(|| format!("line {}: malformed sample", i + 1))?;
        *totals.entry(row.subset).or_insert(0) += 1;
    }
    Ok(totals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, rows: &[(&str, usize)]) -> std::path::PathBuf {
        let p =
            std::env::temp_dir().join(format!("atlas-bfcl-ds-{name}-{}.jsonl", std::process::id()));
        let mut text = String::new();
        for (subset, n) in rows {
            for i in 0..*n {
                text.push_str(&format!(
                    r#"{{"subset":"{subset}","sample_id":"{subset}_{i}","messages":[],"tools":[],"tool_choice":"auto"}}"#
                ));
                text.push('\n');
            }
        }
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn the_draw_takes_the_first_n_of_each_subset() {
        let p = write(
            "head",
            &[
                ("simple_python", 10),
                ("simple_java", 10),
                ("irrelevance", 10),
            ],
        );
        let spec = DrawSpec {
            categories: vec!["non_live".into()],
            category_pct: [("non_live".to_string(), 50.0)].into_iter().collect(),
            subset_floor: None,
        };
        let s = load(&p, &spec).unwrap();
        assert_eq!(
            s.len(),
            10,
            "50% of both selected subsets, and irrelevance is excluded"
        );
        let ids: Vec<&str> = s.iter().map(|x| x.sample_id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "simple_java_0",
                "simple_java_1",
                "simple_java_2",
                "simple_java_3",
                "simple_java_4",
                "simple_python_0",
                "simple_python_1",
                "simple_python_2",
                "simple_python_3",
                "simple_python_4"
            ],
            "head(n), not a random sample"
        );
    }

    #[test]
    fn totals_are_counted_without_parsing_whole_samples() {
        let p = write("totals", &[]);
        std::fs::write(
            &p,
            concat!(
                "{\"subset\":\"multiple\"}\n",
                "{\"subset\":\"multiple\"}\n",
                "{\"subset\":\"multiple\"}\n",
                "{\"subset\":\"live_simple\"}\n",
                "{\"subset\":\"live_simple\"}\n",
            ),
        )
        .unwrap();
        let t = totals(&p).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t["multiple"], 3);
        assert_eq!(t["live_simple"], 2);
    }

    #[test]
    fn an_empty_draw_is_an_error_rather_than_a_zero_sample_run() {
        let p = write("empty", &[("live_relevance", 4)]);
        let err = load(&p, &DrawSpec::golden()).unwrap_err().to_string();
        assert!(err.contains("selected no samples"), "{err}");
    }

    #[test]
    fn a_malformed_line_names_its_line_number() {
        let p = write("bad", &[("multiple", 1)]);
        let mut text = std::fs::read_to_string(&p).unwrap();
        text.push_str("{not json}\n");
        std::fs::write(&p, text).unwrap();
        let err = load(&p, &DrawSpec::full()).unwrap_err().to_string();
        assert!(err.contains("line 2: malformed sample"), "{err}");
    }
}
