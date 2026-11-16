// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: `tokens --probe FILE`, a tokenizer probe for serializer experiments: FILE is a JSON array of strings
//! (or a JSON-lines file of strings); every string is tokenized as one text and its token count and pieces are printed.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::serializer::HfTokenizer;

/// Reads the probe texts: a JSON array of strings, or one JSON string per line.
fn read_texts(path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let values: Vec<Value> = match serde_json::from_str::<Value>(&raw) {
        Ok(Value::Array(a)) => a,
        _ => raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?,
    };
    values
        .into_iter()
        .map(|v| match v {
            Value::String(s) => Ok(s),
            other => bail!("probe entries must be strings, got {other}"),
        })
        .collect()
}

/// Prints `index<TAB>count` per text, and the pieces when `verbose`.
pub fn cmd_probe(release: &Path, file: &Path, verbose: bool) -> Result<i32> {
    let tok = HfTokenizer::from_release(release)?;
    for (i, t) in read_texts(file)?.iter().enumerate() {
        let pieces = tok.pieces(t)?;
        println!("{i}\t{}", pieces.len());
        if verbose {
            let line: Vec<String> = pieces.iter().map(|(_, p)| format!("{p:?}")).collect();
            println!("   {}", line.join(" "));
        }
    }
    Ok(0)
}
