// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Readout and label file formats of the routing gate; no upstream code.
//! Responsibility: the readouts file (a safetensors file: `X` f32 `[n, d]`, `n_tokens` i64 `[n]`, `ok` u8 `[n]`, with
//! the prompt ids and a meta record in the header metadata) and the join of readouts with `labels.jsonl`.
//! The Python harness used an `.npz` here; safetensors needs no NumPy and is already read by this crate.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::{Dropped, Record};
use crate::route_gate::mat::Mat;
use crate::route_gate::tensorio::{OutTensor, TensorFile, write_safetensors};

/// Labeled rows joined with their readouts.
pub struct Inputs {
    pub rows: Vec<Record>,
    pub x: Mat,
    pub dropped: Dropped,
    pub meta: Value,
}

/// Writes a readouts file. `x` is row-major `[ids.len(), d]`.
pub fn write_readouts(
    path: &Path,
    ids: &[String],
    x: &[f32],
    d: usize,
    n_tokens: &[i64],
    ok: &[bool],
    meta: &Value,
) -> Result<()> {
    let n = ids.len();
    if x.len() != n * d || n_tokens.len() != n || ok.len() != n {
        bail!("write_readouts: inconsistent lengths");
    }
    let tensors = vec![
        ("X".to_string(), OutTensor::f32(&[n, d], x.to_vec())),
        (
            "n_tokens".to_string(),
            OutTensor::i64(&[n], n_tokens.to_vec()),
        ),
        (
            "ok".to_string(),
            OutTensor::u8(&[n], ok.iter().map(|&b| u8::from(b)).collect()),
        ),
    ];
    let mut metadata = BTreeMap::new();
    metadata.insert("ids_json".to_string(), serde_json::to_string(ids)?);
    metadata.insert("meta_json".to_string(), serde_json::to_string(meta)?);
    write_safetensors(path, &tensors, &metadata)?;
    Ok(())
}

/// Joins `labels.jsonl` with a readouts file and drops rows that cannot be used, counting why.
pub fn load_inputs(
    labels_path: &Path,
    readouts_path: &Path,
    drop_truncated: bool,
    drop_no_answer: bool,
) -> Result<Inputs> {
    let mut labels: HashMap<String, Value> = HashMap::new();
    for ln in std::fs::read_to_string(labels_path)
        .with_context(|| format!("read {}", labels_path.display()))?
        .lines()
    {
        if !ln.trim().is_empty() {
            let r: Value = serde_json::from_str(ln)?;
            if let Some(id) = r["id"].as_str() {
                labels.insert(id.to_string(), r);
            }
        }
    }
    let file = TensorFile::open(readouts_path)?;
    let metadata = file.metadata();
    let ids: Vec<String> =
        serde_json::from_str(metadata.get("ids_json").map_or("[]", String::as_str))?;
    let xt = file.read("X")?;
    let ok = file.read("ok")?;
    let d = xt.shape.get(1).copied().unwrap_or(0);
    let mut dropped = Dropped::default();
    let (mut rows, mut data) = (Vec::new(), Vec::new());
    for (k, pid) in ids.iter().enumerate() {
        let row = &xt.data[k * d..(k + 1) * d];
        let label = labels.get(pid).filter(|r| !r["passed"].is_null());
        let Some(r) = label else {
            dropped.no_label += 1;
            continue;
        };
        if ok.data[k] == 0.0 || !row.iter().all(|v| v.is_finite()) {
            dropped.no_readout += 1;
        } else if drop_truncated && r["truncated"].as_bool() == Some(true) {
            dropped.truncated += 1;
        } else if drop_no_answer && r["reason"].as_str() == Some("no_answer") {
            dropped.no_answer += 1;
        } else {
            rows.push(Record {
                id: pid.clone(),
                source: r["source"].as_str().unwrap_or_default().to_string(),
                prompt: r["prompt"].as_str().unwrap_or_default().to_string(),
                passed: r["passed"].as_bool().unwrap_or(false),
            });
            data.extend_from_slice(row);
        }
    }
    let x = if rows.is_empty() {
        Mat::zeros(0, 1)
    } else {
        Mat::new(rows.len(), d, data)?
    };
    let meta = metadata
        .get("meta_json")
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(Value::Null);
    Ok(Inputs {
        rows,
        x,
        dropped,
        meta,
    })
}
