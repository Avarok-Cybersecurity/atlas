// SPDX-License-Identifier: AGPL-3.0-only

//! Safetensors header parsing and shard discovery for
//! [`super::FastSafetensorsLoader`].

use crate::weights::WeightDtype;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Index file format: `{ "weight_map": { "tensor_name": "shard_filename" } }`.
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Parsed entry from a safetensor header.
pub(super) struct TensorMeta {
    pub(super) name: String,
    pub(super) dtype: WeightDtype,
    pub(super) shape: Vec<usize>,
    /// Absolute byte offset in the file where the tensor bytes start.
    pub(super) abs_offset: u64,
    /// Tensor byte length.
    pub(super) len: usize,
}

/// Discover which safetensor files to load. Mirrors the resolution order in
/// `crate::weights::SafetensorsLoader::load`.
///
/// Returns `(shard_files, Some(tensor→shard))` when an index is present and
/// `(shard_files, None)` when we're loading everything in the listed files.
#[allow(clippy::type_complexity)]
pub(super) fn resolve_shards(
    model_dir: &Path,
) -> Result<(Vec<PathBuf>, Option<HashMap<String, String>>)> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let consolidated_index = model_dir.join("consolidated.safetensors.index.json");

    let actual_index = if index_path.exists() {
        Some(index_path)
    } else if consolidated_index.exists() {
        Some(consolidated_index)
    } else {
        None
    };

    if let Some(ip) = actual_index {
        let json = std::fs::read_to_string(&ip)
            .with_context(|| format!("Failed to read {}", ip.display()))?;
        let index: SafetensorsIndex = serde_json::from_str(&json)?;
        let mut shards: Vec<String> = index
            .weight_map
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        let files: Vec<PathBuf> = shards.iter().map(|s| model_dir.join(s)).collect();
        return Ok((files, Some(index.weight_map)));
    }

    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok((vec![single], None));
    }

    let mut shards: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                (n.starts_with("model.safetensors-") || n.starts_with("consolidated-"))
                    && n.ends_with(".safetensors")
            })
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        bail!(
            "No safetensor files found in {}. Expected model.safetensors*, \
             consolidated.safetensors*, or consolidated-*-of-*.safetensors",
            model_dir.display()
        );
    }
    Ok((shards, None))
}

/// Parse the safetensors header of an open file. Tensor offsets in the header
/// are relative to the start of the data section; we convert to absolute file
/// offsets up-front so downstream callers only need the fd.
pub(super) fn parse_header(file: &mut File) -> Result<Vec<TensorMeta>> {
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;
    if header_size > 64 * 1024 * 1024 {
        bail!("safetensor header too large ({} bytes)", header_size);
    }
    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)?;
    let data_start: u64 = 8 + header_size as u64;

    let json: Value = serde_json::from_slice(&header_buf)?;
    let obj = json
        .as_object()
        .context("invalid safetensor header (not an object)")?;

    let mut out = Vec::with_capacity(obj.len());
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = info["dtype"].as_str().unwrap_or("BF16");
        let dtype = match dtype_str {
            "F32" => WeightDtype::FP32,
            "BF16" => WeightDtype::BF16,
            "U8" => WeightDtype::UInt8,
            "F8_E4M3" => WeightDtype::FP8E4M3,
            "F8_E8M0" => WeightDtype::FP8E8M0,
            "I64" => WeightDtype::Int64,
            other => bail!("Unsupported safetensors dtype '{other}' for tensor {name}"),
        };
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let offsets = info["data_offsets"]
            .as_array()
            .context("tensor missing data_offsets")?;
        let rel_start = offsets[0].as_u64().context("bad data_offsets[0]")?;
        let rel_end = offsets[1].as_u64().context("bad data_offsets[1]")?;
        let len = (rel_end - rel_start) as usize;
        out.push(TensorMeta {
            name: name.clone(),
            dtype,
            shape,
            abs_offset: data_start + rel_start,
            len,
        });
    }
    // Sort by offset so the reader does sequential disk access.
    out.sort_by_key(|t| t.abs_offset);
    Ok(out)
}
