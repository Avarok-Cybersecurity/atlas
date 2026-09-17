// SPDX-License-Identifier: AGPL-3.0-only

//! Test-only safetensors-header readers for the qwen4_exp checkpoints.
//!
//! Parse a checkpoint's `model.safetensors.index.json` and shard headers
//! WITHOUT uploading anything, so an offline test can check a real snapshot's
//! layout for the price of a few reads instead of a 75 GB load. Split out of
//! `qwen4_exp.rs` for the 500-LoC cap.

use anyhow::{Context, Result};

/// The PLE table's shard layout, read straight from a checkpoint's
/// safetensors header.
///
/// Exists so a test can rebuild the segmented row cache WITHOUT loading a
/// 75 GB model — the gather is the one part of PLE whose failure is invisible
/// downstream, so it needs a cheap isolated arm.
pub fn ple_shard_layout(snapshot: &str) -> Result<(Vec<(std::path::PathBuf, u64)>, u64)> {
    use std::io::{Read, Seek, SeekFrom};
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        std::path::Path::new(snapshot).join("model.safetensors.index.json"),
    )?)?;
    let map = idx["weight_map"].as_object().context("weight_map")?;
    let mut names: Vec<(usize, &String)> = map
        .keys()
        .filter(|k| k.contains(".ngram_embedding.shard_"))
        .map(|k| {
            let n = k
                .rsplit("shard_")
                .next()
                .and_then(|r| r.split('.').next())
                .and_then(|r| r.parse().ok())
                .unwrap_or(usize::MAX);
            (n, k)
        })
        .collect();
    names.sort();
    anyhow::ensure!(!names.is_empty(), "no PLE shards in {snapshot}");

    // Header per FILE, read once and reused. The released NVFP4 checkpoint
    // spreads these 128 shards across ten `model-plefp8-*.safetensors`, so an
    // offset is only meaningful against its own file's `data_start` — computing
    // every one against shard 0's file put each row in the wrong place, when it
    // did not simply refuse to load.
    let mut headers: std::collections::HashMap<String, (serde_json::Value, u64)> =
        std::collections::HashMap::new();
    let mut shards = Vec::with_capacity(names.len());
    let mut rows_per = 0u64;
    for (i, name) in &names {
        let file = map[name.as_str()].as_str().context("shard file")?;
        if !headers.contains_key(file) {
            let path = std::path::Path::new(snapshot).join(file);
            let mut fh = std::fs::File::open(&path)?;
            let mut len = [0u8; 8];
            fh.read_exact(&mut len)?;
            let hlen = u64::from_le_bytes(len);
            let mut hdr = vec![0u8; hlen as usize];
            fh.seek(SeekFrom::Start(8))?;
            fh.read_exact(&mut hdr)?;
            headers.insert(file.to_owned(), (serde_json::from_slice(&hdr)?, 8 + hlen));
        }
        let (hdr, data_start) = &headers[file];
        let e = &hdr[name.as_str()];
        let off = e["data_offsets"][0].as_u64().context("data_offsets")?;
        let rows = e["shape"][0].as_u64().context("shape")?;
        if *i == 0 {
            rows_per = rows;
        }
        anyhow::ensure!(
            rows == rows_per,
            "shard {i} has {rows} rows, not {rows_per}"
        );
        shards.push((std::path::Path::new(snapshot).join(file), data_start + off));
    }
    Ok((shards, rows_per))
}
