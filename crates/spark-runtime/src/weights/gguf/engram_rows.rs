// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The engram tables of DeepSeek-V4.1 Flash, read by row on demand. Split
//! from `expert_stream.rs` (500-LoC cap).

use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use super::container::Q2Group;
use super::expert_stream::{ShardFiles, block_layer, pread};

/// One engram table: `blk.N.engram_embd.weight`, GGUF dims `[head_dim, rows]`,
/// one quant block per row.
#[derive(Clone, Copy, Debug)]
pub struct EngramTable {
    pub layer: usize,
    pub shard: usize,
    pub base: u64,
    pub rows: usize,
    pub row_bytes: usize,
    pub head_dim: usize,
    pub ggml_type_id: u32,
}

/// Row reads from the engram tables. The tables are ~30 GiB each and a token
/// touches 24 rows per table, so nothing is cached: every call is `ids.len()`
/// positional reads of one block.
pub struct EngramRowReader {
    files: Arc<ShardFiles>,
    tables: Vec<EngramTable>,
}

impl EngramRowReader {
    pub fn new(files: Arc<ShardFiles>) -> Result<Self> {
        let mut tables = Vec::new();
        for (shard, t) in files.with_suffix("engram_embd.weight") {
            let Some(layer) = block_layer(&t.name) else {
                continue;
            };
            ensure!(
                t.dims.len() == 2,
                "{}: expected [head_dim, rows], got {:?}",
                t.name,
                t.dims
            );
            let (head_dim, rows) = (t.dims[0], t.dims[1]);
            let (qk, bb) = t.ggml_type.block_layout(Q2Group::G128)?;
            ensure!(
                qk == head_dim,
                "{}: head_dim {head_dim} is not one {qk}-element block per row",
                t.name
            );
            tables.push(EngramTable {
                layer,
                shard,
                base: files.abs_offset(shard, t),
                rows,
                row_bytes: bb,
                head_dim,
                ggml_type_id: t.ggml_type.id(),
            });
        }
        ensure!(!tables.is_empty(), "no engram_embd tensors in this GGUF");
        tables.sort_by_key(|t| t.layer);
        Ok(EngramRowReader { files, tables })
    }

    pub fn tables(&self) -> &[EngramTable] {
        &self.tables
    }

    pub fn table(&self, layer: usize) -> Option<&EngramTable> {
        self.tables.iter().find(|t| t.layer == layer)
    }

    /// Read rows `ids` of layer `layer`'s table into `dst`
    /// (`ids.len() * row_bytes` bytes, in `ids` order).
    pub fn read_rows(&self, layer: usize, ids: &[u64], dst: &mut [u8]) -> Result<()> {
        let t = self
            .table(layer)
            .with_context(|| format!("layer {layer} has no engram table"))?;
        ensure!(
            dst.len() == ids.len() * t.row_bytes,
            "dst is {} bytes for {} rows of {}",
            dst.len(),
            ids.len(),
            t.row_bytes
        );
        let file = self.files.file(t.shard);
        for (i, &id) in ids.iter().enumerate() {
            ensure!(
                (id as usize) < t.rows,
                "engram row {id} >= {} (layer {layer})",
                t.rows
            );
            let off = t.base + id * t.row_bytes as u64;
            pread(file, off, &mut dst[i * t.row_bytes..(i + 1) * t.row_bytes])
                .with_context(|| format!("engram layer {layer} row {id}"))?;
        }
        Ok(())
    }
}
