// SPDX-License-Identifier: AGPL-3.0-only

//! Build an atlas-core [`ModelConfig`] from a bare GGUF file's metadata, so a
//! directory containing only a `.gguf` (no `config.json`) can be served.
//!
//! `config_from_gguf` lives in atlas-core (which cannot see this crate's GGUF
//! parser); this module bridges the two — it impls the atlas-core [`GgufMeta`]
//! accessor over [`GgufFile`] and supplies the two tensor-section facts the
//! builder needs (vocab rows + presence of an untied `output.weight`).

use std::path::Path;

use anyhow::{Context, Result};
use atlas_core::config::{GgufConfigInputs, GgufMeta, ModelConfig, config_from_gguf};

use super::container::GgufFile;
use super::find_gguf;

impl GgufMeta for GgufFile {
    fn get_u64(&self, key: &str) -> Option<u64> {
        GgufFile::get_u64(self, key)
    }
    fn get_f64(&self, key: &str) -> Option<f64> {
        GgufFile::get_f64(self, key)
    }
    fn get_str(&self, key: &str) -> Option<&str> {
        GgufFile::get_str(self, key)
    }
    fn get_arr_len(&self, key: &str) -> Option<usize> {
        GgufFile::arr_len(self, key)
    }
}

/// Build a [`ModelConfig`] from the `.gguf` in `model_dir`, with no `config.json`.
///
/// vocab is taken from the `token_embd.weight` shape (ggml dims are
/// `[embedding_length, vocab]`, so the trailing dim is the vocab), and tied
/// embeddings are inferred from the absence of an `output.weight` tensor.
pub fn config_from_gguf_dir(model_dir: &Path) -> Result<ModelConfig> {
    let path = find_gguf(model_dir)
        .with_context(|| format!("no .gguf file in {}", model_dir.display()))?;
    let file =
        std::fs::File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    // SAFETY: same mmap contract as the GGUF weight loader; the map outlives the
    // borrow below and the file is not mutated concurrently.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let gguf = GgufFile::parse(&mmap)
        .with_context(|| format!("failed to parse GGUF metadata: {}", path.display()))?;

    let token_embd_vocab = gguf
        .tensor("token_embd.weight")
        .and_then(|t| t.dims.last().copied());
    let has_output_weight = gguf.tensor("output.weight").is_some();

    let inputs = GgufConfigInputs {
        meta: &gguf,
        token_embd_vocab,
        has_output_weight,
    };
    config_from_gguf(&inputs).context("failed to build ModelConfig from GGUF metadata")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GgufMeta bridge must forward to the inherent getters (no recursion)
    /// and remap `get_arr_len` → `arr_len`.
    #[test]
    fn gguf_meta_bridge_forwards() {
        // Hand-built minimal GGUF: one UINT32 KV + one STRING-array KV.
        let mut b: Vec<u8> = Vec::new();
        let push_u32 = |b: &mut Vec<u8>, v: u32| b.extend_from_slice(&v.to_le_bytes());
        let push_u64 = |b: &mut Vec<u8>, v: u64| b.extend_from_slice(&v.to_le_bytes());
        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        push_u32(&mut b, 0x4655_4747); // "GGUF"
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 0); // tensor_count
        push_u64(&mut b, 2); // kv_count
        // key = "qwen3.block_count" : UINT32 = 28
        push_str(&mut b, "qwen3.block_count");
        push_u32(&mut b, 4); // UINT32
        push_u32(&mut b, 28);
        // key = "tokenizer.ggml.tokens" : ARRAY<STRING> len 3
        push_str(&mut b, "tokenizer.ggml.tokens");
        push_u32(&mut b, 9); // ARRAY
        push_u32(&mut b, 8); // elem type STRING
        push_u64(&mut b, 3); // len
        for s in ["a", "bb", "ccc"] {
            push_str(&mut b, s);
        }
        // Pad to the 32-byte alignment boundary so the (empty) tensor-data
        // section start is within the buffer.
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        let gguf = GgufFile::parse(&b).unwrap();
        let m: &dyn GgufMeta = &gguf;
        assert_eq!(m.get_u64("qwen3.block_count"), Some(28));
        assert_eq!(m.get_arr_len("tokenizer.ggml.tokens"), Some(3));
        assert_eq!(m.get_str("nonexistent"), None);
    }
}
