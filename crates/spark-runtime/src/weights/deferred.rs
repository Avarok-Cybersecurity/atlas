// SPDX-License-Identifier: AGPL-3.0-only

//! Tensors the checkpoint loader deliberately does NOT upload, and the hook a
//! model's own weight loader uses to say so.
//!
//! # Two ways a tensor becomes deferred
//!
//! 1. **`is_ngram_table`** — a NAME rule baked into the runtime, for the
//!    LongCat / Qwen3.8-Flash-Next embedding tables. Always on, because those
//!    tables (63 GB to 102 GB) cannot be uploaded on any box we serve.
//! 2. **[`DeferHook`]** — a predicate the MODEL's loader supplies, keyed on
//!    `(name, on-disk dtype)`. Off unless a loader asks for it, and plumbed
//!    exactly like `skip_vision`: `serve_phases/weights.rs` asks
//!    `ModelWeightLoader::defer_predicate` once and hands the answer to
//!    whichever checkpoint loader runs.
//!
//! Both end in the same place — a [`DeferredTensor`] in the store's
//! `deferred` map — so a consumer never has to know which rule withheld it.
//!
//! # Why a loader would ask for this
//!
//! The rule is NOT "this tensor is huge". It is "the bytes that reach the
//! device are not the bytes on disk, and the model's loader is the only thing
//! that knows how to make them".
//!
//! `nvidia/GLM-5.3-Flash-NVFP4` ships the MTP block's 288 routed experts as
//! full-width BF16 while the forward is w4a16 end to end. Every one of those
//! tensors is read once, by `glm5_next_load::bind_expert`, quantised, and
//! replaced by an NVFP4 triple ~3.6x smaller. Uploading the BF16 first is
//! ~7.25 GB per EP=2 rank of device residency that lasts from the sweep until
//! `prune_after_load` — i.e. across the WHOLE 45-layer build, which on a
//! unified-memory box is where the host-memory guard fires. Measured
//! 2026-09-21 on a 2-rank boot: survived at a 2048-token shape (host floor
//! 1.74 GB), killed rank 0 at layer 38 at the 32K shape (935 MB against a
//! 1200 MB floor) BEFORE the MTP bind could free anything.
//!
//! Deferring makes that transient structurally impossible: the BF16 never
//! leaves the page cache, and the only expert bytes that reach the device are
//! the NVFP4 ones, for the experts this rank owns.
//!
//! # Contract, both halves of it
//!
//! * The loader that defers a tensor MUST read it back — nothing else will.
//!   A predicate that over-matches withholds a tensor a binder expects, and
//!   `WeightStore::get` then fails by NAME at bind time rather than producing
//!   a subtly wrong model. That is the intended failure mode; it is still the
//!   loader author's job not to reach it.
//! * The OOM pre-flight MUST agree with the predicate. A deferred tensor is
//!   never part of the sweep peak, so counting it at its on-disk size refuses
//!   loads that fit — the same reasoning that already excludes the n-gram
//!   tables. The pre-flight and the retain loop are handed the SAME closure
//!   for exactly this reason.
//!
//! # 🪤 Keyed on the ON-DISK dtype, and F16 is never offered
//!
//! A deferred tensor is served from its own bytes in the shard, so the
//! predicate is asked about the width those bytes actually have. F16 is the
//! one width both disk loaders REWRITE on the way to the store
//! (`f16_to_bf16_bytes`), so a (path, offset) locator would hand its reader
//! F16 where the rest of the engine expects BF16. F16 tensors are therefore
//! never deferred — by the retain loop and by the pre-flight alike, which is
//! the only way the two can stay the same rule.

use anyhow::{Context, Result};

use super::WeightDtype;

/// A model loader's answer to "which tensors will I read from disk myself?".
///
/// `(tensor name, store dtype) -> defer`. `Arc` because both checkpoint
/// loaders hold it for the length of a load and the fast loader's pre-flight
/// and retain loop both call it; `Send + Sync` because the fast loader's
/// pipeline is threaded.
pub type DeferHook = std::sync::Arc<dyn Fn(&str, WeightDtype) -> bool + Send + Sync>;

/// Where a skipped tensor lives, so a consumer can read it in place.
#[derive(Clone, Debug)]
pub struct DeferredTensor {
    /// Shard file containing the tensor.
    pub path: std::path::PathBuf,
    /// ABSOLUTE byte offset of the tensor's first element in that file
    /// (safetensors header length + the tensor's `data_offsets[0]`).
    pub offset: u64,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

impl DeferredTensor {
    /// The tensor's on-disk footprint.
    ///
    /// Only the fixed-width dtypes a safetensors header can carry reach here,
    /// so `numel * byte_size` is exact — the block-based GGUF dtypes
    /// ([`WeightDtype::PackedQ2_0`] and the K-quants) report 0 bytes/element
    /// and are never deferred by either rule.
    pub fn byte_size(&self) -> usize {
        let numel: usize = self.shape.iter().product();
        numel * self.dtype.byte_size()
    }

    /// Read the whole tensor's raw bytes into host memory.
    ///
    /// The consumer for a deferred tensor is usually a row cache that faults
    /// single rows off NVMe; this is the other shape of consumer — a loader
    /// that needs the tensor ONCE, at bind time, to derive the buffer it will
    /// actually upload.
    ///
    /// 🪤 `read_exact` rather than `read_to_end`: a short read means the shard
    /// is truncated or the offset is wrong, and both must be an error here
    /// rather than a shorter tensor whose element count is rejected later with
    /// a message pointing at the wrong thing. Seek-then-read rather than
    /// `pread`, because `WeightStore` builds on Windows too.
    pub fn read_host_bytes(&self) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let n = self.byte_size();
        let mut f = std::fs::File::open(&self.path)
            .with_context(|| format!("deferred tensor: opening {}", self.path.display()))?;
        f.seek(SeekFrom::Start(self.offset))?;
        let mut buf = vec![0u8; n];
        f.read_exact(&mut buf).with_context(|| {
            format!(
                "deferred tensor: reading {n} B at offset {} of {}",
                self.offset,
                self.path.display()
            )
        })?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_shard(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.safetensors");
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    #[test]
    fn byte_size_is_the_on_disk_footprint() {
        let d = DeferredTensor {
            path: std::path::PathBuf::from("x"),
            offset: 0,
            shape: vec![2048, 4096],
            dtype: WeightDtype::BF16,
        };
        assert_eq!(d.byte_size(), 2048 * 4096 * 2);
    }

    #[test]
    fn host_read_returns_exactly_the_tensor_at_its_offset() {
        // 8 filler bytes stand in for a header, then four BF16 values.
        let payload: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (_d, path) = write_shard(&[vec![0xFFu8; 8], payload.clone()].concat());
        let d = DeferredTensor {
            path,
            offset: 8,
            shape: vec![2, 2],
            dtype: WeightDtype::BF16,
        };
        assert_eq!(d.read_host_bytes().unwrap(), payload);
    }

    /// Write a one-tensor-per-entry safetensors file and return its path.
    fn write_safetensors(
        dir: &std::path::Path,
        entries: &[(&str, &str, Vec<usize>, usize)],
    ) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        let mut end = 0usize;
        for (name, dtype, shape, len) in entries {
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [end, end + len],
                }),
            );
            end += len;
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut blob = (header.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(&header);
        blob.extend_from_slice(&vec![0u8; end]);
        let path = dir.join("model.safetensors");
        std::fs::write(&path, blob).unwrap();
        path
    }

    /// 🔴 The other half of the contract: a deferred tensor is never swept, so
    /// the OOM pre-flight must not count it. Before this, the official GLM-5.3
    /// export's ~7.25 GB of full-width MTP experts were charged to a peak they
    /// never joined.
    #[test]
    fn the_preflight_estimate_drops_deferred_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_safetensors(
            dir.path(),
            &[
                ("big.weight", "BF16", vec![8, 8], 128),
                ("small.weight", "BF16", vec![4], 8),
            ],
        );
        let files = [path];

        let all = super::super::estimate_load_bytes(&files, &|_| false, &|_, _| false).unwrap();
        assert_eq!(all, 136);

        let without = super::super::estimate_load_bytes(&files, &|_| false, &|n, d| {
            n == "big.weight" && d == WeightDtype::BF16
        })
        .unwrap();
        assert_eq!(without, 8, "the deferred tensor must leave the peak");

        // 🪤 Keyed on the dtype too: the same NAME at a width the predicate
        // does not claim is still counted, because it is still uploaded.
        let wrong_dtype = super::super::estimate_load_bytes(&files, &|_| false, &|n, d| {
            n == "big.weight" && d == WeightDtype::UInt8
        })
        .unwrap();
        assert_eq!(wrong_dtype, 136);
    }

    /// F16 is rewritten on the way to the store, so it can never be served
    /// from its on-disk bytes — the estimate counts it however loudly the
    /// predicate claims it, which is exactly what the loaders then do.
    #[test]
    fn an_f16_tensor_is_counted_even_when_the_predicate_claims_everything() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_safetensors(dir.path(), &[("h.weight", "F16", vec![8, 8], 128)]);
        let got = super::super::estimate_load_bytes(&[path], &|_| false, &|_, _| true).unwrap();
        assert_eq!(got, 128);
    }

    #[test]
    fn a_truncated_shard_is_an_error_not_a_short_tensor() {
        let (_d, path) = write_shard(&[0u8; 8]);
        let d = DeferredTensor {
            path,
            offset: 0,
            shape: vec![64, 64],
            dtype: WeightDtype::BF16,
        };
        assert!(d.read_host_bytes().is_err());
    }
}
