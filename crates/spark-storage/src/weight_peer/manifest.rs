// SPDX-License-Identifier: AGPL-3.0-only
//
// Weight-staging manifest types + address math (un-gated, CUDA-free, verbs-free).
//
// This half is exactly what the RDMA clients import: `WeightManifest`,
// `WeightTensorRecord`, `rail_for_tensor`, `tensor_remote_addr`. Keeping it free
// of any back-edge into the server (`serve`/`shard`) is what makes the LoRA
// carve-out (`weight_lora_rdma`) a clean lift.

use serde::{Deserialize, Serialize};

/// One tensor's placement inside a staged model, mirroring the safetensors
/// header exactly: `offset_in_shard` is the ABSOLUTE file offset (8-byte size
/// prefix + header + the tensor's data-section start), `len` is the raw
/// contiguous byte count (`data_offsets[1] - data_offsets[0]`). The client
/// RDMA-READs exactly `[shard_base + offset_in_shard .. + len)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightTensorRecord {
    /// HuggingFace tensor name — the `WeightStore` key, verbatim.
    pub name: String,
    /// Raw safetensors dtype string (`"BF16"`, `"F8_E4M3"`, `"I8"`, …). The
    /// client maps it via `WeightDtype::from_safetensors_str` — the same closed
    /// mapping the disk loaders use.
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute byte offset of the tensor's first byte within its shard file.
    pub offset_in_shard: u64,
    /// Tensor byte length (authoritative — do NOT recompute from shape; packed
    /// NVFP4 lengths differ from `numel * byte_size`).
    pub len: u64,
    /// Index into [`WeightManifest::shard_files`].
    pub shard_index: u32,
    /// True for tensors from `extra_weights.safetensors` (grafted MTP etc.):
    /// loaded with NO expert-skip filter, exactly like the disk loaders.
    pub extra: bool,
}

/// A staged model's manifest: the geometry the client needs to reconstruct a
/// byte-identical `WeightStore`. Published as length-prefixed JSON right after
/// the client's model request. The per-shard `(base, rkey)` MR handles ride the
/// verbs handshake separately (see the module doc).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightManifest {
    pub version: u32,
    /// The resolved model id/path the peer staged (echo for the client's log).
    pub model_id: String,
    /// Shard file names, shard-indexed. `shard_index` in each tensor record
    /// indexes this list; the verbs `layers` vector is per-shard in this order.
    pub shard_files: Vec<String>,
    /// Byte length of each shard file, shard-indexed (parallels `shard_files`).
    pub shard_lens: Vec<u64>,
    pub tensors: Vec<WeightTensorRecord>,
}

impl WeightManifest {
    pub const VERSION: u32 = 1;

    /// Number of shard files (== the per-rail `layers` MR count the peer
    /// publishes and the client validates against).
    pub fn num_shards(&self) -> usize {
        self.shard_files.len()
    }

    /// Total registered bytes across all shards (the whole-file MRs) — the
    /// figure charged against the blade `CommitLedger` once per staged model.
    pub fn total_shard_bytes(&self) -> u64 {
        self.shard_lens.iter().sum()
    }
}

/// Rail selection for a tensor under dual-rail striping: tensor `N` is served
/// over rail `N % n_rails`. Factored out (un-gated) so the striping is unit-
/// testable off the RDMA path — the client's read loop calls this so the tested
/// logic and the shipped logic are the same. `n_rails` is clamped to `>= 1`.
pub fn rail_for_tensor(tensor_index: usize, n_rails: usize) -> usize {
    tensor_index % n_rails.max(1)
}

/// Absolute peer virtual address of a tensor's first byte: the shard's whole-
/// file REMOTE_READ MR base plus the tensor's ABSOLUTE in-shard offset (the
/// safetensors data-section offset, which already includes the 8-byte size
/// prefix + header). The client RDMA-READs `[addr .. addr + len)`. Factored out
/// (un-gated) so the address math is unit-testable off the RDMA path.
pub fn tensor_remote_addr(shard_base: u64, offset_in_shard: u64) -> u64 {
    shard_base + offset_in_shard
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
