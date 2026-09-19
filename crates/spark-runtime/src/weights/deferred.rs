// SPDX-License-Identifier: AGPL-3.0-only

use super::WeightDtype;

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
