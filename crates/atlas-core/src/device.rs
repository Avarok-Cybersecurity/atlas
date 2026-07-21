// SPDX-License-Identifier: AGPL-3.0-only

//! Device-context wrapper + per-hardware constants.
//!
//! `sm121` is a pure-constants module (NUM_SMS, SMEM_PER_SM, …) that
//! kernel-launch heuristics in spark-model consume regardless of the
//! active backend — keeping it always-compiled lets metal-feature
//! builds reuse the same dispatch tables and code paths.
//!
//! `AtlasDevice` is the cudarc-wrapped device handle, gated behind
//! the `cuda` feature.

/// SM121 hardware constants for DGX Spark GB10.
pub mod sm121 {
    /// Number of streaming multiprocessors
    pub const NUM_SMS: u32 = 48;

    /// Slot capacity of the split-K paged-attention decode workspace.
    ///
    /// The split-K decode kernel writes one `[o[head_dim], m, l]` partial per
    /// `(seq, q_head, split)` triple — workspace layout
    /// `[num_seqs, num_q_heads, num_splits, head_dim + 2]`, grid
    /// `(num_q_heads, num_splits, num_seqs)`; see
    /// `kernels/gb10/common/paged_decode_attn.cu`. The buffer is allocated for
    /// this many slots (`buffers::sizes`), so every dispatch MUST satisfy
    ///
    /// ```text
    /// num_seqs * num_q_heads * num_splits <= SPLITK_WORKSPACE_SLOTS
    /// ```
    ///
    /// or the kernel writes past the end of the workspace. Use
    /// `qwen3_attention::split_k_num_splits`, which enforces this; do not
    /// open-code the split count. Raising the split count for occupancy
    /// therefore requires raising this constant (and the allocation) first.
    pub const SPLITK_WORKSPACE_SLOTS: u32 = NUM_SMS;

    /// Shared memory per SM (bytes)
    pub const SMEM_PER_SM: usize = 99 * 1024; // 99 KB

    /// Max registers per thread
    pub const MAX_REGS_PER_THREAD: u32 = 255;

    /// Max threads per block
    pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

    /// Warp size
    pub const WARP_SIZE: u32 = 32;

    /// Memory bandwidth (GB/s) — LPDDR5X unified
    pub const MEMORY_BW_GBS: f64 = 273.0;

    /// Compute capability
    pub const COMPUTE_MAJOR: u32 = 12;
    pub const COMPUTE_MINOR: u32 = 1;
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    use crate::error::{AtlasError, Result};

    /// Wrapper around a cudarc CudaContext with SM121-specific configuration.
    #[derive(Clone)]
    pub struct AtlasDevice {
        pub ctx: Arc<CudaContext>,
        pub ordinal: usize,
    }

    impl AtlasDevice {
        /// Initialize an Atlas device on the given GPU ordinal.
        pub fn new(ordinal: usize) -> Result<Self> {
            let ctx = CudaContext::new(ordinal).map_err(AtlasError::CudaDriver)?;
            Ok(Self { ctx, ordinal })
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::AtlasDevice;
