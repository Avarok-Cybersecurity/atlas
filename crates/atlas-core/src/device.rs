// SPDX-License-Identifier: AGPL-3.0-only

use cudarc::driver::CudaContext;
use std::sync::Arc;

use crate::error::{AtlasError, Result};

/// SM121 hardware constants for DGX Spark GB10.
pub mod sm121 {
    /// Number of streaming multiprocessors
    pub const NUM_SMS: u32 = 48;

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
