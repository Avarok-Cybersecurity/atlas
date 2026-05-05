// SPDX-License-Identifier: AGPL-3.0-only

use cudarc::driver::CudaStream;
use std::sync::Arc;

use crate::device::AtlasDevice;
use crate::error::Result;

/// CUDA stream wrapper for asynchronous kernel execution.
pub struct AtlasStream {
    pub stream: Arc<CudaStream>,
    pub device: AtlasDevice,
}

impl AtlasStream {
    /// Create a new CUDA stream on the given device.
    pub fn new(device: &AtlasDevice) -> Result<Self> {
        let stream = device
            .ctx
            .new_stream()
            .map_err(crate::error::AtlasError::CudaDriver)?;
        Ok(Self {
            stream,
            device: device.clone(),
        })
    }
}
