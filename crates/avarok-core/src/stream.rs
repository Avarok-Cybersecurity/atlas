// SPDX-License-Identifier: AGPL-3.0-only

use cudarc::driver::CudaStream;
use std::sync::Arc;

use crate::device::AvarokDevice;
use crate::error::Result;

/// CUDA stream wrapper for asynchronous kernel execution.
pub struct AvarokStream {
    pub stream: Arc<CudaStream>,
    pub device: AvarokDevice,
}

impl AvarokStream {
    /// Create a new CUDA stream on the given device.
    pub fn new(device: &AvarokDevice) -> Result<Self> {
        let stream = device
            .ctx
            .new_stream()
            .map_err(crate::error::AvarokError::CudaDriver)?;
        Ok(Self {
            stream,
            device: device.clone(),
        })
    }
}
