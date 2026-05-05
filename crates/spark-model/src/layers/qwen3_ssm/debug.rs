// SPDX-License-Identifier: AGPL-3.0-only

//! Debug print helpers.

use super::*;

impl Qwen3SsmLayer {
    /// Debug: read first N BF16 values from device and log them.
    pub(super) fn debug_bf16(gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize) {
        let mut buf = vec![0u8; n * 2];
        if gpu.copy_d2h(ptr, &mut buf).is_err() {
            return;
        }
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let lo = buf[i * 2];
                let hi = buf[i * 2 + 1];
                f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
            })
            .collect();
        tracing::info!("  SSM {label}: {:?}", vals);
    }

    /// Debug: read first N FP32 values from device and log them.
    pub(super) fn debug_f32(gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize) {
        let mut buf = vec![0u8; n * 4];
        if gpu.copy_d2h(ptr, &mut buf).is_err() {
            return;
        }
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                f32::from_le_bytes([buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]])
            })
            .collect();
        tracing::info!("  SSM {label}: {:?}", vals);
    }
}
