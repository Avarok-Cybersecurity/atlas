// SPDX-License-Identifier: AGPL-3.0-only

//! DEBUG-level diagnostic readbacks for `MoeLayer::forward` (decode).
//!
//! Hoisted from `forward.rs` to keep that file under the 500 LoC cap.
//! These are pure diagnostic readbacks (d2h copy + tracing) gated behind
//! `tracing::enabled!(DEBUG)`; they perform no compute and do not affect
//! the ordered kernel-launch sequence. Each helper mirrors the original
//! inline block 1:1.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Read `n` BF16 values from `ptr` as f32 (upper-16-bit reconstruction).
fn read_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 2];
    gpu.copy_d2h(ptr, &mut buf)?;
    Ok((0..n)
        .map(|i| {
            let bits = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
}

/// Post-routing: log expert indices (u32) and weights (f32).
pub(super) fn dump_routing(
    gpu: &dyn GpuBackend,
    indices_dev: DevicePtr,
    weights_dev: DevicePtr,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let k = top_k as usize;
    let mut idx_buf = vec![0u8; k * 4];
    let mut wt_buf = vec![0u8; k * 4];
    gpu.copy_d2h(indices_dev, &mut idx_buf)?;
    gpu.copy_d2h(weights_dev, &mut wt_buf)?;
    let indices: Vec<u32> = (0..k)
        .map(|i| {
            u32::from_le_bytes([
                idx_buf[i * 4],
                idx_buf[i * 4 + 1],
                idx_buf[i * 4 + 2],
                idx_buf[i * 4 + 3],
            ])
        })
        .collect();
    let weights: Vec<f32> = (0..k)
        .map(|i| {
            f32::from_le_bytes([
                wt_buf[i * 4],
                wt_buf[i * 4 + 1],
                wt_buf[i * 4 + 2],
                wt_buf[i * 4 + 3],
            ])
        })
        .collect();
    tracing::info!("  MoE experts: {:?}, weights: {:.4?}", indices, weights);
    Ok(())
}

/// NVFP4 path: dump expert/shared gate+up scratch outputs (slot 0).
pub(super) fn dump_gate_up(
    gpu: &dyn GpuBackend,
    expert_gate_out: DevicePtr,
    expert_up_out: DevicePtr,
    shared_gate_scratch: DevicePtr,
    shared_up_scratch: DevicePtr,
    stream: u64,
) -> Result<()> {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    tracing::info!(
        "  MoE gate_out[slot0,0..8]: {:?}",
        read_bf16(gpu, expert_gate_out, 8)?
    );
    tracing::info!(
        "  MoE up_out[slot0,0..8]: {:?}",
        read_bf16(gpu, expert_up_out, 8)?
    );
    tracing::info!(
        "  MoE shared_gate_scratch[0..8]: {:?}",
        read_bf16(gpu, shared_gate_scratch, 8)?
    );
    tracing::info!(
        "  MoE shared_up_scratch[0..8]: {:?}",
        read_bf16(gpu, shared_up_scratch, 8)?
    );
    Ok(())
}

/// Dump expert down output (slot 0) and shared-expert output.
pub(super) fn dump_down(
    gpu: &dyn GpuBackend,
    expert_down_out: DevicePtr,
    shared_out: DevicePtr,
    stream: u64,
) -> Result<()> {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    tracing::info!(
        "  MoE down_out[slot0,0..8]: {:?}",
        read_bf16(gpu, expert_down_out, 8)?
    );
    tracing::info!(
        "  MoE shared_out[0..8]: {:?}",
        read_bf16(gpu, shared_out, 8)?
    );
    Ok(())
}

/// Dump the final blended MoE output (first 4 elements).
pub(super) fn dump_output(gpu: &dyn GpuBackend, output: DevicePtr, stream: u64) -> Result<()> {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let mut buf = vec![0u8; 8];
    gpu.copy_d2h(output, &mut buf)?;
    let vals: Vec<f32> = (0..4)
        .map(|i| {
            let lo = buf[i * 2];
            let hi = buf[i * 2 + 1];
            f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
        })
        .collect();
    tracing::info!("  MoE output: {:?}", vals);
    Ok(())
}
