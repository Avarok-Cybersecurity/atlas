// SPDX-License-Identifier: AGPL-3.0-only

//! Per-operation drift-dump helper for full-attention prefill.
//!
//! Companion to the existing `ATLAS_GDN_DUMP` infrastructure in
//! `super::super::qwen3_ssm::debug`. This one captures named operation
//! outputs at user-selected absolute layer indices for the master drift
//! table study (`bench/fp8_dgx2_drift/`).
//!
//! Activation:
//!     ATLAS_OP_DUMP=<dir>                       (required to enable; no-op when unset)
//!     ATLAS_OP_DUMP_LAYERS=0,7,11,15,19,23,...  (csv of absolute layer
//!                                                indices; default = all)
//!     ATLAS_OP_DUMP_OPS=q_proj,k_proj,...       (csv of op names; default = all)
//!
//! Filename convention:
//!     <ATLAS_OP_DUMP>/atlas_op_L{abs_layer}_{op}.bin
//! Format: headerless little-endian f32, last-token slice of `n_elements`.
//! BF16 source tensors are widened to f32 on the host before write so
//! the Python comparator can load them with a single np.fromfile call.
//!
//! Overwrites on every call so the LAST chunk's last-token capture
//! survives in chunked-prefill scenarios.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

fn parse_csv(env: &str) -> Vec<String> {
    std::env::var(env)
        .ok()
        .map(|s| {
            s.split(',')
                .filter(|p| !p.trim().is_empty())
                .map(|p| p.trim().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn parse_csv_usize(env: &str) -> Vec<usize> {
    std::env::var(env)
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Returns Some(dir) if ATLAS_OP_DUMP is set, the layer index is allowed,
/// and the op name is allowed. Returns None otherwise (no-op fast path).
fn op_dump_dir(layer_idx: usize, op: &str) -> Option<String> {
    let dir = std::env::var("ATLAS_OP_DUMP").ok()?;
    if dir.is_empty() {
        return None;
    }
    let layers = parse_csv_usize("ATLAS_OP_DUMP_LAYERS");
    if !layers.is_empty() && !layers.contains(&layer_idx) {
        return None;
    }
    let ops = parse_csv("ATLAS_OP_DUMP_OPS");
    if !ops.is_empty() && !ops.iter().any(|o| o == op) {
        return None;
    }
    Some(dir)
}

/// Snapshot `n_elements` BF16 values starting at `ptr + byte_offset`,
/// widen to f32 little-endian, write to
/// `<ATLAS_OP_DUMP>/atlas_op_L{layer_idx}_{op}.bin`.
pub(crate) fn dump_bf16(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    byte_offset: usize,
    n_elements: usize,
    layer_idx: usize,
    op: &str,
    stream: u64,
) -> Result<()> {
    let Some(dir) = op_dump_dir(layer_idx, op) else {
        return Ok(());
    };
    gpu.synchronize(stream)?;
    let mut buf = vec![0u16; n_elements];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 2) };
    gpu.copy_d2h(ptr.offset(byte_offset), bytes)?;
    let vals: Vec<f32> = buf
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let bytes_f32: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = std::path::Path::new(&dir).join(format!("atlas_op_L{layer_idx}_{op}.bin"));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(&path, &bytes_f32)?;
    tracing::info!(
        "ATLAS_OP_DUMP: wrote {} ({n_elements} f32, bf16-source)",
        path.display()
    );
    Ok(())
}

/// Same as `dump_bf16` but for tensors already stored as f32 on device
/// (e.g. fp32-residual builds).
#[allow(dead_code)]
pub(crate) fn dump_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    byte_offset: usize,
    n_elements: usize,
    layer_idx: usize,
    op: &str,
    stream: u64,
) -> Result<()> {
    let Some(dir) = op_dump_dir(layer_idx, op) else {
        return Ok(());
    };
    gpu.synchronize(stream)?;
    let mut buf = vec![0f32; n_elements];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 4) };
    gpu.copy_d2h(ptr.offset(byte_offset), bytes)?;
    let bytes_f32: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = std::path::Path::new(&dir).join(format!("atlas_op_L{layer_idx}_{op}.bin"));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(&path, &bytes_f32)?;
    tracing::info!(
        "ATLAS_OP_DUMP: wrote {} ({n_elements} f32 native)",
        path.display()
    );
    Ok(())
}
