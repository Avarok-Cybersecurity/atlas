// SPDX-License-Identifier: AGPL-3.0-only
//! Host helpers for the `gdn_woa_oracle` example: env parsing, the LCG, device
//! upload and readback, pointer tables and the ULP diff.
//!
//! provenance-id: 526f6e616c6420522e205374657369616b

use anyhow::Result;
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::SLAB_ENTRIES;

pub(super) fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

pub(super) fn env_list(k: &str, d: &[usize]) -> Vec<usize> {
    match std::env::var(k) {
        Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        Err(_) => d.to_vec(),
    }
}

/// Deterministic LCG in [-1, 1).
pub(super) struct Rng(pub(super) u64);
impl Rng {
    pub(super) fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

pub(super) fn upload_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(&bytes, p)?;
    Ok(p)
}
pub(super) fn upload_bf16(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(&bytes, p)?;
    Ok(p)
}
pub(super) fn read_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub(super) fn read_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
pub(super) fn ptr_table(g: &dyn GpuBackend, ptrs: &[DevicePtr]) -> Result<DevicePtr> {
    let mut bytes = vec![0u8; SLAB_ENTRIES * 8];
    for (i, p) in ptrs.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&p.0.to_le_bytes());
    }
    let t = g.alloc(bytes.len())?;
    g.copy_h2d(&bytes, t)?;
    Ok(t)
}

/// (mismatching elements, max ULP distance) between two f32 buffers.
pub(super) fn diff_f32(a: &[f32], b: &[f32]) -> (usize, u32) {
    let mut n = 0usize;
    let mut ulp = 0u32;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            let d = (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs();
            ulp = ulp.max(d.min(u32::MAX as u64) as u32);
        }
    }
    (n, ulp)
}
