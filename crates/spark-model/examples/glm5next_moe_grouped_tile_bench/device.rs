// SPDX-License-Identifier: AGPL-3.0-only

//! Host/device upload-download plumbing and the kernel launch this bench calls once per
//! variant per iteration.
//!
//! Split out of `glm5next_moe_grouped_tile_bench.rs` to keep that file under the 500-LoC
//! cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::NUM_EXPERTS;

pub(crate) fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

pub(crate) fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

pub(crate) fn up_u64(g: &dyn GpuBackend, v: &[u64]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

pub(crate) fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

pub(crate) fn up_i32(g: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}

pub(crate) fn dn_i32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<i32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub(crate) fn dn_raw(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2: DevicePtr,
    c: DevicePtr,
    off: DevicePtr,
    stid: DevicePtr,
    n: usize,
    kk: usize,
    max_m_tiles: u32,
    n_tile: u32,
    threads: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([(n as u32).div_ceil(n_tile), max_m_tiles, NUM_EXPERTS as u32])
        .block([threads, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2)
        .arg_ptr(c)
        .arg_ptr(off)
        .arg_ptr(stid)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(0)
}
