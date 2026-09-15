// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for `kernels/gb10/common/exl3_reconstruct.cu` — split from
//! `weights/exl3.rs` (500-LoC cap).
//!
//! Two layers:
//!
//!  * [`reconstruct_had_f16_into`] / [`transpose_f16_to_bf16_into`]: ONE
//!    launch each into caller-owned buffers on a caller-chosen stream — no
//!    allocation, no synchronization. This is the form the native dense
//!    PREFILL tier (`spark-model` `ops/exl3_dense/reconstruct.rs`) runs on
//!    the hot path, once per weight per call, into stage-owned scratch.
//!  * [`reconstruct_had_f16_device`] / [`reconstruct_had_bf16`]: the
//!    LOAD-TIME forms — allocate the result (caller owns it) on the default
//!    stream; `reconstruct_had_bf16` synchronizes before freeing its f16
//!    temporary. Both are thin wrappers over the `_into` forms, so the two
//!    layers cannot disagree on the launch geometry.
//!
//! Launch contract (from the kernel header): `exl3_reconstruct_had_k{K}_cb{CB}`
//! grid `(out/128, in/128, 1)`, block 256, args `(f16 unpacked [in, out],
//! packed trellis, suh, svh, packed_blocks_n = out/16, packed_n_offset = 0)`;
//! `exl3_f16_to_bf16_t` grid `(ceil(out/32), ceil(in/32))`, block `(32, 8)`,
//! args `(src f16 [in, out], dst bf16 [out, in], in, out)`.

use anyhow::{Result, bail, ensure};

use super::{Exl3Codebook, MODULE};
use crate::gpu::{DevicePtr, GpuBackend};
use crate::kernel_args::KernelLaunch;

/// Reconstruct one EXL3 tensor into `dst_f16` — f16 `[in, out]` row-major,
/// `in * out` elements, caller-owned — on `stream`. Stream-ordered, no
/// allocation, no host sync. `cb_index` is the KERNEL codebook index
/// (0 = 3INST, 1 = MCG, 2 = MUL1 — [`Exl3Codebook`]'s discriminant).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_had_f16_into(
    gpu: &dyn GpuBackend,
    trellis: DevicePtr,
    suh: DevicePtr,
    svh: DevicePtr,
    in_dim: usize,
    out_dim: usize,
    k_bits: u32,
    cb_index: u32,
    dst_f16: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        in_dim >= 128
            && in_dim.is_multiple_of(128)
            && out_dim >= 128
            && out_dim.is_multiple_of(128),
        "EXL3 reconstruct needs both dims divisible by 128, got [{in_dim}, {out_dim}]"
    );
    ensure!(
        (1..=8).contains(&k_bits),
        "EXL3 K must be 1..=8, got {k_bits}"
    );
    ensure!(
        cb_index <= 2,
        "EXL3 codebook index must be 0..=2, got {cb_index}"
    );
    ensure!(
        !trellis.is_null() && !suh.is_null() && !svh.is_null() && !dst_f16.is_null(),
        "EXL3 reconstruct: null trellis/suh/svh/destination pointer"
    );
    let name = format!("exl3_reconstruct_had_k{k_bits}_cb{cb_index}");
    let kernel = match gpu.kernel(MODULE, &name) {
        Ok(k) => k,
        Err(e) => bail!("EXL3 kernel {name} unavailable on this target: {e}"),
    };
    KernelLaunch::new(gpu, kernel)
        .grid([(out_dim / 128) as u32, (in_dim / 128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(dst_f16)
        .arg_ptr(trellis)
        .arg_ptr(suh)
        .arg_ptr(svh)
        .arg_u32((out_dim / 16) as u32) // packed_blocks_n
        .arg_u32(0) // packed_n_offset
        .launch(stream)
}

/// `dst_bf16 [out, in] = transpose(src_f16 [in, out])` with ONE f32-exact
/// f16 -> bf16 rounding per element (Atlas's `[N, K]` weight layout) on
/// `stream`. Stream-ordered, no allocation, no host sync; never in place.
pub fn transpose_f16_to_bf16_into(
    gpu: &dyn GpuBackend,
    src_f16: DevicePtr,
    dst_bf16: DevicePtr,
    in_dim: usize,
    out_dim: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        in_dim >= 1 && out_dim >= 1,
        "EXL3 transpose: empty [{in_dim}, {out_dim}]"
    );
    ensure!(
        !src_f16.is_null() && !dst_bf16.is_null() && src_f16.0 != dst_bf16.0,
        "EXL3 transpose: null or aliased source/destination"
    );
    let transpose = gpu.kernel(MODULE, "exl3_f16_to_bf16_t")?;
    KernelLaunch::new(gpu, transpose)
        .grid([
            (out_dim.div_ceil(32)) as u32,
            (in_dim.div_ceil(32)) as u32,
            1,
        ])
        .block([32, 8, 1])
        .arg_ptr(src_f16)
        .arg_ptr(dst_bf16)
        .arg_u32(in_dim as u32)
        .arg_u32(out_dim as u32)
        .launch(stream)
}

/// Reconstruct an EXL3 tensor to the upstream-native f16 `[in, out]`
/// row-major layout on the GPU (the reconstruct kernel's coalesced store
/// order). Returns a fresh f16 buffer of `in * out` elements (caller owns).
///
/// * `trellis` — device ptr to the packed `.trellis` int16 data
///   (`(in/16) * (out/16) * 16 * k_bits` u16s, uploaded by the caller).
/// * `suh` / `svh` — device ptrs to the f16 sign vectors (`in` / `out` f16s).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_had_f16_device(
    gpu: &dyn GpuBackend,
    trellis: DevicePtr,
    suh: DevicePtr,
    svh: DevicePtr,
    in_dim: usize,
    out_dim: usize,
    k_bits: u32,
    cb: Exl3Codebook,
) -> Result<DevicePtr> {
    ensure!(
        in_dim.is_multiple_of(128) && out_dim.is_multiple_of(128),
        "EXL3 reconstruct needs both dims divisible by 128, got [{in_dim}, {out_dim}]"
    );
    let stream = gpu.default_stream();
    let f16_out = gpu.alloc(in_dim * out_dim * 2)?;
    if let Err(e) = reconstruct_had_f16_into(
        gpu, trellis, suh, svh, in_dim, out_dim, k_bits, cb as u32, f16_out, stream,
    ) {
        gpu.free(f16_out).ok();
        return Err(e);
    }
    Ok(f16_out)
}

/// Reconstruct an EXL3 tensor to Atlas-layout BF16 `[out, in]` on the GPU.
/// Returns a fresh BF16 buffer of `out * in` elements (caller owns it).
///
/// Reconstructs to the f16 `[in, out]` layout first, then transposes to
/// Atlas's `[out, in]` row-major with a single f32-exact f16->bf16 rounding.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_had_bf16(
    gpu: &dyn GpuBackend,
    trellis: DevicePtr,
    suh: DevicePtr,
    svh: DevicePtr,
    in_dim: usize,
    out_dim: usize,
    k_bits: u32,
    cb: Exl3Codebook,
) -> Result<DevicePtr> {
    let f16_tmp = reconstruct_had_f16_device(gpu, trellis, suh, svh, in_dim, out_dim, k_bits, cb)?;
    let stream = gpu.default_stream();
    let out = match gpu.alloc(out_dim * in_dim * 2) {
        Ok(p) => p,
        Err(e) => {
            gpu.free(f16_tmp).ok();
            return Err(e);
        }
    };
    let launch = transpose_f16_to_bf16_into(gpu, f16_tmp, out, in_dim, out_dim, stream);
    gpu.synchronize(stream).ok();
    gpu.free(f16_tmp).ok();
    if let Err(e) = launch {
        gpu.free(out).ok();
        return Err(e);
    }
    Ok(out)
}
