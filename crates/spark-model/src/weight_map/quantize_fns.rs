// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Quantize a BF16 dense weight to FP8 E4M3 on GPU.
///
/// Allocates FP8 weight buffer + per-row scale buffer, runs GPU quantization
/// kernel. Called once at model load time (not on the hot path).
pub fn quantize_to_fp8(
    bf16_weight: &DenseWeight,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    quantize_kernel: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<Fp8DenseWeight> {
    use spark_runtime::kernel_args::KernelLaunch;

    // Allocate FP8 weight buffer: N * K bytes
    let fp8_buf = gpu.alloc(n * k)?;
    // Allocate per-row scale buffer: N * 4 bytes (f32)
    let scale_buf = gpu.alloc(n * 4)?;

    // Launch quantization kernel: Grid=(N), Block=(256)
    KernelLaunch::new(gpu, quantize_kernel)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(fp8_buf)
        .arg_ptr(scale_buf)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    // Synchronize to ensure quantization completes before using the buffers
    gpu.synchronize(stream)?;

    Ok(Fp8DenseWeight {
        weight: fp8_buf,
        row_scale: scale_buf,
    })
}

/// Load an FP8 E4M3 checkpoint weight with per-row f32 scales.
///
/// Expects two tensors in the store:
///   - `{name}.weight`: FP8E4M3 [N, K] (1 byte per element)
///   - `{name}.weight_scale`: f32 `[N]` per-row dequant scale
///
/// Both are already on GPU from safetensors mmap — no conversion needed.
/// Returns an [`Fp8Weight`] ready for the `w8a16_gemv` LUT kernel.
pub fn load_fp8_weight(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<Fp8Weight> {
    let w = store.get(&format!("{name}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {name}.weight, got {:?}",
        w.dtype,
    );
    ensure!(
        w.shape.len() == 2,
        "Expected 2D weight for {name}, got {:?}",
        w.shape
    );
    let n = w.shape[0];
    let k = w.shape[1];

    // FP8 weight bytes: already on GPU from WeightStore, 1 byte per element
    let weight_ptr = w.ptr;

    // Per-row scale: try `.weight_scale` (per-row f32 [N])
    let scale_key = format!("{name}.weight_scale");
    let s = store.get(&scale_key).with_context(|| {
        format!("Missing per-row scale tensor {scale_key} for FP8 weight {name}")
    })?;
    ensure!(
        s.shape.len() == 1 && s.shape[0] == n,
        "Expected [{n}] shape for {scale_key}, got {:?}",
        s.shape,
    );

    // Scale tensor may be BF16 or f32 on disk. If BF16, convert to f32 on CPU.
    let row_scale_ptr = if s.dtype == WeightDtype::FP32 {
        // Already f32 on GPU — use directly
        s.ptr
    } else if s.dtype == WeightDtype::BF16 {
        // BF16 → f32 conversion on CPU, upload to GPU
        let mut bf16_buf = vec![0u8; n * 2];
        gpu.copy_d2h(s.ptr, &mut bf16_buf)?;
        let mut f32_buf = vec![0u8; n * 4];
        for i in 0..n {
            let bf16_bytes = [bf16_buf[i * 2], bf16_buf[i * 2 + 1]];
            let val = bf16_bytes_to_f32(bf16_bytes);
            let f32_bytes = val.to_le_bytes();
            f32_buf[i * 4..i * 4 + 4].copy_from_slice(&f32_bytes);
        }
        let f32_ptr = gpu.alloc(n * 4)?;
        gpu.copy_h2d(&f32_buf, f32_ptr)?;
        f32_ptr
    } else {
        anyhow::bail!(
            "Unsupported dtype {:?} for {scale_key}, expected FP32 or BF16",
            s.dtype,
        );
    };

    Ok(Fp8Weight {
        weight: weight_ptr,
        row_scale: row_scale_ptr,
        n: n as u32,
        k: k as u32,
    })
}
