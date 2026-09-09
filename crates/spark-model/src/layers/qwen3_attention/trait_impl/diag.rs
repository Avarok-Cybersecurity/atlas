// SPDX-License-Identifier: AGPL-3.0-only

//! Debug read-back helpers for the attention layer (split out of `trait_impl.rs` under the 500-line cap).

use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Debug: read back BF16 GPU tensor and compute L2 norm + first 4 values.
pub(crate) fn diag_norm(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    let _ = gpu.synchronize(stream);
    let mut buf = vec![0u16; n_elements];
    // SAFETY: `buf` is `vec![0u16; n_elements]` on the line above, so
    // `buf.len() == n_elements` and `n_elements * 2 == buf.len() *
    // size_of::<u16>()` — the span is exactly the Vec's buffer, all of it
    // zero-initialised. `bytes` is the sole reference derived from `buf` while it
    // is live: it is dead after the `copy_d2h` below, before `buf.iter()` runs.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 2) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let vals: Vec<f32> = buf
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if vals.len() >= 4 {
        format!(
            "[{:.4},{:.4},{:.4},{:.4}]",
            vals[0], vals[1], vals[2], vals[3]
        )
    } else {
        format!("{:?}", &vals[..vals.len().min(4)])
    };
    tracing::info!("DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements}");
}

/// Debug: read back FP32 GPU tensor and compute L2 norm + first 4 values.
/// Used by the DeepSeek-V4 multi-seq decode diagnostic path (post/comb-attn
/// holographic tensors are FP32). V4-only — no non-V4 caller.
pub fn diag_norm_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    let _ = gpu.synchronize(stream);
    let mut buf = vec![0f32; n_elements];
    // SAFETY: `buf` is `vec![0f32; n_elements]` on the line above, so
    // `buf.len() == n_elements` and `n_elements * 4 == buf.len() *
    // size_of::<f32>()` — the span is exactly the Vec's buffer, all of it
    // zero-initialised. `bytes` is the sole reference derived from `buf` while it
    // is live: it is dead after the `copy_d2h` below, before `buf.iter()` runs.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 4) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let norm: f32 = buf.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = buf.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if buf.len() >= 4 {
        format!("[{:.4},{:.4},{:.4},{:.4}]", buf[0], buf[1], buf[2], buf[3])
    } else {
        format!("{:?}", &buf[..buf.len().min(4)])
    };
    tracing::info!(
        "DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements} (FP32)"
    );
}

// The `OnceLock<bool>` static that lived here is now
// `layers::ops::ModelLevers::gemma4_diag`, resolved when the model is built
// and carried on `ForwardContext`.
