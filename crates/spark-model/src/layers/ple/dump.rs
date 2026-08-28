// SPDX-License-Identifier: AGPL-3.0-only

//! Highway taps for the qwen4_exp bisect.
//!
//! Every PIECE of this port is pinned to the reference — the n-gram ids are
//! bit-exact, the mHC kernels and the PLE gate/conv match to cosine 0.99999,
//! the NVMe gather is bit-exact — and the model still does not produce
//! coherent text. That combination says the fault is in the COMPOSITION, and
//! composition is exactly what per-kernel probes cannot see.
//!
//! So: dump the `hc_mult`-wide residual highway at named points and diff it
//! against the reference layer by layer.
//!
//! **The sub-layer boundary is what makes this affordable.** Tapping after a
//! block's `hc_post` but BEFORE the next `hc_pre` means the reference only has
//! to reproduce that block — for layer 0 that is the GDN projections alone,
//! with none of the 512-expert MoE. Only the taps that come after an MoE need
//! experts, and even then top-10 routing over a short prompt touches a few
//! dozen, not 512.
//!
//! Off unless `ATLAS_QWEN4EXP_DUMP` names a directory. Writes
//! `<dir>/L{layer:02}_{tag}.bin` as raw little-endian FP32, `[T, hc*H]`.

use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Directory from `ATLAS_QWEN4EXP_DUMP`, resolved once.
fn dump_dir() -> Option<&'static str> {
    static DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let d = std::env::var("ATLAS_QWEN4EXP_DUMP")
            .ok()
            .filter(|s| !s.is_empty());
        if let Some(ref path) = d {
            let _ = std::fs::create_dir_all(path);
            tracing::warn!(
                "ATLAS_QWEN4EXP_DUMP={path}: taping the mHC highway to disk. \
                 This SYNCHRONIZES and copies D2H at every tap — a debug aid, \
                 not a serving mode."
            );
        }
        d
    })
    .as_deref()
}

/// One-shot: refuse to overwrite a tap that already exists.
///
/// `SSM_LAYER_CALL_COUNTER` is a global that never resets, so the second
/// request of a run labels its taps L36+, the third L72+, and so on. Left
/// alone, that means a second request silently leaves the FIRST request's
/// L00 files in place while adding mislabelled ones — and a bisect then
/// compares a stale tap against a fresh reference and calls it a divergence.
///
/// So the first prefill after startup wins and everything later is ignored.
/// The intended use is exactly that: start the server, send one request,
/// read the taps.
fn claim(path: &str) -> bool {
    !std::path::Path::new(path).exists()
}

/// Tap the FP32 highway. No-op unless the dump directory is set.
///
/// Synchronizes before reading, so it must never run inside CUDA-graph
/// capture — which is already true of this model's path (`ATLAS_DEBUG_NO_GRAPH`).
pub fn tap_highway(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    layer: usize,
    tag: &str,
    num_tokens: usize,
    hc_dim: usize,
    stream: u64,
) {
    let Some(dir) = dump_dir() else {
        return;
    };
    let path = format!("{dir}/L{layer:02}_{tag}.bin");
    if !claim(&path) {
        return;
    }
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let mut raw = vec![0u8; num_tokens * hc_dim * 4];
    if gpu.copy_d2h(streams, &mut raw).is_err() {
        return;
    }
    let path = format!("{dir}/L{layer:02}_{tag}.bin");
    if let Err(e) = std::fs::write(&path, &raw) {
        tracing::warn!("highway tap {path}: {e}");
    }
}

/// Tap a BF16 buffer (the embedding, a block output) the same way.
pub fn tap_bf16(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    layer: usize,
    tag: &str,
    n_elements: usize,
    stream: u64,
) {
    let Some(dir) = dump_dir() else {
        return;
    };
    let path = format!("{dir}/L{layer:02}_{tag}.bf16.bin");
    if !claim(&path) {
        return;
    }
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let mut raw = vec![0u8; n_elements * 2];
    if gpu.copy_d2h(ptr, &mut raw).is_err() {
        return;
    }
    if let Err(e) = std::fs::write(&path, &raw) {
        tracing::warn!("highway tap {path}: {e}");
    }
}

/// Tap an FP32 buffer of `n_elements` (the injection vector, a gate).
pub fn tap_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    layer: usize,
    tag: &str,
    n_elements: usize,
    stream: u64,
) {
    let Some(dir) = dump_dir() else {
        return;
    };
    let path = format!("{dir}/L{layer:02}_{tag}.bin");
    if !claim(&path) {
        return;
    }
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let mut raw = vec![0u8; n_elements * 4];
    if gpu.copy_d2h(ptr, &mut raw).is_err() {
        return;
    }
    if let Err(e) = std::fs::write(&path, &raw) {
        tracing::warn!("highway tap {path}: {e}");
    }
}
