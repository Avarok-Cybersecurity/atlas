// SPDX-License-Identifier: AGPL-3.0-only

//! Debug print helpers + env-gated GDN intermediate dumper for
//! per-layer numerical comparison vs an HF CPU oracle. See
//! DEBUGGING_METHODOLOGY.md §3-§5 + bench/longcode/hang-forensics/.
//!
//! Set `ATLAS_GDN_DUMP=<dir>` to enable, and optionally
//! `ATLAS_GDN_DUMP_LAYERS=0,15,29` (comma-list of SSM-layer indices) to
//! pick which layers to capture (default: `0`).
//!
//! Filenames: `gdnsub_step0_L{idx}_{stage}.bin` where stage ∈
//! {conv, l2, gdn, gnorm}. Headerless little-endian BF16, last-token
//! slice of the prefill. The Python comparator
//! (`gdn_chain_diff_a3b.py`) reads this format.

use std::sync::atomic::{AtomicBool, AtomicUsize};

use super::*;

/// Max SSM layers we can track. A3B has 30 SSM layers; 27B has 48.
/// Set well above either.
pub(super) const MAX_SSM_LAYERS: usize = 64;

/// Increments on each entry to `prefill_inner`. First prefill: 0..N-1
/// for the N SSM layers in model order. Subsequent prefills: N..2N-1
/// etc. — those never match user-selected layer indices in the env
/// list, so per-(layer, stage) latches below stay LIFO-clean.
pub(super) static SSM_LAYER_CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

// One latch per (layer_idx, stage). The const-fn AtomicBool::new
// constructor permits this initializer pattern in stable Rust.
macro_rules! atomic_bool_array {
    () => {{
        // `const ELEM` + `[ELEM; N]` is the canonical fixed-size
        // array-of-atomics idiom; clippy's interior-mutable-const lint
        // is a known false positive for it.
        #[allow(clippy::declare_interior_mutable_const)]
        const ELEM: AtomicBool = AtomicBool::new(false);
        [ELEM; MAX_SSM_LAYERS]
    }};
}
pub(super) static DUMP_CONV: [AtomicBool; MAX_SSM_LAYERS] = atomic_bool_array!();
pub(super) static DUMP_L2: [AtomicBool; MAX_SSM_LAYERS] = atomic_bool_array!();
pub(super) static DUMP_GDN: [AtomicBool; MAX_SSM_LAYERS] = atomic_bool_array!();
pub(super) static DUMP_GNORM: [AtomicBool; MAX_SSM_LAYERS] = atomic_bool_array!();

fn dump_layers_from_env() -> Vec<usize> {
    std::env::var("ATLAS_GDN_DUMP_LAYERS")
        .unwrap_or_else(|_| "0".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

/// Snapshot `n_elements` BF16 values starting at `ptr + byte_offset` to
/// `<ATLAS_GDN_DUMP>/gdnsub_step0_L{layer_idx}_{stage}.bin`. Skips
/// when `ATLAS_GDN_DUMP` is unset or `layer_idx` not in
/// `ATLAS_GDN_DUMP_LAYERS`. Latches per-(layer, stage) so each pair
/// dumps at most once across the process lifetime.
pub(super) fn maybe_dump_gdn_buf(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    byte_offset: usize,
    n_elements: usize,
    layer_idx: usize,
    stage: &str,
    _unused_latch: &[AtomicBool; MAX_SSM_LAYERS],
    stream: u64,
) -> Result<()> {
    // Dumps the LAST CHUNK's last token: under chunked prefill, this
    // function fires once per scheduler chunk per SSM layer. We do NOT
    // latch — we OVERWRITE the dump file on every call so the final
    // file on disk corresponds to the LAST chunk's last-token capture
    // (= position L-1 of the full prefill, not position chunk_len-1 of
    // the first chunk). The `layer_idx` modulo num-SSM-layers handles
    // the monotonic SSM_LAYER_CALL_COUNTER wrapping across multiple
    // scheduler chunks so each chunk re-dumps the same layers.
    let dir = match std::env::var("ATLAS_GDN_DUMP") {
        Ok(d) if !d.is_empty() => d,
        _ => return Ok(()),
    };
    // Resolve layer_idx % num_linear_attention_layers via env hint
    // (default 30 for A3B; 48 for dense 27B). Set via
    // ATLAS_GDN_DUMP_N_SSM=<count> if needed.
    let n_ssm: usize = std::env::var("ATLAS_GDN_DUMP_N_SSM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let effective_layer = layer_idx % n_ssm.max(1);
    if effective_layer >= MAX_SSM_LAYERS {
        return Ok(());
    }
    let layers = dump_layers_from_env();
    if !layers.contains(&effective_layer) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let bytes = n_elements * 2;
    let mut buf = vec![0u8; bytes];
    gpu.copy_d2h(ptr.offset(byte_offset), &mut buf)?;
    let path =
        std::path::Path::new(&dir).join(format!("gdnsub_step0_L{effective_layer}_{stage}.bin"));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(&path, &buf)?;
    tracing::info!(
        "ATLAS_GDN_DUMP: wrote {} ({} bf16 elements, raw_idx={layer_idx}, effective_layer={effective_layer})",
        path.display(),
        n_elements
    );
    Ok(())
}

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
