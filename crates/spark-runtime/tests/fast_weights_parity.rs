// SPDX-License-Identifier: AGPL-3.0-only

//! Parity test: FastSafetensorsLoader must produce byte-identical weights
//! to the mmap-based SafetensorsLoader for the same file.
//!
//! Builds a tiny synthetic safetensors file in a tempdir, loads it with both
//! loaders against a MockGpuBackend, and asserts every tensor's bytes match.

#![cfg(unix)]

use spark_runtime::fast_weights::FastSafetensorsLoader;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};
use std::io::Write;

/// Build a minimal `model.safetensors` with two BF16 tensors and one U8 tensor.
/// Layout written by hand so the test doesn't depend on the safetensors crate
/// for encoding (decoding is still needed, used by the baseline loader).
fn write_test_safetensors(dir: &std::path::Path) -> std::path::PathBuf {
    // Tensor A: BF16, shape [4, 8] = 64 bytes.
    // Tensor B: BF16, shape [2, 2] = 8 bytes.
    // Tensor C: U8,   shape [16]   = 16 bytes.
    let a_bytes: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let b_bytes: Vec<u8> = (0..8).map(|i| (128 + i) as u8).collect();
    let c_bytes: Vec<u8> = (0..16).map(|i| (200 + i) as u8).collect();

    let header = serde_json::json!({
        "a": { "dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64] },
        "b": { "dtype": "BF16", "shape": [2, 2], "data_offsets": [64, 72] },
        "c": { "dtype": "U8",   "shape": [16],   "data_offsets": [72, 88] },
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();

    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(&header_bytes).unwrap();
    f.write_all(&a_bytes).unwrap();
    f.write_all(&b_bytes).unwrap();
    f.write_all(&c_bytes).unwrap();
    f.sync_all().unwrap();
    path
}

#[test]
fn fast_and_mmap_loaders_agree() {
    let tmp = tempdir_like();
    write_test_safetensors(&tmp);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new()
        .load(&tmp, &gpu_base, 0)
        .expect("baseline load");
    assert_eq!(base.len(), 3);

    let gpu_fast = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    // Force the buffered-read path: tmpfs rejects O_DIRECT on most kernels,
    // but we disable it explicitly so the test is deterministic.
    fast.try_direct_io = false;
    let new = fast.load(&tmp, &gpu_fast, 0).expect("fast load");
    assert_eq!(new.len(), 3);

    for name in ["a", "b", "c"] {
        let wb = base.get(name).unwrap();
        let wn = new.get(name).unwrap();
        assert_eq!(wb.shape, wn.shape, "shape mismatch for {name}");
        assert_eq!(wb.dtype, wn.dtype, "dtype mismatch for {name}");
        let bb = gpu_base.read_alloc(wb.ptr).unwrap();
        let bn = gpu_fast.read_alloc(wn.ptr).unwrap();
        assert_eq!(bb, bn, "byte mismatch for {name}");
    }

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn fast_loader_with_direct_io_if_supported() {
    // Best-effort O_DIRECT test — silently succeeds (by falling back to
    // buffered) if the filesystem rejects O_DIRECT.
    let tmp = tempdir_like();
    write_test_safetensors(&tmp);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new().load(&tmp, &gpu_base, 0).unwrap();

    let gpu_fast = MockGpuBackend::new();
    let fast = FastSafetensorsLoader::new(); // try_direct_io = true by default
    let new = fast
        .load(&tmp, &gpu_fast, 0)
        .expect("fast load with O_DIRECT attempted");
    assert_eq!(new.len(), 3);

    for name in ["a", "b", "c"] {
        let bb = gpu_base.read_alloc(base.get(name).unwrap().ptr).unwrap();
        let bn = gpu_fast.read_alloc(new.get(name).unwrap().ptr).unwrap();
        assert_eq!(bb, bn, "byte mismatch for {name} (O_DIRECT path)");
    }

    std::fs::remove_dir_all(&tmp).ok();
}

/// Creates a unique temp directory without pulling in the tempfile crate.
fn tempdir_like() -> std::path::PathBuf {
    let pid = std::process::id();
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("atlas-fwp-{pid}-{ns}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}
