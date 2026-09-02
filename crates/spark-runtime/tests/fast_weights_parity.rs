// SPDX-License-Identifier: AGPL-3.0-only

//! Parity test: FastSafetensorsLoader must produce byte-identical weights
//! to the mmap-based SafetensorsLoader for the same file.
//!
//! Builds a tiny synthetic safetensors file in a tempdir, loads it with both
//! loaders against a MockGpuBackend, and asserts every tensor's bytes match.

#![cfg(unix)]

use spark_runtime::fast_weights::FastSafetensorsLoader;
use spark_runtime::gpu::GpuBackend;
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

/// One EXL3 quartet (`p.trellis` I16 [1, 2, 64] = 256 B, `p.suh` F16 [16],
/// `p.svh` F16 [32], `p.mul1` I32 scalar) in file order, then a BF16
/// bystander of `tail_bytes`. Every byte is a distinct pattern so a slot
/// mix-up shows up as a byte mismatch.
fn write_exl3_safetensors(dir: &std::path::Path, tail_bytes: usize) -> std::path::PathBuf {
    let trellis: Vec<u8> = (0..256).map(|i| (i * 7 % 251) as u8).collect();
    let suh: Vec<u8> = (0..32).map(|i| (100 + i) as u8).collect();
    let svh: Vec<u8> = (0..64).map(|i| (150 + i) as u8).collect();
    let mul1 = 0x83DC_D12Du32.to_le_bytes().to_vec();
    let tail: Vec<u8> = (0..tail_bytes).map(|i| (i % 253) as u8).collect();
    let mut off = 0usize;
    let mut span = |len: usize| {
        let s = [off, off + len];
        off += len;
        s
    };
    let header = serde_json::json!({
        "p.trellis": { "dtype": "I16", "shape": [1, 2, 64], "data_offsets": span(256) },
        "p.suh":     { "dtype": "F16", "shape": [16], "data_offsets": span(32) },
        "p.svh":     { "dtype": "F16", "shape": [32], "data_offsets": span(64) },
        "p.mul1":    { "dtype": "I32", "shape": [],   "data_offsets": span(4) },
        "tail.weight": { "dtype": "BF16", "shape": [tail_bytes / 2], "data_offsets": span(tail_bytes) },
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(&header_bytes).unwrap();
    for part in [&trellis, &suh, &svh, &mul1, &tail] {
        f.write_all(part).unwrap();
    }
    f.sync_all().unwrap();
    path
}

fn read_tensor(gpu: &MockGpuBackend, t: &spark_runtime::weights::WeightTensor) -> Vec<u8> {
    let mut buf = vec![0u8; t.byte_size()];
    gpu.copy_d2h(t.ptr, &mut buf).unwrap();
    buf
}

/// With a pool predicate admitting `p`, the quartet lands in two arenas
/// (one alloc each) as `.offset()` views, byte-identical to the mmap
/// loader, `.suh/.svh` still F16; the bystander keeps its own allocation;
/// release frees everything exactly once.
#[test]
fn pooled_exl3_quartet_is_byte_identical_and_arena_owned() {
    use atlas_core::scope::ModelResource;
    let tmp = tempdir_like();
    write_exl3_safetensors(&tmp, 64);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new().load(&tmp, &gpu_base, 0).unwrap();

    let gpu_fast = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let mut seen: Vec<(String, Vec<usize>)> = Vec::new();
    let seen_cell = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sc = seen_cell.clone();
    fast.pool_predicate = Some(std::sync::Arc::new(move |prefix: &str, shape: &[usize]| {
        sc.lock()
            .unwrap()
            .push((prefix.to_string(), shape.to_vec()));
        prefix == "p"
    }));
    let mut new = fast.load(&tmp, &gpu_fast, 0).expect("fast load");
    seen.extend(seen_cell.lock().unwrap().drain(..));
    assert_eq!(seen, vec![("p".to_string(), vec![1, 2, 64])]);

    assert_eq!(new.len(), 5);
    assert_eq!(new.arena_count(), 2, "trellis + aux arenas");
    // trellis arena 256 B; aux arena = three 256-B slots (32 / 64 / 4 B).
    assert_eq!(new.pooled_bytes(), 256 + 3 * 256);
    // 2 arenas + 1 bystander — not 5 per-tensor allocations.
    assert_eq!(gpu_fast.alloc_count(), 3);
    for name in ["p.trellis", "p.suh", "p.svh", "p.mul1", "tail.weight"] {
        let wb = base.get(name).unwrap();
        let wn = new.get(name).unwrap();
        assert_eq!(wb.shape, wn.shape, "shape mismatch for {name}");
        assert_eq!(wb.dtype, wn.dtype, "dtype mismatch for {name}");
        assert_eq!(
            read_tensor(&gpu_base, wb),
            read_tensor(&gpu_fast, wn),
            "byte mismatch for {name}"
        );
        assert_eq!(
            new.is_pooled(wn.ptr),
            name != "tail.weight",
            "{name} pooled?"
        );
    }
    assert_eq!(
        new.get("p.suh").unwrap().dtype,
        spark_runtime::weights::WeightDtype::F16
    );
    // The aux views sit at 256-B slots of one arena, in file order.
    let suh = new.get("p.suh").unwrap().ptr;
    assert_eq!(new.get("p.svh").unwrap().ptr, suh.offset(256));
    assert_eq!(new.get("p.mul1").unwrap().ptr, suh.offset(512));

    new.release(&gpu_fast).expect("released");
    assert_eq!(
        gpu_fast.alloc_count(),
        0,
        "arenas freed once, members never"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

/// No predicate: the same file loads per tensor, exactly as before.
#[test]
fn without_a_predicate_nothing_is_pooled() {
    let tmp = tempdir_like();
    write_exl3_safetensors(&tmp, 64);
    let gpu = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let store = fast.load(&tmp, &gpu, 0).unwrap();
    assert_eq!(store.arena_count(), 0);
    assert_eq!(gpu.alloc_count(), 5);
    std::fs::remove_dir_all(&tmp).ok();
}

/// The arenas are allocated up front; if the shard then fails mid-copy (the
/// bystander's per-tensor alloc AND its managed fallback both fail), the
/// arenas are freed once on the way out — nothing waits for the backend's
/// teardown sweep.
#[test]
fn mid_shard_failure_frees_the_arenas_once() {
    let tmp = tempdir_like();
    write_exl3_safetensors(&tmp, 8192);
    let gpu = MockGpuBackend::new();
    gpu.set_max_allocation_bytes(4096);
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    fast.pool_predicate = Some(std::sync::Arc::new(|prefix: &str, _: &[usize]| {
        prefix == "p"
    }));
    let err = fast
        .load(&tmp, &gpu, 0)
        .err()
        .expect("the 8 KiB bystander cannot be allocated");
    assert!(format!("{err:#}").contains("exceeds mock limit"), "{err:#}");
    assert_eq!(
        gpu.alloc_count(),
        0,
        "both arenas rolled back, nothing else was allocated"
    );
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
