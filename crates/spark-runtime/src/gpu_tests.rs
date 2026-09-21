// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for [`crate::gpu`], kept in a sibling file so `gpu.rs` stays
//! under the 500-LoC cap. `#[path]`-included, so `super::` still resolves to
//! the `gpu` module exactly as it did inline.
//!
//! ★ Deliberately absent: a test for `set_baseline_free_bytes` /
//! `baseline_free_bytes`. That pair reads and writes a PROCESS-GLOBAL cell
//! (`run_metrics::metrics().baseline_free_bytes`) which has no reset hook, so
//! any test that writes it makes every later reader in this binary
//! order-dependent — and a test that only reads it passes or fails based on
//! whether some other test ran first. That is a flaky test wearing a coverage
//! badge. Covering the `0 => None` contract honestly needs either a reset hook
//! on `RunMetrics` or a harness that owns the process; until one exists this
//! stays an acknowledged gap rather than a test that lies.

use super::mock::MockGpuBackend;
use super::*;

#[test]
fn test_mock_alloc_free() {
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(1024).unwrap();
    assert!(!ptr.is_null());
    assert_eq!(gpu.alloc_count(), 1);
    gpu.free(ptr).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn mock_free_rejects_interior_and_repeated_pointers() {
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(1024).unwrap();
    let interior = ptr.offset(256);
    assert!(gpu.free(interior).is_err());
    assert_eq!(gpu.alloc_count(), 1, "interior free must preserve owner");
    gpu.free(ptr).unwrap();
    assert!(gpu.free(ptr).is_err());
}

#[test]
fn test_mock_copy_roundtrip() {
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(8).unwrap();
    let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
    gpu.copy_h2d(&src, ptr).unwrap();
    let mut dst = [0u8; 8];
    gpu.copy_d2h(ptr, &mut dst).unwrap();
    assert_eq!(src, dst);
}

#[test]
fn test_device_ptr_offset() {
    let ptr = DevicePtr(0x1000);
    assert_eq!(ptr.offset(256).0, 0x1100);
}
