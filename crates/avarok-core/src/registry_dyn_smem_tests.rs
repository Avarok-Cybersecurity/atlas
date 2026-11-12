// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free rules of the launch-size floors. `discover` needs a loaded module
//! and is exercised on the device; everything it feeds is checked here with
//! fabricated function handles, which are only ever used as map keys.

use super::{DYN_SMEM_SUFFIX, DynSmemFloors, floor_symbol, raised};
use crate::registry::RawCudaFunc;
use cudarc::driver::LaunchConfig;

fn cfg(shared_mem_bytes: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (8, 4, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes,
    }
}

fn func(addr: usize) -> RawCudaFunc {
    RawCudaFunc(addr as *mut std::ffi::c_void)
}

/// ORACLE: `AVAROK_DYN_SMEM_SYMBOL` in kernels/ada/common/prefill_dyn_smem.cuh
/// pastes `_dyn_smem_bytes` onto the expanded entry name; the PTX of
/// kernels/ada/common/inferspark_prefill.cu carries
/// `inferspark_prefill_64_dyn_smem_bytes = 90112`.
#[test]
fn the_floor_symbol_is_the_entry_name_plus_the_shared_suffix() {
    assert_eq!(DYN_SMEM_SUFFIX, "_dyn_smem_bytes");
    assert_eq!(
        floor_symbol("inferspark_prefill_64"),
        "inferspark_prefill_64_dyn_smem_bytes"
    );
    assert_eq!(
        floor_symbol("inferspark_prefill_paged_turbo3k_turbo8v"),
        "inferspark_prefill_paged_turbo3k_turbo8v_dyn_smem_bytes"
    );
}

/// A function with no floor launches with exactly what the caller passed:
/// every kernel outside the dynamic-tile family, on every target.
#[test]
fn a_function_without_a_floor_is_untouched() {
    let floors = DynSmemFloors::default();
    for bytes in [0, 4096, 70_400] {
        let out = floors.apply(func(0x1000), cfg(bytes));
        assert_eq!(out.shared_mem_bytes, bytes);
        assert_eq!(out.grid_dim, (8, 4, 1));
        assert_eq!(out.block_dim, (256, 1, 1));
    }
}

/// The dispatch sites of the prefill family pass 0 dynamic bytes; a recorded
/// floor is what they launch with, and only for that function.
#[test]
fn a_recorded_floor_raises_only_its_own_function() {
    let floors = DynSmemFloors::default();
    floors.record(func(0x2000), 90_112);
    assert_eq!(floors.apply(func(0x2000), cfg(0)).shared_mem_bytes, 90_112);
    assert_eq!(floors.apply(func(0x3000), cfg(0)).shared_mem_bytes, 0);
    assert_eq!(floors.floor(func(0x2000)), Some(90_112));
    assert_eq!(floors.floor(func(0x3000)), None);
}

/// A floor never LOWERS a caller's size: a site that already asks for more
/// (tiles plus scratch) keeps it.
#[test]
fn a_floor_never_lowers_a_larger_request() {
    assert_eq!(raised(cfg(100_000), Some(90_112)).shared_mem_bytes, 100_000);
    assert_eq!(raised(cfg(70_400), Some(70_400)).shared_mem_bytes, 70_400);
    assert_eq!(raised(cfg(1024), Some(70_400)).shared_mem_bytes, 70_400);
    assert_eq!(raised(cfg(1024), None).shared_mem_bytes, 1024);
}

/// Re-recording a function (a second resolution of the same entry) replaces
/// its floor rather than accumulating.
#[test]
fn re_recording_replaces_the_floor() {
    let floors = DynSmemFloors::default();
    floors.record(func(0x4000), 70_400);
    floors.record(func(0x4000), 90_112);
    assert_eq!(floors.floor(func(0x4000)), Some(90_112));
}
