// SPDX-License-Identifier: AGPL-3.0-only

//! Peer-specific MmapSlotArena test (moved from the pre-split snapshot_swap.rs).

use super::*;

/// `MmapSlotArena` over a real page-aligned heap buffer round-trips bytes.
#[test]
fn mmap_slot_arena_roundtrips() {
    let slot_bytes = 4096usize;
    let n = 3usize;
    // Page-aligned heap buffer (AlignedBuf moved to atlas-tier as a private
    // helper — allocate directly here).
    let mut p: *mut libc::c_void = std::ptr::null_mut();
    let rc = unsafe { libc::posix_memalign(&mut p, 4096, slot_bytes * n) };
    assert!(rc == 0 && !p.is_null(), "posix_memalign failed rc={rc}");
    {
        let mut arena = unsafe { MmapSlotArena::new(p as *mut u8, slot_bytes, n) };
        let pat = vec![0x3C_u8; slot_bytes];
        arena.write_slot(1, &pat).unwrap();
        let mut out = vec![0u8; slot_bytes];
        arena.read_slot(1, &mut out).unwrap();
        assert_eq!(out, pat);
        assert!(
            arena.write_slot(3, &pat).is_err(),
            "slot out of range rejected"
        );
    } // arena dropped before the backing buffer is freed
    unsafe { libc::free(p) };
}
