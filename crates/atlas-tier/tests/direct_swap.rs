// SPDX-License-Identifier: AGPL-3.0-only

//! Public-API integration tests for [`DirectSwapFile`] — real O_DIRECT I/O
//! under `target/atlas-tier-tests`, with an EINVAL-skip when the filesystem
//! (tmpfs/overlay) refuses O_DIRECT so containerized CI doesn't break.

use std::path::Path;

use atlas_tier::{DirectSwapFile, Residency, SwapStore, VecSlotArena};

/// A real-filesystem dir for O_DIRECT (tmpfs/overlay EINVALs on O_DIRECT —
/// tolerated as a skip so containerized CI doesn't break).
fn o_direct_file(record_bytes: usize, tag: &str) -> Option<(DirectSwapFile, std::path::PathBuf)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/atlas-tier-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("dsf-{tag}-{}.swap", std::process::id()));
    match DirectSwapFile::create(&path, record_bytes) {
        Ok(f) => Some((f, path)),
        Err(e) => {
            let unsupported = e.chain().any(|cause| {
                cause
                    .downcast_ref::<std::io::Error>()
                    .and_then(std::io::Error::raw_os_error)
                    .is_some_and(|code| code == libc::EINVAL || code == libc::EOPNOTSUPP)
            });
            // Opt-in enforcement: CI on a real disk sets this so a silent skip
            // can't green-light a run that never touched the O_DIRECT path.
            if std::env::var_os("ATLAS_TIER_REQUIRE_O_DIRECT").is_some() || !unsupported {
                panic!(
                    "DirectSwapFile setup failed instead of reporting unsupported O_DIRECT: {e:#}"
                );
            }
            eprintln!("skipping O_DIRECT test (filesystem refused O_DIRECT): {e:#}");
            None
        }
    }
}

/// A page-aligned (4 KiB) mutable sub-slice of `storage`, which must be
/// over-allocated by at least 4096 bytes past `len`. Lets a test deterministically
/// hit the O_DIRECT *aligned fast-path* (plain `Vec`s almost never are).
fn page_aligned(storage: &mut [u8], len: usize) -> &mut [u8] {
    let pad = (4096 - (storage.as_ptr() as usize & 0xfff)) & 0xfff;
    &mut storage[pad..pad + len]
}

/// A deterministically non-page-aligned sub-slice of an over-allocated buffer.
fn page_unaligned(storage: &mut [u8], len: usize) -> &mut [u8] {
    let offset = usize::from(storage.as_ptr() as usize & 0xfff == 0);
    &mut storage[offset..offset + len]
}

#[test]
fn direct_swap_file_rejects_bad_record_bytes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/atlas-tier-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dsf-bad.swap");
    assert!(
        DirectSwapFile::create(&path, 0).is_err(),
        "zero record_bytes rejected"
    );
    assert!(
        DirectSwapFile::create(&path, 1000).is_err(),
        "non-4KiB multiple rejected"
    );
}

/// O_DIRECT write/read round-trips through the page-aligned bounce with caller
/// buffers that are deterministically unaligned.
#[test]
fn direct_swap_file_roundtrips_unaligned_records() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "rt") else {
        return;
    };
    assert_eq!(f.record_bytes(), rb);
    let mut pat_storage = vec![0u8; rb + 1];
    let pat = page_unaligned(&mut pat_storage, rb);
    assert_ne!(pat.as_ptr() as usize & 0xfff, 0, "write uses bounce");
    for (i, b) in pat.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    f.write_record(3, pat).unwrap(); // sparse: slot 3 before slot 0
    f.write_record(0, &vec![0xEE; rb]).unwrap();
    let mut out_storage = vec![0u8; rb + 1];
    let out = page_unaligned(&mut out_storage, rb);
    assert_ne!(out.as_ptr() as usize & 0xfff, 0, "read uses bounce");
    f.read_record(3, out).unwrap();
    assert_eq!(out, pat, "record 3 byte-identical");
    f.read_record(0, out).unwrap();
    assert_eq!(out, vec![0xEE; rb], "record 0 byte-identical");
    // Size validation is a hard error, not a short IO.
    assert!(f.write_record(1, &pat[..100]).is_err());
    let mut short = vec![0u8; 100];
    assert!(f.read_record(0, &mut short).is_err());
    let _ = std::fs::remove_file(path);
}

/// End-to-end: the residency spills to a REAL O_DIRECT file and faults back
/// byte-identical (the exact peer configuration, minus RDMA).
#[test]
fn residency_over_o_direct_swap_byte_identical() {
    let rb = 4096usize;
    let Some((f, path)) = o_direct_file(rb, "resid") else {
        return;
    };
    let mut r = Residency::new(VecSlotArena::new(rb, 2), f).unwrap();
    for k in 0..8u64 {
        r.put_blob(k, &vec![k as u8; rb]).unwrap();
    }
    assert_eq!(r.total_keys(), 8);
    assert!(
        r.stats().spills_to_disk >= 6,
        "cold keys spilled to the O_DIRECT file"
    );
    let mut out = vec![0u8; rb];
    for k in 0..8u64 {
        assert!(r.get_blob(k, &mut out).unwrap(), "key {k}");
        assert_eq!(
            out,
            vec![k as u8; rb],
            "key {k} byte-identical through O_DIRECT"
        );
    }
    let _ = std::fs::remove_file(path);
}

/// Page-aligned caller buffers round-trip without alignment errors. This proves
/// the input partition, not which internal branch handled it.
#[test]
fn direct_swap_aligned_buffers_roundtrip() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "aligned") else {
        return;
    };
    let mut wstore = vec![0u8; rb + 4096];
    let w = page_aligned(&mut wstore, rb);
    assert_eq!(
        w.as_ptr() as usize & 0xfff,
        0,
        "write buffer is page-aligned → fast-path"
    );
    for (i, b) in w.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    f.write_record(7, w).unwrap(); // aligned src → direct pwrite

    let mut rstore = vec![0u8; rb + 4096];
    let rbuf = page_aligned(&mut rstore, rb);
    assert_eq!(
        rbuf.as_ptr() as usize & 0xfff,
        0,
        "read buffer is page-aligned → fast-path"
    );
    f.read_record(7, rbuf).unwrap(); // aligned dst → direct pread
    for (i, b) in rbuf.iter().enumerate() {
        assert_eq!(*b, (i % 251) as u8, "aligned fast-path byte {i}");
    }
    let _ = std::fs::remove_file(path);
}

/// The other half of "the swap file is 0 bytes": once a record IS written, the
/// file grows immediately. `write_record` is a bare `pwrite`, so it extends
/// `i_size` on the first byte at offset 0 — meaning a 0-byte file proves zero
/// write calls, never a silently-dropped write. Skips where O_DIRECT is refused
/// (tmpfs), like the other tests in this file.
#[test]
fn direct_swap_file_grows_on_first_write() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "grow") else {
        return;
    };
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        0,
        "freshly created (truncated) swap file starts empty"
    );
    f.write_record(0, &vec![0x7C; rb]).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        rb as u64,
        "one record written ⇒ file is exactly one record long"
    );
    // Sparse addressing: a high slot sets i_size to (slot + 1) * record_bytes.
    f.write_record(4, &vec![0x7D; rb]).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 5 * rb as u64);
    let _ = std::fs::remove_file(path);
}
