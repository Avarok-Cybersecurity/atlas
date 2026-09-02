// SPDX-License-Identifier: AGPL-3.0-only

//! Is this shard on a NETWORK filesystem?
//!
//! The loader's per-shard choice between `O_DIRECT` and buffered reads is a
//! TENSOR-COUNT heuristic (`direct_io_tensor_cap`): O_DIRECT wins on a local
//! NVMe below ~5k tensors/shard, buffered wins above it. That heuristic knows
//! nothing about where the bytes actually live, and on a network mount it can
//! pick the worst option available — O_DIRECT over NFS bypasses the page
//! cache and turns every tensor into a synchronous round trip, exactly the
//! access pattern `--fast-load-prefetch-shards` exists to avoid.
//!
//! It only worked out by accident on the EXL3 checkpoints: their shards carry
//! ~37k tensors each, so the count cap already forced the buffered path on
//! dgx-00, where `/tank` is an NFS mount of gx10's disk. A checkpoint with
//! few large shards (a plain NVFP4 export) on the same mount would take
//! O_DIRECT and pay for it.
//!
//! So ask the filesystem instead of inferring it. On Linux `statfs(2)`
//! reports a magic number per mount; the network ones are enumerated below.
//! Everywhere else — and on any error — the answer is [`None`] ("unknown"),
//! and the caller keeps its existing flag-driven behaviour. A probe that
//! cannot answer must never change what the loader does, and must never fail
//! a load: this is an optimisation hint, not a precondition.

use std::path::Path;

/// Filesystem magics that mean "the bytes are on another machine".
///
/// From `linux/magic.h` plus the out-of-tree ones (Lustre, GlusterFS) that
/// ship their own headers. Kept as a table rather than a match so the list
/// reads as data and a new entry costs one line.
#[cfg(target_os = "linux")]
const NETWORK_FS_MAGIC: &[(i64, &str)] = &[
    (0x6969, "nfs"),
    (0x517B, "smb"),
    (0xFF53_4D42u32 as i64, "cifs"),
    (0xFE53_4D42u32 as i64, "smb2"),
    (0x0BD0_0BD0, "lustre"),
    (0x0102_1997, "9p"),
    (0x00C3_6400, "ceph"),
    (0x0116_1970, "gfs/gfs2"),
    (0x1373_4C6E, "afs"),
    (0x5346_414F, "openafs"),
    (0x7461_7266, "fuse.sshfs-or-similar"),
];

/// The mount kind behind `path`, when it can be determined.
///
/// `Some(Some(name))` = a network filesystem, named for the log line.
/// `Some(None)` = a local filesystem. `None` = could not tell (non-Linux,
/// `statfs` failed, path unreadable) — callers must treat this as "keep
/// doing whatever you were doing".
#[cfg(target_os = "linux")]
pub(super) fn network_fs(path: &Path) -> Option<Option<&'static str>> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    // statfs() follows the path, so a shard file works as well as its
    // directory; the caller passes whichever it has.
    let cstr = CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `buf` is only read after a zero return, which is exactly when
    // the kernel has filled it; `cstr` is NUL-terminated and outlives the
    // call (statfs copies nothing out of it).
    let magic = unsafe {
        let mut buf: libc::statfs = std::mem::zeroed();
        if libc::statfs(cstr.as_ptr(), &mut buf) != 0 {
            return None;
        }
        // `f_type` is `__fsword_t`: i64 on 64-bit glibc (where the cast is a
        // no-op, hence the allow), i32 on 32-bit, and an unsigned word on
        // some musl versions — the cast is what keeps this compiling on all
        // of them.
        #[allow(clippy::unnecessary_cast)]
        let m = buf.f_type as i64;
        m
    };
    Some(
        NETWORK_FS_MAGIC
            .iter()
            .find(|(m, _)| *m == magic)
            .map(|(_, name)| *name),
    )
}

#[cfg(not(target_os = "linux"))]
pub(super) fn network_fs(_path: &Path) -> Option<Option<&'static str>> {
    None
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    /// The probe must answer for an ordinary local path, and must answer
    /// "not network" for it — otherwise every local load would silently lose
    /// O_DIRECT.
    #[test]
    fn a_local_path_probes_as_non_network() {
        let dir = std::env::temp_dir();
        // tmpfs (0x01021994) is local and not in the table; so is ext4/xfs.
        assert_eq!(network_fs(&dir), Some(None), "probing {}", dir.display());
    }

    /// A path that does not exist cannot be probed — the caller must get
    /// "unknown" and keep its flag-driven behaviour rather than a panic or a
    /// wrong answer.
    #[test]
    fn an_unprobeable_path_is_unknown_not_a_guess() {
        let missing = Path::new("/nonexistent-atlas-fs-probe/shard.safetensors");
        assert_eq!(network_fs(missing), None);
    }

    /// The table is the contract: NFS must be in it (it is the mount this
    /// exists for) and the magics must be distinct.
    #[test]
    fn the_magic_table_is_sane() {
        assert!(
            NETWORK_FS_MAGIC
                .iter()
                .any(|(m, n)| *m == 0x6969 && *n == "nfs")
        );
        let mut seen: Vec<i64> = NETWORK_FS_MAGIC.iter().map(|(m, _)| *m).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(before, seen.len(), "duplicate magic in NETWORK_FS_MAGIC");
    }
}
