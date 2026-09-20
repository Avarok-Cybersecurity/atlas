// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! The file reads behind the expert paths: positional reads that any
//! number of threads may issue on one descriptor, the page-cache drop after
//! a buffered miss, and the `O_DIRECT` side (the descriptor, the segment
//! description, the read that tolerates a window past the end of the file).
//! Split from `expert_stream.rs` (500-LoC cap).

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Positional read of exactly `dst.len()` bytes at `offset`. No file position
/// is shared, so any number of threads may read one `File` at once.
pub fn pread(file: &File, offset: u64, dst: &mut [u8]) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(dst, offset)
            .with_context(|| format!("pread {} bytes at {offset}", dst.len()))
    }
    #[cfg(not(unix))]
    {
        let _ = (file, offset, dst);
        bail!("expert streaming needs positional reads (unix)")
    }
}

/// `pread`, then drop the range from the page cache: the bytes now live in
/// the arena and the cache copy is dead weight. With a 100 GiB page-locked
/// arena on a 121 GiB Spark, leaving 12 MiB of cache behind every miss made
/// the kernel reclaim into swap in the middle of a step (single steps of 2 to
/// 18 s on the 09-19 standard). `ATLAS_DS41_KEEP_PAGE_CACHE=1` keeps the old
/// behaviour.
pub fn pread_uncached(file: &File, offset: u64, dst: &mut [u8]) -> Result<()> {
    pread(file, offset, dst)?;
    #[cfg(target_os = "linux")]
    if !keep_page_cache() {
        use std::os::unix::io::AsRawFd;
        // SAFETY: an advisory call on an open descriptor; the kernel checks
        // the range.
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                dst.len() as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn keep_page_cache() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_DS41_KEEP_PAGE_CACHE").is_ok_and(|v| v == "1"))
}

/// Open `path` for direct reads (offsets, lengths and buffers aligned to
/// the 512 B logical block by the caller). `ATLAS_DS41_DIRECT_READS=0`
/// keeps every read on the page cache.
pub(super) fn open_direct(path: &Path) -> Option<File> {
    if std::env::var("ATLAS_DS41_DIRECT_READS").is_ok_and(|v| v == "0") {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        None
    }
}

/// One file range of an expert's slot image, for the direct miss path:
/// bytes `slot_off..slot_off + len` of the slot are `file_off..` of `shard`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectSeg {
    pub shard: usize,
    pub file_off: u64,
    pub slot_off: usize,
    pub len: usize,
}

/// `pread` into `dst` until at least `need` bytes have landed (`need <=
/// dst.len()`): the direct miss path reads 512 B windows whose tail may run
/// past the end of the file, where only the tensor's own bytes are owed.
pub fn pread_at_least(file: &File, offset: u64, dst: &mut [u8], need: usize) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        let mut done = 0usize;
        while done < need {
            let n = file
                .read_at(&mut dst[done..], offset + done as u64)
                .with_context(|| format!("pread {} bytes at {offset}", dst.len()))?;
            if n == 0 {
                bail!("pread at {offset}: end of file after {done} of {need} bytes",);
            }
            done += n;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (file, offset, dst, need);
        bail!("expert streaming needs positional reads (unix)")
    }
}
