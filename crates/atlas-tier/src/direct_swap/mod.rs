// SPDX-License-Identifier: AGPL-3.0-only

//! [`DirectSwapFile`] — the NVMe cold tier, one implementation per platform.
//!
//! The contract ([`crate::traits::SwapStore`]) and its invariants are identical
//! everywhere: fixed-stride records addressed by `disk_slot`, `record_bytes` a
//! non-zero 4 KiB multiple, the file growing sparsely as slots are allocated.
//! Only the I/O primitive differs, so only the I/O primitive is split:
//!
//!   * `unix` — `O_DIRECT` (Linux) + `pread`/`pwrite` on a raw fd, staging
//!     through a page-aligned bounce when the caller's buffer is not aligned.
//!     On non-Linux unix `O_DIRECT` does not exist and the file opens buffered,
//!     which is harmless: the NVMe cold tier only ever runs on the Linux fleet.
//!   * `windows` — buffered `seek_read`/`seek_write`. Windows' nearest
//!     equivalent, `FILE_FLAG_NO_BUFFERING`, imposes sector-alignment rules on
//!     every buffer and offset and buys nothing here for the same reason: this
//!     tier is not driven on Windows. Correctness over an unused fast path.
//!
//! Splitting by file rather than by `#[cfg]` attribute keeps each
//! implementation readable on its own and stops a change to one platform from
//! silently editing the other.

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::DirectSwapFile;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::DirectSwapFile;

/// Shared by both implementations so the error text of a misuse is identical
/// on every platform.
#[allow(dead_code)]
pub(crate) fn validate_record_bytes(record_bytes: usize) -> anyhow::Result<()> {
    if record_bytes == 0 || !record_bytes.is_multiple_of(4096) {
        anyhow::bail!(
            "DirectSwapFile: record_bytes ({record_bytes}) must be a non-zero 4 KiB multiple"
        );
    }
    Ok(())
}
