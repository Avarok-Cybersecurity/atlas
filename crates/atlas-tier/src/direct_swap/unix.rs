// SPDX-License-Identifier: AGPL-3.0-only

//! Unix [`DirectSwapFile`]: the real `O_DIRECT` NVMe cold tier, plus its
//! page-aligned bounce-buffer plumbing. See `mod.rs` for the platform split.

use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use anyhow::{Result, bail};

use crate::direct_swap::validate_record_bytes;
use crate::traits::SwapStore;

/// `O_DIRECT` is a Linux-only open flag (macOS has no equivalent — `F_NOCACHE`
/// is an `fcntl`, not an open flag). The type must still compile on every unix
/// (the workspace has a macOS/metal CI job); off Linux it simply opens buffered.
/// That is harmless: the NVMe cold tier only ever runs on the Linux fleet.
#[cfg(target_os = "linux")]
const DIRECT_FLAGS: i32 = libc::O_DIRECT;
#[cfg(not(target_os = "linux"))]
const DIRECT_FLAGS: i32 = 0;

/// O_DIRECT fixed-stride swap file on NVMe (the peer's cold tier). `record_bytes`
/// MUST be a 4 KiB multiple (O_DIRECT) — the SSM snapshot blob (66,846,720 B =
/// 16,320 × 4 KiB) already is. Records are addressed by `disk_slot` at
/// `disk_slot * record_bytes`; the file grows sparsely as slots are allocated.
///
/// Buffers passed to read/write must be page-aligned for O_DIRECT. The peer's
/// callers pass the mmap'd arena scratch (page-aligned); the residency scratch
/// is a plain Vec — see `read/write_record` which stage through an aligned
/// bounce only when the caller's buffer isn't aligned.
pub struct DirectSwapFile {
    fd: OwnedFd,
    record_bytes: usize,
    /// Page-aligned bounce for callers whose buffer isn't O_DIRECT-aligned.
    bounce: AlignedBuf,
}

impl DirectSwapFile {
    pub fn create(path: &Path, record_bytes: usize) -> Result<Self> {
        validate_record_bytes(record_bytes)?;
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(DIRECT_FLAGS)
            .open(path)
            .map_err(|e| anyhow::anyhow!("open O_DIRECT {}: {e}", path.display()))?;
        Ok(Self {
            fd: OwnedFd::from(f),
            record_bytes,
            bounce: AlignedBuf::new(record_bytes),
        })
    }

    fn offset(&self, disk_slot: usize) -> libc::off_t {
        (disk_slot as u64 * self.record_bytes as u64) as libc::off_t
    }
}

impl SwapStore for DirectSwapFile {
    fn record_bytes(&self) -> usize {
        self.record_bytes
    }

    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.record_bytes {
            bail!(
                "write_record: {} bytes, expected {}",
                bytes.len(),
                self.record_bytes
            );
        }
        let off = self.offset(disk_slot);
        let src = if is_aligned(bytes.as_ptr()) {
            bytes.as_ptr()
        } else {
            self.bounce.as_mut_slice().copy_from_slice(bytes);
            self.bounce.ptr()
        };
        let n = unsafe {
            libc::pwrite(
                self.fd.as_raw_fd(),
                src as *const libc::c_void,
                self.record_bytes,
                off,
            )
        };
        if n != self.record_bytes as isize {
            bail!("pwrite record {disk_slot} returned {n}, errno {}", errno());
        }
        Ok(())
    }

    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        if out.len() != self.record_bytes {
            bail!(
                "read_record: {} bytes, expected {}",
                out.len(),
                self.record_bytes
            );
        }
        let off = self.offset(disk_slot);
        if is_aligned(out.as_ptr()) {
            let n = unsafe {
                libc::pread(
                    self.fd.as_raw_fd(),
                    out.as_mut_ptr() as *mut libc::c_void,
                    self.record_bytes,
                    off,
                )
            };
            if n != self.record_bytes as isize {
                bail!("pread record {disk_slot} returned {n}, errno {}", errno());
            }
        } else {
            // Stage through the aligned bounce, then copy out. `&self` — the
            // bounce is interior; take a raw ptr (single-threaded peer loop).
            let bp = self.bounce.ptr();
            let n = unsafe {
                libc::pread(
                    self.fd.as_raw_fd(),
                    bp as *mut libc::c_void,
                    self.record_bytes,
                    off,
                )
            };
            if n != self.record_bytes as isize {
                bail!(
                    "pread(bounce) record {disk_slot} returned {n}, errno {}",
                    errno()
                );
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bp, out.as_mut_ptr(), self.record_bytes);
            }
        }
        Ok(())
    }
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

fn is_aligned(p: *const u8) -> bool {
    (p as usize) & 0xfff == 0
}

/// A page-aligned heap buffer (posix_memalign) for O_DIRECT staging.
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for AlignedBuf {}
impl AlignedBuf {
    fn new(len: usize) -> Self {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut p, 4096, len) };
        assert!(
            rc == 0 && !p.is_null(),
            "posix_memalign({len}) failed rc={rc}"
        );
        Self {
            ptr: p as *mut u8,
            len,
        }
    }
    fn ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr as *mut libc::c_void) }
    }
}
