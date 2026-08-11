// SPDX-License-Identifier: AGPL-3.0-only

//! Page-aligned O_DIRECT I/O primitives for [`super::FastSafetensorsLoader`].

use anyhow::{Result, bail};
use std::alloc::{Layout, alloc, dealloc};
use std::fs::File;
use std::path::Path;

/// O_DIRECT requires offsets, sizes, and buffers to be aligned to the
/// filesystem block size. 4 KiB is the upper bound on every Linux arch we
/// target and satisfies every fs we've seen.
pub(crate) const O_DIRECT_ALIGN: usize = 4096;

/// Heap buffer aligned to [`O_DIRECT_ALIGN`]. `Send` so it can cross a channel
/// from the reader thread to the copier thread.
///
/// `cap` is the allocation; `init` is how much of it has actually been written.
/// The two differ on every short read — the `pread` window is rounded UP to a
/// 4 KiB boundary, so a tensor at the tail of a file routinely leaves the last
/// fragment untouched. Only `init` bytes may ever be observed: a `&[u8]` over
/// uninitialised heap is undefined behaviour in Rust the moment it is formed,
/// regardless of whether anything reads it.
pub(crate) struct AlignedBuffer {
    ptr: *mut u8,
    cap: usize,
    /// Bytes written so far, starting at `ptr`. Invariant: `init <= cap`.
    init: usize,
    layout: Layout,
}

// SAFETY: `AlignedBuffer` owns the allocation pointed to by `ptr`; the
// pointer is created by `std::alloc::alloc` with the recorded `layout` and
// is freed in `Drop` with the matching layout. The struct holds no shared
// references and exposes no `&self` API that aliases the buffer, so moving
// it between threads only moves the unique owner of the allocation. We do
// not implement `Sync`: concurrent `&AlignedBuffer` readers are not a
// pattern Atlas uses (each shard is owned by a single reader thread).
unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(cap_bytes: usize) -> Self {
        let cap = cap_bytes.max(1).div_ceil(O_DIRECT_ALIGN) * O_DIRECT_ALIGN;
        let layout = Layout::from_size_align(cap, O_DIRECT_ALIGN).expect("valid layout");
        // SAFETY: `layout` has non-zero size (`cap_bytes.max(1)` rounded up to
        // a 4 KiB multiple), which is `alloc`'s only precondition. The null
        // return is handled immediately below.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            ptr,
            cap,
            init: 0,
            layout,
        }
    }

    /// The bytes that have actually been written, and only those.
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is the live allocation from `new` and stays valid for
        // `&self`. The length is `init`, not `cap`: `mark_initialised` only
        // ever advances it over bytes a completed `pread` wrote, so every byte
        // in the returned slice is initialised. `init <= cap` is asserted
        // there, so the slice stays inside the allocation.
        unsafe { std::slice::from_raw_parts(self.ptr, self.init) }
    }

    /// Spare capacity to `pread` into: `[init, cap)`.
    ///
    /// Returns a raw pointer rather than a `&mut [u8]` precisely because the
    /// bytes are uninitialised — handing out a slice would be the same UB this
    /// type exists to avoid.
    fn spare_ptr(&mut self) -> *mut u8 {
        // SAFETY: `init <= cap` (invariant), so the offset is in-bounds of the
        // allocation, or one-past-the-end when the buffer is full.
        unsafe { self.ptr.add(self.init) }
    }

    fn spare_len(&self) -> usize {
        self.cap - self.init
    }

    /// Record that `n` more bytes at the front of the spare region are now
    /// initialised. Callers must have actually written them.
    fn mark_initialised(&mut self, n: usize) {
        self.init += n;
        assert!(
            self.init <= self.cap,
            "AlignedBuffer: read reported {} bytes into a {}-byte allocation",
            self.init,
            self.cap
        );
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc(self.layout)` in `new`, is non-null,
        // and `Drop` runs once — so this is the single matching `dealloc`.
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn open_direct(path: &Path) -> std::io::Result<File> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::FromRawFd;
    let cstr = CString::new(path.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let flags = libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC;
    // SAFETY: `cstr` is a NUL-terminated C string that outlives the call
    // (`open` copies the path), and `flags` contains no `O_CREAT`, so the
    // variadic `mode` argument is not consulted.
    let fd = unsafe { libc::open(cstr.as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fd` is a fresh, valid, owned descriptor (the `< 0` case
    // returned above) that nothing else holds, so `File` may take ownership.
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn open_direct(_path: &Path) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "O_DIRECT only supported on Linux",
    ))
}

/// Read a tensor's bytes into an aligned buffer. Returns the buffer and the
/// offset within it where the tensor's `len` bytes start.
///
/// With `using_direct=true`, the read window is widened to the nearest 4 KiB
/// boundaries so the offset/size/buffer alignment constraints are met. If the
/// aligned window extends past end-of-file, we let the kernel return a short
/// read for the trailing fragment (Linux ≥ 2.6 accepts this for O_DIRECT on
/// mainstream filesystems — we only require that the tensor's exact bytes
/// have been populated, which the post-loop check enforces).
pub(crate) fn read_tensor_aligned(
    fd: std::os::unix::io::RawFd,
    abs_offset: u64,
    len: usize,
    using_direct: bool,
) -> Result<(AlignedBuffer, usize)> {
    let (window_start, window_len, slice_off) = if using_direct {
        let ws = abs_offset - (abs_offset % O_DIRECT_ALIGN as u64);
        let unaligned_end = abs_offset + len as u64;
        let aligned_end = unaligned_end.div_ceil(O_DIRECT_ALIGN as u64) * O_DIRECT_ALIGN as u64;
        let wl = (aligned_end - ws) as usize;
        (ws, wl, (abs_offset - ws) as usize)
    } else {
        (abs_offset, len, 0usize)
    };

    // `AlignedBuffer::new` rounds up, so its capacity is >= window_len; the
    // loop below never asks for more than the spare region holds.
    let mut buf = AlignedBuffer::new(window_len);
    let mut filled = 0usize;
    while filled < window_len {
        let want = (window_len - filled).min(buf.spare_len());
        // SAFETY: `spare_ptr()` points at `buf`'s first uninitialised byte and
        // `want` is clamped to the spare region, so the kernel writes only
        // inside the allocation. `fd` is the caller's open descriptor.
        let ret = unsafe {
            libc::pread(
                fd,
                buf.spare_ptr() as *mut libc::c_void,
                want,
                (window_start + filled as u64) as libc::off_t,
            )
        };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            // On O_DIRECT, a non-aligned tail at EOF may return EINVAL on
            // some filesystems. Accept the short read as long as we've
            // already covered the tensor's exact range.
            if filled >= slice_off + len {
                break;
            }
            bail!(
                "pread failed at offset {}: {err}",
                window_start + filled as u64
            );
        }
        if ret == 0 {
            break; // EOF
        }
        // Only now are those bytes initialised — before this the buffer must
        // not be viewed as a `&[u8]` at all.
        buf.mark_initialised(ret as usize);
        filled += ret as usize;
    }
    if filled < slice_off + len {
        bail!(
            "short read: got {} bytes, need at least {} (tensor spans offset {}..{})",
            filled,
            slice_off + len,
            abs_offset,
            abs_offset + len as u64
        );
    }
    Ok((buf, slice_off))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    /// Write `bytes` to a scratch file and return it (kept open by the caller).
    fn scratch(tag: &str, bytes: &[u8]) -> (std::path::PathBuf, File) {
        let p = std::env::temp_dir().join(format!(
            "atlas-dio-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_all().unwrap();
        (p.clone(), std::fs::File::open(&p).unwrap())
    }

    /// The exact-fit case: buffered read of a tensor wholly inside the file.
    #[test]
    fn buffered_read_returns_exactly_the_tensor() {
        let data: Vec<u8> = (0..255u8).cycle().take(8192).collect();
        let (path, f) = scratch("exact", &data);
        let (buf, off) = read_tensor_aligned(f.as_raw_fd(), 100, 512, false).unwrap();
        assert_eq!(off, 0);
        assert_eq!(&buf.as_slice()[off..off + 512], &data[100..612]);
        std::fs::remove_file(path).ok();
    }

    /// THE UB CASE. The aligned window is rounded up past end-of-file, so the
    /// trailing fragment is never written by `pread`. `as_slice()` must expose
    /// only the bytes the kernel actually delivered — never the whole 4 KiB-
    /// rounded allocation, which would be a `&[u8]` over uninitialised heap.
    ///
    /// File is 5000 bytes; the tensor is the last 100. The aligned window is
    /// 4096..8192 (4096 bytes) but only 904 bytes exist.
    #[test]
    fn short_read_at_eof_exposes_only_written_bytes() {
        let data: Vec<u8> = (0..251u8).cycle().take(5000).collect();
        let (path, f) = scratch("shorteof", &data);
        let (buf, off) = read_tensor_aligned(f.as_raw_fd(), 4900, 100, true).unwrap();
        assert_eq!(off, 4900 - 4096, "slice offset into the aligned window");
        assert_eq!(
            buf.as_slice().len(),
            5000 - 4096,
            "slice must stop at the last byte pread wrote, not at the \
             4 KiB-rounded capacity"
        );
        assert_eq!(&buf.as_slice()[off..off + 100], &data[4900..5000]);
        std::fs::remove_file(path).ok();
    }

    /// A tensor that the file cannot actually satisfy must be an error, not a
    /// slice of whatever the allocator handed back.
    #[test]
    fn read_past_end_of_file_errors() {
        let (path, f) = scratch("past", &[7u8; 1024]);
        let err = match read_tensor_aligned(f.as_raw_fd(), 512, 4096, false) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a 4096-byte tensor cannot come out of a 1024-byte file"),
        };
        assert!(err.contains("short read"), "{err}");
        std::fs::remove_file(path).ok();
    }

    /// A zero-length tensor still yields a usable (empty) view.
    #[test]
    fn zero_length_tensor_is_empty_not_uninitialised() {
        let (path, f) = scratch("zero", &[1u8; 4096]);
        let (buf, off) = read_tensor_aligned(f.as_raw_fd(), 0, 0, false).unwrap();
        assert!(buf.as_slice()[off..off].is_empty());
        std::fs::remove_file(path).ok();
    }
}
