// SPDX-License-Identifier: AGPL-3.0-only

//! [`PageAlignedBuf`] — a 4 KiB-aligned heap buffer, so a blob moving through
//! [`crate::Residency`] can go straight to O_DIRECT.
//!
//! Why this is not a `Vec<u8>`: [`crate::DirectSwapFile`]'s read/write paths
//! check `ptr & 0xfff == 0` and, when it fails, stage the whole record through
//! an internal page-aligned bounce. A glibc `Vec` allocation essentially never
//! satisfies that, so the residency's single scratch buffer made **every**
//! O_DIRECT write and read pay an extra full-record memcpy. At the SSM
//! snapshot record size (66,846,720 B) that is a wasted ~66 MB pass — ~19 ms
//! measured for a same-size host memcpy — on both spill-to-disk and
//! fault-from-disk. Allocating the scratch aligned deletes it.
//!
//! `std::alloc` rather than `posix_memalign` keeps the crate's Windows arm
//! building without widening its libc-only-on-unix dependency budget.

/// A page-aligned (4 KiB) owned byte buffer. Zero-length is the [`Default`],
/// which allocates nothing — that is what makes `std::mem::take` usable on a
/// field of this type.
pub(crate) struct PageAlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: exclusive ownership of a plain heap allocation; no interior sharing.
unsafe impl Send for PageAlignedBuf {}

const PAGE: usize = 4096;

impl PageAlignedBuf {
    /// Zeroed buffer of `len` bytes, 4 KiB-aligned. `len == 0` allocates nothing.
    pub(crate) fn new(len: usize) -> Self {
        if len == 0 {
            return Self::default();
        }
        let layout = std::alloc::Layout::from_size_align(len, PAGE)
            .expect("PageAlignedBuf: len > 0 and PAGE is a valid power-of-two align");
        // SAFETY: non-zero layout size; freed in `Drop` with the same layout.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "PageAlignedBuf: alloc of {len} B failed");
        Self { ptr, len }
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` owns `len` initialized bytes for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: as above, and `&mut self` guarantees exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Default for PageAlignedBuf {
    fn default() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling().as_ptr(),
            len: 0,
        }
    }
}

impl Drop for PageAlignedBuf {
    fn drop(&mut self) {
        if self.len == 0 {
            return; // the dangling Default — nothing was allocated
        }
        let layout = std::alloc::Layout::from_size_align(self.len, PAGE)
            .expect("PageAlignedBuf: layout was valid at construction");
        // SAFETY: allocated by `new` with this exact layout, never freed twice.
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}
