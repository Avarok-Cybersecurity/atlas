// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-2 POSIX reference backend. Single pinned bounce buffer, `pread` +
// `cuMemcpyHtoDAsync`, stream-sync after every memcpy to avoid the next
// pread overwriting in-flight DMA. Slow-but-deterministic; used by tests as
// the oracle the io_uring backend is compared against.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;

use super::{ReadRequest, StorageBackend};
use crate::cuda_min::{PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout};
use crate::layout::Layout;

pub struct PosixBackend {
    layout: Layout,
    bounce: PinnedBuffer,
}

impl PosixBackend {
    pub fn new(layout: Layout) -> Result<Self> {
        let bounce = PinnedBuffer::new(layout.group_bytes() as usize)
            .context("alloc pinned bounce buffer")?;
        Ok(Self { layout, bounce })
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl StorageBackend for PosixBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        let bounce_ptr = self.bounce.ptr;
        for req in requests {
            let fd = self.layout.fd(req.group.layer);
            let off = self.layout.offset(req.group) as i64;
            let n = unsafe { libc::pread(fd, bounce_ptr, bytes, off) };
            if n != bytes as isize {
                bail!(
                    "pread {bytes}@{off} returned {n}, errno {}",
                    std::io::Error::last_os_error()
                );
            }
            // The pinned bounce buffer is shared across all requests in this
            // call; we must let the H→D DMA complete before the next pread
            // overwrites the buffer, otherwise the second cuMemcpyHtoDAsync
            // will read partial / stale bytes. Phase-3 io_uring backend uses
            // multiple registered buffers and avoids this serialization.
            copy_h_to_d_async(req.dst_dev_ptr, bounce_ptr as *const c_void, bytes, stream)?;
            stream_sync(stream)?;
        }
        Ok(())
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!(
                "write_from_host: src len {} != group bytes {bytes}",
                src.len()
            );
        }
        // O_DIRECT requires page-aligned source. Stage through the pinned
        // bounce buffer (which is page-aligned per cuMemAllocHost contract).
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.bounce.ptr as *mut u8, bytes);
        }
        let fd = self.layout.fd(key.layer);
        let off = self.layout.offset(key) as i64;
        let n = unsafe { libc::pwrite(fd, self.bounce.ptr, bytes, off) };
        if n != bytes as isize {
            bail!(
                "pwrite {bytes}@{off} returned {n}, errno {}",
                std::io::Error::last_os_error()
            );
        }
        // fsync would be needed for crash durability; skipped for the test
        // path where the file is single-process / single-run.
        let _ = fd;
        Ok(())
    }

    fn group_layout(&self) -> GroupLayout {
        self.layout.spec
    }
}

impl PosixBackend {
    /// Test helper: drop the page cache for the layer files so subsequent
    /// reads actually hit NVMe (otherwise small tests trivially read from RAM).
    pub fn drop_pagecache(&self) {
        for layer in 0..self.layout.spec.num_layers {
            let fd = self.layout.fd(layer);
            unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{GroupLayout, KvKind};

    fn tempdir(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("atlas-storage-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    #[ignore = "requires GPU"]
    fn write_then_read_round_trip() {
        // CUDA must be initialised before any pinned-host allocation.
        let _ctx = crate::cuda_min::CudaCtx::new(0).expect("cuda init");
        let dir = tempdir("rt");
        let spec = GroupLayout::new(1, 2, 1, 16, 128, 2, 4096);
        let layout = Layout::create(&dir, spec).unwrap();
        let mut backend = PosixBackend::new(layout).unwrap();
        let bytes = backend.layout().group_bytes() as usize;
        let pat: Vec<u8> = (0..bytes).map(|i| (i & 0xFF) as u8).collect();
        let key = GroupKey::new(0, 1, 0, KvKind::V);
        backend.write_from_host(key, &pat).unwrap();
        backend.drop_pagecache();

        let dev = crate::cuda_min::DeviceBuffer::new(bytes).unwrap();
        let req = ReadRequest {
            group: key,
            dst_dev_ptr: dev.ptr,
        };
        // Construct a stream from the (already-existing) ctx to satisfy the
        // backend signature.
        backend.read(&[req], _ctx.stream).unwrap();
        let mut host_back = vec![0_u8; bytes];
        crate::cuda_min::copy_d_to_h_async(
            host_back.as_mut_ptr() as *mut c_void,
            dev.ptr,
            bytes,
            _ctx.stream,
        )
        .unwrap();
        crate::cuda_min::stream_sync(_ctx.stream).unwrap();
        assert_eq!(host_back, pat);
        std::fs::remove_dir_all(&dir).ok();
    }
}
