// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-0 micro-benchmarks for the storage layer. Three paths:
//   1. cuFile (compat or GDS, depending on nvidia-fs availability)
//   2. io_uring + pinned-host bounce + cuMemcpyHtoDAsync
//   3. POSIX pread + cuMemcpyHtoDAsync (floor for comparison)
//
// All benchmarks read from a pre-allocated test file with O_DIRECT, into a
// device or pinned-host buffer of `BUF_BYTES` size, with sequential 4 MiB and
// random 64 KiB workloads. Results in MiB/s.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::Instant;

use crate::cuda_min::{CudaCtx, DeviceBuffer, PinnedBuffer, copy_h_to_d_async, stream_sync};

pub const SEQ_IO_BYTES: usize = 4 * 1024 * 1024;
pub const RAND_IO_BYTES: usize = 64 * 1024;
pub const SEQ_ITERS: usize = 64;
pub const RAND_ITERS: usize = 1024;

#[derive(Debug, Clone, Copy)]
pub struct BenchResult {
    pub mib_per_sec: f64,
    pub iters: usize,
    pub bytes_per_io: usize,
}

fn open_o_direct(path: &Path) -> Result<std::fs::File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .with_context(|| format!("open O_DIRECT {}", path.display()))
}

fn drop_pagecache(fd: RawFd) {
    unsafe {
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

fn rand_offset(file_bytes: u64, io_bytes: usize, i: usize) -> u64 {
    let max = (file_bytes - io_bytes as u64) / RAND_IO_BYTES as u64;
    let pseudo = (i as u64).wrapping_mul(2_654_435_761) % max.max(1);
    pseudo * RAND_IO_BYTES as u64
}

pub fn bench_cufile(
    cufile: &cufile_sys::CuFile,
    fd: RawFd,
    file_bytes: u64,
    dev: &DeviceBuffer,
    io_bytes: usize,
    iters: usize,
    sequential: bool,
) -> Result<BenchResult> {
    use cufile_sys::*;
    let mut descr = CUfileDescr_t {
        type_: CU_FILE_HANDLE_TYPE_OPAQUE_FD,
        handle: CUfileDescrHandle::from_fd(fd),
        fs_ops: std::ptr::null(),
    };
    let mut handle: CUfileHandle_t = std::ptr::null_mut();
    let err = unsafe { (cufile.handle_register)(&mut handle, &mut descr) };
    if err.err != CU_FILE_SUCCESS {
        bail!(
            "cuFileHandleRegister failed: {} ({})",
            err.err,
            err_to_str(err.err)
        );
    }
    let _ = unsafe { (cufile.buf_register)(dev.ptr as *const c_void, dev.bytes, 0) };
    let t = Instant::now();
    for i in 0..iters {
        let off = if sequential {
            ((i % SEQ_ITERS) * io_bytes) as i64
        } else {
            rand_offset(file_bytes, io_bytes, i) as i64
        };
        let n = unsafe { (cufile.read)(handle, dev.ptr as *mut c_void, io_bytes, off as _, 0) };
        if n != io_bytes as isize {
            unsafe {
                let _ = (cufile.buf_deregister)(dev.ptr as *const c_void);
                (cufile.handle_deregister)(handle);
            }
            bail!("cuFileRead returned {n}, expected {io_bytes}");
        }
    }
    let dt = t.elapsed().as_secs_f64();
    unsafe {
        let _ = (cufile.buf_deregister)(dev.ptr as *const c_void);
        (cufile.handle_deregister)(handle);
    }
    let bytes = (io_bytes * iters) as f64;
    Ok(BenchResult {
        mib_per_sec: bytes / dt / (1024.0 * 1024.0),
        iters,
        bytes_per_io: io_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bench_io_uring(
    ctx: &CudaCtx,
    fd: RawFd,
    file_bytes: u64,
    pinned: &PinnedBuffer,
    dev: &DeviceBuffer,
    io_bytes: usize,
    iters: usize,
    sequential: bool,
) -> Result<BenchResult> {
    use io_uring::{IoUring, opcode, types};
    let mut ring = IoUring::builder()
        .setup_sqpoll(2_000)
        .build(64)
        .context("io_uring setup_sqpoll")?;
    let t = Instant::now();
    let chunk = io_bytes;
    let host_ptr = pinned.ptr as *mut u8;
    for i in 0..iters {
        let off = if sequential {
            ((i % SEQ_ITERS) * chunk) as u64
        } else {
            rand_offset(file_bytes, chunk, i)
        };
        let read_e = opcode::Read::new(types::Fd(fd), host_ptr, chunk as u32)
            .offset(off)
            .build()
            .user_data(0);
        unsafe {
            ring.submission()
                .push(&read_e)
                .map_err(|_| anyhow::anyhow!("sq full"))?
        };
        ring.submit_and_wait(1)
            .context("io_uring submit_and_wait")?;
        let mut cq = ring.completion();
        let entry: io_uring::cqueue::Entry = cq.next().expect("cqe");
        if entry.result() != chunk as i32 {
            bail!(
                "io_uring read returned {}, expected {}",
                entry.result(),
                chunk
            );
        }
        drop(cq);
        copy_h_to_d_async(dev.ptr, host_ptr as *const c_void, chunk, ctx.stream)?;
    }
    stream_sync(ctx.stream)?;
    let dt = t.elapsed().as_secs_f64();
    let bytes = (io_bytes * iters) as f64;
    Ok(BenchResult {
        mib_per_sec: bytes / dt / (1024.0 * 1024.0),
        iters,
        bytes_per_io: io_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bench_posix(
    ctx: &CudaCtx,
    fd: RawFd,
    file_bytes: u64,
    pinned: &PinnedBuffer,
    dev: &DeviceBuffer,
    io_bytes: usize,
    iters: usize,
    sequential: bool,
) -> Result<BenchResult> {
    let host_ptr = pinned.ptr as *mut u8;
    let t = Instant::now();
    for i in 0..iters {
        let off = if sequential {
            ((i % SEQ_ITERS) * io_bytes) as i64
        } else {
            rand_offset(file_bytes, io_bytes, i) as i64
        };
        let n = unsafe { libc::pread(fd, host_ptr as *mut c_void, io_bytes, off) };
        if n != io_bytes as isize {
            bail!(
                "pread returned {n}, expected {io_bytes} (errno {})",
                std::io::Error::last_os_error()
            );
        }
        copy_h_to_d_async(dev.ptr, host_ptr as *const c_void, io_bytes, ctx.stream)?;
    }
    stream_sync(ctx.stream)?;
    let dt = t.elapsed().as_secs_f64();
    let bytes = (io_bytes * iters) as f64;
    Ok(BenchResult {
        mib_per_sec: bytes / dt / (1024.0 * 1024.0),
        iters,
        bytes_per_io: io_bytes,
    })
}

pub fn open_test_file(path: &Path) -> Result<(std::fs::File, u64)> {
    let f = open_o_direct(path)?;
    let len = f.metadata()?.len();
    drop_pagecache(f.as_raw_fd());
    Ok((f, len))
}
