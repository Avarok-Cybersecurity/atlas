// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-0 storage capability probe. Detects whether cuFile/GDS engages on the
// current host, benchmarks the candidate backends, and recommends a
// production backend (`cuFile-direct`, `cuFile-compat`, or `io_uring`).

use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use crate::bench::{
    BenchResult, RAND_IO_BYTES, RAND_ITERS, SEQ_IO_BYTES, SEQ_ITERS, bench_cufile, bench_io_uring,
    bench_posix, open_test_file,
};
use crate::cuda_min::{CudaCtx, DeviceBuffer, PinnedBuffer};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum Backend {
    CuFileDirect,
    CuFileCompat,
    IoUring,
    PosixOnly,
    None,
}

#[derive(Debug, Clone)]
pub struct ProbeConfig {
    pub dir: PathBuf,
    pub test_file_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProbeResult {
    pub libcufile_loaded: bool,
    pub libcufile_load_error: Option<String>,
    pub cufile_version: Option<i32>,
    pub nvidia_fs_kmod_loaded: bool,
    pub cufile_driver_open_ok: bool,
    pub cufile_driver_open_error: Option<String>,
    pub cufile_seq_4mib: Option<f64>,
    pub cufile_rand_64kib: Option<f64>,
    pub io_uring_seq_4mib: Option<f64>,
    pub io_uring_rand_64kib: Option<f64>,
    pub posix_seq_4mib: Option<f64>,
    pub posix_rand_64kib: Option<f64>,
    pub recommended: Backend,
    pub recommendation_reason: String,
}

fn fill_test_file(path: &Path, bytes: u64) -> Result<()> {
    if path.exists() && std::fs::metadata(path)?.len() == bytes {
        return Ok(());
    }
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let chunk = vec![0xA5u8; 1 << 20];
    let mut written = 0u64;
    while written < bytes {
        let n = ((bytes - written) as usize).min(chunk.len());
        f.write_all(&chunk[..n])?;
        written += n as u64;
    }
    f.sync_all()?;
    Ok(())
}

fn run_one_bench<F>(label: &str, f: F) -> Option<f64>
where
    F: FnOnce() -> Result<BenchResult>,
{
    match f() {
        Ok(r) => {
            tracing::info!(
                "{label}: {:.1} MiB/s ({} iters @ {} bytes)",
                r.mib_per_sec,
                r.iters,
                r.bytes_per_io
            );
            Some(r.mib_per_sec)
        }
        Err(e) => {
            tracing::warn!("{label}: SKIPPED ({e:#})");
            None
        }
    }
}

pub fn run_probe(cfg: &ProbeConfig) -> Result<ProbeResult> {
    std::fs::create_dir_all(&cfg.dir).with_context(|| format!("mkdir {}", cfg.dir.display()))?;
    let test_path = cfg.dir.join("probe-test.bin");
    fill_test_file(&test_path, cfg.test_file_bytes)?;

    let nvidia_fs = cufile_sys::nvidia_fs_loaded();
    let cufile_load = cufile_sys::CuFile::load();
    let (libcufile_loaded, libcufile_err, cufile) = match cufile_load {
        Ok(c) => (true, None, Some(c)),
        Err(e) => (false, Some(e), None),
    };
    let mut version: Option<i32> = None;
    let mut driver_open_ok = false;
    let mut driver_open_err: Option<String> = None;
    if let Some(cufile) = cufile.as_ref() {
        let mut v = 0i32;
        let r = unsafe { (cufile.get_version)(&mut v) };
        if r.err == cufile_sys::CU_FILE_SUCCESS {
            version = Some(v);
        }
        let r = unsafe { (cufile.driver_open)() };
        if r.err == cufile_sys::CU_FILE_SUCCESS {
            driver_open_ok = true;
        } else {
            driver_open_err = Some(format!("{} ({})", r.err, cufile_sys::err_to_str(r.err)));
        }
    }

    let cuda = CudaCtx::new(0).context("init CUDA context for probe")?;
    let dev = DeviceBuffer::new(SEQ_IO_BYTES)?;
    let pinned = PinnedBuffer::new(SEQ_IO_BYTES)?;
    let (file, file_bytes) = open_test_file(&test_path)?;
    let fd = file.as_raw_fd();

    let cufile_seq = cufile.as_ref().filter(|_| driver_open_ok).and_then(|c| {
        run_one_bench("cuFile seq 4MiB", || {
            bench_cufile(c, fd, file_bytes, &dev, SEQ_IO_BYTES, SEQ_ITERS, true)
        })
    });
    let cufile_rand = cufile.as_ref().filter(|_| driver_open_ok).and_then(|c| {
        run_one_bench("cuFile rand 64KiB", || {
            bench_cufile(c, fd, file_bytes, &dev, RAND_IO_BYTES, RAND_ITERS, false)
        })
    });
    let iou_seq = run_one_bench("io_uring seq 4MiB", || {
        bench_io_uring(
            &cuda,
            fd,
            file_bytes,
            &pinned,
            &dev,
            SEQ_IO_BYTES,
            SEQ_ITERS,
            true,
        )
    });
    let iou_rand = run_one_bench("io_uring rand 64KiB", || {
        bench_io_uring(
            &cuda,
            fd,
            file_bytes,
            &pinned,
            &dev,
            RAND_IO_BYTES,
            RAND_ITERS,
            false,
        )
    });
    let posix_seq = run_one_bench("posix seq 4MiB", || {
        bench_posix(
            &cuda,
            fd,
            file_bytes,
            &pinned,
            &dev,
            SEQ_IO_BYTES,
            SEQ_ITERS,
            true,
        )
    });
    let posix_rand = run_one_bench("posix rand 64KiB", || {
        bench_posix(
            &cuda,
            fd,
            file_bytes,
            &pinned,
            &dev,
            RAND_IO_BYTES,
            RAND_ITERS,
            false,
        )
    });
    let _ = (file, dev, pinned, cuda);

    let (recommended, reason) = decide(nvidia_fs, driver_open_ok, cufile_seq, iou_seq, posix_seq);

    Ok(ProbeResult {
        libcufile_loaded,
        libcufile_load_error: libcufile_err,
        cufile_version: version,
        nvidia_fs_kmod_loaded: nvidia_fs,
        cufile_driver_open_ok: driver_open_ok,
        cufile_driver_open_error: driver_open_err,
        cufile_seq_4mib: cufile_seq,
        cufile_rand_64kib: cufile_rand,
        io_uring_seq_4mib: iou_seq,
        io_uring_rand_64kib: iou_rand,
        posix_seq_4mib: posix_seq,
        posix_rand_64kib: posix_rand,
        recommended,
        recommendation_reason: reason,
    })
}

fn decide(
    nvfs: bool,
    cufile_open: bool,
    cufile: Option<f64>,
    iou: Option<f64>,
    posix: Option<f64>,
) -> (Backend, String) {
    let cf = cufile.unwrap_or(0.0);
    let iu = iou.unwrap_or(0.0);
    let px = posix.unwrap_or(0.0);
    if nvfs && cufile_open && cf >= iu * 1.05 {
        return (
            Backend::CuFileDirect,
            format!("nvidia-fs loaded; cuFile {cf:.0} MiB/s ≥ io_uring {iu:.0} MiB/s"),
        );
    }
    if cufile_open && cf >= iu.max(px) * 1.05 {
        return (
            Backend::CuFileCompat,
            format!("cuFile compat-mode {cf:.0} MiB/s beats io_uring {iu:.0} / posix {px:.0}"),
        );
    }
    if iu >= px && iu > 0.0 {
        return (
            Backend::IoUring,
            format!("io_uring {iu:.0} MiB/s ≥ posix {px:.0} MiB/s"),
        );
    }
    if px > 0.0 {
        return (
            Backend::PosixOnly,
            format!("posix-only fallback {px:.0} MiB/s"),
        );
    }
    (Backend::None, "no backend produced bandwidth".into())
}
