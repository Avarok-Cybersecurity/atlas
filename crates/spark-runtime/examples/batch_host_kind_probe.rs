// SPDX-License-Identifier: AGPL-3.0-only
//! GPU probe: which host-memory kinds the device-to-host copy methods accept on this driver.
//!
//! One arm per process (a CUDA illegal address is sticky): host kind (`alloc` = `cuMemAllocHost`,
//! `register` = ordinary page-aligned memory locked with `GpuBackend::host_register`, the host
//! tier's pool kind) times method (`single`, `loop` = one async copy per entry, `batch` =
//! `cuMemcpyBatchAsync`). 256 descriptors of 16 KiB, read back and compared with the pattern.
//! Exit 0 and `RESULT ok` when the bytes match, exit 1 and `RESULT FAIL <why>` otherwise.
//!
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model --example batch_host_kind_probe \
//!       --features cuda,gpu-examples -- --host register --method batch
use std::alloc::{Layout, alloc_zeroed, dealloc};

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{CopyEntry, GpuBackend};

const DESC: usize = 16 << 10;
const N: usize = 256;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
}

fn run(host_kind: &str, method: &str) -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let total = DESC * N;
    let pattern: Vec<u8> = (0..total)
        .map(|i| (i.wrapping_mul(131) ^ (i >> 9)) as u8)
        .collect();
    let dev = gpu.alloc(total)?;
    gpu.copy_h2d(&pattern, dev)?;
    let stream = gpu.create_stream()?;
    let layout = Layout::from_size_align(total, 4096)?;
    let (host, owned_alloc) = match host_kind {
        "alloc" => (gpu.alloc_host_pinned(total)?, true),
        "register" => {
            // SAFETY: a fresh zeroed region of `total` bytes, freed below.
            let p = unsafe { alloc_zeroed(layout) };
            gpu.host_register(p as usize, total)?;
            (p, false)
        }
        other => bail!("unknown --host {other}"),
    };
    // SAFETY: `host` spans `total` writable bytes owned by this function.
    let host_slice = unsafe { std::slice::from_raw_parts_mut(host, total) };
    if arg("--dir").as_deref() == Some("h2d") {
        host_slice.copy_from_slice(&pattern);
        gpu.memset(dev, 0, total)?;
        match method {
            "single" => gpu.copy_h2d_async_retained(host_slice, dev, stream)?,
            "loop" => {
                for (i, c) in host_slice.chunks(DESC).enumerate() {
                    gpu.copy_h2d_async_retained(c, dev.offset(i * DESC), stream)?;
                }
            }
            "batch" | "batchany" => {
                let entries: Vec<CopyEntry> = (0..N)
                    .map(|i| CopyEntry {
                        dev: dev.offset(i * DESC),
                        host: host as usize + i * DESC,
                        bytes: DESC,
                    })
                    .collect();
                gpu.copy_batch_h2d_async(&entries, stream, method == "batchany")?;
            }
            other => bail!("unknown --method {other}"),
        }
        gpu.synchronize(stream)?;
        let mut back = vec![0u8; total];
        gpu.copy_d2h(dev, &mut back)?;
        if back != pattern {
            bail!("h2d bytes differ");
        }
        return Ok(());
    }
    host_slice.fill(0);
    match method {
        "single" => gpu.copy_d2h_async(dev, host_slice, stream)?,
        "loop" => {
            for (i, c) in host_slice.chunks_mut(DESC).enumerate() {
                gpu.copy_d2h_async(dev.offset(i * DESC), c, stream)?;
            }
        }
        "batch" => {
            let entries: Vec<CopyEntry> = (0..N)
                .map(|i| CopyEntry {
                    dev: dev.offset(i * DESC),
                    host: host as usize + i * DESC,
                    bytes: DESC,
                })
                .collect();
            gpu.copy_batch_d2h_async(&entries, stream)?;
        }
        other => bail!("unknown --method {other}"),
    }
    gpu.synchronize(stream)?;
    if host_slice != pattern.as_slice() {
        bail!("bytes differ");
    }
    if owned_alloc {
        gpu.free_host_pinned(host, total)?;
    } else {
        gpu.host_unregister(host as usize)?;
        // SAFETY: allocated above with the same layout.
        unsafe { dealloc(host, layout) };
    }
    Ok(())
}

fn main() {
    let host = arg("--host").unwrap_or_else(|| "register".into());
    let method = arg("--method").unwrap_or_else(|| "batch".into());
    match run(&host, &method) {
        Ok(()) => println!("RESULT ok host={host} method={method}"),
        Err(e) => {
            println!("RESULT FAIL host={host} method={method}: {e:#}");
            std::process::exit(1);
        }
    }
}
