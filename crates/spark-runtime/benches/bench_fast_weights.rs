// SPDX-License-Identifier: AGPL-3.0-only

//! Benchmark harness: compare SafetensorsLoader (mmap) vs FastSafetensorsLoader
//! on real model directories, writing bytes into MockGpuBackend so the disk →
//! host stage is the bottleneck (MockGpuBackend::copy_h2d is a `Vec::copy_from_slice`).
//!
//! Usage:
//!   ATLAS_FAST_LOAD_BENCH_DIR=/path/to/model cargo run --release \
//!       -p spark-runtime --features test-utils --bin bench_fast_weights
//!
//! The script runs each loader in COLD and WARM mode:
//!   - cold: `posix_fadvise(DONTNEED)` across every shard before the run
//!   - warm: skip the advise (OS page cache already has the pages)
//!
//! Prints a markdown-formatted table of times and speedups.

#![cfg(all(unix, feature = "test-utils"))]

use spark_runtime::fast_weights::FastSafetensorsLoader;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn drop_page_cache(dir: &Path) {
    // Walk shard files and advise the kernel to drop their pages.
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".safetensors") {
            continue;
        }
        // Resolve symlinks — HuggingFace cache stores blobs in a sibling dir.
        let real = match std::fs::canonicalize(&p) {
            Ok(r) => r,
            Err(_) => p.clone(),
        };
        let Ok(f) = std::fs::File::open(&real) else {
            continue;
        };
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
    // A final sync for good measure (flushes dirty pages, not relevant for
    // read-only workloads but cheap).
    unsafe {
        libc::sync();
    }
}

fn bench_one<L: WeightLoader>(tag: &str, loader: L, dir: &Path, cold: bool) -> (f64, usize) {
    if cold {
        drop_page_cache(dir);
    }
    let gpu = MockGpuBackend::new();
    let t0 = Instant::now();
    let store = loader.load(dir, &gpu, 0).expect("load failed");
    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "  {tag:24} cold={cold:5} → {:.3}s ({} tensors)",
        elapsed,
        store.len()
    );
    (elapsed, store.len())
}

struct Row {
    model: String,
    cold_mmap: f64,
    cold_fast_buffered: f64,
    cold_fast_direct: f64,
    cold_fast_auto: f64,
    warm_mmap: f64,
    warm_fast_auto: f64,
    n_tensors: usize,
}

fn run_model(model_path: &Path) -> Option<Row> {
    let label = model_path
        .components()
        .rev()
        .find_map(|c| {
            let s = c.as_os_str().to_string_lossy().to_string();
            if s.starts_with("models--") {
                Some(s.trim_start_matches("models--").replace("--", "/"))
            } else {
                None
            }
        })
        .unwrap_or_else(|| model_path.display().to_string());

    println!("\n=== {label} ===");

    let (c_mmap, n) = bench_one(
        "mmap (baseline)",
        SafetensorsLoader::new(),
        model_path,
        true,
    );

    // Fully buffered (forced): try_direct_io = false
    let mut fast_buf = FastSafetensorsLoader::new();
    fast_buf.try_direct_io = false;
    let (c_fbuf, _) = bench_one("fast buffered", fast_buf, model_path, true);

    // Forced O_DIRECT: try_direct_io=true, cap=usize::MAX disables the heuristic.
    let mut fast_dir = FastSafetensorsLoader::new();
    fast_dir.direct_io_tensor_cap = usize::MAX;
    let (c_fdir, _) = bench_one("fast O_DIRECT (forced)", fast_dir, model_path, true);

    // Auto: default cap (5000). Picks O_DIRECT or buffered per shard.
    let fast_auto = FastSafetensorsLoader::new();
    let (c_fauto, _) = bench_one("fast auto (default)", fast_auto, model_path, true);

    // Warm runs (don't drop cache)
    let (w_mmap, _) = bench_one(
        "mmap (baseline)",
        SafetensorsLoader::new(),
        model_path,
        false,
    );
    let fast_auto2 = FastSafetensorsLoader::new();
    let (w_fauto, _) = bench_one("fast auto (default)", fast_auto2, model_path, false);

    Some(Row {
        model: label,
        cold_mmap: c_mmap,
        cold_fast_buffered: c_fbuf,
        cold_fast_direct: c_fdir,
        cold_fast_auto: c_fauto,
        warm_mmap: w_mmap,
        warm_fast_auto: w_fauto,
        n_tensors: n,
    })
}

fn main() {
    let dirs: Vec<PathBuf> = std::env::args()
        .skip(1)
        .chain(std::env::var("ATLAS_FAST_LOAD_BENCH_DIR").ok())
        .map(PathBuf::from)
        .collect();

    if dirs.is_empty() {
        eprintln!("Usage: bench_fast_weights <model_dir> [<model_dir>...]");
        eprintln!("   or: ATLAS_FAST_LOAD_BENCH_DIR=/path/to/model bench_fast_weights");
        std::process::exit(2);
    }

    let mut rows = Vec::new();
    for dir in &dirs {
        if let Some(r) = run_model(dir) {
            rows.push(r);
        }
    }

    // Print markdown table
    println!("\n## Results\n");
    println!(
        "| Model | Tensors | Cold mmap | Cold buffered | Cold O_DIRECT | Cold auto | Auto speedup (cold) | Warm mmap | Warm auto | Warm speedup |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for r in &rows {
        let cold_auto_speedup = r.cold_mmap / r.cold_fast_auto;
        let warm_auto_speedup = r.warm_mmap / r.warm_fast_auto;
        println!(
            "| {} | {} | {:.2}s | {:.2}s | {:.2}s | {:.2}s | **{:.2}x** | {:.2}s | {:.2}s | **{:.2}x** |",
            r.model,
            r.n_tensors,
            r.cold_mmap,
            r.cold_fast_buffered,
            r.cold_fast_direct,
            r.cold_fast_auto,
            cold_auto_speedup,
            r.warm_mmap,
            r.warm_fast_auto,
            warm_auto_speedup,
        );
    }
}
