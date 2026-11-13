// SPDX-License-Identifier: AGPL-3.0-only
//! GPU microtest for lane L34a (RetroInfer index build and zone selection, row `retro_attn`):
//! `kernels/ada/common/retro_kmeans_ada.cu` and `retro_select_ada.cu` against their CPU mirrors
//! (`port::retro::build::reference`, `port::retro::select::reference`) and L33's oracle
//! (`port::retro::oracle`, `oracle_eval`) on synthetic paged KV with a scrambled block table.
//!
//! * Stages (BF16 Gaussian keys): key mean, initialization, one CSR and one centroid update are
//!   bit-equal to the mirror; one segmented assignment differs only on near ties.
//! * End to end (BF16 Gaussian, BF16 uniform, FP8 uniform): sizes sum to the span, the CSR is a
//!   permutation with ascending members, two builds are bitwise equal, clusters with equal members
//!   have bit-equal value sums and centroids within 1e-2, the final assignment agrees with the
//!   mirror (Gaussian: >= 99.9%) and with L33's oracle (Gaussian up to 16K tokens: >= 99%). Then a
//!   decode-time update appends 1,024 tokens at the offsets and leaves the earlier clusters alone.
//! * Selection: QUERIES random queries on the GPU-built index: retrieval and estimation lists
//!   equal to the reference and to `oracle_eval::select` up to near ties (1e-5 relative).
//! * Timing at BENCH_CTX indexed tokens (default 131072; 0 skips): one layer build of 4 KV heads
//!   and one selection. Targets: 0.1 to 0.5 s per layer build, under 60 us per layer selection.
//!
//!   AVAROK_TARGET_HW=ada cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example retro_build_microtest
//!
//! Env: RETRO_LENS (default "4096 16384 32768"), RETRO_QUERIES (1000), RETRO_BENCH_CTX (131072).
//! Exit 0 = PASS, 1 = a check failed.

#[path = "retro_build_microtest/checks.rs"]
mod checks;
#[path = "retro_build_microtest/fixtures.rs"]
mod fixtures;

use anyhow::Result;
use spark_model::port::retro::build;
use spark_model::port::retro::config;
use spark_model::port::retro::select;
use spark_model::port::retro::state::IndexCounts;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::GpuBackend;

use fixtures::{D, H, Kind, QH, Report, bf16_bytes, dev, initial_job, make_layer, run_build};

/// Build and selection latency at `tokens` indexed tokens (uniform BF16 keys).
fn bench(gpu: &dyn GpuBackend, tokens: u32) -> Result<()> {
    let len = tokens + 68;
    let l = make_layer(gpu, len, Kind::Uniform, 99)?;
    let mut d = dev(gpu, &l)?;
    let job = initial_job(len)?;
    let mut t = Vec::new();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        run_build(gpu, &l, &mut d, &job)?;
        t.push(t0.elapsed().as_secs_f64());
    }
    t.sort_by(f64::total_cmp);
    let nc = job.n_centroids;
    println!(
        "  BENCH build: {} tokens, {nc} clusters x {H} heads: median {:.3} s per layer (target 0.1-0.5)",
        job.n_tokens, t[1]
    );
    let np = config::nprobe(nc, config::DEFAULT_RETRIEVAL_BUDGET);
    d.idx.counts = IndexCounts {
        n_centroids: nc,
        nprobe: np,
        es: config::estimation_clusters(nc, config::DEFAULT_ESTIMATION_BUDGET, np),
        indexed_end: job.end_pos(),
        sink: job.first_pos,
        listed: job.n_tokens,
    };
    gpu.copy_h2d(&d.idx.counts.to_bytes(), d.idx.header)?;
    let q = gpu.alloc(QH as usize * D * 2)?;
    let qv: Vec<f32> = (0..QH as usize * D)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .collect();
    gpu.copy_h2d(&bf16_bytes(&qv), q)?;
    let cfg = d.cfg.clone();
    let stream = gpu.default_stream();
    let mut run = |k: u32| -> Result<f64> {
        let t0 = std::time::Instant::now();
        for _ in 0..k {
            select::select_zones(gpu, &d.idx, q, &cfg, &mut d.zl, stream)?;
        }
        gpu.synchronize(stream)?;
        Ok(t0.elapsed().as_secs_f64())
    };
    run(10)?;
    let us = run(200)? * 1e6 / 200.0;
    println!(
        "  BENCH select: {nc} clusters x {H} heads: {us:.1} us per layer, launches included (target < 60)"
    );
    for p in d.bufs.iter().copied().chain([q, l.kv.k_pool, l.kv.v_pool]) {
        gpu.free(p)?;
    }
    Ok(())
}

fn env(k: &str, default: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| default.to_string())
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    if !build::available(&gpu) || !select::available(&gpu) {
        println!(
            "RESULT FAIL: retro_kmeans_ada / retro_select_ada are not in this build \
             (AVAROK_TARGET_HW=ada)"
        );
        std::process::exit(1);
    }
    let lens: Vec<u32> = env("RETRO_LENS", "4096 16384 32768")
        .split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect();
    let queries: usize = env("RETRO_QUERIES", "1000").parse().unwrap_or(1000);
    let bench_tokens: u32 = env("RETRO_BENCH_CTX", "131072").parse().unwrap_or(131_072);
    let mut rep = Report(0);
    for (i, &n) in lens.iter().enumerate() {
        // The initial index covers n tokens (4 sink + n + 64 window); 1,024 more for one update.
        let init_len = n + 68;
        for kind in [Kind::Gaussian, Kind::Uniform, Kind::Fp8] {
            println!("--- {n} indexed tokens, {}", kind.name());
            let l = make_layer(&gpu, init_len + 1024, kind, 1000 + i as u64)?;
            let mut d = dev(&gpu, &l)?;
            if kind == Kind::Gaussian {
                checks::stages(&gpu, &l, init_len, &d, &mut rep)?;
            }
            let job = checks::end_to_end(&gpu, &l, init_len, &mut d, kind, &mut rep)?;
            if kind != Kind::Fp8 {
                checks::selection(&gpu, &mut d, &job, queries, &mut rep)?;
            }
            checks::update(&gpu, &l, &mut d, &job, &mut rep)?;
            for p in d.bufs.iter().copied().chain([l.kv.k_pool, l.kv.v_pool]) {
                gpu.free(p)?;
            }
        }
    }
    if bench_tokens > 0 {
        println!("--- timing");
        bench(&gpu, bench_tokens)?;
    }
    if rep.0 == 0 {
        println!("RESULT PASS");
        Ok(())
    } else {
        println!("RESULT FAIL: {} checks failed", rep.0);
        std::process::exit(1);
    }
}
