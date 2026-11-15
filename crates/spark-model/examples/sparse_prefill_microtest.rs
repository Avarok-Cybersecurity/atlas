// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after XAttention (antidiagonal block scoring for block-sparse prefill), described in its paper;
// independent implementation, no source copied (the reference repository has no detected license).

//! Responsibility: GPU qualification of the block-sparse prefill kernels
//! (`kernels/ada/common/prefill_block_sparse_ada.cu`, lane L35) against the FP64 CPU reference
//! (`port::sparse_prefill::reference`), on a paged cache with a shuffled block table.
//!
//! Per case (Qwen3.5-4B attention geometry by default: 16 q heads, 4 kv heads, head_dim 256):
//!   * SELECT: the GPU's block masks against the reference's masks from FP64 masses; they may differ
//!     only on blocks within `BAND` of the reference's boundary (`select::mismatches`);
//!   * SPARSE: the GPU output against FP64 attention through the GPU's own masks (normwise and
//!     pointwise bounds), plus a sentinel tail the launch must not write;
//!   * IDENTITY (every candidate block kept): the sparse attention kernel against the production dense
//!     paged kernel (`inferspark_prefill_paged`, `_fp8`) and against FP64 dense attention. The dense
//!     kernel is built for head_dim 256 only, so a case at another head dimension is judged against
//!     FP64 alone (its dense-kernel numbers are printed, not gated).
//! `AVAROK_BENCH_ITERS=N` adds timings (dense kernel, score, sparse attention) per case, plus two
//! 32K-key timing cases (BF16 and FP8 KV) that skip the FP64 oracle.
//!
//! Usage: cargo run --release -p spark-model --example sparse_prefill_microtest \
//!          --features cuda,gpu-examples

#[path = "common/sparse_prefill_fixture.rs"]
mod fixture;

use anyhow::Result;
use fixture::{
    Case, all_ones, build, call, compare, decode_masks, dense, down_u16, down_u32, up, widen,
};
use spark_model::port::sparse_prefill::launch::{self, Buffers, Kernels};
use spark_model::port::sparse_prefill::{Call, Config, Geometry, KvKind, reference, select};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const SENTINEL: u16 = 0x7F7F;
/// Selection may disagree with the FP64 reference only within this mass of its boundary.
const BAND: f64 = 2e-3;
/// SPARSE and IDENTITY bounds on BF16 outputs (values are convex combinations of V, |V| <= 1).
const NORMWISE: f64 = 1e-2;
const POINTWISE: f64 = 3e-2;
/// The head dimension the production dense paged kernel is compiled for (`HDIM` in
/// `kernels/gb10/common/prefill_paged_compute.cuh`: its shared-memory tiles and tile counts are
/// compile-time). At any other `head_dim` its output is not a reference. An earlier GPU run
/// (2026-11-13) went red on exactly the two hd128 cells, with the sparse kernel within 1.7e-3 of
/// FP64 dense attention in both and the dense kernel 0.46 and 0.86 away from it; those cells are
/// judged against FP64 alone.
const DENSE_KERNEL_HDIM: usize = 256;

fn bytes_of(v: &[u32]) -> &[u8] {
    // SAFETY: u32 is plain old data.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// SELECT: (differing blocks, differing blocks outside the band) over every (head, query block).
fn check_select(
    c: &Case,
    p: &reference::Problem,
    g: &Geometry,
    gpu_masks: &[Vec<bool>],
) -> (usize, usize) {
    let (mut diff, mut outside) = (0usize, 0usize);
    let (lo, _) = p.query_blocks();
    for (item, sel) in gpu_masks.iter().enumerate() {
        let (h, i) = (item / g.n_qb as usize, lo + item % g.n_qb as usize);
        match reference::block_mass(p, h, i, c.stride as usize) {
            Some(m) => {
                let want = select::select_greedy(&m, c.threshold as f64);
                let (d, o) = select::mismatches(&want, sel, &m, BAND);
                diff += d;
                outside += o;
            }
            // No complete strided row: the kernel must keep every candidate.
            None => {
                let d = sel.iter().filter(|x| !**x).count();
                diff += d;
                outside += d;
            }
        }
    }
    (diff, outside)
}

fn run_case(gpu: &dyn GpuBackend, c: &Case, seed: u64, stream: u64) -> Result<bool> {
    let b = build(gpu, c, seed)?;
    let n_out = c.q_len * c.nq * c.hd;
    let tail = 64 * c.nq * c.hd;
    let out = up(gpu, &vec![SENTINEL; n_out + tail])?;
    let out_d = up(gpu, &vec![SENTINEL; n_out])?;
    let ca = call(c, &b, out);
    let cfg = Config {
        enabled: true,
        min_tokens: 1,
        threshold: c.threshold,
        stride: c.stride,
    };
    let k = launch::resolve(gpu, c.kv, c.hd as u32, c.stride)
        .ok_or_else(|| anyhow::anyhow!("{}: entries did not resolve", c.name))?;
    let g = Geometry::new(&ca);
    let grid = g.score_grid(gpu.sm_count().unwrap_or(34), c.hd as u32);
    let bufs = Buffers {
        scratch: gpu.alloc(g.score_scratch_bytes(grid, c.stride))?,
        masks: gpu.alloc(g.mask_bytes())?,
        stats: gpu.alloc(launch::STATS_BYTES)?,
    };
    gpu.memset(bufs.stats, 0, launch::STATS_BYTES)?;
    launch::launch_score(gpu, &k, &ca, &g, &cfg, &bufs, grid, stream)?;
    launch::launch_attn(gpu, &k, &ca, &g, bufs.masks, stream)?;
    gpu.synchronize(stream)?;
    let p = b.problem(c);
    let words = down_u32(gpu, bufs.masks, g.items as usize * g.mask_words as usize)?;
    let gpu_masks = decode_masks(&words, &g, &p);
    let s = down_u32(gpu, bufs.stats, 6)?;
    let kept = (s[0] as u64) | ((s[1] as u64) << 32);
    let cand = (s[2] as u64) | ((s[3] as u64) << 32);

    let (diff, outside) = if c.oracle {
        check_select(c, &p, &g, &gpu_masks)
    } else {
        (0, 0)
    };
    let all = down_u16(gpu, out, n_out + tail)?;
    let tail_ok = all[n_out..].iter().all(|x| *x == SENTINEL);
    let (nw, pw) = if c.oracle {
        compare(&all[..n_out], &reference::sparse(&p, &gpu_masks))
    } else {
        (0.0, 0.0)
    };
    println!(
        "--- {} (q {}+{}, kv {}, S {}, thr {}):",
        c.name, c.q_offset, c.q_len, b.kv_len, c.stride, c.threshold
    );
    println!(
        "  select   kept {kept}/{cand} blocks ({:.1}%), differs on {diff}, outside the band {outside}",
        100.0 * kept as f64 / cand.max(1) as f64
    );
    println!("  sparse   normwise {nw:.3e} max {pw:.3e} sentinel tail intact={tail_ok}");
    let mut pass = outside == 0 && nw <= NORMWISE && pw <= POINTWISE && tail_ok && cand > 0;

    // IDENTITY: every candidate kept, against the dense production kernel and FP64 dense attention.
    gpu.copy_h2d(bytes_of(&all_ones(&gpu_masks, g.mask_words)), bufs.masks)?;
    launch::launch_attn(gpu, &k, &ca, &g, bufs.masks, stream)?;
    dense(gpu, &ca, out_d, stream)?;
    gpu.synchronize(stream)?;
    let sp = down_u16(gpu, out, n_out)?;
    let (nd, pd) = compare(&sp, &widen(&down_u16(gpu, out_d, n_out)?));
    let (nr, pr) = if c.oracle {
        compare(&sp, &reference::dense(&p))
    } else {
        (0.0, 0.0)
    };
    let dense_is_reference = c.hd == DENSE_KERNEL_HDIM;
    println!(
        "  identity vs dense kernel normwise {nd:.3e} max {pd:.3e}{}; vs FP64 normwise {nr:.3e} \
         max {pr:.3e}{}",
        if dense_is_reference {
            ""
        } else {
            " (not judged: that kernel is built for head_dim 256)"
        },
        if c.oracle {
            ""
        } else {
            " (FP64 oracle skipped)"
        }
    );
    // Every case needs one identity reference: the dense kernel at its own head dimension, FP64
    // wherever the oracle runs.
    pass &= dense_is_reference || c.oracle;
    pass &= !dense_is_reference || (nd <= NORMWISE && pd <= POINTWISE);
    pass &= nr <= NORMWISE && pr <= POINTWISE;
    if let Some(iters) = std::env::var("AVAROK_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        // The score runs first, which restores the case's real masks for the attention timing.
        bench(gpu, &k, &ca, &g, &cfg, &bufs, grid, out_d, iters, stream)?;
    }
    b.free(gpu);
    for x in [out, out_d, bufs.scratch, bufs.masks, bufs.stats] {
        gpu.free(x).ok();
    }
    println!("  {}", if pass { "ok" } else { "FAIL" });
    Ok(pass)
}

#[allow(clippy::too_many_arguments)]
fn bench(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    c: &Call,
    g: &Geometry,
    cfg: &Config,
    b: &Buffers,
    grid: u32,
    out_d: DevicePtr,
    iters: usize,
    stream: u64,
) -> Result<()> {
    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        f()?;
        gpu.synchronize(stream)?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f()?;
        }
        gpu.synchronize(stream)?;
        Ok(t0.elapsed().as_secs_f64() * 1e6 / iters.max(1) as f64)
    };
    let score = time(&|| launch::launch_score(gpu, k, c, g, cfg, b, grid, stream))?;
    let attn = time(&|| launch::launch_attn(gpu, k, c, g, b.masks, stream))?;
    let dense_us = time(&|| dense(gpu, c, out_d, stream))?;
    println!(
        "  BENCH dense {dense_us:.1} us | score {score:.1} us + sparse attn {attn:.1} us = {:.1} us \
         ({:.2}x)",
        score + attn,
        dense_us / (score + attn)
    );
    Ok(())
}

fn cases() -> Vec<Case> {
    let base = Case {
        name: "",
        kv: KvKind::Bf16,
        nq: 16,
        nkv: 4,
        hd: 256,
        q_offset: 1792,
        q_len: 512,
        stride: 8,
        threshold: 0.9,
        planted: false,
        oracle: true,
    };
    let (fp8, small) = (KvKind::Fp8, (8, 2, 128));
    let mut v = vec![
        Case {
            name: "bf16 hd256 S8 aligned random",
            ..base
        },
        Case {
            name: "bf16 hd256 S8 unaligned planted",
            q_offset: 1800,
            q_len: 500,
            planted: true,
            ..base
        },
        Case {
            name: "fp8 hd256 S8 aligned planted",
            kv: fp8,
            planted: true,
            ..base
        },
        Case {
            name: "fp8 hd256 S4 unaligned random",
            kv: fp8,
            q_offset: 1500,
            q_len: 300,
            stride: 4,
            ..base
        },
        Case {
            name: "bf16 hd256 S16 thr 0.95 planted",
            stride: 16,
            threshold: 0.95,
            planted: true,
            ..base
        },
        Case {
            name: "bf16 hd128 S4 thr 0.8 planted",
            nq: small.0,
            nkv: small.1,
            hd: small.2,
            stride: 4,
            threshold: 0.8,
            planted: true,
            ..base
        },
        Case {
            name: "fp8 hd128 S16 short chunk",
            kv: fp8,
            nq: small.0,
            nkv: small.1,
            hd: small.2,
            q_offset: 2000,
            q_len: 70,
            stride: 16,
            ..base
        },
    ];
    if std::env::var("AVAROK_BENCH_ITERS").is_ok() {
        // GPU-only checks (sentinel, identity against the dense kernel) at 32K keys; TTFT at 128K is
        // the serve cell's job (a GPU script that is not part of this repository).
        let big = Case {
            q_offset: 30_720,
            q_len: 2048,
            planted: true,
            oracle: false,
            ..base
        };
        v.push(Case {
            name: "bench bf16 32K",
            ..big
        });
        v.push(Case {
            name: "bench fp8 32K",
            kv: fp8,
            ..big
        });
    }
    v
}

fn main() -> Result<()> {
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    if !gpu.has_module(launch::MODULE) {
        println!(
            "RESULT: FAIL (module {} is not in this image; build with AVAROK_TARGET_HW=ada)",
            launch::MODULE
        );
        std::process::exit(2);
    }
    println!(
        "=== sparse_prefill microtest: band {BAND}, normwise {NORMWISE}, pointwise {POINTWISE} ==="
    );
    let stream = gpu.create_stream()?;
    let mut pass = true;
    for (n, c) in cases().iter().enumerate() {
        pass &= run_case(gpu, c, 0x5A_0000 + n as u64, stream)?;
    }
    if pass {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}
