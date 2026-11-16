// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: GPU qualification of the RetroInfer exact-zone gather
//! (`kernels/ada/common/retro_gather_ada.cu`) and estimation attention
//! (`kernels/ada/common/retro_attn_ada.cu`), port lane L34b, against the CPU
//! references in `spark_model::port::retro::{gather, attn}` (which the unit
//! tests check against L33's oracle).
//!
//! Gather cells (BF16 and FP8 pools; one row, four verify rows, a capacity that
//! drops whole clusters, the 0.8B shape): every output array must be EXACTLY
//! equal (K and V bits, positions, valid lengths, drop counts, and the sentinel
//! after each head's valid length).
//! Attention cells (4B decode and verify K=4 in one launch, a per-row trim,
//! 0.8B, 27B group 6, estimation empty, exact empty, large logits, fewer
//! entries than warps, nothing attended): relative L2 error per (row, query
//! head) against the f64 reference rounded to BF16 (the output format) at most
//! 2e-3, LSE within 1e-3 relative (both -inf where nothing is attended), and two
//! launches bitwise equal (determinism).
//! `AVAROK_BENCH_ITERS=N` adds microseconds per call at the 128K working point.
//!
//! Usage: cargo run --release -p spark-model --example retro_attn_microtest \
//!          --features cuda,gpu-examples

use anyhow::Result;
use spark_model::layers::qwen3_attention::sparse_seam::KvDtypeTag;
use spark_model::port::retro::attn::{self, AttnKernels, HostAttnIn};
use spark_model::port::retro::gather::{
    self, GatherKernels, HostExact, HostLists, HostPaged, HostZones,
};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::GpuBackend;

#[path = "retro_attn_microtest/fixtures.rs"]
mod fixtures;
use fixtures::{
    AttnFx, BS, GatherFx, HD, SCALE, SENT, attn_fx, attn_launch, bits, down, exact_out, gather_fx,
    gather_launch,
};

fn same<T: PartialEq + std::fmt::Debug>(name: &str, got: &[T], want: &[T]) -> bool {
    match got.iter().zip(want).position(|(a, b)| a != b) {
        None if got.len() == want.len() => true,
        None => {
            println!("  {name}: FAIL length {} vs {}", got.len(), want.len());
            false
        }
        Some(i) => {
            println!("  {name}: FAIL at {i}: got {:?} want {:?}", got[i], want[i]);
            false
        }
    }
}

fn gather_cell(
    gpu: &dyn GpuBackend,
    k: &GatherKernels,
    f: &GatherFx,
    cap: u32,
    name: &str,
    s: u64,
) -> Result<bool> {
    let p = HostPaged {
        k_pool: &f.k,
        v_pool: &f.v,
        dtype: f.dtype,
        block_size: BS,
        kv_heads: f.kvh,
        block_stride: f.stride(),
        k_scale: 0.75,
        v_scale: 1.5,
        block_table: &f.table,
        row_lens: &f.lens,
    };
    let l = HostLists {
        header: &f.header,
        offsets: &f.offsets,
        tokens: &f.tokens,
        cap_centroids: f.capc,
        cap_tokens: f.capt,
    };
    let z = HostZones {
        ids: &f.ids,
        counts: &f.counts,
        cap: f.rcap,
    };
    let mut want = HostExact::filled(
        f.kvh,
        cap,
        u16::from_le_bytes([SENT; 2]),
        i32::from_le_bytes([SENT; 4]),
    );
    gather::reference_gather(&p, &l, &z, &mut want);

    let (base, out) = exact_out(gpu, f.kvh, cap)?;
    let g = gather_launch(gpu, f, out)?;
    gather::launch_gather(gpu, k, &g, s)?;
    gpu.synchronize(s)?;
    let (rows, kvh) = ((f.kvh * cap) as usize, f.kvh as usize);
    let mut ok = same(
        &format!("{name} K"),
        &down::<u16>(gpu, out.k, rows * HD)?,
        &want.k,
    );
    ok &= same(
        &format!("{name} V"),
        &down::<u16>(gpu, out.v, rows * HD)?,
        &want.v,
    );
    ok &= same(
        &format!("{name} positions"),
        &down::<i32>(gpu, out.pos, rows)?,
        &want.pos,
    );
    ok &= same(
        &format!("{name} valid"),
        &down::<i32>(gpu, out.valid, kvh)?,
        &want.valid,
    );
    ok &= same(
        &format!("{name} dropped"),
        &down::<i32>(gpu, out.dropped, kvh)?,
        &want.dropped,
    );
    let verdict = if ok { "exact" } else { "FAIL" };
    println!(
        "  {name:<36} {verdict} (valid {:?}, dropped {:?})",
        want.valid, want.dropped
    );
    for ptr in [
        g.kv.k_pool,
        g.kv.v_pool,
        g.block_table,
        g.row_lens,
        g.retrieval_ids,
        g.counts,
    ] {
        gpu.free(ptr)?;
    }
    for ptr in [
        g.lists.header,
        g.lists.list_offsets,
        g.lists.list_tokens,
        base,
    ] {
        gpu.free(ptr)?;
    }
    Ok(ok)
}

fn attn_cell(gpu: &dyn GpuBackend, f: &AttnFx, name: &str, s: u64) -> Result<bool> {
    let (rows, qh) = (f.rows as usize, f.qh as usize);
    let input = HostAttnIn {
        q: &f.q,
        q_row_stride: qh * HD,
        rows,
        row_trim: f.row_trim as usize,
        q_heads: qh,
        kv_heads: f.kvh as usize,
        scale: SCALE,
        exact: &f.exact,
        est_ids: &f.ids,
        counts: &f.counts,
        es_cap: f.es_cap as usize,
        centroids: &f.cent,
        value_sum: &f.vsum,
        cluster_size: &f.size,
        cap_centroids: f.capc as usize,
    };
    let (want, want_lse) = attn::reference_attention(&input);
    let mut a = attn_launch(gpu, f)?;
    let k = AttnKernels::resolve(gpu, f.qh / f.kvh)?;
    attn::launch_attention(gpu, &k, &a, s)?;
    gpu.synchronize(s)?;
    let out1: Vec<u16> = down(gpu, a.out, rows * qh * HD)?;
    let lse1: Vec<f32> = down(gpu, a.lse, rows * qh)?;
    let first = a.out;
    a.out = gpu.alloc(rows * qh * HD * 2)?;
    attn::launch_attention(gpu, &k, &a, s)?;
    gpu.synchronize(s)?;
    let again: Vec<u16> = down(gpu, a.out, rows * qh * HD)?;
    let mut ok = same(&format!("{name} determinism"), &again, &out1);
    let (mut worst, mut worst_lse) = (0f64, 0f64);
    for rh in 0..rows * qh {
        // The target is the f64 reference rounded to BF16 (the output format), so the error
        // measured is the kernel's f32 arithmetic.
        let w: Vec<f64> = want[rh * HD..][..HD]
            .iter()
            .map(|&x| f64::from(attn::bf16_f32(bits(x))))
            .collect();
        let g: Vec<f64> = out1[rh * HD..][..HD]
            .iter()
            .map(|&b| f64::from(attn::bf16_f32(b)))
            .collect();
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        let err = g
            .iter()
            .zip(&w)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            ok &= err == 0.0;
        } else {
            worst = worst.max(err / norm);
        }
        let (gl, wl) = (f64::from(lse1[rh]), f64::from(want_lse[rh]));
        if wl == f64::NEG_INFINITY {
            ok &= gl == f64::NEG_INFINITY;
        } else {
            worst_lse = worst_lse.max((gl - wl).abs() / wl.abs().max(1.0));
        }
    }
    ok &= worst <= 2e-3 && worst_lse <= 1e-3;
    let verdict = if ok { "pass" } else { "FAIL" };
    println!("  {name:<36} {verdict} (rel L2 {worst:.2e}, LSE {worst_lse:.2e})");
    for ptr in [
        a.q,
        a.exact.k,
        a.exact.v,
        a.exact.pos,
        a.exact.valid,
        a.exact.dropped,
        a.est_ids,
    ] {
        gpu.free(ptr)?;
    }
    for ptr in [
        a.counts,
        a.index.centroids,
        a.index.value_sum,
        a.index.cluster_size,
        first,
        a.out,
        a.lse,
    ] {
        gpu.free(ptr)?;
    }
    Ok(ok)
}

fn time_us(
    gpu: &dyn GpuBackend,
    iters: usize,
    s: u64,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f64> {
    f()?;
    gpu.synchronize(s)?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    gpu.synchronize(s)?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / iters as f64)
}

fn bench(gpu: &dyn GpuBackend, gk: &GatherKernels, iters: usize, s: u64) -> Result<()> {
    // The 4B at 128K: 4 kv heads, 147 retrieval clusters of about 16 tokens (the 8K pool keeps
    // the tail at the 64-token window: about 2,420 gathered entries per head), and attention
    // over about 3,444 exact entries (window plus one update interval) and 1,900 clusters.
    let f = gather_fx(KvDtypeTag::Bf16, 4, 8192, 1, 147, 99);
    let cap = 4 + 64 + 1024 + 17 + 2 * 160 * 16;
    let (base, out) = exact_out(gpu, 4, cap)?;
    let g = gather_launch(gpu, &f, out)?;
    let us = time_us(gpu, iters, s, || gather::launch_gather(gpu, gk, &g, s))?;
    let valid: Vec<i32> = down(gpu, out.valid, 4)?;
    // K and V, read and written.
    let bytes: f64 = valid
        .iter()
        .map(|&n| f64::from(n) * HD as f64 * 2.0 * 4.0)
        .sum();
    println!(
        "  BENCH gather bf16 kv4 valid {valid:?}: {us:.1} us/call, {:.1} GB/s",
        bytes / us / 1e3
    );
    gpu.free(base)?;
    for rows in [1u32, 4] {
        let fx = attn_fx(4, 16, rows, 0, 3444 + rows as usize - 1, 1900, 1.0, 7);
        let a = attn_launch(gpu, &fx)?;
        let k = AttnKernels::resolve(gpu, 4)?;
        let us = time_us(gpu, iters, s, || attn::launch_attention(gpu, &k, &a, s))?;
        println!("  BENCH estimation attention kv4 q16 rows {rows} est 1900: {us:.1} us/call");
    }
    Ok(())
}

fn main() -> Result<()> {
    println!(
        "=== retro_attn microtest: gather (exact) and estimation attention (rel L2 <= 2e-3) ==="
    );
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    if !GatherKernels::available(gpu) || !AttnKernels::available(gpu) {
        println!(
            "RESULT: FAIL (modules {} / {} are not in this image; build with AVAROK_TARGET_HW=ada)",
            gather::MODULE,
            attn::MODULE
        );
        std::process::exit(2);
    }
    let gk = GatherKernels::resolve(gpu)?;
    let s = gpu.create_stream()?;
    let mut pass = true;
    println!("--- gather");
    for dtype in [KvDtypeTag::Bf16, KvDtypeTag::Fp8] {
        let f = gather_fx(dtype, 4, 5000, 1, 40, 1);
        pass &= gather_cell(gpu, &gk, &f, 4096, &format!("{dtype:?} decode"), s)?;
        let f = gather_fx(dtype, 4, 5000, 4, 40, 2);
        pass &= gather_cell(gpu, &gk, &f, 4096, &format!("{dtype:?} verify rows 4"), s)?;
        pass &= gather_cell(
            gpu,
            &gk,
            &f,
            400,
            &format!("{dtype:?} capacity 400 (drops)"),
            s,
        )?;
        let f = gather_fx(dtype, 2, 700, 1, 12, 3);
        pass &= gather_cell(gpu, &gk, &f, 2048, &format!("{dtype:?} 0.8B kv2"), s)?;
    }
    println!("--- estimation attention");
    // (name, kv heads, q heads, rows, row_trim, exact entries, estimation ids, query gain)
    let cells: [(&str, u32, u32, u32, u32, usize, usize, f32); 10] = [
        ("4B decode", 4, 16, 1, 0, 2400, 1900, 1.0),
        ("4B verify K=4 (one launch)", 4, 16, 4, 0, 2404, 1900, 1.0),
        ("4B row trim 3", 4, 16, 1, 3, 2404, 1900, 1.0),
        ("0.8B kv2 q8", 2, 8, 1, 0, 1200, 600, 1.0),
        ("27B group 6", 4, 24, 2, 0, 1500, 900, 1.0),
        ("estimation empty", 4, 16, 2, 0, 2000, 0, 1.0),
        ("exact empty", 4, 16, 1, 0, 0, 1500, 1.0),
        ("large logits", 4, 16, 1, 0, 800, 400, 300.0),
        ("fewer entries than warps", 4, 16, 3, 0, 9, 5, 2.0),
        ("nothing attended (trimmed away)", 4, 16, 2, 64, 64, 0, 1.0),
    ];
    for (i, (name, kvh, qh, rows, trim, ne, nest, gain)) in cells.into_iter().enumerate() {
        pass &= attn_cell(
            gpu,
            &attn_fx(kvh, qh, rows, trim, ne, nest, gain, 40 + i as u64),
            name,
            s,
        )?;
    }
    let iters: usize = std::env::var("AVAROK_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if iters > 0 {
        bench(gpu, &gk, iters, s)?;
    }
    if pass {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}
