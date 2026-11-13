// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Atlas GPU oracle; no prototype source was read.

//! Responsibility: GPU qualification of the Quest page-sparse kernels
//! (`kernels/ada/common/attn_page_sparse_ada.cu`, port lane L30) at the
//! Qwen3.5-4B attention geometry (16 query heads, 4 KV heads, head dim 256,
//! blocks of 16, pages of 64), through the exact launches the selector issues
//! (`spark_model::port::sparse_quest::launch`), against the CPU reference
//! (`sparse_quest::reference`):
//!
//!   1. summaries: a prefill fold (host ranges, chunked) then a verify fold
//!      (device ranges, 4 rows per sequence), byte-equal to the reference's
//!      full build for every full page, BF16 and FP8 keys;
//!   2. scores: every candidate page against the reference's f64 bound
//!      (relative 1e-4);
//!   3. compaction: the View tables and lengths equal to the reference's
//!      compaction of the GPU's own scores (exact top-k set equality);
//!   4. the identity gate: with a budget covering every page the View equals
//!      the dense table, and `paged_decode_attn` over it is bit-identical to
//!      dense attention;
//!   5. the rebuild path: with every summary invalidated, the score kernel
//!      rebuilds the summaries, gives the same scores, and leaves them trusted.
//!
//! `AVAROK_BENCH_ITERS=N` adds the score + compaction time per layer at
//! 128K tokens next to one dense paged-decode layer.
//!
//! Usage: AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!          --example page_sparse_microtest --features cuda,gpu-examples -- [LEN ...]

#[path = "page_sparse_microtest_parts/fixture.rs"]
mod fixture;

use anyhow::Result;
use spark_model::layers::ops;
use spark_model::port::sparse_quest::launch::{self, Budget, Kernels, ViewOut};
use spark_model::port::sparse_quest::reference::{self as r, RefBudget};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::page_summary::PageSummaryBuffers;

use fixture::*;

fn run_case(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    paged: spark_runtime::gpu::KernelHandle,
    len: usize,
    eb: usize,
) -> Result<bool> {
    let s = gpu.default_stream();
    let c = build(gpu, len, eb, 0x51A7 ^ (len as u64) ^ ((eb as u64) << 40))?;
    let l = layout(&c)?;
    let sb = PageSummaryBuffers::alloc(gpu, l.clone(), s)?;
    let lb = bufs(&c, &l, &sb);
    fold_all(gpu, k, &c, &lb, len, s)?;
    let bad = check_summaries(gpu, &c, &l, &sb, 0, (len - 1 + R) / 64)?;
    println!("  [{len} eb{eb}] summaries: {bad} mismatched (page, head) records");
    let mut ok = bad == 0;

    let b = Budget {
        sink_pages: 1,
        recent_tokens: 512,
        topk_pages: 8,
        min_ctx: 1024,
    };
    let stride = launch::score_stride(c.row_stride as u32, PB as u32) as usize;
    let scores = gpu.alloc(GROUPS * stride * 4)?;
    let out_bt = gpu.alloc(GROUPS * R * c.row_stride * 4)?;
    let out_sl = gpu.alloc(GROUPS * R * 4)?;
    let stats = gpu.alloc(16)?;
    gpu.memset(stats, 0, 16)?;
    let view = ViewOut {
        block_table: out_bt,
        seq_lens: out_sl,
        stats,
    };
    launch::launch_score(
        gpu,
        k,
        &lb,
        &rows(&c),
        c.q,
        (NQ * HD) as u32,
        NQ as u32,
        scores,
        &b,
        0,
        s,
    )?;
    launch::launch_compact(
        gpu,
        k,
        BS as u32,
        PB as u32,
        &rows(&c),
        scores,
        &view,
        &b,
        true,
        s,
    )?;
    gpu.synchronize(s)?;
    let gs = as_f32(&down(gpu, scores, GROUPS * stride * 4)?);
    let ob = as_i32(&down(gpu, out_bt, GROUPS * R * c.row_stride * 4)?);
    let ol = as_i32(&down(gpu, out_sl, GROUPS * R * 4)?);
    let rb = RefBudget {
        sink_pages: 1,
        recent_tokens: 512,
        topk_pages: 8,
        min_ctx: 1024,
    };
    let mut worst = 0f64;
    let mut table_bad = 0;
    for (g, table) in c.tables.iter().enumerate() {
        let rp = r::recent_page(len, 512, 64);
        let want = ref_scores(&c, g, rp, 1);
        for (p, w) in want.iter().enumerate().skip(1) {
            let got = f64::from(gs[g * stride + p]);
            worst = worst.max((got - w).abs() / w.abs().max(1.0));
        }
        let rows_g: Vec<&[i32]> = (0..R).map(|_| table.as_slice()).collect();
        let lens_g = &c.lens[g * R..(g + 1) * R];
        let gs_g = &gs[g * stride..(g + 1) * stride];
        let cmp = r::compact(&rows_g, lens_g, gs_g, GEOM, rb, c.row_stride, stride, 8192);
        for (rr, (want_t, want_l)) in cmp.tables.iter().zip(&cmp.lens).enumerate() {
            let row = g * R + rr;
            let got_t = &ob[row * c.row_stride..row * c.row_stride + want_t.len()];
            if ol[row] != *want_l || got_t != want_t.as_slice() {
                table_bad += 1;
            }
        }
    }
    println!(
        "  [{len} eb{eb}] scores: worst relative error {worst:.2e}; View rows mismatched: {table_bad}"
    );
    ok &= worst < 1e-4 && table_bad == 0;

    // Rebuild: invalidate every summary, score again, expect the same scores and trusted records.
    let (mo, ml) = l.meta_region();
    gpu.memset(sb.base.offset(mo as usize), 0, ml as usize)?;
    launch::launch_score(
        gpu,
        k,
        &lb,
        &rows(&c),
        c.q,
        (NQ * HD) as u32,
        NQ as u32,
        scores,
        &b,
        0,
        s,
    )?;
    gpu.synchronize(s)?;
    let gs2 = as_f32(&down(gpu, scores, GROUPS * stride * 4)?);
    let rp = r::recent_page(len, 512, 64);
    let same = (0..GROUPS).all(|g| (1..rp).all(|p| gs2[g * stride + p] == gs[g * stride + p]));
    // Only the candidates [sink, rp) are rebuilt; the sink and recent pages stay invalid.
    let rebuilt = check_summaries(gpu, &c, &l, &sb, 1, rp)?;
    println!(
        "  [{len} eb{eb}] rebuild: scores identical {same}; records not trusted after rebuild: {rebuilt}"
    );
    ok &= same && rebuilt == 0;

    if eb == 2 {
        // Identity gate: the budget covers every page.
        let wide = Budget {
            topk_pages: 1 << 20,
            ..b
        };
        launch::launch_compact(
            gpu,
            k,
            BS as u32,
            PB as u32,
            &rows(&c),
            scores,
            &view,
            &wide,
            false,
            s,
        )?;
        let o_dense = gpu.alloc(GROUPS * R * NQ * HD * 2)?;
        let o_view = gpu.alloc(GROUPS * R * NQ * HD * 2)?;
        let v_pool = c.pool; // values: reuse the key bytes (any finite BF16 data serves the gate)
        for (bt, sl, o) in [(c.bt, c.sl, o_dense), (out_bt, out_sl, o_view)] {
            ops::paged_decode_attn_bf16(
                gpu,
                paged,
                c.q,
                c.pool,
                v_pool,
                o,
                bt,
                sl,
                c.row_stride as u32,
                (GROUPS * R) as u32,
                NQ as u32,
                KVH as u32,
                HD as u32,
                BS as u32,
                1.0 / (HD as f32).sqrt(),
                (NQ * HD) as u32,
                0,
                s,
            )?;
        }
        gpu.synchronize(s)?;
        let a = down(gpu, o_dense, GROUPS * R * NQ * HD * 2)?;
        let bview = down(gpu, o_view, GROUPS * R * NQ * HD * 2)?;
        let tl = as_i32(&down(gpu, out_sl, GROUPS * R * 4)?);
        println!(
            "  [{len}] identity gate: lengths equal {}; attention bit-identical {}",
            tl == c.lens,
            a == bview
        );
        ok &= tl == c.lens && a == bview;
        for p in [o_dense, o_view] {
            gpu.free(p)?;
        }
    }
    for p in [scores, out_bt, out_sl, stats, c.pool, c.bt, c.sl, c.q] {
        gpu.free(p)?;
    }
    sb.free(gpu)?;
    Ok(ok)
}

fn bench(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    paged: spark_runtime::gpu::KernelHandle,
    iters: usize,
) -> Result<()> {
    let s = gpu.default_stream();
    let len = 131_072;
    let c = build(gpu, len, 2, 77)?;
    let l = layout(&c)?;
    let sb = PageSummaryBuffers::alloc(gpu, l.clone(), s)?;
    let lb = bufs(&c, &l, &sb);
    fold_all(gpu, k, &c, &lb, len, s)?;
    let b = Budget {
        sink_pages: 1,
        recent_tokens: 2048,
        topk_pages: 32,
        min_ctx: 32_768,
    };
    let stride = launch::score_stride(c.row_stride as u32, PB as u32) as usize;
    let scores = gpu.alloc(GROUPS * stride * 4)?;
    let out_bt = gpu.alloc(GROUPS * R * c.row_stride * 4)?;
    let out_sl = gpu.alloc(GROUPS * R * 4)?;
    let view = ViewOut {
        block_table: out_bt,
        seq_lens: out_sl,
        stats: DevicePtr::NULL,
    };
    let o = gpu.alloc(GROUPS * R * NQ * HD * 2)?;
    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        f()?;
        gpu.synchronize(s)?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f()?;
        }
        gpu.synchronize(s)?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
    };
    let select = time(&|| {
        launch::launch_score(
            gpu,
            k,
            &lb,
            &rows(&c),
            c.q,
            (NQ * HD) as u32,
            NQ as u32,
            scores,
            &b,
            0,
            s,
        )?;
        launch::launch_compact(
            gpu,
            k,
            BS as u32,
            PB as u32,
            &rows(&c),
            scores,
            &view,
            &b,
            false,
            s,
        )
    })?;
    let attn = |bt, sl| {
        ops::paged_decode_attn_bf16(
            gpu,
            paged,
            c.q,
            c.pool,
            c.pool,
            o,
            bt,
            sl,
            c.row_stride as u32,
            (GROUPS * R) as u32,
            NQ as u32,
            KVH as u32,
            HD as u32,
            BS as u32,
            1.0 / (HD as f32).sqrt(),
            (NQ * HD) as u32,
            0,
            s,
        )
    };
    let dense = time(&|| attn(c.bt, c.sl))?;
    let sparse = time(&|| attn(out_bt, out_sl))?;
    println!(
        "BENCH 128K x {GROUPS} seqs x {R} rows, one layer: select {select:.3} ms, dense attention {dense:.3} ms, \
         sparse attention {sparse:.3} ms (non-split kernel)"
    );
    for p in [scores, out_bt, out_sl, o, c.pool, c.bt, c.sl, c.q] {
        gpu.free(p)?;
    }
    sb.free(gpu)?;
    Ok(())
}

fn main() -> Result<()> {
    let mut lens: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    if lens.is_empty() {
        lens = vec![3000, 5000, 20_011];
    }
    println!(
        "=== page_sparse_microtest: nq={NQ} kvh={KVH} hd={HD} block={BS} page={} rows={R} groups={GROUPS} ===",
        BS * PB
    );
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let Some(k) = Kernels::resolve(gpu) else {
        println!(
            "RESULT: FAIL (module {} is not in this image; build with AVAROK_TARGET_HW=ada)",
            launch::MODULE
        );
        std::process::exit(1);
    };
    let paged = gpu.kernel("paged_decode", "paged_decode_attn")?;
    let mut ok = true;
    for &len in &lens {
        for eb in [2, 1] {
            ok &= run_case(gpu, &k, paged, len, eb)?;
        }
    }
    if let Some(n) = std::env::var("AVAROK_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        bench(gpu, &k, paged, n)?;
    }
    println!("RESULT: {}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
