// SPDX-License-Identifier: AGPL-3.0-only
//! WS1/P3b — tile geometry of the GLM routed-MoE grouped W4A16 GEMM, measured.
//!
//! The kernel `moe_w4a16_grouped_gemm_ptrtable` is 64.04 % of GLM-5.3-Flash's rows=256
//! prefill window and runs at **66.8 GB/s = 24.5 % of the GB10's 273 GB/s roofline**
//! (`runs/ws1-p2-prefill-n1n2/PROFILE.md`). This harness reproduces that number on the
//! REAL production shape without a serve, and prices each tile-geometry variant against
//! it on the same allocation, in the same process.
//!
//! Shape, from `kernels/gb10/glm-5.3-flash/MODEL.toml` + EP=2:
//!   288 global experts, of which **144 are local** (the other 144 carry a NULL weight
//!   pointer, exactly as a remote EP rank does); `hidden = 4096`, `moe_intermediate = 2048`;
//!   `top_k = 8`; gate/up are `N=2048, K=4096` and down is `N=4096, K=2048`.
//!
//! 🔴 The bar is BYTE-IDENTITY, not an error bar. Every variant stages the same BF16
//! operand and walks the same 16-wide mma fragments in the same k order, so its output
//! must equal the base kernel's bit for bit. The harness asserts that and reports the
//! first differing element if it does not — a variant that is fast and different is a
//! failure here, because the production claim is `sha8 d44c9251` unchanged.
//!
//! 🪤 Bytes moved is counted from the experts that are BOTH local and non-empty, because
//! a null-pointer CTA returns before reading a weight byte and an empty expert never
//! launches an M tile. Counting all 288 would overstate the achieved bandwidth 2x.
//!
//!   cargo run -p spark-model --release --example glm5next_moe_grouped_tile_bench \
//!       --features cuda,gpu-examples -- [rows]

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

mod device;
mod variants;

use device::{dn_i32, dn_raw, launch, lcg, up, up_f32, up_i32, up_u64};
use variants::VARIANTS;

const NUM_EXPERTS: usize = 288;
const TOP_K: usize = 8;
const HIDDEN: usize = 4096;
const MOE_INTER: usize = 2048;
/// EP=2 — every second global expert lives on the other rank and its pointer is NULL.
const EP: usize = 2;
const GROUP_SIZE: usize = 16;

const WARMUP: usize = 2;
const ITERS: usize = 10;

fn main() -> Result<()> {
    let rows: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);

    // 🪤 `ptx_modules()` is target t0 = **deepseek-v4-flash**, which SHADOWS
    // `common/moe_w4a16_grouped_gemm.cu` with its own copy. Loading it here would
    // benchmark DeepSeek's kernel, built with DeepSeek's nvcc flags, and would not
    // resolve a single kernel added to `common/`. GLM's own set is the one that
    // carries `--fmad=false` and the shared file.
    let set = avarok_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .ok_or_else(|| anyhow::anyhow!("no (glm-5.3-flash, nvfp4) PTX set in this build"))?;
    println!(
        "PTX set: ({}, {}, {})  {} modules",
        set.target.arch,
        set.target.model,
        set.target.quant,
        set.modules.len()
    );
    let g = AvarokCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &g;
    let k_sort: KernelHandle = gpu.kernel("moe", "moe_sort_by_expert")?;

    let mut s = 0xA713_5EED_u64;
    let te = rows * TOP_K;

    // ── routing: top-k is a SET per row, uniform over all 288 global experts ──
    let mut ids: Vec<u32> = Vec::with_capacity(te);
    for _ in 0..rows {
        let mut picked: Vec<u32> = Vec::with_capacity(TOP_K);
        while picked.len() < TOP_K {
            let e = (lcg(&mut s) as usize % NUM_EXPERTS) as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }

    // ── weights: expert e is LOCAL iff e % EP == 0, else a NULL pointer (remote rank) ──
    // Both projection shapes share one allocation per expert sized for the larger of the
    // two, so the 679.5 MB the profile measured is on the device exactly once.
    let max_bytes_packed = MOE_INTER.max(HIDDEN) * HIDDEN.max(MOE_INTER) / 2;
    let max_bytes_scale = MOE_INTER.max(HIDDEN) * (HIDDEN.max(MOE_INTER) / GROUP_SIZE);
    let mut packed_ptrs = vec![0u64; NUM_EXPERTS];
    let mut scale_ptrs = vec![0u64; NUM_EXPERTS];
    let mut scale2 = vec![0f32; NUM_EXPERTS];
    let mut n_local = 0usize;
    // 🔴 ONE arena, not 144 `alloc`s. The first cut of this harness gave every local expert
    // its own allocation and measured the BASE kernel at 43.8 GB/s where the production
    // profile measures 66.8 — a 1.5x harness artefact that would have mispriced every
    // variant. The weight loader hands the ptr table offsets into a contiguous arena, so the
    // harness does too; `GLM_TILE_BENCH_SEPARATE_ALLOC=1` restores the per-expert form to
    // re-measure the gap.
    let separate = std::env::var("GLM_TILE_BENCH_SEPARATE_ALLOC").as_deref() == Ok("1");
    let n_local_total = NUM_EXPERTS.div_ceil(EP);
    let arena_p = if separate {
        DevicePtr(0)
    } else {
        gpu.alloc(n_local_total * max_bytes_packed)?
    };
    let arena_s = if separate {
        DevicePtr(0)
    } else {
        gpu.alloc(n_local_total * max_bytes_scale)?
    };
    {
        let mut pbuf = vec![0u8; max_bytes_packed];
        let mut sbuf = vec![0u8; max_bytes_scale];
        for b in pbuf.iter_mut() {
            *b = lcg(&mut s) as u8;
        }
        for b in sbuf.iter_mut() {
            // E4M3 codes 0x30..0x48 — ~0.25..4.0, no zeros, no NaN.
            *b = 0x30 + (lcg(&mut s) % 0x18) as u8;
        }
        for e in 0..NUM_EXPERTS {
            if e % EP != 0 {
                continue; // remote — NULL pointer, exactly as EP does
            }
            // Perturb so no two experts hold byte-identical weights.
            pbuf[e] ^= 0x5A;
            sbuf[e % max_bytes_scale] = 0x30 + (e % 0x18) as u8;
            if separate {
                packed_ptrs[e] = up(gpu, &pbuf)?.0;
                scale_ptrs[e] = up(gpu, &sbuf)?.0;
            } else {
                let pp = arena_p.offset(n_local * max_bytes_packed);
                let sp = arena_s.offset(n_local * max_bytes_scale);
                gpu.copy_h2d(&pbuf, pp)?;
                gpu.copy_h2d(&sbuf, sp)?;
                packed_ptrs[e] = pp.0;
                scale_ptrs[e] = sp.0;
            }
            n_local += 1;
            scale2[e] = 0.5 + (e % 64) as f32 / 64.0;
        }
    }
    println!(
        "weights: {} local experts, {} arena",
        n_local,
        if separate {
            "SEPARATE allocs"
        } else {
            "ONE contiguous"
        }
    );
    let d_packed_ptrs = up_u64(gpu, &packed_ptrs)?;
    let d_scale_ptrs = up_u64(gpu, &scale_ptrs)?;
    let d_scale2 = up_f32(gpu, &scale2)?;

    // ── activations ──
    let a_bytes: Vec<u8> = (0..rows.max(te) * HIDDEN.max(MOE_INTER))
        .flat_map(|_| {
            // BF16 in [-1, 1): exponent clamped so no inf/NaN reaches the mma.
            let m = (lcg(&mut s) % 0x8000) as u16;
            (0x3000u16 | (m & 0x0FFF) | ((lcg(&mut s) as u16 & 1) << 15)).to_le_bytes()
        })
        .collect();
    let d_a = up(gpu, &a_bytes)?;

    // ── device sort ──
    let d_ids = up_i32(gpu, &ids.iter().map(|x| *x as i32).collect::<Vec<_>>())?;
    let d_stid = gpu.alloc(te * 4)?;
    let d_seid = gpu.alloc(te * 4)?;
    let d_off = gpu.alloc((NUM_EXPERTS + 1) * 4)?;
    let d_t2p = gpu.alloc(te * 4)?;
    KernelLaunch::new(gpu, k_sort)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(d_ids)
        .arg_ptr(d_stid)
        .arg_ptr(d_seid)
        .arg_ptr(d_off)
        .arg_ptr(d_t2p)
        .arg_u32(te as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(TOP_K as u32)
        .launch(0)?;
    gpu.synchronize(0)?;
    let off = dn_i32(gpu, d_off, NUM_EXPERTS + 1)?;

    let counts: Vec<i32> = (0..NUM_EXPERTS).map(|e| off[e + 1] - off[e]).collect();
    let busiest = counts.iter().copied().max().unwrap_or(0) as u32;
    let swept: usize = (0..NUM_EXPERTS)
        .filter(|e| e % EP == 0 && counts[*e] > 0)
        .count();
    let mean = te as f64 / NUM_EXPERTS as f64;

    println!(
        "rows={rows} top_k={TOP_K} slots={te} experts={NUM_EXPERTS} local={n_local} \
         swept(local & non-empty)={swept}  mean rows/expert {mean:.2}  busiest {busiest}"
    );

    // ── the two production projections ──
    for (label, n_out, kk, gather) in [
        ("gate/up  N=2048 K=4096", MOE_INTER, HIDDEN, true),
        ("down     N=4096 K=2048", HIDDEN, MOE_INTER, false),
    ] {
        // Weight bytes ONE sweep of the swept experts costs: NVFP4 is 0.5 B packed plus
        // 1/16 B of E4M3 block scale per element.
        let bytes = swept as f64 * (n_out * kk) as f64 * (0.5 + 1.0 / 16.0);
        let stid = if gather { d_stid } else { DevicePtr(0) };
        let c_bytes = te * n_out * 2;
        let d_c = gpu.alloc(c_bytes)?;

        println!("\n  {label}   one sweep = {:.1} MB", bytes / 1.0e6);
        println!(
            "  {:<18} {:>8} {:>10} {:>9} {:>8}  identity",
            "variant", "ms", "GB/s", "% of 273", "vs base"
        );

        // ── CONTROL: what a pure coalesced stream of these exact bytes costs ──
        // 🔴 Run FIRST, so every "% of roofline" below has a measured denominator beside
        // the datasheet one. See the kernel's own comment for why this is not optional.
        if let Ok(kp) = gpu.kernel("moe_w4a16", "moe_w4a16_grouped_stream_probe") {
            let pb = (n_out * kk / 2) as u32;
            let sb = (n_out * kk / GROUP_SIZE) as u32;
            let d_sink = gpu.alloc(NUM_EXPERTS * 4)?;
            let gx = (pb / 16).div_ceil(256 * 8).max(1);
            let go = |_: usize| -> Result<()> {
                KernelLaunch::new(gpu, kp)
                    .grid([gx, 1, NUM_EXPERTS as u32])
                    .block([256, 1, 1])
                    .arg_ptr(d_packed_ptrs)
                    .arg_ptr(d_scale_ptrs)
                    .arg_ptr(d_sink)
                    .arg_u32(NUM_EXPERTS as u32)
                    .arg_u32(pb)
                    .arg_u32(sb)
                    .launch(0)
            };
            for i in 0..WARMUP {
                go(i)?;
            }
            gpu.synchronize(0)?;
            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                go(i)?;
            }
            gpu.synchronize(0)?;
            let ms = t0.elapsed().as_secs_f64() * 1.0e3 / ITERS as f64;
            let gbs = bytes / (ms * 1.0e6);
            println!(
                "  {:<18} {ms:>8.3} {gbs:>10.1} {:>8.1}%          MEASURED CEILING (no dequant, no mma)",
                "STREAM control",
                100.0 * gbs / 273.0
            );
        }

        let mut base_out: Option<Vec<u8>> = None;
        let mut base_ms = 0f64;
        for v in VARIANTS {
            let k = match gpu.kernel("moe_w4a16", v.kernel) {
                Ok(k) => k,
                Err(e) => {
                    println!("  {:<18} UNRESOLVED: {e}", v.name);
                    continue;
                }
            };
            let max_m_tiles = busiest.div_ceil(v.m_tile).max(1);

            gpu.memset_async(d_c, 0, c_bytes, 0)?;
            for _ in 0..WARMUP {
                launch(
                    gpu,
                    k,
                    d_a,
                    d_packed_ptrs,
                    d_scale_ptrs,
                    d_scale2,
                    d_c,
                    d_off,
                    stid,
                    n_out,
                    kk,
                    max_m_tiles,
                    v.n_tile,
                    v.threads,
                )?;
            }
            gpu.synchronize(0)?;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                launch(
                    gpu,
                    k,
                    d_a,
                    d_packed_ptrs,
                    d_scale_ptrs,
                    d_scale2,
                    d_c,
                    d_off,
                    stid,
                    n_out,
                    kk,
                    max_m_tiles,
                    v.n_tile,
                    v.threads,
                )?;
            }
            gpu.synchronize(0)?;
            let ms = t0.elapsed().as_secs_f64() * 1.0e3 / ITERS as f64;
            let gbs = bytes / (ms * 1.0e6);

            let out = dn_raw(gpu, d_c, c_bytes)?;
            let identity = match &base_out {
                None => {
                    base_out = Some(out);
                    base_ms = ms;
                    "—".to_string()
                }
                Some(b) => match b.iter().zip(&out).position(|(x, y)| x != y) {
                    None => "BYTE-IDENTICAL".to_string(),
                    Some(i) => format!(
                        "🔴 DIFFERS at elem {} (row {}, n {}): base {:02x?} got {:02x?}",
                        i / 2,
                        i / 2 / n_out,
                        (i / 2) % n_out,
                        &b[i & !1..(i & !1) + 2],
                        &out[i & !1..(i & !1) + 2]
                    ),
                },
            };
            println!(
                "  {:<18} {ms:>8.3} {gbs:>10.1} {:>8.1}% {:>7.2}x  {identity}",
                v.name,
                100.0 * gbs / 273.0,
                if base_ms > 0.0 { base_ms / ms } else { 1.0 }
            );
        }
        if base_out.is_none() {
            bail!("the base kernel did not resolve — nothing to compare against");
        }
    }

    Ok(())
}
