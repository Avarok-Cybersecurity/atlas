// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + timings for the window desk (port plan lane L36, row `kv_window_desk`).
//!
//! WHAT THIS PINS (each row prints PASS/FAIL; any FAIL exits non-zero):
//!  * **The view kernel** `kv_window_view_compact` (`kernels/ada/common/kv_page_fetch_ada.cu`)
//!    against its host oracle `port::kv_window::selector::compact_row`: every table entry up to
//!    the new length and every length equal, on random tables with hidden blocks, verify-style
//!    rows (lengths one apart), padding rows (length 0), the identity (hole = -1) and a table
//!    whose rows are all hidden but their last block.
//!  * **The page API round trip** (`spark_runtime::kv_host_tier::pages`) on a real paged KV
//!    cache of the Qwen3.8-27B attention shape (16 layers, 4 KV heads, head dim 256, FP8,
//!    blocks of 16): demote N blocks to registered host slabs, clobber them, fetch them into
//!    fresh blocks, every byte equal.
//!
//! TIMINGS: the demote (D2H) and fetch (H2D) GB/s of that round trip, device time from the page
//! API's timing events, for the receipt R9 of report 77 section 12.5 (restore at least 10 GB/s).
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example kv_window_view_microtest --features cuda,gpu-examples [-- BLOCKS]
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use spark_model::port::kv_window::selector::{MODULE, VIEW_KERNEL, compact_row};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{GpuBackend, KernelArg, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::kv_host_tier::pages::{PageApi, PageStats, SlabArena};
use spark_runtime::kv_host_tier::{PageDeps, RestoreState};

const BS: u32 = 16;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

fn to_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_bytes(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One kernel case: `rows` tables of `mb` entries with lengths `lens`, hole `hole` (-1 = none).
fn view_case(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    name: &str,
    tables: &[Vec<u32>],
    lens: &[u32],
    mb: usize,
    hole: i32,
) -> Result<usize> {
    let rows = tables.len();
    let mut flat = vec![0i32; rows * mb];
    for (r, t) in tables.iter().enumerate() {
        for (j, &b) in t.iter().enumerate() {
            flat[r * mb + j] = b as i32;
        }
    }
    let lens_i: Vec<i32> = lens.iter().map(|&l| l as i32).collect();
    let alloc = |bytes: usize| gpu.alloc(bytes.max(4));
    let (d_in, d_lens, d_out, d_olens, d_hole) = (
        alloc(rows * mb * 4)?,
        alloc(rows * 4)?,
        alloc(rows * mb * 4)?,
        alloc(rows * 4)?,
        alloc(4)?,
    );
    gpu.copy_h2d(&to_bytes(&flat), d_in)?;
    gpu.copy_h2d(&to_bytes(&lens_i), d_lens)?;
    gpu.copy_h2d(&to_bytes(&vec![-7i32; rows * mb]), d_out)?;
    gpu.copy_h2d(&hole.to_le_bytes(), d_hole)?;
    let (n, m, b) = (
        (rows as u32).to_le_bytes(),
        (mb as u32).to_le_bytes(),
        BS.to_le_bytes(),
    );
    let stream = gpu.default_stream();
    gpu.launch_typed(
        k,
        [rows as u32, 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Buffer(d_in),
            KernelArg::Buffer(d_lens),
            KernelArg::Buffer(d_out),
            KernelArg::Buffer(d_olens),
            KernelArg::Buffer(d_hole),
            KernelArg::Bytes(&n),
            KernelArg::Bytes(&m),
            KernelArg::Bytes(&b),
        ],
    )?;
    gpu.synchronize(stream)?;
    let mut out = vec![0u8; rows * mb * 4];
    let mut olens = vec![0u8; rows * 4];
    gpu.copy_d2h(d_out, &mut out)?;
    gpu.copy_d2h(d_olens, &mut olens)?;
    let (out, olens) = (from_bytes(&out), from_bytes(&olens));
    let hole_u = (hole >= 0).then_some(hole as u32);
    let mut bad = 0usize;
    for (r, (t, &l)) in tables.iter().zip(lens).enumerate() {
        let (view, len) = compact_row(t, l, BS, hole_u);
        let got_len = olens[r];
        let n_read = (len.div_ceil(BS) as usize).min(view.len());
        let got: Vec<u32> = out[r * mb..r * mb + n_read]
            .iter()
            .map(|&x| x as u32)
            .collect();
        if got_len != len as i32 || got != view[..n_read] {
            bad += 1;
            if bad <= 3 {
                eprintln!(
                    "  {name} row {r}: len {got_len} want {len}, table {got:?} want {:?}",
                    &view[..n_read]
                );
            }
        }
    }
    for p in [d_in, d_lens, d_out, d_olens, d_hole] {
        gpu.free(p)?;
    }
    println!(
        "view {name:<28} rows {rows:>4} x {mb:>5}: {}",
        if bad == 0 { "PASS" } else { "FAIL" }
    );
    Ok(usize::from(bad != 0))
}

fn view_rows(gpu: &dyn GpuBackend, k: KernelHandle, rng: &mut Lcg) -> Result<usize> {
    let hole = 9_999i32;
    let mut fails = 0;
    for (name, rows, mb) in [
        ("random", 64usize, 300usize),
        ("long (262K at bs 16)", 8, 16_384),
    ] {
        let mut tables = Vec::new();
        let mut lens = Vec::new();
        for r in 0..rows {
            let blocks = 1 + (rng.next() as usize % mb);
            let mut t: Vec<u32> = (0..blocks as u32).map(|i| i * 7 + r as u32).collect();
            for e in t.iter_mut().take(blocks - 1) {
                if !rng.next().is_multiple_of(4) {
                    *e = hole as u32;
                }
            }
            t.resize(mb, 0);
            lens.push(if r % 9 == 8 {
                0
            } else {
                (blocks as u32 - 1) * BS + 1 + (rng.next() % 16) as u32
            });
            tables.push(t);
        }
        fails += view_case(gpu, k, name, &tables, &lens, mb, hole)?;
        fails += view_case(gpu, k, &format!("{name}, identity"), &tables, &lens, mb, -1)?;
    }
    // Verify rows: one table, lengths one apart.
    let t: Vec<u32> = (0..40)
        .map(|i| if (3..30).contains(&i) { hole as u32 } else { i })
        .collect();
    let tables = vec![t; 6];
    let lens: Vec<u32> = (0..6).map(|r| 39 * BS + 2 + r).collect();
    fails += view_case(gpu, k, "verify rows", &tables, &lens, 40, hole)?;
    Ok(fails)
}

fn page_round_trip(gpu: &AvarokCudaBackend, blocks: usize, rng: &mut Lcg) -> Result<usize> {
    let cfg = KvCacheConfig {
        block_size: BS as usize,
        num_kv_heads: 4,
        head_dim: 256,
        num_layers: 16,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let mut kv = PagedKvCache::new(cfg, 2 * blocks + 1, gpu)?;
    let src: Vec<u32> = (0..blocks)
        .map(|_| kv.alloc_block())
        .collect::<Result<_>>()?;
    let page_bytes = kv.block_regions(src[0]).bytes();
    let mut want = Vec::with_capacity(blocks);
    for &b in &src {
        let mut all = Vec::with_capacity(page_bytes);
        for &(p, n) in &kv.block_regions(b).regions {
            let bytes: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            gpu.copy_h2d(&bytes, p)?;
            all.extend_from_slice(&bytes);
        }
        want.push(all);
    }
    let arena = Arc::new(SlabArena::new(
        page_bytes,
        64 << 20,
        (blocks * page_bytes + (64 << 20)) as u64,
    ));
    ensure!(
        arena.reserve(gpu, blocks) >= blocks,
        "host slabs for {blocks} pages"
    );
    let api = PageApi::with_stats(
        PageDeps {
            arena: arena.clone(),
            block_bytes: page_bytes,
            copy_stream: gpu.create_copy_stream()?,
        },
        Arc::new(PageStats::default()),
    );
    let pages: Vec<u32> = (0..blocks as u32).collect();
    let regions: Vec<_> = src.iter().map(|&b| kv.block_regions(b)).collect();
    let stream = gpu.default_stream();
    let rep = api.demote(gpu, 1, &pages, &regions, stream)?;
    ensure!(
        rep.refused.is_empty(),
        "{} pages refused",
        rep.refused.len()
    );
    let t = rep.ticket.context("demote ticket")?;
    ensure!(api.wait(gpu, &t)? == RestoreState::Ready, "demote failed");
    api.release(t);
    for &b in &src {
        for &(p, n) in &kv.block_regions(b).regions {
            gpu.copy_h2d(&vec![0x5A; n], p)?;
        }
    }
    let dst: Vec<u32> = (0..blocks)
        .map(|_| kv.alloc_block())
        .collect::<Result<_>>()?;
    let dregions: Vec<_> = dst.iter().map(|&b| kv.block_regions(b)).collect();
    let t = api.fetch(gpu, 1, &pages, &dregions, stream)?;
    ensure!(api.wait(gpu, &t)? == RestoreState::Ready, "fetch failed");
    api.release(t);
    let mut bad = 0usize;
    for (i, &b) in dst.iter().enumerate() {
        let mut got = Vec::with_capacity(page_bytes);
        for &(p, n) in &kv.block_regions(b).regions {
            let mut buf = vec![0u8; n];
            gpu.copy_d2h(p, &mut buf)?;
            got.extend_from_slice(&buf);
        }
        if got != want[i] {
            bad += 1;
        }
    }
    let s = api.stats().snapshot();
    let gbs = |bytes: u64, secs: f64| {
        if secs > 0.0 {
            bytes as f64 / secs / 1e9
        } else {
            0.0
        }
    };
    println!(
        "pages {blocks} x {} KiB: bytes {} | demote {:.1} GB/s, fetch {:.1} GB/s (R9 bound >= 10 GB/s: {})",
        page_bytes >> 10,
        if bad == 0 { "PASS" } else { "FAIL" },
        gbs(s.demote_bytes, s.demote_seconds),
        gbs(s.fetch_bytes, s.fetch_seconds),
        if gbs(s.fetch_bytes, s.fetch_seconds) >= 10.0 {
            "PASS"
        } else {
            "FAIL"
        },
    );
    api.quiesce(gpu);
    arena.release_all(gpu)?;
    Ok(usize::from(bad != 0))
}

fn main() -> Result<()> {
    let blocks: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()
        .context("BLOCKS must be a whole number")?
        .unwrap_or(512);
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let k = gpu.kernel(MODULE, VIEW_KERNEL).with_context(|| {
        format!("{MODULE}::{VIEW_KERNEL} absent: build with AVAROK_TARGET_HW=ada")
    })?;
    let mut rng = Lcg(0x1336_2026_0d5c_0001);
    let mut fails = view_rows(&gpu, k, &mut rng)?;
    fails += page_round_trip(&gpu, blocks, &mut rng)?;
    ensure!(fails == 0, "{fails} row(s) failed");
    println!("all rows PASS");
    Ok(())
}
