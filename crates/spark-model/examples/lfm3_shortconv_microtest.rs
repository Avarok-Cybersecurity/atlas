// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: GPU qualification of the LFM3 gated short-convolution
//! kernels (`kernels/ada/common/lfm3_shortconv_ada.cu`) at the LFM3-1.7B
//! width (2048 channels), against the CPU reference
//! (`spark_model::port::lfm3_conv::reference`). Every check is EXACT
//! equality of BF16 bits: the kernels and the reference perform the same
//! un-fused f32 operations in the same order.
//!
//! Checks, per prompt length T (default 1 2 5 6 7 8 63 64 65 4096):
//!   * one-shot prefill (fresh and warm windows): y and the final window;
//!   * the same chunk in two pieces (split at 5 and at T / 2): equal to one shot;
//!   * three decode tokens after the prefill: equal to prefilling T + 3;
//!   * a sentinel tail after y, which no launch may write.
//! Then packed prefill of seven sequences (boundaries inside and on tile
//! edges, a zero-length and two shorter-than-window sequences) through a
//! device pointer table, and decode of 1, 8, 16 and 20 sequences (two
//! launches) with distinct windows.
//! `AVAROK_BENCH_ITERS=N` adds microseconds per call: decode at 1, 8 and 16
//! sequences, prefill (both launches) at 4096 tokens.
//!
//! Launches go through `port::lfm3_conv::launch`, the module the layer uses.
//!
//! Usage: cargo run --release -p spark-model --example lfm3_shortconv_microtest \
//!          --features cuda,gpu-examples -- [T ...]

use anyhow::Result;
use spark_model::port::lfm3_conv::launch::{self, ConvKernels, SeqLayout, Strides};
use spark_model::port::lfm3_conv::reference as r;
use spark_model::port::lfm3_conv::{CONV_WIDTH, STATE_LEN};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const DIM: usize = 2048;
const SENTINEL: u16 = 0x7F7F;
const SENTINEL_ROWS: usize = 3;

fn up<T: Copy>(gpu: &dyn GpuBackend, v: &[T]) -> Result<DevicePtr> {
    let bytes = std::mem::size_of_val(v).max(16);
    let p = gpu.alloc(bytes)?;
    // SAFETY: a plain-old-data slice viewed as bytes for the upload.
    let raw =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    if !raw.is_empty() {
        gpu.copy_h2d(raw, p)?;
    }
    Ok(p)
}

fn down(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Index and count of differing BF16 values.
fn diff(got: &[u16], want: &[u16]) -> Option<(usize, usize)> {
    let first = got.iter().zip(want).position(|(a, b)| a != b)?;
    Some((first, got.iter().zip(want).filter(|(a, b)| a != b).count()))
}

fn report(name: &str, got: &[u16], want: &[u16]) -> bool {
    match diff(got, want) {
        None if got.len() == want.len() => {
            println!("  {name:<34} exact ({} values)", want.len());
            true
        }
        None => {
            println!("  {name:<34} FAIL length {} vs {}", got.len(), want.len());
            false
        }
        Some((at, n)) => {
            println!(
                "  {name:<34} FAIL {n} differ, first at {at}: got 0x{:04x} want 0x{:04x}",
                got[at], want[at]
            );
            false
        }
    }
}

/// Prefill `t` rows of `bcx` (already on the device at `d_bcx`, row offset
/// `row0`) for the window at `d_state`, writing y at row `row0` of `d_y`.
#[allow(clippy::too_many_arguments)]
fn chunk(
    gpu: &dyn GpuBackend,
    k: &ConvKernels,
    d_bcx: DevicePtr,
    d_w: DevicePtr,
    d_y: DevicePtr,
    d_state: DevicePtr,
    row0: usize,
    t: usize,
    stream: u64,
) -> Result<()> {
    launch::prefill_chunk(
        gpu,
        k,
        d_bcx.offset(row0 * 3 * DIM * 2),
        d_w,
        d_y.offset(row0 * DIM * 2),
        SeqLayout::Single(d_state),
        t as u32,
        Strides::dense(DIM as u32),
        stream,
    )
}

fn check_len(
    gpu: &dyn GpuBackend,
    k: &ConvKernels,
    t: usize,
    warm: bool,
    stream: u64,
) -> Result<bool> {
    let extra = 3;
    let p = r::generate(
        0x1F3_0000 + t as u64 * 2 + warm as u64,
        t + extra,
        DIM,
        warm,
    );
    let mut s_ref = p.state.clone();
    let y_ref_all = r::prefill(&p.bcx, &p.weight, &mut s_ref, t + extra, DIM);
    let mut s_ref_t = p.state.clone();
    let y_ref_t = r::prefill(&p.bcx[..t * 3 * DIM], &p.weight, &mut s_ref_t, t, DIM);

    let d_bcx = up(gpu, &p.bcx)?;
    let d_w = up(gpu, &p.weight)?;
    let y_init = vec![SENTINEL; (t + extra + SENTINEL_ROWS) * DIM];
    let d_y = up(gpu, &y_init)?;
    let mut pass = true;
    println!("--- T={t} warm={warm}");

    // One shot.
    let d_s = up(gpu, &p.state)?;
    chunk(gpu, k, d_bcx, d_w, d_y, d_s, 0, t, stream)?;
    gpu.synchronize(stream)?;
    let y = down(gpu, d_y, y_init.len())?;
    pass &= report("prefill y", &y[..t * DIM], &y_ref_t);
    pass &= report(
        "prefill window",
        &down(gpu, d_s, DIM * STATE_LEN)?,
        &s_ref_t,
    );
    pass &= report("sentinel tail", &y[t * DIM..], &y_init[t * DIM..]);

    // Three decode tokens continue the window.
    let states = [d_s];
    for i in t..t + extra {
        launch::decode(
            gpu,
            k,
            d_bcx.offset(i * 3 * DIM * 2),
            d_w,
            d_y.offset(i * DIM * 2),
            Strides::dense(DIM as u32),
            &states,
            stream,
        )?;
    }
    gpu.synchronize(stream)?;
    let y = down(gpu, d_y, y_init.len())?;
    pass &= report(
        "decode after prefill y",
        &y[..(t + extra) * DIM],
        &y_ref_all,
    );
    pass &= report(
        "decode after prefill window",
        &down(gpu, d_s, DIM * STATE_LEN)?,
        &s_ref,
    );
    pass &= report(
        "decode sentinel tail",
        &y[(t + extra) * DIM..],
        &y_init[(t + extra) * DIM..],
    );

    // Two-piece chunks equal one shot.
    for split in [5usize, t / 2] {
        if split == 0 || split >= t {
            continue;
        }
        let d_s2 = up(gpu, &p.state)?;
        let d_y2 = up(gpu, &y_init)?;
        chunk(gpu, k, d_bcx, d_w, d_y2, d_s2, 0, split, stream)?;
        chunk(gpu, k, d_bcx, d_w, d_y2, d_s2, split, t - split, stream)?;
        gpu.synchronize(stream)?;
        let y2 = down(gpu, d_y2, t * DIM)?;
        pass &= report(&format!("chunked at {split} y"), &y2, &y_ref_t);
        pass &= report(
            &format!("chunked at {split} window"),
            &down(gpu, d_s2, DIM * STATE_LEN)?,
            &s_ref_t,
        );
        gpu.free(d_s2).ok();
        gpu.free(d_y2).ok();
    }
    for x in [d_bcx, d_w, d_y, d_s] {
        gpu.free(x).ok();
    }
    Ok(pass)
}

fn check_packed(gpu: &dyn GpuBackend, k: &ConvKernels, stream: u64) -> Result<bool> {
    let lens = [3usize, 61, 0, 64, 2, 130, 6];
    let mut cu = vec![0i32];
    for l in lens {
        cu.push(cu.last().copied().unwrap_or(0) + l as i32);
    }
    let total = *cu.last().unwrap_or(&0) as usize;
    let p = r::generate(0x1F3_9000, total, DIM, false);
    let mut rng = r::Rng::new(0x1F3_9001);
    let init: Vec<Vec<u16>> = lens
        .iter()
        .map(|_| rng.bf16_vec(DIM * STATE_LEN, 2.0))
        .collect();
    let mut s_ref = init.clone();
    let cu_usize: Vec<usize> = cu.iter().map(|&v| v as usize).collect();
    let y_ref = r::prefill_packed(&p.bcx, &p.weight, &cu_usize, &mut s_ref, DIM);

    let d_bcx = up(gpu, &p.bcx)?;
    let d_w = up(gpu, &p.weight)?;
    let d_y = up(gpu, &vec![SENTINEL; total * DIM])?;
    let d_states = init
        .iter()
        .map(|s| up(gpu, s))
        .collect::<Result<Vec<_>>>()?;
    let table: Vec<u64> = d_states.iter().map(|d| d.0).collect();
    let d_table = up(gpu, &table)?;
    let d_cu = up(gpu, &cu)?;
    let layout = SeqLayout::Packed {
        cu_seqlens: d_cu,
        state_table: d_table,
        num_seqs: lens.len() as u32,
    };
    launch::prefill_chunk(
        gpu,
        k,
        d_bcx,
        d_w,
        d_y,
        layout,
        total as u32,
        Strides::dense(DIM as u32),
        stream,
    )?;
    gpu.synchronize(stream)?;
    println!("--- packed {lens:?}");
    let mut pass = report("packed y", &down(gpu, d_y, total * DIM)?, &y_ref);
    for (s, d) in d_states.iter().enumerate() {
        pass &= report(
            &format!("packed window {s}"),
            &down(gpu, *d, DIM * STATE_LEN)?,
            &s_ref[s],
        );
    }
    for x in [d_bcx, d_w, d_y, d_table, d_cu].into_iter().chain(d_states) {
        gpu.free(x).ok();
    }
    Ok(pass)
}

fn check_decode_batch(
    gpu: &dyn GpuBackend,
    k: &ConvKernels,
    n: usize,
    stream: u64,
) -> Result<bool> {
    let p = r::generate(0x1F3_A000 + n as u64, n, DIM, false);
    let mut rng = r::Rng::new(0x1F3_A100 + n as u64);
    let init: Vec<Vec<u16>> = (0..n).map(|_| rng.bf16_vec(DIM * STATE_LEN, 2.0)).collect();
    let d_bcx = up(gpu, &p.bcx)?;
    let d_w = up(gpu, &p.weight)?;
    let d_y = up(gpu, &vec![SENTINEL; n * DIM])?;
    let d_states = init
        .iter()
        .map(|s| up(gpu, s))
        .collect::<Result<Vec<_>>>()?;
    launch::decode(
        gpu,
        k,
        d_bcx,
        d_w,
        d_y,
        Strides::dense(DIM as u32),
        &d_states,
        stream,
    )?;
    gpu.synchronize(stream)?;
    println!("--- decode batch n={n}");
    let y = down(gpu, d_y, n * DIM)?;
    let mut pass = true;
    for i in 0..n {
        let mut s = init[i].clone();
        let want = r::decode_step(
            &p.bcx[i * 3 * DIM..(i + 1) * 3 * DIM],
            &p.weight,
            &mut s,
            DIM,
        );
        let ok_y = y[i * DIM..(i + 1) * DIM] == want[..];
        let ok_s = down(gpu, d_states[i], DIM * STATE_LEN)? == s;
        if !(ok_y && ok_s) {
            println!("  row {i}: FAIL (y exact={ok_y}, window exact={ok_s})");
        }
        pass &= ok_y && ok_s;
    }
    if pass {
        println!("  {n} rows exact (y and window)");
    }
    for x in [d_bcx, d_w, d_y].into_iter().chain(d_states) {
        gpu.free(x).ok();
    }
    Ok(pass)
}

/// Microseconds per call (host wall time over `iters` launches, one sync).
fn bench(gpu: &dyn GpuBackend, k: &ConvKernels, iters: usize, stream: u64) -> Result<()> {
    let p = r::generate(0x1F3_B000, 4096, DIM, false);
    let d_bcx = up(gpu, &p.bcx)?;
    let d_w = up(gpu, &p.weight)?;
    let d_y = gpu.alloc(4096 * DIM * 2)?;
    // `generate(.., warm = false)` gives a zero window.
    let d_states = (0..16)
        .map(|_| up(gpu, &p.state))
        .collect::<Result<Vec<_>>>()?;
    for n in [1usize, 8, 16] {
        launch::decode(
            gpu,
            k,
            d_bcx,
            d_w,
            d_y,
            Strides::dense(DIM as u32),
            &d_states[..n],
            stream,
        )?;
        gpu.synchronize(stream)?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            launch::decode(
                gpu,
                k,
                d_bcx,
                d_w,
                d_y,
                Strides::dense(DIM as u32),
                &d_states[..n],
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("  BENCH decode seqs={n:<3} {us:9.2} us/call");
    }
    let layout = SeqLayout::Single(d_states[0]);
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        launch::prefill_chunk(
            gpu,
            k,
            d_bcx,
            d_w,
            d_y,
            layout,
            4096,
            Strides::dense(DIM as u32),
            stream,
        )?;
    }
    gpu.synchronize(stream)?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let gbs = (4096.0 * DIM as f64 * 4.0 * 2.0) / (us * 1e3);
    println!("  BENCH prefill T=4096      {us:9.2} us/call ({gbs:.1} GB/s of BCx + y)");
    for x in [d_bcx, d_w, d_y].into_iter().chain(d_states) {
        gpu.free(x).ok();
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut lens: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    if lens.is_empty() {
        lens = vec![1, 2, 5, 6, 7, 8, 63, 64, 65, 4096];
    }
    println!(
        "=== lfm3_shortconv microtest: dim={DIM} width={CONV_WIDTH} window={STATE_LEN}; exact BF16 ==="
    );
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    if !gpu.has_module(launch::MODULE) {
        println!(
            "RESULT: FAIL (module {} is not in this image; build with AVAROK_TARGET_HW=ada)",
            launch::MODULE
        );
        std::process::exit(2);
    }
    let k = ConvKernels::resolve(gpu)?;
    let stream = gpu.create_stream()?;
    let mut pass = true;
    for &t in &lens {
        for warm in [false, true] {
            pass &= check_len(gpu, &k, t.max(1), warm, stream)?;
        }
    }
    pass &= check_packed(gpu, &k, stream)?;
    for n in [1usize, 8, 16, 20] {
        pass &= check_decode_batch(gpu, &k, n, stream)?;
    }
    let iters: usize = std::env::var("AVAROK_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if iters > 0 {
        bench(gpu, &k, iters, stream)?;
    }
    if pass {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}
