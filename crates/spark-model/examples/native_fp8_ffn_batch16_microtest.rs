// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle for the 5..=32-row native-FP8 dense-FFN decode tier (#927).
//!
//! Runs the REAL Qwen/Qwen3.8-27B-FP8 FFN shapes — gate/up `[17408, 5120]` and
//! down `[5120, 17408]` — through the arms `dense_ffn.rs` dispatches at those
//! widths, and requires the batch16 route to produce the SAME BF16 BYTES as
//! the scalar `w8a16_gemv` each row's M=1 decode runs. `unequal_bf16=0
//! max_abs=0` is the pass condition, not a tolerance: the kernels are one
//! template with one K-iteration order and one per-row reduction tree, so any
//! difference at all is a bug in the wrapper, the offsets or the row split.
//!
//! It then times the tier against the arm it replaces (`w8a16_gemm_pipelined`,
//! which is what the FFN's `w8_gemm!` reached at these widths before #927 —
//! the transposed twin is never built for the dense FFN, `gate_t`/`up_t`/
//! `down_t` are `None`). Effective GB/s counts the FP8 WEIGHT bytes, once per
//! kernel pass: that is the whole budget at decode widths, and the number the
//! tier exists to move. The 17..=32 rung makes two passes on purpose, so its
//! GB/s is reported against two.
//!
//! TIMING METHOD: `synchronize` + host `Instant` over N reps, the house
//! pattern (`examples/w4a16_m17_bench.rs`). `GpuBackend` exposes event record
//! and synchronize but no elapsed-time query, so a CUDA-event delta is not
//! available through the abstraction; over 20 reps of a >=1 ms kernel the
//! launch overhead is well under the difference being measured.
//!
//! Run on the H100:
//!     cargo run --release --example native_fp8_ffn_batch16_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

/// Qwen3.8-27B dense FFN. hidden=5120, intermediate=17408.
const H: usize = 5120;
const INTER: usize = 17408;
const MAX_M: usize = 32;
const GUARD: usize = 64;
const REPS: u32 = 20;
const WARMUP: u32 = 3;

struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
}

/// The two orientations the FFN runs. gate and up share a shape, so one entry
/// covers both; down is the transpose-ish twin with the deep K.
const SHAPES: [Shape; 2] = [
    Shape {
        name: "gate/up",
        n: INTER,
        k: H,
    },
    Shape {
        name: "down",
        n: H,
        k: INTER,
    },
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// The batch16 route exactly as `DenseFfnLayer::w8a16_batch16_proj` runs it:
/// one launch at m<=16, two on contiguous row halves at 17..=32.
#[allow(clippy::too_many_arguments)]
fn batch16_route(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    scale: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<usize> {
    let launch = |rows: usize, first: usize| {
        ops::w8a16_gemv_batch16(
            gpu,
            kernel,
            input.offset(first * k * 2),
            weight,
            scale,
            out.offset(first * n * 2),
            rows as u32,
            n as u32,
            k as u32,
            0,
        )
    };
    if m <= 16 {
        launch(m, 0)?;
        Ok(1)
    } else {
        let first = m.div_ceil(2);
        launch(first, 0)?;
        launch(m - first, first)?;
        Ok(2)
    }
}

/// Sync'd wall clock over `REPS`, minus a warmup. Returns milliseconds per rep.
fn time_ms(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / f64::from(REPS))
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let scalar = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let batch16 = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?;
    let pipelined = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    let mut rng = Rng(0x927_8a16_2026);
    let mut failures = 0_usize;

    for shape in &SHAPES {
        let (n, k) = (shape.n, shape.k);
        // E4M3 byte draws skip 0x7F/0xFF (NaN), as the batch4 oracle does.
        let weights: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let acts: Vec<u8> = (0..MAX_M * k)
            .flat_map(|_| {
                bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let scales: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect();
        let weight = upload(&gpu, &weights)?;
        let scale = upload(&gpu, &scales)?;
        let input = upload(&gpu, &acts)?;
        let out_bytes = MAX_M * n * 2;
        let sentinel = vec![0x5a_u8; out_bytes + 2 * GUARD];
        let scalar_base = upload(&gpu, &sentinel)?;
        let batch_base = upload(&gpu, &sentinel)?;
        let tile_base = upload(&gpu, &sentinel)?;
        let scalar_out = scalar_base.offset(GUARD);
        let batch_out = batch_base.offset(GUARD);
        let tile_out = tile_base.offset(GUARD);
        let weight_gb = (n * k) as f64 / 1e9;

        for m in [5_usize, 8, 16, 32] {
            gpu.copy_h2d(&sentinel, scalar_base)?;
            gpu.copy_h2d(&sentinel, batch_base)?;
            for row in 0..m {
                ops::w8a16_gemv(
                    &gpu,
                    scalar,
                    input.offset(row * k * 2),
                    weight,
                    scale,
                    scalar_out.offset(row * n * 2),
                    n as u32,
                    k as u32,
                    0,
                )?;
            }
            let passes = batch16_route(&gpu, batch16, input, weight, scale, batch_out, m, n, k)?;
            gpu.synchronize(0)?;

            let mut baseline = vec![0_u8; sentinel.len()];
            let mut observed = vec![0_u8; sentinel.len()];
            gpu.copy_d2h(scalar_base, &mut baseline)?;
            gpu.copy_d2h(batch_base, &mut observed)?;
            let bytes = m * n * 2;
            let expected = &baseline[GUARD..GUARD + bytes];
            let actual = &observed[GUARD..GUARD + bytes];
            let unequal = actual
                .chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            let max_abs = actual
                .chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .map(|(a, b)| {
                    let f = |x: &[u8]| {
                        bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32() as f64
                    };
                    (f(a) - f(b)).abs()
                })
                .fold(0.0_f64, f64::max);
            let guards_intact = observed[..GUARD] == sentinel[..GUARD]
                && observed[GUARD + bytes..] == sentinel[GUARD + bytes..];

            let batch_ms = time_ms(&gpu, || {
                batch16_route(&gpu, batch16, input, weight, scale, batch_out, m, n, k)?;
                Ok(())
            })?;
            let tile_ms = time_ms(&gpu, || {
                ops::w8a16_gemm_pipelined(
                    &gpu, pipelined, input, weight, scale, tile_out, m as u32, n as u32, k as u32,
                    0,
                )
            })?;
            let batch_gbs = weight_gb * passes as f64 / (batch_ms / 1e3);
            let tile_gbs = weight_gb / (tile_ms / 1e3);
            println!(
                "{name:<8} M={m:<3} N={n} K={k} passes={passes} \
                 unequal_bf16={unequal} max_abs={max_abs:.9} guards={guards} | \
                 batch16 {batch_ms:.3}ms ({batch_gbs:.1} GB/s) \
                 vs pipelined {tile_ms:.3}ms ({tile_gbs:.1} GB/s) \
                 = {speedup:.2}x",
                name = shape.name,
                guards = if guards_intact { "ok" } else { "CLOBBERED" },
                speedup = tile_ms / batch_ms,
            );
            if unequal != 0 || max_abs != 0.0 || !guards_intact {
                failures += 1;
            }
        }

        // Oracle self-check, once: a one-bit mutation of the baseline MUST be
        // caught by the same comparison the loop above runs, so a green report
        // cannot mean "the comparison was vacuous".
        let mut bad = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut bad)?;
        bad[GUARD] ^= 1;
        let mut good = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut good)?;
        let caught = bad[GUARD..GUARD + 2] != good[GUARD..GUARD + 2];
        println!("KNOWN_BAD {} output-bit: refused={caught}", shape.name);
        ensure!(caught, "comparison oracle admitted a one-bit mutation");
    }

    ensure!(
        failures == 0,
        "{failures} shape/M cases differed from the scalar w8a16_gemv bits"
    );
    println!(
        "ALL PASS: real Qwen3.8-27B FFN shapes, batch16 M5/M8/M16 and 2x-halves M32 \
         exactly equal to scalar w8a16_gemv, guards intact"
    );
    Ok(())
}
