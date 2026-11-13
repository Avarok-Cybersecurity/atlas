// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the keep-packed GGUF microtest's DRAM-bound timing rows (port lane L19): GB/s of
//! packed weight bytes for the decode GEMV at M = 1 and M = 8 on the 4B FFN shapes, for IQ4_XS, Q4_K and
//! Q6_K, the served block shape against the warp-per-row A/B arm, each call reading a distinct weight
//! copy (more than 64 MB in rotation, twice the RTX 4060 Ti's 32 MB L2), rotated A B B A.
//!
//! The bars (report 77, section 12.5, rows R2 and R3): the block shape at M = 1 reaches 85% of the card's
//! 288 GB/s, and its M = 8 call takes at most 25% longer than its M = 1 call (one weight pass for eight
//! rows). A miss prints `BAR MISS`; it does not fail the correctness gates.

use anyhow::{Context, Result};
use spark_model::port::gguf_q4::launch::{self, MmvqShape};
use spark_model::port::gguf_q4::reference::ggml_type;
use spark_model::port::gguf_q4::{Q4Type, from_gguf_tensor, gguf_of, release};
use spark_model::port::w4::{W4Box, W4Scratch};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::util::{Rng, random_weight, time_us, u16_bytes, upload};

/// Bytes the rotation must exceed (2x the 4060 Ti's L2).
const ROTATION_BYTES: usize = 64 << 20;
/// The card's DRAM bandwidth (RTX 4060 Ti 16 GB: 128-bit GDDR6 at 18 Gbps).
const PEAK_GBS: f64 = 288.0;
/// R2: M = 1 at or above this share of [`PEAK_GBS`].
const M1_BAR: f64 = 0.85;
/// R3: M = 8 at most this multiple of the M = 1 time.
const M8_BAR: f64 = 1.25;

const SHAPES: [(&str, usize, usize); 2] = [("gate/up", 9216, 2560), ("down", 2560, 9216)];

fn gbs(bytes: usize, us: f64) -> f64 {
    bytes as f64 / us / 1e3
}

/// One warp-shape decode: the q8_1 rows, then the warp-per-row GEMV, on the weight's own kernels.
fn warp_decode(
    gpu: &dyn GpuBackend,
    w: &W4Box,
    x: DevicePtr,
    rows: usize,
    out: DevicePtr,
    s: &W4Scratch,
) -> Result<()> {
    let d = gguf_of(&**w).context("a live GGUF weight")?;
    launch::q8_rows(gpu, &d.kernels, x, s.ws, rows, d.k, 0)?;
    launch::mmvq(
        gpu,
        &d.kernels,
        MmvqShape::Warp,
        d.blocks,
        s.ws,
        out,
        d.n,
        d.k,
        rows,
        0,
    )
}

/// Run every row; returns `(r2_held, r3_held)` over the IQ4_XS rows (the format the 27B ships).
pub fn rows(gpu: &dyn GpuBackend, rng: &mut Rng) -> Result<(bool, bool)> {
    println!(
        "timing (GB/s of packed weight bytes, distinct copies over {} MB, A B B A; peak {PEAK_GBS} GB/s):",
        ROTATION_BYTES >> 20
    );
    let (mut r2, mut r3) = (true, true);
    for ty in [Q4Type::Iq4Xs, Q4Type::Q4K, Q4Type::Q6K] {
        for (name, n, k) in SHAPES {
            let bytes = launch::weight_bytes(ty, n, k);
            let raw = random_weight(rng, ty, n, k);
            let copies = (ROTATION_BYTES.div_ceil(bytes) + 1).max(2);
            let ws: Vec<W4Box> = (0..copies)
                .map(|_| from_gguf_tensor(gpu, ggml_type(ty), &raw, n, k, 0))
                .collect::<Result<_>>()?;
            let x = upload(gpu, &u16_bytes(&rng.bf16_bits(8 * k)))?;
            let out = gpu.alloc(8 * n * 2)?;
            let sb = ws[0].scratch_bytes(8);
            let s = W4Scratch {
                ws: gpu.alloc(sb)?,
                ws_bytes: sb,
                tickets: DevicePtr::NULL,
                xq: DevicePtr::NULL,
                xs: DevicePtr::NULL,
            };
            let mut block_us = [0f64; 2];
            for (slot, m) in [1usize, 8].into_iter().enumerate() {
                let block = |i: usize| ws[i % ws.len()].decode(gpu, x, m, out, &s, 0);
                let warp = |i: usize| warp_decode(gpu, &ws[i % ws.len()], x, m, out, &s);
                let b1 = time_us(gpu, block)?;
                let w1 = time_us(gpu, warp)?;
                let w2 = time_us(gpu, warp)?;
                let b2 = time_us(gpu, block)?;
                let (bu, wu) = ((b1 + b2) / 2.0, (w1 + w2) / 2.0);
                block_us[slot] = bu;
                let tag = if m == 1 {
                    let held = gbs(bytes, bu) >= M1_BAR * PEAK_GBS;
                    if ty == Q4Type::Iq4Xs {
                        r2 &= held;
                    }
                    if held { "BAR PASS" } else { "BAR MISS" }
                } else {
                    "INFO"
                };
                println!(
                    "  [{tag}] {ty:?} {name:<7} M={m} N={n} K={k}: block {bu:.1} us {:.0} GB/s | warp {wu:.1} us {:.0} GB/s{}",
                    gbs(bytes, bu),
                    gbs(bytes, wu),
                    if m == 1 {
                        format!(" (R2 bar {:.0} GB/s)", M1_BAR * PEAK_GBS)
                    } else {
                        String::new()
                    }
                );
            }
            let ratio = block_us[1] / block_us[0];
            let held = ratio <= M8_BAR;
            if ty == Q4Type::Iq4Xs {
                r3 &= held;
            }
            println!(
                "  [{}] {ty:?} {name:<7} M=8 / M=1 time ratio {ratio:.2} (R3 bar {M8_BAR})",
                if held { "BAR PASS" } else { "BAR MISS" }
            );
            for w in &ws {
                release(gpu, &**w)?;
            }
            for p in [x, out, s.ws] {
                gpu.free(p)?;
            }
        }
    }
    Ok((r2, r3))
}
