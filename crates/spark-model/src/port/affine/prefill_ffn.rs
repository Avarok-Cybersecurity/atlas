// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the dense FFN's whole prompt path on the affine W4A8 int8 route (lane L17b), chunk by chunk:
//! gate and up in one `_many` grid over ONE quantization of x, then the fused SwiGLU + group sums + int8
//! quantizer (`affine4_swiglu_act_quant_ada`) writing h over the gate output, then the down projection's int8
//! GEMM, each with its device-side BF16 fallback.
//!
//! It gives the per-projection path's bytes exactly: the dense FFN's W4 dispatch otherwise runs `prefill` for
//! gate, then up (each quantizing x itself), the layer's `moe_silu_mul` over `rows * inter`, then `prefill` for
//! down (quantizing h), with the same chunks (`select::prefill_chunks` at `prefill_chunk_cap(16)`). Every output
//! row depends only on its own inputs and the weights, the `_many` grid gives each projection the bits of its
//! own grid, and the fused kernel's h and codes are `moe_silu_mul`'s and the plain quantizer's (the GPU
//! microtest pins both). What it saves per chunk: one quantization of x, the separate SwiGLU launch and the down
//! projection's second read of h.
//!
//! [`ffn_int8`] declines (`Ok(false)`, nothing launched) unless all three weights are live 4-bit group-64
//! affine weights with `int8_prefill` on, the shapes chain (gate and up `[inter, hidden]`, down `[hidden,
//! inter]`), the rows exceed the decode window, the six kernels resolve and the scratch has the int8 buffers.
//! The caller keeps its per-projection path for every decline, and only offers SwiGLU layers (the fusion is
//! `silu(gate) * up`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{Packed, flags, launch, run_down_packed, run_packed};
use crate::port::affine::affine_of;
use crate::port::w4::{W4Linear, W4Scratch, select};

#[cfg(test)]
#[path = "prefill_ffn_tests.rs"]
mod tests;

/// The buffers of one dense FFN prompt call. `x` is `[rows, hidden]` BF16; `gate_out` and `up_out` are
/// `[rows, inter]` BF16 (row stride `inter`), `gate_out` receives h = bf16(silu(gate) * up) in place, as the
/// layer's own SwiGLU leaves it; `out` is `[rows, hidden]` BF16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FfnIo {
    pub x: DevicePtr,
    pub rows: usize,
    pub gate_out: DevicePtr,
    pub up_out: DevicePtr,
    pub out: DevicePtr,
}

/// The three weights as int8-route views, or `None` when this path does not serve them.
fn int8_views(gate: &dyn W4Linear, up: &dyn W4Linear, down: &dyn W4Linear) -> Option<[Packed; 3]> {
    let view = |w: &dyn W4Linear| -> Option<Packed> {
        let a = affine_of(w)?;
        if a.int8_prefill { Packed::of(&a) } else { None }
    };
    let (g, u, d) = (view(gate)?, view(up)?, view(down)?);
    let chained = g.n == u.n && g.k == u.k && d.k == g.n && d.n == g.k;
    chained.then_some([g, u, d])
}

/// The largest decode window of the three weights (16 for affine weights): prompt calls must be wider.
fn decode_window(gate: &dyn W4Linear, up: &dyn W4Linear, down: &dyn W4Linear) -> usize {
    gate.max_decode_rows()
        .max(up.max_decode_rows())
        .max(down.max_decode_rows())
}

/// The dense FFN's prompt path on the int8 route: `out = down(silu(gate(x)) * up(x))` for `io.rows` rows, with
/// h left in `io.gate_out`. `Ok(false)`: declined, nothing launched (see the module docs).
pub fn ffn_int8(
    gate: &dyn W4Linear,
    up: &dyn W4Linear,
    down: &dyn W4Linear,
    gpu: &dyn GpuBackend,
    io: FfnIo,
    s: &W4Scratch,
    stream: u64,
) -> Result<bool> {
    let window = decode_window(gate, up, down);
    if io.rows <= window {
        return Ok(false);
    }
    let Some([g, u, d]) = int8_views(gate, up, down) else {
        return Ok(false);
    };
    let (hidden, inter) = (g.k, g.n);
    let aligned = |p: DevicePtr| !p.is_null() && p.0.is_multiple_of(16);
    let io_ok = [io.x, io.gate_out, io.up_out].into_iter().all(aligned)
        && !io.out.is_null()
        && io.out.0.is_multiple_of(4);
    if !io_ok || !launch::kernels_present(gpu) {
        return Ok(false);
    }
    // The int8 buffers are carved per call from `s` (`Int8Act::carve`, the same test); check before any launch,
    // so a missing region declines the whole layer instead of running half of it on BF16.
    if ![s.xq, s.xs].into_iter().all(aligned) {
        if gpu.op_cache().once("affine_prefill:ffn_without_scratch") {
            tracing::warn!(
                "affine prefill: the int8 FFN path has no xq/xs scratch; the per-projection path runs"
            );
        }
        return Ok(false);
    }
    flags(gpu, stream)?;
    if gpu.op_cache().once("affine_prefill:ffn_int8_route") {
        tracing::info!(
            "[avarok] affine prefill: dense FFN on the int8 route (gate+up one grid over one quantization, \
             fused SwiGLU + sums + int8 quant, down), first at {} rows",
            io.rows
        );
    }
    let cap = select::prefill_chunk_cap(window);
    for (first, len) in select::prefill_chunks(io.rows, cap, window) {
        let xc = io.x.offset(first * hidden * 2);
        let gc = io.gate_out.offset(first * inter * 2);
        let uc = io.up_out.offset(first * inter * 2);
        let oc = io.out.offset(first * hidden * 2);
        let ran = run_packed(&[(g, gc), (u, uc)], gpu, xc, len, s, true, stream)?;
        ensure!(
            ran,
            "affine int8 FFN: gate+up declined after the checks (rows {first}+{len})"
        );
        let ran = run_down_packed(&d, gpu, gc, uc, inter, gc, inter, len, oc, s, stream)?;
        ensure!(
            ran,
            "affine int8 FFN: the fused down projection declined after the checks (rows {first}+{len})"
        );
    }
    Ok(true)
}
