// SPDX-License-Identifier: AGPL-3.0-only
// Route design after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce (report
// 72, brief W). Idea only; no TensorFold code is in this file.

//! Responsibility: the quantize-then-free install the three affine routes share (dense FFN, GDN,
//! attention): quantize each BF16 projection in order, hand every device buffer of the result to
//! the store's derived ledger, and free each source before the next projection is quantized.
//!
//! A `#[path]` child of `affine4_from_bf16.rs`, re-exported there for `gdn_affine4_from_bf16.rs`
//! and `attn_affine4_from_bf16.rs`.
//!
//! **LOAD PEAK.** Projection i's source is freed before projection i + 1 is quantized
//! ([`quantize_then_free`]), and the quantizer synchronizes its stream and frees its own staging
//! before it returns, so the peak is the checkpoint plus one projection's staging.
//!
//! **OWNERSHIP.** A 4-bit weight's four buffers (codes, scales, biases, tickets; lane L17a's
//! `AffineLinear`) and another width's three (tiles, scales, biases; lane L17c's `AffineNWeight`,
//! whose tickets are scratch) are adopted into the ledger (#736), which frees them at teardown,
//! and so is a Marlin weight's codes, scales, zero points and locks (lane L18b, only with
//! `marlin_w4a16` on, which swaps the 4-bit quantizer per projection; then `marlin::route::disown`
//! makes `marlin::release` a no-op). The install checks that the adopted bytes equal
//! `device_bytes()`, so no buffer is left without an owner. Never `port::affine::release` (or
//! `widths::release`) a route-installed weight: the ledger would free it a second time.
//!
//! **INT8 PROMPT GEMM.** With `affine4_prefill_int8` on, `port::affine::prefill::prepare` (lane
//! L17b) runs once per install, before any quantize, so its lookups and its flag region are on the
//! eager load path and no prompt GEMM inside a captured graph allocates.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::super::w4_install::LayerCtx;
use crate::port::affine::{affine_of, widths};
use crate::port::affine_opts::W4Role;
use crate::port::w4::{W4Box, W4Linear, W4Opts};

/// Every projection is non-empty and its K a whole number of groups (N needs no alignment: the
/// tiles pad rows to 128).
pub(crate) fn shapes_ok(projections: &[(usize, usize)], group: u16) -> bool {
    let g = group as usize;
    g > 0
        && !projections.is_empty()
        && projections
            .iter()
            .all(|&(n, k)| n > 0 && k > 0 && k.is_multiple_of(g))
}

/// Ledger labels of one role: codes (or tiles), scales, biases, tickets.
pub(crate) fn labels(role: W4Role) -> [&'static str; 4] {
    match role {
        W4Role::Gdn => [
            "ssm affine from bf16",
            "ssm affine from bf16 scales",
            "ssm affine from bf16 biases",
            "ssm affine from bf16 tickets",
        ],
        W4Role::Attn => [
            "attn affine from bf16",
            "attn affine from bf16 scales",
            "attn affine from bf16 biases",
            "attn affine from bf16 tickets",
        ],
        _ => [
            "ffn affine from bf16",
            "ffn affine from bf16 scales",
            "ffn affine from bf16 biases",
            "ffn affine from bf16 tickets",
        ],
    }
}

/// The device buffers of an affine weight with their sizes: four for a 4-bit `AffineLinear`,
/// three for an `AffineNWeight` of another width, none for any other `W4Linear`.
pub(crate) fn buffers_of(w: &dyn W4Linear) -> Vec<(DevicePtr, usize)> {
    if let Some(a) = affine_of(w) {
        let l = a.layout();
        return vec![
            (a.words, l.words),
            (a.scales, l.scales),
            (a.biases, l.scales),
            (a.tickets, a.tickets_bytes()),
        ];
    }
    match w
        .as_any()
        .and_then(|x| x.downcast_ref::<widths::AffineNWeight>())
    {
        Some(m) => {
            let tiles = widths::layout::tile_words(m.n(), m.k(), m.bits()) * 4;
            let table = widths::layout::table_elems(m.n(), m.k()) * 2;
            vec![
                (m.tiles_ptr(), tiles),
                (m.scales_ptr(), table),
                (m.biases_ptr(), table),
            ]
        }
        None => crate::port::marlin::route::buffers_of(w).unwrap_or_default(),
    }
}

/// The Marlin RTN arm (lane L18b, row `marlin_w4a16`, off on every target): `Some` when the row is
/// on, the route is 4-bit and Marlin serves the shape; `None` (nothing looked up or allocated with
/// the row off) when the affine quantizer runs. Another width never becomes Marlin's int4.
fn marlin_or_none(opts: &W4Opts, site: Site<'_>, p: Projection<'_>) -> Result<Option<W4Box>> {
    if opts.bits != 4 {
        return Ok(None);
    }
    crate::port::marlin::route::quantize_projection(
        site.gpu,
        p.src,
        p.n,
        p.k,
        opts.clip_search,
        site.stream,
        p.name,
    )
    .with_context(|| {
        format!(
            "marlin RTN int4 at load of {} [{}x{}] (group 128, clip {}); \
             AVAROK_MARLIN_W4A16=0 keeps the affine route",
            p.name, p.n, p.k, opts.clip_search
        )
    })
}

/// Adopt every buffer of `w` into the store's ledger under `role`'s labels; returns the bytes.
fn adopt(store: &WeightStore, role: W4Role, w: &dyn W4Linear) -> usize {
    let d = store.derived();
    let mut bytes = 0;
    for ((ptr, b), label) in buffers_of(w).into_iter().zip(labels(role)) {
        d.adopt(label, ptr, b);
        bytes += b;
    }
    bytes
}

/// Free one BF16 source. A loader concat buffer may sit in the ledger (#736): take it out first,
/// or teardown frees it a second time. A checkpoint tensor stays in the store's map as a dead
/// entry, as after the FP8 route's free; nothing reads it again.
fn free_source(store: &WeightStore, gpu: &dyn GpuBackend, src: DevicePtr) -> Result<()> {
    store.derived().disown(src);
    gpu.free(src)
}

/// Run `quantize(i)` then `free(i)` for each projection in order: projection i's source is gone
/// before projection i + 1 is quantized. Stops at the first error (the sources from there on stay
/// allocated and owned by the caller).
pub(crate) fn quantize_then_free<T>(
    count: usize,
    mut quantize: impl FnMut(usize) -> Result<T>,
    mut free: impl FnMut(usize) -> Result<()>,
) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(quantize(i)?);
        free(i)?;
    }
    Ok(out)
}

/// One projection's source, shape and name (for the error context).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Projection<'a> {
    pub src: DevicePtr,
    pub n: usize,
    pub k: usize,
    pub name: &'a str,
}

/// Where an install runs: the parts of lane L23b's `LayerCtx` it reads.
#[derive(Clone, Copy)]
pub(crate) struct Site<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub store: &'a WeightStore,
    pub stream: u64,
    pub lp: &'a str,
}

/// Adapter: the [`Site`] of lane L23b's `LayerCtx` (the one place its fields are read, beside
/// `ctx.config` in each route's `install`).
pub(crate) fn site_of<'a>(ctx: &'a LayerCtx<'_>) -> Site<'a> {
    Site {
        gpu: ctx.gpu,
        store: ctx.store,
        stream: ctx.stream,
        lp: ctx.lp,
    }
}

/// The env variable of `role`'s row, for the error context.
fn env_of(role: W4Role) -> &'static str {
    use crate::layers::ops::port_rows::{
        attn_affine4_from_bf16, dense_affine4_from_bf16, gdn_affine4_from_bf16,
    };
    match role {
        W4Role::Gdn => gdn_affine4_from_bf16::ENV,
        W4Role::Attn => attn_affine4_from_bf16::ENV,
        _ => dense_affine4_from_bf16::ENV,
    }
}

/// Quantize every projection in order, adopt each weight's buffers into the ledger, and free each
/// BF16 source once its quantize has returned. An error leaves the earlier weights adopted
/// (teardown frees them) and the later sources allocated; the load fails, as the FP8 route's does.
pub(crate) fn install_projections(
    role: W4Role,
    opts: &W4Opts,
    site: Site<'_>,
    projections: &[Projection<'_>],
) -> Result<Vec<W4Box>> {
    // The int8 prompt GEMM's entries and flag region (lane L17b), resolved and created here on the
    // eager load path: a wide verify can run the prompt GEMM inside a captured graph, where an
    // allocation fails. False (nothing allocated) where the Ada modules are absent.
    if opts.int8_prefill {
        crate::port::affine::prefill::prepare(site.gpu, site.stream)?;
    }
    let env = env_of(role);
    let quantize = |i: usize| -> Result<W4Box> {
        let p = projections[i];
        let w = match marlin_or_none(opts, site, p)? {
            Some(w) => w,
            None => crate::port::affine::quantize_from_bf16(
                site.gpu,
                p.src,
                p.n,
                p.k,
                opts,
                site.stream,
            )
            .with_context(|| {
                format!(
                    "affine {}-bit at load of {} [{}x{}] (group {}, clip {}); {env}=0 keeps BF16",
                    opts.bits, p.name, p.n, p.k, opts.group, opts.clip_search
                )
            })?,
        };
        let adopted = adopt(site.store, role, &*w);
        // A Marlin weight's buffers now belong to the ledger only (false, a no-op, for affine ones).
        crate::port::marlin::route::disown(&*w);
        ensure!(
            adopted == w.device_bytes(),
            "affine route: {} adopted {adopted} of its {} device bytes; a buffer has no owner",
            p.name,
            w.device_bytes()
        );
        Ok(w)
    };
    let free = |i: usize| -> Result<()> { free_source(site.store, site.gpu, projections[i].src) };
    quantize_then_free(projections.len(), quantize, free)
}

#[cfg(test)]
#[path = "affine4_shared_tests.rs"]
mod tests;
