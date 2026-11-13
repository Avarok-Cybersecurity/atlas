// SPDX-License-Identifier: AGPL-3.0-only
// Route design after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce (report 72,
// brief W). Idea only; no TensorFold code is in this file.

//! Responsibility: the load-time BF16 -> affine route for the GDN (linear-attention) projections:
//! whether a load takes it, which layers qualify, and the quantize-then-free install.
//!
//! **WHAT.** On a plain-BF16 dense checkpoint the loader concatenates each GDN layer's `in_proj_qkv`
//! and `in_proj_z` into the BF16 `[Q|K|V|Z]` buffer. With this route on, that buffer and `out_proj`
//! are quantized ONCE to affine codes (`port::affine::quantize_from_bf16`), each source is freed right
//! after its own quantize, and the two `W4Box`es go to `Qwen3SsmLayer::set_w4_proj` (lane L22a), which
//! serves decode, verify (M <= 16) and prefill through `W4Linear`.
//!
//! **WHY.** On a 4B the GDN projections are 2.0 GB of BF16 read per decode token (24 layers); at 4
//! bits and group 64 they keep 23.6 MB a layer against 83.9 MB, 1.45 GB less per token. Expected, to
//! be measured.
//!
//! **THE PRICE.** A GDN projection error feeds a recurrent state, so this is a numerics lever of its
//! own (`[defaults] gdn_affine4_from_bf16`, false everywhere, `AVAROK_GDN_AFFINE4_FROM_BF16=0|1`),
//! held to the D7 gate plus a long-context degeneration check. `AVAROK_W4_GDN_GROUP` lets the GDN
//! projections use a finer group (32) than the FFN (`port::affine_opts`).
//!
//! **DECLINES.** The shared table (`affine4_from_bf16::decide`); `gdn_fp8_from_bf16` on wins. Per
//! layer, any K (`hidden` for `[Q|K|V|Z]`, `value` for `out_proj`) that is not a whole number of
//! groups keeps BF16; N needs no alignment (the tiles pad rows to 128).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::affine4_from_bf16::{
    Projection, Site, affine_opts_of, install_projections, plan_role, shapes_ok, site_of,
};
use super::gdn_fp8_from_bf16::GdnShapes;
use super::w4_install::{LayerCtx, PlanInputs, RoutePlan};
use crate::port::affine_opts::W4Role;
use crate::port::w4::{W4Box, W4Opts};
use crate::weight_map::DenseWeight;

/// `[n, k]` of `[Q|K|V|Z]` and `out_proj`.
pub(super) fn gdn_projections(s: GdnShapes) -> [(usize, usize); 2] {
    [(s.qkvz, s.hidden), (s.hidden, s.value)]
}

/// A layer qualifies when both projections are non-empty and their K a whole number of groups.
pub(super) fn layer_ok(s: GdnShapes, group: u16) -> bool {
    shapes_ok(&gdn_projections(s), group)
}

/// The GDN route for this load (row `gdn_affine4_from_bf16`, options
/// `port::affine_opts::opts_for(W4Role::Gdn)`, which honor `AVAROK_W4_GDN_GROUP`). Logs nothing and,
/// with the row off, looks nothing up.
pub(super) fn plan(gpu: &dyn GpuBackend, i: &PlanInputs) -> RoutePlan {
    let lever = crate::layers::ops::target_defaults_port::resolved_port().gdn_affine4_from_bf16;
    plan_role(W4Role::Gdn, gpu, lever, i)
}

/// The GDN install with plain inputs: quantize `qkvz` (the loader's `[Q|K|V|Z]` concat) then
/// `out_proj`, freeing each BF16 source after its own quantize. `Ok(None)` when the layer does not
/// qualify; then neither source was touched.
pub(super) fn install_gdn(
    opts: &W4Opts,
    site: Site<'_>,
    shapes: GdnShapes,
    qkvz: DenseWeight,
    out_proj: DenseWeight,
) -> Result<Option<[W4Box; 2]>> {
    if !layer_ok(shapes, opts.group) {
        return Ok(None);
    }
    let [pq, po] = gdn_projections(shapes);
    let qkvz_name = format!("{}.linear_attn.in_proj_qkvz (concat)", site.lp);
    let out_name = format!("{}.linear_attn.out_proj.weight", site.lp);
    let projections = [
        Projection {
            src: qkvz.weight,
            n: pq.0,
            k: pq.1,
            name: &qkvz_name,
        },
        Projection {
            src: out_proj.weight,
            n: po.0,
            k: po.1,
            name: &out_name,
        },
    ];
    let v = install_projections(W4Role::Gdn, opts, site, &projections)?;
    let two: [W4Box; 2] = v
        .try_into()
        .map_err(|_| anyhow::anyhow!("affine GDN route: two weights expected for {}", site.lp))?;
    Ok(Some(two))
}

/// `qkvz` is the loader's BF16 `[Q|K|V|Z]` concat `[ssm_qkvz_size, hidden]`, `out_proj` is
/// `[hidden, value_dim]`. `[in_proj_qkvz, out_proj]` for `Qwen3SsmLayer::set_w4_proj`. On `Some`,
/// both BF16 sources were freed; on `None` (the plan is not this route, or the layer does not
/// qualify) neither was touched. The caller builds the layer with its BF16 `in_proj_qkvz` and
/// `out_proj_dense` slots empty, as the FP8 route does.
pub(super) fn install(
    plan: &RoutePlan,
    ctx: &LayerCtx<'_>,
    qkvz: DenseWeight,
    out_proj: DenseWeight,
) -> Result<Option<[W4Box; 2]>> {
    let Some(opts) = affine_opts_of(plan, W4Role::Gdn) else {
        return Ok(None);
    };
    install_gdn(
        &opts,
        site_of(ctx),
        GdnShapes::of(ctx.config),
        qkvz,
        out_proj,
    )
}

#[cfg(test)]
#[path = "gdn_affine4_from_bf16_tests.rs"]
mod tests;
