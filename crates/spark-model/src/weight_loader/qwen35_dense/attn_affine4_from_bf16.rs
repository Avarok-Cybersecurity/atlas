// SPDX-License-Identifier: AGPL-3.0-only
// Route design after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce (report 72,
// brief W). Idea only; no TensorFold code is in this file.

//! Responsibility: the load-time BF16 -> affine route for the full-attention q/k/v/o projections:
//! whether a load takes it, which layers qualify, and the quantize-then-free install.
//!
//! **WHAT.** On a plain-BF16 dense checkpoint, with this route on, each attention layer's q, k, v and
//! o (q doubled when gated) are quantized ONCE to affine codes (`port::affine::quantize_from_bf16`),
//! each BF16 source freed right after its own quantize, and the four `W4Box`es go to
//! `Qwen3AttentionLayer::set_w4_proj` (lane L22b), which serves decode, verify (M <= 16) and prefill
//! through `W4Linear` (q/k/v sharing one input).
//!
//! **WHY.** The 4B's 8 attention layers are 0.59 GB of BF16 read per decode token; at 4 bits and
//! group 64 they keep 20.6 MB a layer against 73.4 MB, 0.42 GB less per token. Expected, to be
//! measured.
//!
//! **THE PRICE.** A numerics lever of its own (`[defaults] attn_affine4_from_bf16`, false everywhere,
//! `AVAROK_ATTN_AFFINE4_FROM_BF16=0|1`), independent of the FFN and GDN rows; D7 gate.
//!
//! **DECLINES.** The shared table (`affine4_from_bf16::decide`); `attn_fp8_from_bf16` on wins. Per
//! layer, any K (`hidden` for q/k/v, `heads * head_dim` for o) that is not a whole number of groups
//! keeps BF16.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::affine4_from_bf16::{
    Projection, Site, affine_opts_of, install_projections, plan_role, shapes_ok, site_of,
};
use super::attn_fp8_from_bf16::AttnShapes;
use super::w4_install::{LayerCtx, PlanInputs, RoutePlan};
use crate::port::affine_opts::W4Role;
use crate::port::w4::{W4Box, W4Opts};
use crate::weight_map::DenseWeight;

const ATTN_NAMES: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

/// A layer qualifies when all four projections are non-empty and their K a whole number of groups.
pub(super) fn layer_ok(s: AttnShapes, group: u16) -> bool {
    shapes_ok(&s.projections(), group)
}

/// The attention route for this load (row `attn_affine4_from_bf16`, options
/// `port::affine_opts::opts_for(W4Role::Attn)`). Logs nothing and, with the row off, looks nothing up.
pub(super) fn plan(gpu: &dyn GpuBackend, i: &PlanInputs) -> RoutePlan {
    let lever = crate::layers::ops::target_defaults_port::resolved_port().attn_affine4_from_bf16;
    plan_role(W4Role::Attn, gpu, lever, i)
}

/// The attention install with plain inputs: quantize q, k, v and o (in that order), freeing each BF16
/// source after its own quantize. `Ok(None)` when the layer does not qualify; then no source was
/// touched.
pub(super) fn install_attn(
    opts: &W4Opts,
    site: Site<'_>,
    shapes: AttnShapes,
    srcs: [DenseWeight; 4],
) -> Result<Option<[W4Box; 4]>> {
    if !layer_ok(shapes, opts.group) {
        return Ok(None);
    }
    let names = ATTN_NAMES.map(|p| format!("{}.self_attn.{p}.weight", site.lp));
    let mut projections = Vec::with_capacity(4);
    for ((s, (n, k)), name) in srcs.iter().zip(shapes.projections()).zip(&names) {
        projections.push(Projection {
            src: s.weight,
            n,
            k,
            name,
        });
    }
    let v = install_projections(W4Role::Attn, opts, site, &projections)?;
    let four: [W4Box; 4] = v.try_into().map_err(|_| {
        anyhow::anyhow!(
            "affine attention route: four weights expected for {}",
            site.lp
        )
    })?;
    Ok(Some(four))
}

/// `srcs` = BF16 q, k, v, o (shapes of `attn_fp8_from_bf16::AttnShapes`). `[q, k, v, o]` for
/// `Qwen3AttentionLayer::set_w4_proj`. On `Some` every source was freed; on `None` (the plan is not
/// this route, or the layer does not qualify) none was touched. The caller builds the layer with its
/// BF16 q/k/v/o slots empty, as the FP8 route does.
pub(super) fn install(
    plan: &RoutePlan,
    ctx: &LayerCtx<'_>,
    srcs: [DenseWeight; 4],
) -> Result<Option<[W4Box; 4]>> {
    let Some(opts) = affine_opts_of(plan, W4Role::Attn) else {
        return Ok(None);
    };
    install_attn(&opts, site_of(ctx), AttnShapes::of(ctx.config), srcs)
}

#[cfg(test)]
#[path = "attn_affine4_from_bf16_tests.rs"]
mod tests;
