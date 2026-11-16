// SPDX-License-Identifier: AGPL-3.0-only
// Route design after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce: a BF16
// checkpoint's linears quantized to MLX affine codes once at load and served on the lane MMA
// GEMV. Idea only; no TensorFold code is in this file. The kernels and the
// quantizer it calls are lanes L17a's, L17b's and L17c's (`port::affine`, see
// TENSORFOLD-ATTRIBUTION.md).

//! Responsibility: the load-time BF16 -> affine route for the dense FFN, plus the decision table
//! and the plan adapter the GDN and attention affine routes share (`gdn_affine4_from_bf16.rs`,
//! `attn_affine4_from_bf16.rs`; the shared install is the child `affine4_shared.rs`).
//!
//! **WHAT.** On a plain-BF16 dense checkpoint (`Nvfp4Variant::Bf16Raw`: Holo 3.1 0.8B and 4B,
//! Qwen3.5 dense) each layer's gate, up and down are quantized ONCE at load by the GPU quantizer
//! (`port::affine::quantize_from_bf16`) into MLX affine codes: per group of `group` inputs a bf16
//! scale and bias, value `s * q + b`. Each projection's BF16 source is freed as soon as its own
//! quantize has returned, so the load peak is the checkpoint plus one projection's staging. The
//! weights come back as `W4Box`es for `DenseFfnLayer::set_w4` (lane L21), which serves decode and
//! verify rows 1..=16 on the lane GEMV and prompts through `W4Linear::prefill` (lane L17b) or its
//! BF16 fallback.
//!
//! **WHY.** Decode is bandwidth bound. At 4 bits and group 64 the 4B FFN keeps 39.8 MB a layer
//! against 141.6 MB of BF16 (and 70.8 MB of FP8): 3.26 GB less per decode token than BF16 over 32
//! layers, 1.0 GB less than the FP8 route. An expectation, to be measured.
//!
//! **THE PRICE.** Round-to-nearest 4-bit codes are a numerics change larger than E4M3, so the
//! route is a target declaration (`[defaults] dense_affine4_from_bf16`, false on every target)
//! with `AVAROK_DENSE_AFFINE4_FROM_BF16=0|1` as the A/B; what flips it is a quality gate (NLL, KL, coherence, recall).
//!
//! **DECLINES** (the table, [`decide`], first failing clause wins): not a BF16 checkpoint; tensor
//! parallel (no sharded affine weight exists); a pending LoRA adapter (`set_lora_weights` has no
//! affine arm); the FP8 route of the same layer type is on (FP8 wins, and the plan says so); a
//! width and group the kernels do not serve; the kernels absent (every non-Ada target: the modules
//! are Ada files, `has_module` answers false and no lookup is issued). Per layer, any `[n, k]` with
//! `k % group != 0` or a source that is not BF16 of the config's shape keeps that layer on BF16.
//!
//! **SPLIT OF WORK with lane L23b** (`w4_install.rs`, the only caller): `plan` decides and logs
//! nothing (L23b prints one boot line per route from the plan's `why`), and with the row off it
//! looks nothing up; `install` quantizes, adopts the buffers and frees the sources, and does not
//! install into the layer, tally residency or run the self-check. [`facts_of`], [`route_plan`],
//! [`affine_opts_of`] and `shared::site_of` (plus `ctx.config` in each `install`) are the only
//! lines that read L23b's types, so a contract change touches only them.

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::w4_install::{LayerCtx, PlanInputs, ROW_OFF, RoutePlan};
use crate::layers::ops::target_defaults::Resolved;
use crate::port::affine::{AffineKernels, widths};
use crate::port::affine_opts::{AffineOpts, W4Role, opts_for, supported};
use crate::port::w4::{W4Box, W4Format, W4Opts};

#[path = "affine4_shared.rs"]
mod shared;
pub(super) use shared::{Projection, Site, install_projections, shapes_ok, site_of};
#[cfg(test)]
pub(super) use shared::{buffers_of, labels};

/// One route's inputs, as data, so the CPU table test pins each clause without a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AffineInputs {
    /// The route row after its `AVAROK_*` override.
    pub lever: bool,
    /// `Nvfp4Variant::Bf16Raw`: there is BF16 to quantize.
    pub bf16_checkpoint: bool,
    /// `tp_world_size == 1`.
    pub single_gpu: bool,
    /// A LoRA adapter is pending (`adapter_max_rank > 0`).
    pub lora: bool,
    /// The FP8 route of the same layer type is on; FP8 wins.
    pub fp8_wins: bool,
    /// The width, group and clip depth are a served combination.
    pub format: bool,
    /// The kernels of that width and group resolve.
    pub kernels: bool,
}

/// Why a route with its lever on did not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Decline {
    NotBf16,
    TensorParallel,
    Lora,
    Fp8Wins,
    Format,
    Kernels,
}

impl Decline {
    /// The boot line's words for this clause.
    pub(super) fn reason(self) -> &'static str {
        match self {
            Decline::NotBf16 => "not a plain BF16 checkpoint",
            Decline::TensorParallel => "tensor parallel (no sharded affine weight)",
            Decline::Lora => "a LoRA adapter is pending (no affine LoRA arm)",
            Decline::Fp8Wins => "the FP8 route of this layer type is on, and FP8 wins",
            Decline::Format => "the width and group are not a served combination",
            Decline::Kernels => "the affine kernels are not compiled for this target",
        }
    }
}

/// The decision table. Pure; `None` when the lever is off (not a decline), else the first failing
/// clause in the documented order, else `Ok`.
pub(super) fn decide(i: AffineInputs) -> Option<Result<(), Decline>> {
    if !i.lever {
        return None;
    }
    Some(if !i.bf16_checkpoint {
        Err(Decline::NotBf16)
    } else if !i.single_gpu {
        Err(Decline::TensorParallel)
    } else if i.lora {
        Err(Decline::Lora)
    } else if i.fp8_wins {
        Err(Decline::Fp8Wins)
    } else if !i.format {
        Err(Decline::Format)
    } else if !i.kernels {
        Err(Decline::Kernels)
    } else {
        Ok(())
    })
}

/// The per-load facts the table reads besides the lever and the options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LoadFacts {
    pub bf16_checkpoint: bool,
    pub tp_world_size: usize,
    pub lora: bool,
    pub fp8_wins: bool,
}

/// What the table answered for one route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// The lever is off: nothing was looked up.
    Off,
    /// The lever is on and a clause declined it.
    Declined(Decline),
    /// Take it.
    On,
}

/// Every kernel the width and group need resolves. `has_module` first, so a target without the
/// Ada modules issues no lookup at all; only called once every other clause holds.
pub(super) fn kernels_present(gpu: &dyn GpuBackend, opts: AffineOpts) -> bool {
    if opts.bits == 4 {
        AffineKernels::available(gpu)
            && AffineKernels::resolve(gpu).is_ok_and(|k| k.group(opts.group).is_ok())
    } else {
        widths::available(gpu) && widths::kernels::Kernels::resolve(gpu, opts.bits).is_ok()
    }
}

/// The route's verdict for this load. With the lever off nothing is looked up; the kernel lookups
/// run only once every other clause holds (FP8 winning, a bad format or a TP load look nothing up).
pub(super) fn verdict(
    gpu: &dyn GpuBackend,
    lever: bool,
    f: LoadFacts,
    opts: AffineOpts,
) -> Verdict {
    if !lever {
        return Verdict::Off;
    }
    let mut i = AffineInputs {
        lever: true,
        bf16_checkpoint: f.bf16_checkpoint,
        single_gpu: f.tp_world_size.max(1) == 1,
        lora: f.lora,
        fp8_wins: f.fp8_wins,
        format: supported(opts.bits, opts.group, opts.clip_search).is_ok(),
        kernels: true,
    };
    if decide(i) == Some(Ok(())) {
        i.kernels = kernels_present(gpu, opts);
    }
    match decide(i) {
        None => Verdict::Off,
        Some(Ok(())) => Verdict::On,
        Some(Err(d)) => Verdict::Declined(d),
    }
}

/// Adapter: the facts of lane L23b's `PlanInputs` (the one place its fields are read).
fn facts_of(i: &PlanInputs) -> LoadFacts {
    LoadFacts {
        bf16_checkpoint: i.bf16,
        tp_world_size: i.tp_world_size,
        lora: i.lora,
        fp8_wins: i.fp8_wins,
    }
}

/// Adapter: lane L23b's `RoutePlan` for a verdict. `format` is `Some` only on [`Verdict::On`];
/// `opts` are the resolved options (the defaults when the lever is off, which resolves nothing);
/// `why` is "on", [`ROW_OFF`] (the literal lane L23b's boot line keys on) or the declining clause.
pub(super) fn route_plan(
    role: W4Role,
    v: Verdict,
    opts: Option<AffineOpts>,
    from_env: bool,
) -> RoutePlan {
    let o = opts.unwrap_or(AffineOpts::DEFAULT);
    let (format, why) = match v {
        Verdict::On => (
            Some(W4Format::Affine {
                bits: o.bits,
                group: o.group,
            }),
            "on",
        ),
        Verdict::Off => (None, ROW_OFF),
        Verdict::Declined(d) => (None, d.reason()),
    };
    RoutePlan {
        role,
        format,
        opts: o.w4(),
        from_env,
        why,
    }
}

/// Adapter: the quantizer options of a plan that took this role's affine route, or `None` (row
/// off, declined, another role's plan, or another format).
pub(super) fn affine_opts_of(plan: &RoutePlan, role: W4Role) -> Option<W4Opts> {
    match plan.format {
        Some(W4Format::Affine { bits, group }) if plan.role == role => Some(W4Opts {
            bits,
            group,
            ..plan.opts
        }),
        _ => None,
    }
}

/// `role`'s plan from its resolved lever (passed in so a test can drive it). Logs nothing.
pub(super) fn plan_role(
    role: W4Role,
    gpu: &dyn GpuBackend,
    lever: Resolved<bool>,
    i: &PlanInputs,
) -> RoutePlan {
    if !lever.value {
        return route_plan(role, Verdict::Off, None, lever.from_env());
    }
    let opts = opts_for(role);
    let v = verdict(gpu, true, facts_of(i), opts);
    route_plan(role, v, Some(opts), lever.from_env())
}

/// The dense-FFN route for this load (row `dense_affine4_from_bf16`, options
/// `port::affine_opts::opts_for(W4Role::Ffn)`).
pub(super) fn plan(gpu: &dyn GpuBackend, i: &PlanInputs) -> RoutePlan {
    let rows = crate::layers::ops::target_defaults_port::resolved_port();
    plan_role(W4Role::Ffn, gpu, rows.dense_affine4_from_bf16, i)
}

/// The FFN's `[n, k]` of gate, up and down.
pub(super) fn ffn_projections(hidden: usize, inter: usize) -> [(usize, usize); 3] {
    [(inter, hidden), (inter, hidden), (hidden, inter)]
}

const FFN_NAMES: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// Per-layer qualification: gate, up and down are BF16 of the config's shape on the store, and
/// both extents are whole groups. A layer that fails keeps its BF16 path.
pub(super) fn ffn_layer_ok(
    store: &WeightStore,
    lp: &str,
    hidden: usize,
    inter: usize,
    group: u16,
) -> bool {
    let p = ffn_projections(hidden, inter);
    shapes_ok(&p, group)
        && FFN_NAMES.iter().zip(p).all(|(name, (n, k))| {
            store
                .get(&format!("{lp}.mlp.{name}.weight"))
                .is_ok_and(|w| w.dtype == WeightDtype::BF16 && w.shape == [n, k])
        })
}

/// The FFN install with plain inputs: quantize one dense layer's gate, up and down (in that
/// order), freeing each BF16 checkpoint tensor after its own quantize. `Ok(None)` when the layer
/// does not qualify ([`ffn_layer_ok`]); then nothing was touched.
pub(super) fn install_ffn(
    opts: &W4Opts,
    site: Site<'_>,
    hidden: usize,
    inter: usize,
) -> Result<Option<[W4Box; 3]>> {
    if !ffn_layer_ok(site.store, site.lp, hidden, inter, opts.group) {
        return Ok(None);
    }
    let names: [String; 3] = FFN_NAMES.map(|p| format!("{}.mlp.{p}.weight", site.lp));
    let mut projections = Vec::with_capacity(3);
    for (name, (n, k)) in names.iter().zip(ffn_projections(hidden, inter)) {
        let src = site
            .store
            .get(name)
            .with_context(|| format!("affine FFN route: {name}"))?
            .ptr;
        projections.push(Projection { src, n, k, name });
    }
    let v = install_projections(W4Role::Ffn, opts, site, &projections)?;
    let three: [W4Box; 3] = v
        .try_into()
        .map_err(|_| anyhow::anyhow!("affine FFN route: three weights expected for {}", site.lp))?;
    Ok(Some(three))
}

/// Quantize `{lp}.mlp.{gate,up,down}_proj.weight` (BF16 on the store) and free each BF16 source
/// after its quantize. `[gate, up, down]` for `DenseFfnLayer::set_w4`; `Ok(None)` when the plan is
/// not this route or the layer does not qualify (then nothing was freed).
pub(super) fn install(plan: &RoutePlan, ctx: &LayerCtx<'_>) -> Result<Option<[W4Box; 3]>> {
    let Some(opts) = affine_opts_of(plan, W4Role::Ffn) else {
        return Ok(None);
    };
    let (hidden, inter) = (ctx.config.hidden_size, super::ffn_inter(ctx.config));
    install_ffn(&opts, site_of(ctx), hidden, inter)
}

#[cfg(test)]
#[path = "affine4_from_bf16_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "affine4_from_bf16_tests_b.rs"]
mod tests_b;
