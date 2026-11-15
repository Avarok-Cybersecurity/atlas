// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the per-load plan of the 4-bit (`W4Linear`) routes: the types lane L23a's
//! decision tables read and answer with ([`PlanInputs`], [`RoutePlan`], [`LayerCtx`]), the FP8
//! precedence ([`Fp8Wins`]), the four route rows ([`Route`]), the plan of one load
//! ([`LoadPlans`]), the silent format plan the residency credit reads ([`plan_formats`]) and one
//! boot line per route ([`boot_line`]). A `#[path]` child of `w4_install.rs`, which re-exports
//! every item, so callers name them as `w4_install::*`.
//!
//! **PRECEDENCE** per layer type: the FP8 route of that type wins, then the affine route, then
//! (FFN only) NVFP4. **LOOKUPS:** none while every 4-bit row is off; with a row on, a route's
//! kernels are looked up only once every other clause of its table holds.

use avarok_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::super::nvfp4_from_bf16::{self, Nvfp4FromBf16, RouteInputs as Nvfp4Inputs};
use super::super::w4_residency::W4Formats;
use super::super::{affine4_from_bf16, attn_affine4_from_bf16, gdn_affine4_from_bf16};
use crate::layers::ops::port_rows;
use crate::port::w4::{W4Format, W4Opts};
use crate::port::w4_stats::W4Role;

/// The `why` of a route whose row is off (L23a's and the NVFP4 table's literal).
pub const ROW_OFF: &str = "row off";

/// The per-load facts one route's decision table reads (lane L23a).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanInputs {
    /// The checkpoint is plain BF16 (`Nvfp4Variant::Bf16Raw`).
    pub bf16: bool,
    pub tp_world_size: usize,
    /// A LoRA adapter is pending (`adapter_max_rank > 0`).
    pub lora: bool,
    /// The FP8 route of the same layer type resolved on: FP8 wins.
    pub fp8_wins: bool,
}

/// What one route answered for this load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutePlan {
    pub role: W4Role,
    /// `Some` when the route runs.
    pub format: Option<W4Format>,
    /// The quantizer options (meaningful for the affine routes only).
    pub opts: W4Opts,
    /// The row came from its `AVAROK_*` variable rather than the target's declaration.
    pub from_env: bool,
    /// "on", [`ROW_OFF`], or the declining clause, for the boot line.
    pub why: &'static str,
}

impl RoutePlan {
    /// A route whose row is off.
    pub fn off(role: W4Role) -> Self {
        Self {
            role,
            format: None,
            opts: W4Opts::default(),
            from_env: false,
            why: ROW_OFF,
        }
    }

    /// The route runs on this load.
    pub fn on(&self) -> bool {
        self.format.is_some()
    }
}

/// Where an install runs (lane L23a).
#[derive(Clone, Copy)]
pub struct LayerCtx<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub store: &'a WeightStore,
    pub stream: u64,
    /// The layer's tensor prefix (`config.layer_prefix(i)`).
    pub lp: &'a str,
    pub config: &'a ModelConfig,
}

/// Whether each layer type's FP8 route resolved on for this load (FP8 wins over every 4-bit
/// route).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp8Wins {
    pub ffn: bool,
    pub gdn: bool,
    pub attn: bool,
}

/// The four route rows, in boot-line order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    FfnAffine,
    FfnNvfp4,
    GdnAffine,
    AttnAffine,
}

impl Route {
    pub fn row(self) -> &'static str {
        match self {
            Self::FfnAffine => "dense_affine4_from_bf16",
            Self::FfnNvfp4 => "dense_nvfp4_from_bf16",
            Self::GdnAffine => "gdn_affine4_from_bf16",
            Self::AttnAffine => "attn_affine4_from_bf16",
        }
    }

    pub fn env(self) -> &'static str {
        match self {
            Self::FfnAffine => port_rows::dense_affine4_from_bf16::ENV,
            Self::FfnNvfp4 => port_rows::dense_nvfp4_from_bf16::ENV,
            Self::GdnAffine => port_rows::gdn_affine4_from_bf16::ENV,
            Self::AttnAffine => port_rows::attn_affine4_from_bf16::ENV,
        }
    }
}

/// Any 4-bit route row resolves on. With none on, nothing in this module looks anything up.
pub fn any_row_on() -> bool {
    crate::port::affine_opts::any_route_on() || port_rows::dense_nvfp4_from_bf16::row()
}

/// The factory's loader choice: the dense Qwen3.5 loader takes this checkpoint.
pub fn dense_loader_takes(config: &ModelConfig) -> bool {
    super::super::fp8_from_bf16::dense_loader_for(config)
}

/// Whether each FP8 route would resolve on, without its boot line: the same clauses as
/// `Fp8FromBf16::resolve`, `GdnFp8FromBf16::resolve` and `AttnFp8FromBf16::resolve`. A route's
/// kernels are looked up only when its lever is on and every other clause holds. For the
/// pre-load prediction; the loader passes what it actually resolved.
pub fn fp8_wins_silent(
    gpu: &dyn GpuBackend,
    bf16: bool,
    tp_world_size: usize,
    lora: bool,
) -> Fp8Wins {
    use super::super::fp8_from_bf16::{QUANT_MODULE, RouteInputs, route_selected};
    use crate::layers::dense_ffn::ada_kernels::Fp8FfnKernels;
    let levers = crate::layers::ops::target_defaults::resolved();
    let clauses = route_selected(RouteInputs {
        lever: true,
        bf16_checkpoint: bf16,
        single_gpu: tp_world_size.max(1) == 1,
        lora,
        kernels: true,
    });
    let quant = || crate::layers::try_target_kernel(gpu, QUANT_MODULE, QUANT_MODULE).0 != 0;
    let on = |lever: bool, gemv: &dyn Fn() -> bool| lever && clauses && quant() && gemv();
    Fp8Wins {
        ffn: on(levers.dense_fp8_from_bf16.value, &|| {
            Fp8FfnKernels::resolve(gpu).gemv.0 != 0
        }),
        gdn: on(levers.gdn_fp8_from_bf16.value, &|| {
            Fp8FfnKernels::m1_gemv(gpu).0 != 0
        }),
        attn: on(levers.attn_fp8_from_bf16.value, &|| {
            Fp8FfnKernels::m1_gemv(gpu).0 != 0
        }),
    }
}

/// The NVFP4 FFN route's table as a [`RoutePlan`].
pub fn nvfp4_plan(from_env: bool, i: Nvfp4Inputs) -> RoutePlan {
    let why = nvfp4_from_bf16::why_declined(i);
    RoutePlan {
        role: W4Role::Ffn,
        format: why.is_none().then_some(W4Format::Nvfp4),
        opts: W4Opts::default(),
        from_env,
        why: why.unwrap_or("on"),
    }
}

/// Every route's plan for one load.
#[derive(Clone, Copy, Debug)]
pub struct LoadPlans {
    pub ffn_affine: RoutePlan,
    pub ffn_nvfp4: RoutePlan,
    pub gdn: RoutePlan,
    pub attn: RoutePlan,
    /// The resolved NVFP4 quantizer and kernels, `Some` exactly when `ffn_nvfp4` is on.
    pub nvfp4: Option<Nvfp4FromBf16>,
}

impl LoadPlans {
    /// Every row off.
    pub fn off() -> Self {
        Self {
            ffn_affine: RoutePlan::off(W4Role::Ffn),
            ffn_nvfp4: RoutePlan::off(W4Role::Ffn),
            gdn: RoutePlan::off(W4Role::Gdn),
            attn: RoutePlan::off(W4Role::Attn),
            nvfp4: None,
        }
    }

    /// Plan every route. Silent; looks nothing up when no 4-bit row is on.
    pub fn plan(
        gpu: &dyn GpuBackend,
        bf16: bool,
        tp_world_size: usize,
        lora: bool,
        fp8: Fp8Wins,
    ) -> Self {
        if !any_row_on() {
            return Self::off();
        }
        let inputs = |fp8_wins| PlanInputs {
            bf16,
            tp_world_size,
            lora,
            fp8_wins,
        };
        let ffn_affine = affine4_from_bf16::plan(gpu, &inputs(fp8.ffn));
        let gdn = gdn_affine4_from_bf16::plan(gpu, &inputs(fp8.gdn));
        let attn = attn_affine4_from_bf16::plan(gpu, &inputs(fp8.attn));
        let lever = port_rows::dense_nvfp4_from_bf16::resolved();
        let affine_ffn = ffn_affine.on();
        let (nvfp4, i) =
            Nvfp4FromBf16::resolve(gpu, lever, bf16, tp_world_size, lora, fp8.ffn, affine_ffn);
        Self {
            ffn_affine,
            ffn_nvfp4: nvfp4_plan(lever.from_env(), i),
            gdn,
            attn,
            nvfp4,
        }
    }

    /// The FFN's route: affine when it runs, else NVFP4 (whose table declines under affine).
    pub fn ffn(&self) -> RoutePlan {
        if self.ffn_affine.on() {
            self.ffn_affine
        } else {
            self.ffn_nvfp4
        }
    }

    /// The 4-bit format each layer type takes from BF16 (the residency prediction's input).
    pub fn formats(&self) -> W4Formats {
        W4Formats {
            ffn: self.ffn().format,
            gdn: self.gdn.format,
            attn: self.attn.format,
        }
    }

    /// `(route, plan)` in boot-line order.
    pub fn routes(&self) -> [(Route, RoutePlan); 4] {
        [
            (Route::FfnAffine, self.ffn_affine),
            (Route::FfnNvfp4, self.ffn_nvfp4),
            (Route::GdnAffine, self.gdn),
            (Route::AttnAffine, self.attn),
        ]
    }
}

/// The silent pre-load / post-load format plan the residency credit reads (`w4_residency.rs`).
pub fn plan_formats(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    bf16: bool,
    tp_world_size: usize,
    lora: bool,
) -> W4Formats {
    if !any_row_on() || !dense_loader_takes(config) {
        return W4Formats::default();
    }
    let fp8 = fp8_wins_silent(gpu, bf16, tp_world_size, lora);
    LoadPlans::plan(gpu, bf16, tp_world_size, lora, fp8).formats()
}

/// One route's boot line, or `None` when its row is off.
pub fn boot_line(route: Route, p: &RoutePlan) -> Option<String> {
    if p.why == ROW_OFF {
        return None;
    }
    let state = match p.format {
        Some(f) => format!("ON, {f}"),
        None => format!("declined, {}", p.why),
    };
    Some(format!(
        "[avarok] 4-bit route {}: {state} (row on from the {}). {}=0 turns it off.",
        route.row(),
        if p.from_env {
            "environment"
        } else {
            "target declaration"
        },
        route.env(),
    ))
}
