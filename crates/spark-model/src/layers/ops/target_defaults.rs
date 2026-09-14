// SPDX-License-Identifier: AGPL-3.0-only

//! Resolving the compiled target's serving defaults — **baked first,
//! environment second**.
//!
//! # The defect this closes
//!
//! Maintainer review of the H100 integration branch, 2026-09-11 (tbraun96):
//!
//! > There is no arch separation at all. H100 builds compile GB10's kernel
//! > tree. Every Hopper/GB10 divergence is expressed as an env lever set by an
//! > H100 recipe living outside this repo — not as arch-selected code. "No
//! > interference" rests on discipline rather than structure.
//!
//! Every lever here used to read the environment and fall back to a literal
//! that described GB10. The H100 configuration therefore lived in a launch
//! script nobody in this repository could see, review or test, and "GB10 is
//! unaffected" was a promise about which prefixes people remembered to type.
//!
//! Now the fallback is [`atlas_kernels::TARGET_DEFAULTS`], baked by
//! `build.rs` from the ONE `kernels/<hw>/HARDWARE.toml` this binary compiled
//! (`[defaults]`). A GB10 build cannot carry Hopper's numbers, an H100 serve
//! needs no prefixes, and every value is reviewable beside the arch it
//! belongs to.
//!
//! # The override grammar
//!
//! | input | meaning |
//! |---|---|
//! | variable absent | the baked target default |
//! | `0`, `false`, `off`, `no` (any case, trimmed) | OFF — explicit override |
//! | any other value, including empty | ON — explicit override |
//!
//! ⚠️ **`VAR=0` NOW MEANS OFF.** These levers were PRESENCE-gated
//! (`var_os(..).is_some()`), chosen so an A/B recipe could stay a bare `VAR=1`
//! prefix with no "`=0` means on" trap. Presence cannot express "off", and
//! once a target's default can be ON, an operator with no way to turn a lever
//! off is back to editing launch scripts. The trap the old rule avoided is
//! gone in the direction that matters: `VAR=0` now means what it reads as.
//! `VAR=1` is unchanged everywhere.
//!
//! The legacy `ATLAS_NO_*` kill switches stay PRESENCE-gated and still force
//! their lever OFF, so no script that predates this file changes meaning.
//! `ATLAS_NO_DECODE_SPLIT_SILU` is the one in this table.
//!
//! # One resolution, one log line
//!
//! [`resolved`] is the SSOT: every consumer below reads it, and
//! `spark-server` prints it as `target defaults (<hw>): …` with the
//! environment-sourced values marked. A lever resolved in two places is a
//! lever that can disagree with the line that claims to report it.
//!
//! # Adding a lever — the contract
//!
//! ONE commit touches all of: the field in
//! `atlas_kernels::TargetDefaults`, the parse arm in
//! `atlas-kernels/build_defaults.rs`, the row in EVERY
//! `kernels/<hw>/HARDWARE.toml` that has a `[defaults]` table, the
//! [`TargetLevers`] field and its arm in [`resolve`], the field in
//! [`format_levers`]'s line, and a test. `parse_defaults` panics on an
//! unknown key, so a half-landed lever fails the build rather than reading as
//! agreement with the baseline.
//!
//! ★ And the commit that does all that is the one landing the lever's
//! CONSUMER. A row whose dispatch site does not exist yet cannot be graded,
//! answers nothing when an operator sets its variable, and puts an ` (env)`
//! tag in the boot line against a decision that changes no code. So a kernel
//! PR brings its own row; this module ships only the rows whose arms are
//! already here.

use atlas_kernels::attn_splitk::{self, SplitkPolicy};

use super::gemm_quant::{DENSE_GEMV_BATCHM_DECODE_MAX_M, DENSE_GEMV_BATCHM_MAX_M};

/// Where a resolved value came from — the whole point of the log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// `kernels/<hw>/HARDWARE.toml` `[defaults]`.
    Target,
    /// An `ATLAS_*` variable in the process environment.
    Env,
}

impl Source {
    /// The suffix the serve log appends to an environment-sourced value.
    pub fn tag(self) -> &'static str {
        match self {
            Source::Target => "",
            Source::Env => " (env)",
        }
    }
}

/// A resolved lever and where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved<T> {
    pub value: T,
    pub source: Source,
}

impl<T> Resolved<T> {
    fn target(value: T) -> Self {
        Self {
            value,
            source: Source::Target,
        }
    }
    fn env(value: T) -> Self {
        Self {
            value,
            source: Source::Env,
        }
    }
    /// True when the environment overrode the target's declaration.
    pub fn from_env(&self) -> bool {
        self.source == Source::Env
    }
}

/// The override grammar for a boolean lever, as a pure function.
///
/// `raw` is the positive variable's value (`None` = absent). `legacy_off` is
/// the presence of the matching `ATLAS_NO_*` kill switch, which wins over
/// everything: it is the escape hatch an operator reaches for while a serve
/// misbehaves, and a hatch that a stale positive variable can veto is not one.
pub fn resolve_toggle(default_on: bool, raw: Option<&str>, legacy_off: bool) -> Resolved<bool> {
    if legacy_off {
        return Resolved::env(false);
    }
    match raw {
        None => Resolved::target(default_on),
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "no" => Resolved::env(false),
            _ => Resolved::env(true),
        },
    }
}

/// The BF16 decode head's batched-GEMV band.
///
/// 🔴 Read `layers/ops/gemm_quant.rs` before touching the DEFAULT. The band's
/// upper edge decides whether a width lands on the batched GEMV or on a
/// REASSOCIATING tile GEMM, and the A/B behind GB10's 8 measured the GEMV
/// NEGATIVE above it (-14.4% at C=16, commits 84d5b763c / 78d276832).
///
/// Clamped to [`DENSE_GEMV_BATCHM_MAX_M`], the kernel's compile-time row
/// bound — `dense_gemv_batchm` refuses above it rather than writing 16 of m
/// rows, and a lever that produced an `Err` at every decode step would be a
/// worse failure than ignoring the excess. An unparseable or `0` environment
/// value keeps the target's declaration; the value is a BAND, not a switch, so
/// there is no "off".
pub fn resolve_batchm_max(default_max: u32, raw: Option<&str>) -> Resolved<u32> {
    let clamp = |v: u32| v.min(DENSE_GEMV_BATCHM_MAX_M);
    match raw
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|&v| v > 0)
    {
        Some(v) => Resolved::env(clamp(v)),
        None => Resolved::target(clamp(default_max)),
    }
}

/// Upper `M` for the W8A8 dense-FFN prefill, per projection shape.
///
/// Unlike [`resolve_batchm_max`] a parsed **0 is honoured**, because 0 is a
/// meaningful operator answer here ("never take the W8A8 arm on this shape")
/// and silently ignoring it would make `…=0` read as agreement with the
/// target — the same silent-agreement failure `parse_defaults` panics over.
/// Anything that is not a u32 falls back to the target's declaration.
pub fn resolve_max_m(default_max: u32, raw: Option<&str>) -> Resolved<u32> {
    match raw.and_then(|v| v.trim().parse::<u32>().ok()) {
        Some(v) => Resolved::env(v),
        None => Resolved::target(default_max),
    }
}

/// Every serving lever this target declares, resolved against the environment.
///
/// Field order is the order the serve log prints them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetLevers {
    /// `kernels/<hw>` this binary was compiled from, for the log line only.
    pub hw: &'static str,
    pub lm_head_batchm_max: Resolved<u32>,
    pub ssm_batched_recurrent: Resolved<bool>,
    pub gdn_prefill_tc: Resolved<bool>,
    pub ssm_ba_gates_hopper: Resolved<bool>,
    pub fp8_act_quant_hopper: Resolved<bool>,
    pub decode_split_silu: Resolved<bool>,
    pub attn_decode_splitk: Resolved<SplitkPolicy>,
    /// The `w8a16_gemm_m16` tier on the dense-FFN decode arm (#927).
    pub ffn_m16_tc: Resolved<bool>,
    /// The `w8a16_gemm_m16` tiers on the decode Q/K/V and o_proj (#927).
    pub attn_m16_tc: Resolved<bool>,
    /// The `dense_gemm_m16_bf16` arm on the BF16 decode head (#927).
    pub lm_head_m16_tc: Resolved<bool>,
    /// `w8a16_gemv_batch16_ncol{2,4}` on the decode attention projections
    /// (#927). No serving receipt on any target — off everywhere.
    pub attn_ncol_gemv: Resolved<bool>,
    pub ffn_gateup_fused: Resolved<bool>,
    pub w8a8_prefill_max_m_widening: Resolved<u32>,
    pub w8a8_prefill_max_m_narrowing: Resolved<u32>,
}

/// The whole table, as a pure function of the baked declaration and a variable
/// lookup — so the resolution is testable for ANY target from a CPU test, on
/// any host, without touching the process environment.
pub fn resolve(
    defaults: &atlas_kernels::TargetDefaults,
    mut var: impl FnMut(&str) -> Option<String>,
) -> TargetLevers {
    let split_silu_off = var("ATLAS_NO_DECODE_SPLIT_SILU").is_some();

    TargetLevers {
        hw: defaults.hw,
        lm_head_batchm_max: resolve_batchm_max(
            defaults.lm_head_batchm_max,
            var("ATLAS_LM_HEAD_BATCHM_MAX").as_deref(),
        ),
        w8a8_prefill_max_m_widening: resolve_max_m(
            defaults.w8a8_prefill_max_m_widening,
            var("ATLAS_W8A8_PREFILL_MAX_M_WIDENING").as_deref(),
        ),
        w8a8_prefill_max_m_narrowing: resolve_max_m(
            defaults.w8a8_prefill_max_m_narrowing,
            var("ATLAS_W8A8_PREFILL_MAX_M_NARROWING").as_deref(),
        ),
        // `ATLAS_SSM_BATCHED_RECURRENT` was `== "1"` in `gdn_flags::from_env`;
        // under the 2026-09-11 grammar `=0` now turns it OFF instead of
        // reading as absent. Everything that ever set it set it to `1`, so no
        // existing recipe changes meaning. The DEFAULT is the target's:
        // `kernels/hopper` declares ON (+6% on the serve, md5-identical
        // output), which is the line that used to live in an external launch
        // script. `--ssm-batched-recurrent` on the CLI still outranks both.
        ssm_batched_recurrent: resolve_toggle(
            defaults.ssm_batched_recurrent,
            var("ATLAS_SSM_BATCHED_RECURRENT").as_deref(),
            false,
        ),
        // ⚠️ `ATLAS_GDN_PREFILL_TC` was PRESENCE-gated and is now grammar-gated
        // like its neighbours, so `=0` turns it OFF instead of on. Everything
        // that ever set it set it to `1`; the A/B recipes in
        // `GDN-PREFILL-ATTRIBUTION.md` are unaffected.
        gdn_prefill_tc: resolve_toggle(
            defaults.gdn_prefill_tc,
            var("ATLAS_GDN_PREFILL_TC").as_deref(),
            false,
        ),
        // The Hopper BA-gates twin (#928). Hopper declares it ON; the twin is
        // BIT-IDENTICAL to its gb10 parent by construction, so unlike every
        // other Hopper-owned row this one carries no accuracy question and no
        // `ATLAS_NO_*` legacy spelling — `ATLAS_SSM_BA_GATES_HOPPER=0` is the
        // whole A/B, under the 2026-09-11 grammar above.
        ssm_ba_gates_hopper: resolve_toggle(
            defaults.ssm_ba_gates_hopper,
            var("ATLAS_SSM_BA_GATES_HOPPER").as_deref(),
            false,
        ),
        // The Hopper FP8 activation-quant twin (#928, round-16 receipt § 2.1).
        // Hopper declares it ON. The twin is BIT-IDENTICAL to its gb10 parent,
        // so the row carries no accuracy question and no `ATLAS_NO_*` legacy
        // spelling — the lever is new, so there is no older script for a
        // presence rule to keep faith with. It is also not the whole rule: the
        // twin is 0.76x-0.95x at M <= 25 for K in {5120, 6144}, so it passes a
        // CTA-count floor (`layers/ops/fp8_act_quant_floor.rs`) before it takes
        // a launch. `ATLAS_FP8_ACT_QUANT_HOPPER=0` declines the twin at EVERY
        // width, which is the A/B.
        fp8_act_quant_hopper: resolve_toggle(
            defaults.fp8_act_quant_hopper,
            var("ATLAS_FP8_ACT_QUANT_HOPPER").as_deref(),
            false,
        ),
        // DECLARATION plus the legacy kill switch, and no positive variable:
        // `decode_split_silu` never had one. `ATLAS_NO_DECODE_SPLIT_SILU`
        // stays PRESENCE-gated and unchanged, so every script that predates
        // this file means what it meant.
        decode_split_silu: resolve_toggle(defaults.decode_split_silu, None, split_silu_off),
        // The paged-decode split-K policy (#928). The RULE is
        // `atlas_kernels::attn_splitk::resolve_policy`, not a fourth copy of
        // the rung order here: `spark-runtime`'s buffer arena has to reach the
        // same answer to size the split-K workspace, and it sits BELOW this
        // crate. One pure function, two callers — a second spelling is how the
        // grid comes to index past the allocation, silently, into device memory
        // it does not own. This table is still the only thing that REPORTS it.
        attn_decode_splitk: {
            let (policy, from_env) = attn_splitk::resolve_policy(
                defaults.attn_decode_splitk,
                var("ATLAS_ATTN_DECODE_SPLITK").as_deref(),
            );
            if from_env {
                Resolved::env(policy)
            } else {
                Resolved::target(policy)
            }
        },
        // Two rows for ONE kernel family, because round 6 measured the FFN
        // arm and the attention arms moving in opposite directions on the same
        // serve. `ATLAS_M16_TC` is the round-6 umbrella that arms both; it is
        // folded in HERE rather than in the consumer so that an umbrella can
        // never DISARM a target's declaration, which would make the recipe
        // depend on export order.
        ffn_m16_tc: resolve_toggle(
            defaults.ffn_m16_tc,
            var("ATLAS_FFN_M16_TC")
                .or_else(|| var("ATLAS_M16_TC"))
                .as_deref(),
            false,
        ),
        attn_m16_tc: resolve_toggle(
            defaults.attn_m16_tc,
            var("ATLAS_ATTN_M16_TC")
                .or_else(|| var("ATLAS_M16_TC"))
                .as_deref(),
            false,
        ),
        // NOT under `ATLAS_M16_TC`. The umbrella is round 6's, which predates
        // this arm and never measured it; folding the head in would silently
        // widen what an old recipe means. Its own variable, or the target's
        // declaration.
        lm_head_m16_tc: resolve_toggle(
            defaults.lm_head_m16_tc,
            var("ATLAS_LM_HEAD_M16_TC").as_deref(),
            false,
        ),
        // `ATLAS_NO_ATTN_DECODE_BATCH` is the pre-existing kill switch for the
        // whole batched attention-decode family, and it OUTRANKS both the
        // declaration and the positive variable: a switch that turns a family
        // off must not be silently narrowed by a new row underneath it.
        attn_ncol_gemv: resolve_toggle(
            defaults.attn_ncol_gemv,
            var("ATLAS_ATTN_NCOL_GEMV").as_deref(),
            var("ATLAS_NO_ATTN_DECODE_BATCH").is_some(),
        ),
        // DECLARATION plus `ATLAS_FFN_GATEUP_FUSED`, which is the A/B a Hopper
        // round runs against the new default. `=0` kills the arm and returns
        // the layer to two cuBLASLt calls; there is no positive spelling that
        // arms it on a target whose tree lacks `silu_mul_strided.cu`, because
        // the handle probe would then fail the boot audit closed.
        ffn_gateup_fused: resolve_toggle(
            defaults.ffn_gateup_fused,
            var("ATLAS_FFN_GATEUP_FUSED").as_deref(),
            false,
        ),
    }
}

/// The process-wide resolution.
///
/// `OnceLock`-cached for the reason every lever it replaces was: these are
/// read per projection per layer per step, `std::env::var` allocates and takes
/// the process-wide environment lock (measured on GB10: 0.57 us
/// single-threaded, **5.76 us at 16 threads**), and the route must be CONSTANT
/// across CUDA-graph replays — a per-call read could change the captured
/// launch set between capture and replay.
pub fn resolved() -> &'static TargetLevers {
    static LEVERS: std::sync::OnceLock<TargetLevers> = std::sync::OnceLock::new();
    LEVERS.get_or_init(|| {
        resolve(&atlas_kernels::TARGET_DEFAULTS, |name| {
            std::env::var(name).ok()
        })
    })
}

/// The baked declaration this binary carries, for the serve log's header and
/// for callers that must stay pure over their own inputs (`ModelLevers`).
pub fn declared() -> &'static atlas_kernels::TargetDefaults {
    &atlas_kernels::TARGET_DEFAULTS
}

/// `target defaults (<hw>): …` — one line naming every resolved value and
/// which came from the environment.
///
/// Built here rather than in `spark-server` so the line and the resolution are
/// the same code: a log that formats its own idea of the table is how a dead
/// lever stays invisible for a campaign (`serve_flags.rs`'s own lesson).
pub fn summary_line() -> String {
    format_levers(resolved())
}

/// [`summary_line`] over a table the caller already has — pure, so the line can
/// be graded for ANY target from a CPU test without touching the process
/// environment or sealing the `OnceLock`.
pub fn format_levers(l: &TargetLevers) -> String {
    let onoff =
        |r: Resolved<bool>| format!("{}{}", if r.value { "on" } else { "off" }, r.source.tag());
    // `u32::MAX` is the no-cap baseline, not a chosen bound. Printing
    // 4294967295 in the serve log would read as a decision someone made.
    let cap = |v: u32| {
        if v == u32::MAX {
            "max".to_string()
        } else {
            v.to_string()
        }
    };
    format!(
        "target defaults ({hw}): sm_count={sms} \
         lm_head_batchm_max={batchm}{batchm_src} \
         ssm_batched_recurrent={recurrent} gdn_prefill_tc={gdn_tc} \
         ssm_ba_gates_hopper={ba_gates} decode_split_silu={silu} \
         attn_decode_splitk={splitk}{splitk_src} ffn_m16_tc={ffn_m16_tc} \
         attn_m16_tc={attn_m16_tc} lm_head_m16_tc={lm_head_m16_tc} \
         attn_ncol_gemv={attn_ncol_gemv} ffn_gateup_fused={gateup} \
         fp8_act_quant_hopper={act_quant} \
         w8a8_prefill_max_m={w8a8_wide}/{w8a8_narrow}{w8a8_src}",
        hw = if l.hw.is_empty() { "unknown" } else { l.hw },
        // Not a resolvable lever — it is a FACT about the part, cross-checked
        // at boot against the driver. Printed on this line because the levers
        // that will read it (grid sizing) are on it, and a reader comparing
        // two campaign logs needs both in one grep.
        sms = atlas_kernels::TARGET_SM_COUNT,
        batchm = l.lm_head_batchm_max.value,
        batchm_src = l.lm_head_batchm_max.source.tag(),
        recurrent = onoff(l.ssm_batched_recurrent),
        gdn_tc = onoff(l.gdn_prefill_tc),
        ba_gates = onoff(l.ssm_ba_gates_hopper),
        act_quant = onoff(l.fp8_act_quant_hopper),
        silu = onoff(l.decode_split_silu),
        splitk = l.attn_decode_splitk.value.label(),
        splitk_src = l.attn_decode_splitk.source.tag(),
        ffn_m16_tc = onoff(l.ffn_m16_tc),
        attn_m16_tc = onoff(l.attn_m16_tc),
        lm_head_m16_tc = onoff(l.lm_head_m16_tc),
        attn_ncol_gemv = onoff(l.attn_ncol_gemv),
        gateup = onoff(l.ffn_gateup_fused),
        // Printed as widening/narrowing. `max` reads as "no cap" rather than
        // 4294967295, which would look like a number someone chose.
        w8a8_wide = cap(l.w8a8_prefill_max_m_widening.value),
        w8a8_narrow = cap(l.w8a8_prefill_max_m_narrowing.value),
        w8a8_src = l.w8a8_prefill_max_m_widening.source.tag(),
    )
}

/// The declaration a target that says nothing gets — kept in sync with
/// `atlas-kernels/build_defaults.rs::baseline` by
/// `target_defaults_tests::the_baseline_band_is_the_frozen_one`.
pub const BASELINE_BATCHM_MAX: u32 = DENSE_GEMV_BATCHM_DECODE_MAX_M;

#[cfg(test)]
#[path = "target_defaults_tests.rs"]
mod tests;
