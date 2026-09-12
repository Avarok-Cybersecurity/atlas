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

/// Every serving lever this target declares, resolved against the environment.
///
/// Field order is the order the serve log prints them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetLevers {
    /// `kernels/<hw>` this binary was compiled from, for the log line only.
    pub hw: &'static str,
    pub lm_head_batchm_max: Resolved<u32>,
    pub ssm_batched_recurrent: Resolved<bool>,
    pub decode_split_silu: Resolved<bool>,
    /// The `w8a16_gemm_m16` tier on the dense-FFN decode arm (#927).
    pub ffn_m16_tc: Resolved<bool>,
    /// The `w8a16_gemm_m16` tiers on the decode Q/K/V and o_proj (#927).
    pub attn_m16_tc: Resolved<bool>,
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
        // DECLARATION plus the legacy kill switch, and no positive variable:
        // `decode_split_silu` never had one. `ATLAS_NO_DECODE_SPLIT_SILU`
        // stays PRESENCE-gated and unchanged, so every script that predates
        // this file means what it meant.
        decode_split_silu: resolve_toggle(defaults.decode_split_silu, None, split_silu_off),
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
    format!(
        "target defaults ({hw}): sm_count={sms} \
         lm_head_batchm_max={batchm}{batchm_src} \
         ssm_batched_recurrent={recurrent} decode_split_silu={silu} \
         ffn_m16_tc={ffn_m16_tc} attn_m16_tc={attn_m16_tc}",
        hw = if l.hw.is_empty() { "unknown" } else { l.hw },
        // Not a resolvable lever — it is a FACT about the part, cross-checked
        // at boot against the driver. Printed on this line because the levers
        // that will read it (grid sizing) are on it, and a reader comparing
        // two campaign logs needs both in one grep.
        sms = atlas_kernels::TARGET_SM_COUNT,
        batchm = l.lm_head_batchm_max.value,
        batchm_src = l.lm_head_batchm_max.source.tag(),
        recurrent = onoff(l.ssm_batched_recurrent),
        silu = onoff(l.decode_split_silu),
        ffn_m16_tc = onoff(l.ffn_m16_tc),
        attn_m16_tc = onoff(l.attn_m16_tc),
    )
}

/// The declaration a target that says nothing gets — kept in sync with
/// `atlas-kernels/build_defaults.rs::baseline` by
/// `target_defaults_tests::the_baseline_band_is_the_frozen_one`.
pub const BASELINE_BATCHM_MAX: u32 = DENSE_GEMV_BATCHM_DECODE_MAX_M;

#[cfg(test)]
#[path = "target_defaults_tests.rs"]
mod tests;
