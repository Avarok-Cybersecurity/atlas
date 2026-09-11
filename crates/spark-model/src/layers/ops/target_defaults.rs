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
//! The legacy `ATLAS_NO_*` kill switches (`ATLAS_FFN_NO_BATCH16`,
//! `ATLAS_NO_DECODE_SPLIT_SILU`) stay PRESENCE-gated and still force their
//! lever OFF, so no script that predates this file changes meaning.
//! `ATLAS_NO_GDN_HOPPER` is the third, and it keeps the `== "1"` spelling it
//! shipped with rather than the presence rule — same principle, applied to the
//! grammar that variable was documented and used with.
//!
//! # One resolution, one log line
//!
//! [`resolved`] is the SSOT: every consumer below reads it, and
//! `spark-server` prints it as `target defaults (<hw>): …` with the
//! environment-sourced values marked. A lever resolved in two places is a
//! lever that can disagree with the line that claims to report it.

use super::dispatch_config::{CublasScope, parse_cublas_scope};
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
/// NEGATIVE above it (-14.4% at C=16, commits 84d5b763c / 78d276832). H100
/// declares 16 because on that machine the TILE GEMM is what loses at decode
/// widths (#927: 224 ms/step at 16 active rows).
///
/// Clamped to [`DENSE_GEMV_BATCHM_MAX_M`], the kernel's compile-time row
/// bound — `dense_gemv_batchm` refuses above it rather than writing 16 of m
/// rows, and a lever that produced an `Err` at every decode step would be a
/// worse failure than ignoring the excess. An unparseable or `0` environment
/// value keeps the target's declaration; the value is a BAND, not a switch, so
/// there is no "off".
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

/// `ssm_decode_ring_slots`: `auto` (size it at preflight, #915) or a depth.
///
/// An unparseable declaration resolves to `auto` rather than failing the boot:
/// the ring's auto-fit is the safe arm, and `spark-server`'s
/// `parse_decode_ring_slots` is the validator for the operator-facing spelling.
pub fn resolve_ring_slots(declared: &str) -> Option<usize> {
    if declared.trim() == "auto" {
        return None;
    }
    declared
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|&n| n <= atlas_kernels::DECODE_ROLLBACK_RING_SLOTS)
}

/// Every serving lever this target declares, resolved against the environment.
///
/// Field order is the order the serve log prints them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetLevers {
    /// `kernels/<hw>` this binary was compiled from, for the log line only.
    pub hw: &'static str,
    pub cublas: Resolved<CublasScope>,
    pub ffn_batch16_tier: Resolved<bool>,
    pub ffn_m16_tc: Resolved<bool>,
    pub attn_m16_tc: Resolved<bool>,
    pub attn_ncol_gemv: Resolved<bool>,
    pub lm_head_m16_tc: Resolved<bool>,
    pub lm_head_batchm_max: Resolved<u32>,
    pub ssm_batched_recurrent: Resolved<bool>,
    pub gdn_decode_hopper: Resolved<bool>,
    pub gdn_prefill_tc: Resolved<bool>,
    pub ssm_ba_gates_hopper: Resolved<bool>,
    pub decode_split_silu: Resolved<bool>,
    pub ssm_decode_ring_slots: Resolved<Option<usize>>,
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
    let present = |var: &mut dyn FnMut(&str) -> Option<String>, name: &str| var(name).is_some();

    let cublas_raw = var("ATLAS_CUBLAS_GEMM");
    let cublas = match &cublas_raw {
        Some(raw) => Resolved::env(parse_cublas_scope(Some(raw)).0),
        None => Resolved::target(parse_cublas_scope(Some(defaults.cublas_gemm_scope)).0),
    };

    let batch16_raw = var("ATLAS_FFN_BATCH16");
    let batch16_off = present(&mut var, "ATLAS_FFN_NO_BATCH16");
    let split_silu_off = present(&mut var, "ATLAS_NO_DECODE_SPLIT_SILU");
    // NOT `present`: see the `gdn_decode_hopper` note below for why this one
    // kill switch keeps its `== "1"` spelling.
    let gdn_hopper_off = var("ATLAS_NO_GDN_HOPPER").as_deref() == Some("1");
    // `ATLAS_M16_TC` is the round-6 UMBRELLA: it arms both M16 tensor-core
    // tiers at once. Kept because that is the recipe round 6 was measured
    // with; it can only turn them ON, never off, so a target that declares one
    // of them on is not silently disarmed by an umbrella somebody exported.
    let umbrella = var("ATLAS_M16_TC");
    let umbrella_on = resolve_toggle(false, umbrella.as_deref(), false).value;
    let arm = |lever: Resolved<bool>| -> Resolved<bool> {
        if umbrella_on && !lever.value {
            Resolved::env(true)
        } else {
            lever
        }
    };

    let ffn_m16 = resolve_toggle(
        defaults.ffn_m16_tc,
        var("ATLAS_FFN_M16_TC").as_deref(),
        false,
    );
    let attn_m16 = resolve_toggle(
        defaults.attn_m16_tc,
        var("ATLAS_ATTN_M16_TC").as_deref(),
        false,
    );

    TargetLevers {
        hw: defaults.hw,
        cublas,
        ffn_batch16_tier: resolve_toggle(
            defaults.ffn_batch16_tier,
            batch16_raw.as_deref(),
            batch16_off,
        ),
        ffn_m16_tc: arm(ffn_m16),
        attn_m16_tc: arm(attn_m16),
        attn_ncol_gemv: resolve_toggle(
            defaults.attn_ncol_gemv,
            var("ATLAS_ATTN_NCOL_GEMV").as_deref(),
            false,
        ),
        lm_head_m16_tc: resolve_toggle(
            defaults.lm_head_m16_tc,
            var("ATLAS_LM_HEAD_M16_TC").as_deref(),
            false,
        ),
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
        ssm_batched_recurrent: resolve_toggle(
            defaults.ssm_batched_recurrent,
            var("ATLAS_SSM_BATCHED_RECURRENT").as_deref(),
            false,
        ),
        // The Hopper GDN DECODE twins (#927). Every target declares them OFF:
        // they are bit-identical to their gb10 parents, so the row is a pure
        // speed claim, and H100 round 12 measured it negative three ways —
        // 0.83x at contiguous n=1 in the microtest, +6.8% per C=1 step in nsys,
        // -0.4% on the serve A/B (`GDN-DECODE-ATTRIBUTION.md`).
        //
        // ⚠️ `ATLAS_NO_GDN_HOPPER` keeps its ORIGINAL `== "1"` spelling, not the
        // presence rule the other two legacy kill switches use. It shipped
        // documented as "`=1` and not presence, so `ATLAS_NO_GDN_HOPPER=0` does
        // NOT disable the tier"; making it presence-gated here would change what
        // an existing `=0` in a recipe means, which is the one thing the legacy
        // rung exists to prevent. It still OUTRANKS the positive lever, the way
        // `ATLAS_FFN_NO_BATCH16` outranks `ATLAS_FFN_BATCH16`.
        gdn_decode_hopper: resolve_toggle(
            defaults.gdn_decode_hopper,
            var("ATLAS_GDN_DECODE_HOPPER").as_deref(),
            gdn_hopper_off,
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
        decode_split_silu: resolve_toggle(defaults.decode_split_silu, None, split_silu_off),
        // DECLARATION ONLY — never an environment read. `ATLAS_SSM_DECODE_RING`
        // has its own grammar (`1` = the full depth, `0` = no ring) and its own
        // precedence rung, BELOW the published depth, in
        // `ssm_reserve::decode_rollback_ring_slots_with`. Reading it here as a
        // plain integer would give `=1` two contradictory meanings in one
        // binary. `spark-server` publishes a non-`auto` declaration as the
        // default depth, where the CLI's explicit `N` still outranks it.
        ssm_decode_ring_slots: Resolved::target(resolve_ring_slots(defaults.ssm_decode_ring_slots)),
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
/// for the preflight that compares it against the running device.
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
    let c = l.cublas.value;
    let scope = if c.any() {
        let mut names = Vec::new();
        for (on, name) in [
            (c.ffn, "ffn"),
            (c.attn, "attn"),
            (c.ssm, "ssm"),
            (c.head, "head"),
        ] {
            if on {
                names.push(name);
            }
        }
        names.join(",")
    } else {
        "off".to_string()
    };
    format!(
        "target defaults ({hw}): cublas_gemm_scope={scope}{cublas_src} \
         ffn_batch16_tier={batch16} ffn_m16_tc={ffn_m16} attn_m16_tc={attn_m16} \
         attn_ncol_gemv={ncol} lm_head_m16_tc={head_m16} \
         lm_head_batchm_max={batchm}{batchm_src} ssm_batched_recurrent={recurrent} \
         gdn_decode_hopper={gdn_decode} gdn_prefill_tc={gdn_tc} \
         ssm_ba_gates_hopper={ba_gates} decode_split_silu={silu} \
         ssm_decode_ring_slots={ring}{ring_src} \
         w8a8_prefill_max_m={w8a8_wide}/{w8a8_narrow}{w8a8_src}",
        hw = if l.hw.is_empty() { "unknown" } else { l.hw },
        cublas_src = l.cublas.source.tag(),
        batch16 = onoff(l.ffn_batch16_tier),
        ffn_m16 = onoff(l.ffn_m16_tc),
        attn_m16 = onoff(l.attn_m16_tc),
        ncol = onoff(l.attn_ncol_gemv),
        head_m16 = onoff(l.lm_head_m16_tc),
        batchm = l.lm_head_batchm_max.value,
        batchm_src = l.lm_head_batchm_max.source.tag(),
        recurrent = onoff(l.ssm_batched_recurrent),
        gdn_decode = onoff(l.gdn_decode_hopper),
        gdn_tc = onoff(l.gdn_prefill_tc),
        ba_gates = onoff(l.ssm_ba_gates_hopper),
        silu = onoff(l.decode_split_silu),
        ring = match l.ssm_decode_ring_slots.value {
            Some(n) => n.to_string(),
            None => "auto".to_string(),
        },
        ring_src = l.ssm_decode_ring_slots.source.tag(),
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
