// SPDX-License-Identifier: AGPL-3.0-only
//
// `kernels/<hw>/HARDWARE.toml` `[defaults]` — the per-target SERVING levers,
// parsed here and baked by build.rs into `atlas_kernels::TARGET_DEFAULTS`.
// Included via `#[path = "build_defaults.rs"] mod build_defaults;`.
//
// WHY THIS FILE EXISTS. Maintainer review, 2026-09-11 (tbraun96): "There is no
// arch separation at all. H100 builds compile GB10's kernel tree. Every
// Hopper/GB10 divergence is expressed as an env lever set by an H100 recipe
// living outside this repo — not as arch-selected code. 'No interference'
// rests on discipline rather than structure." Every lever below WAS a line in
// that external recipe. Declaring them beside the target's arch and memory
// facts makes the recipe a property OF the target: an H100 serve with nothing
// in its environment reproduces the measured configuration, and no GB10 serve
// can be reached by an H100 recipe, because the two are different files.
//
// Its own file, with no `super::` dependencies, so
// `tests/target_defaults.rs` can compile the SAME code against the REAL
// `kernels/*/HARDWARE.toml`: cargo never runs a build script's `#[cfg(test)]`
// modules, so a rule that lives only inside build.rs is a rule nothing tests.
// Same posture as `build_flags.rs` and `build_arch.rs`.

/// One target's `[defaults]` table, owned (build-time shape).
///
/// Mirrors `atlas_kernels::TargetDefaults` field for field; [`literal`] emits
/// that type's `const` initialiser. Two shapes rather than one because the
/// runtime type is `&'static str` + `Copy` (it is a baked constant) and a
/// parser needs `String`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Defaults {
    pub hw: String,
    pub lm_head_batchm_max: u32,
    pub ssm_batched_recurrent: bool,
    pub decode_split_silu: bool,
    pub w8a8_prefill_max_m_widening: u32,
    pub w8a8_prefill_max_m_narrowing: u32,
}

/// What a target that declares NO `[defaults]` table gets.
///
/// ★ These are the values every resolver in spark-model hardcoded before this
/// table existed, which is what makes the table additive: `kernels/metal`,
/// `kernels/strix` and `kernels/strix-hip` declare nothing and are byte-for-
/// byte unaffected. `kernels/gb10` declares exactly these values EXPLICITLY —
/// not to change anything, but so the file that describes GB10 says what GB10
/// serves with, and so `tests/target_defaults.rs` can assert the two agree.
pub(crate) fn baseline(hw: &str) -> Defaults {
    Defaults {
        hw: hw.to_string(),
        // `ops::DENSE_GEMV_BATCHM_DECODE_MAX_M` in spark-model. Duplicated as a
        // literal because atlas-kernels is BELOW spark-model in the dependency
        // graph and cannot name it; `spark-model`'s resolver asserts the two
        // agree (`target_defaults_tests::the_baseline_band_is_the_frozen_one`).
        lm_head_batchm_max: 8,
        ssm_batched_recurrent: false,
        decode_split_silu: true,
        // No cap. Absence is the correct declaration for every arch on which
        // W8A8 does not lose to W8A16 at large M, which is every arch that has
        // not measured otherwise — H100 included (2.0-3.1x at every M).
        w8a8_prefill_max_m_widening: u32::MAX,
        w8a8_prefill_max_m_narrowing: u32::MAX,
    }
}

/// What a target that declares no `[hardware] sm_count` gets.
///
/// ★ 48 — `atlas_core::device::sm121::NUM_SMS`, the GB10 constant every
/// caller hardcoded. A target that says nothing therefore resolves exactly as
/// it did, including `kernels/metal`, `kernels/strix` and
/// `kernels/strix-hip`, whose SM/CU counts this number does not describe.
pub(crate) const BASELINE_SM_COUNT: u32 = 48;

/// `[hardware] sm_count`, or [`BASELINE_SM_COUNT`].
///
/// A non-positive or non-integer declaration PANICS, for the reason an unknown
/// `[defaults]` key does: a zero SM count would divide a grid-sizing rule by
/// nothing and a string would read as "absent", i.e. as agreement with a
/// constant named after another card — the exact failure this row closes.
pub(crate) fn parse_sm_count(hw: &str, hw_toml: &toml::Value) -> u32 {
    let Some(value) = hw_toml.get("hardware").and_then(|h| h.get("sm_count")) else {
        return BASELINE_SM_COUNT;
    };
    let n = value.as_integer().unwrap_or_else(|| {
        panic!("kernels/{hw}/HARDWARE.toml: [hardware] sm_count must be an integer")
    });
    u32::try_from(n).ok().filter(|&v| v > 0).unwrap_or_else(|| {
        panic!("kernels/{hw}/HARDWARE.toml: [hardware] sm_count = {n} is not a positive u32")
    })
}

/// Read `kernels/<hw>/HARDWARE.toml` and parse its `[hardware] sm_count`.
///
/// Missing or unparseable file falls back to the baseline, exactly as
/// [`read_defaults`] does and for the same reason (the `ATLAS_SKIP_BUILD` path
/// may have no `kernels/` tree at all).
pub(crate) fn read_sm_count(kernels_root: &std::path::Path, hw: &str) -> u32 {
    let path = kernels_root.join(hw).join("HARDWARE.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return BASELINE_SM_COUNT;
    };
    let Ok(toml) = text.parse::<toml::Value>() else {
        return BASELINE_SM_COUNT;
    };
    parse_sm_count(hw, &toml)
}

/// The generated `TARGET_SM_COUNT` initialiser, appended beside
/// [`literal`]'s.
pub(crate) fn sm_count_literal(sm_count: u32) -> String {
    format!(
        "// Auto-generated by build.rs from kernels/<hw>/HARDWARE.toml [hardware] sm_count.\n\
         /// Streaming multiprocessors on the hardware this binary's kernels were\n\
         /// built for, from `kernels/<hw>/HARDWARE.toml` `[hardware] sm_count`.\n\
         ///\n\
         /// Cross-checked at boot against the driver's own\n\
         /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`\n\
         /// (`spark_runtime::cuda_backend::arch_preflight::check_sm_count`), which\n\
         /// WARNS on a mismatch and names both numbers.\n\
         pub const TARGET_SM_COUNT: u32 = {sm_count};\n"
    )
}

/// Parse `[defaults]` out of a `kernels/<hw>/HARDWARE.toml`.
///
/// Every key is optional and falls back to [`baseline`], so a target declares
/// only what it differs on. An UNKNOWN key panics: a typo'd lever name would
/// otherwise read as "this target agrees with the baseline", which is the one
/// failure mode a per-target default table must not have — it is exactly the
/// silent agreement the review objected to, re-created inside the fix.
pub(crate) fn parse_defaults(hw: &str, hw_toml: &toml::Value) -> Defaults {
    let mut out = baseline(hw);
    let Some(table) = hw_toml.get("defaults").and_then(|d| d.as_table()) else {
        return out;
    };

    let boolean = |key: &str, v: &toml::Value| -> bool {
        v.as_bool().unwrap_or_else(|| {
            panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} must be a bool")
        })
    };
    let unsigned = |key: &str, v: &toml::Value| -> u32 {
        let n = v.as_integer().unwrap_or_else(|| {
            panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} must be an integer")
        });
        u32::try_from(n).unwrap_or_else(|_| {
            panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} = {n} is not a u32")
        })
    };

    for (key, value) in table {
        match key.as_str() {
            "lm_head_batchm_max" => out.lm_head_batchm_max = unsigned(key, value),
            "w8a8_prefill_max_m_widening" => out.w8a8_prefill_max_m_widening = unsigned(key, value),
            "w8a8_prefill_max_m_narrowing" => {
                out.w8a8_prefill_max_m_narrowing = unsigned(key, value)
            }
            "ssm_batched_recurrent" => out.ssm_batched_recurrent = boolean(key, value),
            "decode_split_silu" => out.decode_split_silu = boolean(key, value),
            other => panic!(
                "kernels/{hw}/HARDWARE.toml: [defaults] has no key `{other}`. \
                 The lever list is the field list of `TargetDefaults` \
                 (crates/atlas-kernels/src/target_defaults.rs); adding a lever \
                 means adding it there, in `build_defaults.rs`, in the \
                 spark-model resolver and in every target's table, in the one \
                 commit that lands the arm which reads it."
            ),
        }
    }
    out
}

/// The generated `const` initialiser build.rs writes into `OUT_DIR`.
///
/// Emitted even under `ATLAS_SKIP_BUILD=1` (which returns before any kernel is
/// compiled): the constant is CONFIGURATION, not a kernel blob, and every CPU
/// gate — the whole test suite — runs under that flag. A skip build that
/// emitted nothing would make `TARGET_DEFAULTS` unresolvable in exactly the
/// builds that test it.
pub(crate) fn literal(d: &Defaults) -> String {
    format!(
        "// Auto-generated by build.rs from kernels/{hw}/HARDWARE.toml [defaults] — do not edit.\n\
         pub const TARGET_DEFAULTS: TargetDefaults = TargetDefaults {{\n\
         \x20   hw: \"{hw}\",\n\
         \x20   lm_head_batchm_max: {batchm},\n\
         \x20   ssm_batched_recurrent: {batched_recurrent},\n\
         \x20   decode_split_silu: {split_silu},\n\
         \x20   w8a8_prefill_max_m_widening: {w8a8_wide},\n\
         \x20   w8a8_prefill_max_m_narrowing: {w8a8_narrow},\n\
         }};\n",
        hw = d.hw,
        batchm = d.lm_head_batchm_max,
        batched_recurrent = d.ssm_batched_recurrent,
        split_silu = d.decode_split_silu,
        w8a8_wide = d.w8a8_prefill_max_m_widening,
        w8a8_narrow = d.w8a8_prefill_max_m_narrowing,
    )
}

/// Read `kernels/<hw>/HARDWARE.toml` and parse its `[defaults]`.
///
/// A MISSING or unparseable file falls back to [`baseline`] rather than
/// panicking, because this runs on the `ATLAS_SKIP_BUILD` path too, where the
/// `kernels/` tree may not be present at all (a vendored crate, a docs build).
/// The normal build already panics on a bad HARDWARE.toml in
/// `resolve_targets`, so nothing is silently excused twice.
pub(crate) fn read_defaults(kernels_root: &std::path::Path, hw: &str) -> Defaults {
    let path = kernels_root.join(hw).join("HARDWARE.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return baseline(hw);
    };
    let Ok(toml) = text.parse::<toml::Value>() else {
        return baseline(hw);
    };
    parse_defaults(hw, &toml)
}
