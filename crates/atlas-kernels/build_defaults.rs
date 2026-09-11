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
    pub cublas_gemm_scope: String,
    pub ffn_batch16_tier: bool,
    pub ffn_m16_tc: bool,
    pub attn_m16_tc: bool,
    pub attn_ncol_gemv: bool,
    pub lm_head_m16_tc: bool,
    pub lm_head_batchm_max: u32,
    pub ssm_batched_recurrent: bool,
    pub gdn_prefill_tc: bool,
    pub decode_split_silu: bool,
    pub ssm_decode_ring_slots: String,
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
        cublas_gemm_scope: "off".to_string(),
        ffn_batch16_tier: false,
        ffn_m16_tc: false,
        attn_m16_tc: false,
        attn_ncol_gemv: false,
        lm_head_m16_tc: false,
        // `ops::DENSE_GEMV_BATCHM_DECODE_MAX_M` in spark-model. Duplicated as a
        // literal because atlas-kernels is BELOW spark-model in the dependency
        // graph and cannot name it; `spark-model`'s resolver asserts the two
        // agree (`target_defaults_tests::the_baseline_band_is_the_frozen_one`).
        lm_head_batchm_max: 8,
        ssm_batched_recurrent: false,
        gdn_prefill_tc: false,
        decode_split_silu: true,
        ssm_decode_ring_slots: "auto".to_string(),
    }
}

/// Parse `[defaults]` out of a `kernels/<hw>/HARDWARE.toml`.
///
/// Every key is optional and falls back to [`baseline`], so a target declares
/// only what it differs on. An UNKNOWN key panics: a typo'd lever name would
/// otherwise read as "this target agrees with the baseline", which is the one
/// failure mode a per-target default table must not have — it is exactly the
/// silent-agreement the review objected to, re-created inside the fix.
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
    let string = |key: &str, v: &toml::Value| -> String {
        v.as_str()
            .unwrap_or_else(|| {
                panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} must be a string")
            })
            .to_string()
    };

    for (key, value) in table {
        match key.as_str() {
            "cublas_gemm_scope" => out.cublas_gemm_scope = string(key, value),
            "ffn_batch16_tier" => out.ffn_batch16_tier = boolean(key, value),
            "ffn_m16_tc" => out.ffn_m16_tc = boolean(key, value),
            "attn_m16_tc" => out.attn_m16_tc = boolean(key, value),
            "attn_ncol_gemv" => out.attn_ncol_gemv = boolean(key, value),
            "lm_head_m16_tc" => out.lm_head_m16_tc = boolean(key, value),
            "lm_head_batchm_max" => {
                let n = value.as_integer().unwrap_or_else(|| {
                    panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} must be an integer")
                });
                out.lm_head_batchm_max = u32::try_from(n).unwrap_or_else(|_| {
                    panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} = {n} is not a u32")
                });
            }
            "ssm_batched_recurrent" => out.ssm_batched_recurrent = boolean(key, value),
            "gdn_prefill_tc" => out.gdn_prefill_tc = boolean(key, value),
            "decode_split_silu" => out.decode_split_silu = boolean(key, value),
            "ssm_decode_ring_slots" => out.ssm_decode_ring_slots = string(key, value),
            other => panic!(
                "kernels/{hw}/HARDWARE.toml: [defaults] has no key `{other}`. \
                 The lever list is the field list of `TargetDefaults` \
                 (crates/atlas-kernels/src/target_defaults.rs); adding a lever \
                 means adding it there, in `build_defaults.rs` and in the \
                 spark-model resolver, in one commit."
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
         \x20   cublas_gemm_scope: \"{cublas}\",\n\
         \x20   ffn_batch16_tier: {ffn_batch16},\n\
         \x20   ffn_m16_tc: {ffn_m16},\n\
         \x20   attn_m16_tc: {attn_m16},\n\
         \x20   attn_ncol_gemv: {attn_ncol},\n\
         \x20   lm_head_m16_tc: {lm_head_m16},\n\
         \x20   lm_head_batchm_max: {batchm},\n\
         \x20   ssm_batched_recurrent: {batched_recurrent},\n\
         \x20   gdn_prefill_tc: {gdn_tc},\n\
         \x20   decode_split_silu: {split_silu},\n\
         \x20   ssm_decode_ring_slots: \"{ring}\",\n\
         }};\n",
        hw = d.hw,
        cublas = d.cublas_gemm_scope,
        ffn_batch16 = d.ffn_batch16_tier,
        ffn_m16 = d.ffn_m16_tc,
        attn_m16 = d.attn_m16_tc,
        attn_ncol = d.attn_ncol_gemv,
        lm_head_m16 = d.lm_head_m16_tc,
        batchm = d.lm_head_batchm_max,
        batched_recurrent = d.ssm_batched_recurrent,
        gdn_tc = d.gdn_prefill_tc,
        split_silu = d.decode_split_silu,
        ring = d.ssm_decode_ring_slots,
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
