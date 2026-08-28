// SPDX-License-Identifier: AGPL-3.0-only
//
// Parse helpers for build.rs. Included via `#[path = "build_parse.rs"] mod build_parse;`
// so types defined in build.rs (`SamplingCat`, `ModelTypeMatch`,
// `DflashRaw`) are reachable via `super::`.

use std::collections::HashMap;

use super::{DflashRaw, ModelTypeMatch, SamplingCat};

#[path = "build_parse_behavior.rs"]
mod behavior;
pub(super) use behavior::*;

pub(super) fn parse_kernel_toml(
    kernel_dir: &std::path::Path,
    vendor: &str,
) -> (Vec<String>, HashMap<String, String>) {
    let kernel_toml_path = kernel_dir.join("KERNEL.toml");
    let kernel_toml: toml::Value = toml::from_str(
        &std::fs::read_to_string(&kernel_toml_path)
            .unwrap_or_else(|e| panic!("{}: {e}", kernel_toml_path.display())),
    )
    .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", kernel_toml_path.display()));
    println!("cargo:rerun-if-changed={}", kernel_toml_path.display());

    // Per-vendor extra flag keys. NVIDIA reads `extra_nvcc_flags`; Apple
    // reads `extra_metal_flags`. KERNEL.toml may declare both — only the
    // vendor-matching list is forwarded so flags don't bleed across
    // toolchains (e.g. nvcc's `--fmad=false` is invalid for xcrun metal).
    let flag_key = match vendor {
        "apple" | "metal" => "extra_metal_flags",
        _ => "extra_nvcc_flags",
    };
    let extra_flags: Vec<String> = kernel_toml
        .get("build")
        .and_then(|b| b.get(flag_key))
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default();

    let module_overrides: HashMap<String, String> = kernel_toml
        .get("modules")
        .and_then(|m| m.as_table())
        .map(|t| {
            t.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect()
        })
        .unwrap_or_default();

    (extra_flags, module_overrides)
}

/// Parse `[shadow_exempt]` from a KERNEL.toml: `module = ["kernel", ...]`.
///
/// Declares `(module, kernel)` pairs a model shadow may omit without that being
/// drift — superseded entry points `common/` still carries but nothing
/// dispatches. Each needs a stated reason in the TOML comment.
///
/// WARNING-SCOPED ONLY. The caller uses this to filter the build warning; the
/// pairs stay in `TargetPtxSet::shadowed_dropped`, so the startup audit still
/// hard-errors if a model's dispatch actually resolves one. An exemption can
/// never turn a real missing kernel into a silent pass.
pub(super) fn parse_shadow_exempt(kernel_dir: &std::path::Path) -> Vec<(String, String)> {
    let path = kernel_dir.join("KERNEL.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let toml: toml::Value =
        toml::from_str(&text).unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
    let Some(table) = toml.get("shadow_exempt").and_then(|v| v.as_table()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (module, kernels) in table {
        let list = kernels.as_array().unwrap_or_else(|| {
            panic!(
                "{}: [shadow_exempt] {module} must be an array of kernel names",
                path.display()
            )
        });
        for k in list {
            let name = k.as_str().unwrap_or_else(|| {
                panic!(
                    "{}: [shadow_exempt] {module} entries must be strings",
                    path.display()
                )
            });
            out.push((module.clone(), name.to_string()));
        }
    }
    out.sort();
    out
}

/// Parse `[expected_absent]` from a MODEL.toml.
///
/// ```toml
/// [expected_absent.mla_absorbed]
/// mla_batched_gemv = "no MLA: this checkpoint is standard GQA (kv_lora_rank 0)"
/// ```
///
/// Each entry names a `(module, kernel)` lookup this model's dispatch may issue
/// and fail to resolve WITHOUT that being an error. The value is a MANDATORY
/// reason — a bare list would become a place to silence the boot gate, which is
/// the failure mode the gate exists to remove. TRANSITIONAL: the right fix is
/// to gate the lookup on config so it is never issued (see
/// `qwen3_attention::init_arch_gates`). Anything neither gated nor listed here
/// fails the boot audit.
pub(super) fn parse_expected_absent(model_dir: &std::path::Path) -> Vec<(String, String)> {
    let path = model_dir.join("MODEL.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let toml: toml::Value =
        toml::from_str(&text).unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
    let Some(table) = toml.get("expected_absent").and_then(|v| v.as_table()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (module, kernels) in table {
        let entries = kernels.as_table().unwrap_or_else(|| {
            panic!(
                "{}: [expected_absent.{module}] must be a table of `kernel = \"reason\"` \
                 — every entry needs a stated reason",
                path.display()
            )
        });
        for (kernel, reason) in entries {
            let reason = reason.as_str().unwrap_or_else(|| {
                panic!(
                    "{}: [expected_absent.{module}] {kernel} must be a reason string",
                    path.display()
                )
            });
            assert!(
                !reason.trim().is_empty(),
                "{}: [expected_absent.{module}] {kernel} needs a stated reason",
                path.display()
            );
            out.push((module.clone(), kernel.clone()));
        }
    }
    out.sort();
    out
}

/// Parse sampling presets from MODEL.toml `[sampling.*]` sections.
pub(super) fn parse_sampling_presets(
    model_dir: &std::path::Path,
) -> (SamplingCat, SamplingCat, SamplingCat, SamplingCat) {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return (
            SamplingCat::default(),
            SamplingCat::default(),
            SamplingCat::default(),
            SamplingCat::default(),
        );
    }
    println!("cargo:rerun-if-changed={}", model_toml_path.display());
    let content = std::fs::read_to_string(&model_toml_path)
        .unwrap_or_else(|e| panic!("{}: {e}", model_toml_path.display()));
    let toml: toml::Value = toml::from_str(&content)
        .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", model_toml_path.display()));

    let parse_cat = |key: &str| -> SamplingCat {
        let section = toml.get("sampling").and_then(|s| s.get(key));
        match section {
            Some(v) => SamplingCat {
                temperature: v
                    .get("temperature")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.7) as f32,
                top_p: v.get("top_p").and_then(|t| t.as_float()).unwrap_or(0.95) as f32,
                top_k: v.get("top_k").and_then(|t| t.as_integer()).unwrap_or(20) as u32,
                presence_penalty: v
                    .get("presence_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                frequency_penalty: v
                    .get("frequency_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                repetition_penalty: v
                    .get("repetition_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(1.0) as f32,
                dry_multiplier: v
                    .get("dry_multiplier")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                dry_base: v.get("dry_base").and_then(|t| t.as_float()).unwrap_or(1.75) as f32,
                dry_allowed_length: v
                    .get("dry_allowed_length")
                    .and_then(|t| t.as_integer())
                    .unwrap_or(2) as u32,
                lz_penalty: v
                    .get("lz_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                // NO unwrap_or: an absent min_p must stay None so the
                // server's --default-min-p keeps owning the field.
                min_p: v.get("min_p").and_then(|t| t.as_float()).map(|p| p as f32),
                // Same rule as min_p: absent stays None so --default-top-n-sigma
                // keeps owning the field for every model that does not declare it.
                top_n_sigma: v
                    .get("top_n_sigma")
                    .and_then(|t| t.as_float())
                    .map(|p| p as f32),
            },
            None => SamplingCat::default(),
        }
    };

    (
        parse_cat("thinking_text"),
        parse_cat("thinking_coding"),
        parse_cat("non_thinking"),
        parse_cat("tools"),
    )
}

/// Parse `[[model_types]]` from MODEL.toml.
///
/// Each entry maps a `(model_type, optional hidden_size)` pair to this kernel target.
/// Missing `hidden_size` = wildcard (matches any hidden_size not caught by a more specific entry).
pub(super) fn parse_model_types(model_dir: &std::path::Path) -> Vec<ModelTypeMatch> {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return Vec::new();
    }
    let content = std::fs::read_to_string(&model_toml_path).unwrap_or_default();
    let toml: toml::Value = match toml::from_str(&content) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let Some(entries) = toml.get("model_types").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let mt = entry.get("model_type")?.as_str()?.to_string();
            let hs = entry
                .get("hidden_size")
                .and_then(|v| v.as_integer())
                .map(|v| v as usize);
            Some(ModelTypeMatch {
                model_type: mt,
                hidden_size: hs,
            })
        })
        .collect()
}

/// Parse `[model] match_names` from MODEL.toml.
///
/// Checkpoint-reference needles (case-insensitive substrings of the HF id /
/// `--model-name` / resolved model dir) that identify checkpoints this
/// target serves. Consulted at runtime only to break a tie when several
/// targets declare the same `(model_type, hidden_size)` — see
/// `src/resolve.rs`. Empty entries are rejected: an empty needle would
/// match every reference and silently win every tie.
pub(super) fn parse_match_names(model_dir: &std::path::Path) -> Vec<String> {
    let path = model_dir.join("MODEL.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let toml: toml::Value =
        toml::from_str(&text).unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
    let Some(arr) = toml
        .get("model")
        .and_then(|m| m.get("match_names"))
        .and_then(|v| v.as_array())
    else {
        return Vec::new();
    };
    arr.iter()
        .map(|v| {
            let s = v.as_str().unwrap_or_else(|| {
                panic!(
                    "{}: [model] match_names entries must be strings",
                    path.display()
                )
            });
            assert!(
                !s.trim().is_empty(),
                "{}: [model] match_names entries must be non-empty — an empty needle \
                 would match every checkpoint reference",
                path.display()
            );
            // The needles are emitted into generated Rust as `"{needle}"`
            // string literals (build_codegen.rs) with no escaping; a quote
            // or backslash would produce an uncompilable target_ptx.rs with
            // an error pointing nowhere near this file. Reject here, where
            // the operator can see which TOML entry to fix. (No legitimate
            // HF id / model-dir needle contains either character.)
            assert!(
                !s.contains('"') && !s.contains('\\'),
                "{}: [model] match_names entry {s:?} contains a quote or backslash — \
                 needles are emitted verbatim into generated Rust string literals \
                 and checkpoint references never contain these characters",
                path.display()
            );
            s.to_string()
        })
        .collect()
}

/// Parse `[model] kernel_source` from MODEL.toml.
///
/// Names ANOTHER kernel-target directory whose per-quant kernel sources
/// this target compiles instead of shipping its own copies (SSOT for
/// architecturally-identical checkpoints — qwen3.8-27b reuses
/// qwen3.6-27b's .cu tree verbatim). Everything else in MODEL.toml
/// (sampling, behavior, model_types, match_names, expected_absent) still
/// belongs to this target. `build.rs` validates the referent exists and
/// refuses redirect chains.
pub(super) fn parse_kernel_source(model_dir: &std::path::Path) -> Option<String> {
    let path = model_dir.join("MODEL.toml");
    let text = std::fs::read_to_string(&path).ok()?;
    let toml: toml::Value =
        toml::from_str(&text).unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
    let src = toml.get("model")?.get("kernel_source")?;
    let s = src
        .as_str()
        .unwrap_or_else(|| panic!("{}: [model] kernel_source must be a string", path.display()));
    assert!(
        !s.trim().is_empty(),
        "{}: [model] kernel_source must name a kernel target directory",
        path.display()
    );
    Some(s.to_string())
}

/// Parse `[dflash]` from MODEL.toml. Returns `None` when the section is
/// absent (model has no DFlash drafter pairing). The build emits the parsed
/// values into the static `TargetPtxSet::dflash` field that spark-server
/// reads at runtime to satisfy `--dflash` without an explicit `--draft-model`.
pub(super) fn parse_dflash(model_dir: &std::path::Path) -> Option<DflashRaw> {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&model_toml_path).unwrap_or_default();
    let toml: toml::Value = toml::from_str(&content).ok()?;
    let dflash = toml.get("dflash")?;
    let draft_model = dflash.get("draft_model")?.as_str()?.to_string();
    let gamma = dflash
        .get("gamma")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize)
        .unwrap_or(16);
    let window_size = dflash
        .get("window_size")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize)
        .unwrap_or(0);
    let mask_token_id = dflash
        .get("mask_token_id")
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(0);
    let target_layer_ids: Vec<usize> = dflash
        .get("target_layer_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_integer().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_default();
    Some(DflashRaw {
        draft_model,
        gamma,
        window_size,
        mask_token_id,
        target_layer_ids,
    })
}
