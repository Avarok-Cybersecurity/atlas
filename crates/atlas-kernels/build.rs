// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

/// Per-category sampling defaults parsed from MODEL.toml `[sampling.*]`.
#[derive(Debug, Clone)]
struct SamplingCat {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repetition_penalty: f32,
    // DRY sampler params (see SamplingCategory in atlas-kernels/src/lib.rs
    // for full rationale). Defaults disable DRY; individual MODEL.toml
    // `[sampling.*]` tables opt in when needed.
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: u32,
    // LZ penalty (arXiv:2504.20131). Frequency-weighted n-gram penalty
    // over the recent token window. 0.0 = disabled. 0.2 is the SGLang
    // reference value; lossless on AIME/GPQA at that strength.
    lz_penalty: f32,
}

impl Default for SamplingCat {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
        }
    }
}

/// A `(model_type, optional hidden_size)` pair declaring which models a kernel target supports.
#[derive(Debug, Clone)]
struct ModelTypeMatch {
    model_type: String,
    hidden_size: Option<usize>,
}

/// A resolved (hw, model, quant) compilation target.
struct Target {
    hw: String,
    model: String,
    quant: String,
    arch: String,
    /// Per-model quant dir (for KERNEL.toml and optional override .cu files).
    model_kernel_dir: PathBuf,
    /// Common quant dir (hw_dir/quant/) with shared .cu files.
    common_kernel_dir: Option<PathBuf>,
    extra_flags: Vec<String>,
    module_overrides: HashMap<String, String>,
    sampling_thinking_text: SamplingCat,
    sampling_thinking_coding: SamplingCat,
    sampling_non_thinking: SamplingCat,
    sampling_tools: SamplingCat,
    behavior_thinking_in_tools: bool,
    behavior_max_thinking_budget: u32,
    behavior_thinking_default: bool,
    behavior_fp8_kv_calibration_tokens: usize,
    behavior_default_kv_dtype: String,
    behavior_default_num_drafts: u32,
    behavior_disable_tool_steering: bool,
    behavior_tool_call_parser: String,
    behavior_enable_loop_watchdog: bool,
    behavior_min_p_floor: f32,
    behavior_temperature_max: f32,
    behavior_think_loop_min_repeats: u32,
    behavior_think_loop_scan_window: u32,
    behavior_confidence_early_stop: bool,
    behavior_confidence_run_length: u32,
    behavior_fuzzy_repeat_tolerance_div: u32,
    behavior_max_inter_tool_prose: u32,
    behavior_max_post_think_content_tokens: u32,
    behavior_tscg: bool,
    behavior_disable_tool_grammar: bool,
    behavior_rollback_resteer: bool,
    behavior_rom_head: String,
    behavior_tool_retry: bool,
    /// Which `(model_type, hidden_size)` pairs this kernel target supports.
    /// Parsed from `[[model_types]]` in MODEL.toml.
    model_type_matches: Vec<ModelTypeMatch>,
    /// `[dflash]` section if present in MODEL.toml — drafter pairing for
    /// block-diffusion speculative decoding. `None` when the model has no
    /// associated DFlash drafter checkpoint.
    dflash: Option<DflashRaw>,
}

#[derive(Default, Clone)]
struct DflashRaw {
    draft_model: String,
    gamma: usize,
    window_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    // Without these the cargo cache short-circuits when only the target
    // selection env vars change, leaving an out-of-date kernel registry
    // baked into the binary (e.g. defaulting to qwen3-next-only after a
    // prior `ATLAS_TARGET_MODEL=*` build).
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_MODEL");
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_QUANT");
    // ATLAS_EXTRA_NVCC_FLAGS — global nvcc-flag override read by
    // `build_target::NvidiaTarget::compile`. Used for kernel bisection
    // tests (e.g. `-DATLAS_FAST_SOFTMAX_EXP=1` to flip the softmax
    // polynomial vs `__expf` `#ifdef` choice).
    println!("cargo:rerun-if-env-changed=ATLAS_EXTRA_NVCC_FLAGS");
    // Auto-skip the kernel build on macOS unless an explicit Apple Metal
    // target was selected. The default `gb10` target is NVIDIA-only and
    // cannot find nvcc on a Mac. Phase 2 onwards will populate
    // `kernels/metal/` and let `ATLAS_TARGET_HW=metal` drive a real build.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let hw_explicit = env::var("ATLAS_TARGET_HW").is_ok();
    let auto_skip_macos = target_os == "macos" && !hw_explicit;
    let skip_env = matches!(
        env::var("ATLAS_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    );
    if skip_env || auto_skip_macos {
        // Stub both the cuda-side (`ptx_modules`, `all_ptx_sets`) AND
        // the metal-side (`metallib_modules`) generated APIs so any
        // consumer of either keeps type-checking under the skip path.
        let stub = "// Auto-generated by build.rs (skip stub — no kernel compiler invoked).\n\
            pub fn ptx_modules() -> Vec<(&'static str, &'static str)> { Vec::new() }\n\
            pub fn metallib_modules() -> Vec<(&'static str, &'static [u8])> { Vec::new() }\n\
            pub fn all_ptx_sets() -> Vec<TargetPtxSet> { Vec::new() }\n";
        std::fs::write(out_dir.join("target_ptx.rs"), stub).expect("write skip stub target_ptx.rs");
        println!(
            "cargo:rustc-env=ATLAS_KERNEL_SET_HASH={}",
            content_hash(stub)
        );
        println!("cargo:rustc-env=ATLAS_PTX_DIR={}", out_dir.display());
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    // ── Resolve targets (supports wildcards) ──
    let targets = resolve_targets(workspace_root);

    assert!(
        !targets.is_empty(),
        "No kernel targets resolved. Check ATLAS_TARGET_* env vars."
    );

    // ── Resolve compute target (compiler) from HARDWARE.toml vendor ──
    // This abstraction supports NVIDIA (nvcc→PTX), AMD (hipcc→HSACO),
    // Apple (xcrun→metallib), Intel (icpx→SPIR-V). Only NVIDIA is implemented.
    let hw_dir = workspace_root
        .join("kernels")
        .join(env::var("ATLAS_TARGET_HW").unwrap_or_else(|_| "gb10".into()));
    let hw_toml_path = hw_dir.join("HARDWARE.toml");
    let hw_toml: toml::Value = {
        let content = std::fs::read_to_string(&hw_toml_path)
            .unwrap_or_else(|_| panic!("Cannot read {}", hw_toml_path.display()));
        content
            .parse()
            .unwrap_or_else(|e| panic!("Invalid HARDWARE.toml: {e}"))
    };
    let vendor_str = hw_toml
        .get("hardware")
        .and_then(|h| h.get("vendor"))
        .and_then(|v| v.as_str());
    let compute_target = resolve_compute_target(vendor_str);
    let output_ext = compute_target.output_extension();
    let output_is_text = compute_target.output_is_text();

    // Per-target: (target_idx, vec of (stem, module_name))
    let mut all_target_modules: Vec<Vec<(String, String)>> = Vec::new();

    // 2026-05-24 dedup+parallel: pre-walk every (target, cu_file) pair
    // and split into two queues:
    //   - `compile_jobs`: unique (source, arch, sorted_flags) tuples
    //     that need nvcc — run in parallel via thread::scope.
    //   - `copy_jobs`: cache-hits where another target already produced
    //     this exact binary — run sequentially after compile (microsec
    //     each, no point parallelising).
    //
    // Pre-dedup baseline on 13 model targets × ~85 shared kernels
    // = ~1100 sequential nvcc invocations. After dedup: ~85 × 2
    // flag-variants = ~170 compiles, ~930 file copies. After parallel:
    // 170 / N_CORES nvcc batches. Net ~30-50× wall-clock improvement on
    // a 16-core build host.
    struct CompileJob {
        cu_file: std::path::PathBuf,
        arch: String,
        extra_flags: Vec<String>,
        out_file: std::path::PathBuf,
    }
    let mut compile_jobs: Vec<CompileJob> = Vec::new();
    let mut copy_jobs: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    let mut compile_cache: std::collections::HashMap<
        (std::path::PathBuf, String, Vec<String>),
        std::path::PathBuf,
    > = std::collections::HashMap::new();

    let source_ext = compute_target.source_extension();
    for (idx, target) in targets.iter().enumerate() {
        let cu_files = collect_cu_files(
            target.common_kernel_dir.as_deref(),
            &target.model_kernel_dir,
            source_ext,
        );
        assert!(
            !cu_files.is_empty(),
            "No .{} files found for target ({}, {}, {})",
            compute_target.source_extension(),
            target.hw,
            target.model,
            target.quant,
        );

        // Gather work for this target (no compilation yet — that runs
        // in parallel after the full work plan is built).
        for cu_file in &cu_files {
            let stem = cu_file.file_stem().unwrap().to_str().unwrap().to_string();
            let out_file = out_dir.join(format!("t{idx}__{stem}.{output_ext}"));

            // Dedup key: same (source, arch, sorted-flags) → identical
            // binary output. Sort flags so flag-order doesn't bust the cache.
            let mut sorted_flags = target.extra_flags.clone();
            sorted_flags.sort();
            let key = (cu_file.clone(), target.arch.clone(), sorted_flags);

            if let Some(existing) = compile_cache.get(&key) {
                copy_jobs.push((existing.clone(), out_file));
            } else {
                compile_jobs.push(CompileJob {
                    cu_file: cu_file.clone(),
                    arch: target.arch.clone(),
                    extra_flags: target.extra_flags.clone(),
                    out_file: out_file.clone(),
                });
                compile_cache.insert(key, out_file);
            }
        }

        for cu_file in &cu_files {
            println!("cargo:rerun-if-changed={}", cu_file.display());
        }

        // Collect (stem, module_name) pairs sorted by module_name
        let mut modules: Vec<(String, String)> = cu_files
            .iter()
            .map(|f| {
                let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
                let module_name = target
                    .module_overrides
                    .get(&stem)
                    .cloned()
                    .unwrap_or_else(|| stem.clone());
                (stem, module_name)
            })
            .collect();
        modules.sort_by(|a, b| a.1.cmp(&b.1));

        all_target_modules.push(modules);

        println!(
            "cargo:rerun-if-changed={}",
            target.model_kernel_dir.display()
        );
        if let Some(ref common) = target.common_kernel_dir {
            println!("cargo:rerun-if-changed={}", common.display());
        }
        let n_overrides = find_cu_files(&target.model_kernel_dir, source_ext).len();
        println!(
            "cargo:warning=atlas-kernels: compiled {} kernels for target {} ({}, {}, {}){}",
            cu_files.len(),
            idx,
            target.hw,
            target.model,
            target.quant,
            if n_overrides > 0 {
                format!(" ({n_overrides} model-specific overrides)")
            } else {
                String::new()
            },
        );
    }

    // ── Parallel compile of unique jobs ──
    let nvcc_invocations = compile_jobs.len();
    let cache_hits = copy_jobs.len();
    let total = nvcc_invocations + cache_hits;

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(nvcc_invocations.max(1));

    if nvcc_invocations > 0 {
        let compute = &*compute_target;
        let errors_mutex: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        let next_idx = std::sync::atomic::AtomicUsize::new(0);

        std::thread::scope(|s| {
            for _ in 0..n_threads {
                let next_idx = &next_idx;
                let compile_jobs = &compile_jobs;
                let errors_mutex = &errors_mutex;
                s.spawn(move || {
                    loop {
                        let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= compile_jobs.len() {
                            break;
                        }
                        let job = &compile_jobs[i];
                        if let Err(e) = compute.compile(
                            &job.cu_file,
                            &job.out_file,
                            &job.arch,
                            &job.extra_flags,
                        ) {
                            errors_mutex.lock().unwrap().push(e);
                        }
                    }
                });
            }
        });

        let errors = errors_mutex.into_inner().unwrap();
        if !errors.is_empty() {
            panic!("Kernel compilation failed:\n{}", errors.join("\n"));
        }
    }

    // ── Sequential copy for cache-hits (microseconds each, no point parallelising) ──
    for (src, dst) in &copy_jobs {
        std::fs::copy(src, dst).unwrap_or_else(|e| {
            panic!(
                "Failed to copy cached {} → {}: {e}",
                src.display(),
                dst.display(),
            )
        });
    }

    // Dedup+parallel summary.
    if total > 0 {
        println!(
            "cargo:warning=atlas-kernels: dedup+parallel: {nvcc_invocations}/{total} unique nvcc \
             invocations ({cache_hits} cache hits, {:.1}× dedup), {n_threads} parallel workers",
            total as f64 / nvcc_invocations.max(1) as f64,
        );
    }

    // ── Generate target_ptx.rs ──
    let generated =
        generate_target_ptx_rs(&targets, &all_target_modules, output_ext, output_is_text);
    let gen_path = out_dir.join("target_ptx.rs");
    std::fs::write(&gen_path, &generated)
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", gen_path.display()));

    // Force atlas-kernels lib recompilation whenever the generated kernel set
    // changes. cargo does NOT track this build-script-generated `include!`d
    // file as a recompile trigger, so without this the lib can embed a STALE
    // module set (the 2026-06-04 98-vs-99 / dropped-pipelined-GEMM bug). The
    // content hash is surfaced as a rustc-env that lib.rs references via env!;
    // a changed hash invalidates the crate's fingerprint → fresh rebuild.
    println!(
        "cargo:rustc-env=ATLAS_KERNEL_SET_HASH={}",
        content_hash(&generated)
    );
    println!("cargo:rustc-env=ATLAS_PTX_DIR={}", out_dir.display());
}

/// FNV-1a 64-bit content fingerprint → 12 hex chars. Deterministic, no deps.
/// Used to force atlas-kernels recompilation when the generated kernel set
/// changes (cargo doesn't track build-script-generated `include!` files).
fn content_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

/// Resolve all compilation targets from env vars, expanding wildcards.
fn resolve_targets(workspace_root: &std::path::Path) -> Vec<Target> {
    let hw = env::var("ATLAS_TARGET_HW").unwrap_or_else(|_| "gb10".into());
    let model_spec = env::var("ATLAS_TARGET_MODEL").unwrap_or_else(|_| "qwen3-next-80b-a3b".into());
    let quant_spec = env::var("ATLAS_TARGET_QUANT").unwrap_or_else(|_| "nvfp4".into());

    let hw_dir = workspace_root.join("kernels").join(&hw);
    assert!(
        hw_dir.is_dir(),
        "Hardware kernel directory not found: {}",
        hw_dir.display()
    );

    // Parse HARDWARE.toml (shared across all models for this hw)
    let hw_toml_path = hw_dir.join("HARDWARE.toml");
    let hw_toml: toml::Value = toml::from_str(
        &std::fs::read_to_string(&hw_toml_path)
            .unwrap_or_else(|e| panic!("{}: {e}", hw_toml_path.display())),
    )
    .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", hw_toml_path.display()));
    let arch = hw_toml["hardware"]["arch"]
        .as_str()
        .expect("hardware.arch must be a string in HARDWARE.toml")
        .to_string();
    // Vendor steers per-vendor flag-key parsing in parse_kernel_toml
    // (e.g. extra_metal_flags vs extra_nvcc_flags).
    let target_vendor = hw_toml["hardware"]["vendor"]
        .as_str()
        .unwrap_or("nvidia")
        .to_string();
    println!("cargo:rerun-if-changed={}", hw_toml_path.display());

    // Expand model wildcard (exclude the `common/` shared-kernel dir,
    // which has no MODEL.toml).
    let models: Vec<String> = if model_spec == "*" {
        list_subdirs(&hw_dir)
            .into_iter()
            .filter(|d| hw_dir.join(d).join("MODEL.toml").exists())
            .collect()
    } else {
        vec![model_spec]
    };

    let mut targets = Vec::new();
    for model in &models {
        let model_dir = hw_dir.join(model);
        if !model_dir.is_dir() {
            panic!("Model kernel directory not found: {}", model_dir.display());
        }

        // Expand quant wildcard
        let quants: Vec<String> = if quant_spec == "*" {
            list_subdirs(&model_dir)
        } else {
            vec![quant_spec.clone()]
        };

        for quant in &quants {
            let model_kernel_dir = model_dir.join(quant);
            // Shared kernels live in `kernels/<hw>/common/` and apply to
            // every (model, quant) target on this hardware. Most kernels
            // here are dtype-agnostic (BF16 norms/embeds/attn) — the dir
            // is named `common` rather than after a single quant because
            // its contents span BF16, FP8, NVFP4, W4A16, W8A16, and
            // turbo3/4/8 KV-cache flavours. Per-model specialisations
            // still live under `kernels/<hw>/<model>/<quant>/`.
            let common_kernel_dir = hw_dir.join("common");

            // At least one of common or model-specific dir must exist
            let has_model_dir = model_kernel_dir.is_dir();
            let has_common_dir = common_kernel_dir.is_dir();
            assert!(
                has_model_dir || has_common_dir,
                "No kernel directory found for ({model}, {quant}). \
                 Expected {} or {}.",
                model_kernel_dir.display(),
                common_kernel_dir.display(),
            );

            // KERNEL.toml: MERGE common + model-specific. The old
            // prefer-model-else-common selection meant a model toml fully
            // SHADOWED common — silently dropping common's build flags
            // (gb10 model targets lost -DTQ_PLUS_SIGNS; the metal per-quant
            // toml lost -ffast-math) and common's [modules] overrides (the
            // chunk>=2 prefill misdispatch class, previously worked around
            // by propagating mappings into all 13 model tomls). Semantics
            // now: common parses first as the base; the model toml appends
            // flags (deduped, model last) and wins per-key on [modules].
            let mut extra_flags: Vec<String> = Vec::new();
            let mut module_overrides: HashMap<String, String> = HashMap::new();
            if has_common_dir && common_kernel_dir.join("KERNEL.toml").exists() {
                let (f, m) = parse_kernel_toml(&common_kernel_dir, &target_vendor);
                extra_flags.extend(f);
                module_overrides.extend(m);
            }
            if has_model_dir && model_kernel_dir.join("KERNEL.toml").exists() {
                let (f, m) = parse_kernel_toml(&model_kernel_dir, &target_vendor);
                for flag in f {
                    if !extra_flags.contains(&flag) {
                        extra_flags.push(flag);
                    }
                }
                module_overrides.extend(m);
            }

            // Parse sampling presets, behavior, and model_types from MODEL.toml
            let (s_tt, s_tc, s_nt, s_tools) = parse_sampling_presets(&model_dir);
            let pb = parse_behavior(&model_dir);
            let model_type_matches = parse_model_types(&model_dir);
            let dflash = parse_dflash(&model_dir);

            targets.push(Target {
                hw: hw.clone(),
                model: model.clone(),
                quant: quant.clone(),
                arch: arch.clone(),
                model_kernel_dir,
                common_kernel_dir: if has_common_dir {
                    Some(common_kernel_dir)
                } else {
                    None
                },
                extra_flags,
                module_overrides,
                sampling_thinking_text: s_tt,
                sampling_thinking_coding: s_tc,
                sampling_non_thinking: s_nt,
                sampling_tools: s_tools,
                behavior_thinking_in_tools: pb.thinking_in_tools,
                behavior_max_thinking_budget: pb.max_thinking_budget,
                behavior_thinking_default: pb.thinking_default,
                behavior_fp8_kv_calibration_tokens: pb.fp8_kv_calibration_tokens,
                behavior_default_kv_dtype: pb.default_kv_dtype,
                behavior_default_num_drafts: pb.default_num_drafts,
                behavior_disable_tool_steering: pb.disable_tool_steering,
                behavior_tool_call_parser: pb.tool_call_parser,
                behavior_enable_loop_watchdog: pb.enable_loop_watchdog,
                behavior_min_p_floor: pb.min_p_floor,
                behavior_temperature_max: pb.temperature_max,
                behavior_think_loop_min_repeats: pb.think_loop_min_repeats,
                behavior_think_loop_scan_window: pb.think_loop_scan_window,
                behavior_confidence_early_stop: pb.confidence_early_stop,
                behavior_confidence_run_length: pb.confidence_run_length,
                behavior_fuzzy_repeat_tolerance_div: pb.fuzzy_repeat_tolerance_div,
                behavior_max_inter_tool_prose: pb.max_inter_tool_prose,
                behavior_max_post_think_content_tokens: pb.max_post_think_content_tokens,
                behavior_tscg: pb.tscg,
                behavior_disable_tool_grammar: pb.disable_tool_grammar,
                behavior_rollback_resteer: pb.rollback_resteer,
                behavior_rom_head: pb.rom_head,
                behavior_tool_retry: pb.tool_retry,
                model_type_matches,
                dflash,
            });
        }
    }

    // Sort by (model, quant) for deterministic ordering
    targets.sort_by(|a, b| (&a.model, &a.quant).cmp(&(&b.model, &b.quant)));
    targets
}

/// List subdirectory names (not files) in a directory, sorted.
fn list_subdirs(dir: &std::path::Path) -> Vec<String> {
    let mut dirs: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            if entry.file_type().ok()?.is_dir() {
                Some(entry.file_name().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    dirs.sort();
    dirs
}

#[path = "build_parse.rs"]
mod build_parse;
use build_parse::{
    parse_behavior, parse_dflash, parse_kernel_toml, parse_model_types, parse_sampling_presets,
};

/// Collect kernel-source files with shadowing: common dir provides the
/// base set, model-specific dir can override individual files by matching
/// filename. `source_ext` is the per-vendor extension (e.g. "cu" for
/// NVIDIA, "metal" for Apple).
fn collect_cu_files(
    common_dir: Option<&std::path::Path>,
    model_dir: &std::path::Path,
    source_ext: &str,
) -> Vec<PathBuf> {
    let mut files: HashMap<String, PathBuf> = HashMap::new();

    // Base layer: common kernels
    if let Some(common) = common_dir {
        for f in find_cu_files(common, source_ext) {
            let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
            files.insert(stem, f);
        }
    }

    // Override layer: model-specific kernel files shadow common ones
    for f in find_cu_files(model_dir, source_ext) {
        let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
        files.insert(stem, f);
    }

    let mut result: Vec<PathBuf> = files.into_values().collect();
    result.sort();
    result
}

/// Find all kernel-source files (extension `source_ext`) in a directory.
/// Returns empty vec if dir doesn't exist.
fn find_cu_files(kernel_dir: &std::path::Path, source_ext: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(kernel_dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|e| e.to_str()) == Some(source_ext) {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}

#[path = "build_codegen.rs"]
mod build_codegen;
use build_codegen::generate_target_ptx_rs;

#[path = "build_target.rs"]
mod build_target;
use build_target::resolve_compute_target;
// Force recompilation 1775404930
