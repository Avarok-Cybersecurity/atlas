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
    // Model-declared min-p. None = MODEL.toml is silent, so the server's
    // --default-min-p stands (see SamplingCategory in src/lib.rs).
    min_p: Option<f32>,
    top_n_sigma: Option<f32>,
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
            min_p: None,
            top_n_sigma: None,
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
    /// This target's own model directory. It can differ from the owner of the
    /// kernel sources when MODEL.toml declares `kernel_source`.
    model_config_dir: PathBuf,
    /// Per-model quant dir (for KERNEL.toml and optional override .cu files).
    model_kernel_dir: PathBuf,
    /// Common quant dir (hw_dir/quant/) with shared .cu files.
    common_kernel_dir: Option<PathBuf>,
    extra_flags: Vec<String>,
    module_overrides: HashMap<String, String>,
    /// `(module, kernel)` pairs declared in `[shadow_exempt]` — kernels this
    /// target may drop from `common/` WITHOUT it being drift, each with a
    /// stated reason in its KERNEL.toml. Merged common-then-model, same as
    /// `extra_flags`. Filters the build WARNING only: the pairs stay in
    /// `shadowed_dropped`, so the startup audit still hard-errors if the
    /// model's dispatch actually resolves one.
    shadow_exempt: Vec<(String, String)>,
    /// `(module, kernel)` pairs declared in MODEL.toml `[expected_absent]` —
    /// lookups this model's dispatch may issue and fail to resolve WITHOUT that
    /// being an error, each with a mandatory stated reason. Carried onto
    /// `TargetPtxSet::expected_absent` and read by the boot audit, which fails
    /// closed on every unresolved lookup that is NOT declared here.
    expected_absent: Vec<(String, String)>,
    sampling_thinking_text: SamplingCat,
    sampling_thinking_coding: SamplingCat,
    sampling_non_thinking: SamplingCat,
    sampling_tools: SamplingCat,
    behavior_thinking_in_tools: bool,
    behavior_max_thinking_budget: u32,
    behavior_effort_capped_at_ceiling: bool,
    behavior_thinking_default: bool,
    behavior_fp8_kv_calibration_tokens: usize,
    behavior_default_kv_dtype: String,
    behavior_default_num_drafts: u32,
    behavior_disable_tool_steering: bool,
    behavior_disable_cwd_hint_injection: bool,
    behavior_use_sampling_presets_for_core: bool,
    behavior_tool_call_parser: String,
    behavior_enable_loop_watchdog: bool,
    behavior_enable_think_loop_watchdog: bool,
    behavior_honor_eos_inside_thinking: bool,
    behavior_min_reasoning_floor_tokens: u32,
    behavior_cap_thinking_at_max_tokens: bool,
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
    behavior_preserve_thinking: Option<bool>,
    /// Which `(model_type, hidden_size)` pairs this kernel target supports.
    /// Parsed from `[[model_types]]` in MODEL.toml.
    model_type_matches: Vec<ModelTypeMatch>,
    /// `[model] match_names` — checkpoint-reference needles that break the
    /// tie when several targets declare the same `(model_type, hidden_size)`
    /// (Qwen3.6-27B vs Qwen3.8-27B have bit-identical configs).
    /// `validate_collision_match_names` panics if a colliding target omits
    /// them; non-colliding targets leave this empty.
    match_names: Vec<String>,
    /// `[dflash]` section if present in MODEL.toml — drafter pairing for
    /// block-diffusion speculative decoding. `None` when the model has no
    /// associated DFlash drafter checkpoint.
    dflash: Option<DflashRaw>,
    /// `[[serve_presets]]` entries from MODEL.toml — named, validated serve
    /// configurations (checkpoint + flag/env defaults) this target declares.
    /// `validate_serve_presets` panics on a name reused across targets or
    /// equal to a target directory name; targets that declare none leave this
    /// empty.
    serve_presets: Vec<ServePresetRaw>,
}

#[derive(Default, Clone)]
struct DflashRaw {
    draft_model: String,
    gamma: usize,
    window_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
}

/// One `[[serve_presets]]` entry as parsed; mirrors `src/serve_preset.rs`.
#[derive(Default, Clone)]
struct ServePresetRaw {
    name: String,
    hf_id: String,
    hf_revision: String,
    description: String,
    /// `[serve_presets.flags]` as `(recipe key, value text)`, in file order.
    flags: Vec<(String, String)>,
    /// `[serve_presets.env]` as `(variable, value text)`, in file order.
    env: Vec<(String, String)>,
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
            pub fn ptx_modules() -> Vec<(&'static str, &'static [u8])> { Vec::new() }\n\
            pub fn metallib_modules() -> Vec<(&'static str, &'static [u8])> { Vec::new() }\n\
            pub fn all_ptx_sets() -> Vec<TargetPtxSet> { Vec::new() }\n";
        std::fs::write(out_dir.join("target_ptx.rs"), stub).expect("write skip stub target_ptx.rs");
        println!(
            "cargo:rustc-env=ATLAS_KERNEL_SET_HASH={}",
            content_hash(stub)
        );
        println!("cargo:rustc-env=ATLAS_PTX_DIR={}", out_dir.display());
        // No compiler ran, so this binary can attest to nothing. Emitted
        // explicitly rather than left unset so a reader of lib.rs does not have
        // to infer it: a record written from a skip build excuses no future
        // diff, exactly as a record written before attestations existed.
        println!("cargo:rustc-env=ATLAS_TARGET_CLOSURES={{}}");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    // ── Resolve targets (supports wildcards) ──
    let targets = resolve_targets(workspace_root);

    assert!(
        !targets.is_empty(),
        "{}",
        build_diagnose::no_targets(
            &workspace_root.join("kernels"),
            &env::var("ATLAS_TARGET_HW").unwrap_or_else(|_| build_diagnose::DEFAULT_HW.into()),
            &env::var("ATLAS_TARGET_MODEL").unwrap_or_else(|_| "*".into()),
            &env::var("ATLAS_TARGET_QUANT").unwrap_or_else(|_| "nvfp4".into()),
        )
    );

    // ── Resolve compute target (compiler) from HARDWARE.toml vendor ──
    // This abstraction supports NVIDIA (nvcc→PTX), AMD (hipcc→HSACO),
    // Apple (xcrun→metallib), Intel (icpx→SPIR-V). Only NVIDIA is implemented.
    let hw_dir = workspace_root
        .join("kernels")
        .join(env::var("ATLAS_TARGET_HW").unwrap_or_else(|_| build_diagnose::DEFAULT_HW.into()));
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
    // NOTE: a `force_br32_prefill` HARDWARE.toml key used to be re-emitted here
    // as `cargo:rustc-env=ATLAS_HW_FORCE_BR32`, for an `option_env!` reader in
    // spark-model's prefill dispatch. That reader was dropped by the squash
    // merge that landed Strix support, leaving an emit nothing consumed — and
    // the contract could not have worked anyway: `cargo:rustc-env` only reaches
    // the crate whose build script emits it, so it can never reach spark-model.
    // BR=32 on RDNA3.5 is enforced by two live mechanisms instead, and the key
    // was redundant even when it had a reader:
    //   * kernel side  — `#if defined(__SCALE__) #define BR64 32` in
    //     kernels/gb10/common/prefill_paged_compute.cuh (and an unconditional
    //     `#define BR64 32` in the strix-hip shadow), which is what keeps the
    //     _64 kernels inside the 64 KB LDS cap.
    //   * host side    — `cfg!(atlas_scale)` picks the matching 32-row grid
    //     stride in ops/prefill_attn_main_a.rs (non-paged) and
    //     ops/prefill_attn_main_b.rs (paged); the cfg comes from spark-model's
    //     own build.rs reading ATLAS_TARGET_HW.
    // Re-adding a HARDWARE.toml key for this would re-create the dead lever.
    let compute_target = resolve_compute_target(vendor_str);
    let output_ext = compute_target.output_extension();
    let uses_cuda_api = compute_target.uses_cuda_module_api();

    // Per-target: (target_idx, vec of (stem, module_name))
    let mut all_target_modules: Vec<Vec<(String, String)>> = Vec::new();
    // Per-target: (module, kernel) pairs this model's files dropped by shadowing
    // their `common/` namesakes. Baked into the binary so the startup audit can
    // separate "dropped by a fork" (fatal) from "never built for this target"
    // (expected). See `shadowed_dropped_pairs`.
    let mut all_target_drops: Vec<Vec<(String, String)>> = Vec::new();

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

    // ── Native ROCm/HIP path (vendor="hip") staging ──
    // Pairs with HipTarget in build_target.rs + the libcuda→HIP shim. Only the
    // hip vendor path runs any of this; NVIDIA/SCALE/Apple are untouched
    // (`is_hip` is false → every block below is a no-op).
    //   (a) Stage the CUDA→HIP compat-header dir (atlas-kernels/hip/compat) and
    //       export ATLAS_HIP_COMPAT_INCLUDE — HipTarget::compile force-includes
    //       hip/hip_runtime.h and `-I`s this dir so the unmodified `.cu` find
    //       cuda_runtime.h / cuda_bf16.h / cuda_fp8.h forwarded to hip/*.
    //   (b) Per-source mask-widen: gfx1151 wavefronts are 64-wide, so HIP's
    //       `__shfl_*_sync`/`__ballot_sync` masks and `__activemask()` must be
    //       64-bit. We mirror each `.cu`/`.cuh` into OUT_DIR with the widen
    //       transform applied (kernels/ is never mutated in place) and compile
    //       the mirror. NVIDIA/SCALE compile the originals directly.
    let is_hip = vendor_str == Some("hip");
    let hip_mirror_dir = out_dir.join("hip_mirror");
    if is_hip {
        let compat_dir = manifest_dir.join("hip").join("compat");
        assert!(
            compat_dir.join("cuda_runtime.h").exists(),
            "HIP compat headers missing at {} — expected the staged \
             atlas-kernels/hip/compat dir (cuda_runtime.h, cuda_bf16.h, cuda_fp8.h).",
            compat_dir.display()
        );
        println!(
            "cargo:rustc-env=ATLAS_HIP_COMPAT_INCLUDE={}",
            compat_dir.display()
        );
        // HipTarget::compile reads this via std::env::var in THIS build-script
        // process (the `cargo:rustc-env` directive only reaches the compiled
        // crate, not our own child hipcc invocations). Set before the parallel
        // compile scope is spawned, so this is single-threaded → sound.
        unsafe {
            std::env::set_var("ATLAS_HIP_COMPAT_INCLUDE", &compat_dir);
        }
        println!("cargo:rerun-if-changed={}", compat_dir.display());
        std::fs::create_dir_all(&hip_mirror_dir)
            .unwrap_or_else(|e| panic!("create hip_mirror dir: {e}"));
    }

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

            // HIP: compile a mask-widened MIRROR of the source (kernels/ is
            // never mutated). The whole source directory is mirrored once so
            // sibling `"foo.cuh"` includes still resolve next to the mirrored
            // `.cu`. NVIDIA/SCALE keep `compile_source = cu_file` (the original).
            let compile_source = if is_hip {
                hip_mirror_source(cu_file, &hip_mirror_dir, source_ext)
            } else {
                cu_file.clone()
            };

            // Dedup key: same (source, arch, sorted-flags) → identical
            // binary output. Sort flags so flag-order doesn't bust the cache.
            let mut sorted_flags = target.extra_flags.clone();
            sorted_flags.sort();
            let key = (compile_source.clone(), target.arch.clone(), sorted_flags);

            if let Some(existing) = compile_cache.get(&key) {
                copy_jobs.push((existing.clone(), out_file));
            } else {
                compile_jobs.push(CompileJob {
                    cu_file: compile_source.clone(),
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

        let drops = shadowed_dropped_pairs(
            target.common_kernel_dir.as_deref(),
            &target.model_kernel_dir,
            source_ext,
            &target.module_overrides,
        );
        // Warning-visible subset: everything except the pairs `common/`
        // declares (with a reason) as superseded in `[shadow_exempt]`. The
        // FULL `drops` list still goes into `TargetPtxSet::shadowed_dropped`
        // below, so the startup gate (`kernel_audit::classify_failures`)
        // remains fail-closed for exempt pairs too — an exemption silences
        // build-log noise, never a runtime miss.
        let reportable: Vec<&(String, String)> = drops
            .iter()
            .filter(|(m, f)| {
                !target
                    .shadow_exempt
                    .iter()
                    .any(|(em, ef)| em == m && ef == f)
            })
            .collect();
        if !reportable.is_empty() {
            println!(
                "cargo:warning=atlas-kernels: ({}, {}) drops {} kernel(s) by shadowing common/: {}. \
                 A dropped kernel fails CLOSED (try_kernel -> handle 0). If the model genuinely \
                 cannot use it this is fine; the startup audit will only hard-error if the model's \
                 dispatch actually requests it.",
                target.model,
                target.quant,
                reportable.len(),
                reportable
                    .iter()
                    .map(|(m, f)| format!("{m}::{f}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
        all_target_drops.push(drops);

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

    // ── HIP: build the libcuda→HIP shim (libcuda.so) ──
    // The runtime is unchanged (cudarc links `-lcuda`); on AMD the CUDA driver
    // symbols it imports are re-exported by libcuda_hip_shim.cpp mapped onto HIP
    // (cuModuleLoadData→hipModuleLoadData, cuLaunchKernel→hipModuleLaunchKernel,
    // …). We compile it to OUT_DIR/libcuda.so and emit a search-path directive so
    // the final link/load resolves `-lcuda` to the shim. NVIDIA/SCALE skip this.
    if is_hip {
        build_hip_shim(&manifest_dir, &out_dir);
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
    let generated = generate_target_ptx_rs(
        &targets,
        &all_target_modules,
        &all_target_drops,
        output_ext,
        uses_cuda_api,
    );
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

    // ── Bake what each target was compiled from ──
    // Read back by the benchmark gate so a record can attest to the BINARY's
    // sources rather than to whatever the tree held when the record was
    // written — the two differ whenever a build is stale, a tree is dirty, or
    // an image is carried between boxes, all of which happen during a gate
    // campaign.
    println!(
        "cargo:rustc-env=ATLAS_TARGET_CLOSURES={}",
        closure_attestation(workspace_root, &targets, compute_target.as_ref())
    );
}

/// Per-target closure hashes, as one line of JSON.
///
/// Returns `{}` — attesting to nothing — whenever any part cannot be computed:
/// an unknown compiler, sources that will not resolve, an unresolvable
/// `#include`. Every such case costs a future re-run, which is the direction a
/// provenance hash must fail in; a placeholder would let two different builds
/// compare equal.
fn closure_attestation(
    workspace_root: &std::path::Path,
    targets: &[Target],
    compute_target: &dyn build_target::ComputeTarget,
) -> String {
    let Some(compiler) = compute_target.compiler_id() else {
        return "{}".into();
    };
    let mut map = serde_json::Map::new();
    for target in targets {
        let sources = collect_cu_files(
            target.common_kernel_dir.as_deref(),
            &target.model_kernel_dir,
            compute_target.source_extension(),
        );
        if sources.is_empty() {
            continue;
        }
        let configs = [
            workspace_root
                .join("kernels")
                .join(&target.hw)
                .join("HARDWARE.toml"),
            target
                .common_kernel_dir
                .as_ref()
                .map(|d| d.join("KERNEL.toml"))
                .unwrap_or_default(),
            target.model_config_dir.join("MODEL.toml"),
            target.model_kernel_dir.join("KERNEL.toml"),
        ]
        .into_iter()
        .filter(|p| p.is_file())
        .collect();

        let inputs = atlas_closure::ClosureInputs {
            sources,
            configs,
            flags: target.extra_flags.clone(),
            arch: target.arch.clone(),
            compiler: compiler.clone(),
        };
        let Ok(closure) = atlas_closure::hash_with_report(workspace_root, &inputs) else {
            continue;
        };
        // Surfaced, not swallowed. A quoted include naming no file is hashed by
        // name rather than content, so a NEW one silently widens what the gate
        // cannot see. There are two in the tree today, both dead `#if` arms of
        // the 27B's vendored q4k code; a third deserves a look.
        for entry in &closure.unresolved {
            println!("cargo:warning=closure: unresolved include ({entry})");
        }
        map.insert(
            format!("{}/{}/{}", target.hw, target.model, target.quant),
            serde_json::json!({
                "hash": closure.digest,
                "arch": target.arch,
                "compiler": compiler,
                "flags": target.extra_flags,
            }),
        );
    }
    serde_json::Value::Object(map).to_string()
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

// ──────────────────────────── HIP (vendor="hip") helpers ────────────────────────────
// These run ONLY on the native ROCm/HIP path. NVIDIA/SCALE/Apple never call them.

/// Mirror `src` (a `.cu`/`.cuh`) into `mirror_root` with the 64-wide-wavefront
/// mask-widen transform applied, and return the mirrored path to compile.
///
/// gfx1151 wavefronts are 64-wide, so HIP's `__shfl_*_sync` / `__ballot_sync`
/// masks and the result of `__activemask()` must be 64-bit (NVIDIA's are 32).
/// The entire source DIRECTORY is mirrored on first touch so that sibling
/// `#include "foo.cuh"` resolves next to the mirrored `.cu` (clang resolves
/// quoted includes relative to the including file). `kernels/` is never mutated.
fn hip_mirror_source(
    src: &std::path::Path,
    mirror_root: &std::path::Path,
    source_ext: &str,
) -> PathBuf {
    let src_dir = src.parent().expect("kernel source has no parent dir");
    // One mirror subdir per source dir, keyed by a stable hash of its abs path
    // (avoids collisions between e.g. common/ and qwen3.6-27b/nvfp4/).
    let dir_key = content_hash(&src_dir.to_string_lossy());
    let mirror_dir = mirror_root.join(dir_key);
    std::fs::create_dir_all(&mirror_dir)
        .unwrap_or_else(|e| panic!("create hip mirror subdir: {e}"));

    // Transform every `.cu`/`.cuh` in the source dir into the mirror (once per
    // file per build; cheap, and keeps headers in lockstep with their `.cu`).
    if let Ok(entries) = std::fs::read_dir(src_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            let is_kernel_src = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == source_ext || e == "cuh" || e == "h")
                .unwrap_or(false);
            if !p.is_file() || !is_kernel_src {
                continue;
            }
            let name = p.file_name().unwrap();
            let dst = mirror_dir.join(name);
            let text = std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("read {} for HIP mirror: {e}", p.display()));
            let widened = widen_warp_masks(&text);
            // Only rewrite when content changed (keeps mtimes stable across
            // incremental builds → no needless hipcc recompiles).
            let needs_write = match std::fs::read_to_string(&dst) {
                Ok(existing) => existing != widened,
                Err(_) => true,
            };
            if needs_write {
                std::fs::write(&dst, &widened)
                    .unwrap_or_else(|e| panic!("write HIP mirror {}: {e}", dst.display()));
            }
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }

    mirror_dir.join(src.file_name().unwrap())
}

/// Append `ULL` to the warp-mask literal that is the first argument of every
/// `__shfl*_sync(` / `__ballot_sync(`, and widen `unsigned int <v> = __activemask();`
/// to `unsigned long long`. Bit-exact reproduction of the transform that produced
/// the port's strix-hip-real tree (e.g. `__shfl_down_sync(0xFFFFFFFF, ...)` →
/// `__shfl_down_sync(0xFFFFFFFFULL, ...)`, `0xf` → `0xfULL`). Anything that isn't
/// a sync-call mask literal (e.g. byte-extraction `& 0xFFFFFFFF`) is left alone.
fn widen_warp_masks(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 256);
    // Sync-call prefixes whose FIRST argument is a warp mask.
    const SYNC_CALLS: &[&str] = &[
        "__shfl_sync(",
        "__shfl_up_sync(",
        "__shfl_down_sync(",
        "__shfl_xor_sync(",
        "__ballot_sync(",
    ];
    let mut i = 0usize;
    while i < bytes.len() {
        // Match a sync call at position i.
        let mut matched = None;
        for call in SYNC_CALLS {
            if src[i..].starts_with(call) {
                matched = Some(call.len());
                break;
            }
        }
        if let Some(call_len) = matched {
            out.push_str(&src[i..i + call_len]);
            i += call_len;
            // Skip whitespace before the mask literal.
            while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                out.push(bytes[i] as char);
                i += 1;
            }
            // If a hex literal follows, emit it and append ULL (unless already
            // suffixed with L/UL/LL/ULL). Decimal/identifier masks left untouched.
            if i + 1 < bytes.len()
                && bytes[i] == b'0'
                && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X')
            {
                let start = i;
                i += 2;
                while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                    i += 1;
                }
                let lit = &src[start..i];
                // Consume any existing integer suffix (u/U/l/L) so we don't double it.
                let mut suffix_end = i;
                while suffix_end < bytes.len()
                    && matches!(bytes[suffix_end], b'u' | b'U' | b'l' | b'L')
                {
                    suffix_end += 1;
                }
                let existing_suffix = &src[i..suffix_end];
                out.push_str(lit);
                // Normalize to a 64-bit literal. If it already ends in LL/ULL keep it.
                if existing_suffix.to_ascii_uppercase().contains("LL") {
                    out.push_str(existing_suffix);
                } else {
                    out.push_str("ULL");
                }
                i = suffix_end;
            }
            continue;
        }
        // `__activemask()` is captured into a variable typed `unsigned int`; widen
        // that declaration to 64-bit. Match `unsigned int <id> = __activemask()`.
        if src[i..].starts_with("unsigned int ") {
            // Look ahead on the same statement for `= __activemask()`.
            let stmt_end = src[i..].find(';').map(|o| i + o).unwrap_or(bytes.len());
            if src[i..stmt_end].contains("__activemask()") {
                out.push_str("unsigned long long ");
                i += "unsigned int ".len();
                continue;
            }
        }
        // Default: copy one byte (UTF-8 safe: kernels are ASCII, but guard anyway).
        let ch_len = src[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
        out.push_str(&src[i..i + ch_len]);
        i += ch_len;
    }
    out
}

/// Compile the libcuda→HIP shim to `out_dir/libcuda.so` and emit the link
/// search-path directive. The runtime keeps importing the CUDA driver API;
/// this `.so` re-exports those symbols mapped onto HIP.
fn build_hip_shim(manifest_dir: &std::path::Path, out_dir: &std::path::Path) {
    // Windows: build the real cu*/cudart HIP shim as cuda.dll + import libs and
    // stage the runtime DLLs for packaging (cudarc dlopens the driver at run
    // time). See build_hip_shim_windows.
    if cfg!(windows) {
        build_hip_shim_windows(manifest_dir, out_dir);
        return;
    }
    let hipcc = std::env::var("ATLAS_HIPCC").unwrap_or_else(|_| "/opt/rocm/bin/hipcc".into());
    // Three HIP shims so the HIP target resolves every CUDA lib spark emits:
    //   libcuda.so     — cu* driver API, real (libcuda_hip_shim.cpp)
    //   libcudart.so   — cudart runtime API, real (libcudart_hip_shim.cpp)
    //   libcublasLt.so — cuBLASLt, stub (opt-in ATLAS_CUBLAS_GEMM path only)
    // Before this, only libcuda.so existed, so native-HIP never linked (`-lcudart`
    // / `-lcublasLt` had no provider). hipcc links libamdhip64 into each shim, so
    // the AMD runtime is pulled via the shim's DT_NEEDED at serve time (needs
    // /opt/rocm/lib on LD_LIBRARY_PATH alongside this OUT_DIR).
    for (src_name, so_name) in [
        ("libcuda_hip_shim.cpp", "libcuda.so"),
        ("libcudart_hip_shim.cpp", "libcudart.so"),
        ("libcublaslt_stub.cpp", "libcublasLt.so"),
    ] {
        let src = manifest_dir.join("hip").join(src_name);
        assert!(src.exists(), "HIP shim source missing at {}", src.display());
        println!("cargo:rerun-if-changed={}", src.display());
        let out = out_dir.join(so_name);
        let status = std::process::Command::new(&hipcc)
            .args([
                "-shared",
                "-fPIC",
                src.to_str().unwrap(),
                "-o",
                out.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("failed to run hipcc for {so_name} ({hipcc}): {e}"));
        assert!(
            status.success(),
            "hipcc failed building {so_name} from {}",
            src.display()
        );
    }
    // OUT_DIR first on the link search path so `-lcuda`/`-lcudart`/`-lcublasLt`
    // resolve to the shims.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!(
        "cargo:warning=atlas-kernels: built HIP shims (libcuda/libcudart/libcublasLt) in {}",
        out_dir.display()
    );
}

/// Windows native-HIP runtime shim. Builds ONE `cuda.dll` from the real cu*
/// (libcuda_hip_shim.cpp) and cudart (libcudart_hip_shim.cpp) HIP mappings plus
/// the cuBLASLt stub, linked against `amdhip64.lib`, and generates its import
/// library. cudarc dlopens `["cuda","nvcuda"]` at runtime, so the DLL is copied
/// to `nvcuda.dll`; spark's own FFI links `-lcuda`/`-lcudart`/`-lcublasLt`, so
/// the import lib is copied to all three names (each carries every export, the
/// linker binds each symbol once). The runtime DLLs (`nvcuda.dll` +
/// `amdhip64.dll`) are staged into OUT_DIR for the packaging step to bundle
/// beside `spark.exe`. Mirrors the Linux `.so` shims — same mappings, proven to
/// link on gfx1151. (Hosted CI has no AMD GPU, so CI proves compile+link+package,
/// not execution.)
fn build_hip_shim_windows(manifest_dir: &std::path::Path, out_dir: &std::path::Path) {
    let hipcc = std::env::var("ATLAS_HIPCC")
        .expect("ATLAS_HIPCC must point at the Windows HIP SDK hipcc for the hip target");
    let hip_path =
        std::env::var("HIP_PATH").expect("HIP_PATH must be set for the windows hip build");
    let hip_root = std::path::Path::new(&hip_path);

    // 1. Compile the three shims to objects (host C++ over HIP; no -fPIC on MSVC).
    let sources = [
        "libcuda_hip_shim.cpp",
        "libcudart_hip_shim.cpp",
        "libcublaslt_stub.cpp",
    ];
    let mut objs = Vec::new();
    for name in sources {
        let src = manifest_dir.join("hip").join(name);
        assert!(src.exists(), "HIP shim source missing at {}", src.display());
        println!("cargo:rerun-if-changed={}", src.display());
        let obj = out_dir.join(format!("{name}.obj"));
        let status = std::process::Command::new(&hipcc)
            .args(["-c", "-O2"])
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("hipcc -c failed for {name} ({e})"));
        assert!(status.success(), "hipcc failed compiling {name}");
        objs.push(obj);
    }

    // 2. Export list: exactly the extern symbols the objects define, read back
    // with MSVC `dumpbin /SYMBOLS` (on PATH via msvc-dev-cmd, like cl/lib —
    // the HIP SDK does not ship llvm-nm at a stable path). Reading the objects
    // means the .def can never drift from the sources. Defined externals are
    // `SECTn ... External | <name>`; undefined imports (the hip* the shim calls)
    // are `UNDEF` and start with `hip`, so filtering on `External`, not `UNDEF`,
    // and a `cu` name prefix keeps exactly cu*/cudart/cublasLt.
    let mut exports = Vec::new();
    for obj in &objs {
        let out = std::process::Command::new("dumpbin")
            .arg("/SYMBOLS")
            .arg(obj)
            .output()
            .unwrap_or_else(|e| panic!("dumpbin /SYMBOLS failed on {} ({e})", obj.display()));
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if line.contains("External")
                && !line.contains("UNDEF")
                && let Some(name) = line.split_whitespace().last()
                && name.starts_with("cu")
            {
                exports.push(name.to_string());
            }
        }
    }
    exports.sort();
    exports.dedup();
    assert!(
        !exports.is_empty(),
        "no cu*/cudart/cublasLt exports found in shim objects"
    );
    let def = out_dir.join("atlas_hip_cuda.def");
    std::fs::write(&def, format!("EXPORTS\n{}\n", exports.join("\n"))).expect("write cuda.def");

    // 3. Link cuda.dll + its import lib against amdhip64 (hipcc -> clang -> lld-link).
    let dll = out_dir.join("cuda.dll");
    let implib = out_dir.join("cuda.lib");
    let amdhip = hip_root.join("lib").join("amdhip64.lib");
    let mut link = std::process::Command::new(&hipcc);
    link.arg("-shared");
    for obj in &objs {
        link.arg(obj);
    }
    let status = link
        .arg(&amdhip)
        .arg("-o")
        .arg(&dll)
        .arg(format!("-Wl,/DEF:{}", def.display()))
        .arg(format!("-Wl,/IMPLIB:{}", implib.display()))
        .status()
        .unwrap_or_else(|e| panic!("hipcc -shared (cuda.dll) failed ({e})"));
    assert!(status.success(), "linking cuda.dll failed");

    // 4. cudarc dlopens nvcuda.dll; spark links cuda/cudart/cublasLt.lib. One DLL,
    // one import lib carrying every export, copied to each needed name.
    std::fs::copy(&dll, out_dir.join("nvcuda.dll")).expect("copy cuda.dll -> nvcuda.dll");
    for lib in ["cudart.lib", "cublasLt.lib"] {
        std::fs::copy(&implib, out_dir.join(lib))
            .unwrap_or_else(|e| panic!("copy import lib -> {lib}: {e}"));
    }

    // 5. Stage the HIP runtime DLL for packaging. On Windows it is versioned
    // (amdhip64_6.dll, not amdhip64.dll), so glob `amdhip64*.dll` under the SDK
    // bin. If absent (a driverless CI runner may ship only the import lib), it
    // is not fatal: amdhip64 is an AMD-driver component present on any real AMD
    // Windows host, so the shipped shim resolves it there. Informational, not a
    // warning, so a green build is not noisy.
    let mut staged_runtime = false;
    if let Ok(entries) = std::fs::read_dir(hip_root.join("bin")) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("amdhip64") && name.ends_with(".dll") {
                std::fs::copy(entry.path(), out_dir.join(&*name))
                    .unwrap_or_else(|e| panic!("stage {name}: {e}"));
                staged_runtime = true;
            }
        }
    }
    if !staged_runtime {
        println!(
            "cargo:warning=atlas-kernels: no amdhip64*.dll in {}\\bin to bundle — it is an AMD-driver component, present on real AMD Windows hosts at runtime.",
            hip_root.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // Record the dir holding nvcuda.dll/amdhip64.dll so the packaging step bundles them.
    println!(
        "cargo:rustc-env=ATLAS_HIP_RUNTIME_DIR={}",
        out_dir.display()
    );
    println!(
        "cargo:warning=atlas-kernels: built Windows HIP runtime shim (cuda.dll/nvcuda.dll + import libs, {} exports) at {}",
        exports.len(),
        out_dir.display()
    );
}

/// Resolve all compilation targets from env vars, expanding wildcards.
fn resolve_targets(workspace_root: &std::path::Path) -> Vec<Target> {
    let kernels_root = workspace_root.join("kernels");

    let hw = env::var("ATLAS_TARGET_HW").unwrap_or_else(|_| build_diagnose::DEFAULT_HW.into());
    let model_spec = env::var("ATLAS_TARGET_MODEL").unwrap_or_else(|_| "*".into());
    let quant_spec = env::var("ATLAS_TARGET_QUANT").unwrap_or_else(|_| "nvfp4".into());

    let hw_dir = kernels_root.join(&hw);
    assert!(
        hw_dir.is_dir(),
        "{}",
        build_diagnose::unknown_hw(&kernels_root, &hw)
    );

    // Nothing else in the build names the hardware it chose. Say it here,
    // before any of it can fail somewhere that never mentions the env var —
    // but AFTER the target is known to exist, so a rejected target is not
    // also announced as the one being built.
    build_diagnose::warn_if_defaulted(&kernels_root, &hw, &model_spec, &quant_spec);

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
            panic!("{}", build_diagnose::unknown_model(&hw_dir, &hw, model));
        }

        // `[model] kernel_source` — this target compiles another target's
        // kernel sources instead of shipping its own copies (SSOT for
        // architecturally-identical checkpoints: qwen3.8-27b reuses
        // qwen3.6-27b's .cu files verbatim rather than vendoring 26
        // duplicates). MODEL.toml (sampling/behavior/model_types/…) still
        // comes from THIS target's own directory; only the quant kernel
        // dirs are redirected. Chains are refused — a source must own its
        // sources.
        let kernel_src_dir = match parse_kernel_source(&model_dir) {
            Some(src) => {
                let src_dir = hw_dir.join(&src);
                assert!(
                    src_dir.is_dir() && src_dir.join("MODEL.toml").exists(),
                    "{model}/MODEL.toml: kernel_source = \"{src}\" does not name a kernel \
                     target directory under {}",
                    hw_dir.display()
                );
                assert!(
                    parse_kernel_source(&src_dir).is_none(),
                    "{model}/MODEL.toml: kernel_source = \"{src}\" itself redirects — \
                     chains are not allowed; point at the target that owns the sources"
                );
                src_dir
            }
            None => model_dir.clone(),
        };

        // Expand quant wildcard (against the source dir, which owns the
        // per-quant kernel subdirectories).
        let quants: Vec<String> = if quant_spec == "*" {
            list_subdirs(&kernel_src_dir)
        } else {
            vec![quant_spec.clone()]
        };

        for quant in &quants {
            let model_kernel_dir = kernel_src_dir.join(quant);
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
            // Shadow-drop exemptions merge the same way: common/ declares the
            // repo-wide superseded kernels, a model KERNEL.toml adds the ones
            // only IT may drop (e.g. mistral's divergent RoPE convention).
            let mut shadow_exempt: Vec<(String, String)> = Vec::new();
            if has_common_dir && common_kernel_dir.join("KERNEL.toml").exists() {
                let (f, m) = parse_kernel_toml(&common_kernel_dir, &target_vendor);
                extra_flags.extend(f);
                module_overrides.extend(m);
                shadow_exempt.extend(parse_shadow_exempt(&common_kernel_dir));
            }
            if has_model_dir && model_kernel_dir.join("KERNEL.toml").exists() {
                let (f, m) = parse_kernel_toml(&model_kernel_dir, &target_vendor);
                for flag in f {
                    if !extra_flags.contains(&flag) {
                        extra_flags.push(flag);
                    }
                }
                module_overrides.extend(m);
                shadow_exempt.extend(parse_shadow_exempt(&model_kernel_dir));
            }
            shadow_exempt.sort();
            shadow_exempt.dedup();

            // Parse sampling presets, behavior, and model_types from MODEL.toml.
            // MODEL.toml is a build INPUT (needles, sampling, behavior are
            // codegen'd into the binary) but lives one level above the
            // rerun-tracked quant dir — without this line an edit to it (e.g.
            // a match_names change that MOVES routing) does not rebuild, and
            // the binary keeps serving stale routing until an unrelated build.
            println!(
                "cargo:rerun-if-changed={}",
                model_dir.join("MODEL.toml").display()
            );
            let (s_tt, s_tc, s_nt, s_tools) = parse_sampling_presets(&model_dir);
            let pb = parse_behavior(&model_dir);
            let model_type_matches = parse_model_types(&model_dir);
            let match_names = parse_match_names(&model_dir);
            let dflash = parse_dflash(&model_dir);
            let expected_absent = parse_expected_absent(&model_dir);
            let serve_presets = parse_serve_presets(&model_dir);

            targets.push(Target {
                hw: hw.clone(),
                model: model.clone(),
                quant: quant.clone(),
                arch: arch.clone(),
                model_config_dir: model_dir.clone(),
                model_kernel_dir,
                common_kernel_dir: if has_common_dir {
                    Some(common_kernel_dir)
                } else {
                    None
                },
                extra_flags,
                module_overrides,
                shadow_exempt,
                expected_absent,
                sampling_thinking_text: s_tt,
                sampling_thinking_coding: s_tc,
                sampling_non_thinking: s_nt,
                sampling_tools: s_tools,
                behavior_thinking_in_tools: pb.thinking_in_tools,
                behavior_max_thinking_budget: pb.max_thinking_budget,
                behavior_effort_capped_at_ceiling: pb.effort_capped_at_ceiling,
                behavior_thinking_default: pb.thinking_default,
                behavior_fp8_kv_calibration_tokens: pb.fp8_kv_calibration_tokens,
                behavior_default_kv_dtype: pb.default_kv_dtype,
                behavior_default_num_drafts: pb.default_num_drafts,
                behavior_disable_tool_steering: pb.disable_tool_steering,
                behavior_disable_cwd_hint_injection: pb.disable_cwd_hint_injection,
                behavior_use_sampling_presets_for_core: pb.use_sampling_presets_for_core,
                behavior_tool_call_parser: pb.tool_call_parser,
                behavior_enable_loop_watchdog: pb.enable_loop_watchdog,
                behavior_enable_think_loop_watchdog: pb.enable_think_loop_watchdog,
                behavior_honor_eos_inside_thinking: pb.honor_eos_inside_thinking,
                behavior_min_reasoning_floor_tokens: pb.min_reasoning_floor_tokens,
                behavior_cap_thinking_at_max_tokens: pb.cap_thinking_at_max_tokens,
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
                behavior_preserve_thinking: pb.preserve_thinking,
                model_type_matches,
                match_names,
                dflash,
                serve_presets,
            });
        }
    }

    // Sort by (model, quant) for deterministic ordering
    targets.sort_by(|a, b| (&a.model, &a.quant).cmp(&(&b.model, &b.quant)));
    validate_collision_match_names(&targets);
    validate_serve_presets(&targets);
    targets
}

/// Fail the BUILD when a `[[serve_presets]]` name is not unique across every
/// compiled target, or collides with a kernel-target directory name.
///
/// `spark serve <name>` resolves a preset by name before it tries the HF
/// cache, so two targets declaring the same preset name would serve whichever
/// sorted first — the same silent build-order pick the match_names rule
/// exists to forbid. A preset named like a target directory is refused for
/// the same reason `--kernel-target` takes a directory name: one vocabulary,
/// two meanings, and the error would surface as a wrong checkpoint rather
/// than as a message.
fn validate_serve_presets(targets: &[Target]) {
    let mut seen: HashMap<String, &str> = HashMap::new();
    for t in targets {
        for p in &t.serve_presets {
            let key = p.name.to_ascii_lowercase();
            if let Some(owner) = seen.get(&key)
                && *owner != t.model
            {
                panic!(
                    "serve preset {:?} is declared by both kernel targets {:?} and {:?} — \
                     preset names must be unique across kernels/<hw>/*/MODEL.toml",
                    p.name, owner, t.model
                );
            }
            seen.insert(key.clone(), &t.model);
            assert!(
                !targets
                    .iter()
                    .any(|o| o.model.eq_ignore_ascii_case(&p.name)),
                "kernels/<hw>/{}/MODEL.toml: serve preset {:?} is named like a kernel target \
                 directory — pick a name that is not a target (e.g. suffix the quant/variant)",
                t.model,
                p.name
            );
        }
    }
}

/// Fail the BUILD when differently-named targets collide on a
/// `(model_type, hidden_size)` declaration without every participant
/// declaring explicit `[model] match_names`.
///
/// Runtime resolution (`atlas_kernels::resolve`) breaks such a collision by
/// matching the needles against the checkpoint reference and hard-errors
/// when it cannot — but a colliding target with NO needles declared could
/// never win that tie-break, which is a latent unserveable configuration
/// nobody would notice until a serve fails. Config shape cannot distinguish
/// e.g. Qwen3.6-27B from Qwen3.8-27B (bit-identical configs), so the
/// disambiguation must be declared, and declared at build time.
fn validate_collision_match_names(targets: &[Target]) {
    let mut by_pair: HashMap<(&str, Option<usize>), Vec<&Target>> = HashMap::new();
    for t in targets {
        for m in &t.model_type_matches {
            by_pair
                .entry((m.model_type.as_str(), m.hidden_size))
                .or_default()
                .push(t);
        }
    }
    for ((model_type, hidden), group) in by_pair {
        let mut names: Vec<&str> = group.iter().map(|t| t.model.as_str()).collect();
        names.sort();
        names.dedup();
        if names.len() < 2 {
            continue; // no collision (multi-quant variants of one target are fine)
        }
        for t in &group {
            assert!(
                !t.match_names.is_empty(),
                "kernel targets {names:?} all declare (model_type \"{model_type}\", \
                 hidden_size {hidden:?}) in [[model_types]], but target \"{}\" has no \
                 [model] match_names — runtime resolution could never select it \
                 unambiguously. Add e.g. `match_names = [\"{}\"]` to \
                 kernels/<hw>/{}/MODEL.toml (needles are case-insensitive substrings \
                 of the checkpoint's HF id / --model-name / model dir).",
                t.model,
                t.model,
                t.model,
            );
        }
        // Presence is not enough: a target whose every needle is DOMINATED by
        // a colliding sibling's needle (any reference containing needle X
        // also contains a substring needle Y of the sibling) can never be the
        // unique match in this tier — the tier degrades to a guaranteed
        // Ambiguous startup error for exactly the references it exists to
        // route. That shape shipped once (qwen3.5-27b's needle set was a
        // strict subset of qwen3.6-27b's on their shared qwen3_6_moe/5120
        // entry) and this rejects it at build time.
        for a in &group {
            for b in &group {
                if a.model == b.model {
                    continue;
                }
                let winnable = a.match_names.iter().any(|na| {
                    let na = na.to_lowercase();
                    !b.match_names
                        .iter()
                        .any(|nb| na.contains(&nb.to_lowercase()))
                });
                assert!(
                    winnable,
                    "kernel target \"{}\" collides with \"{}\" on (model_type \
                     \"{model_type}\", hidden_size {hidden:?}), but every one of its \
                     match_names {:?} contains one of \"{}\"'s needles {:?} — any \
                     reference matching \"{}\" also matches \"{}\", so \"{}\" can \
                     never win the tie and the tier always hard-errors Ambiguous. \
                     Give \"{}\" a needle the sibling does not shadow, or remove \
                     the colliding [[model_types]] entry.",
                    a.model,
                    b.model,
                    a.match_names,
                    b.model,
                    b.match_names,
                    a.model,
                    b.model,
                    a.model,
                    a.model,
                );
            }
        }
    }
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

// Entry-point resolution for `shadowed_dropped_pairs`. Lives in its own file so
// `tests/kernel_shadow_detector.rs` can compile the SAME code a build script
// would otherwise keep untestable — a build script's `#[cfg(test)]` modules are
// never run by `cargo test`, which is how the previous text-scan shipped a hole
// with no test that could have noticed.
#[path = "build_shadow.rs"]
mod build_shadow;
use build_parse::{
    parse_behavior, parse_dflash, parse_expected_absent, parse_kernel_source, parse_kernel_toml,
    parse_match_names, parse_model_types, parse_sampling_presets, parse_serve_presets,
    parse_shadow_exempt,
};
use build_shadow::shadowed_missing_symbols;

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

#[path = "build_diagnose.rs"]
mod build_diagnose;
// Force recompilation 1775404930

/// Every `(module, kernel)` this target's model-specific files drop relative to
/// their `common/` namesakes.
///
/// SHADOWING DRIFT. Shadowing is whole-file, not per-symbol: a model file
/// replaces its common namesake entirely. When `common/` later gains a kernel
/// the fork never picked up, that kernel silently vanishes for this model —
/// `try_kernel` returns handle 0 and whatever depends on it fails CLOSED. That
/// is exactly how the 27B lost the four multi-sequence decode kernels
/// (`gated_delta_rule_decode_f32_{norm,conv_norm,strided,strided_norm}`),
/// pinning every concurrent decode to the per-sequence fallback until
/// 2026-07-26.
///
/// This list is baked into the binary (`TargetPtxSet::shadowed_dropped`) so the
/// startup kernel audit can tell the two failure classes apart:
///
///   * a lookup that failed AND is in this list — the kernel EXISTS in
///     `common/`, this model's fork dropped it. A build defect: fatal.
///   * a lookup that failed and is NOT — the kernel was never compiled for this
///     target at all (typically another architecture's: MLA, hyper-connection,
///     CSA). Expected, and merely informational.
///
/// Without that split the audit prints ~26 benign entries for a Qwen model and
/// the one that matters hides among them.
fn shadowed_dropped_pairs(
    common_dir: Option<&std::path::Path>,
    model_dir: &std::path::Path,
    source_ext: &str,
    module_overrides: &HashMap<String, String>,
) -> Vec<(String, String)> {
    let Some(common) = common_dir else {
        return Vec::new();
    };
    let common_by_stem: HashMap<String, PathBuf> = find_cu_files(common, source_ext)
        .into_iter()
        .map(|f| (f.file_stem().unwrap().to_str().unwrap().to_string(), f))
        .collect();
    let mut out = Vec::new();
    for f in find_cu_files(model_dir, source_ext) {
        let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
        let Some(common_f) = common_by_stem.get(&stem) else {
            continue;
        };
        let module = module_overrides.get(&stem).cloned().unwrap_or(stem);
        for func in shadowed_missing_symbols(common_f, &f) {
            out.push((module.clone(), func));
        }
    }
    out.sort();
    out
}
