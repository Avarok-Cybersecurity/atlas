// SPDX-License-Identifier: AGPL-3.0-only
//
// Compute-target abstraction for build.rs. Included via
// `#[path = "build_target.rs"] mod build_target;` so the public
// surface (`ComputeTarget` trait + `resolve_compute_target` factory)
// is reachable through `super::build_target::*`.

use std::path::PathBuf;
use std::process::Command;

use super::build_codegen::find_cuda_dir;

// ── Compute target abstraction ─────────────────────────────────────────

/// Build-time kernel compilation target. Abstracts away the specific
/// compiler and output format so the same build.rs works for NVIDIA
/// (nvcc → PTX text), Apple (xcrun → metallib binary), AMD (hipcc →
/// HSACO binary), or Intel (icpx → SPIR-V binary).
pub(super) trait ComputeTarget: Send + Sync {
    fn source_extension(&self) -> &str;
    fn output_extension(&self) -> &str;
    /// Whether this backend exposes the CUDA module API — i.e. the
    /// runtime loads kernels via `cuModuleLoadData` and the codegen must
    /// emit the `all_ptx_sets()` registry. True for NVIDIA and for SCALE
    /// (SCALE is CUDA-compatible; it just emits AMD-GPU binary objects
    /// instead of PTX text). False for Metal, which has its own module
    /// API and registry path.
    fn uses_cuda_module_api(&self) -> bool;
    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String>;
}

/// Warning numbers the strict kernel-compile validation ("clippy for kernels")
/// intentionally tolerates. Documented like `_typos.toml`: every entry needs a
/// rationale comment. Empty is the goal — every kernel warning fixed at source
/// rather than suppressed.
const KERNEL_STRICT_DIAG_SUPPRESS: &[u32] = &[
    // (none — all kernel warnings are resolved at source)
];

struct NvidiaTarget {
    nvcc: PathBuf,
}

impl ComputeTarget for NvidiaTarget {
    fn source_extension(&self) -> &str {
        "cu"
    }
    fn output_extension(&self) -> &str {
        "ptx"
    }
    fn uses_cuda_module_api(&self) -> bool {
        true
    }

    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        let mut args = vec!["--ptx".into(), format!("-arch={arch}"), "-O3".into()];
        // The MSVC host compiler defaults to C++14, so nvcc on Windows rejects
        // the kernels' C++17 fold expressions and structured bindings. Force the
        // dialect there (nvcc -std=c++17 sets device + cl.exe host std). Gated to
        // Windows so the Linux/macOS nvcc builds are byte-for-byte unchanged.
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
            args.push("-std=c++17".into());
        }
        args.extend(extra_flags.iter().cloned());
        // Strict kernel-compile validation — "clippy for kernels". Promote every
        // nvcc/cudafe warning (unused variables #550-D, reorder, deprecated
        // declarations, …) to a hard error so latent kernel-quality issues fail
        // the build instead of silently accruing — the Windows/MSVC nvcc leg of
        // the release matrix first surfaced a pile of these that the Linux build
        // tolerated. On by default; set ATLAS_KERNEL_NO_STRICT=1 to disable
        // locally (only the literal "1" disables, so ATLAS_KERNEL_NO_STRICT=0
        // still means strict-on and never silently opts out). A clean compile
        // emits byte-identical PTX — `--Werror` only changes behaviour when a
        // warning actually fires — so this never alters kernel output.
        if std::env::var("ATLAS_KERNEL_NO_STRICT").as_deref() != Ok("1") {
            args.push("--Werror".into());
            args.push("all-warnings".into());
            for num in KERNEL_STRICT_DIAG_SUPPRESS {
                args.push("-Xcudafe".into());
                args.push(format!("--diag_suppress={num}"));
            }
        }
        // ATLAS_EXTRA_NVCC_FLAGS — global override for kernel bisection
        // tests. Whitespace-separated list of additional nvcc args
        // (typically `-D<MACRO>=1` to flip `#ifdef`-gated kernel paths).
        // Phase 2c day 2: used with `-DATLAS_FAST_SOFTMAX_EXP=1` to
        // re-enable the pre-Phase-2b sw_exp polynomial and bench the
        // `__expf` regression hypothesis.
        if let Ok(s) = std::env::var("ATLAS_EXTRA_NVCC_FLAGS") {
            for tok in s.split_whitespace() {
                args.push(tok.to_string());
            }
        }
        args.push(source.to_str().unwrap().into());
        args.push("-o".into());
        args.push(output.to_str().unwrap().into());

        let status = Command::new(&self.nvcc)
            .args(&args)
            .status()
            .map_err(|e| format!("Failed to run nvcc: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("nvcc --ptx failed for {}", source.display()))
        }
    }
}

/// Apple Metal compilation target: `.metal` → `.metallib` via the
/// two-step `xcrun -sdk macosx metal -c` (→ AIR) then
/// `xcrun -sdk macosx metallib` (→ metallib) pipeline.
struct AppleTarget {
    xcrun: PathBuf,
}

impl ComputeTarget for AppleTarget {
    fn source_extension(&self) -> &str {
        "metal"
    }
    fn output_extension(&self) -> &str {
        "metallib"
    }
    fn uses_cuda_module_api(&self) -> bool {
        false
    }

    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        // Two-step pipeline: source.metal → tmp.air → output.metallib.
        // The intermediate AIR file lives next to the metallib in OUT_DIR
        // so cargo's incremental cache treats it as a build artifact.
        let air_path = output.with_extension("air");

        let mut metal_args: Vec<String> = vec!["-sdk".into(), "macosx".into(), "metal".into()];
        // arch maps directly to `-std=` (e.g. metal3.1, metal3.0, metal2.4).
        if !arch.is_empty() {
            metal_args.push(format!("-std={arch}"));
        }
        metal_args.push("-c".into());
        metal_args.push("-O3".into());
        metal_args.extend(extra_flags.iter().cloned());
        metal_args.push(source.to_str().unwrap().into());
        metal_args.push("-o".into());
        metal_args.push(air_path.to_str().unwrap().into());

        let status = Command::new(&self.xcrun)
            .args(&metal_args)
            .status()
            .map_err(|e| format!("Failed to run xcrun metal: {e}"))?;
        if !status.success() {
            return Err(format!(
                "xcrun metal compile failed for {}",
                source.display()
            ));
        }

        let metallib_args: Vec<&str> = vec![
            "-sdk",
            "macosx",
            "metallib",
            air_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ];
        let status = Command::new(&self.xcrun)
            .args(&metallib_args)
            .status()
            .map_err(|e| format!("Failed to run xcrun metallib: {e}"))?;
        if !status.success() {
            return Err(format!(
                "xcrun metallib link failed for {}",
                source.display()
            ));
        }
        Ok(())
    }
}

/// SCALE (scale-lang.com) compilation target: recompiles the **unmodified
/// CUDA** `.cu` sources for AMD GPUs. SCALE is a drop-in `nvcc` shim
/// (clang-19 based) — but it emits an **AMD GPU code object** (ELF
/// relocatable), not PTX text, and does **not** accept `--ptx`. The device
/// object is produced via `--cuda-device-only -c`. Target arch (e.g.
/// `gfx1151`) selects the per-arch toolchain dir `targets/<arch>/bin/nvcc`
/// (equivalent to `source scaleenv <arch>` without needing a sourced shell).
struct ScaleTarget {
    /// SCALE install root (the `scale-<ver>-Linux` dir containing
    /// `targets/` and `bin/scaleenv`).
    scale_root: PathBuf,
}

impl ComputeTarget for ScaleTarget {
    fn source_extension(&self) -> &str {
        "cu"
    }
    fn output_extension(&self) -> &str {
        // AMD GPU ELF relocatable produced by `--cuda-device-only -c`.
        "o"
    }
    fn uses_cuda_module_api(&self) -> bool {
        // SCALE is a CUDA-compatible toolkit: the runtime loads these
        // AMD-GPU code objects via `cuModuleLoadData`, same as NVIDIA.
        true
    }

    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        // Per-arch SCALE toolchain dir. `targets/<arch>/bin/nvcc` is the
        // arch-pinned compiler (what `scaleenv <arch>` puts on PATH).
        let nvcc = self.scale_root.join("targets").join(arch).join("bin/nvcc");
        if !nvcc.exists() {
            return Err(format!(
                "SCALE arch toolchain not found: {} — `{}` is not a SCALE \
                 target (check kernels/<hw>/HARDWARE.toml `arch` and the \
                 installed SCALE `targets/` dir).",
                nvcc.display(),
                arch
            ));
        }

        // SCALE's `cuModuleLoadData` (HSA loader) accepts the AMD-GPU ELF
        // *relocatable* emitted by `--cuda-device-only -c` and performs the
        // final link itself at module-load time. Verified on gfx1151 / SCALE
        // 1.7.1: the relocatable loads, `cuModuleGetFunction` resolves, and
        // `cuLaunchKernel` runs correctly. Do NOT pre-link with `ld.lld
        // -shared`: that ELF DYN is rejected with CUDA_ERROR_INVALID_IMAGE.
        // Emit the relocatable directly as the loadable module.
        let mut args: Vec<String> = vec!["--cuda-device-only".into(), "-c".into(), "-O3".into()];
        args.extend(extra_flags.iter().cloned());
        args.push(source.to_str().unwrap().into());
        args.push("-o".into());
        args.push(output.to_str().unwrap().into());

        let status = Command::new(&nvcc)
            .args(&args)
            .status()
            .map_err(|e| format!("Failed to run SCALE nvcc ({}): {e}", nvcc.display()))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "SCALE `--cuda-device-only -c` failed for {} (arch {arch})",
                source.display()
            ))
        }
    }
}

/// Native ROCm/HIP compilation target (no SCALE): compiles the (unmodified
/// except for mechanical compat) CUDA `.cu` sources directly with `hipcc` for
/// an AMD `gfx*` arch, emitting a loadable HIP code object. Validated on
/// gfx1151 — 72/92 kernels compile via this recipe with only compat headers +
/// a mask-widen sed; the tensor-core kernels are ported to AMD WMMA separately.
///
/// Pairs with the `libcuda.so` → HIP shim (atlas-kernels/hip/libcuda_hip_shim.cpp):
/// the unchanged cudarc runtime loads these objects via `cuModuleLoadData`
/// (→ `hipModuleLoadData`). build.rs stages the CUDA→HIP compat-header dir and
/// exports its path via `ATLAS_HIP_COMPAT_INCLUDE`; the per-source mask-widen
/// sed runs during the source-mirror step (see build.rs HIP branch).
struct HipTarget {
    hipcc: PathBuf,
}

impl ComputeTarget for HipTarget {
    fn source_extension(&self) -> &str {
        "cu"
    }
    fn output_extension(&self) -> &str {
        // HIP code object (loadable by hipModuleLoadData via the libcuda shim).
        "co"
    }
    fn uses_cuda_module_api(&self) -> bool {
        // The runtime still calls the CUDA driver API; the libcuda→HIP shim
        // maps cuModuleLoadData/cuLaunchKernel/… onto HIP. Same registry path.
        true
    }

    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        // CUDA→HIP compat headers (cuda_runtime.h/cuda_bf16.h/cuda_fp8.h that
        // forward to hip/* and alias __nv_* types) staged by build.rs. Placed
        // first on the include path so the unmodified .cu finds them.
        let compat = std::env::var("ATLAS_HIP_COMPAT_INCLUDE").map_err(|_| {
            "ATLAS_HIP_COMPAT_INCLUDE not set — build.rs must stage the CUDA→HIP \
             compat-header dir before compiling HIP kernels."
                .to_string()
        })?;
        // `-x hip` + force-include hip_runtime.h (kernels that only include
        // cuda_bf16.h otherwise lack blockIdx/threadIdx). `--genco` emits a
        // loadable code object.
        let mut args: Vec<String> = vec![
            "-x".into(),
            "hip".into(),
            "--genco".into(),
            format!("--offload-arch={arch}"),
            "-O3".into(),
            format!("-I{compat}"),
            "-include".into(),
            "hip/hip_runtime.h".into(),
        ];
        // The Windows HIP SDK (ROCm 6.4 clang) lacks the CUDA mask-arg warp
        // intrinsics (__shfl_*_sync/__any_sync/__all_sync/__activemask) that
        // Linux ROCm ships. Force-include the compat shim AFTER hip_runtime.h
        // (so the base __shfl*/__ballot it maps onto are already declared). It
        // is `-I{compat}` on the include path. Windows-only: Linux HIP declares
        // these itself and must not get a second definition.
        if cfg!(windows) {
            args.push("-include".into());
            args.push("atlas_hip_win_shims.h".into());
        }
        // Translate nvcc-specific flags to their hipcc/clang equivalents so
        // KERNEL.toml `extra_nvcc_flags` (authored for nvcc) work on HIP.
        // `--fmad=false` (disable FMA contraction for determinism) → clang's
        // `-ffp-contract=off`; preserving this matters for numeric parity.
        for f in extra_flags {
            match f.as_str() {
                "--fmad=false" => args.push("-ffp-contract=off".into()),
                "--fmad=true" => args.push("-ffp-contract=fast".into()),
                s if s.starts_with("--fmad=") => {}
                other => args.push(other.into()),
            }
        }
        args.push(source.to_str().unwrap().into());
        args.push("-o".into());
        args.push(output.to_str().unwrap().into());

        let status = Command::new(&self.hipcc)
            .args(&args)
            .status()
            .map_err(|e| format!("Failed to run hipcc ({}): {e}", self.hipcc.display()))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "hipcc --genco failed for {} (arch {arch})",
                source.display()
            ))
        }
    }
}

fn find_hipcc() -> PathBuf {
    if let Ok(p) = std::env::var("ATLAS_HIPCC") {
        return PathBuf::from(p);
    }
    let canonical = PathBuf::from("/opt/rocm/bin/hipcc");
    if canonical.exists() {
        return canonical;
    }
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let p = dir.join("hipcc");
            if p.exists() {
                return p;
            }
        }
    }
    panic!("hipcc not found — install ROCm or set ATLAS_HIPCC to its path.");
}

fn find_xcrun() -> PathBuf {
    // Cargo's macOS hosts always have `/usr/bin/xcrun`; PATH lookup is a
    // safety net for unusual toolchain layouts (CI runners with custom
    // Xcode roots, nix shells, etc.).
    let canonical = PathBuf::from("/usr/bin/xcrun");
    if canonical.exists() {
        return canonical;
    }
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let p = dir.join("xcrun");
            if p.exists() {
                return p;
            }
        }
    }
    panic!(
        "xcrun not found — install Xcode Command Line Tools \
         (xcode-select --install) or set PATH to a directory containing xcrun."
    );
}

/// Resolve the compilation target from the HARDWARE.toml vendor field.
/// Falls back to NVIDIA if no vendor is specified.
pub(super) fn resolve_compute_target(vendor: Option<&str>) -> Box<dyn ComputeTarget> {
    match vendor.unwrap_or("nvidia") {
        "nvidia" | "cuda" => {
            let nvcc = find_cuda_dir().join("bin/nvcc");
            Box::new(NvidiaTarget { nvcc })
        }
        "apple" | "metal" => {
            let xcrun = find_xcrun();
            Box::new(AppleTarget { xcrun })
        }
        "amd" | "rocm" | "scale" => {
            let scale_root = super::build_codegen::find_scale_dir();
            Box::new(ScaleTarget { scale_root })
        }
        // Native ROCm/HIP path (no SCALE) — pairs with the libcuda→HIP shim.
        "hip" => Box::new(HipTarget {
            hipcc: find_hipcc(),
        }),
        other => panic!(
            "Unsupported compute vendor '{other}'. Supported: nvidia, apple, amd, hip.\n\
             To add support for a new vendor, implement the ComputeTarget trait \n\
             in atlas-kernels/build_target.rs and atlas-core/src/compute.rs."
        ),
    }
}
