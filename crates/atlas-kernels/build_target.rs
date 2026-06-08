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
    /// Whether the compiled output is human-readable text (PTX) or an
    /// opaque binary blob (metallib / HSACO / SPIR-V). Drives whether
    /// the codegen emits `include_str!` or `include_bytes!`.
    fn output_is_text(&self) -> bool;
    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String>;
}

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
    fn output_is_text(&self) -> bool {
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
        args.extend(extra_flags.iter().cloned());
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
    fn output_is_text(&self) -> bool {
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
        other => panic!(
            "Unsupported compute vendor '{other}'. Supported: nvidia, apple.\n\
             To add support for a new vendor, implement the ComputeTarget trait \n\
             in atlas-kernels/build_target.rs and atlas-core/src/compute.rs."
        ),
    }
}
