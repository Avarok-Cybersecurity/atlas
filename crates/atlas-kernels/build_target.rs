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
/// compiler and output format so the same build.rs works for NVIDIA (nvcc→PTX),
/// AMD (hipcc→HSACO), Apple (xcrun→metallib), or Intel (icpx→SPIR-V).
///
/// Only NVIDIA is implemented today. Other vendors panic at resolve time.
pub(super) trait ComputeTarget {
    fn source_extension(&self) -> &str;
    fn output_extension(&self) -> &str;
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

    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        let mut args = vec!["--ptx".into(), format!("-arch={arch}"), "-O3".into()];
        args.extend(extra_flags.iter().cloned());
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

/// Resolve the compilation target from the HARDWARE.toml vendor field.
/// Falls back to NVIDIA if no vendor is specified.
pub(super) fn resolve_compute_target(vendor: Option<&str>) -> Box<dyn ComputeTarget> {
    match vendor.unwrap_or("nvidia") {
        "nvidia" | "cuda" => {
            let nvcc = find_cuda_dir().join("bin/nvcc");
            Box::new(NvidiaTarget { nvcc })
        }
        other => panic!(
            "Unsupported compute vendor '{other}'. Only 'nvidia' is implemented.\n\
             To add support for a new vendor, implement the ComputeTarget trait \n\
             in atlas-kernels/build.rs and atlas-core/src/compute.rs."
        ),
    }
}
