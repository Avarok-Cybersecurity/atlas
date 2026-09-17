// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    // `atlas_scale` mirrors spark-runtime's build.rs cfg: it marks the SCALE/
    // AMD targets. On these the GPU-visible pool is a unified APU GTT (~60 GB)
    // that cannot hold the FP8 source checkpoint co-resident with its NVFP4
    // requant result, so the weight loader frees each FP8 source tensor right
    // after requant (see `quantized_from_fp8`). NVIDIA targets leave the cfg
    // unset and keep the current resident-source behavior byte-for-byte.
    println!("cargo:rustc-check-cfg=cfg(atlas_scale)");
    // `atlas_hip` is the strict subset of atlas_scale for the NATIVE-HIP
    // vendor (hipcc — not the SCALE PTX-recompile path). HIP lacks the FP8
    // *prefill* GEMM kernels (fp8_gemm*/w8a16* are inline-PTX, not yet
    // WMMA-ported), so the FP8→FP8 predequant-for-prefill path has no kernel
    // there; on atlas_hip we skip predequant and use the NVFP4 (w4a16 WMMA)
    // prefill instead. SCALE recompiles the PTX and keeps the FP8 prefill path.
    println!("cargo:rustc-check-cfg=cfg(atlas_hip)");

    // ── VENDOR, not name ──────────────────────────────────────────────────
    // These two cfgs used to key on `hw.starts_with("strix")` and
    // `hw == "strix-hip"`, which made the host build a function of the
    // DIRECTORY NAME. The kernel side is not: `prefill_paged_compute.cuh`
    // pins `#define BR64 32` under `#if defined(__SCALE__)`, i.e. for EVERY
    // SCALE target regardless of what its directory is called, and
    // `ops/prefill_attn_main_{a,b}.rs` must select the matching 32-row host
    // grid stride or the two disagree about the tile the kernel was compiled
    // for. A second SCALE target whose name does not begin with "strix"
    // (`kernels/r9700`, gfx1201) would therefore have compiled 32-row kernels
    // and launched them with a 64-row stride — silently, with no build error.
    //
    // So the signal is `[hardware].vendor` in the same `kernels/<hw>/
    // HARDWARE.toml` that atlas-kernels/build.rs reads to pick the compiler:
    // `amd` (SCALE) and `hip` (native ROCm) are the two vendors that reach a
    // SCALE/HIP device, and `hip` alone is the native-HIP subset. Behaviour is
    // unchanged for every existing target — strix declares `amd`, strix-hip
    // declares `hip`, gb10/hopper/b200 declare `nvidia`, metal declares
    // `apple` — and an unset ATLAS_TARGET_HW still resolves to gb10, i.e.
    // neither cfg.
    let hw = std::env::var("ATLAS_TARGET_HW").unwrap_or_else(|_| "gb10".to_string());
    let hardware_toml =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|crates| crates.parent())
            .expect("crates/<crate> sits two levels below the workspace root")
            .join("kernels")
            .join(&hw)
            .join("HARDWARE.toml");
    println!("cargo:rerun-if-changed={}", hardware_toml.display());

    // An unreadable or vendorless file leaves both cfgs unset — the NVIDIA
    // behaviour, which is what a build that cannot see the kernels tree got
    // before this too. A wrong `atlas_scale` would mis-stride prefill on an
    // NVIDIA box; a missing one costs a bring-up bug on a target that has not
    // booted yet. Silence in the safe direction.
    match hardware_vendor(&hardware_toml).as_deref() {
        Some("amd") => println!("cargo:rustc-cfg=atlas_scale"),
        Some("hip") => {
            println!("cargo:rustc-cfg=atlas_scale");
            println!("cargo:rustc-cfg=atlas_hip");
        }
        _ => {}
    }
}

/// `[hardware].vendor` from a `HARDWARE.toml`, or `None` if it cannot be read.
///
/// Same key, same file and same fallible shape as spark-runtime's
/// `hardware_arch` — one hardware declaration, read the same way by both.
fn hardware_vendor(path: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc: toml::Value = text.parse().ok()?;
    Some(doc.get("hardware")?.get("vendor")?.as_str()?.to_string())
}
