// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../avarok-kernels/build_backend.rs"]
mod build_backend;

fn main() {
    // FIRST, before any early return. `rustc-check-cfg` does not cross crates,
    // so each crate must register the names itself or `unexpected_cfgs` fires
    // on whichever target takes the early path -- the scar spark-storage's
    // build script already carries. The RULE lives in build_backend.rs so it
    // can be tested; see crates/avarok-kernels/tests/backend_resolution.rs.
    build_backend::register_cfgs();
    if let Some(os) = build_backend::target_os_from_env() {
        build_backend::emit(std::env::var_os("CARGO_FEATURE_CUDA").is_some(), &os);
    }
    println!("cargo:rerun-if-env-changed=AVAROK_TARGET_HW");
    // `avarok_scale` mirrors spark-runtime's build.rs cfg: it marks the SCALE/
    // AMD (gfx1151) targets (`strix`, `strix-hip`). On these the GPU-visible
    // pool is a unified APU GTT (~60 GB) that cannot hold the FP8 source
    // checkpoint co-resident with its NVFP4 requant result, so the weight
    // loader frees each FP8 source tensor right after requant (see
    // `quantized_from_fp8`). NVIDIA targets leave the cfg unset and keep the
    // current resident-source behavior byte-for-byte.
    println!("cargo:rustc-check-cfg=cfg(avarok_scale)");
    // `avarok_hip` is the strict subset of avarok_scale for the NATIVE-HIP target
    // (`strix-hip`, hipcc — not the SCALE PTX-recompile `strix`). HIP lacks the
    // FP8 *prefill* GEMM kernels (fp8_gemm*/w8a16* are inline-PTX, not yet
    // WMMA-ported), so the FP8→FP8 predequant-for-prefill path has no kernel
    // there; on avarok_hip we skip predequant and use the NVFP4 (w4a16 WMMA)
    // prefill instead. SCALE recompiles the PTX and keeps the FP8 prefill path.
    println!("cargo:rustc-check-cfg=cfg(avarok_hip)");
    let hw = std::env::var("AVAROK_TARGET_HW").unwrap_or_default();
    if hw.starts_with("strix") {
        println!("cargo:rustc-cfg=avarok_scale");
    }
    if hw == "strix-hip" {
        println!("cargo:rustc-cfg=avarok_hip");
    }
}
