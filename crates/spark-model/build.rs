// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
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
