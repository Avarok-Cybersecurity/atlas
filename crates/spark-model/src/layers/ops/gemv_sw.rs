// SPDX-License-Identifier: AGPL-3.0-only

//! Decode W4A16 GEMV launchers whose grid is coupled to the CUDA
//! `N_PER_BLOCK` / `N_PER_BLOCK_SW` defines.
//!
//! The single-warp kernel (`w4a16_gemv_sw`) is bit-identical to the 64-thread
//! base (`examples/w4a16_gemv_sw_microtest.rs`). Shipping it as the default
//! decode GEMV is a free occupancy win — **if and only if** the launch grid
//! stays coupled: 8 outputs/block vs the base kernel's 4. Swapping the kernel
//! without swapping the grid writes the wrong outputs.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// Base `w4a16_gemv`: 4 outputs / 256-thread block.
/// SSOT with `kernels/**/w4a16_gemv.cu` `#define N_PER_BLOCK 4`.
pub const W4A16_GEMV_OUTS_PER_BLOCK: u32 = 4;

/// Single-warp `w4a16_gemv_sw`: 8 outputs / 256-thread block.
/// SSOT with `#define N_PER_BLOCK_SW 8`.
pub const W4A16_GEMV_SW_OUTS_PER_BLOCK: u32 = 8;

pub fn w4a16_gemv_grid_x(n: u32) -> u32 {
    div_ceil(n, W4A16_GEMV_OUTS_PER_BLOCK)
}

pub fn w4a16_gemv_sw_grid_x(n: u32) -> u32 {
    div_ceil(n, W4A16_GEMV_SW_OUTS_PER_BLOCK)
}

/// Kill-switch polarity for lossless SW GEMV. ON unless `ATLAS_NO_GEMV_SW` is
/// exactly `"1"`. `=0` does **not** disable (same `== "1"` reading as
/// `ATLAS_NO_LM_HEAD_BATCH_GEMV`).
pub fn gemv_sw_from(no_gemv_sw: Option<&str>) -> bool {
    no_gemv_sw != Some("1")
}

/// SW kernel when the model lever is on **and** the handle resolved.
pub fn use_gemv_sw(lever: bool, sw_handle: KernelHandle) -> bool {
    lever && sw_handle.0 != 0
}

/// Single-warp-per-output W4A16 GEMV (M=1). Grid: `(ceil(N/8), 1, 1)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_sw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_sw_grid_x(n), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Decode GEMV: software-pipelined single-warp when the lever and handle agree.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_decode_gemv(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    gemv_sw: KernelHandle,
    use_sw: bool,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if use_gemv_sw(use_sw, gemv_sw) {
        w4a16_gemv_sw(gpu, gemv_sw, input, weight, output, n, k, stream)
    } else {
        super::quant_dispatch::w4a16_gemv(gpu, gemv, input, weight, output, n, k, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::KernelHandle;
    use std::fs;
    use std::path::{Path, PathBuf};

    #[test]
    fn gemv_sw_ships_on_and_only_the_one_value_kills() {
        assert!(gemv_sw_from(None), "unset → ON");
        assert!(gemv_sw_from(Some("0")), "`=0` is NOT off");
        assert!(gemv_sw_from(Some("")), "empty is NOT off");
        assert!(!gemv_sw_from(Some("1")), "`=1` is the kill");
    }

    #[test]
    fn sw_requires_both_the_lever_and_a_live_handle() {
        assert!(use_gemv_sw(true, KernelHandle(1)));
        assert!(
            !use_gemv_sw(true, KernelHandle(0)),
            "missing kernel falls back"
        );
        assert!(!use_gemv_sw(false, KernelHandle(1)), "kill switch wins");
        assert!(!use_gemv_sw(false, KernelHandle(0)));
    }

    #[test]
    fn sw_grid_covers_every_output_and_is_half_base_when_n_divisible_by_8() {
        for n in 1..=64 {
            assert!(w4a16_gemv_sw_grid_x(n) * W4A16_GEMV_SW_OUTS_PER_BLOCK >= n);
            assert!(w4a16_gemv_grid_x(n) * W4A16_GEMV_OUTS_PER_BLOCK >= n);
        }
        for n in [8u32, 16, 256, 5120, 14336] {
            assert_eq!(
                w4a16_gemv_sw_grid_x(n) * 2,
                w4a16_gemv_grid_x(n),
                "N={n}: SW is 8 outs/block, base is 4 — grid_x must be half"
            );
        }
    }

    fn kernel_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
    }

    fn named_cu(file_name: &str) -> Vec<PathBuf> {
        fn visit(d: &Path, name: &str, out: &mut Vec<PathBuf>) {
            let Ok(rd) = std::fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    visit(&p, name, out);
                } else if p.file_name().is_some_and(|n| n == name) {
                    out.push(p);
                }
            }
        }
        let root = kernel_root();
        let mut files = Vec::new();
        visit(&root, file_name, &mut files);
        files.sort();
        files
    }

    /// POSITIVE: every copy of the GEMV sources pins the same occupancy
    /// constants the Rust launchers use. A new backend copy that changes
    /// `N_PER_BLOCK_SW` without updating the launcher writes the wrong N
    /// slice — silent, not a CUDA error.
    ///
    /// PROVEN BY: changing either `#define` in one `.cu` copy turns this red.
    #[test]
    fn cuda_n_per_block_matches_rust_ssot() {
        let gemv = named_cu("w4a16_gemv.cu");
        assert!(
            gemv.len() >= 3,
            "expected gb10 + strix + strix-hip copies, got {gemv:?}"
        );
        let want_base = format!("#define N_PER_BLOCK {W4A16_GEMV_OUTS_PER_BLOCK}");
        let want_sw = format!("#define N_PER_BLOCK_SW {W4A16_GEMV_SW_OUTS_PER_BLOCK}");
        for p in &gemv {
            let src = fs::read_to_string(p).unwrap();
            assert!(
                src.contains(&want_base),
                "{} missing {want_base}",
                p.display()
            );
            assert!(src.contains(&want_sw), "{} missing {want_sw}", p.display());
        }
        let fused = named_cu("w4a16_gemv_fused.cu");
        assert!(
            !fused.is_empty(),
            "dual_sw / silu_input_sw live in w4a16_gemv_fused.cu"
        );
        for p in &fused {
            let src = fs::read_to_string(p).unwrap();
            assert!(src.contains(&want_sw), "{} missing {want_sw}", p.display());
        }
    }

    /// POSITIVE: SW GEMV must share the 2-chunk K16 pipeline with the 64-thread
    /// kernel. A stride-64 sequential `acc += a*w` copy was 1 ULP lossy on GB10
    /// (`w4a16_gemv_sw_microtest`: gdn in_proj 99.992%, K-tail 99.976%).
    ///
    /// PROVEN BY: restoring `k16 += 64u` in `w4a16_gemv.cu` or dropping
    /// `orig_lane * 2u` from `w4a16_gemv_partial` turns this red.
    #[test]
    fn sw_partial_shares_pipelined_k16_loop() {
        for p in named_cu("w4a16_gemv.cu") {
            let src = fs::read_to_string(&p).unwrap();
            assert!(
                src.contains("orig_lane * 2u"),
                "{}: w4a16_gemv_partial must start k16 at orig_lane*2",
                p.display()
            );
            assert!(
                src.contains("k16 < K16 + 1u"),
                "{}: pipelined K16+1 bound missing",
                p.display()
            );
            assert!(
                !src.contains("k16 += 64u"),
                "{}: stride-64 sequential loop drifted back in",
                p.display()
            );
        }
        for p in named_cu("w4a16_gemv_fused.cu") {
            let src = fs::read_to_string(&p).unwrap();
            assert!(
                src.contains("w4a16_dual_partial"),
                "{}: dual and dual_sw must share w4a16_dual_partial",
                p.display()
            );
            assert!(
                src.contains("orig_lane * 2u"),
                "{}: dual_partial must start k16 at orig_lane*2",
                p.display()
            );
        }
    }

    /// NEGATIVE: attention decode must not launch the base GEMV directly.
    /// A new `ops::w4a16_gemv(` site there ships the 64-thread kernel on
    /// the default path even though `nvfp4_decode_gemv` exists.
    ///
    /// PROVEN BY: restoring any of the pre-PR call sites turns this red.
    #[test]
    fn attention_decode_does_not_call_base_w4a16_gemv() {
        let attn = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/layers/qwen3_attention");
        let mut offenders = Vec::new();
        for rel in [
            "decode/attention_forward.rs",
            "decode/attention_forward_v4.rs",
            "decode/attention_forward_oproj.rs",
            "decode/attention_forward_mla.rs",
            "decode/attention_forward_kv.rs",
            "trait_impl/multi_seq/qkv.rs",
            "trait_impl/multi_seq/attn.rs",
            "trait_impl/multi_seq/attn/o_proj.rs",
            "trait_impl/multi_seq/mla.rs",
        ] {
            let src = fs::read_to_string(attn.join(rel)).unwrap();
            if src.contains("ops::w4a16_gemv(") {
                offenders.push(rel);
            }
        }
        assert!(
            offenders.is_empty(),
            "use nvfp4_decode_gemv (N/8 grid) not ops::w4a16_gemv: {offenders:?}"
        );
    }
}
