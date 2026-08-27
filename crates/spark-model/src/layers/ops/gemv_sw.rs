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
    use spark_runtime::gpu::mock::MockGpuBackend;
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

    #[test]
    fn decode_dispatch_uses_the_selected_handle_and_matching_grid() {
        for (lever, sw_handle, expected_handle, expected_grid_x) in [
            (true, KernelHandle(22), 22, 2),
            (false, KernelHandle(22), 11, 3),
            (true, KernelHandle(0), 11, 3),
        ] {
            let gpu = MockGpuBackend::new();
            w4a16_decode_gemv(
                &gpu,
                KernelHandle(11),
                sw_handle,
                lever,
                DevicePtr::NULL,
                &QuantizedWeight::null(),
                DevicePtr::NULL,
                9,
                128,
                0,
            )
            .unwrap();
            let launches = gpu.launches_snapshot();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].func, expected_handle);
            assert_eq!(launches[0].grid, [expected_grid_x, 1, 1]);
            assert_eq!(launches[0].block, [256, 1, 1]);
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

    /// Split the declaration starting at `sig` into (parameter list, body).
    /// The body is brace-matched, so nested blocks are kept and the next
    /// function is not swept in.
    fn fn_signature_and_body<'a>(src: &'a str, sig: &str) -> (&'a str, &'a str) {
        let start = src
            .find(sig)
            .unwrap_or_else(|| panic!("signature `{sig}` not found"));
        let open = start
            + src[start..]
                .find('{')
                .expect("no body brace after signature");
        let mut depth = 0usize;
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return (&src[start..open], &src[open..=open + i]);
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced braces after `{sig}`");
    }

    fn fn_body<'a>(src: &'a str, sig: &str) -> &'a str {
        fn_signature_and_body(src, sig).1
    }

    /// (file, partial signature, `__constant__` table it must NOT index,
    ///  callers that must hand it a shared-staged copy)
    const DECODE_PARTIALS: &[(&str, &str, &str, &[(&str, &str)])] = &[
        (
            "w4a16_gemv.cu",
            "__device__ __forceinline__ float w4a16_gemv_partial(",
            "E2M1_LUT",
            &[("w4a16_gemv", "s_lut"), ("w4a16_gemv_sw", "warp_lut")],
        ),
        (
            "w4a16_gemv_fused.cu",
            "__device__ __forceinline__ float w4a16_dual_partial(",
            "E2M1_LUT_FUSED_W4",
            &[
                ("w4a16_gemv_dual", "s_lut"),
                ("w4a16_gemv_dual_sw", "warp_lut"),
            ],
        ),
        (
            "w4a16_gemv_fused.cu",
            "__device__ __forceinline__ float w4a16_silu_partial(",
            "E2M1_LUT_FUSED_W4",
            &[("w4a16_gemv_silu_input_sw", "warp_lut")],
        ),
    ];

    /// POSITIVE: every M=1 decode GEMV partial must index a SHARED-staged copy
    /// of the E2M1 table, never `__constant__` memory directly.
    ///
    /// WHY (this is the PR #479 regression, not style): the table index is a
    /// data-dependent weight nibble. `__constant__` is a BROADCAST cache — a
    /// warp request replays once per distinct address, and 32 lanes over NVFP4
    /// weights cover ~14 of the 16 entries, so one lookup costs ~14
    /// transactions. There is exactly one lookup per weight element, i.e. on
    /// every FMA of the decode inner loop. Shared memory answers all 16
    /// indices from 16 distinct banks in one conflict-free transaction.
    ///
    /// Staging is numerically inert — `s_lut[i]` is a bit-exact FP32 copy — so
    /// this coexists with `sw_partial_shares_pipelined_k16_loop`, which pins
    /// the association order that buys base/SW bit-identity.
    ///
    /// PROVEN BY: swapping any `lut[byte_val ...]` back to
    /// `E2M1_LUT[byte_val ...]` turns this red (SASS: 1 LDS + 32 indexed
    /// `LDC c[0x3][R]` instead of 33 LDS + 1 LDC in `w4a16_gemv`).
    #[test]
    fn decode_gemv_partials_index_a_shared_staged_lut() {
        for &(file, sig, table, callers) in DECODE_PARTIALS {
            let partial = sig
                .strip_suffix('(')
                .and_then(|sig| sig.split_whitespace().last())
                .expect("partial signature ends in a function name");
            let paths = named_cu(file);
            assert!(paths.len() >= 3, "{file}: expected 3 backend copies");
            for path in paths {
                let src = fs::read_to_string(&path).unwrap();
                let where_ = format!("{}::{sig}", path.display());
                let (params, body) = fn_signature_and_body(&src, sig);
                assert!(
                    !body.contains(&format!("{table}[byte_val")),
                    "{where_}: data-dependent index into __constant__ {table} \
                     serializes the warp — take the staged `lut` instead"
                );
                assert!(
                    body.contains("lut[byte_val"),
                    "{where_}: must dequant through the staged `lut` parameter"
                );
                assert!(
                    params.contains("const float* __restrict__ lut"),
                    "{where_}: must accept the staged table as `const float* __restrict__ lut`"
                );
                for &(caller, expected_lut) in callers {
                    let cb = fn_body(&src, &format!("void {caller}("));
                    assert!(
                        cb.contains("__shared__ float s_lut"),
                        "{}::{caller}: must stage the E2M1 table in shared memory",
                        path.display()
                    );
                    let calls: Vec<_> = cb
                        .lines()
                        .filter(|line| line.contains(&format!("{partial}(")))
                        .collect();
                    assert!(
                        !calls.is_empty(),
                        "{}::{caller}: no {partial} call",
                        path.display()
                    );
                    for call in calls {
                        assert!(
                            call.contains(&format!(", {expected_lut})")),
                            "{}::{caller}: `{partial}` must receive {expected_lut}, got `{}`",
                            path.display(),
                            call.trim()
                        );
                        assert!(
                            !call.contains(table),
                            "{}::{caller}: `{partial}` received constant-memory {table}",
                            path.display()
                        );
                    }
                }
            }
        }
    }

    /// STRUCTURAL: the single-warp kernels must stage the LUT PER WARP and
    /// stay free of block barriers.
    ///
    /// Two invariants the shared staging leans on, both silently breakable:
    ///   1. `w4a16_gemv_sw` / `_dual_sw` / `_silu_input_sw` early-return on
    ///      `n >= N`, which is warp-uniform (`n = blockIdx.x*8 + tid/32`) but
    ///      NOT block-uniform. A `__syncthreads()` after that return is a
    ///      divergent barrier — undefined behaviour, not a compile error. So
    ///      the staging must publish with `__syncwarp()`, and these kernels
    ///      must contain no `__syncthreads()` at all (the documented
    ///      "no smem, no __syncthreads in the reduction" property).
    ///   2. One private 16-float row per warp, so the row count must track
    ///      `N_PER_BLOCK_SW`. A hardcoded 8 would index out of bounds the day
    ///      that define moves. 8 rows = 512 B/block, ~50x under the smem that
    ///      would cap occupancy at 8 blocks/SM, so it is occupancy-neutral.
    ///
    /// PROVEN BY: replacing `__syncwarp()` with `__syncthreads()`, or writing
    /// `s_lut[8][16]`, turns this red.
    #[test]
    fn sw_gemv_stages_the_lut_per_warp_without_a_block_barrier() {
        let want_rows = "__shared__ float s_lut[N_PER_BLOCK_SW][16]";
        for (file, kernels, helper) in [
            (
                "w4a16_gemv.cu",
                &["w4a16_gemv_sw"][..],
                "stage_e2m1_lut_warp",
            ),
            (
                "w4a16_gemv_fused.cu",
                &["w4a16_gemv_dual_sw", "w4a16_gemv_silu_input_sw"][..],
                "stage_e2m1_lut_fused_warp",
            ),
        ] {
            for path in named_cu(file) {
                let src = fs::read_to_string(&path).unwrap();
                let hb = fn_body(&src, &format!("void {helper}("));
                assert!(
                    hb.contains("__syncwarp()"),
                    "{}::{helper}: warp-scoped staging must publish with __syncwarp()",
                    path.display()
                );
                for k in kernels {
                    let kb = fn_body(&src, &format!("void {k}("));
                    assert!(
                        kb.contains(want_rows),
                        "{}::{k}: per-warp LUT rows must be sized by N_PER_BLOCK_SW",
                        path.display()
                    );
                    assert!(
                        kb.contains(&format!("{helper}(s_lut[local_out], lane)")),
                        "{}::{k}: must stage its own warp row",
                        path.display()
                    );
                    assert!(
                        !kb.contains("__syncthreads()"),
                        "{}::{k}: block barrier after a warp-uniform early return is \
                         divergent UB — and it would undo the barrier-free reduction",
                        path.display()
                    );
                }
            }
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
            if src.contains("w4a16_gemv(") {
                offenders.push(rel);
            }
        }
        assert!(
            offenders.is_empty(),
            "use nvfp4_decode_gemv (N/8 grid) not ops::w4a16_gemv: {offenders:?}"
        );
    }
}
