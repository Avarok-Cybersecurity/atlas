// SPDX-License-Identifier: AGPL-3.0-only

//! Contract tests for `conv1d_update_biased_strided`.
//!
//! Two independent guards, both CPU-only so they run in the no-GPU CI:
//!
//! 1. A `MockGpuBackend` GEOMETRY test — the mock hands out one shared kernel
//!    handle, so `(grid, block)` is the only per-launch identity. It pins the
//!    "ONE launch for n rows" property that the whole strided variant exists
//!    for: a regression back to a per-row loop shows up as n launches.
//! 2. A SOURCE test over `kernels/**/causal_conv1d.cu` — `cuLaunchKernel`'s
//!    `void**` form reads one host word per COMPILED parameter, so a launcher
//!    and a kernel that disagree on arity do not fail loudly: the driver
//!    either ignores trailing args (silent wrong strides) or reads one past
//!    the end of the arg array. This pins arity on both sides, pins the two
//!    strided index expressions, and pins that all three target mirrors carry
//!    a byte-identical copy of the entry. (`strix`/`strix-hip` are SYMLINKS to
//!    the `gb10` copy of `causal_conv1d.cu` today, so that last one is free
//!    now — it is what catches a future de-symlink that forgets this entry.)

use super::conv1d_update_biased_strided;
use crate::weight_map::DenseWeight;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, mock::MockGpuBackend};

// Lightning-30B mamba-2 conv shapes.
const D_XBC: u32 = 6144;
const D_CONV: u32 = 4;
const IN_PROJ_SIZE: u32 = 10304;

#[test]
fn one_launch_for_n_rows_at_lightning_geometry() {
    let gpu = MockGpuBackend::new();
    let g: &dyn GpuBackend = &gpu;
    let k = g
        .kernel("causal_conv1d", "causal_conv1d_update_strided")
        .unwrap();
    let w = DenseWeight {
        weight: g.alloc((D_XBC * D_CONV) as usize * 2).unwrap(),
    };

    for n in [2u32, 4, 8] {
        let gpu = MockGpuBackend::new();
        let g: &dyn GpuBackend = &gpu;
        let conv_state = g.alloc((n * D_XBC * D_CONV) as usize * 4).unwrap();
        let input = g.alloc((n * IN_PROJ_SIZE) as usize * 2).unwrap();
        let bias = g.alloc(D_XBC as usize * 4).unwrap();
        let out = g.alloc((n * D_XBC) as usize * 2).unwrap();

        conv1d_update_biased_strided(
            g,
            k,
            conv_state,
            input,
            &w,
            bias,
            out,
            D_XBC,
            D_CONV,
            n,
            IN_PROJ_SIZE,
            D_XBC,
            0,
        )
        .unwrap();

        let l = gpu.launches_snapshot();
        assert_eq!(l.len(), 1, "n={n}: strided conv must be ONE launch, not n");
        // ceil(6144/256) = 24 CTAs on x, one row per blockIdx.y.
        assert_eq!(l[0].grid, [24, n, 1], "n={n}: grid");
        assert_eq!(l[0].block, [256, 1, 1], "n={n}: block");
    }
}

#[test]
fn null_bias_is_still_accepted() {
    // The kernel branches on `bias != nullptr`; the wrapper must pass the
    // caller's pointer through verbatim rather than substituting one.
    let gpu = MockGpuBackend::new();
    let g: &dyn GpuBackend = &gpu;
    let k = KernelHandle(1);
    let w = DenseWeight {
        weight: g.alloc(64).unwrap(),
    };
    conv1d_update_biased_strided(
        g,
        k,
        g.alloc(64).unwrap(),
        g.alloc(64).unwrap(),
        &w,
        DevicePtr::NULL,
        g.alloc(64).unwrap(),
        D_XBC,
        D_CONV,
        1,
        IN_PROJ_SIZE,
        D_XBC,
        0,
    )
    .unwrap();
    assert_eq!(gpu.launch_count(), 1);
}

// ── source contract ───────────────────────────────────────────────────────

const MIRRORS: &[&str] = &[
    "gb10/common/causal_conv1d.cu",
    "strix/common/causal_conv1d.cu",
    "strix-hip/common/causal_conv1d.cu",
];

/// Drop `//` line comments. CUDA signatures carry both commas AND parens
/// inside comments (`// [dim, d_conv] BF16 (shared)`), which is exactly how a
/// naive scan reads a 1-parameter signature ending at the `)` of `(shared)`.
fn strip_line_comments(s: &str) -> String {
    s.lines()
        .map(|l| l.split_once("//").map_or(l, |(h, _)| h))
        .collect::<Vec<_>>()
        .join("\n")
}

/// `extern "C" __global__ void NAME(` … `)` signature text, comments stripped.
fn kernel_signature(src: &str, name: &str) -> String {
    let src = strip_line_comments(src);
    let pat = format!("void {name}(");
    let at = src
        .find(&pat)
        .unwrap_or_else(|| panic!("kernel `{name}` not found"));
    let open = at + pat.len() - 1;
    let close = src[open..]
        .find(')')
        .unwrap_or_else(|| panic!("unterminated signature for `{name}`"))
        + open;
    src[open + 1..close].to_string()
}

fn kernel_src(mirror: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels")
        .join(mirror);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn kernel_arity_matches_the_launcher_arg_count() {
    let src = kernel_src(MIRRORS[0]);
    let sig = kernel_signature(&src, "causal_conv1d_update_strided");
    let params = sig.split(',').filter(|s| !s.trim().is_empty()).count();
    assert_eq!(
        params, 10,
        "causal_conv1d_update_strided must take exactly the 8 base args plus \
         input_stride/output_stride; got:\n{sig}"
    );

    // The launcher lives next door; count its `.arg_*` chain.
    let launcher = include_str!("ssm_mamba_strided.rs");
    let body = launcher
        .split_once("pub fn conv1d_update_biased_strided(")
        .expect("launcher fn")
        .1;
    let body = body.split_once("\n}\n").expect("launcher body end").0;
    let args = body.matches(".arg_").count();
    assert_eq!(
        args, params,
        "launcher passes {args} args but the kernel compiles {params} parameters — \
         cuLaunchKernel reads one host word per COMPILED parameter, so this \
         mismatch is silent at runtime"
    );
}

#[test]
fn strided_entry_indexes_input_and_output_by_the_strides() {
    let src = kernel_src(MIRRORS[0]);
    let at = src.find("void causal_conv1d_update_strided(").unwrap();
    let body = &src[at..];
    for needle in [
        "new_input[(unsigned long long)b * input_stride + ch]",
        "output[(unsigned long long)b * output_stride + ch]",
    ] {
        assert!(
            body.contains(needle),
            "strided entry lost `{needle}` — a batch=n launch would then read \
             row b>=1 from the wrong offset (correct at n=1, silently corrupt above)"
        );
    }
    // conv_state stays DENSE and unparameterised by design: the multi-seq
    // caller proves slot contiguity instead. If someone adds a stride arg the
    // arity test above fires too, but say why here.
    assert!(
        body.contains("conv_state + (b * dim + ch) * d_conv"),
        "conv_state must keep the dense pool-slot layout"
    );
}

#[test]
fn every_target_mirror_carries_the_same_entry() {
    let want = {
        let src = kernel_src(MIRRORS[0]);
        let at = src.find("void causal_conv1d_update_strided(").unwrap();
        let end = src[at..].find("\n}\n").unwrap() + at;
        src[at..end].to_string()
    };
    for m in &MIRRORS[1..] {
        let src = kernel_src(m);
        let at = src
            .find("void causal_conv1d_update_strided(")
            .unwrap_or_else(|| panic!("{m} is missing causal_conv1d_update_strided"));
        let end = src[at..].find("\n}\n").unwrap() + at;
        assert_eq!(
            &src[at..end],
            want,
            "{m} diverged from the gb10 copy of causal_conv1d_update_strided"
        );
    }
}
