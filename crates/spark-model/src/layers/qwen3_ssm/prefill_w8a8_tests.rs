// SPDX-License-Identifier: AGPL-3.0-only

//! Clause-by-clause contract for the SSM QKVZ cuBLASLt arm's selection rule.
//! Every case defaults to SELECTING and perturbs exactly one thing, so a
//! failure names the clause. The numerics are the GPU microtest's job.

use super::prefill_w8a8::qkvz_cublas_selected;
use crate::layers::ops::cublas_fp8_m_pad;
use crate::weight_map::WeightQuantFormat;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

/// Qwen3.8-27B GDN: hidden 5120, fused `in_proj_qkvz` N = 10240 + 6144.
const H: u32 = 5120;
const QKVZ: u32 = 16384;
/// The 2026-09-11 H100 trace prompt.
const M: u32 = 1193;
const KMAJOR_K: KernelHandle = KernelHandle(0xC0DE);
const KMAJOR_BUF: DevicePtr = DevicePtr(0x1000);

fn m_pad() -> u32 {
    cublas_fp8_m_pad(M)
}

fn out_bytes() -> usize {
    m_pad() as usize * QKVZ as usize * 2
}

fn scale_bytes() -> usize {
    m_pad() as usize * (H as usize / 128) * 4
}

#[allow(clippy::too_many_arguments)]
fn selected(
    cublas_ssm: bool,
    fmt: WeightQuantFormat,
    n: u32,
    k: u32,
    out_capacity: usize,
    kmajor_k: KernelHandle,
    kmajor_buf: DevicePtr,
    kmajor_capacity: usize,
) -> bool {
    qkvz_cublas_selected(
        cublas_ssm,
        fmt,
        m_pad(),
        n,
        k,
        out_capacity,
        kmajor_k,
        kmajor_buf,
        kmajor_capacity,
    )
}

fn ready() -> bool {
    selected(
        true,
        WeightQuantFormat::Fp8BlockScaled,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(),
    )
}

#[test]
fn selected_at_the_real_qkvz_shape_when_the_ssm_family_is_armed() {
    assert!(ready(), "the H100 QKVZ shape must take the cuBLASLt arm");
}

/// The whole point of the scoped lever: arming the dense FFN must leave this
/// projection on the in-tree kernel. `ATLAS_CUBLAS_GEMM=1` used to arm both,
/// and the SSM side cost 167772160 B of BF16 weight dequant per layer.
#[test]
fn not_selected_when_the_ssm_family_is_not_armed() {
    assert!(!selected(
        false,
        WeightQuantFormat::Fp8BlockScaled,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(),
    ));
}

#[test]
fn not_selected_for_non_block_scaled_weights() {
    // cuBLASLt is told the weight scales are a BLK128x128 `[N/128, K/128]`
    // grid; a per-row `[N]` vector read as that grid is silent garbage.
    for fmt in [
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8SingleScale,
    ] {
        assert!(
            !selected(
                true,
                fmt,
                QKVZ,
                H,
                out_bytes(),
                KMAJOR_K,
                KMAJOR_BUF,
                scale_bytes()
            ),
            "{fmt:?} must not take the block-scaled cuBLASLt arm"
        );
    }
}

#[test]
fn not_selected_for_unaligned_shapes() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    let case = |n, k| {
        selected(
            true,
            f,
            n,
            k,
            out_bytes(),
            KMAJOR_K,
            KMAJOR_BUF,
            scale_bytes(),
        )
    };
    // N not a multiple of 128: the weight scale grid is `[N/128, K/128]`.
    assert!(!case(QKVZ + 1, H));
    // K not a multiple of 128: the activation quantizer emits one FP32 scale
    // per 128-wide K group, so a ragged tail has no scale.
    assert!(!case(QKVZ, H + 1));
    // K/128 not a multiple of 4: cuBLAS requires that column stride to be one
    // ("Scaling factors layouts"), and Atlas hands the grid over as-is.
    assert!(!case(QKVZ, 128 * 3));
    assert!(case(QKVZ, 128 * 4));
}

/// The cuBLASLt helper rounds M up to 16 and WRITES the phantom rows. The
/// destination is an arena buffer shared with its neighbours, so a short one
/// must fall back rather than spill into the next buffer.
#[test]
fn not_selected_when_the_output_buffer_cannot_hold_the_padded_m() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    let short = out_bytes() - 1;
    assert!(!selected(
        true,
        f,
        QKVZ,
        H,
        short,
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes()
    ));
    assert!(selected(
        true,
        f,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes()
    ));
}

/// Without the VEC128 transpose the GEMM is fast and WRONG (H100 2026-09-11:
/// rel_rms 7.7e-2 / ~33 000 BF16 ULP against the in-tree kernel on identical
/// FP8 bytes). Missing kernel, missing scratch or a scratch too small for the
/// padded M all have to decline.
#[test]
fn not_selected_without_a_usable_kmajor_scale_adapter() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    // Guarded on the default layout: `ATLAS_CUBLAS_SCALE_LAYOUT=rowmajor` is a
    // deliberate measurement control that needs no adapter, and the selector
    // says so — but it is a process-wide `OnceLock`, so this test only claims
    // the default.
    if !crate::layers::ops::cublas_scale_layout_kmajor() {
        return;
    }
    assert!(!selected(
        true,
        f,
        QKVZ,
        H,
        out_bytes(),
        KernelHandle(0),
        KMAJOR_BUF,
        scale_bytes()
    ));
    assert!(!selected(
        true,
        f,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        DevicePtr::NULL,
        scale_bytes()
    ));
    assert!(!selected(
        true,
        f,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes() - 1
    ));
}
