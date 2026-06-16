// SPDX-License-Identifier: AGPL-3.0-only

//! ATLAS_MTP_DEBUG_NORMS=1 diagnostic readbacks for `MtpHead::forward_one`.
//!
//! Hoisted from `forward.rs` to keep that file under the 500 LoC cap.
//! These are pure diagnostic readbacks (d2h copy + tracing) gated behind
//! the env var; they perform no compute and do not affect the ordered
//! kernel-launch sequence. Each helper mirrors the original inline block
//! 1:1.

use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// True when `ATLAS_MTP_DEBUG_NORMS=1` is set in the environment.
#[inline]
pub(super) fn enabled() -> bool {
    std::env::var("ATLAS_MTP_DEBUG_NORMS").as_deref() == Ok("1")
}

/// L2 norm of a BF16 GPU buffer (NaN reads back as NaN, err ⇒ NaN).
pub(super) fn mtp_dbg_l2(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> f64 {
    let mut b = vec![0u8; n * 2];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return f64::NAN;
    }
    b.chunks_exact(2)
        .map(|c| {
            let f = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
            f * f
        })
        .sum::<f64>()
        .sqrt()
}

/// L2 norm of a BF16 GPU buffer (err ⇒ -1.0); used by the step-12 dump.
fn mtp_dbg_l2_neg1(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> f64 {
    let mut b = vec![0u8; n * 2];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return -1.0;
    }
    b.chunks_exact(2)
        .map(|c| {
            let f = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
            f * f
        })
        .sum::<f64>()
        .sqrt()
}

/// Step 1-3 readback: embed / normed-embed / normed-hidden / concat norms.
pub(super) fn dump_concat(
    gpu: &dyn GpuBackend,
    embed_out: DevicePtr,
    normed_embed: DevicePtr,
    normed_hidden: DevicePtr,
    concat_out: DevicePtr,
    h: u32,
    stream: u64,
) {
    if !enabled() {
        return;
    }
    gpu.synchronize(stream).ok();
    tracing::warn!(
        "MTP_DBG s1-embed ||={:.4} s2-n_embed ||={:.4} s2-n_hidden ||={:.4} s3-concat ||={:.4}",
        mtp_dbg_l2(gpu, embed_out, h as usize),
        mtp_dbg_l2(gpu, normed_embed, h as usize),
        mtp_dbg_l2(gpu, normed_hidden, h as usize),
        mtp_dbg_l2(gpu, concat_out, (h * 2) as usize),
    );
}

/// Step 4 readback: fc-projected hidden norm.
pub(super) fn dump_fc(gpu: &dyn GpuBackend, hidden: DevicePtr, h: u32, stream: u64) {
    if !enabled() {
        return;
    }
    gpu.synchronize(stream).ok();
    tracing::warn!(
        "MTP_DBG s4-fc_hidden ||={:.4}",
        mtp_dbg_l2(gpu, hidden, h as usize)
    );
}

/// Step 7 readback: pre-gate attention output + gate norms.
pub(super) fn dump_attn(
    gpu: &dyn GpuBackend,
    attn_out: DevicePtr,
    gate_ptr: DevicePtr,
    nq_hd: usize,
    stream: u64,
) {
    if !enabled() {
        return;
    }
    gpu.synchronize(stream).ok();
    tracing::warn!(
        "MTP_DBG s7-attn_out(pre-gate) ||={:.4}  gate ||={:.4}",
        mtp_dbg_l2(gpu, attn_out, nq_hd),
        mtp_dbg_l2(gpu, gate_ptr, nq_hd)
    );
}

/// Step 12 readback: localize the constant-0 draft via input_hidden →
/// final_normed → logits L2 norms.
pub(super) fn dump_final(
    gpu: &dyn GpuBackend,
    target_hidden: DevicePtr,
    final_normed: DevicePtr,
    logits: DevicePtr,
    h: u32,
    v: u32,
    stream: u64,
) {
    if !enabled() {
        return;
    }
    gpu.synchronize(stream).ok();
    // The residual stream is always BF16, so the saved hidden is BF16.
    let hin = mtp_dbg_l2_neg1(gpu, target_hidden, h as usize);
    tracing::warn!(
        "MTP_DEBUG_NORMS: ||input_hidden||={:.4} ||final_normed||={:.4} ||logits||={:.4}",
        hin,
        mtp_dbg_l2_neg1(gpu, final_normed, h as usize),
        mtp_dbg_l2_neg1(gpu, logits, v as usize)
    );
}
