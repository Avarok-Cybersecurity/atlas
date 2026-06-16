// SPDX-License-Identifier: AGPL-3.0-only

//! ATLAS_DUMP_GDN / ATLAS_DUMP_EXPERT_IDS debug helpers for
//! `Qwen3SsmLayer::prefill_inner`.
//!
//! Hoisted from `trait_prefill.rs` to keep that file under the 500 LoC
//! cap. These are pure diagnostic readbacks (d2h copy + eprintln/tracing)
//! gated behind env vars; they perform no compute and do not affect the
//! ordered kernel-launch sequence. Each helper mirrors the original
//! inline block 1:1.

use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// True when `ATLAS_DUMP_GDN` is set in the environment.
#[inline]
pub(super) fn dump_gdn_enabled() -> bool {
    std::env::var("ATLAS_DUMP_GDN").is_ok()
}

/// Read `n_f32` BF16 values from `ptr`, return their L2 norm.
fn bf16_l2(gpu: &dyn GpuBackend, ptr: DevicePtr, n_bytes: usize) -> f64 {
    let mut b = vec![0u8; n_bytes];
    let _ = gpu.copy_d2h(ptr, &mut b);
    let mut ss = 0f64;
    for c in b.chunks_exact(2) {
        let x = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
        ss += (x as f64) * (x as f64);
    }
    ss.sqrt()
}

/// Step 1 readback: `hidden_in` and `normed_in` BF16 norms.
pub(super) fn dump_rms_norm(
    gpu: &dyn GpuBackend,
    hidden: DevicePtr,
    normed: DevicePtr,
    stream: u64,
) {
    if !dump_gdn_enabled() {
        return;
    }
    let _ = gpu.synchronize(stream);
    eprintln!("[gdn] hidden_in norm={:.3}", bf16_l2(gpu, hidden, 64 * 2));
    eprintln!("[gdn] normed_in norm={:.3}", bf16_l2(gpu, normed, 64 * 2));
}

/// Step 4/5 readback: GDN decay + beta gate FP32 slices.
pub(super) fn dump_gates(gpu: &dyn GpuBackend, gates_buf: DevicePtr, nv: usize, stream: u64) {
    if !dump_gdn_enabled() {
        return;
    }
    let _ = gpu.synchronize(stream);
    let fp32 = 4usize;
    let n = nv;
    let mut bd = vec![0u8; n * 4];
    let _ = gpu.copy_d2h(gates_buf, &mut bd);
    let vd: Vec<f32> = bd
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut bb = vec![0u8; n * 4];
    let _ = gpu.copy_d2h(gates_buf.offset(nv * fp32), &mut bb);
    let vb: Vec<f32> = bb
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!(
        "[gdn] decay[0..4]={:?} beta[0..4]={:?}",
        &vd[..4.min(vd.len())],
        &vb[..4.min(vb.len())]
    );
}

/// Step 6 readback: `conv_out` BF16 norm (first 64 elements).
pub(super) fn dump_conv(gpu: &dyn GpuBackend, conv_out_buf: DevicePtr, stream: u64) {
    if !dump_gdn_enabled() {
        return;
    }
    let _ = gpu.synchronize(stream);
    eprintln!("[gdn] conv_out norm={:.3}", bf16_l2(gpu, conv_out_buf, 128));
}

/// Step 7 readback: post-L2-norm q norm + v_in norm.
pub(super) fn dump_l2(gpu: &dyn GpuBackend, conv_out_buf: DevicePtr, key_dim: usize, stream: u64) {
    if !dump_gdn_enabled() {
        return;
    }
    let bf16 = 2usize;
    let _ = gpu.synchronize(stream);
    eprintln!(
        "[gdn] post_l2norm_q norm={:.3}",
        bf16_l2(gpu, conv_out_buf, 128)
    );
    let _ = gpu.synchronize(stream);
    eprintln!(
        "[gdn] v_in norm={:.3}",
        bf16_l2(gpu, conv_out_buf.offset(key_dim * 2 * bf16), 128)
    );
}

/// `gdnmag!`-equivalent: BF16 norm + nonfinite flag + first-6 values for a tag.
pub(super) fn dump_mag(gpu: &dyn GpuBackend, tag: &str, ptr: DevicePtr, n: usize, stream: u64) {
    if !dump_gdn_enabled() {
        return;
    }
    let _ = gpu.synchronize(stream);
    let mut b = vec![0u8; n * 2];
    let _ = gpu.copy_d2h(ptr, &mut b);
    let mut ss = 0f64;
    let mut nf = false;
    for c in b.chunks_exact(2) {
        let x = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
        if !x.is_finite() {
            nf = true;
        }
        ss += (x as f64) * (x as f64);
    }
    let mut first = [0f32; 6];
    for ii in 0..6.min(b.len() / 2) {
        first[ii] = f32::from_bits((u16::from_le_bytes([b[ii * 2], b[ii * 2 + 1]]) as u32) << 16);
    }
    eprintln!(
        "[gdn] {} norm={:.6} nonfinite={} first={:?}",
        tag,
        ss.sqrt(),
        nf,
        first
    );
}

/// ATLAS_DUMP_EXPERT_IDS=1 residual_add_rms_norm input attribution:
/// logs `hidden`, `out_proj`, and their sum for the last token.
pub(super) fn dump_prenorm_inputs(
    gpu: &dyn GpuBackend,
    hidden: DevicePtr,
    out_proj_buf: DevicePtr,
    num_tokens: usize,
    h: usize,
    stream: u64,
) -> anyhow::Result<()> {
    if std::env::var("ATLAS_DUMP_EXPERT_IDS").ok().as_deref() != Some("1") {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let offset = (num_tokens - 1) * h * 2;
    let read = |p: DevicePtr| -> Vec<f32> {
        let mut buf = vec![0u8; h * 2];
        let _ = gpu.copy_d2h(p, &mut buf);
        buf.chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect()
    };
    let v_h = read(hidden.offset(offset));
    let n_h = v_h.iter().map(|x| x * x).sum::<f32>().sqrt();
    let v_o = read(out_proj_buf.offset(offset));
    let n_o = v_o.iter().map(|x| x * x).sum::<f32>().sqrt();
    tracing::info!(
        "ATLAS_PRENORM_HIDDEN last_tok: |x|={:.4} first5={:?}",
        n_h,
        &v_h[..5]
    );
    tracing::info!(
        "ATLAS_PRENORM_OUTPROJ last_tok: |x|={:.4} first5={:?}",
        n_o,
        &v_o[..5]
    );
    let v_sum: Vec<f32> = v_h.iter().zip(v_o.iter()).map(|(a, b)| a + b).collect();
    let n_sum = v_sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    tracing::info!(
        "ATLAS_PRENORM_SUM (hidden+out_proj): |x|={:.4} first5={:?}",
        n_sum,
        &v_sum[..5]
    );
    Ok(())
}
