// SPDX-License-Identifier: AGPL-3.0-only
//
//! ATLAS_DUMP_EXPERT_IDS=1 — per-MoE-fire diagnostic dumps shared by
//! both the FP8 (`forward_prefill_fp8.rs`) and NVFP4
//! (`forward_prefill.rs`) routed-expert prefill paths.
//!
//! All helpers are no-ops when the env var is unset (single `var()`
//! lookup per call; the device-to-host copies + sync only happen when
//! enabled). They synchronize the active stream before reading, so the
//! values reflect post-kernel state.
//!
//! Used during the 2026-05-20 MoE bug hunt — three compounding bugs in
//! the routed-expert path (kernel v1, missing zero-init, wrong
//! `max_m_tiles`). The dumps were essential for localizing the
//! amplification (chunk-4 L0 expert 200 up_proj |x|=28 vs HF ~5)
//! and verifying the fix landed it in [0.977, 1.021] of HF baseline
//! across all 40 layers. See `project_qwen36_moe_v2_fix` memory.
//!
//! Toggle on a running server with `-e ATLAS_DUMP_EXPERT_IDS=1`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[inline]
pub fn enabled() -> bool {
    std::env::var("ATLAS_DUMP_EXPERT_IDS").ok().as_deref() == Some("1")
}

/// Read a `[num_elements]` BF16 row at `ptr + offset_bytes` to a host
/// `Vec<f32>` (converting BF16 → f32 via shift-left 16).
fn read_bf16_row(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    offset_bytes: usize,
    num_elements: usize,
) -> Vec<f32> {
    let mut buf = vec![0u8; num_elements * 2];
    let _ = gpu.copy_d2h(ptr.offset(offset_bytes), &mut buf);
    buf.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// |x| + first5 of a BF16 row at the last-token position.
fn last_tok_stats(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, width: usize) -> (f32, Vec<f32>) {
    let offset = (n - 1) * width * 2;
    let v = read_bf16_row(gpu, ptr, offset, width);
    let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let first5 = v.iter().take(5).copied().collect();
    (mag, first5)
}

/// Log the gate INPUT (router_in / post-norm hidden) magnitude + first5
/// at the last token.
pub fn dump_gate_input(
    gpu: &dyn GpuBackend,
    stream: u64,
    router_in: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let (mag, first5) = last_tok_stats(gpu, router_in, n as usize, h as usize);
    tracing::info!(
        "ATLAS_GATE_INPUT last_tok: |x|={:.4}  first5={:?}",
        mag,
        first5
    );
    Ok(())
}

/// Log the top-10 gate logits at the last token (post-matmul, pre-softmax).
pub fn dump_gate_logits(
    gpu: &dyn GpuBackend,
    stream: u64,
    gate_logits: DevicePtr,
    n: u32,
    num_experts: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let offset = (n - 1) as usize * num_experts as usize * 2;
    let logits = read_bf16_row(gpu, gate_logits, offset, num_experts as usize);
    let mut idx_val: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    idx_val.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top10: Vec<(usize, f32)> = idx_val.iter().take(10).copied().collect();
    let mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let var: f32 = logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits.len() as f32;
    tracing::info!(
        "ATLAS_GATE_LOGITS last_tok: top10_(idx,val)={:?} mean={:.4} std={:.4}",
        top10,
        mean,
        var.sqrt()
    );
    Ok(())
}

/// Log the top-K expert indices + weights (post-softmax/sigmoid + renorm)
/// at the last token.
pub fn dump_expert_ids(
    gpu: &dyn GpuBackend,
    stream: u64,
    indices_dev: DevicePtr,
    weights_dev: DevicePtr,
    n: u32,
    top_k: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let offset = (n - 1) as usize * top_k as usize * 4;
    let mut idx_buf = vec![0u8; top_k as usize * 4];
    let mut w_buf = vec![0u8; top_k as usize * 4];
    let _ = gpu.copy_d2h(indices_dev.offset(offset), &mut idx_buf);
    let _ = gpu.copy_d2h(weights_dev.offset(offset), &mut w_buf);
    let ids: Vec<u32> = idx_buf
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let ws: Vec<f32> = w_buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    tracing::info!(
        "ATLAS_EXPERT_IDS last_tok: indices={:?} weights={:?} sum={:.4}",
        ids,
        ws,
        ws.iter().sum::<f32>()
    );
    Ok(())
}

/// Log per-expert token counts (sorted-expert histogram).  Fires ONCE
/// per process so the log doesn't flood.  Warns if any expert exceeds
/// `max_m_tiles * 64` (= the kernel's row cap → truncation).
pub fn dump_expert_load(
    gpu: &dyn GpuBackend,
    stream: u64,
    expert_offsets: DevicePtr,
    num_experts: usize,
    num_tokens: usize,
    avg_per_expert: usize,
    max_m_tiles: u32,
) {
    if !enabled() {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if gpu.synchronize(stream).is_err() {
            return;
        }
        let dump_n = num_experts + 1;
        let mut eo_buf = vec![0u8; dump_n * 4];
        let _ = gpu.copy_d2h(expert_offsets, &mut eo_buf);
        let eo: Vec<u32> = eo_buf
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let counts: Vec<u32> = (0..num_experts).map(|i| eo[i + 1] - eo[i]).collect();
        let max_cnt = *counts.iter().max().unwrap_or(&0);
        let min_cnt = *counts.iter().min().unwrap_or(&0);
        let max_idx = counts.iter().position(|&x| x == max_cnt).unwrap_or(0);
        let kernel_max = max_m_tiles * 64;
        tracing::info!(
            "ATLAS_EXPERT_LOAD: n_tokens={} avg={} max={} (expert {}) min={} max_m_tiles={} kernel_cap={} truncated={}",
            num_tokens,
            avg_per_expert,
            max_cnt,
            max_idx,
            min_cnt,
            max_m_tiles,
            kernel_max,
            max_cnt > kernel_max
        );
    });
}

/// Dump the routed-only MoE output buffer (= post-unpermute_reduce,
/// PRE-shared-blend) at the last token.
pub fn dump_routed_only(
    gpu: &dyn GpuBackend,
    stream: u64,
    output: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let (mag, first5) = last_tok_stats(gpu, output, n as usize, h as usize);
    tracing::info!(
        "ATLAS_ROUTED_ONLY last_tok: |x|={:.4} first5={:?}",
        mag,
        first5
    );
    Ok(())
}

/// Dump the shared-expert output (pre-sigmoid-gate) at the last token.
pub fn dump_shared_out(
    gpu: &dyn GpuBackend,
    stream: u64,
    shared_down_out: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let (mag, first5) = last_tok_stats(gpu, shared_down_out, n as usize, h as usize);
    tracing::info!(
        "ATLAS_SHARED_OUT last_tok: |x|={:.4} first5={:?}",
        mag,
        first5
    );
    Ok(())
}

/// Dump the shared-expert gate scalar (dot + sigmoid) at the last token.
pub fn dump_shared_gate(
    gpu: &dyn GpuBackend,
    stream: u64,
    input: DevicePtr,
    gate_weight: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let offset = (n - 1) as usize * h as usize * 2;
    let v_in = read_bf16_row(gpu, input, offset, h as usize);
    let v_g = read_bf16_row(gpu, gate_weight, 0, h as usize);
    let dot: f32 = v_in.iter().zip(v_g.iter()).map(|(a, b)| a * b).sum();
    let sig = 1.0 / (1.0 + (-dot).exp());
    tracing::info!(
        "ATLAS_SHARED_GATE last_tok: dot={:.4} sigmoid={:.6}",
        dot,
        sig
    );
    Ok(())
}

/// Dump the final MoE output buffer (= routed + shared blend) at the
/// last token.  Called after `moe_batched_blend`.
pub fn dump_moe_out(
    gpu: &dyn GpuBackend,
    stream: u64,
    output: DevicePtr,
    n: u32,
    h: u32,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let (mag, first5) = last_tok_stats(gpu, output, n as usize, h as usize);
    tracing::info!("ATLAS_MOE_OUT last_tok: |x|={:.4} first5={:?}", mag, first5);
    Ok(())
}
