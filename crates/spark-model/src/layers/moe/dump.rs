// SPDX-License-Identifier: AGPL-3.0-only
//
//! AVAROK_DUMP_EXPERT_IDS=1 — per-MoE-fire diagnostic dumps shared by
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
//! Toggle on a running server with `-e AVAROK_DUMP_EXPERT_IDS=1`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[inline]
pub fn enabled() -> bool {
    std::env::var("AVAROK_DUMP_EXPERT_IDS").ok().as_deref() == Some("1")
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
        "AVAROK_GATE_INPUT last_tok: |x|={:.4}  first5={:?}",
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
        "AVAROK_GATE_LOGITS last_tok: top10_(idx,val)={:?} mean={:.4} std={:.4}",
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
        "AVAROK_EXPERT_IDS last_tok: indices={:?} weights={:?} sum={:.4}",
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
    // Log-once latch (see `avarok_core::scope`). It holds no model-derived
    // value — the message is rebuilt from the arguments every call — so a
    // stale entry cannot produce a wrong answer, only a suppressed duplicate
    // line after a model swap. Scoping it would thread a logging concern
    // through the call path to prevent one repeated INFO line.
    // Backend-scoped latch: `dump_expert_load` holds a `GpuBackend` and
    // nothing else, and a `static Once` meant only the first model ever
    // dumped its expert load.
    if gpu.op_cache().once("dump:moe_expert_load") {
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
            "AVAROK_EXPERT_LOAD: n_tokens={} avg={} max={} (expert {}) min={} max_m_tiles={} kernel_cap={} truncated={}",
            num_tokens,
            avg_per_expert,
            max_cnt,
            max_idx,
            min_cnt,
            max_m_tiles,
            kernel_max,
            max_cnt > kernel_max
        );
    }
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
        "AVAROK_ROUTED_ONLY last_tok: |x|={:.4} first5={:?}",
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
        "AVAROK_SHARED_OUT last_tok: |x|={:.4} first5={:?}",
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
        "AVAROK_SHARED_GATE last_tok: dot={:.4} sigmoid={:.6}",
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
    tracing::info!(
        "AVAROK_MOE_OUT last_tok: |x|={:.4} first5={:?}",
        mag,
        first5
    );
    Ok(())
}

/// `AVAROK_MOE_ROUTER_MARGIN=1` — how much numerical headroom the router GEMM
/// actually has, in exact BF16 ULPs.
///
/// WHY THIS EXISTS. The router GEMM is pinned to the scalar kernel because a
/// rerouted one "flips top-k on borderline tokens deterministically"
/// (2026-08-12 BFCL regression, see `forward_prefill.rs`). It is also 15% of
/// the MoE stage at 7.0 TFLOP/s — 22% of this box's peak — so the pin is
/// expensive. BFCL is a slow, noisy, DOWNSTREAM proxy for the thing that
/// actually breaks; this measures the thing itself.
///
/// THE MECHANISM IS DISCRETE. Top-k is a step function of the logits: the
/// selected set can only change if a perturbation exceeds the gap between the
/// k-th and (k+1)-th largest logit. So a candidate kernel does NOT need
/// bit-exact logits — it needs a maximum error smaller than that gap. This
/// reports the gap distribution, which converts "is it safe?" into a budget:
/// a kernel whose logits differ by at most N ULPs can flip at most the tokens
/// reported at `<=N`.
///
/// ULPs, NOT ABSOLUTE ERROR, because the logits are stored BF16 — an 8-bit
/// mantissa. Two adjacent representable values differ by one ULP, so a gap of
/// 0 means an EXACT TIE, where the selection is decided by the sort's
/// tie-break and ANY change of bit pattern can reorder it. Those tokens are
/// unprotectable by an error bound and are counted separately.
///
/// The gap is computed on the raw bit patterns: for IEEE floats the bit
/// pattern of a positive value is monotonic in the value, so mapping each
/// BF16 to a total-order key makes the ULP distance an integer subtraction —
/// exact, with no float arithmetic of our own to muddy the measurement.
///
/// Sample size is the reason this is practical: one 8K prefill is ~8000 tokens
/// x 48 layers, so ~380K independent routing decisions in a single request.
pub fn dump_router_margin(
    gpu: &dyn GpuBackend,
    stream: u64,
    gate_logits: DevicePtr,
    n: u32,
    num_experts: u32,
    top_k: u32,
    fp32_logits: bool,
) -> Result<()> {
    if std::env::var("AVAROK_MOE_ROUTER_MARGIN").ok().as_deref() != Some("1") {
        return Ok(());
    }
    if top_k == 0 || num_experts <= top_k {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let ne = num_experts as usize;
    let k = top_k as usize;
    // ULPs are a property of the STORED type, so the same token can be an exact
    // tie in BF16 and a comfortable gap in F32 — which is precisely what this
    // has to be able to show, with one metric across both arms.
    let esz = if fp32_logits { 4usize } else { 2usize };
    let mut raw = vec![0u8; n as usize * ne * esz];
    let _ = gpu.copy_d2h(gate_logits, &mut raw);

    // Total-order key: for IEEE floats, positives are monotonic in their bit
    // pattern and negatives are reversed. Mapping both onto one increasing
    // unsigned makes the ULP distance a plain subtraction, with no float
    // arithmetic of our own to muddy the measurement. Widened to u32 so the two
    // dtypes share one path; BF16 keys just occupy the low 16 bits.
    let key32 = |b: u32| -> u32 {
        if b & 0x8000_0000 != 0 {
            !b
        } else {
            b | 0x8000_0000
        }
    };
    let key16 = |b: u16| -> u32 { (if b & 0x8000 != 0 { !b } else { b | 0x8000 }) as u32 };

    let mut ties = 0usize;
    let mut le: [usize; 5] = [0; 5]; // gap <= 0,1,2,4,8 ULPs
    let mut min_gap = u32::MAX;
    let mut sum_gap = 0u64;
    let mut row: Vec<u32> = Vec::with_capacity(ne);
    for t in 0..n as usize {
        row.clear();
        let bytes = &raw[t * ne * esz..(t + 1) * ne * esz];
        if fp32_logits {
            row.extend(
                bytes
                    .chunks_exact(4)
                    .map(|c| key32(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
            );
        } else {
            row.extend(
                bytes
                    .chunks_exact(2)
                    .map(|c| key16(u16::from_le_bytes([c[0], c[1]]))),
            );
        }
        // Partial select: only the k/k+1 boundary matters, so this is O(ne)
        // rather than a full sort per token.
        let (_, kth, rest) = row.select_nth_unstable_by(k - 1, |a, b| b.cmp(a));
        let kth = *kth;
        let next = *rest.iter().max().unwrap_or(&0);
        let gap = kth.saturating_sub(next);
        sum_gap += gap as u64;
        min_gap = min_gap.min(gap);
        if gap == 0 {
            ties += 1;
        }
        for (i, bound) in [0u32, 1, 2, 4, 8].iter().enumerate() {
            if gap <= *bound {
                le[i] += 1;
            }
        }
    }
    let tot = n as f64;
    tracing::info!(
        "moe-router-margin dtype={} n={n} topk={top_k} ties={ties} ({:.3}%) min_gap={min_gap} \
         mean_gap={:.1} | flippable at <=1ulp {} ({:.3}%)  <=2 {} ({:.3}%)  \
         <=4 {} ({:.3}%)  <=8 {} ({:.3}%)",
        if fp32_logits { "f32" } else { "bf16" },
        ties as f64 / tot * 100.0,
        sum_gap as f64 / tot,
        le[1],
        le[1] as f64 / tot * 100.0,
        le[2],
        le[2] as f64 / tot * 100.0,
        le[3],
        le[3] as f64 / tot * 100.0,
        le[4],
        le[4] as f64 / tot * 100.0,
    );
    Ok(())
}
