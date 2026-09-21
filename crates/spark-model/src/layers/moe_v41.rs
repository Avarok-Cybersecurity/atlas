// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash MoE on streamed, still-quantized experts.
//!
//! The routed experts never leave their Q2_K / Q3_K blocks: per token the
//! router picks `topk` of 384, the [`ExpertLru`](spark_runtime::weights::expert_stream::ExpertLru) gathers those experts' raw
//! slices into its device-visible slots (misses read from the SSD), and each
//! expert runs as three K-quant GEMVs on the raw blocks (`kquant_mmvq_q2_k_w` for
//! gate and up, `kquant_mmvq_q3_k_w` for down) with the activation quantised to
//! q8_1 in between. The shared expert every token goes through is resident bf16
//! and runs on the dense GEMM.
//!
//! Routing follows the CPU reference (`deepseek_v41_ref::moe::gate`) exactly:
//! the logits are an f32-accumulated GEMM of the bf16 input against the bf16
//! gate weight, then `sqrt(softplus(logit / temp))`, and the top-k is chosen by
//! `score + correction_bias` while the weights are the unbiased scores,
//! renormalised and scaled by `route_scale`. The selection runs on the CPU from
//! the downloaded logits (`[tokens, 384]` f32), which keeps the tie-breaking the
//! reference's and costs nothing at decode.
//!
//! Oracle: `moe_v41_tests.rs`, synthetic Q2_K / Q3_K experts on a pinned arena
//! against the CPU decoders + q8_1 emulation + the reference's expert math.

use spark_runtime::gpu::{DevicePtr, KernelHandle};

use crate::layers::ops::ResidentMat;

mod device_route;
mod forward;

/// Layers the device slot table covers (V4.1 has 40; the loader checks).
pub(crate) const SLOT_TABLE_LAYERS: usize = 64;
mod init;
mod route;
mod shared;
mod single;

const MODULE: &str = "moe_v41";
const GEMM_MODULE: &str = "gemm";

#[derive(Clone, Debug)]
pub struct MoeV41Cfg {
    pub dim: usize,
    pub inter: usize,
    pub n_routed: usize,
    pub topk: usize,
    pub gate_temp: f32,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub max_tokens: usize,
}

/// One layer's resident MoE weights: the router and the shared expert.
pub struct MoeV41LayerWeights {
    pub layer: u32,
    /// bf16 `[n_routed, dim]`
    pub gate_w: DevicePtr,
    /// f32 `[n_routed]`, host: the selection runs on the CPU
    pub gate_bias: Vec<f32>,
    /// The same bias on the device, for the device-side selection.
    pub gate_bias_dev: DevicePtr,
    /// `[inter, dim]`, `[dim, inter]`, `[inter, dim]`: bf16 or the GGUF's
    /// Q2_K (w1, w3) / Q3_K (w2) blocks on the routed experts' kernels
    pub shared_w1: ResidentMat,
    pub shared_w2: ResidentMat,
    pub shared_w3: ResidentMat,
}

/// The router of one layer alone: what predicting a layer's selection from
/// an earlier layer's input needs (see `MoeV41::predict_launch`).
pub struct RouterWeights {
    pub layer: u32,
    /// bf16 `[n_routed, dim]`
    pub gate_w: DevicePtr,
    pub gate_bias: Vec<f32>,
}

/// `ATLAS_DS41_PREFETCH_K` (default 0 = off), read once: how many of the
/// next layer's predicted experts to start reading while this layer computes
/// (needs `ATLAS_DS41_READER_POOL=1`). Off by default: on the 09-19 standard
/// the predictor's router launch (2.7 ms a token) and the wasted reads on
/// the shared disk cost more than the caught misses saved (K=4: 12.50/16.67,
/// K=6: 13.63/16.39, against 14.20/17.73 without).
pub fn prefetch_k() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("ATLAS_DS41_PREFETCH_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

struct Kernels {
    gemm: KernelHandle,
    /// bf16 shared expert at m = 1 (the tiled GEMM idles 15 of 16 rows)
    gemv: KernelHandle,
    gemm_f32out: KernelHandle,
    /// router logits at m <= 8: strict-order GEMV, bit-identical to gemm_f32out
    router_gemv: KernelHandle,
    /// the same logits from the products staged in shared memory, N blocks (decode)
    router_gemv_staged: KernelHandle,
    q8_rows: KernelHandle,
    mmvq_q2k: KernelHandle,
    mmvq_q3k: KernelHandle,
    /// the single-token arm: every routed expert in one launch a projection
    mmvq_q2k_experts: KernelHandle,
    mmvq_q3k_experts: KernelHandle,
    /// rows (warps) a block of the expert batch, 2 / 4 / 8 (ATLAS_DS41_EXPERT_WARPS)
    experts_warps: u32,
    swiglu: KernelHandle,
    /// the swiglu and the q8_1 rows of its output in one launch (decode)
    swiglu_q8: KernelHandle,
    accumulate: KernelHandle,
    finish: KernelHandle,
    gather: KernelHandle,
    scatter_add: KernelHandle,
    sum_rows: KernelHandle,
    quant_d2s6: KernelHandle,
    quant_d4: KernelHandle,
    mmq_q2k_nc: KernelHandle,
    mmq_q2k_wc: KernelHandle,
    mmq_q3k_nc: KernelHandle,
    mmq_q3k_wc: KernelHandle,
    route_select: KernelHandle,
    slot_table_set: KernelHandle,
}

/// Where one call's time went (wall clock, host side).
#[derive(Clone, Copy, Debug, Default)]
pub struct MoeV41Timing {
    pub route_ms: f64,
    pub fetch_ms: f64,
    pub compute_ms: f64,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
}

impl MoeV41Timing {
    pub fn add(&mut self, o: &MoeV41Timing) {
        self.route_ms += o.route_ms;
        self.fetch_ms += o.fetch_ms;
        self.compute_ms += o.compute_ms;
        self.hits += o.hits;
        self.misses += o.misses;
        self.bytes_read += o.bytes_read;
    }
}

/// What `stage_m1` leaves for `compute_m1`: the distinct expert count and
/// the routing (for callers and diagnostics), with the host-span timing.
pub struct MoeV41Stage {
    pub ne: usize,
    pub weights: Vec<f32>,
    pub indices: Vec<usize>,
    pub timing: MoeV41Timing,
}

pub struct MoeV41 {
    /// The last forward's timing.
    pub last: std::cell::Cell<MoeV41Timing>,
    /// sync at the end of the forward so the step timer reads GPU time (diag only)
    timing_sync: bool,
    pub cfg: MoeV41Cfg,
    k: Kernels,
    logits: DevicePtr,
    /// the next layer's router on this layer's input (`[n_routed]` f32)
    pred_logits: DevicePtr,
    /// gathered rows of one expert group, `[m, dim]` bf16
    a_rows: DevicePtr,
    /// the group's q8_1 activations (plain rows or the MMQ layout)
    a_q8: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    h: DevicePtr,
    h_q8: DevicePtr,
    down_out: DevicePtr,
    /// `[m * topk]` i32 token rows and f32 routing weights, group-major
    rows_dev: DevicePtr,
    weight_dev: DevicePtr,
    /// The device selection's header (miss flag, picks, weight bits, plan
    /// slots) and the `[64 layers][n_routed]` slot table (-1 = not resident).
    route_hdr: DevicePtr,
    slot_table: DevicePtr,
    /// gate / up / down block pointers of the token's experts, `3 * topk`
    ptrs_dev: DevicePtr,
    sg: DevicePtr,
    su: DevicePtr,
    sh: DevicePtr,
    sd: DevicePtr,
    acc: DevicePtr,
    out: DevicePtr,
    /// The shared expert's own q8_1 scratch for its SwiGLU output (`[1,
    /// inter]`) on the side stream, so it never shares `h_q8` with the routed
    /// experts running at the same time; the token's input rows (`a_q8`) are
    /// written once on the forking stream and read by both.
    sh_q8: DevicePtr,
    /// The side stream the single-token shared expert runs on, and the two
    /// events that fence it: `ev_in` (the input is ready, recorded on the
    /// main stream) and `ev_out` (`sd` is ready, recorded on the side).
    side: u64,
    ev_in: u64,
    ev_out: u64,
}

/// `ATLAS_DS41_SHARED_SIDE` (default on; `0` = off), read once: at decode the
/// shared expert runs on a side stream while the main stream does the router
/// and the host picks the experts (the GPU is otherwise idle for that span);
/// its output is added into `acc` after the routed experts, as before, so the
/// accumulation order and the bits are unchanged.
pub fn shared_side() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !std::env::var("ATLAS_DS41_SHARED_SIDE").is_ok_and(|v| v == "0"))
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// The reference's `Gate.forward` on f32 logits: returns
/// (`weights[tokens, topk]`, `indices[tokens, topk]`) in torch's top-k order.
pub fn route_from_logits(
    logits: &[f32],
    tokens: usize,
    bias: &[f32],
    c: &MoeV41Cfg,
) -> (Vec<f32>, Vec<usize>) {
    let n = c.n_routed;
    let mut weights = Vec::with_capacity(tokens * c.topk);
    let mut indices = Vec::with_capacity(tokens * c.topk);
    for t in 0..tokens {
        let scores: Vec<f32> = (0..n)
            .map(|e| softplus(logits[t * n + e] / c.gate_temp).sqrt())
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            (scores[b] + bias[b])
                .partial_cmp(&(scores[a] + bias[a]))
                .expect("finite scores")
        });
        let picked = &order[..c.topk];
        let mut wt: Vec<f32> = picked.iter().map(|&e| scores[e]).collect();
        if c.norm_topk_prob && c.topk > 1 {
            let sum: f32 = wt.iter().sum::<f32>() + 1e-20;
            for v in &mut wt {
                *v /= sum;
            }
        }
        for v in &mut wt {
            *v *= c.route_scale;
        }
        weights.extend(wt);
        indices.extend_from_slice(picked);
    }
    (weights, indices)
}

/// A `[tokens, dim]` bf16 device buffer's bytes, for callers staging inputs.
pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| {
            let b = x.to_bits();
            let lsb = (b >> 16) & 1;
            ((b.wrapping_add(0x7FFF + lsb) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

#[cfg(test)]
#[path = "moe_v41_route_tests.rs"]
mod route_tests;
#[cfg(test)]
#[path = "moe_v41_tests.rs"]
mod tests;
