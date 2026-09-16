// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash MoE on streamed, still-quantized experts.
//!
//! The routed experts never leave their Q2_K / Q3_K blocks: per token the
//! router picks `topk` of 384, the [`ExpertLru`] gathers those experts' raw
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

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::expert_stream::{ExpertLru, ExpertSource};

use crate::layers::ops::{
    self, KQUANT_MODULE, Q2K_MMQ_SMEM, Q3K_MMQ_SMEM, ResidentMat, kquant_mmq_act_bytes,
    kquant_mmq_gemm, kquant_mmvq_experts_w, kquant_mmvq_w, kquant_q8_1_rows,
    kquant_q8_1_rows_bytes,
};
use crate::weight_map::DenseWeight;

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
    /// `[inter, dim]`, `[dim, inter]`, `[inter, dim]`: bf16 or the GGUF's
    /// Q2_K (w1, w3) / Q3_K (w2) blocks on the routed experts' kernels
    pub shared_w1: ResidentMat,
    pub shared_w2: ResidentMat,
    pub shared_w3: ResidentMat,
}

struct Kernels {
    gemm: KernelHandle,
    /// bf16 shared expert at m = 1 (the tiled GEMM idles 15 of 16 rows)
    gemv: KernelHandle,
    gemm_f32out: KernelHandle,
    /// router logits at m <= 8: strict-order GEMV, bit-identical to gemm_f32out
    router_gemv: KernelHandle,
    q8_rows: KernelHandle,
    mmvq_q2k: KernelHandle,
    mmvq_q3k: KernelHandle,
    /// the single-token arm: every routed expert in one launch a projection
    mmvq_q2k_experts: KernelHandle,
    mmvq_q3k_experts: KernelHandle,
    swiglu: KernelHandle,
    accumulate: KernelHandle,
    finish: KernelHandle,
    gather: KernelHandle,
    scatter_add: KernelHandle,
    quant_d2s6: KernelHandle,
    quant_d4: KernelHandle,
    mmq_q2k_nc: KernelHandle,
    mmq_q2k_wc: KernelHandle,
    mmq_q3k_nc: KernelHandle,
    mmq_q3k_wc: KernelHandle,
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
    /// gate / up / down block pointers of the token's experts, `3 * topk`
    ptrs_dev: DevicePtr,
    sg: DevicePtr,
    su: DevicePtr,
    sh: DevicePtr,
    sd: DevicePtr,
    acc: DevicePtr,
    out: DevicePtr,
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

impl MoeV41 {
    pub fn new(gpu: &dyn GpuBackend, cfg: MoeV41Cfg) -> Result<Self> {
        ensure!(
            cfg.dim % 256 == 0 && cfg.inter % 256 == 0,
            "K-quant experts need dim and inter to be multiples of 256 (got {} / {})",
            cfg.dim,
            cfg.inter
        );
        let m = cfg.max_tokens;
        let alloc = |bytes: usize| gpu.alloc(bytes.max(16));
        Ok(MoeV41 {
            last: std::cell::Cell::new(MoeV41Timing::default()),
            timing_sync: std::env::var("ATLAS_DS41_DIAG").is_ok_and(|v| v == "1"),
            k: Kernels {
                gemm: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16")?,
                gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
                gemm_f32out: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16_f32out")?,
                router_gemv: gpu.kernel(MODULE, "moe_v41_router_gemv_f32out")?,
                q8_rows: gpu.kernel(KQUANT_MODULE, "kquant_q8_1_rows_bf16")?,
                mmvq_q2k: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q2_k_w")?,
                mmvq_q3k: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q3_k_w")?,
                mmvq_q2k_experts: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q2_k_experts_w")?,
                mmvq_q3k_experts: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q3_k_experts_w")?,
                swiglu: gpu.kernel(MODULE, "moe_v41_swiglu")?,
                accumulate: gpu.kernel(MODULE, "moe_v41_accumulate")?,
                finish: gpu.kernel(MODULE, "moe_v41_finish")?,
                gather: gpu.kernel(MODULE, "moe_v41_gather_rows")?,
                scatter_add: gpu.kernel(MODULE, "moe_v41_scatter_add")?,
                quant_d2s6: gpu.kernel(KQUANT_MODULE, "atlas_q8_1_quantize_d2s6_bf16")?,
                quant_d4: gpu.kernel(KQUANT_MODULE, "atlas_q8_1_quantize_d4_bf16")?,
                mmq_q2k_nc: gpu.kernel(KQUANT_MODULE, "atlas_q2_k_mmq128_nc")?,
                mmq_q2k_wc: gpu.kernel(KQUANT_MODULE, "atlas_q2_k_mmq128_wc")?,
                mmq_q3k_nc: gpu.kernel(KQUANT_MODULE, "atlas_q3_k_mmq128_nc")?,
                mmq_q3k_wc: gpu.kernel(KQUANT_MODULE, "atlas_q3_k_mmq128_wc")?,
            },
            logits: alloc(m * cfg.n_routed * 4)?,
            a_rows: alloc(m * cfg.dim * 2)?,
            a_q8: alloc(
                kquant_mmq_act_bytes(m as u32, cfg.dim as u32)
                    .max(kquant_q8_1_rows_bytes(m as u32, cfg.dim as u32)),
            )?,
            gate_out: alloc(m * cfg.inter * 2)?,
            up_out: alloc(m * cfg.inter * 2)?,
            h: alloc(m * cfg.inter * 2)?,
            h_q8: alloc(
                kquant_mmq_act_bytes(m as u32, cfg.inter as u32)
                    .max(kquant_q8_1_rows_bytes(m as u32, cfg.inter as u32)),
            )?,
            down_out: alloc(m * cfg.dim * 2)?,
            rows_dev: alloc(m * cfg.topk * 4)?,
            weight_dev: alloc(m * cfg.topk * 4)?,
            ptrs_dev: alloc(3 * cfg.topk * 8)?,
            sg: alloc(m * cfg.inter * 2)?,
            su: alloc(m * cfg.inter * 2)?,
            sh: alloc(m * cfg.inter * 2)?,
            sd: alloc(m * cfg.dim * 2)?,
            acc: alloc(m * cfg.dim * 4)?,
            out: alloc(m * cfg.dim * 2)?,
            cfg,
        })
    }

    fn launch_n(
        &self,
        gpu: &dyn GpuBackend,
        k: KernelHandle,
        n: usize,
        stream: u64,
        f: impl FnOnce(KernelLaunch) -> KernelLaunch,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        f(KernelLaunch::new(gpu, k)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1]))
        .launch(stream)
    }

    /// Router logits on the GPU (f32-accumulated), the selection on the CPU.
    pub fn route(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        self.route_launch(gpu, w, x, m, stream)?;
        self.route_select(gpu, w, m, stream)
    }

    /// The router GEMM/GEMV into `self.logits`: device work only (the half a
    /// CUDA graph can hold).
    pub fn route_launch(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        ensure!(
            m >= 1 && m <= c.max_tokens,
            "moe_v41: {m} tokens outside 1..={}",
            c.max_tokens
        );
        // the gate at decode: one thread per logit in strict k order (the same
        // numbers as the tiled kernel, one pass over the gate rows)
        let (kernel, grid, block) = if m <= 8 {
            (
                self.k.router_gemv,
                [(c.n_routed as u32).div_ceil(64), m as u32, 1],
                [64, 1, 1],
            )
        } else {
            (
                self.k.gemm_f32out,
                [(c.n_routed as u32).div_ceil(16), (m as u32).div_ceil(16), 1],
                [16, 16, 1],
            )
        };
        KernelLaunch::new(gpu, kernel)
            .grid(grid)
            .block(block)
            .arg_ptr(x)
            .arg_ptr(w.gate_w)
            .arg_ptr(self.logits)
            .arg_u32(m as u32)
            .arg_u32(c.n_routed as u32)
            .arg_u32(c.dim as u32)
            .launch(stream)
    }

    /// The selection: drain the stream, download the logits `route_launch`
    /// wrote, pick the top-k on the CPU as the reference does.
    pub fn route_select(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        m: usize,
        stream: u64,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let c = &self.cfg;
        gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; m * c.n_routed * 4];
        gpu.copy_d2h(self.logits, &mut bytes)?;
        let logits: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        ensure!(
            w.gate_bias.len() == c.n_routed,
            "gate bias has {} entries for {} experts",
            w.gate_bias.len(),
            c.n_routed
        );
        Ok(route_from_logits(&logits, m, &w.gate_bias, c))
    }

    /// One layer's MoE for `m` tokens: routed experts from the cache, plus the
    /// shared expert. Returns the bf16 `[m, dim]` output and the routing.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<S: ExpertSource + ?Sized>(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        lru: &mut ExpertLru,
        src: &S,
        x: DevicePtr,
        m: usize,
        reader_threads: usize,
        stream: u64,
    ) -> Result<(DevicePtr, Vec<f32>, Vec<usize>)> {
        let c = &self.cfg;
        let t0 = std::time::Instant::now();
        let (weights, indices) = self.route(gpu, w, x, m, stream)?;
        let t1 = std::time::Instant::now();
        // this token batch's experts, gathered once
        lru.begin_token();
        let before = lru.stats();
        let keys: Vec<(u32, u32)> = indices.iter().map(|&e| (w.layer, e as u32)).collect();
        let slots = lru.fetch_many(src, &keys, reader_threads)?;
        let after = lru.stats();
        let t2 = std::time::Instant::now();
        let (rows_host, w_host, plan) = self.plan(&indices, &weights, m);
        gpu.copy_h2d_async(&rows_host, self.rows_dev, stream)?;
        gpu.copy_h2d_async(&w_host, self.weight_dev, stream)?;
        gpu.memset_async(self.acc, 0, m * c.dim * 4, stream)?;
        if m == 1 {
            // the single-token arm: the token is every expert's activation, so
            // quantise it once and run each projection as one launch over the
            // experts (pointer table), the routing weight folded in at the
            // SwiGLU and the expert rows summed into `acc` in plan order, the
            // same order and the same per-row math as the loop below
            let ne = self.upload_expert_table(gpu, &plan, &slots, stream)?;
            self.routed_m1(gpu, x, ne, stream)?;
        }
        for &(a0, off, r) in plan.iter().filter(|_| m > 1) {
            let slot = slots[a0];
            let rows_ptr = DevicePtr(self.rows_dev.0 + (off * 4) as u64);
            let w_ptr = DevicePtr(self.weight_dev.0 + (off * 4) as u64);
            KernelLaunch::new(gpu, self.k.gather)
                .grid([r as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_ptr(rows_ptr)
                .arg_ptr(self.a_rows)
                .arg_u32(c.dim as u32)
                .launch(stream)?;
            if r <= 8 {
                // the decode GEMV: plain q8_1 rows, one weight read shared by the rows
                kquant_q8_1_rows(
                    gpu,
                    self.k.q8_rows,
                    self.a_rows,
                    self.a_q8,
                    r as u32,
                    c.dim as u32,
                    stream,
                )?;
                kquant_mmvq_w(
                    gpu,
                    self.k.mmvq_q2k,
                    slot.gate,
                    self.a_q8,
                    self.gate_out,
                    c.inter as u32,
                    c.dim as u32,
                    r as u32,
                    stream,
                )?;
                kquant_mmvq_w(
                    gpu,
                    self.k.mmvq_q2k,
                    slot.up,
                    self.a_q8,
                    self.up_out,
                    c.inter as u32,
                    c.dim as u32,
                    r as u32,
                    stream,
                )?;
            } else {
                // the prefill MMQ: tensor cores on the raw blocks, D2S6 activations for Q2_K
                ops::quantize_act_q8_1(
                    gpu,
                    self.k.quant_d2s6,
                    self.a_rows,
                    self.a_q8,
                    r as u32,
                    c.dim as u32,
                    stream,
                )?;
                kquant_mmq_gemm(
                    gpu,
                    self.k.mmq_q2k_nc,
                    self.k.mmq_q2k_wc,
                    self.a_q8,
                    slot.gate,
                    self.gate_out,
                    r as u32,
                    c.inter as u32,
                    c.dim as u32,
                    Q2K_MMQ_SMEM,
                    stream,
                )?;
                kquant_mmq_gemm(
                    gpu,
                    self.k.mmq_q2k_nc,
                    self.k.mmq_q2k_wc,
                    self.a_q8,
                    slot.up,
                    self.up_out,
                    r as u32,
                    c.inter as u32,
                    c.dim as u32,
                    Q2K_MMQ_SMEM,
                    stream,
                )?;
            }
            self.launch_n(gpu, self.k.swiglu, r * c.inter, stream, |l| {
                l.arg_ptr(self.gate_out)
                    .arg_ptr(self.up_out)
                    .arg_ptr(w_ptr)
                    .arg_ptr(self.h)
                    .arg_u32(r as u32)
                    .arg_u32(c.inter as u32)
                    .arg_f32(c.swiglu_limit)
            })?;
            if r <= 8 {
                kquant_q8_1_rows(
                    gpu,
                    self.k.q8_rows,
                    self.h,
                    self.h_q8,
                    r as u32,
                    c.inter as u32,
                    stream,
                )?;
                kquant_mmvq_w(
                    gpu,
                    self.k.mmvq_q3k,
                    slot.down,
                    self.h_q8,
                    self.down_out,
                    c.dim as u32,
                    c.inter as u32,
                    r as u32,
                    stream,
                )?;
            } else {
                ops::quantize_act_q8_1(
                    gpu,
                    self.k.quant_d4,
                    self.h,
                    self.h_q8,
                    r as u32,
                    c.inter as u32,
                    stream,
                )?;
                kquant_mmq_gemm(
                    gpu,
                    self.k.mmq_q3k_nc,
                    self.k.mmq_q3k_wc,
                    self.h_q8,
                    slot.down,
                    self.down_out,
                    r as u32,
                    c.dim as u32,
                    c.inter as u32,
                    Q3K_MMQ_SMEM,
                    stream,
                )?;
            }
            KernelLaunch::new(gpu, self.k.scatter_add)
                .grid([r as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.acc)
                .arg_ptr(self.down_out)
                .arg_ptr(rows_ptr)
                .arg_u32(c.dim as u32)
                .launch(stream)?;
        }
        self.shared_expert(gpu, w, x, m, stream)?;
        if self.timing_sync {
            // ATLAS_DS41_DIAG=1: make compute_ms the GPU time, not the launch time
            gpu.synchronize(stream)?;
        }
        self.last.set(MoeV41Timing {
            route_ms: (t1 - t0).as_secs_f64() * 1e3,
            fetch_ms: (t2 - t1).as_secs_f64() * 1e3,
            compute_ms: t2.elapsed().as_secs_f64() * 1e3,
            hits: after.hits - before.hits,
            misses: after.misses - before.misses,
            bytes_read: after.bytes_read - before.bytes_read,
        });
        Ok((self.out, weights, indices))
    }

    /// Group the `(token, k)` assignments by expert (ascending expert id):
    /// the token rows and routing weights group-major, and the plan
    /// `(first assignment index, row offset, rows)` per group.
    #[allow(clippy::type_complexity)]
    fn plan(
        &self,
        indices: &[usize],
        weights: &[f32],
        m: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<(usize, usize, usize)>) {
        let c = &self.cfg;
        let mut groups: std::collections::BTreeMap<usize, Vec<(i32, f32, usize)>> =
            std::collections::BTreeMap::new();
        for t in 0..m {
            for kk in 0..c.topk {
                let a = t * c.topk + kk;
                groups
                    .entry(indices[a])
                    .or_default()
                    .push((t as i32, weights[a], a));
            }
        }
        let mut rows_host: Vec<u8> = Vec::with_capacity(m * c.topk * 4);
        let mut w_host: Vec<u8> = Vec::with_capacity(m * c.topk * 4);
        let mut plan: Vec<(usize, usize, usize)> = Vec::with_capacity(groups.len());
        for members in groups.values() {
            let off = rows_host.len() / 4;
            for &(t, rw, _a) in members {
                rows_host.extend_from_slice(&t.to_le_bytes());
                w_host.extend_from_slice(&rw.to_le_bytes());
            }
            plan.push((members[0].2, off, members.len()));
        }
        (rows_host, w_host, plan)
    }

    /// The single-token arm's pointer table: gate, up and down block pointers
    /// of the plan's experts, `3 * ne` entries into `ptrs_dev`. Returns `ne`.
    fn upload_expert_table(
        &self,
        gpu: &dyn GpuBackend,
        plan: &[(usize, usize, usize)],
        slots: &[spark_runtime::weights::expert_stream::ExpertSlot],
        stream: u64,
    ) -> Result<usize> {
        let ne = plan.len();
        let mut ptrs: Vec<u8> = Vec::with_capacity(3 * ne * 8);
        for which in 0..3 {
            for &(a0, _, _) in plan {
                let slot = slots[a0];
                let p = [slot.gate, slot.up, slot.down][which];
                ptrs.extend_from_slice(&p.0.to_le_bytes());
            }
        }
        gpu.copy_h2d_async(&ptrs, self.ptrs_dev, stream)?;
        Ok(ne)
    }

    /// The single-token routed experts from the pointer table: the token
    /// quantised once, each projection one launch over the `ne` experts, the
    /// routing weight folded in at the SwiGLU, the rows summed into `acc` in
    /// plan order. Device work only.
    fn routed_m1(&self, gpu: &dyn GpuBackend, x: DevicePtr, ne: usize, stream: u64) -> Result<()> {
        let c = &self.cfg;
        let table = |which: usize| DevicePtr(self.ptrs_dev.0 + (which * ne * 8) as u64);
        kquant_q8_1_rows(gpu, self.k.q8_rows, x, self.a_q8, 1, c.dim as u32, stream)?;
        for (which, out) in [(0usize, self.gate_out), (1, self.up_out)] {
            kquant_mmvq_experts_w(
                gpu,
                self.k.mmvq_q2k_experts,
                table(which),
                self.a_q8,
                out,
                c.inter as u32,
                c.dim as u32,
                1,
                ne as u32,
                0,
                stream,
            )?;
        }
        self.launch_n(gpu, self.k.swiglu, ne * c.inter, stream, |l| {
            l.arg_ptr(self.gate_out)
                .arg_ptr(self.up_out)
                .arg_ptr(self.weight_dev)
                .arg_ptr(self.h)
                .arg_u32(ne as u32)
                .arg_u32(c.inter as u32)
                .arg_f32(c.swiglu_limit)
        })?;
        kquant_q8_1_rows(
            gpu,
            self.k.q8_rows,
            self.h,
            self.h_q8,
            ne as u32,
            c.inter as u32,
            stream,
        )?;
        kquant_mmvq_experts_w(
            gpu,
            self.k.mmvq_q3k_experts,
            table(2),
            self.h_q8,
            self.down_out,
            c.dim as u32,
            c.inter as u32,
            1,
            ne as u32,
            kquant_q8_1_rows_bytes(1, c.inter as u32) as u32,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k.scatter_add)
            .grid([ne as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.acc)
            .arg_ptr(self.down_out)
            .arg_ptr(self.rows_dev)
            .arg_u32(c.dim as u32)
            .launch(stream)
    }

    /// The shared expert added into `acc`, then `acc` finished into `out`.
    /// Device work only.
    fn shared_expert(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        // shared expert: bf16 (GEMV / tiled GEMM) or the GGUF's K-quant
        // blocks on the routed experts' kernels (GEMV at m <= 8, MMQ above)
        let kq = |a: DevicePtr,
                  a_q8: DevicePtr,
                  wt: ResidentMat,
                  out: DevicePtr,
                  n: usize,
                  kdim: usize|
         -> Result<()> {
            let (mu, nu, ku) = (m as u32, n as u32, kdim as u32);
            match wt {
                ResidentMat::Bf16(p) if m == 1 => ops::dense_gemv(
                    gpu,
                    self.k.gemv,
                    a,
                    &DenseWeight { weight: p },
                    out,
                    nu,
                    ku,
                    stream,
                ),
                ResidentMat::Bf16(p) => ops::dense_gemm(
                    gpu,
                    self.k.gemm,
                    a,
                    &DenseWeight { weight: p },
                    out,
                    mu,
                    nu,
                    ku,
                    stream,
                ),
                ResidentMat::Q2K(b) if m <= 8 => {
                    kquant_q8_1_rows(gpu, self.k.q8_rows, a, a_q8, mu, ku, stream)?;
                    kquant_mmvq_w(gpu, self.k.mmvq_q2k, b, a_q8, out, nu, ku, mu, stream)
                }
                ResidentMat::Q3K(b) if m <= 8 => {
                    kquant_q8_1_rows(gpu, self.k.q8_rows, a, a_q8, mu, ku, stream)?;
                    kquant_mmvq_w(gpu, self.k.mmvq_q3k, b, a_q8, out, nu, ku, mu, stream)
                }
                ResidentMat::Q2K(b) => {
                    ops::quantize_act_q8_1(gpu, self.k.quant_d2s6, a, a_q8, mu, ku, stream)?;
                    kquant_mmq_gemm(
                        gpu,
                        self.k.mmq_q2k_nc,
                        self.k.mmq_q2k_wc,
                        a_q8,
                        b,
                        out,
                        mu,
                        nu,
                        ku,
                        Q2K_MMQ_SMEM,
                        stream,
                    )
                }
                ResidentMat::Q3K(b) => {
                    ops::quantize_act_q8_1(gpu, self.k.quant_d4, a, a_q8, mu, ku, stream)?;
                    kquant_mmq_gemm(
                        gpu,
                        self.k.mmq_q3k_nc,
                        self.k.mmq_q3k_wc,
                        a_q8,
                        b,
                        out,
                        mu,
                        nu,
                        ku,
                        Q3K_MMQ_SMEM,
                        stream,
                    )
                }
            }
        };
        kq(x, self.a_q8, w.shared_w1, self.sg, c.inter, c.dim)?;
        kq(x, self.a_q8, w.shared_w3, self.su, c.inter, c.dim)?;
        self.launch_n(gpu, self.k.swiglu, m * c.inter, stream, |l| {
            l.arg_ptr(self.sg)
                .arg_ptr(self.su)
                .arg_ptr(DevicePtr(0))
                .arg_ptr(self.sh)
                .arg_u32(m as u32)
                .arg_u32(c.inter as u32)
                .arg_f32(c.swiglu_limit)
        })?;
        kq(self.sh, self.h_q8, w.shared_w2, self.sd, c.dim, c.inter)?;
        self.launch_n(gpu, self.k.accumulate, m * c.dim, stream, |l| {
            l.arg_ptr(self.acc)
                .arg_ptr(self.sd)
                .arg_u32((m * c.dim) as u32)
        })?;
        self.launch_n(gpu, self.k.finish, m * c.dim, stream, |l| {
            l.arg_ptr(self.acc)
                .arg_ptr(self.out)
                .arg_u32((m * c.dim) as u32)
        })?;
        Ok(())
    }

    /// The HOST span of a single-token step between the router GEMV and the
    /// expert compute: the selection (drains the stream), the cache fetch,
    /// and the rows / weights / pointer-table uploads into fixed buffers.
    /// `route_launch` must have been issued on `stream` first.
    pub fn stage_m1<S: ExpertSource + ?Sized>(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        lru: &mut ExpertLru,
        src: &S,
        reader_threads: usize,
        stream: u64,
    ) -> Result<MoeV41Stage> {
        let t0 = std::time::Instant::now();
        let (weights, indices) = self.route_select(gpu, w, 1, stream)?;
        let t1 = std::time::Instant::now();
        lru.begin_token();
        let before = lru.stats();
        let keys: Vec<(u32, u32)> = indices.iter().map(|&e| (w.layer, e as u32)).collect();
        let slots = lru.fetch_many(src, &keys, reader_threads)?;
        let after = lru.stats();
        let t2 = std::time::Instant::now();
        let (rows_host, w_host, plan) = self.plan(&indices, &weights, 1);
        gpu.copy_h2d_async(&rows_host, self.rows_dev, stream)?;
        gpu.copy_h2d_async(&w_host, self.weight_dev, stream)?;
        let ne = self.upload_expert_table(gpu, &plan, &slots, stream)?;
        Ok(MoeV41Stage {
            ne,
            weights,
            indices,
            timing: MoeV41Timing {
                route_ms: (t1 - t0).as_secs_f64() * 1e3,
                fetch_ms: (t2 - t1).as_secs_f64() * 1e3,
                compute_ms: 0.0,
                hits: after.hits - before.hits,
                misses: after.misses - before.misses,
                bytes_read: after.bytes_read - before.bytes_read,
            },
        })
    }

    /// The single-token expert compute after `stage_m1`: the accumulator
    /// cleared, the routed experts from the pointer table, the shared expert,
    /// the finish into `out`. Device work only, no per-token scalar
    /// arguments (`ne` is the top-k: one token's picks are distinct experts),
    /// so a CUDA graph of it replays for any token once `stage_m1` has
    /// refilled `rows_dev` / `weight_dev` / `ptrs_dev`.
    pub fn compute_m1(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        ne: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let c = &self.cfg;
        ensure!(
            ne == c.topk,
            "moe_v41: {ne} distinct experts for one token, expected the top-{}",
            c.topk
        );
        gpu.memset_async(self.acc, 0, c.dim * 4, stream)?;
        self.routed_m1(gpu, x, ne, stream)?;
        self.shared_expert(gpu, w, x, 1, stream)?;
        Ok(self.out)
    }

    pub fn out_ptr(&self) -> DevicePtr {
        self.out
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.logits,
            self.a_rows,
            self.a_q8,
            self.rows_dev,
            self.gate_out,
            self.up_out,
            self.h,
            self.h_q8,
            self.down_out,
            self.weight_dev,
            self.sg,
            self.su,
            self.sh,
            self.sd,
            self.acc,
            self.out,
        ] {
            gpu.free(p)?;
        }
        Ok(())
    }
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
#[path = "moe_v41_tests.rs"]
mod tests;
