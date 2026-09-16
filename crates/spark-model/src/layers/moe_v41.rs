// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash MoE on streamed, still-quantized experts.
//!
//! The routed experts never leave their Q2_K / Q3_K blocks: per token the
//! router picks `topk` of 384, the [`ExpertLru`] gathers those experts' raw
//! slices into its device-visible slots (misses read from the SSD), and each
//! expert runs as three K-quant GEMVs on the raw blocks (`kquant_mmvq_q2_k` for
//! gate and up, `kquant_mmvq_q3_k` for down) with the activation quantised to
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
    self, KQUANT_MODULE, kquant_mmvq, kquant_q8_1_rows, kquant_q8_1_rows_bytes,
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
    /// bf16 `[inter, dim]`, `[dim, inter]`, `[inter, dim]`
    pub shared_w1: DevicePtr,
    pub shared_w2: DevicePtr,
    pub shared_w3: DevicePtr,
}

struct Kernels {
    gemm: KernelHandle,
    gemm_f32out: KernelHandle,
    q8_rows: KernelHandle,
    mmvq_q2k: KernelHandle,
    mmvq_q3k: KernelHandle,
    swiglu: KernelHandle,
    accumulate: KernelHandle,
    finish: KernelHandle,
}

pub struct MoeV41 {
    pub cfg: MoeV41Cfg,
    k: Kernels,
    logits: DevicePtr,
    x_q8: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    h: DevicePtr,
    h_q8: DevicePtr,
    down_out: DevicePtr,
    weight_dev: DevicePtr,
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
            k: Kernels {
                gemm: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16")?,
                gemm_f32out: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16_f32out")?,
                q8_rows: gpu.kernel(KQUANT_MODULE, "kquant_q8_1_rows_bf16")?,
                mmvq_q2k: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q2_k")?,
                mmvq_q3k: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q3_k")?,
                swiglu: gpu.kernel(MODULE, "moe_v41_swiglu")?,
                accumulate: gpu.kernel(MODULE, "moe_v41_accumulate")?,
                finish: gpu.kernel(MODULE, "moe_v41_finish")?,
            },
            logits: alloc(m * cfg.n_routed * 4)?,
            x_q8: alloc(kquant_q8_1_rows_bytes(m as u32, cfg.dim as u32))?,
            gate_out: alloc(cfg.inter * 2)?,
            up_out: alloc(cfg.inter * 2)?,
            h: alloc(cfg.inter * 2)?,
            h_q8: alloc(kquant_q8_1_rows_bytes(1, cfg.inter as u32))?,
            down_out: alloc(cfg.dim * 2)?,
            weight_dev: alloc(4)?,
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
        let c = &self.cfg;
        ensure!(
            m >= 1 && m <= c.max_tokens,
            "moe_v41: {m} tokens outside 1..={}",
            c.max_tokens
        );
        KernelLaunch::new(gpu, self.k.gemm_f32out)
            .grid([(c.n_routed as u32).div_ceil(16), (m as u32).div_ceil(16), 1])
            .block([16, 16, 1])
            .arg_ptr(x)
            .arg_ptr(w.gate_w)
            .arg_ptr(self.logits)
            .arg_u32(m as u32)
            .arg_u32(c.n_routed as u32)
            .arg_u32(c.dim as u32)
            .launch(stream)?;
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
        let (weights, indices) = self.route(gpu, w, x, m, stream)?;
        // this token batch's experts, gathered once
        lru.begin_token();
        let keys: Vec<(u32, u32)> = indices.iter().map(|&e| (w.layer, e as u32)).collect();
        let slots = lru.fetch_many(src, &keys, reader_threads)?;
        // activations to q8_1 once
        kquant_q8_1_rows(
            gpu,
            self.k.q8_rows,
            x,
            self.x_q8,
            m as u32,
            c.dim as u32,
            stream,
        )?;
        gpu.memset_async(self.acc, 0, m * c.dim * 4, stream)?;
        let x_row_bytes = kquant_q8_1_rows_bytes(1, c.dim as u32);
        for t in 0..m {
            let x_q8_t = DevicePtr(self.x_q8.0 + (t * x_row_bytes) as u64);
            for kk in 0..c.topk {
                let slot = slots[t * c.topk + kk];
                let rw = weights[t * c.topk + kk];
                kquant_mmvq(
                    gpu,
                    self.k.mmvq_q2k,
                    slot.gate,
                    x_q8_t,
                    self.gate_out,
                    c.inter as u32,
                    c.dim as u32,
                    1,
                    stream,
                )?;
                kquant_mmvq(
                    gpu,
                    self.k.mmvq_q2k,
                    slot.up,
                    x_q8_t,
                    self.up_out,
                    c.inter as u32,
                    c.dim as u32,
                    1,
                    stream,
                )?;
                gpu.copy_h2d_async(&rw.to_le_bytes(), self.weight_dev, stream)?;
                self.launch_n(gpu, self.k.swiglu, c.inter, stream, |l| {
                    l.arg_ptr(self.gate_out)
                        .arg_ptr(self.up_out)
                        .arg_ptr(self.weight_dev)
                        .arg_ptr(self.h)
                        .arg_u32(1)
                        .arg_u32(c.inter as u32)
                        .arg_f32(c.swiglu_limit)
                })?;
                kquant_q8_1_rows(
                    gpu,
                    self.k.q8_rows,
                    self.h,
                    self.h_q8,
                    1,
                    c.inter as u32,
                    stream,
                )?;
                kquant_mmvq(
                    gpu,
                    self.k.mmvq_q3k,
                    slot.down,
                    self.h_q8,
                    self.down_out,
                    c.dim as u32,
                    c.inter as u32,
                    1,
                    stream,
                )?;
                let acc_t = DevicePtr(self.acc.0 + (t * c.dim * 4) as u64);
                self.launch_n(gpu, self.k.accumulate, c.dim, stream, |l| {
                    l.arg_ptr(acc_t)
                        .arg_ptr(self.down_out)
                        .arg_u32(c.dim as u32)
                })?;
                // the scratch buffers are reused per (token, expert): keep the launches ordered
                gpu.synchronize(stream)?;
            }
        }
        // shared expert, dense bf16
        let dense = |a: DevicePtr, wt: DevicePtr, out: DevicePtr, n: usize, kdim: usize| {
            ops::dense_gemm(
                gpu,
                self.k.gemm,
                a,
                &DenseWeight { weight: wt },
                out,
                m as u32,
                n as u32,
                kdim as u32,
                stream,
            )
        };
        dense(x, w.shared_w1, self.sg, c.inter, c.dim)?;
        dense(x, w.shared_w3, self.su, c.inter, c.dim)?;
        self.launch_n(gpu, self.k.swiglu, m * c.inter, stream, |l| {
            l.arg_ptr(self.sg)
                .arg_ptr(self.su)
                .arg_ptr(DevicePtr(0))
                .arg_ptr(self.sh)
                .arg_u32(m as u32)
                .arg_u32(c.inter as u32)
                .arg_f32(c.swiglu_limit)
        })?;
        dense(self.sh, w.shared_w2, self.sd, c.dim, c.inter)?;
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
        gpu.synchronize(stream)?;
        Ok((self.out, weights, indices))
    }

    pub fn out_ptr(&self) -> DevicePtr {
        self.out
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.logits,
            self.x_q8,
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
