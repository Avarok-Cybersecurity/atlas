// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! `MoeV41` construction and teardown: the kernel handles and the
//! `max_tokens` workspace.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{GEMM_MODULE, Kernels, MODULE, MoeV41, MoeV41Cfg, MoeV41Timing};
use crate::layers::ops::{KQUANT_MODULE, kquant_mmq_act_bytes, kquant_q8_1_rows_bytes};

impl MoeV41 {
    pub fn new(gpu: &dyn GpuBackend, cfg: MoeV41Cfg) -> Result<Self> {
        ensure!(
            cfg.dim.is_multiple_of(256) && cfg.inter.is_multiple_of(256),
            "K-quant experts need dim and inter to be multiples of 256 (got {} / {})",
            cfg.dim,
            cfg.inter
        );
        let m = cfg.max_tokens;
        // The single-token arm (`routed_m1`) runs every selected expert as a
        // row of the per-expert buffers, so they hold at least `topk` rows
        // however small `max_tokens` is; the token-indexed buffers keep `m`.
        let me = m.max(cfg.topk);
        let alloc = |bytes: usize| gpu.alloc(bytes.max(16));
        // Warps a block of the six-expert GEMV batch: 8 (the `_w8` entries)
        // by default, the fastest of 2 / 4 / 8 on the 09-19 probe (18.9 vs
        // 18.2 vs 18.0 tok/s hot); 2 / 4 select `_w2` / `_w`, same bytes out.
        let experts_warps: u32 = std::env::var("ATLAS_DS41_EXPERT_WARPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|w| matches!(w, 2 | 4 | 8))
            .unwrap_or(8);
        let sfx = match experts_warps {
            2 => "2",
            8 => "8",
            _ => "",
        };
        Ok(MoeV41 {
            last: std::cell::Cell::new(MoeV41Timing::default()),
            timing_sync: std::env::var("ATLAS_DS41_DIAG").is_ok_and(|v| v == "1"),
            k: Kernels {
                gemm: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16")?,
                gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
                gemm_f32out: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16_f32out")?,
                router_gemv: gpu.kernel(MODULE, "moe_v41_router_gemv_f32out")?,
                router_gemv_staged: gpu.kernel(MODULE, "moe_v41_router_gemv_f32out_staged")?,
                q8_rows: gpu.kernel(KQUANT_MODULE, "kquant_q8_1_rows_bf16")?,
                mmvq_q2k: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q2_k_w")?,
                mmvq_q3k: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q3_k_w")?,
                mmvq_q2k_experts: gpu
                    .kernel(KQUANT_MODULE, &format!("kquant_mmvq_q2_k_experts_w{sfx}"))?,
                mmvq_q3k_experts: gpu
                    .kernel(KQUANT_MODULE, &format!("kquant_mmvq_q3_k_experts_w{sfx}"))?,
                experts_warps,
                swiglu: gpu.kernel(MODULE, "moe_v41_swiglu")?,
                accumulate: gpu.kernel(MODULE, "moe_v41_accumulate")?,
                finish: gpu.kernel(MODULE, "moe_v41_finish")?,
                gather: gpu.kernel(MODULE, "moe_v41_gather_rows")?,
                scatter_add: gpu.kernel(MODULE, "moe_v41_scatter_add")?,
                sum_rows: gpu.kernel(MODULE, "moe_v41_sum_rows")?,
                quant_d2s6: gpu.kernel(KQUANT_MODULE, "atlas_q8_1_quantize_d2s6_bf16")?,
                quant_d4: gpu.kernel(KQUANT_MODULE, "atlas_q8_1_quantize_d4_bf16")?,
                mmq_q2k_nc: gpu.kernel(KQUANT_MODULE, "atlas_q2_k_mmq128_nc")?,
                mmq_q2k_wc: gpu.kernel(KQUANT_MODULE, "atlas_q2_k_mmq128_wc")?,
                mmq_q3k_nc: gpu.kernel(KQUANT_MODULE, "atlas_q3_k_mmq128_nc")?,
                mmq_q3k_wc: gpu.kernel(KQUANT_MODULE, "atlas_q3_k_mmq128_wc")?,
            },
            logits: alloc(m * cfg.n_routed * 4)?,
            pred_logits: alloc(cfg.n_routed * 4)?,
            a_rows: alloc(m * cfg.dim * 2)?,
            a_q8: alloc(
                kquant_mmq_act_bytes(m as u32, cfg.dim as u32)
                    .max(kquant_q8_1_rows_bytes(m as u32, cfg.dim as u32)),
            )?,
            // [2 * me, inter]: the single-token path runs gate and up as one
            // 2 * ne expert batch and reads up at gate_out + ne * inter.
            gate_out: alloc(2 * me * cfg.inter * 2)?,
            up_out: alloc(me * cfg.inter * 2)?,
            h: alloc(me * cfg.inter * 2)?,
            h_q8: alloc(
                kquant_mmq_act_bytes(me as u32, cfg.inter as u32)
                    .max(kquant_q8_1_rows_bytes(me as u32, cfg.inter as u32)),
            )?,
            down_out: alloc(me * cfg.dim * 2)?,
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

    pub fn out_ptr(&self) -> DevicePtr {
        self.out
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.logits,
            self.pred_logits,
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
