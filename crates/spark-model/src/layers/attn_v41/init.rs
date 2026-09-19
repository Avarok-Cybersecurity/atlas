// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! [`AttnV41`] construction and teardown: kernel lookup, the RoPE tables and
//! the per-call workspaces (`new`), and their release (`free`). Split from
//! `attn_v41.rs` (500-LoC cap).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{AttnV41, AttnV41Cfg, GEMM_MODULE, Kernels, MODULE, upload_f32};
use crate::layers::deepseek_v41_ref::attn::freqs_cis;
use crate::layers::deepseek_v41_ref::compress::yarn_freqs_cis;
use crate::layers::ops::{KQUANT_MODULE, kquant_mmq_act_bytes, kquant_q8_1_rows_bytes};

impl AttnV41 {
    pub fn new(gpu: &dyn GpuBackend, cfg: AttnV41Cfg) -> Result<Self> {
        ensure!(
            cfg.rope_dim.is_multiple_of(2) && cfg.rope_dim / 2 <= 256,
            "rope_dim {} not supported",
            cfg.rope_dim
        );
        ensure!(
            (cfg.head_dim * cfg.n_heads).is_multiple_of(cfg.groups),
            "n_heads * hd not divisible by groups"
        );
        ensure!(
            cfg.head_dim.is_multiple_of(32) && cfg.index_hd.is_multiple_of(32),
            "head dims must be multiples of the 32-block"
        );
        let k = Kernels {
            gemm: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            q8_rows: gpu.kernel(KQUANT_MODULE, "kquant_q8_1_rows_bf16")?,
            mmvq_q2k_w: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q2_k_w")?,
            mmvq_q2k_groups_w: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q2_k_groups_w")?,
            quant_d2s6: gpu.kernel(KQUANT_MODULE, "atlas_q8_1_quantize_d2s6_bf16")?,
            mmq_q2k_nc: gpu.kernel(KQUANT_MODULE, "atlas_q2_k_mmq128_nc")?,
            mmq_q2k_wc: gpu.kernel(KQUANT_MODULE, "atlas_q2_k_mmq128_wc")?,
            rmsnorm_bf16: gpu.kernel(MODULE, "attn_v41_rmsnorm_bf16")?,
            rmsnorm_f32: gpu.kernel(MODULE, "attn_v41_rmsnorm_f32")?,
            rope: gpu.kernel(MODULE, "attn_v41_rope")?,
            act_quant: gpu.kernel(MODULE, "attn_v41_act_quant_fp8")?,
            fp4_quant: gpu.kernel(MODULE, "attn_v41_fp4_quant")?,
            gemm_f32: gpu.kernel(MODULE, "attn_v41_gemm_f32")?,
            gemv_f32_staged: gpu.kernel(MODULE, "attn_v41_gemv_f32_staged")?,
            pool: gpu.kernel(MODULE, "attn_v41_pool")?,
            index_score: gpu.kernel(MODULE, "attn_v41_index_score")?,
            sparse_attn: gpu.kernel(MODULE, "attn_v41_sparse_attn")?,
            slice_cols: gpu.kernel(MODULE, "attn_v41_slice_cols")?,
            scatter_cols: gpu.kernel(MODULE, "attn_v41_scatter_cols")?,
            scale_bf16: gpu.kernel(MODULE, "attn_v41_scale_bf16")?,
            ring_put: gpu.kernel(MODULE, "attn_v41_ring_put")?,
        };
        let plain = freqs_cis(cfg.rope_dim, cfg.max_seq, cfg.rope_theta);
        let yarn = yarn_freqs_cis(
            cfg.rope_dim,
            cfg.max_seq,
            cfg.orig_seq,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        );
        let flat = |v: &[(f32, f32)]| -> Vec<f32> { v.iter().flat_map(|&(c, s)| [c, s]).collect() };
        let fc_plain = upload_f32(gpu, &flat(&plain))?;
        let fc_yarn = upload_f32(gpu, &flat(&yarn))?;
        let m = cfg.max_tokens;
        let (nh, hd, nhi, ihd) = (cfg.n_heads, cfg.head_dim, cfg.index_heads, cfg.index_hd);
        let max_width = cfg.max_seq;
        let max_topk = cfg.window + cfg.index_topk;
        let alloc = |bytes: usize| gpu.alloc(bytes.max(16));
        // `n_heads * head_dim`: the grouped `wo_a` decode path quantises the
        // whole rotated attention output row once (primitives::wo_a_grouped).
        let kmax = cfg
            .dim
            .max(cfg.q_rank)
            .max(cfg.gw())
            .max(cfg.groups * cfg.o_rank)
            .max(nh * hd);
        Ok(AttnV41 {
            a_q8: alloc(
                kquant_q8_1_rows_bytes(8, kmax as u32)
                    .max(kquant_mmq_act_bytes(m as u32, kmax as u32)),
            )?,
            qr_raw: alloc(m * cfg.q_rank * 2)?,
            qr: alloc(m * cfg.q_rank * 2)?,
            q: alloc(m * nh * hd * 2)?,
            kv_raw: alloc(m * hd * 2)?,
            kv: alloc(m * hd * 2)?,
            o: alloc(m * nh * hd * 2)?,
            o_rot: alloc(m * nh * hd * 2)?,
            og: alloc(m * cfg.groups * cfg.o_rank * 2)?,
            slice_in: alloc(m * cfg.gw() * 2)?,
            slice_out: alloc(m * cfg.o_rank * 2)?,
            out: alloc(m * cfg.dim * 2)?,
            pos: alloc(m * 4)?,
            head_pos: alloc(m * nh * 4)?,
            idx_pos: alloc(m * nhi * 4)?,
            grp_pos: alloc(m * 4)?,
            idx_dev: alloc(m * max_topk * 4)?,
            ckv: alloc(m * hd * 4)?,
            cscore: alloc(m * hd * 4)?,
            pooled: alloc(m * hd * 4)?,
            latent_raw: alloc(m * hd * 2)?,
            latent: alloc(m * hd * 2)?,
            ik_raw: alloc(m * ihd * 2)?,
            ik: alloc(m * ihd * 2)?,
            iq: alloc(m * nhi * ihd * 2)?,
            iw_raw: alloc(m * nhi * 2)?,
            iw: alloc(m * nhi * 2)?,
            score: alloc(m * max_width * 4)?,
            decode_pos: None,
            decode_idx: None,
            cfg,
            k,
            fc_plain,
            fc_yarn,
        })
    }

    /// The layer output buffer, bf16 `[max_tokens, dim]`: a pointer a captured
    /// decode step bakes.
    pub fn out_ptr(&self) -> DevicePtr {
        self.out
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.fc_plain,
            self.fc_yarn,
            self.qr_raw,
            self.qr,
            self.q,
            self.kv_raw,
            self.kv,
            self.o,
            self.o_rot,
            self.og,
            self.slice_in,
            self.slice_out,
            self.out,
            self.pos,
            self.head_pos,
            self.idx_pos,
            self.grp_pos,
            self.idx_dev,
            self.ckv,
            self.cscore,
            self.pooled,
            self.latent_raw,
            self.latent,
            self.ik_raw,
            self.ik,
            self.iq,
            self.iw_raw,
            self.iw,
            self.score,
        ] {
            gpu.free(p)?;
        }
        Ok(())
    }
}
