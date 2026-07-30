// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::layers::qwen3_attention::types::MlaWeights;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_v4_flash_n2_output(
        &self,
        c: &MultiSeqCtx<'_>,
        mla: &MlaWeights,
        attn_out: DevicePtr,
        o_out: DevicePtr,
        nq: u32,
        hd: u32,
        bf16: usize,
        stream: u64,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        let h = c.h as u32;
        let o_lora = mla.o_lora_rank as u32;
        let q_dim = nq * hd;
        let o_groups = c.fwd.config.o_groups.max(1) as u32;
        let group_in = q_dim / o_groups;
        let latent_dim = o_groups * o_lora;
        let o_latent = c.fwd.buffers.o_latent();
        let pack_in = c.fwd.buffers.splitk_workspace();
        let pack_out = pack_in.offset(2 * group_in as usize * bf16);

        for g in 0..o_groups {
            let row_bytes = group_in as usize * bf16;
            let src0 = attn_out.offset((g * group_in) as usize * bf16);
            let src1 = attn_out.offset((q_dim as usize + (g * group_in) as usize) * bf16);
            gpu.copy_d2d_async(src0, pack_in, row_bytes, stream)?;
            gpu.copy_d2d_async(src1, pack_in.offset(row_bytes), row_bytes, stream)?;

            if let Some(ref woa) = mla.wo_a_fp8 {
                let w_off = g as usize * o_lora as usize * group_in as usize;
                let s_off = g as usize * (o_lora as usize / 128) * (group_in as usize / 128) * 4;
                ops::w8a16_gemv_batch2(
                    gpu,
                    self.w8a16_gemv_batch4_k,
                    pack_in,
                    woa.weight.offset(w_off),
                    woa.row_scale.offset(s_off),
                    pack_out,
                    o_lora,
                    group_in,
                    stream,
                )?;
                let out_row = o_lora as usize * bf16;
                let dst0 = o_latent.offset((g * o_lora) as usize * bf16);
                let dst1 = o_latent.offset((latent_dim as usize + (g * o_lora) as usize) * bf16);
                gpu.copy_d2d_async(pack_out, dst0, out_row, stream)?;
                gpu.copy_d2d_async(pack_out.offset(out_row), dst1, out_row, stream)?;
            } else {
                for token in 0..2usize {
                    let in_g =
                        attn_out.offset((token * q_dim as usize + (g * group_in) as usize) * bf16);
                    let out_g = o_latent
                        .offset((token * latent_dim as usize + (g * o_lora) as usize) * bf16);
                    let weight = crate::weight_map::DenseWeight {
                        weight: mla
                            .wo_a
                            .weight
                            .offset(g as usize * o_lora as usize * group_in as usize * bf16),
                    };
                    ops::dense_gemv(
                        gpu,
                        self.dense_gemv_k,
                        in_g,
                        &weight,
                        out_g,
                        o_lora,
                        group_in,
                        stream,
                    )?;
                }
            }
        }

        if let Some(ref wob) = mla.wo_b_fp8 {
            ops::w8a16_gemv_batch2(
                gpu,
                self.w8a16_gemv_batch4_k,
                o_latent,
                wob.weight,
                wob.row_scale,
                o_out,
                h,
                latent_dim,
                stream,
            )?;
        } else {
            for token in 0..2usize {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    o_latent.offset(token * latent_dim as usize * bf16),
                    &mla.wo_b,
                    o_out.offset(token * c.h * bf16),
                    h,
                    latent_dim,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
