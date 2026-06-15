// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash decode path. Reuses low-rank Q projection (wq_a→norm→wq_b)
//! from the MLA path, but uses direct KV projection (no absorption) and
//! grouped low-rank O projection (wo_a→wo_b).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Run the DeepSeek-V4-Flash decode chain. Returns the O-projection
    /// output (`ctx.buffers.qkv_output()`).
    pub(super) fn attention_forward_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &super::attention_forward_mla::DecodeMlaArgs,
    ) -> Result<DevicePtr> {
        let super::attention_forward_mla::DecodeMlaArgs {
            normed,
            q_out,
            k_out,
            v_out,
            q_dim,
            h,
            nq,
            hd,
            eps,
            bs,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("attention_forward_v4 called without MLA config");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash decode requires pre-uploaded metadata");

        let q_lora = mla.q_lora_rank as u32;
        let mla_rope = mla.rope as u32;
        let o_lora = mla.o_lora_rank as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let profile = ctx.profile;
        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let _t = std::time::Instant::now();
                    let _r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    V4 {}: {:.0}µs", $label, _t.elapsed().as_micros());
                    _r
                } else {
                    $body
                }
            }};
        }

        // ── Step 1: Q latent → norm → expand ──
        let q_latent = ctx.buffers.ssm_ba();
        prof!("wq_a", {
            if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed,
                    wqa_nvfp4,
                    q_latent,
                    q_lora,
                    h,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wq_a,
                    q_latent,
                    q_lora,
                    h,
                    stream,
                )
            }
        })?;
        prof!("q_norm", {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                q_latent,
                &mla.q_a_norm,
                q_latent,
                1,
                q_lora,
                eps,
                stream,
            )
        })?;
        prof!("wq_b", {
            if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    q_latent,
                    wqb_nvfp4,
                    q_out,
                    q_dim,
                    q_lora,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    q_latent,
                    &mla.wq_b,
                    q_out,
                    q_dim,
                    q_lora,
                    stream,
                )
            }
        })?;
        if self.attn_layer_idx == 0 {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_out,
                q_dim as usize,
                stream,
                "V4-decode L0 Q after proj",
            );
        }

        // ── Step 2: Direct KV projection ──
        let kv_dim = nkv * hd;
        prof!("wkv", {
            if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed,
                    wkva_nvfp4,
                    k_out,
                    kv_dim,
                    h,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wkv_a,
                    k_out,
                    kv_dim,
                    h,
                    stream,
                )
            }
        })?;
        // K=V for V4-Flash direct KV projection
        ctx.gpu
            .copy_d2d_async(k_out, v_out, (kv_dim as usize) * 2, stream)?;
        if self.attn_layer_idx == 0 {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                "V4-decode L0 K after proj",
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                v_out,
                kv_dim as usize,
                stream,
                "V4-decode L0 V after copy",
            );
        }

        // ── Step 3: RoPE for Q and K ──
        // V4-Flash: rope dims are at offset `nope` per head (matching MLA layout),
        // not at the beginning. Extract → RoPE → writeback.
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        let k_rope_tmp = q_latent; // reuse after wq_b is done
        prof!("rope_extract", {
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                q_out,
                q_rope_tmp,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )
        })?;
        prof!("k_rope_extract", {
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                k_out,
                k_rope_tmp,
                1,
                1,
                hd,
                mla.nope as u32,
                mla_rope,
                hd,
                stream,
            )
        })?;
        prof!("rope", {
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_k,
                q_rope_tmp,
                k_rope_tmp,
                meta.positions,
                1,
                nq,
                1,
                mla_rope,
                mla_rope,
                mla.yarn_inv_freq,
                ctx.config.rope_theta as f32,
                stream,
            )
        })?;
        prof!("rope_writeback", {
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                q_rope_tmp,
                q_out,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )
        })?;
        prof!("k_rope_writeback", {
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                k_rope_tmp,
                k_out,
                1,
                1,
                hd,
                mla.nope as u32,
                mla_rope,
                hd,
                stream,
            )
        })?;
        if self.attn_layer_idx == 0 {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                "V4-decode L0 K after RoPE",
            );
            // Diagnostic: rope region of K (offset nope=448)
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset((mla.nope * 2) as usize),
                (kv_dim - mla.nope as u32) as usize,
                stream,
                "V4-decode L0 K rope after RoPE",
            );
            // Diagnostic: rope region of Q head 0 (offset nope=448)
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_out.offset((mla.nope * 2) as usize),
                (hd - mla.nope as u32) as usize,
                stream,
                "V4-decode L0 Q rope after RoPE",
            );
        }

        // ── Step 4: Write K/V to paged cache ──
        let kv_stride = kv_dim;
        prof!("kv_write", {
            self.write_kv_cache(
                ctx.gpu,
                k_out,
                v_out,
                kv_cache,
                meta.slot,
                1,
                1,
                hd,
                bs as u32,
                kv_stride,
                kv_stride,
                stream,
                ctx.graph_capture,
            )
        })?;

        // ── Step 5: Paged decode attention ──
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        prof!("paged_attn", {
            self.run_paged_decode(
                ctx.gpu,
                q_out,
                kv_cache,
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                nq * hd,
                ctx.buffers.splitk_workspace(),
                stream,
            )
        })?;
        if self.attn_layer_idx == 0 {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out,
                (nq * hd) as usize,
                stream,
                "V4-decode L0 attn_out",
            );
        }

        // ── Step 6: Grouped low-rank O projection (wo_a → wo_b) ──
        let o_latent = ctx.buffers.norm_output();
        let o_out = ctx.buffers.qkv_output();
        prof!("wo_a", {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                attn_out,
                &mla.wo_a,
                o_latent,
                o_lora,
                nq * hd,
                stream,
            )
        })?;
        prof!("wo_b", {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                o_latent,
                &mla.wo_b,
                o_out,
                h,
                o_lora,
                stream,
            )
        })?;
        if self.attn_layer_idx == 0 {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out,
                h as usize,
                stream,
                "V4-decode L0 o_out",
            );
        }

        Ok(o_out)
    }
}
