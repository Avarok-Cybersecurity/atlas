// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{
    DeepseekV4MtpHead, DeepseekV4MtpProposerState, MTP_META_OFFSET, v4_mtp_k1_state_enabled,
};
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;

impl DeepseekV4MtpHead {
    /// Batch-build model-native V4 MTP prompt rows from the target's live
    /// pre-hc_head FP32 streams. The target arena is consumed from front to
    /// back: after one microbatch is converted, the MTP body may overwrite
    /// those rows while later target rows remain untouched.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_stream_rows_inner(
        &self,
        next_tokens: &[u32],
        target_streams: DevicePtr,
        first_position: usize,
        state: &mut DeepseekV4MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        if std::env::var("ATLAS_V4_MTP_PROMPT_PREFILL").ok().as_deref() != Some("1")
            || next_tokens.is_empty()
        {
            return Ok(0);
        }
        anyhow::ensure!(
            v4_mtp_k1_state_enabled(),
            "V4 prompt MTP prefill requires ATLAS_V4_MTP_K1_STATE=1"
        );
        anyhow::ensure!(
            std::env::var("ATLAS_V4_MTP_USE_MHC").ok().as_deref() != Some("0"),
            "V4 prompt MTP prefill requires model-native multi-stream mHC"
        );
        anyhow::ensure!(
            state.seq_len == first_position,
            "V4 prompt MTP rows must be cold-contiguous: state rows={} first_position={first_position}",
            state.seq_len
        );

        let h = ctx.config.hidden_size;
        let hc_mult = ctx.config.hc_mult;
        anyhow::ensure!(hc_mult > 0, "V4 prompt MTP prefill requires hc_mult > 0");
        let rows_per_pass = (ctx.buffers.max_batch_tokens() / hc_mult).min(512);
        anyhow::ensure!(
            rows_per_pass > 0,
            "V4 prompt MTP prefill arena cannot hold one {hc_mult}-stream row"
        );
        let eps = ctx.config.rms_norm_eps as f32;
        let stream_row_bytes = hc_mult * h * 4;
        let t0 = std::time::Instant::now();
        let mut done = 0usize;

        while done < next_tokens.len() {
            let c = (next_tokens.len() - done).min(rows_per_pass);
            let position = first_position + done;
            let shifted = &next_tokens[done..done + c];

            // h branch: F32 target streams -> BF16 -> hnorm -> h_proj.
            // residual/norm_output each have max_batch_tokens rows, so c is
            // capped such that c*hc_mult fits both buffers.
            let streams_bf16 = ctx.buffers.residual();
            let normed_streams = ctx.buffers.norm_output();
            let h_branch = streams_bf16;
            ops::mtp_hc_f32_to_bf16_legacy(
                ctx.gpu,
                self.mtp_hc_f32_to_bf16_k,
                target_streams.offset(done * stream_row_bytes),
                streams_bf16,
                (c * hc_mult * h) as u32,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                streams_bf16,
                &self.module.hnorm,
                normed_streams,
                (c * hc_mult) as u32,
                h as u32,
                eps,
                stream,
            )?;
            ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_pipelined_k,
                normed_streams,
                &self.module.h_proj,
                h_branch,
                (c * hc_mult) as u32,
                h as u32,
                h as u32,
                stream,
            )?;

            // e branch: shifted target token -> embedding -> enorm -> e_proj.
            let token_ids = ctx.buffers.token_ids();
            let token_bytes =
                unsafe { std::slice::from_raw_parts(shifted.as_ptr() as *const u8, c * 4) };
            ctx.gpu.copy_h2d_async(token_bytes, token_ids, stream)?;
            let embed = ctx.buffers.hidden_states();
            ops::batched_embed(
                ctx.gpu,
                self.batched_embed_k,
                token_ids,
                self.embed_tokens.weight,
                embed,
                c as u32,
                h as u32,
                stream,
            )?;
            let normed_embed = ctx.buffers.moe_output();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                embed,
                &self.module.enorm,
                normed_embed,
                c as u32,
                h as u32,
                eps,
                stream,
            )?;
            let e_branch = ctx.buffers.attn_output();
            ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_pipelined_k,
                normed_embed,
                &self.module.e_proj,
                e_branch,
                c as u32,
                h as u32,
                h as u32,
                stream,
            )?;
            ops::mtp_hproj_broadcast_add_batched(
                ctx.gpu,
                self.mtp_hproj_broadcast_add_k,
                h_branch,
                e_branch,
                ctx.buffers.hc_streams(),
                c as u32,
                hc_mult as u32,
                h as u32,
                position as u32,
                stream,
            )?;

            // Grow the private MTP cache and stage one-sequence paged metadata.
            let mut kv_cache = self.kv_cache.lock();
            let bs = kv_cache.block_size();
            let end_row = state.seq_len + c;
            let blocks_needed = end_row.div_ceil(bs);
            while state.block_table.len() < blocks_needed {
                state.block_table.push(kv_cache.alloc_block()?);
            }

            let positions: Vec<u32> = (position..position + c).map(|p| p as u32).collect();
            let slots: Vec<i64> = (state.seq_len..end_row)
                .map(|row| (state.block_table[row / bs] as i64) * (bs as i64) + (row % bs) as i64)
                .collect();
            let block_table: Vec<i32> = state
                .block_table
                .iter()
                .map(|&block| block as i32)
                .collect();
            let pos_bytes = c * 4;
            let slot_off = (pos_bytes + 7) & !7;
            let seq_len_off = slot_off + c * 8;
            let block_table_off = (seq_len_off + 4 + 3) & !3;
            let meta_bytes = block_table_off + block_table.len() * 4;
            let scratch_bytes = ctx.buffers.scratch_bytes();
            let meta_off = if MTP_META_OFFSET + meta_bytes <= scratch_bytes {
                MTP_META_OFFSET
            } else if meta_bytes + 64 <= scratch_bytes {
                scratch_bytes - meta_bytes
            } else {
                anyhow::bail!(
                    "V4 prompt MTP metadata ({meta_bytes} B) exceeds scratch ({scratch_bytes} B)"
                );
            };
            let meta_base = ctx.buffers.scratch().offset(meta_off);
            let mut meta = vec![0u8; meta_bytes];
            let positions_bytes =
                unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, pos_bytes) };
            meta[..pos_bytes].copy_from_slice(positions_bytes);
            let slots_bytes =
                unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, c * 8) };
            meta[slot_off..slot_off + c * 8].copy_from_slice(slots_bytes);
            meta[seq_len_off..seq_len_off + 4].copy_from_slice(&(end_row as i32).to_le_bytes());
            let block_table_bytes = unsafe {
                std::slice::from_raw_parts(block_table.as_ptr() as *const u8, block_table.len() * 4)
            };
            meta[block_table_off..].copy_from_slice(block_table_bytes);
            ctx.gpu.copy_h2d_async(&meta, meta_base, stream)?;

            let mtp_meta = AttnMetadataDev {
                positions: meta_base,
                positions_h: meta_base,
                positions_w: meta_base,
                slot: meta_base.offset(slot_off),
                seq_len: meta_base.offset(seq_len_off),
                block_table: meta_base.offset(block_table_off),
                max_blocks_per_seq: block_table.len() as u32,
                num_seqs: 1,
                seq_slot: DevicePtr::NULL,
            };
            let mtp_ctx = ForwardContext {
                buffers: ctx.buffers,
                gpu: ctx.gpu,
                config: ctx.config,
                attn_metadata: Some(mtp_meta),
                profile: ctx.profile,
                comm: None,
                graph_capture: false,
                gdn_exact_replay: false,
                token_ids: Some(token_ids),
                routed_lora_layers: None,
                midchunk_capture: None,
            };
            let mut disk_block_ids = Vec::new();
            let mut disk_last_offloaded = vec![0u32; ctx.config.num_hidden_layers + 1];
            self.module.body.prefill(
                ctx.buffers.hidden_states(),
                ctx.buffers.residual(),
                c,
                state.body_state.as_mut(),
                &mut kv_cache,
                state.seq_len,
                &mut state.block_table,
                &mut disk_block_ids,
                &mut disk_last_offloaded,
                0,
                &mtp_ctx,
                stream,
            )?;
            ctx.gpu.synchronize(stream).map_err(|e| {
                anyhow::anyhow!(
                    "V4 prompt MTP body prefill failed at position {position} ({c} rows): {e}"
                )
            })?;
            drop(kv_cache);

            state.seq_len = end_row;
            state.last_pair_key = Some(position + c - 1);
            done += c;
        }

        tracing::info!(
            "V4 MTP prompt prefill: {} rows at positions {}..{} in {:.1} ms",
            done,
            first_position,
            first_position + done,
            t0.elapsed().as_secs_f64() * 1e3,
        );
        Ok(done)
    }
}
