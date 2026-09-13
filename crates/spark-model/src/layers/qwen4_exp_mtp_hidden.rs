// SPDX-License-Identifier: AGPL-3.0-only

//! `draft_hidden`, split out of `qwen4_exp_mtp.rs` to keep it under the
//! 500-line cap.

use super::*;

impl Qwen4ExpMtpHead {
    pub fn draft_hidden(
        &self,
        last_token: u32,
        target_streams: DevicePtr,
        position: usize,
        state: &mut Qwen4ExpMtpState,
        h_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let hc = ctx.config.hc_mult.max(1) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row = h as usize * 2;

        // The body reads its highway through the private arena. Snapshot the
        // input before the combiner writes that arena: chained drafts read
        // from the same arena, so input and output can alias. Highway elements
        // are FP32, even though collapsed hidden rows are BF16.
        let hc_bytes = hc as usize * h as usize * 4;

        // ── DIAGNOSTIC (ATLAS_QWEN4EXP_MTP_DIFF=1) ──
        // The bisect proved the BODY forward dirties state the target still
        // needs, but not WHICH buffer. Rather than keep guessing, fingerprint
        // the shared buffers either side of the call and name the ones that
        // changed. Taken BEFORE the combiner runs, so the baseline is the
        // TARGET's state — an earlier version sampled it after the combiner had
        // already written hc_streams, which made hc_streams a false positive.
        let diff = std::env::var("ATLAS_QWEN4EXP_MTP_DIFF").as_deref() == Ok("1");
        let probes: Vec<(&str, DevicePtr, usize)> = if diff {
            ctx.gpu.synchronize(stream).ok();
            vec![
                ("hc_streams", ctx.buffers.hc_streams(), hc_bytes.min(4096)),
                ("hc_post", ctx.buffers.hc_post(), 256),
                ("hc_comb", ctx.buffers.hc_comb(), 256),
                ("hc_lowrank_scratch", ctx.buffers.hc_lowrank_scratch(), 4096),
                ("hidden_states", ctx.buffers.hidden_states(), row),
                ("residual", ctx.buffers.residual(), row),
                ("norm_output", ctx.buffers.norm_output(), row),
                (
                    "scratch@target_meta",
                    ctx.buffers.scratch().offset(32768),
                    4096,
                ),
            ]
        } else {
            Vec::new()
        };
        let before: Vec<u64> = probes
            .iter()
            .map(|(_, p, n)| crate::speculative::hidden_fingerprint(ctx.gpu, *p, *n / 2))
            .collect();

        // The draft's highway lives in the DRAFT's arena, not the target's.
        // Nothing below writes a buffer the target owns.
        let body_streams = self.arena.hc_streams();
        self.draft_combine(last_token, target_streams, body_streams, ctx, stream)?;

        // No save/restore of target buffers: the draft runs in its own arena.
        // ── 3. Body decode against the module's OWN cache ──
        let mut kv_cache = self.kv_cache.lock().expect("mtp kv cache poisoned");
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        // Same layout + same shared packer the DeepSeek-V4 head uses, at a
        // DISTINCT scratch offset so this never clobbers the target metadata.
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let meta_buf = super::super::mtp_meta::pack_mtp_attn_meta(
            position as u32,
            global_slot,
            (state.seq_len + 1) as i32,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;
        let meta = crate::layer::AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: state.block_table.len() as u32,
            num_seqs: 1,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr::NULL,
        };

        let mtp_ctx = ForwardContext {
            // This private arena contains one draft row, regardless of the
            // accepted row selected from the target's verification highway.
            hc_row_offset: 0,
            attn_metadata: Some(meta),
            // The draft body must not issue an EP all-reduce: it is rank-0 only
            // and `ensure_loadable` refuses ep_world_size > 1 outright.
            comm: None,
            // Full-width drafter geometry, not the target's per-rank counts.
            // See `Qwen4ExpMtpHead::cfg`.
            config: &self.cfg,
            // Host-built metadata + H2D uploads are illegal under capture.
            graph_capture: false,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
            // ★ the draft's OWN arena — the whole point of this design.
            buffers: &self.arena,
            ..*ctx
        };

        let mut disk_block_ids: Vec<u32> = Vec::new();
        let mut disk_last_offloaded: Vec<u32> = vec![0u32; 1];
        // Result captured, NOT `?`: an early return here would leave the
        // draft's streams in the target's persistent highway.
        let body_res = self.module.body.decode(
            self.buf.body_scratch,
            self.buf.residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        );
        drop(kv_cache);
        if body_res.is_err() {
            ctx.gpu
                .copy_d2d_async(self.buf.streams, body_streams, hc_bytes, stream)?;
            body_res?;
        }

        // ── 4. mHC head: collapse the module's streams → h_out ──
        // qwen4_exp's head is LOW-RANK, which is exactly the arm DeepSeek-V4's
        // MTP asserts against; call the low-rank collapse directly.
        let head = self
            .module
            .hc_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp MTP: module has no hc_head"))?;
        let lowrank = head.lowrank.as_ref().ok_or_else(|| {
            anyhow::anyhow!("qwen4_exp MTP: hc_head is not low-rank; this model's is")
        })?;
        // Collapse the DRAFT's highway (which the body just wrote through
        // `body_streams`), not the saved target one.
        let collapse = ops::hc_head_lowrank(
            ctx.gpu,
            self.hc_head_k,
            body_streams,
            lowrank,
            h_out,
            self.buf.head_scratch,
            1,
            h,
            hc,
            eps,
            stream,
        );

        // Keep the private arena's output highway for the next autoregressive
        // draft. The input snapshot belongs to the combiner; restoring it here
        // would make every later draft consume the preceding draft's INPUT.

        if diff {
            ctx.gpu.synchronize(stream).ok();
            let changed: Vec<&str> = probes
                .iter()
                .zip(before.iter())
                .filter(|((_, p, n), b)| {
                    crate::speculative::hidden_fingerprint(ctx.gpu, *p, *n / 2) != **b
                })
                .map(|((name, _, _), _)| *name)
                .collect();
            tracing::info!(
                "qwen4_exp MTP diff: buffers changed across the draft body = {:?} \
                 (target hc_streams should NOT appear — the draft arena is private; anything \
                 else that appears is shared state the target still needs)",
                changed
            );
        }
        collapse?;

        state.seq_len += 1;
        Ok(())
    }
}
