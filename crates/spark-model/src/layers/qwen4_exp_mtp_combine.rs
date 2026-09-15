// SPDX-License-Identifier: AGPL-3.0-only

//! Combining the draft streams and running the draft bodies, split out of
//! `qwen4_exp_mtp.rs` to keep it under the 500-line cap.

use super::*;

impl Qwen4ExpMtpHead {
    /// One draft step. Writes the draft's final hidden state (post-mHC-head,
    /// pre-LM-head) into `h_out`; the caller applies its own final norm and LM
    /// head. `target_streams` is the target's four-stream highway for the
    /// position that just produced `last_token`.
    #[allow(clippy::too_many_arguments)]
    /// Steps 1-2 of a draft: embedding branch, hidden branch, and the
    /// combine that writes the body's INPUT highway to `streams_out`.
    ///
    /// Snapshots `target_streams` into private scratch first, so `streams_out`
    /// may alias it (chained drafts read the arena row they then overwrite).
    /// Shared by the per-sequence path (row 0) and the batched path (row i).
    pub fn draft_combine(
        &self,
        last_token: u32,
        target_streams: DevicePtr,
        streams_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let hc = ctx.config.hc_mult.max(1) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row = h as usize * 2;
        let hc_bytes = hc as usize * h as usize * 4;
        ctx.gpu
            .copy_d2d_async(target_streams, self.buf.streams, hc_bytes, stream)?;

        // ── 1. Embedding branch ──
        let src = self.embed_tokens.weight.offset(last_token as usize * row);
        ctx.gpu.copy_d2d_async(src, self.buf.embed, row, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            self.buf.embed,
            &self.module.pre_fc_norm_embedding,
            self.buf.normed_embed,
            1,
            h,
            eps,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            self.buf.normed_embed,
            &self.module.fc_embedding,
            self.buf.embed_proj,
            h,
            h,
            stream,
        )?;

        // ── 2. Hidden branch — DTYPE-CORRECT end to end ──
        // The highway is FP32; the projections are BF16 GEMVs. So:
        //   a) `hc_pre_stage_bf16` reads the FP32 streams and writes the GROUPED
        //      norm as BF16. It is the model's own kernel for exactly this
        //      (per-stream RMS, offset-from-1 scale, `[hc*H]` weight) — which is
        //      also independent confirmation that `pre_fc_norm_hidden [10240]`
        //      normalizes the four-stream highway.
        //   b) `fc_hidden` is applied PER STREAM as a BF16 GEMV.
        //   c) `qhc_mtp_combine_streams` writes the FP32 highway from those
        //      per-stream BF16 rows plus the broadcast embedding projection.
        // An earlier version ran BF16 ops directly over the FP32 buffer — silent
        // garbage, and the reason the first accept measurements were meaningless.
        ops::hc_pre_stage_bf16_norm(
            ctx.gpu,
            self.hc_stage_k,
            self.buf.streams,
            self.module.pre_fc_norm_hidden.weight,
            self.buf.normed_streams,
            1,
            h,
            hc,
            eps,
            stream,
        )?;
        for i in 0..hc as usize {
            let off = i * row;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                self.buf.normed_streams.offset(off),
                &self.module.fc_hidden,
                self.buf.per_stream.offset(off),
                h,
                h,
                stream,
            )?;
        }
        ops::qhc_mtp_combine_streams(
            ctx.gpu,
            self.combine_k,
            self.buf.per_stream,
            self.buf.embed_proj,
            streams_out,
            h,
            hc,
            stream,
        )?;

        Ok(())
    }

    /// Collapse arena highway row `i` into `h_out` (one row) — step 4 of a
    /// draft, addressed by row for the batched path.
    pub fn draft_collapse_row(
        &self,
        i: usize,
        h_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let hc = ctx.config.hc_mult.max(1) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let head = self
            .module
            .hc_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp MTP: module has no hc_head"))?;
        let lowrank = head.lowrank.as_ref().ok_or_else(|| {
            anyhow::anyhow!("qwen4_exp MTP: hc_head is not low-rank; this model's is")
        })?;
        ops::hc_head_lowrank(
            ctx.gpu,
            self.hc_head_k,
            self.arena_streams_row(i, hc as usize, h as usize),
            lowrank,
            h_out,
            self.buf.head_scratch,
            1,
            h,
            hc,
            eps,
            stream,
        )
    }

    /// Step 3 of a draft for `n` sequences at once: the module body over arena
    /// rows 0..n via `decode_multi_seq`, each row against that sequence's own
    /// private KV and draft state.
    ///
    /// The n-sequence attention metadata is drafter-local, in the drafter
    /// arena's scratch, laid out by the arena's derived `decode_meta()` —
    /// `positions u32[R] @0`, `slots i64[R] @8R`, `seq_lens i32[R] @16R`,
    /// `block table i32[R x max_blocks] @24R` — exactly the layout the target's
    /// batched decode uploads, so the attention kernels read what they
    /// expect. `num_seqs = n`, no padding rows.
    ///
    /// Advances every state's `seq_len` by one, as `draft_hidden` does.
    pub fn draft_bodies_batched(
        &self,
        states: &mut [&mut Qwen4ExpMtpState],
        positions: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let n = states.len();
        anyhow::ensure!(
            n >= 1 && n <= self.batch_cap && positions.len() == n,
            "draft_bodies_batched: n={n} (cap {}), {} positions",
            self.batch_cap,
            positions.len()
        );
        let mut kv_cache = self.kv_cache.lock().expect("mtp kv cache poisoned");
        let bs = kv_cache.block_size();
        for st in states.iter_mut() {
            let blocks_needed = (st.seq_len / bs) + 1;
            while st.block_table.len() < blocks_needed {
                st.block_table.push(kv_cache.alloc_block()?);
            }
        }

        // ── n-sequence attention metadata, in the DRAFTER arena's scratch ──
        let lay = self.arena.decode_meta();
        anyhow::ensure!(
            n <= lay.rows(),
            "draft_bodies_batched: n={n} exceeds the arena's {}-row metadata layout",
            lay.rows()
        );
        let max_blocks = states
            .iter()
            .map(|st| st.block_table.len())
            .max()
            .unwrap_or(1)
            .max(1);
        const META_OFF: usize = 32768;
        let need = META_OFF + lay.meta_bytes(max_blocks);
        anyhow::ensure!(
            need <= self.arena.scratch_bytes(),
            "draft_bodies_batched: metadata needs {need} B of arena scratch, have {}",
            self.arena.scratch_bytes()
        );
        let rows = lay.rows();
        let mut positions_u32: Vec<u32> = vec![0; rows];
        let mut slots: Vec<i64> = vec![0; rows];
        let mut seq_lens_i32: Vec<i32> = vec![1; rows];
        let mut bt_flat: Vec<i32> = vec![0; rows * max_blocks];
        for (i, st) in states.iter().enumerate() {
            let pos = st.seq_len;
            positions_u32[i] = positions[i] as u32;
            let block_idx = st.block_table[pos / bs];
            slots[i] = (block_idx as i64) * (bs as i64) + ((pos % bs) as i64);
            seq_lens_i32[i] = (pos + 1) as i32;
            for (j, &b) in st.block_table.iter().take(max_blocks).enumerate() {
                bt_flat[i * max_blocks + j] = b as i32;
            }
        }
        let base = self.arena.scratch().offset(META_OFF);
        let to_bytes_u32 = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let to_bytes_i64 = |v: &[i64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let to_bytes_i32 = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        ctx.gpu
            .copy_h2d_async(&to_bytes_u32(&positions_u32), base, stream)?;
        ctx.gpu
            .copy_h2d_async(&to_bytes_i64(&slots), base.offset(lay.slots_off()), stream)?;
        ctx.gpu.copy_h2d_async(
            &to_bytes_i32(&seq_lens_i32),
            base.offset(lay.seq_lens_off()),
            stream,
        )?;
        ctx.gpu.copy_h2d_async(
            &to_bytes_i32(&bt_flat),
            base.offset(lay.block_table_off()),
            stream,
        )?;
        let meta = crate::layer::AttnMetadataDev {
            positions: base,
            positions_h: base,
            positions_w: base,
            slot: base.offset(lay.slots_off()),
            seq_len: base.offset(lay.seq_lens_off()),
            block_table: base.offset(lay.block_table_off()),
            max_blocks_per_seq: max_blocks as u32,
            num_seqs: n as u32,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr::NULL,
        };
        let mtp_ctx = ForwardContext {
            hc_row_offset: 0,
            attn_metadata: Some(meta),
            // Rank-0 only, no EP collective — same as the per-sequence body.
            comm: None,
            // NOT the target's config: see `Qwen4ExpMtpHead::cfg`. Under TP the
            // target's head counts are per-rank; the drafter is replicated and
            // must read its own full-width geometry.
            config: &self.cfg,
            graph_capture: false,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
            buffers: &self.arena,
            ..*ctx
        };

        let seq_lens: Vec<usize> = states.iter().map(|st| st.seq_len).collect();
        let block_tables: Vec<Vec<u32>> = states.iter().map(|st| st.block_table.clone()).collect();
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            states.iter_mut().map(|st| st.body_state.as_mut()).collect();
        self.module.body.decode_multi_seq(
            self.arena.hidden_states(),
            self.arena.residual(),
            n,
            &mut refs,
            &mut kv_cache,
            &seq_lens,
            &block_tables,
            &mtp_ctx,
            stream,
        )?;
        drop(kv_cache);
        for st in states.iter_mut() {
            st.seq_len += 1;
        }
        Ok(())
    }
}
