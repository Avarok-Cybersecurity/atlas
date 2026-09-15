// SPDX-License-Identifier: AGPL-3.0-only

//! Committing an accepted prefix, split out of `async_chkpt.rs` to keep it
//! under the 500-line cap.

use super::*;

impl TransformerModel {
    /// STree-style in-place verify commit (item #2): the verify kernel
    /// writes directly onto the canonical `h_state`/`conv_state`, so the
    /// surviving prefix is already live and "commit" reduces to a single
    /// index-select on a partial accept (and nothing on a full accept).
    ///
    /// - `num_accepted == k` (full accept): the kernel's final `h_state`
    ///   is the committed state → no-op.
    /// - `0 < num_accepted < k` (partial accept): copy
    ///   `h_state_intermediates[num_accepted - 1]` (state after the last
    ///   accepted token) → `h_state` (+ conv intermediate).
    ///
    /// All verify paths (K=2, K=3, K=4, DFlash) run the kernel directly
    /// on the canonical `h_state` (no `pre_verify_copy_async` scratch-seed),
    /// so on a full accept the live state is already committed and on a
    /// partial accept the single index-select below leaves `h_state`
    /// canonical for every successor (bootstrap decode, gate-flip decode,
    /// concurrent request). No `*_checkpoint` write is needed — the next
    /// `start_checkpoint_async` syncs h_state → checkpoint at prefill time.
    ///
    /// Runs on `secondary_stream`; pair with `sync_secondary`.
    pub(in crate::model::trait_impl) fn commit_accepted_prefix_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        // Width invariant. Together with the `num_accepted == 0` guard below
        // this pins the reachable intermediate index to exactly [0, k-2],
        // which is the invariant the fused K=2/3/4 verify paths rely on when
        // they skip writing `conv_state_intermediates[k-1]`
        // (qwen3_ssm/trait_decode_batched_conv_gdn.rs). Enforcing it here
        // turns that from a global argument about callers into a locally
        // checked precondition: if a caller ever passes a width that would
        // reach index k-1, it errors here instead of silently reading a slot
        // the kernel no longer writes.
        //
        // `num_accepted > k` is nonsense (more tokens committed than
        // verified) and is the shape a bonus-token off-by-one would take —
        // e.g. DFlash passing `gamma` instead of `gamma + 1` as `k` while
        // passing `num_accepted + 1`. Today `verify_dflash_step.rs` passes
        // `k_verify = drafts.len() + 1` against `total_accepted =
        // num_accepted + 1`, so the two agree; this catches the day they
        // stop agreeing.
        if num_accepted > k {
            anyhow::bail!(
                "commit_accepted_prefix: num_accepted ({num_accepted}) > k ({k}) — more \
                 tokens committed than were verified. Check that the caller's `k` is the \
                 VERIFY WIDTH (drafts + 1), not the draft count."
            );
        }

        // Full accept: the verify kernel's final h_state/conv_state is
        // already the canonical committed state — nothing to do.
        if num_accepted == k {
            return self.commit_verify_aux_rows(seq, num_accepted, self.secondary_stream);
        }

        // `num_accepted == 0` has no representable rewind target here: the
        // per-token intermediates are indexed `num_accepted - 1`, and this is
        // `usize` arithmetic in a release build (overflow-checks off), so a 0
        // would wrap to `usize::MAX` and hand `h_intermediate()` an
        // out-of-range index — a wild device pointer straight into
        // `copy_d2d_async`. Every scheduler caller passes >= 1 today (position
        // 0 of a verify batch is accepted by construction; DFlash adds the
        // bonus token via `num_accepted + 1`), but that is a caller
        // convention, not an invariant this function can see. Fail fast so a
        // future caller change surfaces as an error instead of silent memory
        // corruption. A genuine "nothing accepted" rewind belongs in
        // `rollback_ssm_states`, which restores the pre-verify checkpoint.
        if num_accepted == 0 {
            anyhow::bail!(
                "commit_accepted_prefix: num_accepted == 0 (k={k}) has no intermediate to \
                 rewind to — position 0 of a verify batch is accepted by construction. \
                 Use rollback_ssm_states() for a full-reject rewind to the pre-verify \
                 checkpoint."
            );
        }

        // Replay: the per-token intermediates are SHARED scratch, so there is
        // no per-sequence blob to rewind to. Restore the pre-verify checkpoint
        // and re-run the accepted rows instead — the same reconstruction the
        // full-reject path uses, with `num_accepted` rows to replay rather
        // than none. `num_accepted` here counts the committed prefix
        // (`na + 1` from the verdict), and the rows cached are the verify
        // window's, so the replay advances by exactly the committed tokens.
        if self.ssm_pool.intermediates_shared {
            return self.replay_commit_accepted_prefix(seq, num_accepted);
        }

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;
        // Two plans, not one interleaved loop: h blobs and conv blobs have
        // different widths, and a pitched 2-D copy carries ONE width. Both
        // are built in ascending layer order — the same order, and the same
        // (src, dst, bytes) triples, the per-layer loop issued.
        let mut h_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut conv_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

            let nv = self.config.linear_num_value_heads;
            let vd = self.config.linear_value_head_dim;
            let nk = self.config.linear_num_key_heads;
            let kd = self.config.linear_key_head_dim;
            // Pool h STORAGE width (SSOT: ssm_reserve::ssm_h_stored_bytes).
            let h_bytes = self.ssm_pool.h_stored_bytes;
            let conv_bytes = (nk * kd * 2 + nv * vd) * self.config.linear_conv_kernel_dim * 4;

            // Partial accept: rewind live state to the last accepted token's
            // intermediate (state after token `num_accepted-1`).
            let slot = seq.slot_idx;
            let inter_idx = commit_rewind_index(num_accepted);
            h_plan.push(StateCopy {
                src: self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx),
                dst: ssm.h_state,
                bytes: h_bytes,
            });
            conv_plan.push(StateCopy {
                src: self
                    .ssm_pool
                    .conv_intermediate(ssm_layer_idx, slot, inter_idx),
                dst: ssm.conv_state,
                bytes: conv_bytes,
            });

            ssm_layer_idx += 1;
        }
        run_ssm_state_copies(self.gpu.as_ref(), &h_plan, &conv_plan, stream)?;
        // The SSM carry is now committed. The OTHER TWO per-row carries — PLE's
        // rolling conv/history window and QSA's ingested/pooled marks — are
        // still sitting `k - num_accepted` rows ahead, which is the measured
        // degeneration class. No-op unless the K-row BATCHED mHC verify ran
        // (it is what records the per-row PLE snapshots and the verify span).
        self.commit_verify_aux_rows(seq, num_accepted, stream)?;
        // The next decode must wait for PLE's restore as well as the SSM
        // copies. Recording earlier leaves that auxiliary copy unfenced.
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }
}
