// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-commit half of the async SSM checkpoint path: the STree-style
//! in-place commit (`commit_accepted_prefix_dispatch`) and the bulk/loop
//! checkpoint-commit (`commit_verify_state_async_dispatch`). Split from
//! `async_chkpt.rs` per the 500-LoC cap; the shared helpers
//! (`bulk_copy_available`, `bulk_state_copy`, pool-provenance checks) live
//! in the parent module.

use anyhow::Result;
use atlas_core::config::LayerType;

use crate::model::types::TransformerModel;
use crate::traits::SequenceState;

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
            return Ok(());
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

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;
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
            let h_bytes = nv * vd * kd * 4;
            let conv_bytes = (nk * kd * 2 + nv * vd) * self.config.linear_conv_kernel_dim * 4;

            // Partial accept: rewind live state to the last accepted token's
            // intermediate (state after token `num_accepted-1`).
            let slot = seq.slot_idx;
            let inter_idx = num_accepted - 1;
            let h_inter = self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx);
            let conv_inter = self
                .ssm_pool
                .conv_intermediate(ssm_layer_idx, slot, inter_idx);
            self.gpu
                .copy_d2d_async(h_inter, ssm.h_state, h_bytes, stream)?;
            self.gpu
                .copy_d2d_async(conv_inter, ssm.conv_state, conv_bytes, stream)?;

            ssm_layer_idx += 1;
        }
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(in crate::model::trait_impl) fn commit_verify_state_async_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        // Bulk path: 2 launches (full accept / full reject) or 4 (partial)
        // instead of 2-4 cuMemcpyDtoDAsync PER LAYER. This is the copy burst
        // that owns the ~2.35 ms post-argmax stall measured every step in
        // nsys-moe-cap4-0931 — 43.9 h_state + 43.9 conv copies per step on the
        // secondary stream, 93 driver calls, of which only ~0.77 ms is the GPU
        // actually moving bytes.
        //
        // Every branch below mirrors the loop's source/destination exactly,
        // including the partial-accept case writing the intermediate to BOTH
        // the checkpoint and the live buffer (the live-state invariant noted
        // under the loop). Same bytes, same order of families, same stream.
        {
            let pool = &self.ssm_pool;
            let h_bytes = pool.h_bytes;
            let conv_bytes = pool.conv_bytes;
            let slot = seq.slot_idx;
            let ni = pool.num_intermediates;
            let stream = self.secondary_stream;

            let bulk_ok = self.bulk_copy_available(h_bytes) && self.bulk_copy_available(conv_bytes);
            let partial = num_accepted != 0 && num_accepted != k;

            let usable = bulk_ok
                && if partial {
                    self.ssm_intermediates_are_pool_backed(seq)
                } else {
                    self.ssm_state_is_pool_backed(seq)
                };

            if usable {
                if num_accepted == 0 {
                    // Full reject: checkpoint → live.
                    self.bulk_state_copy(
                        pool.h_checkpoint_bases_dev,
                        pool.h_state_bases_dev,
                        slot * h_bytes,
                        slot * h_bytes,
                        h_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.conv_checkpoint_bases_dev,
                        pool.conv_state_bases_dev,
                        slot * conv_bytes,
                        slot * conv_bytes,
                        conv_bytes,
                        stream,
                    )?;
                } else if num_accepted == k {
                    // Full accept: live → checkpoint (commit verify result).
                    self.bulk_state_copy(
                        pool.h_state_bases_dev,
                        pool.h_checkpoint_bases_dev,
                        slot * h_bytes,
                        slot * h_bytes,
                        h_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.conv_state_bases_dev,
                        pool.conv_checkpoint_bases_dev,
                        slot * conv_bytes,
                        slot * conv_bytes,
                        conv_bytes,
                        stream,
                    )?;
                } else {
                    // Partial accept: intermediate[num_accepted-1] → checkpoint
                    // AND → live.
                    let inter_idx = num_accepted - 1;
                    let h_inter_off = (slot * ni + inter_idx) * h_bytes;
                    let conv_inter_off = (slot * ni + inter_idx) * conv_bytes;
                    self.bulk_state_copy(
                        pool.h_intermediate_bases_dev,
                        pool.h_checkpoint_bases_dev,
                        h_inter_off,
                        slot * h_bytes,
                        h_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.conv_intermediate_bases_dev,
                        pool.conv_checkpoint_bases_dev,
                        conv_inter_off,
                        slot * conv_bytes,
                        conv_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.h_intermediate_bases_dev,
                        pool.h_state_bases_dev,
                        h_inter_off,
                        slot * h_bytes,
                        h_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.conv_intermediate_bases_dev,
                        pool.conv_state_bases_dev,
                        conv_inter_off,
                        slot * conv_bytes,
                        conv_bytes,
                        stream,
                    )?;
                }
                self.gpu.record_event(self.secondary_event, stream)?;
                return Ok(());
            }
        }

        // Live-state invariant (2026-06-10 MTP×warm stutter fix): the live
        // h_state/conv_state MUST be canonical after every commit, not just
        // the checkpoint. Leaving live dirty (holding the rejected draft)
        // was safe only when the next op was guaranteed to be another verify
        // (pre_verify copies checkpoint→live). Three real paths run a plain
        // decode() on the live buffer with no restore — spontaneous <think>
        // flipping the scheduler MTP gate, a second concurrent request, and
        // the MTP bootstrap after empty drafts (which then BAKES the dirty
        // live state into the checkpoint via start_checkpoint_async). The
        // phantom rejected token in the GDN memory garbles subsequent
        // decode (token-stutter), and with prefix caching the poisoned
        // decode-KV is immortalized in shared blocks across agentic turns.
        // Cost: one extra D2D pair per SSM layer per reject — same as the
        // pre-verify copy.
        if num_accepted == 0 {
            // Full reject: canonical state untouched; restore live from the
            // checkpoint so any non-verify successor reads canonical state.
            let stream = self.secondary_stream;
            for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
                if self.config.layer_type(i) == LayerType::LinearAttention {
                    let ssm = layer_state
                        .as_any_mut()
                        .downcast_mut::<SsmLayerState>()
                        .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
                    let (Some(h_ckpt), Some(conv_ckpt)) =
                        (ssm.h_state_checkpoint, ssm.conv_state_checkpoint)
                    else {
                        continue;
                    };
                    let nv = self.config.linear_num_value_heads;
                    let vd = self.config.linear_value_head_dim;
                    let nk = self.config.linear_num_key_heads;
                    let kd = self.config.linear_key_head_dim;
                    let h_bytes = nv * vd * kd * 4;
                    let conv_bytes =
                        (nk * kd * 2 + nv * vd) * self.config.linear_conv_kernel_dim * 4;
                    self.gpu
                        .copy_d2d_async(h_ckpt, ssm.h_state, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(conv_ckpt, ssm.conv_state, conv_bytes, stream)?;
                }
            }
            self.gpu
                .record_event(self.secondary_event, self.secondary_stream)?;
            // Ordering: the verify path syncs at entry (verify_*_step
            // sync_secondary); the non-verify successors (gate flip,
            // bootstrap) sync at THEIR entry — see scheduler/mod.rs and
            // mtp_step.rs. No wait here: a commit-side wait would serialize
            // this copy against the next draft and cost ~25% decode wall.
            return Ok(());
        }

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let Some(h_ckpt) = ssm.h_state_checkpoint else {
                    ssm_layer_idx += 1;
                    continue;
                };
                let Some(conv_ckpt) = ssm.conv_state_checkpoint else {
                    ssm_layer_idx += 1;
                    continue;
                };

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if num_accepted == k {
                    // Full accept: scratch → live (commit verify result).
                    self.gpu
                        .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(ssm.conv_state, conv_ckpt, conv_bytes, stream)?;
                } else {
                    // Partial accept: intermediate[num_accepted-1] → checkpoint
                    // AND → live. The live buffer holds state through ALL k
                    // verify tokens (including the rejected draft); restoring
                    // it here keeps live canonical for any non-verify
                    // successor (see the live-state invariant note above).
                    let slot = seq.slot_idx;
                    let inter_idx = num_accepted - 1;
                    let h_inter = self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx);
                    let conv_inter =
                        self.ssm_pool
                            .conv_intermediate(ssm_layer_idx, slot, inter_idx);
                    self.gpu.copy_d2d_async(h_inter, h_ckpt, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(conv_inter, conv_ckpt, conv_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(h_inter, ssm.h_state, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(conv_inter, ssm.conv_state, conv_bytes, stream)?;
                }

                ssm_layer_idx += 1;
            }
        }

        self.gpu.record_event(self.secondary_event, stream)?;
        // Ordering: verify_*_step calls sync_secondary at entry; the
        // non-verify successors that read the live state (MTP gate flip →
        // step_decode_only, bootstrap decode) call sync_secondary at THEIR
        // entry (scheduler/mod.rs, mtp_step.rs). A commit-side wait here
        // would serialize this 250MB copy against the next draft kernels
        // that used to overlap it (~25% decode wall, tq11 360s cap-riders).
        Ok(())
    }
}
