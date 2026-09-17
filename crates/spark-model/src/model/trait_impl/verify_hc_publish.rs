// SPDX-License-Identifier: AGPL-3.0-only

//! Publishing and stashing the per-row verify state, split out of
//! `verify_hc.rs` to keep that file under the 500-line cap.

use super::*;

impl TransformerModel {
    pub(super) fn stash_verify_aux(&self, slot: usize, stash: VerifyAuxRows) -> Result<()> {
        // Per SLOT — see the span stash above for why a single slot is wrong
        // once more than one sequence is verified in a sweep.
        self.pending_verify_aux
            .lock()
            .map_err(|_| anyhow::anyhow!("verify aux stash poisoned"))?
            .insert(slot, stash);
        Ok(())
    }

    /// Copy this sequence's live GDN `h_state`/`conv_state` into the verify
    /// intermediate slot for row `t`, so `commit_accepted_prefix` has a real
    /// snapshot to rewind to.
    ///
    /// The batched verify kernels populate these slots as a side effect of
    /// their own scan; the mHC prefill body has no such side effect, so this
    /// publishes them explicitly. The widths are the SSOT ones
    /// `commit_accepted_prefix_dispatch` copies BACK —
    /// `ssm_pool.h_stored_bytes` for h (it tracks `--ssm-h-dtype`) and the
    /// conv blob computed from config — so the two must not drift apart.
    pub(super) fn publish_verify_row_state(
        &self,
        seq: &mut SequenceState,
        t: usize,
        stream: u64,
    ) -> Result<()> {
        use avarok_core::config::LayerType;

        use crate::layer::SsmLayerState;

        let nv = self.config.linear_num_value_heads;
        let vd = self.config.linear_value_head_dim;
        let nk = self.config.linear_num_key_heads;
        let kd = self.config.linear_key_head_dim;
        let h_bytes = self.ssm_pool.h_stored_bytes;
        let conv_bytes = (nk * kd * 2 + nv * vd) * self.config.linear_conv_kernel_dim * 4;

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            // No MTP pools (`has_mtp == false`) means no speculative verify
            // reaches `commit_accepted_prefix` either — nothing to publish.
            if ssm.h_state_intermediates.is_empty() && ssm.conv_state_intermediates.is_empty() {
                return Ok(());
            }
            let h_dst = *ssm.h_state_intermediates.get(t).ok_or_else(|| {
                anyhow::anyhow!(
                    "mHC verify: row {t} has no h intermediate slot ({} sized) — the \
                     verify width exceeds what ssm_reserve sized, and a partial accept \
                     would rewind onto never-written pool memory",
                    ssm.h_state_intermediates.len()
                )
            })?;
            let c_dst = *ssm.conv_state_intermediates.get(t).ok_or_else(|| {
                anyhow::anyhow!(
                    "mHC verify: row {t} has no conv intermediate slot ({} sized)",
                    ssm.conv_state_intermediates.len()
                )
            })?;
            self.gpu
                .copy_d2d_async(ssm.h_state, h_dst, h_bytes, stream)?;
            self.gpu
                .copy_d2d_async(ssm.conv_state, c_dst, conv_bytes, stream)?;
        }
        Ok(())
    }

    /// Land the auxiliary carries on a commit of `num_accepted` out of `k`
    /// verify rows, leaving `seq_len`/`tokens` to the caller.
    ///
    /// PLE is restored from the per-row snapshot at
    /// `verify_aux_restore_row(num_accepted, k)`; QSA is realigned to the
    /// absolute position `base_pos + num_accepted`. The KV written for the
    /// discarded positions is left alone deliberately: it is past `seq_len` and
    /// the next step overwrites it.
    ///
    /// The scheduler's branch already owns the token/`seq_len` rewind
    /// (`seq_len -= rejected` + `commit_accepted_prefix`); what it cannot do is
    /// walk back the carries a mini-prefill advanced. Splitting the aux half out
    /// lets the scheduler call exactly the missing piece instead of rewinding
    /// `seq_len` a second time.
    pub(in crate::model) fn restore_verify_aux_at(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let stash = self
            .pending_verify_aux
            .lock()
            .map_err(|_| anyhow::anyhow!("verify aux stash poisoned"))?
            .remove(&seq.slot_idx);
        let Some(stash) = stash else {
            anyhow::bail!(
                "commit_verify_aux({num_accepted}/{k}) with no stashed aux snapshot — \
                 decode_verify_hc must run first, or the carries cannot be rewound"
            );
        };
        anyhow::ensure!(
            stash.k == k,
            "commit_verify_aux quotes k={k} but the stash was taken for k={} — the \
             scheduler and the verify disagree about the draft width, and restoring \
             across that mismatch would land the carries on the wrong token",
            stash.k
        );
        anyhow::ensure!(
            num_accepted >= 1 && num_accepted <= k,
            "commit_verify_aux: num_accepted={num_accepted} outside 1..={k}"
        );
        // The scheduler owns the token/`seq_len` rewind and runs it BEFORE the
        // commit; this pins that the two agree about where the sequence landed.
        // If they ever disagree the QSA alignment below would move the marks to
        // a position the sequence is not at, which the next decode reports as
        // "decode at pos N but M tokens ingested" — a confusing symptom a long
        // way from its cause.
        anyhow::ensure!(
            stash.base_pos + num_accepted == seq.seq_len,
            "commit_verify_aux({num_accepted}/{k}): verify started at {} so the \
             sequence should be at {} after committing {num_accepted} rows, but \
             seq_len is {} — the scheduler's rewind and this commit disagree",
            stash.base_pos,
            stash.base_pos + num_accepted,
            seq.seq_len
        );

        // ── The snapshot half: PLE's conv + n-gram history ──
        // Skipped on a FULL accept: no row was discarded, so the live carry is
        // already the committed one (and `hc_publish_rows` never snapshots the
        // last row for exactly that reason).
        if let Some(idx) = verify_aux_restore_row(num_accepted, k) {
            let blobs = stash.rows.get(idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "mHC verify: commit of {num_accepted}/{k} rows needs aux snapshot \
                     {idx}, but the verify stashed only {} — the publish range and \
                     `commit_rewind_index` have drifted apart",
                    stash.rows.len()
                )
            })?;
            self.apply_aux_states(seq, blobs, stream)?;
        }

        // ── The mark half: QSA `ingested`/`pooled` ──
        // ALWAYS, including full accept, and by ABSOLUTE position. The verify
        // advanced the marks by `k`; the sequence kept `num_accepted`. Aligning
        // to `base_pos + num_accepted` is a no-op when they already agree, so
        // this is safe to run on every branch and leaves nothing for the next
        // step's `align_aux` to discover — which matters because a Marconi
        // checkpoint can be taken between the two, and would otherwise
        // serialize marks that are ahead of the sequence.
        self.align_verify_aux_states(seq, stash.base_pos + num_accepted, stream)
    }

    /// Land ALL THREE per-row carries on a partial accept of `num_accepted`
    /// rows out of the K-row batched mHC verify.
    ///
    /// The SSM carry is not here — `commit_accepted_prefix` already rewinds
    /// `h_state`/`conv_state` from the pool intermediates the conv+GDN kernels
    /// wrote. What this adds is the other two, which that function has never
    /// touched:
    ///
    /// * PLE's rolling conv + history window, from the per-row snapshot
    ///   `decode_batched_inner_hc` recorded (`commit_verify_row`); and
    /// * QSA's `ingested`/`pooled` marks, rewound to the ABSOLUTE position
    ///   `base + num_accepted` (`align_aux`) — contiguous marks need no blob.
    ///
    /// Leaving either advanced over the rejected rows is the measured
    /// degeneration: PLE then hashes a window shifted by the discarded
    /// drafts, and QSA trips its own `pos == ingested` assertion or indexes
    /// past the committed prefix.
    ///
    /// No-op unless the BATCHED arm ran: the per-row reference path advances
    /// its carries one row per pass and snapshots between rows 0 and 1
    /// instead.
    pub(in crate::model) fn commit_verify_aux_rows(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        stream: u64,
    ) -> Result<()> {
        let span = self
            .pending_verify_span
            .lock()
            .map_err(|_| anyhow::anyhow!("verify span stash poisoned"))?
            .remove(&seq.slot_idx);
        let Some((base, k)) = span else {
            return Ok(());
        };
        anyhow::ensure!(
            (1..=k).contains(&num_accepted) && seq.seq_len == base + num_accepted,
            "batched mHC commit: {num_accepted}/{k} rows from {base}, but seq_len={}",
            seq.seq_len
        );
        // Full accept consumes the span too; the live carries already agree.
        if num_accepted == k {
            return Ok(());
        }
        let row = commit_rewind_index(num_accepted);
        let to_pos = base + num_accepted;
        for (i, layer) in self.layers.iter().enumerate() {
            let st = seq.layer_states[i].as_mut();
            layer.commit_verify_row(st, row, self.gpu.as_ref(), stream)?;
            layer.align_aux(st, to_pos, self.gpu.as_ref(), stream)?;
        }
        Ok(())
    }
}
