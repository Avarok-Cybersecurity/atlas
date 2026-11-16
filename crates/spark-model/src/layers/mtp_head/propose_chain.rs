// SPDX-License-Identifier: AGPL-3.0-only
// Design after Cinference's single-graph MTP round (https://github.com/satellitedown/cinference, commit 383e5db, src/models/qwen3_5/program/speculative/mtp.cpp:94-201; Apache-2.0, derived from NInfer by Neroued and contributors); no code copied.

//! Responsibility: the device-resident MTP propose chain
//! (`mtp_propose_graph`): one host readback per propose round instead of one
//! per draft position.
//!
//! The pre-sprint chain feeds each draft back through the host: per-sequence
//! `propose` reads every intermediate draft id with a blocking 4-byte copy so
//! the next position can copy that token's embedding row, and the batched
//! propose reads all n ids after every position for the same reason. Under
//! this row the next position gathers its embeddings on the device from the
//! previous position's ids (`batched_embed`, the same bytes the host-indexed
//! copy moves), the ids of every position collect in a device buffer, and the
//! round reads them back once. The arithmetic of every draft is unchanged, so
//! drafts, and therefore outputs, are bit-identical to the per-draft chain;
//! only host waits disappear.
//!
//! Eligible: grammar-free rows, no deferred embedding target, chains of 2 to
//! [`MAX_CHAIN_DRAFTS`] drafts, and none of the observational per-draft
//! readbacks armed (draft confidence, shadow top-k, debug norms), since each
//! of those reads the logits on the host by design. Everything else takes the
//! per-draft chain exactly as before.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::head_v2::{MAX_CHAIN_DRAFTS, MAX_CHAIN_WIDTH};
use super::{MtpHead, MtpProposerState};
use crate::layer::ForwardContext;

/// Whether a per-sequence propose can take the device-resident chain.
pub(super) fn per_seq_eligible(
    chain_fits: bool,
    num_drafts: usize,
    grammar: bool,
    deferred_embed: bool,
    levers: &crate::layers::ops::ModelLevers,
) -> bool {
    chain_fits
        && (2..=MAX_CHAIN_DRAFTS).contains(&num_drafts)
        && !grammar
        && !deferred_embed
        && observational_readbacks_off(levers)
}

/// The per-draft host readbacks `forward_one` performs by design when one of
/// these diagnostics is armed; the chain declines rather than skip them.
pub(super) fn observational_readbacks_off(levers: &crate::layers::ops::ModelLevers) -> bool {
    levers.draft_conf_tau <= 0.0 && levers.shadow_topk == 0 && !levers.mtp_debug_norms
}

/// Arrange a position-major readback (`flat[j * width + i]` = sequence i's
/// draft j) as one draft list per sequence.
pub(super) fn per_sequence<T: Copy>(flat: &[T], width: usize, num_drafts: usize) -> Vec<Vec<T>> {
    (0..width)
        .map(|i| (0..num_drafts).map(|j| flat[j * width + i]).collect())
        .collect()
}

impl MtpHead {
    /// Whether the batched propose of `n` sequences x `num_drafts` can take
    /// the chain (the batched path is already grammar-free and declines when
    /// draft confidence is armed).
    pub(super) fn chain_eligible_batch(
        &self,
        ctx: &ForwardContext,
        n: usize,
        num_drafts: usize,
    ) -> bool {
        self.v2.chain.fits(n, num_drafts) && observational_readbacks_off(ctx.levers)
    }

    /// Per-sequence chain: `num_drafts` `forward_one` positions, drafts read
    /// back once. Same contract as the per-draft loop in `propose` with no
    /// deferred embedding target: returns every draft's real id.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_chain(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let round = self.v2.chain.begin(1)?;
        let mut current_hidden = target_hidden;
        for i in 0..num_drafts {
            self.v2.chain.set_pos(i);
            // Position 0 embeds the committed token from its host id; later
            // positions gather on the device (`HeadV2::chain_embed`), so the
            // id passed here is not read.
            let token = if i == 0 { last_token } else { 0 };
            self.forward_one(
                token,
                current_hidden,
                position + i,
                state,
                ctx,
                stream,
                None,
                None,
            )?;
            current_hidden = ctx.buffers.hidden_states();
        }
        let (ids, _) = self.v2.chain.read(ctx.gpu, num_drafts, false)?;
        drop(round);
        tracing::debug!("MTP propose chain: pos={position} drafts={ids:?} (one readback)");
        Ok(ids)
    }

    /// Batched chain: the M=n-row positions of `propose_batch_impl`, with the
    /// ids (and top-1 log-probabilities when asked) read back once per round.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_batch_chain(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut MtpProposerState],
        ctx: &ForwardContext,
        stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<Vec<u32>>> {
        let n = last_tokens.len();
        ensure!(n <= MAX_CHAIN_WIDTH, "propose chain: {n} sequences");
        let h = ctx.config.hidden_size;
        let want_lp = out_conf.is_some();
        // The LP argmax replaces the plain one only when its kernel exists;
        // without it the per-position path reports 0.0 (certainty), and so
        // does the chain.
        let lp_measured = want_lp && self.argmax_batch_lp_k.0 != 0;
        let round = self.v2.chain.begin(n)?;
        let mut cur_positions = positions.to_vec();
        // Host outputs of forward_batch_position, unused under the chain.
        let mut ids_unused = vec![0u32; n];
        let mut lp_unused = vec![0f32; n];
        for j in 0..num_drafts {
            self.v2.chain.set_pos(j);
            let hiddens_j: Vec<DevicePtr> = if j == 0 {
                target_hiddens.to_vec()
            } else {
                (0..n)
                    .map(|i| ctx.buffers.hidden_states().offset(i * h * 2))
                    .collect()
            };
            // `last_tokens` is read only at position 0 (host-indexed embed).
            self.forward_batch_position(
                last_tokens,
                &hiddens_j,
                &cur_positions,
                states,
                ctx,
                stream,
                &mut ids_unused,
                if want_lp {
                    Some(&mut lp_unused[..])
                } else {
                    None
                },
            )?;
            for p in cur_positions.iter_mut() {
                *p += 1;
            }
            for state in states.iter_mut() {
                state.last_num_drafted = j + 1;
            }
        }
        let (flat, lps) = self.v2.chain.read(ctx.gpu, num_drafts, lp_measured)?;
        drop(round);
        if let Some(c) = out_conf {
            *c = if lp_measured {
                per_sequence(&lps, n, num_drafts)
            } else {
                vec![vec![0.0; num_drafts]; n]
            };
        }
        Ok(per_sequence(&flat, n, num_drafts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readback_is_regrouped_per_sequence() {
        // 3 sequences x 2 drafts, position-major as the device writes it.
        let flat = [10u32, 20, 30, 11, 21, 31];
        assert_eq!(
            per_sequence(&flat, 3, 2),
            vec![vec![10, 11], vec![20, 21], vec![30, 31]]
        );
        assert_eq!(per_sequence(&[7u32], 1, 1), vec![vec![7]]);
    }

    #[test]
    fn eligibility_excludes_every_host_reading_mode() {
        let levers = crate::layers::ops::ModelLevers::default();
        assert!(observational_readbacks_off(&levers));
        assert!(per_seq_eligible(true, 3, false, false, &levers));
        assert!(
            !per_seq_eligible(false, 3, false, false, &levers),
            "row off"
        );
        assert!(
            !per_seq_eligible(true, 1, false, false, &levers),
            "one draft: nothing to save"
        );
        assert!(!per_seq_eligible(
            true,
            MAX_CHAIN_DRAFTS + 1,
            false,
            false,
            &levers
        ));
        assert!(
            !per_seq_eligible(true, 3, true, false, &levers),
            "grammar rows"
        );
        assert!(
            !per_seq_eligible(true, 3, false, true, &levers),
            "deferred embed target"
        );
        let mut conf = levers;
        conf.draft_conf_tau = 0.5;
        assert!(!observational_readbacks_off(&conf));
        let mut shadow = levers;
        shadow.shadow_topk = 4;
        assert!(!observational_readbacks_off(&shadow));
        let mut norms = levers;
        norms.mtp_debug_norms = true;
        assert!(!observational_readbacks_off(&norms));
    }
}
