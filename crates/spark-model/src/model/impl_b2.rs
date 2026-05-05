// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn generate_self_speculative_inner(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<crate::engine::GenerateResult> {
        let logits_ptr = self.prefill(prompt_tokens, seq, stream)?;
        let first_token = self.argmax_on_device(logits_ptr, stream)?;

        let mut output_tokens = Vec::with_capacity(params.max_tokens);
        output_tokens.push(first_token);

        if params.stop_token_ids.contains(&first_token) {
            return Ok(crate::engine::GenerateResult {
                output_tokens,
                finish_reason: "stop".to_string(),
            });
        }

        let mut total_accepted = 0usize;
        let mut total_proposed = 0usize;
        let mut total_steps = 0usize;

        while output_tokens.len() < params.max_tokens {
            let last_token = *output_tokens.last().unwrap();

            // 1. Full-model decode to get token_0
            let logits = self.decode(last_token, seq, stream)?;
            let token_0 = self.argmax_on_device(logits, stream)?;

            // 2. Draft phase: skip SSM layers for cheap predictions
            let seq_len_before_draft = seq.seq_len;
            let tokens_before_draft = seq.tokens.len();

            let mut draft_tokens = Vec::with_capacity(num_drafts);
            let mut draft_token = token_0;
            for _ in 0..num_drafts {
                let logits = self.decode_draft(draft_token, seq, stream)?;
                draft_token = self.argmax_on_device(logits, stream)?;
                draft_tokens.push(draft_token);
            }

            // 3. Rewind to pre-draft state (SSM unchanged since we skipped SSM layers)
            seq.seq_len = seq_len_before_draft;
            seq.tokens.truncate(tokens_before_draft);

            // 4. Verify: run full model on [token_0, d1, ..., dK]
            let mut verify_tokens = vec![token_0];
            verify_tokens.extend_from_slice(&draft_tokens);

            self.checkpoint_ssm_states(seq)?;
            let seq_len_before_verify = seq.seq_len;

            let verified = self.decode_verify(&verify_tokens, seq, stream)?;

            // 5. Compare draft vs verified
            output_tokens.push(token_0);
            let n_drafts = draft_tokens.len();
            let mut num_accepted = 0;

            for i in 0..n_drafts {
                if draft_tokens[i] == verified[i] {
                    output_tokens.push(draft_tokens[i]);
                    num_accepted += 1;
                } else {
                    output_tokens.push(verified[i]);
                    break;
                }
            }

            if num_accepted == n_drafts && n_drafts > 0 {
                output_tokens.push(verified[n_drafts]);
            }

            total_accepted += num_accepted;
            total_proposed += n_drafts;
            total_steps += 1;

            // 6. Rollback extra tokens if needed
            // tokens_added = token_0 (always kept) + accepted drafts
            let tokens_added = 1 + num_accepted;
            let expected_seq_len = seq_len_before_verify + tokens_added;

            if seq.seq_len > expected_seq_len {
                let extra = seq.seq_len - expected_seq_len;
                for _ in 0..extra {
                    seq.seq_len -= 1;
                    seq.tokens.pop();
                }
                // +1 because token_0 is always accepted in the verify batch
                self.rollback_ssm_states(seq, num_accepted + 1)?;
            }

            if let Some(last) = output_tokens.last()
                && params.stop_token_ids.contains(last)
            {
                break;
            }
        }

        output_tokens.truncate(params.max_tokens);

        if total_steps > 0 {
            tracing::info!(
                "Self-speculative decode: {} steps, {}/{} accepted ({:.0}%)",
                total_steps,
                total_accepted,
                total_proposed,
                if total_proposed > 0 {
                    total_accepted as f64 / total_proposed as f64 * 100.0
                } else {
                    0.0
                },
            );
        }

        let finish_reason = if output_tokens
            .last()
            .is_some_and(|t| params.stop_token_ids.contains(t))
        {
            "stop".to_string()
        } else {
            "length".to_string()
        };
        Ok(crate::engine::GenerateResult {
            output_tokens,
            finish_reason,
        })
    }

    pub(super) fn generate_speculative_inner(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
        proposer: &Arc<dyn DraftProposer>,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<crate::engine::GenerateResult> {
        let mut prop_state = proposer.alloc_state(self.gpu.as_ref())?;

        let logits_ptr = self.prefill(prompt_tokens, seq, stream)?;
        let first_token = self.argmax_on_device(logits_ptr, stream)?;

        let mut output_tokens = Vec::with_capacity(params.max_tokens);
        output_tokens.push(first_token);

        if params.stop_token_ids.contains(&first_token) {
            return Ok(crate::engine::GenerateResult {
                output_tokens,
                finish_reason: "stop".to_string(),
            });
        }

        let mut total_accepted = 0usize;
        let mut total_proposed = 0usize;
        let mut total_steps = 0usize;

        while output_tokens.len() < params.max_tokens {
            let last_token = *output_tokens.last().unwrap();

            let logits = self.decode(last_token, seq, stream)?;
            let token_0 = self.argmax_on_device(logits, stream)?;

            let target_hidden = self.hidden_after_norm();
            let position = seq.seq_len;
            let ctx = ForwardContext {
                buffers: &self.buffers,
                gpu: self.gpu.as_ref(),
                config: &self.config,
                attn_metadata: None,
                profile: false,
                // MTP runs on rank 0 only — no EP all_reduce (BUG #26).
                comm: None,
                graph_capture: false,
            };
            let drafts = proposer.propose(
                token_0,
                target_hidden,
                position,
                num_drafts,
                prop_state.as_mut(),
                &ctx,
                stream,
                None,
                None, // grammar_bitmask: internal self-spec callsite, no grammar routing here
                self.dflash_hidden_save,
            )?;
            let n_drafts = drafts.len();

            let mut verify_tokens = vec![token_0];
            verify_tokens.extend_from_slice(&drafts);

            self.checkpoint_ssm_states(seq)?;
            let seq_len_before = seq.seq_len;

            let verified = self.decode_verify(&verify_tokens, seq, stream)?;

            output_tokens.push(token_0);
            let mut num_accepted = 0;

            for i in 0..n_drafts {
                if drafts[i] == verified[i] {
                    output_tokens.push(drafts[i]);
                    num_accepted += 1;
                } else {
                    output_tokens.push(verified[i]);
                    break;
                }
            }

            if num_accepted == n_drafts && n_drafts > 0 {
                output_tokens.push(verified[n_drafts]);
            }

            total_accepted += num_accepted;
            total_proposed += n_drafts;
            total_steps += 1;

            let tokens_added = 1
                + num_accepted
                + if num_accepted == n_drafts && n_drafts > 0 {
                    1
                } else {
                    0
                };
            let expected_seq_len = seq_len_before + tokens_added;

            if seq.seq_len > expected_seq_len {
                let extra = seq.seq_len - expected_seq_len;
                for _ in 0..extra {
                    seq.seq_len -= 1;
                    seq.tokens.pop();
                }
                // +1 because the verify batch begins with token_0 (always
                // committed). `h_state_intermediates[i]` holds state after
                // verify_token[i], so restoring to "state after token_0 +
                // num_accepted drafts" needs index num_accepted, which the
                // helper computes as `arg - 1`. Matches the self-speculative
                // path. Without this, the SSM state lags one token behind
                // the committed sequence on every verify step, accumulating
                // drift that corrupts subsequent decodes on 80B-nvfp4-mtp.
                self.rollback_ssm_states(seq, num_accepted + 1)?;
            }

            proposer.after_verify(num_accepted, prop_state.as_mut(), stream)?;

            if let Some(last) = output_tokens.last()
                && params.stop_token_ids.contains(last)
            {
                if total_steps > 0 {
                    tracing::info!(
                        "Speculative decode: {} steps, {}/{} accepted ({:.0}%)",
                        total_steps,
                        total_accepted,
                        total_proposed,
                        if total_proposed > 0 {
                            total_accepted as f64 / total_proposed as f64 * 100.0
                        } else {
                            0.0
                        },
                    );
                }
                return Ok(crate::engine::GenerateResult {
                    output_tokens,
                    finish_reason: "stop".to_string(),
                });
            }
        }

        output_tokens.truncate(params.max_tokens);

        if total_steps > 0 {
            tracing::info!(
                "Speculative decode: {} steps, {}/{} accepted ({:.0}%)",
                total_steps,
                total_accepted,
                total_proposed,
                if total_proposed > 0 {
                    total_accepted as f64 / total_proposed as f64 * 100.0
                } else {
                    0.0
                },
            );
        }

        Ok(crate::engine::GenerateResult {
            output_tokens,
            finish_reason: "length".to_string(),
        })
    }
}
