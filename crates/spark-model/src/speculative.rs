// SPDX-License-Identifier: AGPL-3.0-only

//! Speculative decoding abstraction (SDD).
//!
//! Defines the [`DraftProposer`] trait for speculative decoding strategies.
//! MTP implements this first; EAGLE-3 can implement later without engine changes.

use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::ForwardContext;

/// Per-sequence state owned by a [`DraftProposer`].
///
/// Stores KV cache, hidden states, or whatever the proposer needs
/// across decode steps. Follows the same downcasting pattern as `LayerState`.
pub trait ProposerState: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A draft token proposer for speculative decoding.
///
/// The engine calls `propose()` after each target decode to get draft tokens,
/// then verifies them with the target model. `after_verify()` lets the
/// proposer trim state (e.g., KV cache) based on how many drafts were accepted.
pub trait DraftProposer: Send + Sync {
    /// Allocate per-sequence proposer state.
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>>;

    /// Propose up to `num_drafts` tokens autoregressively.
    ///
    /// # Arguments
    /// * `last_token` - The last verified token (target model output)
    /// * `target_hidden` - Target model's hidden states after final norm [1, hidden_size] BF16
    /// * `position` - Current sequence position (for RoPE)
    /// * `num_drafts` - Maximum number of draft tokens to produce
    /// * `state` - Per-sequence proposer state
    /// * `ctx` - Shared forward context (buffers, gpu, config)
    /// * `stream` - CUDA stream handle
    /// * `grammar_bitmask` - Optional XGrammar bitmask (ceil(vocab_size/32) i32
    ///   words). When `Some`, drafts are constrained to tokens the grammar
    ///   accepts at the current matcher position; bit `tok` set ⇒ allowed.
    ///   `None` preserves the unconstrained fast path.
    /// * `target_hidden_stack` - Optional pointer to a contiguous buffer of
    ///   `5 × target_hidden × bf16` containing the most-recently-decoded
    ///   token's hidden states captured at the drafter's `target_layer_ids`
    ///   (DFlash uses this; MTP ignores). Layout matches vLLM's
    ///   `combine_hidden_states` input: shallow-to-deep concatenation along
    ///   the feature axis.
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>>;

    /// Read the draft token ID stored on GPU by the last `propose()` call
    /// that used `draft_embed_target = Some(...)`. Returns 0 if not supported.
    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        let _ = gpu;
        Ok(0)
    }

    /// Called after target verification to trim proposer state.
    ///
    /// `num_accepted` indicates how many draft tokens were accepted.
    /// The proposer should trim its KV cache / state to match.
    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        stream: u64,
    ) -> Result<()>;

    /// Free per-sequence proposer state (KV cache blocks, etc.).
    ///
    /// Must be called when a sequence is finished to avoid resource leaks.
    fn free_state(&self, state: &mut dyn ProposerState) -> Result<()> {
        let _ = state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProposerState {
        tokens_proposed: Vec<u32>,
    }

    impl ProposerState for MockProposerState {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn test_proposer_state_downcast() {
        let state: Box<dyn ProposerState> = Box::new(MockProposerState {
            tokens_proposed: vec![42, 99],
        });
        let mock = state.as_any().downcast_ref::<MockProposerState>().unwrap();
        assert_eq!(mock.tokens_proposed, vec![42, 99]);
    }

    #[test]
    fn test_proposer_state_downcast_mut() {
        let mut state: Box<dyn ProposerState> = Box::new(MockProposerState {
            tokens_proposed: vec![],
        });
        let mock = state
            .as_any_mut()
            .downcast_mut::<MockProposerState>()
            .unwrap();
        mock.tokens_proposed.push(7);
        assert_eq!(mock.tokens_proposed, vec![7]);
    }
}
