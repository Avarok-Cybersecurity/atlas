// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: fingerprint a streamed chat response. Wraps the neutral
//! delta stream (`ir::DeltaStream`) and, at the terminal `Finish` delta,
//! attaches `usage.atlas_token_sha` and `usage.atlas_output_sha256`.
//!
//! The token ids come from the `token_ids` the streaming core already attaches
//! to each delta when `return_token_ids` is on. A request that asked only for
//! evidence has that switched on internally by the dispatcher, and this tap
//! strips the ids again before they reach the wire, so the client sees exactly
//! the chunks it would have seen without `atlas_evidence`.

use futures::StreamExt;

use super::{OutputHasher, TokenHasher};
use crate::ir::{DeltaStream, StreamDelta};

/// The per-stream fingerprint state.
pub(crate) struct EvidenceTap {
    tokens: TokenHasher,
    output: OutputHasher,
    expose_ids: bool,
}

impl EvidenceTap {
    /// `expose_ids`: the client itself asked for `return_token_ids`, so the
    /// ids stay on the wire.
    pub(crate) fn new(expose_ids: bool) -> Self {
        Self {
            tokens: TokenHasher::new(),
            output: OutputHasher::new(),
            expose_ids,
        }
    }

    /// Fold a delta's ids into the token fingerprint and hand back what the
    /// wire should carry.
    fn take_ids(&mut self, ids: Vec<u32>) -> Vec<u32> {
        self.tokens.update(&ids);
        if self.expose_ids { ids } else { Vec::new() }
    }

    /// Observe one delta and return it (ids stripped when the client did not
    /// ask for them; the terminal delta carries the fingerprints).
    pub(crate) fn apply(&mut self, delta: StreamDelta) -> StreamDelta {
        match delta {
            StreamDelta::Content { text, token_ids } => {
                self.output.add_content(&text);
                let token_ids = self.take_ids(token_ids);
                StreamDelta::Content { text, token_ids }
            }
            StreamDelta::Reasoning { text, token_ids } => {
                self.output.add_reasoning(&text);
                let token_ids = self.take_ids(token_ids);
                StreamDelta::Reasoning { text, token_ids }
            }
            StreamDelta::ToolCallArgs {
                index,
                fragment,
                token_ids,
            } => {
                let token_ids = self.take_ids(token_ids);
                StreamDelta::ToolCallArgs {
                    index,
                    fragment,
                    token_ids,
                }
            }
            StreamDelta::Finish {
                reason,
                mut usage,
                token_ids,
            } => {
                let token_ids = self.take_ids(token_ids);
                let tokens = std::mem::take(&mut self.tokens).finish();
                let output = std::mem::take(&mut self.output).finish();
                usage.atlas.token_sha = Some(tokens);
                usage.atlas.output_sha = Some(output);
                StreamDelta::Finish {
                    reason,
                    usage,
                    token_ids,
                }
            }
            other @ (StreamDelta::ToolCallStart { .. }
            | StreamDelta::Refusal { .. }
            | StreamDelta::Error { .. }) => other,
        }
    }
}

/// Wrap `deltas` so the terminal delta carries the fingerprints.
pub(crate) fn wrap(deltas: DeltaStream, expose_ids: bool) -> DeltaStream {
    let mut tap = EvidenceTap::new(expose_ids);
    Box::pin(deltas.map(move |d| tap.apply(d)))
}
