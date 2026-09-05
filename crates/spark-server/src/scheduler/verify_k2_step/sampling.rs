// SPDX-License-Identifier: AGPL-3.0-only

//! K=2 MTP samples only rows that will be emitted, using the serial sampler.
//! Serial request policies apply, including sampling and min-p; historical
//! MTP-only sampler kill switches do not override those policies here.

use crate::api::TokenLogprobs;
use crate::scheduler::ActiveSeq;
use crate::scheduler::decode_logits_seq::process_seq_logits;
use crate::scheduler::emit_step::emit_token;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::sched_ctx::SchedCtx;
use spark_model::traits::Model;

pub(super) struct Rows {
    bytes: Vec<u8>,
    vocab: usize,
}

impl Rows {
    pub(super) fn copy(model: &dyn Model) -> anyhow::Result<Self> {
        let vocab = model.vocab_size();
        let mut bytes = vec![0; 2 * vocab * 2];
        model.copy_logits_to_host(model.logits_buffer_ptr(), &mut bytes)?;
        Ok(Self { bytes, vocab })
    }

    pub(super) fn pick(
        &self,
        row: usize,
        seq: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> (u32, Option<TokenLogprobs>) {
        process_seq_logits(seq, &self.bytes, row, self.vocab, 2, false, ctx, false)
    }

    /// Called only after the accepted verdict has been broadcast to EP ranks.
    /// Real emission advances history, grammar and thinking/tool state before
    /// the bonus sample. The serial sampler uses the resulting history length
    /// for its seed; no speculative position offset or rollback is needed.
    pub(super) fn emit_accepted(
        &self,
        first: (u32, Option<TokenLogprobs>),
        seq: &mut ActiveSeq,
        sched: &SchedCtx,
        ctx: &LogitsContext,
    ) -> Option<u32> {
        emit_token(seq, first.0, first.1, sched);
        if seq.finished {
            return None;
        }
        let (bonus, logprobs) = self.pick(1, seq, ctx);
        emit_token(seq, bonus, logprobs, sched);
        Some(bonus)
    }
}

#[cfg(test)]
mod tests;
