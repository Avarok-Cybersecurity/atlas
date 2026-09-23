// SPDX-License-Identifier: AGPL-3.0-only

//! THE chain input of an MTP propose: which buffer draft `j > 0` reads as
//! "the previous position's hidden". One reader for both chains (the
//! single-sequence `propose` and the batched `propose_batch_impl`), so the
//! lever cannot steer one lane and not the other.

use spark_runtime::gpu::DevicePtr;

use super::MtpHead;
use crate::layer::ForwardContext;

impl MtpHead {
    /// Row `i` of the returned buffer is sequence `i`'s hidden from the
    /// position just drafted: the drafter's final-normed hidden
    /// (`norm_output`, what its LM head read) under
    /// `ModelLevers::mtp_chain_postnorm`, else its pre-norm residual stream
    /// (`hidden_states`). Both are written by every forward position and are
    /// read by the next position's `pre_fc_norm_hidden` BEFORE that position
    /// writes either buffer (step 2 precedes steps 4/5, same stream).
    pub(super) fn chain_hidden(ctx: &ForwardContext) -> DevicePtr {
        if ctx.levers.mtp_chain_postnorm {
            ctx.buffers.norm_output()
        } else {
            ctx.buffers.hidden_states()
        }
    }
}
