// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the token-major forward of `Lfm3ConvLayer` for `T`
//! consecutive tokens of ONE sequence: a prefill chunk (`prefill`) and the
//! K-token batched decode (`decode_batched`), which is the same computation.
//!
//! The chunk continues from the sequence's window and leaves the window
//! advanced past it, so chunked prefill equals one-shot prefill and a decode
//! after a prefill equals continuing the prefill (the CPU reference pins both,
//! `reference.rs` and `tests.rs`). Two launches per chunk: y from the initial
//! window, then the window update (in place, ordered after y on the stream).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Lfm3ConvLayer;
use super::launch::{self, SeqLayout, Strides};
use crate::layer::ForwardContext;

impl Lfm3ConvLayer {
    /// `num_tokens` tokens of the sequence whose window is `state`; `hidden`
    /// and `residual` are `[num_tokens, hidden]`.
    pub(crate) fn forward_tokens(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }
        let total = u32::try_from(num_tokens).map_err(|_| {
            anyhow::anyhow!(
                "LFM3 conv layer {}: chunk of {num_tokens} tokens overflows u32",
                self.cfg.layer_idx
            )
        })?;
        let bcx = self.head(hidden, residual, num_tokens, ctx, stream)?;
        let y = ctx.buffers.ssm_deinterleaved();
        launch::prefill_chunk(
            ctx.gpu,
            &self.k.conv,
            bcx,
            self.conv.weight,
            y,
            SeqLayout::Single(state),
            total,
            Strides::dense(self.cfg.hidden as u32),
            stream,
        )?;
        self.tail(hidden, residual, y, num_tokens, ctx, stream)
    }
}
