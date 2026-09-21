// SPDX-License-Identifier: AGPL-3.0-only

//! The final-norm step in front of every lm_head projection.
//!
//! Split from `impl_a3.rs` (500-LoC cap). One helper, called by all 18
//! norm-then-lm-head sites, so a model whose checkpoint has no final norm
//! (`final_norm_identity`) degrades to an identity copy at every one of
//! them instead of each site growing its own branch.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    /// The final-norm step in front of every lm_head projection. For models
    /// whose checkpoint has no final norm (`final_norm_identity` --
    /// qwen4_exp, where the mHC mixer already normalized while collapsing the
    /// streams) this is an identity COPY: `rms_norm` with a ones weight is
    /// NOT identity, it divides each row by its RMS and flattens the logits
    /// by a per-token factor the reference forward does not have.
    pub(super) fn final_norm_apply(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        num_tokens: u32,
        hidden_size: u32,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        if self.config.final_norm_identity {
            return self.gpu.copy_d2d_async(
                input,
                output,
                num_tokens as usize * hidden_size as usize * 2,
                stream,
            );
        }
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            input,
            &self.final_norm,
            output,
            num_tokens,
            hidden_size,
            eps,
            stream,
        )
    }
}
