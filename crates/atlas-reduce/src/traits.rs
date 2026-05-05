// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::error::Result;

/// Reduction operations backend trait.
#[allow(clippy::too_many_arguments)]
pub trait Reduce {
    /// Top-K expert selection: for each token, select top-k experts by score.
    fn topk(
        &self,
        scores_ptr: u64,       // [num_tokens, num_experts] FP32
        topk_ids_ptr: u64,     // [num_tokens, topk] i32 (output)
        topk_weights_ptr: u64, // [num_tokens, topk] FP32 (output)
        num_tokens: u32,
        num_experts: u32,
        topk: u32,
        stream_ptr: u64,
    ) -> Result<()>;

    /// Weighted sum of expert outputs: `output = sum(weights[i] * expert_out[i])` for topk experts.
    fn moe_sum(
        &self,
        expert_outputs_ptr: u64, // [num_tokens * topk, hidden_size] BF16
        topk_weights_ptr: u64,   // [num_tokens, topk] FP32
        output_ptr: u64,         // [num_tokens, hidden_size] BF16
        num_tokens: u32,
        hidden_size: u32,
        topk: u32,
        stream_ptr: u64,
    ) -> Result<()>;

    /// Softmax over last dimension.
    fn softmax(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        num_rows: u32,
        num_cols: u32,
        stream_ptr: u64,
    ) -> Result<()>;
}
