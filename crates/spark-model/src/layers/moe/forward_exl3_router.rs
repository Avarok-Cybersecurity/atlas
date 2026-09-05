// SPDX-License-Identifier: AGPL-3.0-only

//! Experimental decode-shaped routing for small EXL3 verification batches.

fn row_router_required(enabled: bool, exact_pass: bool, rows: usize) -> bool {
    enabled && exact_pass && (2..=4).contains(&rows)
}

use super::*;

/// Experimental single-token expert geometry with additional slot waves.
/// Independent of the router switch so attribution can isolate both stages.
pub(super) fn stable_grid_enabled(ctx: &ForwardContext, rows: usize) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED
        .get_or_init(|| std::env::var("ATLAS_VERIFY_EXL3_STABLE_GRID").as_deref() == Ok("1"));
    row_router_required(enabled, ctx.gdn_exact_replay, rows)
}

impl MoeLayer {
    /// Opt in with `ATLAS_VERIFY_EXL3_ROW_ROUTER=1`. Only the router's
    /// projection and top-k use decode kernels; packed experts stay batched.
    /// Pair with `ATLAS_NO_VERIFY_ROW_FFN=1` to isolate routing from experts
    /// in the mHC verifier. No normal decode or prefill dispatch changes.
    pub(super) fn exl3_row_router(
        &self,
        input: DevicePtr,
        logits: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled = *ENABLED
            .get_or_init(|| std::env::var("ATLAS_VERIFY_EXL3_ROW_ROUTER").as_deref() == Ok("1"));
        if !row_router_required(enabled, ctx.gdn_exact_replay, rows) {
            return Ok(false);
        }
        let h = ctx.config.hidden_size;
        let experts = ctx.config.num_experts;
        for row in 0..rows {
            let src = input.offset(row * h * 2);
            let dst = logits.offset(row * experts * 2);
            if let Some(ref weight) = self.gate_nvfp4 {
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv,
                    self.w4a16_gemv_sw,
                    ctx.levers.gemv_sw,
                    src,
                    weight,
                    dst,
                    self.router_logits_n,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv,
                    src,
                    &self.weights.gate,
                    dst,
                    self.router_logits_n,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(true)
    }

    pub(super) fn exl3_row_topk(
        &self,
        logits: DevicePtr,
        indices: DevicePtr,
        weights: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let experts = ctx.config.num_experts;
        let top_k = ctx.config.num_experts_per_tok;
        for row in 0..rows {
            let src = logits.offset(row * experts * 2);
            let ids = indices.offset(row * top_k * 4);
            let probs = weights.offset(row * top_k * 4);
            if let Some(bias) = self.correction_bias_dev {
                anyhow::ensure!(
                    ctx.config.scoring_func != "sqrtsoftplus"
                        && ctx.config.scoring_func != "softmax",
                    "EXL3 native MoE: correction bias with {:?} is not supported",
                    ctx.config.scoring_func,
                );
                ops::moe_topk_sigmoid(
                    ctx.gpu,
                    self.moe_topk_sigmoid_k,
                    src,
                    bias,
                    ids,
                    probs,
                    experts as u32,
                    top_k as u32,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    stream,
                )?;
            } else {
                ops::moe_topk_softmax(
                    ctx.gpu,
                    self.moe_topk,
                    src,
                    ids,
                    probs,
                    experts as u32,
                    top_k as u32,
                    ctx.config.norm_topk_prob,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::row_router_required;

    #[test]
    fn row_router_is_limited_to_opted_in_small_verify_batches() {
        for rows in 2..=4 {
            assert!(row_router_required(true, true, rows));
            assert!(!row_router_required(false, true, rows));
            assert!(!row_router_required(true, false, rows));
        }
        for rows in [0, 1, 5, 16, 256] {
            assert!(!row_router_required(true, true, rows));
        }
    }
}
