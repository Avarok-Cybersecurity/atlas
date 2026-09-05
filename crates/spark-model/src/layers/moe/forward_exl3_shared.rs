// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-shaped shared expert for the native EXL3 MoE arm.
//!
//! `forward_exl3_after_routing` used to evaluate the (NVFP4-materialized)
//! shared expert through `run_shared_expert_prefill`, i.e. the prefill-tiled
//! `w4a16_gemm` (`M_TILE = 64`) at `m = 1`. nsys on one GB10 (2026-09-05,
//! 4.05bpw, 200-token greedy decode) put that at **50.7% of all GPU time**:
//! 3 launches x 48 layers per token at ~274 us each = ~39 ms of an ~85 ms
//! token, while every EXL3 trellis kernel together was ~21 ms. The NVFP4
//! decode path never paid this because it fuses the shared expert into the
//! routed gate-up / silu-down kernels as an extra slot.
//!
//! This arm runs the same three NVFP4 projections per row through the
//! single-warp `w4a16_gemv_sw` (the router's decode kernel, ~9 us at these
//! shapes) whenever the row count is small. Numerics: the GEMV and the tiled
//! GEMM both compute `sum_k a[k] * dequant(w[n,k])` in fp32 with the FP8 group
//! scale factored out — same math, different reduction order, so outputs are
//! not bit-identical to the old arm (the same contract as every other
//! gemm-vs-gemv decode dispatch in this crate).
//!
//! Kill switch: `ATLAS_EXL3_SHARED_PREFILL_GEMM=1` restores the old arm, for
//! back-to-back A/B only.

use super::*;

/// Row cap for the per-row GEMV arm. Above it the prefill-tiled GEMM wins
/// (one tile covers 64 rows) — same bound as the EXL3 GEMV tier.
pub(super) const EXL3_SHARED_GEMV_MAX_ROWS: u32 = 8;

/// Pure dispatch decision: GEMV arm when the row count is small, the shared
/// expert is NVFP4 (no FP8 twin — that arm has its own kernels), and the
/// kill switch is not set.
pub(super) fn shared_gemv_arm(rows: u32, has_fp8_shared: bool, kill_switch: bool) -> bool {
    (1..=EXL3_SHARED_GEMV_MAX_ROWS).contains(&rows) && !has_fp8_shared && !kill_switch
}

fn kill_switch_set() -> bool {
    static KILL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KILL.get_or_init(|| std::env::var("ATLAS_EXL3_SHARED_PREFILL_GEMM").as_deref() == Ok("1"))
}

impl MoeLayer {
    /// Shared expert for the EXL3 decode arm: writes `silu(gate) * up @ down`
    /// for `num_tokens` rows of `input` into `ctx.buffers.attn_output()`
    /// (BF16 `[num_tokens, hidden]`), scratching `ssm_deinterleaved()` and
    /// `ssm_qkvz()` exactly like `run_shared_expert_prefill` — same buffers,
    /// same output, so the caller's `moe_batched_blend` is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_expert_exl3_decode(
        &self,
        input: DevicePtr,
        num_tokens: u32,
        hidden: u32,
        shared_inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if shared_inter == 0 {
            return Ok(());
        }
        let has_fp8 = self.shared_gate_fp8.is_some()
            || self.shared_up_fp8.is_some()
            || self.shared_down_fp8.is_some();
        if !shared_gemv_arm(num_tokens, has_fp8, kill_switch_set()) {
            return self.run_shared_expert_prefill(
                input,
                num_tokens,
                hidden,
                shared_inter,
                stream,
                stream,
                false,
                ctx,
            );
        }

        let gate_out = ctx.buffers.ssm_deinterleaved();
        let up_out = ctx.buffers.ssm_qkvz();
        let down_out = ctx.buffers.attn_output();

        // Checkpoint-native BF16 shared expert (already GEMV at one row).
        if self.run_bf16_shared_expert(
            input,
            num_tokens,
            hidden,
            shared_inter,
            gate_out,
            up_out,
            down_out,
            ctx,
            stream,
        )? {
            return Ok(());
        }

        let h_row = hidden as usize * 2; // bf16 row strides
        let i_row = shared_inter as usize * 2;
        let w = &self.weights.shared_expert;
        for row in 0..num_tokens as usize {
            let src = input.offset(row * h_row);
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                self.w4a16_gemv_sw,
                ctx.levers.gemv_sw,
                src,
                &w.gate_proj,
                gate_out.offset(row * i_row),
                shared_inter,
                hidden,
                stream,
            )?;
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                self.w4a16_gemv_sw,
                ctx.levers.gemv_sw,
                src,
                &w.up_proj,
                up_out.offset(row * i_row),
                shared_inter,
                hidden,
                stream,
            )?;
        }
        // Same activation kernel as the prefill arm (in place into gate_out).
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            num_tokens * shared_inter,
            stream,
        )?;
        for row in 0..num_tokens as usize {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                self.w4a16_gemv_sw,
                ctx.levers.gemv_sw,
                gate_out.offset(row * i_row),
                &w.down_proj,
                down_out.offset(row * h_row),
                hidden,
                shared_inter,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_arm_covers_decode_and_verify_widths_only() {
        for rows in 1..=EXL3_SHARED_GEMV_MAX_ROWS {
            assert!(shared_gemv_arm(rows, false, false), "rows={rows}");
        }
        assert!(!shared_gemv_arm(0, false, false));
        assert!(!shared_gemv_arm(
            EXL3_SHARED_GEMV_MAX_ROWS + 1,
            false,
            false
        ));
        assert!(!shared_gemv_arm(64, false, false));
    }

    #[test]
    fn fp8_twin_and_kill_switch_keep_the_prefill_arm() {
        assert!(
            !shared_gemv_arm(1, true, false),
            "FP8 shared expert has its own kernels"
        );
        assert!(
            !shared_gemv_arm(1, false, true),
            "ATLAS_EXL3_SHARED_PREFILL_GEMM=1 restores the old arm"
        );
    }
}
