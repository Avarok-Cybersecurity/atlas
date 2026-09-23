// SPDX-License-Identifier: AGPL-3.0-only

//! Step 9 of the batched cross-sequence propose: the FFN for the n stacked
//! rows, split out of [`super::forward_batch`] (500-line cap) together with
//! the ONE decision that names which FFN this head can batch.
//!
//! Two arms:
//!
//! * **dense** (`dense_ffn_generic`, the 27B dense drafter; BF16 or the
//!   weight-only NVFP4 stream under `--mtp-quantization nvfp4`):
//!   gate/up/down as M=n row projections through `proj_rows`, SiLU over the
//!   `[n, inter]` block — the original step 9.
//! * **native-FP8 MoE** (`moe_fp8`, Qwen3.6-35B-A3B-FP8): ONE
//!   `forward_fp8_grouped_decode` call for the n rows — batched router
//!   (`dense_gemm` + `moe_topk_softmax_batched`), rows sorted by expert, every
//!   active expert's weights streamed once. Before this arm an MoE drafter
//!   reported `propose_batch_max() == 1`, so a C=16 step ran 16 single-row
//!   `forward_one`s per draft position (each with a routing D2H sync) plus 16
//!   scalar LM-head GEMVs over the NVFP4 head — ~24.5 ms of the 176 ms step
//!   in the 2026-09-22 nsys profile (G14). The grouped path is the G9 kernel
//!   pair; its GPU oracle (`examples/fp8_moe_grouped_decode_microtest`) proves
//!   each row bit-identical to the M=1 fused kernels `forward_one` runs, given
//!   the same routing. The routing itself is the batched GEMM + top-k pair,
//!   which differs from the per-row GEMV + top-k in FP32 summation order and
//!   can flip a razor-margin expert choice — the SAME caveat the main model's
//!   grouped decode carries, and the only place a batched draft can differ
//!   from a per-seq one (`examples/fp8_moe_grouped_routing_microtest`).
//!
//! Arena aliasing, checked against `forward_batch_position`'s live set: the
//! grouped decode writes `gate_logits`, `scratch[0..2*n*top_k*4)`,
//! `expert_{gate,up,down}_out`, `logits`, `ssm_qkvz`, `attn_output` and
//! `moe_output`. Across step 9 only `hidden_states` (the residual stream) and
//! `norm_output` (the FFN input) are live, and neither is touched; `scratch`
//! and `logits` are rewritten by step 10 only after the FFN has consumed them.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::batch_caps::is_row_proj;
use super::{MtpHead, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Which FFN the batched propose runs for a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProposeFfnArm {
    Dense,
    MoeFp8Grouped,
}

impl ProposeFfnArm {
    /// Name for the "propose_batch active" log line.
    fn name(self) -> &'static str {
        match self {
            Self::Dense => "DENSE",
            Self::MoeFp8Grouped => "MOE-FP8-GROUPED",
        }
    }
}

/// Pure arm selection over the three facts that decide it. A head with
/// neither identity — or, impossibly, both — cannot be batched (`None`):
/// the dense arm needs a row-dispatchable triple ([`dense_rows_ok`]) AND its
/// SiLU kernel, the MoE arm needs
/// the native-FP8 layer. Tested without a GPU below.
pub(super) fn propose_ffn_arm(
    dense_rows: bool,
    silu_mul_k: bool,
    moe_fp8: bool,
) -> Option<ProposeFfnArm> {
    match (dense_rows && silu_mul_k, moe_fp8) {
        (true, false) => Some(ProposeFfnArm::Dense),
        (false, true) => Some(ProposeFfnArm::MoeFp8Grouped),
        _ => None,
    }
}

/// Whether a dense FFN triple can run on the batched row dispatch: each of
/// gate/up/down BF16 or the weight-only NVFP4 stream of a dense head under
/// `--mtp-quantization nvfp4` (`batch_caps::is_row_proj`, `proj_rows`).
pub(super) fn dense_rows_ok(
    ffn: Option<&(ProjectionWeight, ProjectionWeight, ProjectionWeight)>,
) -> bool {
    ffn.is_some_and(|(g, u, d)| is_row_proj(g) && is_row_proj(u) && is_row_proj(d))
}

impl MtpHead {
    /// The arm for THIS head — the single reader for the scope check
    /// (`batch_caps`), the dispatch below and the log line.
    pub(super) fn propose_ffn_arm(&self) -> Option<ProposeFfnArm> {
        propose_ffn_arm(
            dense_rows_ok(self.dense_ffn_generic.as_ref()),
            self.moe_silu_mul_k.is_some(),
            self.moe_fp8.is_some(),
        )
    }

    /// Arm name for the proof-of-engagement log line (`NONE` = not batchable).
    pub(super) fn propose_ffn_arm_name(&self) -> &'static str {
        self.propose_ffn_arm().map_or("NONE", ProposeFfnArm::name)
    }

    /// `hidden[0..n] += FFN(normed2[0..n])` for the n stacked rows.
    pub(super) fn ffn_rows(
        &self,
        normed2: DevicePtr,
        hidden: DevicePtr,
        n: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        let ffn_out = match self.propose_ffn_arm() {
            Some(ProposeFfnArm::Dense) => {
                let inter = if ctx.config.intermediate_size > 0 {
                    ctx.config.intermediate_size as u32
                } else {
                    ctx.config.moe_intermediate_size as u32
                };
                let Some((gate_w, up_w, down_w)) = self.dense_ffn_generic.as_ref() else {
                    anyhow::bail!("propose_batch: no dense FFN (propose_ffn_arm lied)");
                };
                let Some(silu_k) = self.moe_silu_mul_k else {
                    anyhow::bail!("propose_batch: no SiLU kernel (propose_ffn_arm lied)");
                };
                let gate_out = ctx.buffers.expert_gate_out();
                let up_out = ctx.buffers.expert_up_out();
                self.proj_rows(gpu, normed2, gate_w, gate_out, n, inter, h as u32, stream)?;
                self.proj_rows(gpu, normed2, up_w, up_out, n, inter, h as u32, stream)?;
                ops::moe_silu_mul(
                    gpu,
                    silu_k,
                    gate_out,
                    up_out,
                    gate_out,
                    n as u32 * inter,
                    stream,
                )?;
                let ffn_out = ctx.buffers.moe_output();
                self.proj_rows(gpu, gate_out, down_w, ffn_out, n, h as u32, inter, stream)?;
                ffn_out
            }
            Some(ProposeFfnArm::MoeFp8Grouped) => {
                let Some(moe) = self.moe_fp8.as_ref() else {
                    anyhow::bail!("propose_batch: no FP8 MoE layer (propose_ffn_arm lied)");
                };
                // Refuses loudly if `propose_batch` did not gate on
                // `fp8_grouped_decode_ok` first; output lands in moe_output.
                moe.forward_fp8_grouped_decode(normed2, n, ctx, stream)?;
                ctx.buffers.moe_output()
            }
            None => anyhow::bail!("propose_batch: no batchable FFN (can_propose_batch lied)"),
        };
        ops::residual_add(
            gpu,
            self.residual_add_k,
            hidden,
            ffn_out,
            n as u32 * h as u32,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ProposeFfnArm, propose_ffn_arm};

    #[test]
    fn dense_bf16_head_batches_through_the_dense_arm() {
        assert_eq!(
            propose_ffn_arm(true, true, false),
            Some(ProposeFfnArm::Dense)
        );
    }

    #[test]
    fn native_fp8_moe_head_batches_through_the_grouped_arm() {
        // The whole point of the lever: an MoE drafter with FP8 tables is
        // batchable (before, this was the `None` row below).
        assert_eq!(
            propose_ffn_arm(false, false, true),
            Some(ProposeFfnArm::MoeFp8Grouped)
        );
        // The dense SiLU kernel is irrelevant to the MoE arm.
        assert_eq!(
            propose_ffn_arm(false, true, true),
            Some(ProposeFfnArm::MoeFp8Grouped)
        );
    }

    #[test]
    fn per_expert_moe_head_cannot_batch() {
        // BF16-on-disk MTP experts (no FP8 tables) keep the per-seq loop:
        // `moe_forward_generic` has no cross-sequence form.
        assert_eq!(propose_ffn_arm(false, true, false), None);
        assert_eq!(propose_ffn_arm(false, false, false), None);
    }

    #[test]
    fn dense_arm_needs_its_silu_kernel() {
        // A 0-handle SiLU kernel must refuse the dense arm rather than
        // unwrap at dispatch time (the previous code `unwrap()`ed it).
        assert_eq!(propose_ffn_arm(true, false, false), None);
    }

    #[test]
    fn a_head_with_both_identities_is_refused() {
        // Impossible by construction (`MtpHead::new` builds the MoE only when
        // `dense_ffn` is absent); if it ever happens, refuse to batch rather
        // than pick one silently.
        assert_eq!(propose_ffn_arm(true, true, true), None);
    }

    #[test]
    fn arm_names_are_distinct_for_the_log_line() {
        assert_ne!(
            ProposeFfnArm::Dense.name(),
            ProposeFfnArm::MoeFp8Grouped.name()
        );
    }
}
