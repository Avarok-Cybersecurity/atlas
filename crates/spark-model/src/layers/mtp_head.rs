// SPDX-License-Identifier: AGPL-3.0-only

//! MTP (Multi-Token Prediction) head implementing [`DraftProposer`].
//!
//! Single transformer decoder layer trained jointly with the target model.
//! Forward pass: embed+hidden concat → fc → attention → MoE → norm → lm_head → argmax.
//!
//! Weight precision is parameterized via [`MtpQuantization`]: NVFP4 (4-bit),
//! FP8 (8-bit), or BF16 (16-bit). Higher precision improves draft acceptance
//! at the cost of increased MTP forward latency.

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::ForwardContext;
use crate::layers::MoeLayer;
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{
    DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight, quantize_to_fp8, quantize_to_nvfp4,
};

/// Drafter context prefill — **ON by default**, cached once.
///
/// The target prefill captures every position's final-layer hidden and the MTP
/// drafter's KV cache is batch-prefilled over the whole prompt before the first
/// propose(), mirroring vLLM's MTP proposer prefill. The drafter's KV entries
/// are pure functions of its input pair `(embed(token_{i+1}), target_hidden_i)`
/// — a single-layer drafter's K/V do not depend on its own attention outputs —
/// so the prefill needs only the fc + k/v projections + norms + RoPE + cache
/// write, no attention pass.
///
/// Policy, including the kill switch and the coupling to the cross-turn carry
/// (which this half is useless without), lives in
/// `crate::model::drafter_context` — the single source of truth.
pub fn mtp_drafter_prefill_enabled() -> bool {
    crate::model::drafter_context::config().prefill
}

/// Dedicated scratch for the batched drafter prefill (allocated in
/// `MtpHead::new` only when [`mtp_drafter_prefill_enabled`]). All buffers are
/// sized for [`prefill::PREFILL_CHUNK`] rows; dedicated (not aliased onto the
/// shared arena) so the pass has no aliasing hazards against target buffers.
pub(crate) struct MtpPrefillScratch {
    pub embed: DevicePtr,
    pub normed_embed: DevicePtr,
    pub normed_hidden: DevicePtr,
    pub concat: DevicePtr,
    pub fc_out: DevicePtr,
    pub normed2: DevicePtr,
    pub k_out: DevicePtr,
    pub v_out: DevicePtr,
    /// RoPE rotates Q and K in one launch; prefill discards Q, but the kernel
    /// still needs a writable [chunk, nq*hd] region.
    pub q_scratch: DevicePtr,
    /// u32 RoPE positions, one per row.
    pub pos_dev: DevicePtr,
    /// i64 KV slot mapping, one per row.
    pub slot_dev: DevicePtr,
}

/// MTP head weight precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpQuantization {
    /// NVFP4 E2M1 (0.5 bytes/weight) — fastest MTP forward, lowest accuracy.
    Nvfp4,
    /// FP8 E4M3 (1 byte/weight) — balanced.
    Fp8,
    /// BF16 (2 bytes/weight) — highest accuracy, slowest MTP forward.
    Bf16,
}

impl MtpQuantization {
    /// Whether the batched drafter prefill can run at this precision.
    ///
    /// NECESSARY, not sufficient — `prefill::prefill_drafter` remains the
    /// authority and re-checks the actual weight variants and kernel handles.
    /// This predicate exists so the caller can skip the `max_seq_len x hidden`
    /// BF16 prompt-hidden buffer (335 MB at 32k/h=5120, 2.7 GB at 256k) for a
    /// head that could never use it. It is exact by construction:
    /// `quantize_proj` produces `ProjectionWeight::Bf16` for, and only for,
    /// [`MtpQuantization::Bf16`].
    pub fn supports_drafter_prefill(self) -> bool {
        matches!(self, Self::Bf16)
    }
}

impl std::str::FromStr for MtpQuantization {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "nvfp4" | "fp4" => Ok(Self::Nvfp4),
            "fp8" => Ok(Self::Fp8),
            "bf16" => Ok(Self::Bf16),
            _ => anyhow::bail!("Unknown MTP quantization: {s}. Expected: nvfp4, fp8, bf16"),
        }
    }
}

/// Weight storage that can hold any supported precision.
#[allow(dead_code)]
enum ProjectionWeight {
    Nvfp4(QuantizedWeight),
    Fp8(Fp8DenseWeight),
    /// FP8 E4M3 block-scaled from checkpoint (w8a16_gemv LUT kernel).
    /// Used when the checkpoint is FP8 native (native FP8 serving).
    Fp8BlockScaled(Fp8Weight),
    Bf16(DenseWeight),
}

/// Per-sequence MTP proposer state.
pub struct MtpProposerState {
    /// Block table for MTP's own KV cache.
    pub block_table: Vec<u32>,
    /// Current sequence length in MTP's KV cache.
    pub seq_len: usize,
    /// Number of drafts produced in the last propose() call.
    /// Used by after_verify to know how many entries to trim.
    pub last_num_drafted: usize,
    /// Sequence-space pair key of the newest drafter row (a `forward_one`
    /// call with RoPE position `p` writes pair key `p - 1`). `seq_len` alone
    /// cannot locate the drafter in the sequence — without drafter prefill
    /// the row space is compacted (accepted pairs only) and drifts from the
    /// sequence position. `None` until the first row is written.
    pub last_pair_key: Option<usize>,
}

impl ProposerState for MtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// MTP prediction head.
#[allow(dead_code)]
pub struct MtpHead {
    // Norms (always BF16)
    pre_fc_norm_embedding: DenseWeight,
    pre_fc_norm_hidden: DenseWeight,
    input_layernorm: DenseWeight,
    post_attn_layernorm: DenseWeight,
    norm: DenseWeight,

    // Projections (precision depends on MtpQuantization)
    fc: ProjectionWeight,
    q_proj: ProjectionWeight,
    k_proj: ProjectionWeight,
    v_proj: ProjectionWeight,
    o_proj: ProjectionWeight,

    // BF16 fallbacks for Q/K norms
    q_norm: DenseWeight,
    k_norm: DenseWeight,

    // MoE: NVFP4 uses fused MoeLayer; FP8/BF16 uses per-expert storage
    moe_nvfp4: Option<MoeLayer>,
    moe_experts_generic: Option<Vec<(ProjectionWeight, ProjectionWeight, ProjectionWeight)>>,
    moe_shared_generic: Option<(ProjectionWeight, ProjectionWeight, ProjectionWeight)>,
    moe_gate: DenseWeight,
    shared_expert_gate: DenseWeight,

    /// Dense FFN triple `(gate_proj, up_proj, down_proj)` for MTP heads
    /// bundled with dense (non-MoE) checkpoints. When `Some`, the forward
    /// path skips routing/expert dispatch and runs a single MLP. The MoE
    /// fields above are unused/None in that mode.
    dense_ffn_generic: Option<(ProjectionWeight, ProjectionWeight, ProjectionWeight)>,

    // Precision mode
    quant: MtpQuantization,

    /// Reduced vocab size for MTP LM head GEMV (0 = full vocab).
    mtp_vocab_size: u32,

    // Shared weights from target model
    embed_tokens: DenseWeight,
    lm_head_nvfp4: QuantizedWeight,

    // KV cache for MTP attention (1 layer, separate from target)
    kv_cache: Mutex<PagedKvCache>,
    attn_layer_idx: usize,

    // Kernel handles (always needed)
    rms_norm_k: KernelHandle,
    rms_norm_residual_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemv_qg_k: KernelHandle,
    w4a16_gemv_dual_k: KernelHandle,
    rope_k: KernelHandle,
    reshape_cache_k: KernelHandle,
    paged_decode_k: KernelHandle,
    /// MTP KV cache dtype: true = BF16 (matches the main model), false = FP8.
    /// The FP8 path hard-passed k_scale=v_scale=1.0 which collapsed the MTP
    /// attention output to a constant on Qwen3.6-A3B (large deep-layer K/V
    /// magnitudes) → constant draft token 0 → 0% acceptance. BF16 KV (this
    /// head is a single tiny attention layer) fixes it. Gated by mtp_quant.
    kv_bf16: bool,
    residual_add_k: KernelHandle,
    residual_add_rms_norm_k: KernelHandle,
    sigmoid_gate_mul_k: KernelHandle,
    bf16_concat_k: KernelHandle,
    argmax_k: KernelHandle,
    embed_from_argmax_k: KernelHandle,
    /// Fixed device buffer (4 bytes) for deferred draft token ID readback.
    draft_token_id_dev: DevicePtr,
    /// Chain confidence of the last propose (f32 bits; min top-1 softmax
    /// prob across drafts). Written by `forward_one` when
    /// `draft_conf_tau() > 0`; reset to 1.0 at each propose start.
    pub(super) last_conf_bits: std::sync::atomic::AtomicU32,
    // BF16/FP8 kernel handles (None if NVFP4 mode)
    dense_gemv_k: Option<KernelHandle>,
    dense_gemv_fp8w_k: Option<KernelHandle>,
    w8a16_gemv_k: Option<KernelHandle>,
    deinterleave_qg_k: Option<KernelHandle>,
    moe_topk_k: Option<KernelHandle>,
    moe_silu_mul_k: Option<KernelHandle>,
    moe_weighted_sum_blend_k: Option<KernelHandle>,
    /// Batched BF16 GEMM for the drafter-prefill pass (0 when absent).
    dense_gemm_k: KernelHandle,
    /// Drafter-prefill scratch; `None` unless ATLAS_MTP_DRAFTER_PREFILL=1.
    prefill_scratch: Option<MtpPrefillScratch>,
}

impl MtpHead {
    /// Acquire the MTP KV cache mutex. Used by the multi-module
    /// dispatcher (`mtp_multi`) to reclaim blocks during free_state.
    /// `parking_lot::Mutex` does not poison, so this can never fail.
    pub(crate) fn kv_cache_lock(&self) -> parking_lot::MutexGuard<'_, PagedKvCache> {
        self.kv_cache.lock()
    }

    /// Dispatch GEMV to the appropriate kernel based on weight precision.
    fn gemv(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        proj: &ProjectionWeight,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        match proj {
            ProjectionWeight::Nvfp4(w) => {
                ops::w4a16_gemv(gpu, self.w4a16_gemv_k, input, w, output, n, k, stream)
            }
            ProjectionWeight::Fp8(w) => ops::dense_gemv_fp8w(
                gpu,
                self.dense_gemv_fp8w_k.unwrap(),
                input,
                w,
                output,
                n,
                k,
                stream,
            ),
            ProjectionWeight::Fp8BlockScaled(w) => ops::w8a16_gemv(
                gpu,
                self.w8a16_gemv_k.unwrap(),
                input,
                w.weight,
                w.row_scale,
                output,
                n,
                k,
                stream,
            ),
            ProjectionWeight::Bf16(w) => ops::dense_gemv(
                gpu,
                self.dense_gemv_k.unwrap(),
                input,
                w,
                output,
                n,
                k,
                stream,
            ),
        }
    }

    /// Quantize a BF16 weight to the target precision.
    fn quantize_proj(
        bf16: &DenseWeight,
        n: usize,
        k: usize,
        quant: MtpQuantization,
        gpu: &dyn GpuBackend,
        absmax_k: KernelHandle,
        nvfp4_k: KernelHandle,
        fp8_k: KernelHandle,
        stream: u64,
    ) -> Result<ProjectionWeight> {
        match quant {
            MtpQuantization::Nvfp4 => Ok(ProjectionWeight::Nvfp4(quantize_to_nvfp4(
                bf16, n, k, gpu, absmax_k, nvfp4_k, stream,
            )?)),
            MtpQuantization::Fp8 => Ok(ProjectionWeight::Fp8(quantize_to_fp8(
                bf16, n, k, gpu, fp8_k, stream,
            )?)),
            MtpQuantization::Bf16 => Ok(ProjectionWeight::Bf16(*bf16)),
        }
    }
}

mod draft_proposer;
mod forward;
mod moe_forward;
mod new;
mod prefill;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_proposer_state_downcast() {
        let state: Box<dyn ProposerState> = Box::new(MtpProposerState {
            block_table: vec![0, 1, 2],
            seq_len: 42,
            last_num_drafted: 0,
            last_pair_key: None,
        });
        let mtp = state.as_any().downcast_ref::<MtpProposerState>().unwrap();
        assert_eq!(mtp.seq_len, 42);
        assert_eq!(mtp.block_table.len(), 3);
    }
}

/// How many drafter KV rows `after_verify` must drop.
///
/// * Rejected rows always go: `num_drafted - num_accepted`.
/// * With `refeed_accepted` (ATLAS_MTP_REFEED_ACCEPTED), the ACCEPTED rows
///   that were written with the drafter's own hidden also go — that is every
///   accepted draft except the first. Draft 1 consumed the target's verified
///   hidden (`mtp_hidden_save`) and is correct; drafts 2.. each consumed the
///   previous draft's drafter-side residual. Those rows are rebuilt from the
///   catch-up ring on the next propose, with the target's true hidden.
///
/// Never returns more than `num_drafted` — the drafter cannot un-write rows it
/// never wrote, and over-trimming would corrupt the compacted row space by
/// desynchronising `seq_len` from `last_pair_key`.
pub(crate) fn mtp_rows_to_trim(
    num_drafted: usize,
    num_accepted: usize,
    refeed_accepted: bool,
) -> usize {
    let rejected = num_drafted.saturating_sub(num_accepted);
    let accepted_with_drafter_hidden = if refeed_accepted {
        num_accepted.saturating_sub(1)
    } else {
        0
    };
    (rejected + accepted_with_drafter_hidden).min(num_drafted)
}

#[cfg(test)]
mod refeed_trim_tests {
    use super::mtp_rows_to_trim;

    #[test]
    fn flag_off_is_exactly_the_legacy_behaviour() {
        // Legacy: trim only the rejected rows. These are the K=2/3/4 cases
        // the schedulers actually produce.
        assert_eq!(mtp_rows_to_trim(1, 0, false), 1); // K=2 reject
        assert_eq!(mtp_rows_to_trim(1, 1, false), 0); // K=2 accept
        assert_eq!(mtp_rows_to_trim(2, 0, false), 2); // K=3 reject
        assert_eq!(mtp_rows_to_trim(2, 1, false), 1); // K=3 accept-1
        assert_eq!(mtp_rows_to_trim(2, 2, false), 0); // K=3 accept-2
        assert_eq!(mtp_rows_to_trim(3, 3, false), 0); // K=4 accept-3
    }

    #[test]
    fn flag_on_also_drops_accepted_rows_past_the_first() {
        // The first accepted draft used the TARGET hidden — it stays.
        assert_eq!(mtp_rows_to_trim(1, 1, true), 0); // K=2 accept: nothing extra
        assert_eq!(mtp_rows_to_trim(2, 1, true), 1); // K=3 accept-1: rejected only
        assert_eq!(mtp_rows_to_trim(2, 2, true), 1); // K=3 accept-2: drop draft 2
        assert_eq!(mtp_rows_to_trim(3, 2, true), 2); // K=4 accept-2: 1 rejected + 1
        assert_eq!(mtp_rows_to_trim(3, 3, true), 2); // K=4 accept-3: drop drafts 2,3
    }

    #[test]
    fn full_reject_is_identical_with_and_without_the_flag() {
        // Nothing was accepted, so there is no drafter-hidden row to rebuild.
        for d in 0..8 {
            assert_eq!(mtp_rows_to_trim(d, 0, true), mtp_rows_to_trim(d, 0, false));
        }
    }

    #[test]
    fn never_trims_more_rows_than_were_drafted() {
        for d in 0..8 {
            for a in 0..=d + 2 {
                assert!(mtp_rows_to_trim(d, a, true) <= d, "d={d} a={a}");
                assert!(mtp_rows_to_trim(d, a, false) <= d, "d={d} a={a}");
            }
        }
    }
}
