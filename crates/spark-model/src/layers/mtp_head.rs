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

/// ATLAS_MTP_DRAFTER_PREFILL=1 — opt-in drafter context prefill (cached once).
///
/// When set, the target prefill captures every position's final-layer hidden
/// and the MTP drafter's KV cache is batch-prefilled over the whole prompt
/// before the first propose(), mirroring vLLM's MTP proposer prefill. The
/// drafter's KV entries are pure functions of its input pair
/// `(embed(token_{i+1}), target_hidden_i)` — a single-layer drafter's K/V do
/// not depend on its own attention outputs — so the prefill needs only the
/// fc + k/v projections + norms + RoPE + cache write, no attention pass.
pub fn mtp_drafter_prefill_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MTP_DRAFTER_PREFILL").ok().as_deref() == Some("1"))
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

mod forward;
mod moe_forward;
mod new;
mod prefill;

impl DraftProposer for MtpHead {
    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(MtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            last_pair_key: None,
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;

        // Reset chain confidence for this propose (forward_one mins into it).
        self.last_conf_bits
            .store(1.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;

        for i in 0..num_drafts {
            // Only the LAST draft gets GPU-side embedding (it's the one
            // used in the next verify step).
            let embed_target = if i == num_drafts - 1 {
                draft_embed_target
            } else {
                None
            };
            // Grammar-masked drafting (num_drafts==1 path only for now).
            // For num_drafts > 1 we would need to speculatively advance the
            // matcher between drafts and roll back before returning; the
            // current scheduler only uses num_drafts==1, so we pass the same
            // mask for every i and warn loudly if K>1 + grammar combine.
            if grammar_bitmask.is_some() && i > 0 {
                tracing::warn!(
                    "MTP grammar-masked drafting called with num_drafts>1 (i={i}); \
                     mask held fixed across draft positions — acceptance may drop."
                );
            }
            let mask_for_draft = grammar_bitmask;
            let draft = self.forward_one(
                current_token,
                current_hidden,
                position + i,
                mtp_state,
                ctx,
                stream,
                embed_target,
                mask_for_draft,
            )?;
            tracing::debug!(
                "MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} → draft={draft}",
                position + i,
                mtp_state.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
            // For subsequent drafts, use the MTP head's own hidden state
            current_hidden = ctx.buffers.hidden_states();
        }

        mtp_state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.prefill_drafter_impl(prompt_tokens, hiddens, state, ctx, stream)
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .map(|s| s.seq_len)
            .unwrap_or(0)
    }

    fn last_pair_key(&self, state: &mut dyn ProposerState) -> Option<usize> {
        state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .and_then(|s| s.last_pair_key)
    }

    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.drafter_rows_impl(tokens, hiddens, row_base, pos_base, state, ctx, stream)
    }

    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        self.read_deferred_draft_token(gpu)
    }

    fn last_confidence(&self) -> Option<f32> {
        if crate::speculative::draft_conf_tau() <= 0.0 {
            return None;
        }
        Some(f32::from_bits(
            self.last_conf_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        ))
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;

        // Trim rejected drafts from MTP KV cache.
        // num_drafted was recorded in the last propose() call.
        // We trim `num_drafted - num_accepted` entries.
        // e.g. K=2: drafted 1, accepted 0 → trim 1. accepted 1 → trim 0.
        // e.g. K=3: drafted 2, accepted 0 → trim 2. accepted 1 → trim 1. accepted 2 → trim 0.
        let num_drafted = mtp_state.last_num_drafted.max(1);
        let num_to_trim = num_drafted.saturating_sub(num_accepted);
        let old_sl = mtp_state.seq_len;
        if num_to_trim > 0 {
            mtp_state.seq_len = mtp_state.seq_len.saturating_sub(num_to_trim);
            // Trimmed rows have consecutive pair keys; the newest surviving
            // key moves back by the same count.
            if let Some(k) = mtp_state.last_pair_key {
                mtp_state.last_pair_key = Some(k.saturating_sub(num_to_trim));
            }
        }
        tracing::debug!(
            "MTP after_verify: accepted={num_accepted} drafted={num_drafted} trim={num_to_trim} mtp_seq_len: {old_sl} → {}",
            mtp_state.seq_len,
        );
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;
        if !mtp_state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&mtp_state.block_table);
            mtp_state.block_table.clear();
        }
        mtp_state.seq_len = 0;
        Ok(())
    }
}

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
