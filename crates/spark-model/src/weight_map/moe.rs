// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// MoE layer weights.
pub struct MoeWeights {
    /// Router gate: [hidden_size, num_experts] BF16.
    pub gate: DenseWeight,
    /// Shared expert (always active).
    pub shared_expert: ExpertWeight,
    /// Shared expert gate sigmoid weight: `[1]` BF16.
    pub shared_expert_gate: DenseWeight,
    /// Per-expert weights: 512 experts.
    pub experts: Vec<ExpertWeight>,
    /// Optional router pre-normalization weight.
    /// Set for Gemma-4 MoE where the HF reference applies a pure RMSNorm to
    /// the router input followed by a per-dim scale multiplication:
    ///   `router_input = rms_norm(x) * scale * hidden_size^(-0.5)`
    /// Stored as a BF16 `[hidden_size]` vector containing `scale * root_size`
    /// so the existing rms_norm kernel (`output = x/rms(x) * weight`) applies
    /// both steps in one pass. `None` for models that feed the router from
    /// the raw post-attention residual.
    pub router_pre_norm: Option<DenseWeight>,
    /// Optional expert correction bias: `[num_experts]` F32.
    ///
    /// Set for models using the DeepSeek-V3 / MiniMax-M2 loss-free-balancing
    /// routing trick: the bias is added to sigmoid(gate_logits) *only* for
    /// top-k selection; gathered dispatch weights come from the unbiased
    /// sigmoid scores. Consumed by `moe_topk_sigmoid` kernel via its `bias`
    /// argument.
    ///
    /// `None` for softmax-routed Qwen/Gemma MoE. Nemotron-H carries its own
    /// bias in `NemotronMoeWeights::e_score_correction_bias` because its
    /// MoE is a separate layer type (Mamba-2 interleaved) — those paths
    /// don't touch this struct.
    pub correction_bias: Option<DenseWeight>,
}

impl MoeWeights {
    /// Create empty MoeWeights for testing (all null pointers).
    #[cfg(test)]
    pub fn empty(num_experts: usize) -> Self {
        let null_dense = DenseWeight {
            weight: DevicePtr::NULL,
        };
        let null_quant = QuantizedWeight {
            weight: DevicePtr::NULL,
            weight_scale: DevicePtr::NULL,
            weight_scale_2: 1.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        };
        let null_expert = ExpertWeight {
            gate_proj: null_quant,
            up_proj: null_quant,
            down_proj: null_quant,
        };
        Self {
            gate: null_dense,
            shared_expert: null_expert,
            shared_expert_gate: null_dense,
            experts: vec![null_expert; num_experts],
            router_pre_norm: None,
            correction_bias: None,
        }
    }
}

/// All weights for one transformer layer.
pub enum LayerWeights {
    FullAttention {
        input_norm: DenseWeight,
        attn: AttentionWeights,
        post_attn_norm: DenseWeight,
        moe: MoeWeights,
    },
    LinearAttention {
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        moe: MoeWeights,
    },
}

/// BF16 expert weight (before NVFP4 quantization).
#[derive(Debug, Clone, Copy)]
pub struct DenseExpertWeight {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// MTP (Multi-Token Prediction) head weights (all BF16 from safetensors).
///
/// Single decoder layer + concat projection. All projection weights are BF16
/// and get quantized to NVFP4 at load time by the weight loader.
pub struct MtpWeights {
    /// RMSNorm on token embedding before concat: `[hidden_size]` BF16.
    pub pre_fc_norm_embedding: DenseWeight,
    /// RMSNorm on target hidden state before concat: `[hidden_size]` BF16.
    pub pre_fc_norm_hidden: DenseWeight,
    /// Concat projection: `[hidden_size, 2*hidden_size]` BF16.
    pub fc: DenseWeight,
    /// Input layernorm for the attention layer: `[hidden_size]` BF16.
    pub input_layernorm: DenseWeight,
    /// Attention projections (all BF16).
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    /// Post-attention layernorm: `[hidden_size]` BF16.
    pub post_attn_layernorm: DenseWeight,
    /// MoE router gate: [num_experts, hidden_size] BF16.
    /// NULL when `dense_ffn` is `Some` (dense FFN MTP head).
    pub moe_gate: DenseWeight,
    /// Shared expert (BF16). NULL fields when `dense_ffn` is `Some`.
    pub shared_expert: DenseExpertWeight,
    /// Shared expert gate: [1, hidden_size] BF16.
    /// NULL when `dense_ffn` is `Some`.
    pub shared_expert_gate: DenseWeight,
    /// Per-expert weights (512 experts, BF16).
    /// Empty when `dense_ffn` is `Some`.
    pub experts: Vec<DenseExpertWeight>,
    /// Dense FFN triple (`gate_proj`, `up_proj`, `down_proj`) — used by MTP
    /// heads bundled with dense (non-MoE) FP8 checkpoints, e.g.
    /// `Qwen/Qwen3.6-27B-FP8`. When `Some`, the MoE fields above are unused
    /// and the forward path takes the dense MLP shortcut.
    pub dense_ffn: Option<DenseExpertWeight>,
    /// Final output RMSNorm: `[hidden_size]` BF16.
    pub norm: DenseWeight,
    /// The routed + shared experts as the checkpoint's own FP8 block-scaled
    /// tables, when the MTP head ships them that way (Qwen3.6-35B-A3B-FP8:
    /// `mtp.layers.0.mlp.experts.{e}.*_proj.{weight,weight_scale_inv}`, the
    /// per-expert layout the native-FP8 main layers consume). `experts` /
    /// `shared_expert` above are then fresh BF16 dequants of the SAME tensors,
    /// kept for the NVFP4 re-quantization path; a BF16/FP8 head builds its
    /// MoE from these tables instead and calls
    /// [`Self::release_bf16_expert_dequants`]. `None` when the experts are
    /// BF16 on disk (or stacked), in which case there is nothing to release.
    pub fp8_experts: Option<MtpFp8Experts>,
}

/// Native FP8 block-scaled MTP MoE tables: `experts[e]` is routed expert `e`
/// (all `num_experts` present — the drafter is never expert-parallel), plus
/// the shared expert. FP8 bytes are the weight store's; only the widened FP32
/// block scales are owned allocations.
pub struct MtpFp8Experts {
    pub experts: Vec<Fp8ExpertWeight>,
    pub shared_expert: Fp8ExpertWeight,
}

impl MtpWeights {
    /// Free the BF16 dequants of the routed and shared experts once an FP8
    /// `MoeLayer` serves them. Only legal when [`Self::fp8_experts`] is
    /// `Some`: the loader attaches the tables exactly when the on-disk experts
    /// are FP8, and in that case every entry of `experts` / `shared_expert`
    /// is a fresh `dequant_fp8_blockscaled_to_bf16` allocation (never a store
    /// pointer, never a slice of a stacked tensor). Leaves the vectors empty so
    /// a later reader cannot touch the released memory.
    pub fn release_bf16_expert_dequants(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        ensure!(
            self.fp8_experts.is_some(),
            "release_bf16_expert_dequants: no FP8 tables — the BF16 experts are the only copy"
        );
        let mut released = 0usize;
        for de in self.experts.drain(..) {
            for w in [de.gate_proj, de.up_proj, de.down_proj] {
                gpu.free(w.weight)?;
                released += 1;
            }
        }
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };
        let shared = std::mem::replace(
            &mut self.shared_expert,
            DenseExpertWeight {
                gate_proj: null,
                up_proj: null,
                down_proj: null,
            },
        );
        for w in [shared.gate_proj, shared.up_proj, shared.down_proj] {
            if !w.weight.is_null() {
                gpu.free(w.weight)?;
                released += 1;
            }
        }
        tracing::info!(
            "MTP head: released {released} BF16 expert dequants (native FP8 tables serve the MoE)"
        );
        Ok(())
    }
}
