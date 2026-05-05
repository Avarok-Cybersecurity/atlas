// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H standalone MoE FFN layer.
//!
//! Supports two variants:
//!   - **Nano 30B**: Direct MoE — experts operate on full hidden_size.
//!   - **Super 120B**: LatentMoE — routed experts operate in latent space `[moe_latent_size]`,
//!     with fc1/fc2 latent projections bridging hidden↔latent.
//!
//! Forward: RMS norm → gate → sigmoid topK routing → (fc1_latent if latent) →
//!          batched up GEMV → fused relu²+down → weighted_sum → (fc2_latent if latent) →
//!          shared expert up+relu²+down → sum routed+shared → residual add.
//!
//! All expert dispatch is device-side (pointer tables) — zero D2H sync.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use crate::layers::ops;
use crate::weight_map::{DenseWeight, NemotronExpertWeight, NemotronMoeWeights, QuantizedWeight};

/// Device-side pointer table for one projection across all experts.
struct ExpertPtrTable {
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
}

/// Nemotron-H standalone MoE FFN layer.
pub struct NemotronMoeLayer {
    weights: NemotronMoeWeights,
    input_norm: DenseWeight,
    /// LatentMoE dimension (0 = direct, >0 = latent).
    moe_latent_size: usize,
    // Kernel handles — decode (single token)
    rms_norm_residual_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    topk_sigmoid_k: KernelHandle,
    moe_expert_gemv_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    relu2_down_shared_k: KernelHandle,
    weighted_sum_scale_k: KernelHandle,
    residual_add_k: KernelHandle,
    // Kernel handles — prefill (batched GEMM)
    dense_gemm_k: KernelHandle,
    w4a16_gemm_k: KernelHandle,
    // Batched N-token MoE prefill kernels
    topk_sigmoid_batched_k: KernelHandle,
    moe_up_prefill_k: KernelHandle,
    moe_relu2_down_prefill_k: KernelHandle,
    moe_weighted_sum_prefill_k: KernelHandle,
    // Sorted grouped GEMM (Qwen pattern — proven to work)
    moe_sort_k: KernelHandle,
    moe_grouped_gemm_k: KernelHandle,
    moe_relu2_elementwise_k: KernelHandle,
    moe_unpermute_reduce_k: KernelHandle,
    moe_grouped_gemm_n128_k: KernelHandle,
    up_ptrs: ExpertPtrTable,
    down_ptrs: ExpertPtrTable,
    // Transposed expert pointer tables (for N128 grouped GEMM)
    up_ptrs_t: Option<ExpertPtrTable>,
    down_ptrs_t: Option<ExpertPtrTable>,
    // Transposed shared expert weights
    shared_up_t: Option<QuantizedWeight>,
    shared_down_t: Option<QuantizedWeight>,
    // Transposed SSM GEMM kernel handle (for shared expert)
    w4a16_gemm_t_k: KernelHandle,
}

impl NemotronMoeLayer {
    pub fn new(
        weights: NemotronMoeWeights,
        input_norm: DenseWeight,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let up_ptrs = build_ptr_table(&weights.experts, |e| &e.up_proj, gpu)?;
        let down_ptrs = build_ptr_table(&weights.experts, |e| &e.down_proj, gpu)?;

        Ok(Self {
            weights,
            input_norm,
            moe_latent_size: config.moe_latent_size,
            rms_norm_residual_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "rms_norm_residual_f32")
                    .or_else(|_| gpu.kernel("norm", "rms_norm_residual"))?
            } else {
                gpu.kernel("norm", "rms_norm_residual")?
            },
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            topk_sigmoid_k: gpu.kernel("moe_topk_sig", "moe_topk_sigmoid")?,
            moe_expert_gemv_k: gpu.kernel("moe_expert_gemv", "moe_expert_gemv")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            relu2_down_shared_k: gpu.kernel("moe_relu2_fused", "moe_expert_relu2_down_shared")?,
            weighted_sum_scale_k: gpu.kernel("relu2", "moe_weighted_sum_scale")?,
            residual_add_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "f32_residual_add")
                    .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?
            } else {
                gpu.kernel("residual_add", "bf16_residual_add")?
            },
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            topk_sigmoid_batched_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_topk_sigmoid_batched",
            ),
            moe_up_prefill_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_up_prefill",
            ),
            moe_relu2_down_prefill_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_relu2_down_prefill",
            ),
            moe_weighted_sum_prefill_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_weighted_sum_prefill",
            ),
            moe_sort_k: super::try_kernel(gpu, "moe", "moe_sort_by_expert"),
            moe_grouped_gemm_k: super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable",
            ),
            moe_relu2_elementwise_k: super::try_kernel(gpu, "relu2", "relu_squared_inplace"),
            moe_unpermute_reduce_k: super::try_kernel(gpu, "moe", "moe_unpermute_reduce_indexed"),
            moe_grouped_gemm_n128_k: super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_t",
            ),
            up_ptrs,
            down_ptrs,
            up_ptrs_t: None,
            down_ptrs_t: None,
            shared_up_t: None,
            shared_down_t: None,
            w4a16_gemm_t_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t"),
        })
    }
}

impl NemotronMoeLayer {
    /// Transpose expert weights for fast grouped GEMM prefill.
    /// Called from weight loader after construction. Skips expert transposition
    /// when memory is tight (Super 120B: 128 experts × 40 layers would OOM).
    pub fn prepare_prefill_weights(&mut self, gpu: &dyn GpuBackend, config: &ModelConfig) {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;

        // Only transpose routed experts for small models (Nano 30B: 23 MoE layers × 128 experts).
        // Super 120B has 40 MoE layers × 128 experts = 5120 matrices — too much memory.
        // The sorted grouped GEMM still works with non-transposed weights via the base kernel.
        if self.moe_latent_size == 0 {
            let expert_k = h;
            let mut up_t = Vec::new();
            let mut down_t = Vec::new();
            for expert in &self.weights.experts {
                if let Ok(ut) = expert.up_proj.transpose_for_gemm(gpu, inter, expert_k) {
                    up_t.push(ut);
                }
                if let Ok(dt) = expert.down_proj.transpose_for_gemm(gpu, expert_k, inter) {
                    down_t.push(dt);
                }
            }
            if up_t.len() == self.weights.experts.len()
                && let Ok(ptrs) = build_ptr_table_from_weights(&up_t, gpu)
            {
                self.up_ptrs_t = Some(ptrs);
            }
            if down_t.len() == self.weights.experts.len()
                && let Ok(ptrs) = build_ptr_table_from_weights(&down_t, gpu)
            {
                self.down_ptrs_t = Some(ptrs);
            }
        }

        // Transpose shared expert weights (only for direct MoE — Super is too memory-tight)
        if self.moe_latent_size == 0 {
            self.shared_up_t = self
                .weights
                .shared_up
                .transpose_for_gemm(gpu, shared_inter, h)
                .ok();
            self.shared_down_t = self
                .weights
                .shared_down
                .transpose_for_gemm(gpu, h, shared_inter)
                .ok();
        }
    }
}

mod decode_helpers;
mod prefill_fallback;
mod prefill_sorted;

use prefill_sorted::SortedPrefillCtx;

impl TransformerLayer for NemotronMoeLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_inner(hidden, residual, ctx, stream)
    }

    /// Batched MoE prefill: uses GEMM for gate/fc1/fc2/shared, per-token for routing + experts.
    ///
    /// For Super 120B with 40 MoE layers, this replaces O(N * 7 kernel_launches) decode calls
    /// with O(4 GEMMs + N * 3 kernel_launches), cutting TTFT by 30-50%.
    #[allow(clippy::overly_complex_bool_expr)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let scale = ctx.config.routed_scaling_factor as f32;
        let n = num_tokens as u32;

        // ── 1. Batched RMS norm: [N, H] → normed[N, H] + residual update ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // ── 2. Batched Gate GEMM: [N, H] x [H, num_experts]^T → [N, num_experts] ──
        let gate_logits = ctx.buffers.gate_logits();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &self.weights.gate,
            gate_logits,
            n,
            num_experts,
            h as u32,
            stream,
        )?;

        // Check if batched MoE prefill kernels are available
        let has_batched = self.topk_sigmoid_batched_k.0 != 0
            && self.moe_up_prefill_k.0 != 0
            && self.moe_relu2_down_prefill_k.0 != 0
            && self.moe_weighted_sum_prefill_k.0 != 0;

        // ── 3. Shared expert UP ──
        // When batched MoE prefill is available, the shared expert UP is handled
        // inside the batched UP kernel (step 5b). We only pre-compute here for
        // the per-token fallback path or LatentMoE.
        let shared_up_out_base = ctx.buffers.ssm_qkvz();
        let use_batched_moe = has_batched && num_tokens > 1;
        // Always compute shared expert UP — even when batched path overwrites it later.
        // The batched UP kernel writes shared_up_out for shared blocks, but we need
        // this result for the per-token fallback path AND it's harmless to overwrite.
        if let Some(ref sut) = self.shared_up_t {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t_k,
                normed,
                sut,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                &self.weights.shared_up,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        }

        // ── 4. LatentMoE: batched fc1_latent GEMM [N, H] → [N, L] ──
        // Use attn_output as temp buffer (m*max_dim*2, large enough for [N, L]).
        // Cannot use ssm_ba (too small) or moe_output (used later for unpermute).
        let latent = self.moe_latent_size as u32;
        let latent_base = if latent > 0 {
            let fc1 = self.weights.fc1_latent_proj.as_ref().unwrap();
            let latent_buf = ctx.buffers.attn_output();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                fc1,
                latent_buf,
                n,
                latent,
                h as u32,
                stream,
            )?;
            Some(latent_buf)
        } else {
            None
        };

        // ── 5. Batched routing + expert dispatch (N tokens, 4 kernel launches) ──
        // When batched prefill kernels are available, replace the per-token loop
        // (N × 5 launches = 10k+ launches) with 4 batched launches.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(n as usize * top_k as usize * 4);

        // Sorted MoE prefill: sort tokens by expert, then grouped GEMM.
        // This is the proven Qwen pattern — avoids the crashing batched UP/DOWN kernels.
        let use_sorted = use_batched_moe
            && self.moe_sort_k.0 != 0
            && self.moe_grouped_gemm_k.0 != 0
            && self.moe_unpermute_reduce_k.0 != 0;

        let p = SortedPrefillCtx {
            n,
            num_tokens,
            h,
            inter,
            shared_inter,
            num_experts,
            top_k,
            scale,
            latent,
            gate_logits,
            indices_dev,
            weights_dev,
            normed,
            hidden,
            latent_base,
            shared_up_out_base,
        };
        if use_sorted {
            self.prefill_sorted_path(&p, ctx, stream)?;
        } else {
            self.prefill_fallback_path(&p, ctx, stream)?;
        }

        Ok(())
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }
}

fn build_ptr_table_from_weights(
    weights: &[QuantizedWeight],
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = weights.len();
    let packed_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale_2.to_le_bytes())
        .collect();
    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;
    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;
    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;
    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

fn build_ptr_table(
    experts: &[NemotronExpertWeight],
    proj: impl Fn(&NemotronExpertWeight) -> &QuantizedWeight,
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = experts.len();

    let packed_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale_2.to_le_bytes())
        .collect();

    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;

    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;

    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}
