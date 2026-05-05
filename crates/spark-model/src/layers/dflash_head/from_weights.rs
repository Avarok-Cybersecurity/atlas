// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter constructor + target-config validation.
//!
//! Split out of `dflash_head.rs` for file-size budget. Contains
//! [`BlockDiffusionDraftHead::from_weights`] (kernel resolution + KV
//! cache setup) and [`BlockDiffusionDraftHead::validate_against_target`].

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use super::{
    BlockDiffusionDraftHead, DflashKernels, DflashLayer, DflashQuantization, DflashScratch,
};
use crate::weight_loader::DflashWeights;

impl BlockDiffusionDraftHead {
    pub fn from_weights(
        weights: DflashWeights,
        embed_tokens_shared: DevicePtr,
        lm_head_shared: DevicePtr,
        target_hidden_size: usize,
        gamma: Option<usize>,
        window_size: Option<usize>,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
    ) -> Result<Self> {
        // Drafter's `fc` is `[draft_hidden, len(target_layer_ids) * target_hidden]`.
        // We rely on the drafter config's `hidden_size` and the parsed
        // `target_layer_ids` to derive the expected target_hidden, then
        // validate it matches what the caller provided.
        let target_layer_ids = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.clone())
            .unwrap_or_default();
        let mask_token_id = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.mask_token_id)
            .unwrap_or(0);

        if target_layer_ids.is_empty() {
            anyhow::bail!(
                "DFlash drafter config.json has no `dflash_config.target_layer_ids` — \
                 cannot determine which target hidden states to capture"
            );
        }

        let _ = target_hidden_size;

        let num_layers = weights.config.num_hidden_layers;
        let hidden_size = weights.config.hidden_size;
        let intermediate_size = weights.config.intermediate_size;
        let num_q_heads = weights.config.num_attention_heads;
        let num_kv_heads = weights.config.num_key_value_heads;
        let head_dim = weights.config.head_dim;
        let vocab_size = weights.config.vocab_size;
        let gamma_val = gamma.unwrap_or(weights.config.block_size);

        // Allocate the drafter's paged FP8 KV cache. One multi-layer cache,
        // sized for `max_seq_len + γ + 1` positions (prompt + γ drafts +
        // 1 bonus). Block size 16 matches the rest of Atlas.
        let block_size = 16;
        let kv_config = KvCacheConfig {
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let num_blocks = (max_seq_len + gamma_val + 1) / block_size + 1;
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;

        // Resolve kernel handles. All BF16 paths since drafter weights are
        // BF16 (DflashQuantization::Bf16); FP8 cache uses the FP8 reshape +
        // FP8-aware paged-attention kernel. Module/function names verified
        // against existing Atlas resolutions in `qwen3_attention/mod.rs` and
        // `mtp_head.rs` plus the `extern "C" __global__` declarations under
        // `kernels/gb10/nvfp4/`.
        let kernels = DflashKernels {
            // DFlash drafter uses HF's vanilla RMSNorm convention
            // (`out = x * w / RMS(x)`), NOT Atlas's default offset-from-1
            // form (`out = x * (1 + w) / RMS(x)`). Atlas's standard
            // `rms_norm` kernel includes the `+1` for Qwen3-Next-style
            // checkpoints; we must use `rms_norm_vanilla` for the drafter
            // to match the drafter's HF-trained weights exactly.
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            // `rms_norm_residual` lands the post-attn / post-MLP add+norm in
            // a single launch. Atlas exposes this as a separate kernel — see
            // `mtp_head.rs:469` for the established lookup.
            residual_rms_norm: gpu
                .kernel("norm", "rms_norm_residual")
                .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            // Qwen3.6-DFlash uses yarn RoPE — confirmed in the drafter
            // `config.json:rope_scaling.rope_type="yarn"`. Atlas's yarn
            // kernel is `rope::rope_forward_yarn`.
            rope_qwen3: gpu.kernel("rope", "rope_forward_yarn")?,
            // FP8 KV cache writeback. Module name is the .cu stem
            // `reshape_and_cache`, function is `reshape_and_cache_flash_fp8`
            // (qwen3_attention/mod.rs:377-378 uses the same path).
            reshape_cache_fp8: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")?,
            // The Phase-2 γ-block kernel — same module as the existing
            // FP8 paged-prefill kernel (we just pass `causal_mask_enabled=0`
            // via a different dispatcher).
            prefill_attn_dflash_fp8: gpu
                .kernel("prefill_paged_fp8", "inferspark_prefill_paged_fp8")?,
            silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            argmax: gpu.kernel("argmax", "argmax_bf16")?,
            batched_embed: gpu.kernel("embed_from_argmax", "batched_embed")?,
            // Drafter has head_dim=128, but qwen3.6-35b-a3b target's
            // `inferspark_prefill` is compiled with HDIM=256. Using that
            // kernel produces corrupted attn_out for the drafter (kernel
            // reads 256 elements per head when only 128 are valid →
            // garbage in the back half of SMEM tiles → per-head sign-flip
            // pattern across q-heads). The HDIM=128 specialization
            // `inferspark_prefill_h128` lives in the qwen3.6-35b-a3b
            // model-override kernel dir.
            prefill_attn: gpu.kernel("inferspark_prefill_h128", "inferspark_prefill_h128")?,
        };

        // Per-step scratch buffers. BF16 = 2 bytes/element.
        //
        // Sized for `n_attn_slots = ctx_window + γ` rows in the attention
        // path. The first `ctx_window` slots hold projected target ctx
        // (K/V only — Q is zero-padded so its attention output is
        // discarded). The next γ slots hold the noise tokens. lm_head +
        // logits + argmax tail still operates on γ rows (offset past ctx).
        let bf16 = 2usize;
        let g = gamma_val;
        // Phase 2.5n: ctx_window controls how many captured target positions
        // the drafter attends to per step. The drafter was trained over the
        // FULL captured prefix (paper §A.1), but capping at γ=16 cripples it
        // on prompts past a tiny window — Atlas's 6-10% acceptance vs the
        // paper's 70% is dominated by this cap. Default raised to 512;
        // ATLAS_DFLASH_CTX_WINDOW overrides at construction time.
        //
        // Memory cost: scratch buffers scale linearly with `n_attn = γ + cw`.
        // At cw=512: stream_buf = 528 × 2048 × 2 = 2.1 MB; mlp_intermediate
        // = 528 × 6144 × 2 = 6.3 MB; logits = 528 × 248320 × 2 = 257 MB
        // (largest). Total scratch ~280 MB per head — affordable.
        let ctx_window: usize = std::env::var("ATLAS_DFLASH_CTX_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        tracing::info!(
            "DFlash ctx_window = {} (set ATLAS_DFLASH_CTX_WINDOW to override; \
             drafter trained on full captured prefix — larger is better, \
             scratch grows linearly)",
            ctx_window
        );
        let n_attn = g + ctx_window; // total attention slots
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let scratch = DflashScratch {
            stream_buf: gpu.alloc(n_attn * hidden_size * bf16)?,
            norm_buf: gpu.alloc(n_attn * hidden_size * bf16)?,
            q_buf: gpu.alloc(n_attn * q_dim * bf16)?,
            k_buf: gpu.alloc(n_attn * kv_dim * bf16)?,
            v_buf: gpu.alloc(n_attn * kv_dim * bf16)?,
            attn_out: gpu.alloc(n_attn * q_dim * bf16)?,
            mlp_intermediate: gpu.alloc(n_attn * intermediate_size * bf16)?,
            mlp_up: gpu.alloc(n_attn * intermediate_size * bf16)?,
            stream_acc: gpu.alloc(n_attn * hidden_size * bf16)?,
            fc_proj: gpu.alloc(ctx_window * hidden_size * bf16)?,
            logits: gpu.alloc(n_attn * vocab_size * bf16)?,
            draft_tokens_dev: gpu.alloc(n_attn * 4)?,
            position_ids: gpu.alloc(n_attn * 4)?,
        };

        // Pre-compute yarn inv_freq table for drafter RoPE. Mirrors the
        // formula in `mistral_loader.rs:518-577`. Drafter config:
        //   factor = 64.0, beta_fast = 32.0, beta_slow = 1.0,
        //   original_max_position_embeddings = 4096
        let rope_theta = 10_000_000.0f32; // drafter rope_theta
        let rotary_dim = head_dim; // Qwen3.6-DFlash applies rope to full head_dim
        let factor = 64.0f32;
        let beta_fast = 32.0f32;
        let beta_slow = 1.0f32;
        let orig_max_pos = 4096.0f32;
        let dim_f = rotary_dim as f32;
        let n_pairs = rotary_dim / 2;
        let find_correction_dim = |num_rot: f32| -> f32 {
            (dim_f * (orig_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
                / (2.0 * rope_theta.ln())
        };
        let low = find_correction_dim(beta_fast).floor().max(0.0);
        let high = find_correction_dim(beta_slow)
            .ceil()
            .min((rotary_dim - 1) as f32);
        let ramp_denom = if (high - low).abs() < 1e-6 {
            high - low + 0.001
        } else {
            high - low
        };
        let mut inv_freq_table = vec![0.0f32; n_pairs];
        for j in 0..n_pairs {
            let pos_freq = rope_theta.powf((2 * j) as f32 / dim_f);
            let inv_freq_extrap = 1.0 / pos_freq;
            let inv_freq_interp = 1.0 / (factor * pos_freq);
            let ramp = ((j as f32 - low) / ramp_denom).clamp(0.0, 1.0);
            let extrap_factor = 1.0 - ramp;
            inv_freq_table[j] =
                inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor;
        }
        let inv_freq_bytes: Vec<u8> = inv_freq_table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let yarn_inv_freq = gpu.alloc(inv_freq_bytes.len())?;
        gpu.copy_h2d(&inv_freq_bytes, yarn_inv_freq)?;
        tracing::info!(
            "DFlash yarn inv_freq: {} pairs, factor={factor}, beta_fast={beta_fast}, \
             beta_slow={beta_slow}, max_pos={orig_max_pos}, low_dim={low:.1}, high_dim={high:.1}",
            n_pairs,
        );

        let head = Self {
            num_layers,
            hidden_size,
            intermediate_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            draft_vocab_size: weights.config.draft_vocab_size.unwrap_or(vocab_size),
            gamma: gamma_val,
            mask_token_id,
            window_size,
            target_layer_ids,
            target_hidden_size,

            embed_tokens_shared,
            lm_head_shared,
            hidden_norm: weights.hidden_norm,
            norm: weights.norm,
            fc: weights.fc,
            draft_id_to_target_id: None,
            layers: weights
                .layers
                .into_iter()
                .map(|l| DflashLayer {
                    input_layernorm: l.input_layernorm,
                    post_attention_layernorm: l.post_attention_layernorm,
                    q_proj: l.q_proj,
                    k_proj: l.k_proj,
                    v_proj: l.v_proj,
                    o_proj: l.o_proj,
                    q_norm: l.q_norm,
                    k_norm: l.k_norm,
                    gate_proj: l.gate_proj,
                    up_proj: l.up_proj,
                    down_proj: l.down_proj,
                })
                .collect(),
            kv_cache: Mutex::new(kv_cache),
            scratch,
            kernels,
            max_seq_len,
            yarn_inv_freq,
            rope_theta,
            rotary_dim,
            rms_norm_eps: 1e-6,
            ctx_window,
            quant: DflashQuantization::Bf16,
        };

        tracing::info!(
            "BlockDiffusionDraftHead loaded: {} layers, hidden={}, intermediate={}, \
             GQA {}/{}, head_dim={}, γ={}, vocab={}, mask_token_id={}, target_layers={:?}",
            head.num_layers,
            head.hidden_size,
            head.intermediate_size,
            head.num_q_heads,
            head.num_kv_heads,
            head.head_dim,
            head.gamma,
            head.vocab_size,
            head.mask_token_id,
            head.target_layer_ids,
        );

        Ok(head)
    }

    /// Borrow-validate the drafter dimensions against the target's hidden_size
    /// at construction time. Mismatch is a hard error — the `fc` projection
    /// width is baked from `target_hidden_size` and a runtime mismatch would
    /// produce silent garbage (vLLM's loader hits this same check).
    pub fn validate_against_target(&self, target_hidden_size: usize) -> Result<()> {
        if self.target_hidden_size != target_hidden_size {
            anyhow::bail!(
                "DFlash drafter target_hidden_size mismatch: drafter expects {}, target is {}",
                self.target_hidden_size,
                target_hidden_size
            );
        }
        Ok(())
    }
}
