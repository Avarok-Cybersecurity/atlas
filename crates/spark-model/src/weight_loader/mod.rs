// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loading traits and per-model loader implementations.
//!
//! Translates flat [`WeightStore`] into typed [`TransformerLayer`] objects.
//! Each model architecture has its own [`ModelWeightLoader`] implementation
//! that knows the HuggingFace weight name patterns.
//!
//! Submodules contain per-family loaders:
//!   - `qwen3`: Qwen3-Next (NVFP4, hybrid SSM+Attention+MoE)
//!   - `qwen35`: Qwen3.5 MoE (35B, 122B)
//!   - `qwen35_dense`: Qwen3.5 Dense (27B)
//!   - `qwen3_vl`: Qwen3-VL (vision-language)
//!   - `nemotron`: Nemotron-H (Mamba-2 + MoE + Attention)
//!   - `gemma4`: Gemma-4 (pure attention, GeGLU, sliding + full attention)

pub mod dflash_loader;
mod gemma4;
mod minimax;
mod nemotron;
mod qwen3;
mod qwen35;
mod qwen35_dense;
mod qwen3_vl;

pub use dflash_loader::{
    DflashConfig, DflashLayerWeights, DflashSubConfig, DflashWeights, load_dflash_weights,
    store_has_dflash_weights,
};
pub use gemma4::Gemma4WeightLoader;
pub use minimax::MinimaxM2WeightLoader;
pub use nemotron::NemotronHWeightLoader;
pub use qwen3::Qwen3WeightLoader;
pub use qwen3_vl::Qwen3VLWeightLoader;
pub use qwen35::Qwen35WeightLoader;
pub use qwen35_dense::Qwen35DenseWeightLoader;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::VisionEncoder;
use crate::weight_map::{DenseWeight, MtpWeights, Nvfp4Variant, detect_nvfp4_variant};

/// Runtime quantization format for weight dispatch.
///
/// Determines which GEMV/GEMM kernels are used for decode, prefill, and
/// MTP verify. Adding a new quant format requires:
/// 1. Add variant here
/// 2. Add kernel dispatch in the layer forward paths
/// 3. Add weight loading logic in load_moe_qwen35 / attention loader
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// NVFP4 E2M1 — default, highest throughput. Uses w4a16 kernels.
    Nvfp4,
    /// FP8 E4M3 block-scaled — native FP8 serving. Uses w8a16 kernels.
    Fp8,
    // Future: Int4, AWQ, GPTQ, etc.
}

impl QuantFormat {
    /// Peak GPU memory multiplier for OOM pre-flight estimation.
    ///
    /// Accounts for model-building overhead on top of raw weight bytes:
    /// - NVFP4: 1.3x (weight pointers aliased, transposed copies + predequant)
    /// - FP8: 1.5x (zero-copy weights, transposed attention copies, FP8 pointer tables)
    ///
    /// Adding a new format: set the multiplier based on empirical peak/on-disk ratio.
    pub fn peak_memory_multiplier(&self) -> f64 {
        match self {
            // NVFP4: weights are mmap'd (zero-copy), temporary buffers for
            // runtime quantization (FP8→BF16→NVFP4) are freed after each layer.
            // Empirical peak/on-disk ratio on GB10: ~1.15x.
            Self::Nvfp4 => 1.15,
            Self::Fp8 => 1.5,
        }
    }
}

/// Checkpoint weight format, detected from safetensors metadata.
///
/// Determines how raw weight bytes are interpreted and transformed into
/// the runtime NVFP4 format used by Atlas GEMM kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// NVFP4 E2M1 on disk (nvidia ModelOpt or compressed-tensors).
    /// Weights load directly into `QuantizedWeight` with no conversion.
    Nvfp4,
    /// FP8 E4M3 block-scaled on disk (e.g. `quant_method: "fp8"` with `weight_block_size`).
    /// Each weight tensor has a `weight_scale_inv` (BF16 per-block) companion.
    /// At load time: FP8 -> BF16 -> NVFP4 (runtime quantization).
    Fp8BlockScaled,
    /// BF16 dense on disk (unquantized, e.g. attention Q/K/V in Standard NVFP4 models).
    /// At load time: BF16 -> NVFP4 (runtime quantization).
    Bf16Dense,
}

impl WeightFormat {
    /// Detect the weight format from a [`WeightStore`] by probing key names.
    pub fn detect(store: &WeightStore, config: &ModelConfig) -> Self {
        match detect_nvfp4_variant(store, config) {
            Nvfp4Variant::Fp8Dequanted => Self::Fp8BlockScaled,
            Nvfp4Variant::CompressedTensors | Nvfp4Variant::Standard => Self::Nvfp4,
            // Bf16Raw fine-tunes get runtime-quantized to NVFP4 inside the
            // weight loader, so the downstream pipeline sees Nvfp4.
            Nvfp4Variant::Bf16Raw => Self::Nvfp4,
        }
    }

    /// Whether this format requires FP8 -> BF16 dequantization at load time.
    pub fn is_fp8(&self) -> bool {
        matches!(self, Self::Fp8BlockScaled)
    }
}

/// Loads weights from a [`WeightStore`] into typed layer objects.
pub trait ModelWeightLoader {
    /// Whether this loader's weight slicing is TP-aware. **No default** —
    /// every loader MUST declare this explicitly so adding a new model
    /// architecture cannot accidentally inherit a `false` and silently
    /// regress users who pass `--tp-size > 1`.
    ///
    /// Loaders that honour `config.tp_world_size` / `config.tp_rank` when
    /// loading attention Q/K/V/O, MoE gate/up/down, head-parallel SSM
    /// components, and lm_head return `true`. Loaders that always load
    /// full replicated weights return `false`.
    ///
    /// The startup path in `spark-server/src/main.rs` consults this method
    /// to fail-fast at load time when `--tp-size > 1` is requested against
    /// a TP-unaware loader. Extending TP to a new architecture requires:
    ///   1. Wire `slice_for_rank` (in `crate::tp_shard`) per Q/K/V/O,
    ///      gate/up/down, and any head-parallel SSM tensors.
    ///   2. Divide `num_attention_heads` / `num_key_value_heads` per the
    ///      same axis when constructing layer state.
    ///   3. Return `true` from this method.
    ///
    /// See `weight_loader/minimax.rs` for the reference implementation.
    fn supports_tp(&self) -> bool;

    /// Load all transformer layers from the weight store.
    ///
    /// `layer_kv_dtypes` is indexed by attention layer index (0-based sequential
    /// counter over full-attention layers only). Each attention layer receives its
    /// own KV cache dtype, enabling mixed-precision KV caching where boundary
    /// layers use higher precision.
    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>>;

    /// Per-(layer, role) weight precision schedule (C.3, 2026-04-25).
    /// Default impl returns the empty schedule (every lookup yields
    /// `Dtype::Inherit`), preserving the existing per-checkpoint
    /// dtype logic byte-for-byte. Loader-specific implementations
    /// can override to honour MODEL.toml's `[precision]` block.
    fn precision_schedule(
        &self,
        _config: &ModelConfig,
    ) -> crate::precision_schedule::PrecisionSchedule {
        crate::precision_schedule::PrecisionSchedule::default()
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight>;
    /// Load the final RMSNorm weight used before the LM head.
    ///
    /// `gpu` is passed so model-specific loaders can do on-device weight
    /// transforms at load time (e.g. Gemma-4 shifts the learned absolute-
    /// scale weight by -1 into the offset-from-1 convention expected by
    /// Atlas's rms_norm kernel). Loaders that don't need it should ignore
    /// the argument.
    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight>;
    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight>;

    /// Load MTP head weights (returns None if no MTP weights in store).
    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>>;

    /// Load MTP weights for multi-module MTP (DeepSeek-V3 / MiniMax-M2
    /// style: N independent transformer modules, each with its own
    /// attention + MoE + KV cache). Returns an empty `Vec` when the
    /// checkpoint has no MTP modules, a 1-element Vec for single-module
    /// MTP (Qwen3.5 family), or N elements for multi-module.
    ///
    /// Default impl adapts `load_mtp_weights` so existing single-module
    /// loaders don't need to change. MiniMax overrides this directly.
    fn load_mtp_weights_multi(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Vec<MtpWeights>> {
        Ok(self
            .load_mtp_weights(store, config, gpu)?
            .into_iter()
            .collect())
    }

    /// Per-layer (num_kv_heads, head_dim) overrides for heterogeneous
    /// attention models (e.g. Gemma-4 with sliding 16×256 and full 4×512).
    /// Default empty — homogeneous models skip per-layer dims and the KV
    /// cache allocator uses the global (num_kv_heads, head_dim). Populated
    /// by loaders whose models have different attention geometries per
    /// layer. Indexed by attention layer index (same as layer_kv_dtypes).
    fn kv_layer_dims(&self, _config: &ModelConfig) -> Vec<(usize, usize)> {
        Vec::new()
    }

    /// Load DFlash drafter weights from a separate `WeightStore` pointing
    /// at the drafter checkpoint (`z-lab/Qwen3.6-{27B,35B-A3B}-DFlash`).
    /// Default impl returns `None` so loaders that don't yet support
    /// DFlash silently fall through to the existing MTP path. Override in
    /// loaders whose target models pair with a DFlash drafter (Qwen3.5/3.6
    /// family). The same drafter format works across both 27B-dense and
    /// 35B-A3B-MoE targets — only the `target_hidden_size` validated
    /// against the drafter's `fc` input dimension differs.
    fn load_dflash_weights(
        &self,
        _drafter_store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _tp_size: usize,
    ) -> Result<Option<DflashWeights>> {
        Ok(None)
    }

    /// Load vision encoder weights (returns None for text-only models).
    fn load_vision_encoder(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<VisionEncoder>> {
        Ok(None)
    }
}
