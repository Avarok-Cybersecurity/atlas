// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use serde::Deserialize;

/// Deserialize a u32 that may be JSON null (treat null as 0).
fn nullable_u32<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<u32, D::Error> {
    Option::<u32>::deserialize(d).map(|v| v.unwrap_or(0))
}

/// Layer type in a hybrid transformer model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    SlidingAttention,
    LinearAttention,
    /// Standalone MoE FFN layer (Nemotron-H: no mixer, just expert routing + FFN).
    Moe,
}

/// Model configuration parsed from HuggingFace config.json.
///
/// Single source of truth for model dimensions. All kernel launch
/// parameters and buffer sizes derive from this struct.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    // ── Core dimensions ──
    pub hidden_size: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub vocab_size: usize,

    // ── Full attention ──
    #[serde(default)]
    pub num_attention_heads: usize,
    /// Per-layer Q-head counts for heterogeneous attention models. Empty means
    /// every layer uses `num_attention_heads`.
    #[serde(default)]
    pub num_attention_heads_per_layer: Vec<usize>,
    /// GQA: number of K/V heads (≤ `num_attention_heads`). MQA when 1.
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    /// Fraction of `head_dim` that gets RoPE-rotated. 1.0 = full RoPE,
    /// 0.5 = half-rotated (Phi-style). Default 1.0.
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,

    // ── Linear attention (SSM / GDN) ──
    // "linear" = the recurrent state-space / gated-delta-net pathway used
    // by hybrid models (Qwen3.5/3.6, Nemotron-Nano, MiniMax). Per-token
    // updates run in O(1) state instead of O(seq) attention.
    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
    /// 1D causal-conv kernel size on the SSM input (typically 3 or 4).
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,

    // ── MoE ──
    #[serde(default)]
    pub num_experts: usize,
    /// LongCat-Flash zero-computation "identity" experts: the router scores
    /// `num_experts + zero_expert_num` logits, and a token routed to an
    /// expert id `>= num_experts` receives the INPUT itself scaled by the
    /// routing weight instead of an expert FFN. 0 = no zero-experts.
    #[serde(default)]
    pub zero_expert_num: usize,
    /// Top-K experts activated per token (the "A" in 35B-A3B = 3B
    /// active params).
    #[serde(default = "default_one")]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    /// Renormalize routing probabilities so the K active experts sum
    /// to 1 after top-K selection. Qwen3.5+ sets true; older Qwen2 MoE
    /// variants set false.
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// MoE block stride: layer `i` uses MoE iff `i % decoder_sparse_step
    /// == 0`. 1 = every layer is MoE. Mistral / DeepSeek-style stagger
    /// uses 2.
    #[serde(default = "default_one")]
    pub decoder_sparse_step: usize,

    // ── Hybrid layer layout ──
    /// Per-layer kind (FullAttention | LinearAttention | …) parsed from
    /// HF config. When empty, falls back to `full_attention_interval`.
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    /// Stride for full-attention layers in hybrid models when
    /// `layer_types` is empty: every Nth layer is FullAttention, the
    /// rest LinearAttention. 1 = every layer is full attention.
    #[serde(default = "default_one")]
    pub full_attention_interval: usize,
    /// Gemma-4 hybrid-attention sliding window size (0 = full attention).
    /// Sliding layers only attend to the last `sliding_window` KV positions;
    /// full layers (every 6th in Gemma-4) ignore this (effectively 0).
    /// Parsed from HF config.json `sliding_window` field. Uses `nullable_u32`
    /// because Nemotron-H (and some other models) set it to `null` in JSON.
    #[serde(default, deserialize_with = "nullable_u32")]
    pub sliding_window: u32,

    // ── Position embeddings ──
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    // ── Normalization ──
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,

    // ── Tokenizer ──
    /// BOS token ID (null → 0 for models without explicit BOS).
    #[serde(default, deserialize_with = "nullable_u32")]
    pub bos_token_id: u32,
    #[serde(default, deserialize_with = "nullable_u32")]
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// CLI override (`--lm-head-dtype`) for LM-head quantization, set at serve time
    /// (not from config.json). `Some(true)` = force BF16 lm_head; `Some(false)` = force
    /// the model's quantized lm_head; `None` = use the model-config-driven default.
    /// Consumed by `skip_lm_head_quantization()`. Replaces the ATLAS_LMHEAD_BF16 env var.
    #[serde(default)]
    pub lm_head_bf16_override: Option<bool>,
    /// When `skip_lm_head_quantization()` == false, quantize the LM head to FP8
    /// (E4M3, per-row scales, decoded via `w8a16_gemv`) instead of NVFP4.
    /// Set by `--lm-head-dtype fp8`. Additive: leaves the NVFP4/BF16 paths
    /// byte-identical when false.
    #[serde(default)]
    pub lm_head_fp8: bool,

    // ── Model type ──
    #[serde(default)]
    pub model_type: String,

    // ── MTP ──
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,

    // ── DSpark ──
    /// Number of query positions generated by one semi-autoregressive draft pass.
    /// Zero means the checkpoint does not declare checkpoint-native DSpark.
    #[serde(default)]
    pub dspark_block_size: usize,
    /// Token used to initialize the non-anchor positions in a DSpark block.
    #[serde(default)]
    pub dspark_noise_token_id: u32,
    /// Target layers whose hidden states are concatenated for the DSpark input.
    #[serde(default)]
    pub dspark_target_layer_ids: Vec<usize>,
    /// Width of the low-rank Markov token transition head.
    #[serde(default)]
    pub dspark_markov_rank: usize,

    // ── Nemotron-H / Mamba-2 ──
    #[serde(default)]
    pub hybrid_override_pattern: String,
    #[serde(default)]
    pub mamba_num_heads: usize,
    #[serde(default)]
    pub mamba_head_dim: usize,
    #[serde(default)]
    pub ssm_state_size: usize,
    #[serde(default)]
    pub n_groups: usize,
    #[serde(default)]
    pub expand: usize,
    /// Nemotron-H uses `n_routed_experts` (mapped to `num_experts` in parse_config).
    #[serde(default)]
    pub n_routed_experts: usize,
    /// Nemotron-H uses `norm_eps` (mapped to `rms_norm_eps` in parse_config).
    #[serde(default)]
    pub norm_eps: f64,
    /// Nemotron-H conv kernel size (mapped to `linear_conv_kernel_dim` in parse_config).
    #[serde(default)]
    pub conv_kernel: usize,
    /// Nemotron-H shared expert intermediate (mapped to shared_expert_intermediate_size).
    #[serde(default)]
    pub moe_shared_expert_intermediate_size: usize,
    /// Nemotron-H routed scaling factor for expert outputs.
    #[serde(default = "default_one_f64")]
    pub routed_scaling_factor: f64,
    /// Decoder-layer indices that use a dense MLP instead of routed experts.
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    /// LatentMoE: latent projection dimension for routed experts (Super 120B).
    /// When present, routed experts operate in latent space `[moe_latent_size]`
    /// instead of full `[hidden_size]`. Absent for Nano 30B.
    #[serde(default)]
    pub moe_latent_size: usize,
    /// Per-layer MoE intermediate sizes (Nemotron-H Puzzle heterogeneous channel
    /// pruning). Length == `num_hidden_layers`; 0 for non-MoE layers. Empty =
    /// fall back to scalar `moe_intermediate_size` for every MoE layer.
    #[serde(default, skip_deserializing, skip_serializing)]
    pub moe_intermediate_sizes: Vec<usize>,
    /// Per-layer top-K expert counts (Puzzle). Same layout as
    /// `moe_intermediate_sizes`. Empty = use scalar `num_experts_per_tok`.
    #[serde(default, skip_deserializing, skip_serializing)]
    pub num_experts_per_toks: Vec<usize>,

    // ── MLA (Multi-head Latent Attention) — Mistral Small 4 / DeepSeek-V2+ ──
    /// KV latent dimension for compressed cache. 0 = standard attention (no MLA).
    #[serde(default)]
    pub kv_lora_rank: usize,
    /// Per-layer KV cache dimensions (num_kv_heads, head_dim). Populated by
    /// loaders for heterogeneous-attention models (e.g. Gemma-4 with sliding
    /// and full attention having different head counts and dims). Empty for
    /// homogeneous models.
    #[serde(default, skip_deserializing, skip_serializing)]
    pub kv_layer_dims: Vec<(usize, usize)>,
    /// Query latent dimension for low-rank Q projection. 0 = standard Q.
    #[serde(default)]
    pub q_lora_rank: usize,
    /// Non-rotary portion of Q/K per head (NoPE component).
    #[serde(default)]
    pub qk_nope_head_dim: usize,
    /// Rotary portion of Q/K per head (RoPE component).
    #[serde(default)]
    pub qk_rope_head_dim: usize,
    /// Value dimension per head (may differ from head_dim in MLA).
    #[serde(default)]
    pub v_head_dim: usize,

    // ── N-gram embeddings — LongCat-Flash-Lite / Qwen3.8-Flash-Next ──
    // (arxiv 2601.21204: capacity via hashed n-gram lookup tables instead of
    // more experts.) `emb_split_num * (emb_neighbor_num - 1)` embedding
    // tables, each ~`ngram_vocab_size_ratio * vocab_size` rows at
    // `hidden_size / num_tables` dims; ids are a polynomial rolling hash of
    // the current + previous n-1 TOKEN IDS (never hidden states), each
    // looked-up vector is projected to hidden and ADDED to the base token
    // embedding, and the sum is scaled by 1/(1 + num_tables). Reference:
    // bench/ngram_ref/{modeling_longcat_ngram.py, ngram_parity.py}.
    /// N-gram table size multiplier: each table has ~ratio*vocab_size rows
    /// (LongCat-Lite: 78 → ~10.2M rows/table). 0 = no n-gram embeddings.
    #[serde(default)]
    pub ngram_vocab_size_ratio: usize,
    /// Largest n-gram size N (LongCat-Lite: 4 → bigram/trigram/4-gram).
    #[serde(default)]
    pub emb_neighbor_num: usize,
    /// Independent hash splits K per n-gram size (LongCat-Lite: 4).
    #[serde(default)]
    pub emb_split_num: usize,
    /// Rows per n-gram HEAD, absolute (`ngram_vocab_size_base`).
    ///
    /// The Qwen4-Exp form of the same idea LongCat expresses as a ratio:
    /// LongCat says "ratio x vocab_size rows per table", Qwen says
    /// "20,000,000 rows per head" outright. Mutually exclusive with
    /// `ngram_vocab_size_ratio` — whichever the checkpoint declares wins,
    /// and the authoritative per-head sizes/offsets ship as I64 tensors
    /// (`ngram_heads_vocab_sizes` / `ngram_heads_offsets`) which the loader
    /// reads rather than re-deriving. 0 = not a base-form checkpoint.
    #[serde(default)]
    pub ngram_vocab_size_base: usize,
    /// Physical shard count of the n-gram table (`split_ngram_parts`).
    ///
    /// PURELY a file-layout fact, NOT an architectural one: Qwen4-Exp stores
    /// one logical `[sum(head_vocabs), ngram_dim]` table as 128 equal
    /// `shard_N.weight` tensors. The head ranges are independent of the
    /// shard boundaries and a head can straddle several shards, so the row
    /// cache must address the logical table and translate. 0 = unsharded.
    #[serde(default)]
    pub ngram_split_parts: usize,
    /// Decoder layers that carry a PLE (per-layer-embedding) n-gram
    /// injection (`ple_layer_ids`). Qwen4-Exp injects at ONE layer, not at
    /// the token embedding the way LongCat does — which is why this is a
    /// layer list and not a flag. Empty = no PLE.
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    /// Depthwise conv width inside the PLE block (`ple_conv_kernel_size`).
    /// 0 = no conv.
    #[serde(default)]
    pub ple_conv_kernel_size: usize,

    // ── DeepSeek-V4 low-rank / grouped output projection + mHC ──
    /// Output projection latent dimension for low-rank O projection.
    /// DeepSeek-V4 uses `o_lora_rank` to compress the output projection.
    /// 0 = standard O (no low-rank compression).
    #[serde(default)]
    pub o_lora_rank: usize,
    /// Number of block-diagonal groups for the grouped O projection (wo_a).
    /// DeepSeek-V4-Flash splits the n_heads*head_dim attention output into
    /// `o_groups` independent groups, each projected to `o_lora_rank` before the
    /// follow-up wo_b mixes the `o_groups*o_lora_rank` vector back to hidden_size.
    /// 0 = ungrouped (dense O).
    #[serde(default)]
    pub o_groups: usize,
    /// YaRN attention-temperature `mscale` (`rope_scaling.mscale`). HF default
    /// is 1.0 when absent. DeepSeek folds `_mscale` into the rope cos/sin.
    #[serde(default)]
    pub yarn_mscale: f32,
    /// YaRN attention-temperature `mscale_all_dim` (`rope_scaling.mscale_all_dim`).
    /// HF default is 0.0 when absent. Used in the `_mscale` ratio that scales
    /// the rope cos/sin (and, when non-zero, the softmax scale).
    #[serde(default)]
    pub yarn_mscale_all_dim: f32,
    /// Number of hyper-connection residual streams per block (`hc_mult`).
    /// 0 = disabled (every model except DeepSeek-V4). DeepSeek-V4 uses 4.
    #[serde(default)]
    pub hc_mult: usize,
    /// Number of Sinkhorn normalization iterations for the HC mixing matrix
    /// (`hc_sinkhorn_iters`). DeepSeek-V4 default is 20.
    #[serde(default)]
    pub hc_sinkhorn_iters: usize,
    /// Numerical-stability epsilon for HC sigmoid/softmax/Sinkhorn (`hc_eps`).
    /// DeepSeek-V4 default is 1e-6.
    #[serde(default)]
    pub hc_eps: f32,
    /// Rank of the hyper-connection input mixer (`hc_lowrank`).
    ///
    /// Qwen4-Exp mixes the `hc_mult` residual streams through a LOW-RANK
    /// pair — `input_mix_weight_down [r, hc_mult*hidden]` then
    /// `input_mix_weight_up [hc_mult*hidden, r]` — where DeepSeek-V4 uses a
    /// Sinkhorn-normalized square matrix. The two share `hc_mult` and the
    /// stream-major layout but NOT the mixing math, so a non-zero value here
    /// selects the low-rank variant. 0 = DeepSeek-V4's Sinkhorn form.
    #[serde(default)]
    pub hc_lowrank: usize,
    /// The checkpoint carries NO final normalization before `lm_head`: the
    /// real one is applied inside the hyper-connection mixer while the
    /// residual streams collapse. Applying the engine's ones-placeholder RMS
    /// anyway still DIVIDES the hidden by its per-token RMS, which flattens
    /// the logits by a per-token factor (measured 1.16-1.63x vs the reference
    /// forward on qwen4_exp) -- an uninvited temperature multiplier that
    /// argmax survives but sampling does not. When set, the final-norm step
    /// becomes an identity copy.
    #[serde(default)]
    pub final_norm_identity: bool,
    /// Per-layer compression ratios for hybrid attention (CSA/HCA).
    /// 0 = full attention, >0 = compressed attention with that ratio.
    /// Length equals num_hidden_layers. Empty = all layers full attention.
    #[serde(default)]
    pub compress_ratios: Vec<usize>,
    /// Number of semantic-indexer heads used by DeepSeek-V4 CSA layers.
    #[serde(default)]
    pub index_n_heads: usize,
    /// Per-head dimension of the DeepSeek-V4 semantic indexer.
    #[serde(default)]
    pub index_head_dim: usize,
    /// Maximum compressed-history rows selected per query by the semantic indexer.
    #[serde(default)]
    pub index_topk: usize,
    /// Indexer compression ratio, recorded WITHOUT populating
    /// `compress_ratios`.
    ///
    /// Qwen3.8-Flash-Next's QSA indexer is inert below its budget — selection
    /// is `topk(min(budget/ratio, complete_blocks))`, so at
    /// `seq_len <= index_topk` every block is chosen and dense attention is
    /// exact. Keeping `compress_ratios` empty stops DeepSeek-V4's compressor
    /// being dispatched in its place; keeping the ratio here lets a loader
    /// refuse above the budget instead of silently attending densely.
    /// 0 = no indexer.
    #[serde(default)]
    pub index_compress_ratio: usize,
    /// Number of hash-based attention layers (DeepSeek-V4 HCA). 0 = none.
    #[serde(default)]
    pub num_hash_layers: usize,

    // ── YaRN RoPE scaling (Mistral Small 4) ──
    /// YaRN scaling factor (`yarn.factor`). 0.0 = YaRN disabled, use plain RoPE.
    #[serde(default)]
    pub yarn_factor: f32,
    /// YaRN low-rotation cutoff (`yarn.alpha` in Mistral params,
    /// `beta_slow` in HF transformers terminology).
    #[serde(default)]
    pub yarn_beta_slow: f32,
    /// YaRN high-rotation cutoff (`yarn.beta` in Mistral params,
    /// `beta_fast` in HF transformers terminology).
    #[serde(default)]
    pub yarn_beta_fast: f32,
    /// YaRN original context length used for the correction range
    /// (`yarn.original_max_position_embeddings`).
    #[serde(default)]
    pub yarn_original_max_position_embeddings: usize,
    /// Multiplier applied to both YaRN cosine and sine values. 1.0 means no
    /// attention-temperature scaling.
    #[serde(default = "default_one_f32")]
    pub yarn_attention_factor: f32,
    /// llama_4_scaling Q temperature beta (`llama_4_scaling.beta`).
    /// Q is multiplied by `1 + beta * log(1 + floor(pos / original_max_pos))`
    /// after RoPE. 0.0 = disabled. Mistral Small 4 uses 0.1.
    #[serde(default)]
    pub llama_4_scaling_beta: f32,
    /// llama_4_scaling original context length for the Q temperature scale.
    #[serde(default)]
    pub llama_4_scaling_original_max_position_embeddings: usize,

    // ── Vision (Qwen3-VL only) ──
    /// Vision encoder configuration parsed from `vision_config` in config.json.
    /// None for text-only models.
    #[serde(skip)]
    pub vision: Option<VisionConfig>,

    /// Advertised quantization format + algorithm + per-module ignore list.
    /// Populated from `config.json::quantization_config` or a sibling
    /// `hf_quant_config.json` at `parse_config` time. `None` for
    /// un-quantized BF16/FP16 checkpoints. Consumed by the `QuantFormat`
    /// dispatcher (`crates/spark-model/src/quant_format/`) to pick the
    /// correct on-disk loader without guessing from tensor names.
    #[serde(skip)]
    pub quantization_config: Option<QuantizationConfig>,

    // ── Architecture flags (set by parse_config, not from JSON) ──
    /// Whether Q projection includes an output gate (Q+Gate interleaved, 2x q_dim).
    /// False for Qwen3-VL, Nemotron-H, Mistral (ungated Q).
    #[serde(skip)]
    pub attn_gated: bool,
    /// The GDN gated-norm's gate activation is SIGMOID rather than SiLU.
    ///
    /// The reference constructs its `RMSNormGated` with
    /// `activation = output_gate_type or hidden_act`, so on a checkpoint
    /// with `output_gate_type: "sigmoid"` (Qwen3.8-Flash-Next) BOTH the
    /// attention output gate and the GDN norm gate are sigmoid. Every other
    /// Qwen-family GDN model gates with SiLU. Found by the qwen4_exp phase-E
    /// bisect: recurrence proven correct, norm stage off at cos 0.81, and
    /// sigmoid closed it to 0.0.
    #[serde(default)]
    pub gdn_norm_sigmoid: bool,
    /// Whether config.json wraps the LLM config in a nested field (e.g., `text_config`).
    /// Determines weight prefix auto-detection behavior.
    #[serde(skip)]
    pub nested_config: bool,
    /// MRoPE (multi-modal rotary position embedding) section sizes in
    /// `[T, H, W]` order. `[0, 0, 0]` = scalar RoPE (default for Qwen3.5
    /// and earlier). Qwen3.6 uses `[11, 11, 10]`. Summed × 2 == rotary_dim.
    #[serde(skip)]
    pub mrope_section: [usize; 3],
    /// MRoPE channel layout: `true` = round-robin `[T H W T H W …]` (Qwen3.6),
    /// `false` = contiguous `[T…T | H…H | W…W]` (Qwen3-VL non-interleaved).
    /// Ignored when `mrope_section == [0, 0, 0]`.
    #[serde(skip)]
    pub mrope_interleaved: bool,

    // ── Weight key prefix (set by parser for conditional generation models) ──
    #[serde(skip)]
    pub weight_prefix: String,

    /// `--profile`: skip CUDA graphs, sync and time each layer.
    ///
    /// Carried here rather than through `ATLAS_PROFILE`, which `serve.rs` used
    /// to `set_var` at runtime under a `// SAFETY: called before any threads
    /// are spawned` comment that was **already false** — the tokio pool, the
    /// startup blocking thread, the signal listener, the TUI thread and the
    /// OOM watchdog all exist by then, and a concurrent `getenv` during
    /// `setenv` is UB. A field on the config the model already receives has
    /// none of that hazard.
    #[serde(skip)]
    pub profile: bool,

    /// Boot-time expert loading plan (`--expert-category`), resolved from
    /// the model's MODEL.toml at serve time. `None` = load every expert.
    ///
    /// Runtime state like `ep_rank` below, not checkpoint config: it comes
    /// from the command line and the compiled-in category table, and it is
    /// the SSOT both the weight loaders and the router mask read — they must
    /// not be able to disagree about which experts are resident.
    #[serde(skip)]
    pub bel: Option<std::sync::Arc<bel::BelPlan>>,

    // ── Expert Parallelism (set at runtime, not from config.json) ──
    #[serde(skip)]
    pub ep_rank: usize,
    #[serde(skip)]
    pub ep_world_size: usize,

    // ── Tensor Parallelism (set at runtime, not from config.json) ──
    /// TP rank within the TP sub-communicator. 0 if `tp_world_size==1`.
    #[serde(skip)]
    pub tp_rank: usize,
    /// Number of TP ranks. 1 = no TP. Composes with EP statically:
    /// attention/MLP weights are TP-sharded; MoE expert weights are EP-sharded.
    #[serde(skip)]
    pub tp_world_size: usize,

    // ── FP8 KV cache calibration (set at runtime from CLI) ──
    /// Number of warmup tokens for online FP8 KV scale calibration.
    /// 0 = disabled (use static scales from checkpoint or uncalibrated 1.0).
    #[serde(skip)]
    pub fp8_kv_calibration_tokens: usize,
    /// Headroom multiplier on the first-observe absmax when freezing the online
    /// FP8 KV scale (`--fp8-kv-headroom`, default 2.0). The first observe sees
    /// only the first prefill chunk, so the frozen scale covers headroom× its
    /// observed max — later tokens that grow don't clip, at <1 bit of precision.
    #[serde(skip)]
    pub fp8_kv_headroom: f32,

    // ── Gemma-4 specific ──
    /// Final logit softcapping: logits = cap * tanh(logits / cap).
    /// 0.0 = disabled (default for all models except Gemma-4 which uses 30.0).
    #[serde(skip)]
    pub final_logit_softcapping: f32,
    /// Embedding scale factor: embeddings *= scale after lookup.
    /// 0.0 = disabled (default). Gemma models use sqrt(hidden_size).
    #[serde(skip)]
    pub embed_scale: f32,

    // ── MiniMax M2 specific ──
    /// MoE routing activation. "" = default softmax. "sigmoid" = DeepSeek-V3
    /// / MiniMax-M2 style: raw gate logits pass through sigmoid to produce
    /// per-expert scores in (0,1), independent (not normalized across
    /// experts). Top-k selection may use a bias term (see `moe_routing_bias`).
    #[serde(default)]
    pub scoring_func: String,
    /// If true, a per-expert `e_score_correction_bias` tensor is added to
    /// routing scores *for top-k selection only* (not dispatch weighting).
    /// This is the DeepSeek-V3 loss-free balancing trick. The bias tensor
    /// itself lives in the checkpoint (typically one `[num_experts]` vector
    /// per MoE layer).
    #[serde(default)]
    pub use_routing_bias: bool,
    /// QK normalization granularity. "" = none (Qwen3-Next default).
    /// "per_layer" = each attention layer has its own learned q_layernorm /
    /// k_layernorm weight of shape `[head_dim]`, applied after Q/K projection
    /// and before RoPE (MiniMax M2).
    #[serde(default)]
    pub qk_norm_type: String,
    /// Number of sequential MTP draft modules. 0 = no MTP. 1 = existing
    /// Atlas MTP path (Qwen3.5). 3 = MiniMax M2 (each module is a single
    /// transformer layer that predicts one future token).
    #[serde(default)]
    pub num_mtp_modules: usize,
    /// Transformer layers per MTP module. 1 for MiniMax M2 (3 modules × 1
    /// layer = 3 future-token predictors).
    #[serde(default)]
    pub mtp_transformer_layers: usize,
    /// Explicit rotary dimension from config (bypasses partial_rotary_factor
    /// computation). MiniMax M2 ships `rotary_dim: 64` while head_dim=128,
    /// so the rotary factor is 0.5 — we honor the explicit int value when
    /// present for byte-exact rope dim.
    #[serde(default)]
    pub rotary_dim: usize,

    /// Target-model layer indices to capture intermediate hidden states from
    /// for DFlash speculative decoding. Sourced from the drafter's
    /// `dflash_config.target_layer_ids` (e.g., `[1, 10, 19, 28, 37]` for
    /// Qwen3.6-35B-A3B-DFlash). Empty when DFlash is disabled — its presence
    /// gates `TransformerModel::dflash_hidden_save` allocation and the
    /// per-layer capture hooks. Order matters: shallow-to-deep concatenation
    /// is what the drafter's `fc` projection expects.
    #[serde(default)]
    pub dflash_capture_layers: Vec<usize>,
    /// Resolved DFlash drafter γ (block size), set by the factory alongside
    /// `dflash_capture_layers`. Sizes the SSM verify intermediate pools at
    /// the ACTUAL K = γ+1 instead of the legacy 17-wide ceiling — at γ=8,
    /// C=8 that ceiling alone cost ~12 GB of pool (2026-08-19 256K/C8 boot
    /// ledger). `None` = DFlash inactive (or unknown → 17-wide fallback).
    pub dflash_gamma: Option<usize>,

    /// LoRA adapter rank ceiling (`--max-lora-rank`). `0` = LoRA disabled.
    /// Set programmatically before model build (never parsed from the HF
    /// `config.json`); the only consumer is `BufferSizes`, which sizes the
    /// adapter delta scratch from it. `adapter_*` naming avoids the MLA
    /// `*lora_rank` collision (`config.rs:182-207`).
    #[serde(default)]
    pub adapter_max_rank: usize,
}

/// Advertised weight-quantization layout, as declared in the HF
/// `config.json`'s `quantization_config` block (or a sibling
/// `hf_quant_config.json`). This is the authoritative signal for
/// format dispatch — the `QuantFormat` trait prefers this over
/// tensor-name sniffing, matching the dispatch model used by vLLM /
/// TensorRT-LLM / SGLang.
///
/// `quant_method` is the serialization scheme:
///   * `"compressed-tensors"` — Neural Magic / llm-compressor. Uses
///     `weight_packed` + `weight_global_scale` + `input_global_scale`.
///     Commonly paired with `format = "nvfp4-pack-quantized"` or
///     `"float-quantized"`.
///   * `"modelopt"` — NVIDIA TensorRT ModelOpt. Uses `weight` (as the
///     packed FP4 payload when `quant_algo == "NVFP4"`) + `weight_scale`
///     + `weight_scale_2` + `input_scale`.
///   * `"fp8"` — native FP8 block-scaled (e.g. `Qwen/Qwen3.5-35B-A3B-FP8`)
///     with `weight_scale_inv` sibling tensors.
///
/// `ignore_modules` holds the already-expanded list of module-path
/// patterns that should be loaded as dense BF16 rather than quantized.
/// Patterns use HF glob semantics (`*` matches any non-`.` sub-path).
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Raw `quant_method` string from the config. Stable values:
    /// `"compressed-tensors"`, `"modelopt"`, `"fp8"`.
    pub quant_method: String,
    /// ModelOpt-specific algorithm label: `"NVFP4"`, `"FP8"`, …
    /// Empty string for schemes that don't declare one (e.g. plain FP8).
    pub quant_algo: String,
    /// Optional `format` string (compressed-tensors uses this for
    /// `"nvfp4-pack-quantized"` and friends).
    pub format: String,
    /// Module-path globs that should stay BF16 (the "ignore list" in
    /// ModelOpt terminology; `targets`/`exclude_modules` in compressed-
    /// tensors). Example entries: `"lm_head"`,
    /// `"model.layers.*.self_attn*"`.
    pub ignore_modules: Vec<String>,
}

/// Vision encoder configuration for Qwen3-VL models.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Number of ViT transformer blocks (depth=27).
    pub depth: usize,
    /// ViT hidden dimension (1152).
    pub hidden_size: usize,
    /// Number of attention heads (16).
    pub num_heads: usize,
    /// Spatial patch size in pixels (16).
    pub patch_size: usize,
    /// Temporal patch size: still images are replicated this many times (2).
    pub temporal_patch_size: usize,
    /// 2×2 spatial merge: this many patch-lengths merged into one token (2).
    pub spatial_merge_size: usize,
    /// ViT MLP intermediate size (4304).
    pub intermediate_size: usize,
    /// Projection output dimension = LLM hidden_size (2048).
    pub out_hidden_size: usize,
    /// Layer indices after which deepstack mergers are applied ([8, 16, 24]).
    pub deepstack_visual_indexes: Vec<usize>,
    /// Placeholder token ID that marks where vision embeddings get spliced
    /// into the text embedding stream. Qwen3-VL uses 151655; Qwen3.6 uses
    /// 248056. When 0 the runtime falls back to the legacy Qwen3-VL value.
    pub image_pad_token_id: u32,
    /// Placeholder token ID for VIDEO frames, the temporal sibling of
    /// [`Self::image_pad_token_id`]. Qwen3.6/3.8 use 248057. A distinct token
    /// is what lets the position builder tell a video item from an image one
    /// in the token stream, which matters because their MRoPE treatment
    /// differs: an image holds T constant across its whole pad run, a video
    /// advances T once per temporal group. When 0 the runtime falls back to
    /// the family default.
    pub video_pad_token_id: u32,
    /// Resolved vision AREA bound in pixels: the operator's
    /// `--vision-max-pixels`, else the checkpoint's `preprocessor_config.json`,
    /// else `None`.
    ///
    /// ★ THE SINGLE SOURCE OF TRUTH, and it exists because there used to be
    /// two. The CPU preprocessor clamped every image to 1280px on the long
    /// side while the GPU encoder allocated its buffers for 6400 patches —
    /// exactly 1280×1280 — with nothing in the code connecting them. They
    /// agreed only by coincidence, so raising one on 2026-08-14 made every
    /// image above 1280px fail an H2D copy with `CUDA_ERROR_INVALID_VALUE`
    /// from deep inside the scheduler.
    ///
    /// Both now derive from this field, resolved once at config load, before
    /// the encoder is constructed. `None` keeps the historical behaviour on
    /// both sides.
    pub max_pixels: Option<usize>,
}

impl VisionConfig {
    /// Dimension of the merger input (spatial_merge_size² × hidden_size).
    pub fn merger_input_size(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size * self.hidden_size
    }
}

pub(crate) fn default_one() -> usize {
    1
}
pub(crate) fn default_one_f64() -> f64 {
    1.0
}
pub(crate) fn default_one_f32() -> f32 {
    1.0
}
pub(crate) fn default_rope_theta() -> f64 {
    10000.0
}
pub(crate) fn default_rms_eps() -> f64 {
    1e-6
}
pub(crate) fn default_partial_rotary() -> f64 {
    1.0
}
pub(crate) fn default_conv_kernel() -> usize {
    4
}

pub mod bel;
mod dispatch;
mod factory;
mod gguf;
mod methods;
mod parsers;
#[cfg(test)]
mod tests;

pub use dispatch::parse_config;
pub use gguf::{GgufConfigInputs, GgufMeta, config_from_gguf};
pub use parsers::{
    PEFT_SUPPORTED_TARGET_MODULES, PeftAdapterConfig, allow_partial_targets, parse_mistral_params,
    parse_peft_adapter_config, parse_quantization_config,
};
pub(crate) use parsers::{
    parse_deepseek_v4, parse_gemma4_params, parse_laguna, parse_longcat_ngram, parse_minimax_m2,
    parse_qwen4_exp, parse_step3p7, parse_vision_config,
};

pub(crate) fn finalize_config(config: &mut ModelConfig, raw: &serde_json::Value) -> Result<()> {
    if config.quantization_config.is_none() {
        config.quantization_config = parse_quantization_config(raw);
    }
    validate_config(config)
}

/// Post-parse validation for ModelConfig.
/// Checks layer_types length matches num_hidden_layers and SSM field consistency.
pub(crate) fn validate_config(config: &ModelConfig) -> Result<()> {
    if !config.layer_types.is_empty() && config.layer_types.len() != config.num_hidden_layers {
        anyhow::bail!(
            "layer_types length ({}) doesn't match num_hidden_layers ({}) in config.json",
            config.layer_types.len(),
            config.num_hidden_layers,
        );
    }

    if !config.num_attention_heads_per_layer.is_empty()
        && config.num_attention_heads_per_layer.len() != config.num_hidden_layers
    {
        anyhow::bail!(
            "num_attention_heads_per_layer length ({}) doesn't match num_hidden_layers ({}) in config.json",
            config.num_attention_heads_per_layer.len(),
            config.num_hidden_layers,
        );
    }

    let has_ssm =
        config.layer_types.contains(&LayerType::LinearAttention) || config.linear_num_key_heads > 0;
    if has_ssm && config.linear_num_key_heads == 0 && config.mamba_num_heads == 0 {
        anyhow::bail!(
            "SSM model detected but linear_num_key_heads is 0 in config.json. \
             This field is required for SSM/GDN layer initialization."
        );
    }

    if config.mamba_num_heads > 0 {
        if config.mamba_head_dim == 0 {
            anyhow::bail!("mamba_head_dim must be greater than zero");
        }
        if config.ssm_state_size == 0 {
            anyhow::bail!("ssm_state_size must be greater than zero");
        }
        if config.n_groups == 0 {
            anyhow::bail!("n_groups must be greater than zero");
        }
        if !config.mamba2_d_inner().is_multiple_of(config.n_groups) {
            anyhow::bail!("mamba_num_heads * mamba_head_dim must be divisible by n_groups");
        }
    }

    Ok(())
}
