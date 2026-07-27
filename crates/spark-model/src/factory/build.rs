// SPDX-License-Identifier: AGPL-3.0-only

//! `build_model` — entry point that wires up the configured loader,
//! buffers, KV cache, and (optional) DFlash drafter into a `TransformerModel`.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::weights::WeightStore;

use super::loader_for_config;
use super::m2_setup::maybe_run_minimax_m2_moe_transpose;
use super::{DflashBuildArgs, LoraBuildArgs};
use crate::layers::MtpQuantization;
use crate::model::TransformerModel;
use crate::traits::Model;
use crate::weight_loader::load_dflash_weights;

mod kv_summary;

pub fn build_model(
    mut config: ModelConfig,
    store: &WeightStore,
    gpu: Box<dyn GpuBackend>,
    max_batch_tokens: usize,
    kv_block_size: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    mtp_quant: MtpQuantization,
    use_speculative: bool,
    prefix_cache: Box<dyn PrefixCache>,
    mtp_vocab_size: u32,
    comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    self_speculative: bool,
    num_drafts: usize,
    kv_dtype: KvCacheDtype,
    inference_reserve: usize,
    gpu_memory_utilization: f64,
    ssm_cache_slots: usize,
    layer_dtypes: Vec<KvCacheDtype>,
    ssm_checkpoint_interval: usize,
    // Phase 6.1.f: per-sequence HBM cache cap. `Some(N)` enables
    // `--high-speed-swap` HBM-shrink behavior. `None` preserves the
    // pre-Phase-6 unbounded behavior.
    hss_cache_blocks_per_seq: Option<u32>,
    // DFlash speculative-decoding pairing. `None` = no DFlash; existing
    // MTP / no-spec paths unchanged.
    dflash_args: Option<DflashBuildArgs<'_>>,
    // Startup-static LoRA adapter (`--lora-adapter`). `None` = base-only.
    lora_args: Option<LoraBuildArgs<'_>>,
    // NLLB / M2M-100 translation language pair (tokenizer-resolved
    // `(src_lang_id, tgt_lang_id)`), resolved server-side. `None` for all other
    // model types.
    nllb_lang: Option<(u32, u32)>,
    // NLLB / M2M-100 PEFT LoRA adapter directory (`--lora-adapter` for an
    // encoder-decoder checkpoint). `None` = base model.
    nllb_lora_dir: Option<std::path::PathBuf>,
) -> Result<Box<dyn Model>> {
    // NLLB / M2M-100 is an encoder-decoder model that cannot be represented by
    // the decoder-only TransformerModel stack. Serve it with the dedicated
    // `NllbGpuModel`, which reads its weights from the standard `store` — this
    // returns BEFORE `loader_for_config`, so the decoder-only weight loader
    // (and its fail-fast) never runs on this path.
    #[cfg(feature = "cuda")]
    if matches!(config.model_type.as_str(), "m2m_100" | "nllb") {
        let (src, tgt) = nllb_lang.ok_or_else(|| {
            anyhow::anyhow!(
                "NLLB serving requires --src-lang and --tgt-lang (translation language pair)"
            )
        })?;
        let lang = crate::model::nllb::NllbLang {
            src_lang_id: src,
            tgt_lang_id: tgt,
            decoder_start_id: config.eos_token_id,
            eos_id: config.eos_token_id,
            pad_id: 1,
        };
        let model = crate::model::nllb::NllbGpuModel::new(
            &config,
            store,
            gpu,
            lang,
            max_seq_len,
            max_batch_size,
            nllb_lora_dir.as_deref(),
        )?;
        return Ok(Box::new(model));
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (nllb_lang, nllb_lora_dir);

    // ── Step 1: Select weight loader (only model-specific dispatch) ──
    let loader = loader_for_config(&config)?;

    // ── LoRA adapter load (pre-arena, pre-KV-sizing) ──
    // MUST run before `BufferArena::new` and the `gpu.free_memory()`
    // snapshot below: the pool allocation then lands in `used_so_far`, so
    // the KV-cache budget shrinks automatically (positional budgeting —
    // no arithmetic edit needed). Do NOT move this later. Setting
    // `config.adapter_max_rank` here also lets `BufferSizes` size the
    // lora_xa/lora_delta/lora_hact scratch.
    let lora_weights: Option<crate::lora::LoraWeights> = if let Some(ref la) = lora_args {
        config.adapter_max_rank = la.max_lora_rank;
        loader.load_lora_adapters(
            &la.adapters,
            &config,
            gpu.as_ref(),
            la.max_loras,
            la.max_lora_rank,
        )?
    } else {
        None
    };

    // Pre-construction: when DFlash is active, populate the target's
    // capture-layer indices from the drafter's `dflash_config.target_layer_ids`
    // so `TransformerModel::new` allocates the 5×hidden_size capture buffer.
    //
    // HF `output_hidden_states[i]` semantics: index 0 = post-embedding,
    // index k>=1 = post-layer-(k-1). The drafter's `target_layer_ids`
    // are interpreted as HF `output_hidden_states` indices (so layer_id=1
    // means post-layer-0). Atlas captures AFTER `layer.decode()` for the
    // listed `dflash_capture_layers` index — to match HF semantics we
    // subtract 1 from each id (clamped at 0). Set
    // ATLAS_DFLASH_CAPTURE_LAYER_OFFSET=0 to disable this adjustment for
    // a back-to-back A/B test.
    if let Some(ref args) = dflash_args
        && let Some(ref sub) = args.drafter_config.dflash_config
    {
        let offset: i64 = std::env::var("ATLAS_DFLASH_CAPTURE_LAYER_OFFSET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(-1);
        config.dflash_capture_layers = sub
            .target_layer_ids
            .iter()
            .map(|&id| (id as i64 + offset).max(0) as usize)
            .collect();
        tracing::info!(
            "DFlash: target layer capture indices = {:?} (offset={offset} from raw {:?})",
            config.dflash_capture_layers,
            sub.target_layer_ids,
        );
    }

    // ── Step 2: Load weights (model-agnostic from here) ──
    let attn_layer_dtypes: Vec<KvCacheDtype> = if layer_dtypes.is_empty() {
        vec![kv_dtype; config.num_attention_layers()]
    } else {
        layer_dtypes.clone()
    };

    // Populate per-layer KV dims for heterogeneous-attention models (Gemma-4).
    // Homogeneous models return an empty Vec which the KV cache treats as
    // "use global num_kv_heads/head_dim for all layers" (backward compatible).
    config.kv_layer_dims = loader.kv_layer_dims(&config);

    let mut layers = loader.load_layers(store, &config, gpu.as_ref(), &attn_layer_dtypes)?;
    let embed = loader.load_embedding(store, &config, gpu.as_ref())?;
    let final_norm = loader.load_final_norm(store, &config, gpu.as_ref())?;
    let lm_head = loader.load_lm_head(store, &config, gpu.as_ref())?;
    let mtp_weights = loader.load_mtp_weights_multi(store, &config, gpu.as_ref())?;

    // DeepSeek-V4 ships an architecturally distinct MTP module (MLA + mHC), not
    // the Qwen-shaped `MtpWeights`. Load it via the V4-specific path and keep it
    // — the `DeepseekV4MtpHead` proposer is built from it after the model is
    // constructed (it needs the resolved draft NVFP4 LM head + the model's
    // owned GPU backend) and installed via `set_dflash_proposer`. Only built
    // when `--speculative` is set; otherwise the module is loaded for
    // verification then dropped.
    // Only rank 0 runs the MTP draft (no-EP, all experts local). Skip loading it
    // on the worker ranks — they never call propose(), so it would be dead weight.
    let v4_mtp_module =
        if config.model_type == "deepseek_v4" && use_speculative && config.ep_rank == 0 {
            match crate::weight_loader::deepseek_v4::load_v4_mtp_module(
                store,
                &config,
                gpu.as_ref(),
                &attn_layer_dtypes,
            ) {
                Ok(Some(m)) => {
                    tracing::info!(
                        "DeepSeek-V4 MTP draft module loaded OK (num_mtp_modules={})",
                        config.num_mtp_modules
                    );
                    Some(m)
                }
                Ok(None) => {
                    tracing::info!("DeepSeek-V4: no MTP module in checkpoint (MTP off)");
                    None
                }
                Err(e) => {
                    tracing::error!("DeepSeek-V4 MTP module load FAILED: {e:#}");
                    None
                }
            }
        } else {
            None
        };

    // Capability warning: user asked for `--speculative` but the model has no
    // MTP head bundled, so speculative decoding will silently no-op. Surface
    // this loudly so the user knows the flag was inert.
    if use_speculative && mtp_weights.is_empty() {
        tracing::warn!(
            "`--speculative` was requested but no MTP weights were loaded for this \
             model — speculative decoding will be disabled. Either drop `--speculative` \
             or use a checkpoint that ships an MTP head (e.g. `mtp.safetensors`)."
        );
    }
    let vision_encoder = loader.load_vision_encoder(store, &config, gpu.as_ref())?;

    // If the checkpoint's `quantization_config.ignore_modules` lists MTP
    // (e.g. Sehyo/Qwen3.5-35B-A3B-NVFP4 ignores `mtp.*`), the MTP weights
    // were stored as BF16 on disk. Runtime-quantizing them to NVFP4
    // anyway — which is what `mtp_quant` would otherwise do — produces
    // garbage drafts (vllm PR #38832). Force BF16 in that case.
    let effective_mtp_quant = if !mtp_weights.is_empty() {
        let quant_fmt = crate::quant_format::detect_quant_format(&config, store);
        if quant_fmt.is_ignored("mtp.fc.weight")
            || quant_fmt.is_ignored("mtp.layers.0.self_attn.q_proj.weight")
        {
            if mtp_quant != MtpQuantization::Bf16 {
                tracing::info!(
                    "MTP head listed in checkpoint ignore_modules — overriding \
                     --mtp-quantization {:?} → Bf16 to preserve precision",
                    mtp_quant,
                );
            }
            MtpQuantization::Bf16
        } else {
            mtp_quant
        }
    } else {
        mtp_quant
    };

    // ── Step 3: LM-head quantization (NVFP4 / FP8 / BF16-skip) + the
    // draft-only NVFP4 head for MTP — extracted to lm_head_setup.rs
    // (file-size cap; pure code move).
    let (lm_head_nvfp4, lm_head_fp8, mtp_lm_head_nvfp4) = super::lm_head_setup::setup_lm_heads(
        store,
        &lm_head,
        &config,
        gpu.as_ref(),
        use_speculative,
        !mtp_weights.is_empty(),
    )?;

    // Capture the shared embed + resolved draft NVFP4 head for the DeepSeek-V4
    // MTP proposer BEFORE `embed` / `lm_head_nvfp4` / `mtp_lm_head_nvfp4` are
    // moved into `TransformerModel::new`. All are `Copy` (DenseWeight /
    // QuantizedWeight). The draft head resolves to the separate draft-only
    // NVFP4 head (main head kept BF16) or the main NVFP4 head. `None` ⇒ no
    // NVFP4 head available ⇒ the V4 proposer can't draft and is skipped.
    let v4_mtp_embed = embed;
    // DeepSeek-V4-Flash keeps the LM head in BF16; the proposer drafts with the
    // same BF16 head via dense_gemv (drafts are re-verified by the target, so the
    // draft head only affects acceptance). DenseWeight is Copy.
    let v4_mtp_lm_head = lm_head;

    // ── Step 3b: Post-load MoE prefill transpose (MiniMax EP=2 TTFT fix) ──
    //
    // MiniMax M2.7-NVFP4 EP=2 has ~46 GB free at layer-0 load time but
    // ~65 GB free here (the BF16 lm_head just freed ~22 GB during NVFP4
    // quantization). The transpose costs ~59 GB — fits in the post-load
    // window but not the pre-load one. Other loaders (qwen35, qwen3,
    // gemma4) still call `transpose_for_prefill` inline during layer
    // construction; this default-no-op hook doesn't perturb them.
    maybe_run_minimax_m2_moe_transpose(&config, gpu.as_ref(), &mut layers)?;
    // ── Step 4: Create buffer arena ──
    let buffers = BufferArena::new(
        &config,
        max_batch_tokens,
        max_seq_len,
        kv_block_size,
        gpu.as_ref(),
    )?;

    // ── Step 5: Size KV cache from actual free memory ──
    // MLA absorbed: cache compressed latent [kv_lora + rope] instead of expanded [nkv * hd]
    // This gives 12.8x smaller KV cache AND better precision (no expand→cache→read roundtrip)
    let (kv_num_heads, kv_head_dim) = if config.kv_lora_rank > 0 {
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        tracing::info!(
            "MLA absorbed KV cache: 1 head × {} dims ({}+{}) per token (vs {} heads × {})",
            mla_cache_dim,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
            config.num_key_value_heads,
            config.head_dim,
        );
        (1, mla_cache_dim)
    } else {
        (config.num_key_value_heads, config.head_dim)
    };
    let kv_config = KvCacheConfig {
        block_size: kv_block_size,
        num_kv_heads: kv_num_heads,
        head_dim: kv_head_dim,
        num_layers: config.num_attention_layers(),
        dtype: kv_dtype,
        layer_dtypes: layer_dtypes.clone(),
        layer_dims: config.kv_layer_dims.clone(),
        cache_blocks_per_seq: hss_cache_blocks_per_seq,
    };

    if hss_cache_blocks_per_seq.is_some() {
        kv_summary::log_hss_kv_summary(&kv_config);
    }
    // ── gpu_memory_utilization as fraction of TOTAL GPU memory ──
    //
    // User-facing contract (matches vLLM / sparkrun convention):
    //   total_memory × gpu_memory_utilization = hard ceiling on everything
    //   this process consumes (weights + buffers + KV cache + reserves).
    //
    // KV cache gets whatever remains inside that ceiling after deducting
    // prior allocations (model weights, buffer arena, CUDA context/driver)
    // and the inference reserve (SSM state pools, CUDA headroom).  A safety
    // clamp ensures we never exceed what the device can physically provide
    // right now (handles external memory pressure on shared-memory /
    // unified-memory systems like GB10).
    let total_mem = gpu.total_memory()?;
    let actual_free = gpu.free_memory()?;
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let mut used_so_far = total_mem.saturating_sub(actual_free);
    // GB10 is shared (ComfyUI/voxel/etc.). Raw `used_so_far` counts those
    // co-tenants against our --gpu-memory-utilization budget, so a low util
    // needlessly starves the KV pool (vs vLLM, whose util is self-relative).
    //
    // We want the KV pool sized against Atlas's OWN footprint (weights +
    // buffers), excluding co-tenants. Two ways to find that footprint:
    //
    //   1. AUTO (default, preferred): free-at-context-init minus free-now =
    //      exactly what THIS process allocated since startup. Co-tenants that
    //      were already resident at init are in the baseline, so they cancel
    //      out — and it self-corrects as co-tenants come and go (no stale
    //      constant). Requires `set_baseline_free_bytes` to have run (it does
    //      under the real server; absent under the mock backend → we skip it).
    //
    //   2. MANUAL override: ATLAS_KV_EXTERNAL_RESERVE_GB=<co-tenant GB> still
    //      wins when explicitly set (>0), for operators who want to RESERVE
    //      headroom for co-tenants that will arrive LATER (the auto measure
    //      only sees current state).
    //
    // The `.min(actual_free - reserve)` clamp below still guarantees a physical
    // fit regardless of which path set `used_so_far`.
    let manual_reserve_gb = std::env::var("ATLAS_KV_EXTERNAL_RESERVE_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&gb| gb > 0.0);
    if let Some(gb) = manual_reserve_gb {
        let ext = (gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let discounted = used_so_far.saturating_sub(ext);
        tracing::info!(
            "ATLAS_KV_EXTERNAL_RESERVE_GB={gb} (manual override): discounting \
             external/co-tenant memory from KV budget — used_so_far {:.1} GB → \
             Atlas-own {:.1} GB",
            gib(used_so_far),
            gib(discounted),
        );
        used_so_far = discounted;
    } else if let Some(baseline) = spark_runtime::gpu::baseline_free_bytes() {
        // AUTO: bytes this process consumed since context init.
        let atlas_own = baseline.saturating_sub(actual_free);
        // Sanity-gate: baseline must be ≥ free-now, atlas_own positive and no
        // larger than total used (co-tenants can't be negative). If a co-tenant
        // *freed* memory during our load, baseline > free-now still holds and
        // atlas_own just slightly overcounts (conservative — fine). If the
        // numbers are implausible, fall back to raw used_so_far.
        if atlas_own > 0 && atlas_own <= used_so_far {
            tracing::info!(
                "KV budget self-relative (auto): baseline-free {:.1} GB − free-now \
                 {:.1} GB = Atlas-own {:.1} GB; co-tenants {:.1} GB excluded \
                 (set ATLAS_KV_EXTERNAL_RESERVE_GB to override)",
                gib(baseline),
                gib(actual_free),
                gib(atlas_own),
                gib(used_so_far - atlas_own),
            );
            used_so_far = atlas_own;
        } else {
            tracing::warn!(
                "KV budget auto-measure implausible (baseline {:.1} GB, free-now \
                 {:.1} GB, used {:.1} GB) — using raw used_so_far",
                gib(baseline),
                gib(actual_free),
                gib(used_so_far),
            );
        }
    }
    let total_budget = (total_mem as f64 * gpu_memory_utilization) as usize;
    let kv_budget = total_budget
        .saturating_sub(used_so_far)
        .saturating_sub(inference_reserve)
        .min(actual_free.saturating_sub(inference_reserve));
    // Phase 6.1.f: when HBM-shrink is active, size the production cache to
    // `max_batch_size × cache_blocks_per_seq` rather than the unbounded
    // budget-driven sum. This is the *whole point* of the HBM-shrink
    // feature — the production cache becomes write staging only; older
    // blocks live on disk under the orchestrator's control.
    let num_kv_blocks = match hss_cache_blocks_per_seq {
        Some(cap) => {
            // Phase 6.3 (original): pool = max_batch × cap + 1 dummy + 1 spare per seq.
            // Issue #31 (2026-05-08): the cap×bs sizing assumed prefill would
            // fit in cap blocks AND the slide-during-prefill path would handle
            // any overflow. Live-tested: slides during prefill produce silently
            // wrong attention output (the orchestrator-fed disk-read path is
            // wired up for DECODE attention only — Phase 6.2.a — not for
            // prefill — Phase 6.2.b deferred). The companion change in
            // `block_mgmt::ensure_blocks_through_prefill` removes the broken
            // slide; this change resizes the pool so prefill can grow up to
            // `max_seq_len` blocks without hitting "no free blocks". HBM-shrink
            // remains in effect post-prefill: the FIRST decode step finds
            // bt_len > cap and slides down via the orchestrator-aware path
            // (which IS correct).
            //
            // Sizing rationale:
            //   * Per-seq blocks: `max(cap + 1, ceil(max_seq_len / block_size))`
            //     so prefill of any prompt up to max_seq_len fits in HBM.
            //   * +1 dummy slot for OOB-safe paged-kernel reads.
            //
            // For multi-seq HSS where the user wanted strict HBM-shrink, this
            // increases pool size by `(max_seq_len_blocks - cap) × max_batch`
            // bytes per block. The existing post-load OOM check (line 304+)
            // catches infeasible configs at startup with a clear message.
            let max_seq_blocks = max_seq_len.div_ceil(kv_block_size);
            let per_seq = (cap as usize + 1).max(max_seq_blocks);
            let n = max_batch_size * per_seq + 1;
            tracing::info!(
                "--high-speed-swap: HBM cache sized to {n} blocks ({} batch × max(cap={cap}+1, max_seq_len_blocks={max_seq_blocks}) + 1 dummy); \
                 prefill grows monotonically, decode shrinks to cap × bs and streams older blocks from disk via the orchestrator",
                max_batch_size
            );
            n
        }
        None => {
            if kv_budget == 0 {
                anyhow::bail!(
                    "No memory left for KV cache: total GPU = {:.1} GB, \
                     --gpu-memory-utilization {:.0}% → budget {:.1} GB, \
                     but {:.1} GB already consumed + {:.1} GB inference reserve \
                     = {:.1} GB committed.  Raise --gpu-memory-utilization or \
                     use a smaller model.",
                    total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
                    gpu_memory_utilization * 100.0,
                    total_budget as f64 / (1024.0 * 1024.0 * 1024.0),
                    used_so_far as f64 / (1024.0 * 1024.0 * 1024.0),
                    inference_reserve as f64 / (1024.0 * 1024.0 * 1024.0),
                    (used_so_far + inference_reserve) as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
            let n = PagedKvCache::compute_num_blocks(&kv_config, kv_budget)?;
            let max_kv_tokens = n * kv_block_size;
            tracing::info!(
                "KV cache: {:.1} GB total × {:.0}% util = {:.1} GB budget; \
                 {:.1} GB pre-KV + {:.1} GB reserve → {:.1} GB for KV \
                 → {} blocks × {} tok/block = {} max KV tokens",
                total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
                gpu_memory_utilization * 100.0,
                total_budget as f64 / (1024.0 * 1024.0 * 1024.0),
                used_so_far as f64 / (1024.0 * 1024.0 * 1024.0),
                inference_reserve as f64 / (1024.0 * 1024.0 * 1024.0),
                kv_budget as f64 / (1024.0 * 1024.0 * 1024.0),
                n,
                kv_block_size,
                max_kv_tokens,
            );
            n
        }
    };
    let _max_kv_tokens = num_kv_blocks * kv_block_size;
    // Phase 6.1.f / 6.2.c — when --high-speed-swap is on with HBM-shrink, the
    // production KV cache only has to fit the per-seq HBM window, not the full
    // sequence (older blocks live on disk). Compare against `cache_blocks_per_seq`
    // in that mode; the legacy "blocks per max_seq_len" check is invalid for
    // HBM-shrunk pools by design.
    let blocks_per_seq = match hss_cache_blocks_per_seq {
        Some(cap) => cap as usize,
        None => max_seq_len.div_ceil(kv_block_size),
    };
    let max_concurrent = num_kv_blocks / blocks_per_seq.max(1);
    if max_concurrent < max_batch_size {
        // Suggest a max_seq_len that lets the requested batch size fit.
        let suggested_max_seq_len = (num_kv_blocks / max_batch_size.max(1)) * kv_block_size;
        // The check is WORST-CASE: it assumes every concurrent sequence reaches
        // --max-seq-len. With paged KV (blocks allocated on demand) that almost
        // never holds for real agent traffic (mixed/shorter sequences), so a high
        // --max-seq-len (e.g. 64K for long agent contexts) needlessly caps
        // --max-batch-size. ATLAS_KV_OVERCOMMIT=1 downgrades the hard error to a
        // warning: the scheduler admits up to max_batch_size and the pool fills on
        // demand (a genuinely over-long burst gets back-pressured by the block
        // allocator, not a boot-time refusal).
        let overcommit = matches!(
            std::env::var("ATLAS_KV_OVERCOMMIT").as_deref(),
            Ok("1") | Ok("true")
        );
        if overcommit {
            tracing::warn!(
                "KV OVERCOMMIT: pool fits {} seq(s) at full --max-seq-len={} but \
                 --max-batch-size={} requested ({} block(s)/seq, {} block(s) total). \
                 Paged KV allocates on demand; long-context bursts are back-pressured \
                 at the block allocator, not refused at boot.",
                max_concurrent,
                max_seq_len,
                max_batch_size,
                blocks_per_seq,
                num_kv_blocks,
            );
        } else {
            anyhow::bail!(
                "KV cache can hold at most {} concurrent sequence(s) at --max-seq-len={}, \
                 but --max-batch-size={} was requested. \
                 KV pool has {} block(s) of {} tokens each; each sequence needs {} block(s). \
                 Try --max-seq-len {} (keeps max_batch_size={}), reduce --max-batch-size, \
                 or set ATLAS_KV_OVERCOMMIT=1 to allow on-demand paged allocation.",
                max_concurrent,
                max_seq_len,
                max_batch_size,
                num_kv_blocks,
                kv_block_size,
                blocks_per_seq,
                suggested_max_seq_len.max(kv_block_size),
                max_batch_size,
            );
        }
    }
    let kv_cache = PagedKvCache::new(kv_config, num_kv_blocks, gpu.as_ref())?;

    // ── Step 6: Assemble model ──
    // Capture pointers for any post-construction sharing (DFlash drafter
    // shares embed_tokens + lm_head with the target). DenseWeight is Copy
    // so this clones the device pointer cheaply.
    let target_embed_for_dflash = embed.weight;
    let target_lm_head_for_dflash = lm_head.weight;
    // NVFP4 lm_head (Copy) shared with the DFlash drafter so its final logits
    // GEMM uses w4a16 instead of a BF16 dense_gemm on NVFP4-packed bytes.
    let target_lm_head_nvfp4_for_dflash = lm_head_nvfp4;
    let target_hidden_for_dflash = config.hidden_size;

    let mut model = TransformerModel::new(
        config,
        embed,
        final_norm,
        lm_head,
        lm_head_nvfp4,
        lm_head_fp8,
        mtp_lm_head_nvfp4,
        layers,
        buffers,
        kv_cache,
        mtp_weights,
        gpu,
        max_seq_len,
        max_batch_size,
        effective_mtp_quant,
        use_speculative,
        prefix_cache,
        mtp_vocab_size,
        comm,
        self_speculative,
        num_drafts,
        vision_encoder,
        ssm_cache_slots,
        ssm_checkpoint_interval,
    )?;

    // ── Step 6b: DeepSeek-V4 MTP proposer (optional, post-construction) ──
    //
    // Built here (not inside `new()`, which only knows the Qwen-shaped
    // `MtpWeights`) because it needs the model's owned GPU backend, the
    // resolved draft NVFP4 head, and the shared embedding. Installed via the
    // existing proposer setter. DFlash (below) is CLI-exclusive with
    // `--speculative`, so the two never both install.
    if let Some(v4_module) = v4_mtp_module {
        match crate::layers::DeepseekV4MtpHead::new(
            v4_module,
            v4_mtp_embed,
            v4_mtp_lm_head,
            model.config_ref(),
            model.gpu_backend(),
            mtp_vocab_size,
            max_seq_len,
        ) {
            Ok(head) => {
                model.set_dflash_proposer(std::sync::Arc::new(head));
                tracing::info!("DeepSeek-V4 MTP speculative decoding: ENABLED (single-module)");
            }
            Err(e) => tracing::warn!(
                "Failed to build DeepSeek-V4 MTP proposer: {e:#}. Speculative decoding disabled."
            ),
        }
    }

    // ── Step 7: DFlash drafter (optional, post-construction) ──
    //
    // Loaded last because it depends on the target's `embed_tokens` and
    // `lm_head` pointers (the drafter checkpoint omits these — they're
    // shared at runtime, mirroring vLLM PR #40898's `skip_substrs` flow).
    if let Some(args) = dflash_args {
        let weights = load_dflash_weights(
            args.drafter_store,
            &args.drafter_config,
            model.gpu_backend(),
            1, // tp_size for the drafter side: replicated, so always 1
        )?;
        if let Some(weights) = weights {
            let head = crate::layers::BlockDiffusionDraftHead::from_weights(
                weights,
                target_embed_for_dflash,
                target_lm_head_for_dflash,
                target_lm_head_nvfp4_for_dflash,
                target_hidden_for_dflash,
                args.gamma,
                args.window_size,
                model.gpu_backend(),
                max_seq_len,
            )?;
            model.set_dflash_proposer(std::sync::Arc::new(head));
            tracing::info!("DFlash drafter installed as the active proposer");
        } else {
            tracing::warn!(
                "DFlash drafter store had no fc.weight — proposer not installed; \
                 falling back to whatever proposer (if any) the target's MTP path built"
            );
        }
    }

    // ── Step 8: LoRA adapter install (optional, post-construction) ──
    // The pool/tables were loaded up top (pre-KV-sizing); this walk copies
    // the per-layer pairs into the layer structs. M0: layers only STORE the
    // adapter — base output is unchanged until the M1 compute insertions.
    model.set_lora_weights(lora_weights)?;

    Ok(Box::new(model))
}
