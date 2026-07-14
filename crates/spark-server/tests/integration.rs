// SPDX-License-Identifier: AGPL-3.0-only

//! Integration smoke test: loads real model weights and runs inference.
//!
//! Requires GPU hardware and model weights. Set
//! `ATLAS_INTEGRATION_MODEL_DIR` to the absolute path of a HuggingFace
//! snapshot directory before running. Tests are `#[ignore]`d by default;
//! invoke with:
//!
//!   ATLAS_INTEGRATION_MODEL_DIR=/path/to/snapshot \
//!     cargo test -p spark-server --release -- --ignored
//!
//! When the env var is unset, tests skip with a clear message instead of
//! panicking on a missing path.

use anyhow::Result;
use std::path::Path;

/// Default snapshot used when `ATLAS_INTEGRATION_MODEL_DIR` is unset.
/// Pinned to a Qwen3-Next-80B NVFP4 snapshot known to exercise the
/// hybrid SSM+attention path; override via env var to test other models.
const DEFAULT_MODEL_DIR: &str = "~/.cache/huggingface/hub/models--nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4/snapshots/8fb2682f136cf94d932a498f18cb1e428832a912";

/// Resolve the model dir from `ATLAS_INTEGRATION_MODEL_DIR` (env, with
/// `~` expansion) or fall back to `DEFAULT_MODEL_DIR`.
fn model_dir_path() -> std::path::PathBuf {
    let raw = std::env::var("ATLAS_INTEGRATION_MODEL_DIR")
        .unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    if let Some(rest) = raw.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return std::path::PathBuf::from(home).join(rest);
    }
    std::path::PathBuf::from(raw)
}

/// Helper: build model + GPU backend from model directory.
fn setup_model(
    model_dir: &Path,
) -> Result<(
    Box<dyn spark_model::traits::Model>,
    atlas_core::config::ModelConfig,
)> {
    let config_path = model_dir.join("config.json");
    let config_json = std::fs::read_to_string(&config_path)?;
    let config: atlas_core::config::ModelConfig = serde_json::from_str(&config_json)?;
    tracing::info!(
        "Config: {} layers, vocab={}, hidden={}",
        config.num_hidden_layers,
        config.vocab_size,
        config.hidden_size
    );

    let ptx_modules = atlas_kernels::ptx_modules();
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &ptx_modules)?,
    );
    let total = gpu.total_memory()?;
    let free = gpu.free_memory()?;
    tracing::info!(
        "GPU: {:.1} GB total, {:.1} GB free",
        total as f64 / (1 << 30) as f64,
        free as f64 / (1 << 30) as f64,
    );

    let loader = spark_runtime::weights::SafetensorsLoader {
        ep_rank: 0,
        ep_world_size: 1,
        num_experts: 0,
        peak_memory_multiplier: None,
    };
    use spark_runtime::weights::WeightLoader;
    let store = loader.load(model_dir, gpu.as_ref(), 1024 * 1024 * 1024)?;
    tracing::info!(
        "Loaded {} tensors ({:.2} GB)",
        store.len(),
        store.total_bytes() as f64 / (1 << 30) as f64,
    );

    let post_weight_free = gpu.free_memory()?;
    let kv_budget = (post_weight_free as f64 * 0.85) as usize;
    let block_size = 16;
    let kv_config = spark_runtime::kv_cache::KvCacheConfig {
        block_size,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        num_layers: config.num_attention_layers(),
        dtype: spark_runtime::kv_cache::KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let num_blocks =
        spark_runtime::kv_cache::PagedKvCache::compute_num_blocks(&kv_config, kv_budget)?;
    tracing::info!("KV cache: {} blocks", num_blocks);

    let prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache> =
        Box::new(spark_runtime::prefix_cache::NoPrefixCaching);
    let model = spark_model::factory::build_model(
        config.clone(),
        &store,
        gpu,
        4,          // max_batch_tokens: up to 3 spec-decode verification tokens
        block_size, // kv_block_size = 16
        4096,       // max_seq_len
        8,          // max_batch_size
        spark_model::layers::MtpQuantization::Nvfp4,
        false, // use_speculative
        prefix_cache,
        0,     // mtp_vocab_size
        None,  // no EP comm backend
        false, // self_speculative
        1,     // num_drafts
        spark_runtime::kv_cache::KvCacheDtype::Fp8,
        1024 * 1024 * 1024, // inference_reserve: 1 GB
        0.90,               // gpu_memory_utilization
        0,                  // ssm_cache_slots
        Vec::new(),         // layer_dtypes
        0,                  // ssm_checkpoint_interval
        None,               // hss_cache_blocks_per_seq
        None,               // dflash_args (no speculative-decoding pairing)
    )?;

    Ok((model, config))
}

/// Helper: generate tokens from a prompt, returning (generated_token_ids, tok_per_sec).
fn generate(
    model: &dyn spark_model::traits::Model,
    config: &atlas_core::config::ModelConfig,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
) -> Result<(Vec<u32>, f64)> {
    let mut seq = model.alloc_sequence()?;
    let eos = config.eos_token_id;

    // Just do simple prefill + generate like the smoke test
    let logits = model.prefill(prompt_tokens, &mut seq, 0)?;
    let first_token = model.argmax_on_device(logits, 0)?;

    let mut generated = vec![first_token];
    if first_token == eos {
        return Ok((generated, 0.0));
    }

    // Decode remaining tokens
    let start = std::time::Instant::now();
    for _ in 1..max_new_tokens {
        let last = *generated.last().unwrap();
        let logits = model.decode(last, &mut seq, 0)?;
        let token = model.argmax_on_device(logits, 0)?;
        generated.push(token);
        if token == eos {
            break;
        }
    }
    let elapsed = start.elapsed();
    let decode_tokens = generated.len().saturating_sub(1).max(1);
    let tok_per_sec = decode_tokens as f64 / elapsed.as_secs_f64();

    Ok((generated, tok_per_sec))
}

/// Smoke test: parse config, load weights, build model, run one decode step.
#[test]
#[ignore] // Requires GPU + model weights
fn smoke_test_single_decode() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init()
        .ok();

    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set ATLAS_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();

    let (model, config) = setup_model(model_dir)?;
    tracing::info!("Model built successfully");

    // Single decode step (BOS token)
    let mut seq = model.alloc_sequence()?;
    let bos_token = config.bos_token_id;
    let logits_ptr = model.decode(bos_token, &mut seq, 0)?;
    assert!(!logits_ptr.is_null(), "Logits pointer should not be null");
    assert_eq!(seq.seq_len, 1);

    let vocab_size = model.vocab_size();
    let best_idx = model.argmax_on_device(logits_ptr, 0)?;
    tracing::info!("GPU argmax token: {}, vocab_size: {}", best_idx, vocab_size);
    assert!((best_idx as usize) < vocab_size);

    // Verify logits are finite
    let mut logits_host = vec![0u8; vocab_size * 2];
    model.copy_logits_to_host(logits_ptr, &mut logits_host)?;
    let idx = best_idx as usize;
    let lo = logits_host[idx * 2];
    let hi = logits_host[idx * 2 + 1];
    let best_val = f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16);
    tracing::info!("Best logit value: {:.4}", best_val);
    assert!(best_val.is_finite());

    // Extended generation with per-step timing (200 tokens for robust statistics)
    let num_steps = 200;
    let mut tokens = vec![best_idx];
    let mut step_times = Vec::with_capacity(num_steps);
    for step in 0..num_steps {
        let last_token = *tokens.last().unwrap();
        let t0 = std::time::Instant::now();
        let logits_ptr = model.decode(last_token, &mut seq, 0)?;
        let token = model.argmax_on_device(logits_ptr, 0)?;
        let dt = t0.elapsed();
        step_times.push(dt);
        tokens.push(token);
        if step < 10 || step % 50 == 49 {
            tracing::info!(
                "Step {}: token {} (seq_len={}, {:.1}ms)",
                step + 1,
                token,
                seq.seq_len,
                dt.as_secs_f64() * 1000.0,
            );
        }
    }
    // Report timing: exclude first step (graph capture) for steady-state
    let replay_times: Vec<f64> = step_times[1..]
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .collect();
    let avg_ms = replay_times.iter().sum::<f64>() / replay_times.len() as f64;
    let min_ms = replay_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = replay_times.iter().cloned().fold(0.0f64, f64::max);
    let mut sorted = replay_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_ms = sorted[sorted.len() / 2];
    let p95_ms = sorted[(sorted.len() as f64 * 0.95) as usize];
    let capture_ms = step_times[0].as_secs_f64() * 1000.0;
    let total_elapsed: f64 = step_times.iter().map(|t| t.as_secs_f64()).sum();
    let tok_per_sec = num_steps as f64 / total_elapsed;
    tracing::info!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        num_steps,
        total_elapsed,
        tok_per_sec,
    );
    tracing::info!(
        "Step timing: capture={:.1}ms, replay avg={:.1}ms p50={:.1}ms p95={:.1}ms min={:.1}ms max={:.1}ms",
        capture_ms,
        avg_ms,
        p50_ms,
        p95_ms,
        min_ms,
        max_ms,
    );
    assert_eq!(seq.seq_len, num_steps + 1);

    tracing::info!("Integration smoke test passed!");
    Ok(())
}

/// Coherence test: ask a factual question and verify the answer is correct.
///
/// Uses the chat template to encode "What is the capital of France?"
/// and verifies the output contains "Paris".
#[test]
#[ignore] // Requires GPU + model weights
fn coherence_test_capital_of_france() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .try_init()
        .ok();

    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set ATLAS_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();

    let (model, config) = setup_model(model_dir)?;

    // Load tokenizer
    use spark_server::tokenizer::ChatTokenizer;
    let tokenizer = ChatTokenizer::from_model_dir(
        model_dir,
        config.eos_token_id,
        false,
        &config.model_type,
        None,
    )?;

    // Encode prompt with chat template
    let messages = vec![(
        "user".to_string(),
        "What is the capital of France?".to_string(),
    )];
    let prompt_tokens = tokenizer.apply_chat_template(&messages, false, &[])?;
    tracing::info!(
        "Prompt: {} tokens: {:?}",
        prompt_tokens.len(),
        prompt_tokens
    );

    // Generate
    let (gen_tokens, tok_per_sec) = generate(model.as_ref(), &config, &prompt_tokens, 200)?;
    let output_text = tokenizer.decode(&gen_tokens)?;
    tracing::info!(
        "Generated {} tokens ({:.1} tok/s):\n  {}",
        gen_tokens.len(),
        tok_per_sec,
        output_text,
    );

    // Assertions
    let output_lower = output_text.to_lowercase();
    assert!(
        output_lower.contains("paris"),
        "Expected output to contain 'Paris', got: {output_text}"
    );

    // Check for degenerate repetition — but allow short answers that hit EOS.
    // A correct "Paris" answer in 8 tokens is fine; degenerate means 50+ tokens
    // of the same repeated pattern.
    let unique_tokens: std::collections::HashSet<u32> = gen_tokens.iter().copied().collect();
    let hit_eos = gen_tokens.last().copied() == Some(config.eos_token_id);
    if !hit_eos {
        assert!(
            unique_tokens.len() >= 10,
            "Output is degenerate (only {} unique tokens, no EOS): {:?}",
            unique_tokens.len(),
            gen_tokens,
        );
    }

    tracing::info!("Coherence test PASSED: output contains 'Paris'");
    Ok(())
}

/// Streaming test: verifies generate_streaming produces the same tokens
/// and calls the callback for each one.
#[test]
#[ignore] // Requires GPU + model weights
fn streaming_coherence_test() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .try_init()
        .ok();

    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set ATLAS_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();

    let (model, config) = setup_model(model_dir)?;

    use spark_server::tokenizer::ChatTokenizer;
    let tokenizer = ChatTokenizer::from_model_dir(
        model_dir,
        config.eos_token_id,
        false,
        &config.model_type,
        None,
    )?;

    let messages = vec![(
        "user".to_string(),
        "What is the capital of France?".to_string(),
    )];
    let prompt_tokens = tokenizer.apply_chat_template(&messages, false, &[])?;
    tracing::info!("Prompt: {} tokens", prompt_tokens.len());

    let params = spark_runtime::sampler::SamplingParams {
        stop_token_ids: vec![config.eos_token_id],
        ..spark_runtime::sampler::SamplingParams::greedy(200)
    };

    // Collect streamed tokens via callback
    let mut streamed_tokens = Vec::new();
    let start = std::time::Instant::now();
    let result = spark_model::engine::generate_streaming(
        model.as_ref(),
        &prompt_tokens,
        &params,
        |token| {
            streamed_tokens.push(token);
        },
    )?;
    let elapsed = start.elapsed();

    let output_text = tokenizer.decode(&result.output_tokens)?;
    let tok_per_sec = result.output_tokens.len() as f64 / elapsed.as_secs_f64();
    tracing::info!(
        "Streaming: {} tokens in {:.2}s ({:.1} tok/s):\n  {}",
        result.output_tokens.len(),
        elapsed.as_secs_f64(),
        tok_per_sec,
        output_text,
    );

    // Verify callback received same tokens as result
    assert_eq!(
        streamed_tokens, result.output_tokens,
        "Streamed tokens should match final output tokens"
    );

    // Verify coherence
    let output_lower = output_text.to_lowercase();
    assert!(
        output_lower.contains("paris"),
        "Expected output to contain 'Paris', got: {output_text}"
    );

    tracing::info!(
        "Streaming coherence test PASSED: {} tokens streamed, output contains 'Paris'",
        streamed_tokens.len()
    );
    Ok(())
}

/// Speculative decoding test: uses MTP to draft tokens and verify coherence.
///
/// Exercises the full generate_speculative() path through the engine.
#[test]
#[ignore] // Requires GPU + model weights
fn speculative_decode_coherence() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .try_init()
        .ok();

    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set ATLAS_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();

    let (model, config) = setup_model(model_dir)?;
    assert!(model.has_proposer(), "MTP proposer should be enabled");

    use spark_server::tokenizer::ChatTokenizer;
    let tokenizer = ChatTokenizer::from_model_dir(
        model_dir,
        config.eos_token_id,
        false,
        &config.model_type,
        None,
    )?;

    let messages = vec![(
        "user".to_string(),
        "What is the capital of France?".to_string(),
    )];
    let prompt_tokens = tokenizer.apply_chat_template(&messages, false, &[])?;
    tracing::info!("Prompt: {} tokens", prompt_tokens.len());

    let params = spark_runtime::sampler::SamplingParams {
        stop_token_ids: vec![config.eos_token_id],
        ..spark_runtime::sampler::SamplingParams::greedy(200)
    };

    let start = std::time::Instant::now();
    let result =
        spark_model::engine::generate_speculative(model.as_ref(), &prompt_tokens, &params, 2)?;
    let elapsed = start.elapsed();

    let output_text = tokenizer.decode(&result.output_tokens)?;
    let tok_per_sec = result.output_tokens.len() as f64 / elapsed.as_secs_f64();
    tracing::info!(
        "Speculative: {} tokens in {:.2}s ({:.1} tok/s), reason={}:\n  {}",
        result.output_tokens.len(),
        elapsed.as_secs_f64(),
        tok_per_sec,
        result.finish_reason,
        output_text,
    );

    let output_lower = output_text.to_lowercase();
    assert!(
        output_lower.contains("paris"),
        "Expected output to contain 'Paris', got: {output_text}"
    );

    tracing::info!("Speculative decode coherence test PASSED");
    Ok(())
}

/// Legacy /v1/completions echo+logprobs: prompt-token logprob collection
/// during chunked prefill (the loglikelihood-scoring core).
///
/// Verifies, on a real model:
/// 1. entries == prompt_len - 1 (position i scores tokens[i+1]; the
///    final position — scoring the first GENERATED token — is excluded);
/// 2. every logprob is finite and <= 0 (valid log-probability);
/// 3. token_id fields equal the actual next prompt tokens;
/// 4. top-k alternatives are sorted descending and contain the scored
///    token's logprob consistently (the target's logprob can never
///    exceed the top-1 alternative);
/// 5. a control sequence WITHOUT the flag collects nothing (zero-cost
///    default path untouched).
#[test]
#[ignore] // Requires GPU + model weights (ATLAS_INTEGRATION_MODEL_DIR)
fn prompt_logprobs_collection_during_prefill() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .try_init()
        .ok();
    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set ATLAS_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();
    let (model, config) = setup_model(model_dir)?;
    use spark_server::tokenizer::ChatTokenizer;
    let tokenizer = ChatTokenizer::from_model_dir(
        model_dir,
        config.eos_token_id,
        false,
        &config.model_type,
        None,
    )?;

    let prompt = "The capital of France is Paris. The capital of Germany is";
    let prompt_tokens = tokenizer.encode(prompt)?;
    let n = prompt_tokens.len();
    assert!(n >= 4, "prompt too short to exercise scoring");

    // Collecting sequence: k=2 alternatives.
    let mut seq = model.alloc_sequence()?;
    seq.collect_prompt_logprobs = Some(2);
    let _ = model.prefill_chunk(&prompt_tokens, &mut seq, 0, n, true, 0)?;

    assert_eq!(
        seq.prompt_logprobs.len(),
        n - 1,
        "one entry per prompt position scoring the NEXT prompt token"
    );
    for (i, lp) in seq.prompt_logprobs.iter().enumerate() {
        assert_eq!(lp.token_id, prompt_tokens[i + 1], "target at position {i}");
        assert!(lp.logprob.is_finite(), "logprob finite at {i}");
        assert!(lp.logprob <= 0.0, "logprob <= 0 at {i}: {}", lp.logprob);
        assert_eq!(lp.top.len(), 2, "top-k size at {i}");
        assert!(lp.top[0].1 >= lp.top[1].1, "top sorted desc at {i}");
        assert!(
            lp.logprob <= lp.top[0].1 + 1e-4,
            "target logprob cannot exceed top-1 at {i}"
        );
    }
    // The high-certainty continuation ("... Germany is" scored against
    // the actual next token) sanity: total loglikelihood is negative
    // and not astronomically so on a coherent prompt.
    let total: f32 = seq.prompt_logprobs.iter().map(|l| l.logprob).sum();
    assert!(total < 0.0 && total > -200.0, "plausible total ll: {total}");
    model.free_sequence(&mut seq)?;

    // Control: default path collects nothing.
    let mut seq2 = model.alloc_sequence()?;
    let _ = model.prefill_chunk(&prompt_tokens, &mut seq2, 0, n, true, 0)?;
    assert!(
        seq2.prompt_logprobs.is_empty(),
        "no collection without the flag"
    );
    model.free_sequence(&mut seq2)?;
    Ok(())
}
