// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen4ExpMtpHead::new`, split out of `qwen4_exp_mtp.rs` to keep it under
//! the 500-line cap.

use super::*;

impl Qwen4ExpMtpHead {
    // `pub(crate)`: the signature now names `Exl3LmHead`, which is a
    // crate-private type (the native head is an internal dispatch arm). The
    // only caller is `model/impl_b3_accessors.rs`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        module: Qwen4ExpMtpModule,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: Option<crate::weight_map::QuantizedWeight>,
        lm_head_exl3: Option<std::sync::Arc<crate::model::lm_head_exl3::Exl3LmHead>>,
        config: &avarok_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
        max_sequences: usize,
    ) -> Result<Self> {
        // ── Full pre-shard view, derived BEFORE anything is sized. ──
        //
        // The drafter is replicated, so it must be built end-to-end at full
        // width: its weights, its private KV cache, and its arena. Deriving
        // this only for the forward pass (and leaving `new` sizing from the
        // TP-divided config) allocated a 1-KV-head draft cache under a 2-KV-head
        // forward, and `run_mtp_propose_batched` failed its D2D copy with
        // CUDA_ERROR_INVALID_VALUE at C=4 — the batched propose fell back and
        // C=4 read 46.8 tok/s against EP-only's 68.7.
        //
        // Correct whichever config the caller passes: an already-full one
        // (tp_world_size == 1) clones unchanged; a divided one multiplies back.
        let cfg = {
            let mut c = config.clone();
            let tp = config.tp_world_size.max(1);
            if tp > 1 {
                c.num_attention_heads *= tp;
                c.num_key_value_heads *= tp;
                c.linear_num_key_heads *= tp;
                c.linear_num_value_heads *= tp;
                c.tp_world_size = 1;
                c.tp_rank = 0;
            }
            c
        };
        // Shadow: every `config.` below now reads the drafter's own geometry.
        let config = &cfg;
        let h = config.hidden_size;
        let hc = config.hc_mult.max(1);
        let row = h * 2;
        // FP32 highway (4 B/elem) — see the note in `draft_hidden`.
        let streams_bytes = hc * h * 4;

        // The body was built with `attn_idx = 0`, so this pool needs exactly
        // ONE layer — and the body must NEVER be handed the main model's pool,
        // where index 0 is full-attention layer 0's live K/V.
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let num_blocks =
            state::draft_pool_blocks(max_seq_len, kv_config.block_size, max_sequences)?;
        let (bs, kvh, hd) = (
            kv_config.block_size,
            kv_config.num_kv_heads,
            kv_config.head_dim,
        );
        let kv_bytes = [num_blocks, bs, kvh, hd, 2, 2]
            .into_iter()
            .try_fold(1usize, |bytes, dim| bytes.checked_mul(dim))
            .ok_or_else(|| anyhow::anyhow!("Qwen MTP KV pool byte size overflow"))?;
        anyhow::ensure!(
            kv_bytes <= gpu.free_memory()?,
            "Qwen MTP private KV pool needs {kv_bytes} bytes for {max_sequences} sequence slots; insufficient free GPU memory"
        );
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;
        tracing::info!(
            "qwen4_exp MTP head: private KV pool {} blocks x {} tok = {} tokens, \
             {} kv_heads x {} head_dim BF16 (~{:.2} GB), capacity {max_sequences} sequence slots ({kv_bytes} bytes). This is allocated AFTER \
             the main pool and is therefore OUTSIDE the util pledge.",
            num_blocks,
            bs,
            num_blocks * bs,
            kvh,
            hd,
            kv_bytes as f64 / 1e9,
        );

        // T=1 arena for the draft. `max_batch_tokens = 1` keeps every
        // token-scaled buffer at one row; `max_seq_len` still sizes the scratch
        // block-table region, and kv_block_size must match this head's own pool.
        let free_before = gpu.free_memory().unwrap_or(0);
        // Rows 1..cap exist for the batched propose; row 0 is what the
        // per-sequence path always used. See `draft_bodies_batched`.
        let batch_cap = max_sequences.clamp(1, BATCH_CAP);
        let arena = spark_runtime::buffers::BufferArena::new(
            config,
            batch_cap,
            max_seq_len,
            16,
            batch_cap,
            gpu,
        )?;
        let free_after = gpu.free_memory().unwrap_or(0);
        tracing::info!(
            "qwen4_exp MTP head: private {}-row buffer arena costs {:.3} GB (rows 1.. serve \
             the batched propose; row 0 is the per-sequence path). The draft runs entirely \
             inside it, so it cannot reach the target's buffers.",
            batch_cap,
            (free_before.saturating_sub(free_after)) as f64 / 1e9,
        );

        Ok(Self {
            module,
            embed_tokens,
            cfg: cfg.clone(),
            kv_cache: Mutex::new(kv_cache),
            arena,
            batch_cap,
            w4a16_batchm: crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu),
            argmax_batch_k: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_batch"),
            buf: MtpBuffers {
                streams: gpu.alloc(streams_bytes)?,
                normed_streams: gpu.alloc(streams_bytes)?,
                embed: gpu.alloc(row)?,
                normed_embed: gpu.alloc(row)?,
                embed_proj: gpu.alloc(row)?,
                body_scratch: gpu.alloc(row)?,
                residual: gpu.alloc(row)?,
                // `hc_head_lowrank`'s decode-split layout is
                // `t * (hc_mult*h + hc_lowrank) * 4` — normed FP32 [t, hc*H]
                // THEN low FP32 [t, rank]. Sizing this `hc*h*4` (the streams
                // alone) under-allocates by `rank*4` = 1280 B and the collapse
                // writes past the end of the allocation. Measured, not guessed.
                per_stream: gpu.alloc(hc * h * 2)?,
                head_scratch: gpu.alloc((hc * h + config.hc_lowrank.max(1)) * 4)?,
                batch_h_out: gpu.alloc(BATCH_CAP * row)?,
                batch_logits: gpu.alloc(BATCH_CAP * config.vocab_size * 2)?,
                batch_tok: gpu.alloc(BATCH_CAP * 4)?,
                logits_stash: gpu.alloc(config.vocab_size * 2)?,
            },
            // Avarok's offset-from-1 rms_norm, NOT V4's `rms_norm_vanilla`:
            // this checkpoint's norm weights are offset-from-1 like the rest of
            // the qwen4_exp tree.
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            lm_head_nvfp4,
            lm_head_exl3,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: KernelHandle(0),
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            hc_stage_k: gpu.kernel("hyper_connection", "hc_pre_stage_bf16")?,
            combine_k: gpu.kernel("hyper_connection", "qhc_mtp_combine_streams")?,
            shadow_drafts: AtomicU64::new(0),
            shadow_hits: AtomicU64::new(0),
        })
    }
}
