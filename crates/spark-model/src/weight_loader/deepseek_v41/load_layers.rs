// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;
use spark_runtime::weights::expert_stream::{
    EngramRowReader, ExpertLru, ExpertSliceMap, ExpertSource, PinnedArena, ShardFiles,
};

use crate::layer::TransformerLayer;
use crate::layers::attn_v41::{
    AttnV41, AttnV41Cfg, AttnV41LayerWeights, CompressorWeightsGpu, IndexerWeightsGpu, LayerRole,
    SharedV41,
};
use crate::layers::deepseek_v41_layer::{DeepSeekV41Layer, V41Runtime};
use crate::layers::engram_v41::{EngramHashTables, EngramHasher, EngramLayerWeights, EngramV41};
use crate::layers::moe_v41::{MoeV41, MoeV41Cfg, MoeV41LayerWeights};
use crate::weight_map::DenseWeight;

use super::{
    DEFAULT_CANDIDATE_BLOCK, DEFAULT_CANDIDATE_SOURCE, DEFAULT_CANDIDATE_TOPK_BLOCKS, bf16_ptr,
    download_f32, env_usize, f32_ptr, hc_site, resident_mat,
};

pub(super) fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    _layer_kv_dtypes: &[spark_runtime::kv_cache::KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let n_layers = config.num_hidden_layers;
    let (dim, hc) = (config.hidden_size, config.hc_mult);
    ensure!(
        (1..=4).contains(&hc),
        "deepseek-v4.1: hc_mult {hc} outside 1..=4"
    );
    let head_dim = config.head_dim;
    let nh = config.num_attention_heads;

    // ── the streamed tensors: shards, expert map, engram rows ──
    let anchor = store
        .deferred("model.layers.0.ffn.experts_stack.gate")
        .context("deepseek-v4.1: the routed expert stacks were not deferred by the GGUF loader")?;
    let model_dir = anchor.path.parent().context("shard path has no parent")?;
    let files = Arc::new(ShardFiles::open_dir(model_dir)?);
    let slices = ExpertSliceMap::new(files.clone())?;
    let rows = EngramRowReader::new(files.clone())?;
    ensure!(
        slices.num_experts() == config.num_experts,
        "expert stacks hold {} experts, config says {}",
        slices.num_experts(),
        config.num_experts
    );

    // ── geometry ──
    let max_seq = env_usize("ATLAS_DS41_MAX_SEQ", 8192).min(config.max_position_embeddings.max(1));
    let max_tokens = env_usize("ATLAS_DS41_MAX_TOKENS", 2048).min(max_seq);
    let cache_gib = env_usize("ATLAS_DS41_EXPERT_CACHE_GIB", 88);
    let reader_threads = env_usize("ATLAS_DS41_READER_THREADS", 8);
    let ratios: Vec<usize> = config
        .compress_ratios
        .iter()
        .take(n_layers)
        .copied()
        .collect();
    ensure!(
        ratios.len() == n_layers,
        "compress_ratios has {} entries for {n_layers} layers",
        ratios.len()
    );
    let has = |l: usize, t: &str| store.contains(&format!("model.layers.{l}.{t}"));
    let kv_sources: Vec<usize> = (0..n_layers)
        .filter(|&l| has(l, "compressor.wkv.weight"))
        .collect();
    let index_sources: Vec<usize> = (0..n_layers)
        .filter(|&l| has(l, "indexer.wq_b.weight"))
        .collect();
    ensure!(
        !kv_sources.is_empty() && !index_sources.is_empty(),
        "deepseek-v4.1: no compressor / indexer tensors found"
    );
    let cand_src = config
        .candidate_source_layer_id
        .unwrap_or_else(|| env_usize("ATLAS_DS41_CANDIDATE_SOURCE", DEFAULT_CANDIDATE_SOURCE));
    let cand_topk_blocks = if config.candidate_topk_blocks > 0 {
        config.candidate_topk_blocks
    } else {
        DEFAULT_CANDIDATE_TOPK_BLOCKS
    };
    let cand_block = if config.candidate_block_size > 0 {
        config.candidate_block_size
    } else {
        DEFAULT_CANDIDATE_BLOCK
    };
    let attn_cfg = AttnV41Cfg {
        dim,
        n_heads: nh,
        head_dim,
        rope_dim: config.rotary_dim,
        q_rank: config.q_lora_rank,
        o_rank: config.o_lora_rank,
        groups: config.o_groups.max(1),
        window: config.sliding_window as usize,
        eps: config.rms_norm_eps as f32,
        index_heads: config.index_n_heads,
        index_hd: config.index_head_dim,
        index_topk: config.index_topk,
        cand_topk_blocks,
        cand_block,
        max_seq,
        max_tokens,
        rope_theta: config.rope_theta as f32,
        compress_rope_theta: config.compress_rope_theta,
        rope_factor: if config.yarn_factor > 0.0 {
            config.yarn_factor
        } else {
            16.0
        },
        orig_seq: if config.yarn_original_max_position_embeddings > 0 {
            config.yarn_original_max_position_embeddings
        } else {
            65536
        },
        beta_fast: if config.yarn_beta_fast > 0.0 {
            config.yarn_beta_fast
        } else {
            32.0
        },
        beta_slow: if config.yarn_beta_slow > 0.0 {
            config.yarn_beta_slow
        } else {
            1.0
        },
    };
    let moe_cfg = MoeV41Cfg {
        dim,
        inter: config.moe_intermediate_size,
        n_routed: config.num_experts,
        topk: config.num_experts_per_tok,
        gate_temp: 1.0,
        norm_topk_prob: config.norm_topk_prob,
        route_scale: config.routed_scaling_factor as f32,
        swiglu_limit: config.swiglu_limit,
        max_tokens,
    };
    tracing::info!(
        "DeepSeek-V4.1: {n_layers} layers, kv sources {kv_sources:?}, index sources {index_sources:?}, candidate source {cand_src} ({cand_topk_blocks} x {cand_block}), window {}, max_seq {max_seq}, max_tokens {max_tokens}, expert cache {cache_gib} GiB, {reader_threads} readers",
        attn_cfg.window
    );

    // ── engram tables from the GGUF metadata ──
    let token_map = files
        .header(0)
        .get_i64_array("deepseek41.engram.token_map")
        .context("deepseek41.engram.token_map missing from the GGUF metadata")?;
    let tables = Arc::new(EngramHashTables::from_flat(
        config.engram_layer_ids.clone(),
        config.engram_max_ngram_size,
        config.engram_n_heads,
        config.engram_pad_token_id,
        token_map,
        &config.engram_multipliers,
        &config.engram_primes,
        &config.engram_offsets,
    )?);
    let cols = tables.n_hash_cols();

    // ── the runtime shared by all layers ──
    let layout = slices.slot_layout();
    let arena = PinnedArena::alloc(gpu, cache_gib << 30)?;
    let lru = ExpertLru::new(arena.host(), arena.dev(), arena.bytes(), layout)?;
    tracing::info!(
        "DeepSeek-V4.1: expert cache {} slots of {:.2} MiB (page-locked, device-visible)",
        lru.n_slots(),
        layout.bytes as f64 / 1048576.0
    );
    let attn = AttnV41::new(gpu, attn_cfg.clone())?;
    let moe = MoeV41::new(gpu, moe_cfg.clone())?;
    let mut engram = EngramV41::new(
        gpu,
        dim,
        hc,
        config.engram_head_dim,
        cols,
        config.rms_norm_eps as f32,
        max_tokens,
    )?;
    for &l in &config.engram_layer_ids {
        let lp = format!("model.layers.{l}");
        let q = download_f32(gpu, store, &format!("{lp}.engram.wq"))?;
        let k = download_f32(gpu, store, &format!("{lp}.engram.wk"))?;
        ensure!(
            q.len() == hc * dim && k.len() == hc * dim,
            "engram q/k of layer {l}: {} / {} elements, expected {}",
            q.len(),
            k.len(),
            hc * dim
        );
        engram.add_layer(EngramLayerWeights {
            layer: l,
            wkv: bf16_ptr(store, &format!("{lp}.engram.wkv"))?,
            qk: EngramV41::upload_qk(gpu, &q, &k)?,
        });
    }
    let alloc_f32 = |n: usize| gpu.alloc((n * 4).max(16));
    let rt = Arc::new(V41Runtime {
        attn: Mutex::new(attn),
        moe: Mutex::new(moe),
        engram: Mutex::new(engram),
        lru: Mutex::new(lru),
        arena,
        slices,
        rows,
        hasher: Mutex::new(EngramHasher::new(tables.clone(), max_seq)),
        tables,
        shared: Mutex::new(SharedV41::default()),
        step_hashes: Mutex::new(None),
        pre_prev: alloc_f32(max_tokens * hc)?,
        mixes_s: alloc_f32(max_tokens * (2 + hc) * hc)?,
        reader_threads,
        n_layers,
        hc_mult: hc,
        hidden: dim,
        sinkhorn_iters: config.hc_sinkhorn_iters.max(1),
        hc_eps: config.hc_eps,
        norm_eps: config.rms_norm_eps as f32,
        pre_a: alloc_f32(max_tokens * hc)?,
        pre_f: alloc_f32(max_tokens * hc)?,
        post_s: alloc_f32(max_tokens * hc)?,
        comb_s: alloc_f32(max_tokens * hc * hc)?,
        attn_in: gpu.alloc(max_tokens * dim * 2)?,
        attn_cfg: attn_cfg.clone(),
        moe_cfg,
        max_tokens,
        step_moe: Mutex::new(Default::default()),
        step_attn_ms: Mutex::new(0.0),
        step_engram_ms: Mutex::new(0.0),
        step_start: Mutex::new(None),
    });

    // ── kernels shared by every layer ──
    let k_hc_expand = gpu.kernel("hyper_connection", "hc_expand")?;
    let k_hc_post = gpu.kernel("hyper_connection", "hc_post")?;
    let k_mixes_dot = gpu.kernel("hc_v41", "hc_v41_mixes_dot")?;
    let k_mixes_finish = gpu.kernel("hc_v41", "hc_v41_mixes_finish")?;
    let k_collapse = gpu.kernel("hc_v41", "hc_v41_collapse")?;
    // V4.1 norm weights are plain (`w * x_normed`); the shared `rms_norm`
    // kernel applies the zero-centered `(1 + w)` convention, so every
    // DeepSeek-V4.1 norm goes through the vanilla twin (the model-level
    // final norm through `ships_vanilla_norm_weights`).
    let k_rms_norm = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;

    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let lp = format!("model.layers.{l}");
        let ratio = ratios[l];
        let role = LayerRole {
            ratio,
            is_kv_source: kv_sources.contains(&l),
            is_index_source: index_sources.contains(&l),
            is_candidate_source: cand_src == l,
            uses_candidates: cand_src < l,
        };
        let comp = if role.is_kv_source {
            let norm = f32_ptr(
                gpu,
                store,
                &format!("{lp}.compressor.norm.weight"),
                head_dim,
            )?;
            if ratio > 1 {
                Some(CompressorWeightsGpu {
                    kv: f32_ptr(
                        gpu,
                        store,
                        &format!("{lp}.compressor.wkv.weight"),
                        head_dim * dim,
                    )?,
                    gate: Some(f32_ptr(
                        gpu,
                        store,
                        &format!("{lp}.compressor.wgate.weight"),
                        head_dim * dim,
                    )?),
                    norm,
                })
            } else {
                Some(CompressorWeightsGpu {
                    kv: bf16_ptr(store, &format!("{lp}.compressor.wkv.weight"))?,
                    gate: None,
                    norm,
                })
            }
        } else {
            None
        };
        let idx = if role.is_index_source {
            Some(IndexerWeightsGpu {
                wq_b: bf16_ptr(store, &format!("{lp}.indexer.wq_b.weight"))?,
                weights_proj: bf16_ptr(store, &format!("{lp}.indexer.proj.weight"))?,
                wk: if role.is_kv_source {
                    Some(bf16_ptr(store, &format!("{lp}.indexer.wk.weight"))?)
                } else {
                    None
                },
                k_norm: if role.is_kv_source {
                    Some(f32_ptr(
                        gpu,
                        store,
                        &format!("{lp}.indexer.k_norm.weight"),
                        config.index_head_dim,
                    )?)
                } else {
                    None
                },
            })
        } else {
            None
        };
        let attn_w = AttnV41LayerWeights {
            role,
            sink: f32_ptr(gpu, store, &format!("{lp}.attn.attn_sink"), nh)?,
            wq_a: resident_mat(store, &format!("{lp}.attn.wq_a.weight"))?,
            q_norm: f32_ptr(
                gpu,
                store,
                &format!("{lp}.attn.q_norm.weight"),
                config.q_lora_rank,
            )?,
            wq_b: resident_mat(store, &format!("{lp}.attn.wq_b.weight"))?,
            wkv: resident_mat(store, &format!("{lp}.attn.wkv.weight"))?,
            kv_norm: f32_ptr(gpu, store, &format!("{lp}.attn.kv_norm.weight"), head_dim)?,
            wo_a: resident_mat(store, &format!("{lp}.attn.wo_a.weight"))?,
            wo_b: resident_mat(store, &format!("{lp}.attn.wo_b.weight"))?,
            comp,
            idx,
        };
        let gate_bias = download_f32(
            gpu,
            store,
            &format!("{lp}.ffn.gate.e_score_correction_bias"),
        )?;
        ensure!(
            gate_bias.len() == config.num_experts,
            "layer {l}: correction bias has {} entries",
            gate_bias.len()
        );
        let moe_w = MoeV41LayerWeights {
            layer: l as u32,
            gate_w: bf16_ptr(store, &format!("{lp}.ffn.gate.weight"))?,
            gate_bias,
            shared_w1: resident_mat(store, &format!("{lp}.ffn.shared_experts.w1"))?,
            shared_w2: resident_mat(store, &format!("{lp}.ffn.shared_experts.w2"))?,
            shared_w3: resident_mat(store, &format!("{lp}.ffn.shared_experts.w3"))?,
        };
        let engram_index = rt.tables.hash_index(l);
        layers.push(Box::new(DeepSeekV41Layer {
            idx: l,
            role,
            rt: rt.clone(),
            attn_w,
            moe_w,
            engram_index,
            hc_attn: hc_site(gpu, store, &lp, "attn", config)?,
            hc_ffn: hc_site(gpu, store, &lp, "ffn", config)?,
            attn_norm: DenseWeight {
                weight: bf16_ptr(store, &format!("{lp}.attn_norm.weight"))?,
            },
            ffn_norm: DenseWeight {
                weight: bf16_ptr(store, &format!("{lp}.ffn_norm.weight"))?,
            },
            k_hc_expand,
            k_hc_post,
            k_mixes_dot,
            k_mixes_finish,
            k_collapse,
            k_rms_norm,
        }));
    }
    Ok(layers)
}
