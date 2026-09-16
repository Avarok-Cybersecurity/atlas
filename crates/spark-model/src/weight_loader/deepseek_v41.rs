// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash: the loader that assembles [`DeepSeekV41Layer`]s from
//! the seven-shard Q2_K GGUF.
//!
//! Resident (dequantized to bf16 once by the GGUF path, ~2.9 GiB): attention,
//! the compressor and indexer projections, the routers, the shared experts,
//! the engram projections, the norms, the embedding and the Q6_K head. Left
//! on disk and recorded as deferred by the GGUF loader: the 40 x 3 routed
//! expert stacks and the two engram tables, served by `expert_stream`.
//!
//! What the GGUF narrows to bf16 but the kernels want in f32 (norm weights,
//! the attention sinks, the ratio-2 compressor projections, the mHC mixes)
//! is widened here at load; what the kernels want as a product (`engram q * k`)
//! is formed here.
//!
//! The source-layer sets (`kv_source_layer_ids`, `index_source_layer_ids`)
//! are derived from which layers ship compressor / indexer tensors, which the
//! S1 real-file oracle proved equal to the published config. The candidate
//! settings are not in the GGUF metadata; the published `text_config` values
//! are the defaults (candidate source layer 20, 2048 blocks of 8).

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::expert_stream::{
    EngramRowReader, ExpertLru, ExpertSliceMap, ExpertSource, PinnedArena, ShardFiles,
};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::attn_v41::{
    AttnV41, AttnV41Cfg, AttnV41LayerWeights, CompressorWeightsGpu, IndexerWeightsGpu, LayerRole,
    SharedV41,
};
use crate::layers::deepseek_v41_layer::{DeepSeekV41Layer, V41Runtime};
use crate::layers::engram_v41::{EngramHashTables, EngramHasher, EngramLayerWeights, EngramV41};
use crate::layers::moe_v41::{MoeV41, MoeV41Cfg, MoeV41LayerWeights};
use crate::layers::ops::ResidentMat;
use crate::layers::qwen3_attention::HcSiteWeights;
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{DenseWeight, MtpWeights, dense_auto};

pub struct DeepSeekV41WeightLoader;

const DEFAULT_CANDIDATE_SOURCE: usize = 20;
const DEFAULT_CANDIDATE_TOPK_BLOCKS: usize = 2048;
const DEFAULT_CANDIDATE_BLOCK: usize = 8;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn bf16_ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    let t = store.get(name)?;
    ensure!(
        t.dtype == WeightDtype::BF16,
        "{name}: expected bf16, got {:?}",
        t.dtype
    );
    Ok(t.ptr)
}

/// A resident projection as the GGUF path left it: bf16 (expanded on load)
/// or raw Q2_K / Q3_K blocks (`WeightDtype::Q2K` / `Q3K`, the K-quant path).
fn resident_mat(store: &WeightStore, name: &str) -> Result<ResidentMat> {
    let t = store.get(name)?;
    match t.dtype {
        WeightDtype::BF16 => Ok(ResidentMat::Bf16(t.ptr)),
        WeightDtype::Q2K => Ok(ResidentMat::Q2K(t.ptr)),
        WeightDtype::Q3K => Ok(ResidentMat::Q3K(t.ptr)),
        d => anyhow::bail!("{name}: expected bf16, Q2_K or Q3_K, got {d:?}"),
    }
}

fn download_f32(gpu: &dyn GpuBackend, store: &WeightStore, name: &str) -> Result<Vec<f32>> {
    let t = store.get(name)?;
    let n = t.num_elements();
    match t.dtype {
        WeightDtype::BF16 => {
            let mut b = vec![0u8; n * 2];
            gpu.copy_d2h(t.ptr, &mut b)?;
            Ok(b.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect())
        }
        WeightDtype::FP32 => {
            let mut b = vec![0u8; n * 4];
            gpu.copy_d2h(t.ptr, &mut b)?;
            Ok(b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        WeightDtype::Q2K | WeightDtype::Q3K => {
            use spark_runtime::weights::dequant_cpu::{GgmlType, dequant_to_f32};
            let gt = if t.dtype == WeightDtype::Q2K {
                GgmlType::Q2K
            } else {
                GgmlType::Q3K
            };
            let mut b = vec![0u8; t.byte_size()];
            gpu.copy_d2h(t.ptr, &mut b)?;
            let mut out = vec![0f32; n];
            dequant_to_f32(gt, &b, n, &mut out)
                .with_context(|| format!("{name}: CPU dequant of the resident {gt:?} blocks"))?;
            Ok(out)
        }
        other => anyhow::bail!("{name}: cannot widen {other:?} to f32"),
    }
}

fn upload_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len().max(4))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

/// A tensor the kernels read as f32: the store's bf16 copy widened on the device.
fn f32_ptr(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    name: &str,
    expect: usize,
) -> Result<DevicePtr> {
    let v = download_f32(gpu, store, name)?;
    ensure!(
        v.len() == expect,
        "{name}: {} elements, expected {expect}",
        v.len()
    );
    upload_f32(gpu, &v)
}

fn hc_site(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    lp: &str,
    site: &str,
    c: &ModelConfig,
) -> Result<HcSiteWeights> {
    let hc = c.hc_mult;
    let mix_hc = (2 + hc) * hc;
    Ok(HcSiteWeights {
        hc_fn: f32_ptr(
            gpu,
            store,
            &format!("{lp}.hc_{site}_fn"),
            mix_hc * hc * c.hidden_size,
        )?,
        hc_base: f32_ptr(gpu, store, &format!("{lp}.hc_{site}_base"), mix_hc)?,
        hc_scale: f32_ptr(gpu, store, &format!("{lp}.hc_{site}_scale"), 3)?,
        lowrank: None,
    })
}

impl ModelWeightLoader for DeepSeekV41WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[spark_runtime::kv_cache::KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let n_layers = config.num_hidden_layers;
        let (dim, hc) = (config.hidden_size, config.hc_mult);
        ensure!(
            hc >= 1 && hc <= 4,
            "deepseek-v4.1: hc_mult {hc} outside 1..=4"
        );
        let head_dim = config.head_dim;
        let nh = config.num_attention_heads;

        // ── the streamed tensors: shards, expert map, engram rows ──
        let anchor = store
            .deferred("model.layers.0.ffn.experts_stack.gate")
            .context(
                "deepseek-v4.1: the routed expert stacks were not deferred by the GGUF loader",
            )?;
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
        let max_seq =
            env_usize("ATLAS_DS41_MAX_SEQ", 8192).min(config.max_position_embeddings.max(1));
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
            graph_disabled: std::sync::atomic::AtomicBool::new(false),
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

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "model.embed_tokens.weight", gpu)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "model.norm.weight", gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "lm_head.weight", gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}

/// Layer 0 of the real model against the CPU reference, stage by stage: the
/// same five embedding rows through hc_mixes / collapse / norm / attention /
/// hc_post / MoE on the GPU and on the CPU (real weights downloaded from the
/// store, the six routed experts dequantised by the loader's decoders).
/// Ratio-0 layer: no engram, no shared runtime, so every mismatch is local.
#[cfg(all(test, feature = "cuda"))]
mod real_file_tests {
    use super::*;
    use crate::layers::attn_v41::AttnV41LayerState;
    use crate::layers::deepseek_v41_ref::attn::{AttnWeights, freqs_cis};
    use crate::layers::deepseek_v41_ref::compress::{
        CompAttnCfg, IndexerCfg, LayerAttnState, SharedRuntime, attention_any,
    };
    use crate::layers::deepseek_v41_ref::hc::{hc_mixes, hc_post, hc_pre, rms_norm};
    use crate::layers::deepseek_v41_ref::moe::{
        MoeCfg, MoeWeights, gate as ref_gate, moe as ref_moe,
    };
    use crate::layers::ops;
    use spark_runtime::kernel_args::KernelLaunch;
    use spark_runtime::weights::GgufLoader;
    use spark_runtime::weights::WeightLoader;
    use spark_runtime::weights::dequant_cpu::{GgmlType, dequant_to_f32};

    const MODEL_DIR: &str = "/home/rstesiak/models/dsv41-q2k";

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt()
    }

    fn report(what: &str, got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let max = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "  {what:<28} rms got {:.5} want {:.5}  max|want| {:.4}  worst {:.5}  rel {:.2e}",
            rms(got),
            rms(want),
            max,
            worst,
            worst / max.max(1e-12)
        );
        worst / max.max(1e-12)
    }

    fn dl_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
        let mut b = vec![0u8; n * 2];
        gpu.copy_d2h(p, &mut b).unwrap();
        b.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    }

    fn dl_f32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
        let mut b = vec![0u8; n * 4];
        gpu.copy_d2h(p, &mut b).unwrap();
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> DevicePtr {
        let b = crate::layers::moe_v41::bf16_bytes(v);
        let p = gpu.alloc(b.len()).unwrap();
        gpu.copy_h2d(&b, p).unwrap();
        p
    }

    #[test]
    #[ignore = "requires a CUDA GB10 + the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
    fn layer0_matches_the_cpu_reference_on_the_real_weights() {
        let set = atlas_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
            .expect("kernel target");
        let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules)
            .expect("CUDA backend");
        let g: &dyn GpuBackend = &gpu;
        let stream = g.default_stream();
        let dir = std::path::Path::new(MODEL_DIR);
        let config = spark_runtime::weights::config_from_gguf_dir(dir).expect("config from gguf");
        let store = GgufLoader::new().load(dir, g, 0).expect("load gguf");
        let layers = DeepSeekV41WeightLoader
            .load_layers(&store, &config, g, &[])
            .expect("layers");
        assert_eq!(layers.len(), config.num_hidden_layers);
        let (dim, hc, hd, nh) = (
            config.hidden_size,
            config.hc_mult,
            config.head_dim,
            config.num_attention_heads,
        );
        let eps = config.rms_norm_eps as f32;
        let lp = "model.layers.0";
        let ids: Vec<u32> = vec![0, 576, 6440, 315, 9822];
        let m = ids.len();

        // the same input on both sides: the embedding rows
        let embed = store.get("model.embed_tokens.weight").unwrap();
        let mut x = Vec::with_capacity(m * dim);
        for &id in &ids {
            x.extend(dl_bf16(
                g,
                DevicePtr(embed.ptr.0 + (id as usize * dim * 2) as u64),
                dim,
            ));
        }
        // the highway: hc copies of x, pre-mix one-hot on stream 0
        let h: Vec<f32> = (0..m)
            .flat_map(|t| std::iter::repeat_n(x[t * dim..(t + 1) * dim].to_vec(), hc).flatten())
            .collect();
        let onehot: Vec<f32> = (0..m)
            .flat_map(|_| (0..hc).map(|c| if c == 0 { 1.0 } else { 0.0 }))
            .collect();

        // ── hc mixes on the attention site ──
        let hc_fn = download_f32(g, &store, &format!("{lp}.hc_attn_fn")).unwrap();
        let hc_scale = download_f32(g, &store, &format!("{lp}.hc_attn_scale")).unwrap();
        let hc_base = download_f32(g, &store, &format!("{lp}.hc_attn_base")).unwrap();
        let (pre_r, post_r, comb_r) = hc_mixes(
            &h,
            m,
            hc,
            dim,
            &hc_fn,
            &hc_scale,
            &hc_base,
            config.hc_sinkhorn_iters,
            config.hc_eps,
            eps,
        );
        let streams = g.alloc(m * hc * dim * 4).unwrap();
        let hb: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d(&hb, streams).unwrap();
        let site = hc_site(g, &store, lp, "attn", &config).unwrap();
        let (pre_d, post_d, comb_d) = (
            g.alloc(m * hc * 4).unwrap(),
            g.alloc(m * hc * 4).unwrap(),
            g.alloc(m * hc * hc * 4).unwrap(),
        );
        let mix_hc = (2 + hc) * hc;
        let mixes_d = g.alloc(m * mix_hc * 4).unwrap();
        KernelLaunch::new(g, g.kernel("hc_v41", "hc_v41_mixes_dot").unwrap())
            .grid([m as u32, mix_hc as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(site.hc_fn)
            .arg_ptr(mixes_d)
            .arg_u32(dim as u32)
            .arg_u32(hc as u32)
            .arg_f32(eps)
            .launch(stream)
            .unwrap();
        KernelLaunch::new(g, g.kernel("hc_v41", "hc_v41_mixes_finish").unwrap())
            .grid([m as u32, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(mixes_d)
            .arg_ptr(site.hc_scale)
            .arg_ptr(site.hc_base)
            .arg_ptr(pre_d)
            .arg_ptr(post_d)
            .arg_ptr(comb_d)
            .arg_u32(hc as u32)
            .arg_u32(config.hc_sinkhorn_iters as u32)
            .arg_f32(config.hc_eps)
            .launch(stream)
            .unwrap();
        g.synchronize(stream).unwrap();
        report("hc_mixes pre", &dl_f32(g, pre_d, m * hc), &pre_r);
        report("hc_mixes post", &dl_f32(g, post_d, m * hc), &post_r);
        report("hc_mixes comb", &dl_f32(g, comb_d, m * hc * hc), &comb_r);

        // ── collapse with the one-hot, then the attention norm ──
        let y_r = hc_pre(&h, &onehot, m, hc, dim);
        let attn_norm = download_f32(g, &store, &format!("{lp}.attn_norm.weight")).unwrap();
        let x_r = rms_norm(&y_r, &attn_norm, m, dim, eps);
        let onehot_d = g.alloc(m * hc * 4).unwrap();
        g.copy_h2d(
            &onehot
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            onehot_d,
        )
        .unwrap();
        let y_d = g.alloc(m * dim * 2).unwrap();
        KernelLaunch::new(g, g.kernel("hc_v41", "hc_v41_collapse").unwrap())
            .grid([m as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(onehot_d)
            .arg_ptr(y_d)
            .arg_u32(dim as u32)
            .arg_u32(hc as u32)
            .launch(stream)
            .unwrap();
        let x_d = g.alloc(m * dim * 2).unwrap();
        let norm_w = DenseWeight {
            weight: bf16_ptr(&store, &format!("{lp}.attn_norm.weight")).unwrap(),
        };
        ops::rms_norm(
            g,
            g.kernel("rms_norm_vanilla", "rms_norm_vanilla").unwrap(),
            y_d,
            &norm_w,
            x_d,
            m as u32,
            dim as u32,
            eps,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        report("collapse y", &dl_bf16(g, y_d, m * dim), &y_r);
        report("attn_norm x", &dl_bf16(g, x_d, m * dim), &x_r);

        // ── attention (ratio 0) ──
        let w = |n: &str| download_f32(g, &store, &format!("{lp}.attn.{n}")).unwrap();
        let (sink, wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b) = (
            w("attn_sink"),
            w("wq_a.weight"),
            w("q_norm.weight"),
            w("wq_b.weight"),
            w("wkv.weight"),
            w("kv_norm.weight"),
            w("wo_a.weight"),
            w("wo_b.weight"),
        );
        let aw = AttnWeights {
            sink: &sink,
            wq_a: &wq_a,
            q_norm: &q_norm,
            wq_b: &wq_b,
            wkv: &wkv,
            kv_norm: &kv_norm,
            wo_a: &wo_a,
            wo_b: &wo_b,
        };
        let max_seq = 256usize;
        let cc = CompAttnCfg {
            dim,
            n_heads: nh,
            head_dim: hd,
            rope_dim: config.rotary_dim,
            q_rank: config.q_lora_rank,
            o_rank: config.o_lora_rank,
            groups: config.o_groups,
            window: config.sliding_window as usize,
            eps,
            ratio: 0,
            is_kv_source: false,
            is_index_source: false,
            is_candidate_source: false,
            uses_candidates: false,
        };
        let icfg = IndexerCfg {
            n_heads: config.index_n_heads,
            index_hd: config.index_head_dim,
            rope_dim: config.rotary_dim,
            q_rank: config.q_lora_rank,
            dim,
            hd,
            index_topk: config.index_topk,
            cand_topk_blocks: 1,
            cand_block: 1,
            eps,
        };
        let fc = freqs_cis(config.rotary_dim, max_seq, config.rope_theta as f32);
        let mut st = LayerAttnState::new(&cc, max_seq, config.index_head_dim);
        let mut sh = SharedRuntime::default();
        let run_r = attention_any(
            &x_r, m, 0, &aw, None, None, &icfg, &cc, &fc, &mut st, &mut sh,
        );
        // production
        let mut acfg = AttnV41Cfg {
            dim,
            n_heads: nh,
            head_dim: hd,
            rope_dim: config.rotary_dim,
            q_rank: config.q_lora_rank,
            o_rank: config.o_lora_rank,
            groups: config.o_groups,
            window: config.sliding_window as usize,
            eps,
            index_heads: config.index_n_heads,
            index_hd: config.index_head_dim,
            index_topk: config.index_topk,
            cand_topk_blocks: 2048,
            cand_block: 8,
            max_seq,
            max_tokens: 16,
            rope_theta: config.rope_theta as f32,
            compress_rope_theta: config.compress_rope_theta,
            rope_factor: 16.0,
            orig_seq: 65536,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        acfg.max_tokens = 16;
        let attn = AttnV41::new(g, acfg.clone()).unwrap();
        let role = LayerRole {
            ratio: 0,
            ..Default::default()
        };
        let attn_w = AttnV41LayerWeights {
            role,
            sink: f32_ptr(g, &store, &format!("{lp}.attn.attn_sink"), nh).unwrap(),
            wq_a: resident_mat(&store, &format!("{lp}.attn.wq_a.weight")).unwrap(),
            q_norm: f32_ptr(
                g,
                &store,
                &format!("{lp}.attn.q_norm.weight"),
                config.q_lora_rank,
            )
            .unwrap(),
            wq_b: resident_mat(&store, &format!("{lp}.attn.wq_b.weight")).unwrap(),
            wkv: resident_mat(&store, &format!("{lp}.attn.wkv.weight")).unwrap(),
            kv_norm: f32_ptr(g, &store, &format!("{lp}.attn.kv_norm.weight"), hd).unwrap(),
            wo_a: resident_mat(&store, &format!("{lp}.attn.wo_a.weight")).unwrap(),
            wo_b: resident_mat(&store, &format!("{lp}.attn.wo_b.weight")).unwrap(),
            comp: None,
            idx: None,
        };
        let mut ast = AttnV41LayerState::new(g, &acfg, role).unwrap();
        let mut shared = SharedV41::default();
        let x_in = up_bf16(g, &x_r);
        let run = attn
            .forward(g, &attn_w, &mut ast, &mut shared, x_in, m, 0, stream)
            .unwrap();
        assert_eq!(run.idx, run_r.idx, "window idx");
        report("attn q", &dl_bf16(g, run.q, m * nh * hd), &run_r.q);
        report(
            "attn kv rows",
            &dl_bf16(g, run.rows_a, m * hd),
            &run_r.kv_rows,
        );
        report("attn o", &dl_bf16(g, run.o, m * nh * hd), &run_r.o);
        let attn_rel = report("attn out", &dl_bf16(g, run.out, m * dim), &run_r.out);

        // ── hc_post on the attention output ──
        let h_mid_r = hc_post(&run_r.out, &h, &post_r, &comb_r, m, hc, dim);
        let out_d = up_bf16(g, &run_r.out);
        ops::hc_post(
            g,
            g.kernel("hyper_connection", "hc_post").unwrap(),
            out_d,
            streams,
            post_d,
            comb_d,
            streams,
            m as u32,
            dim as u32,
            hc as u32,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        report(
            "hc_post streams",
            &dl_f32(g, streams, m * hc * dim),
            &h_mid_r,
        );

        // ── MoE on the ffn input (reference collapse with attn pre) ──
        let ffn_norm = download_f32(g, &store, &format!("{lp}.ffn_norm.weight")).unwrap();
        let f_in_r = rms_norm(
            &hc_pre(&h_mid_r, &pre_r, m, hc, dim),
            &ffn_norm,
            m,
            dim,
            eps,
        );
        let gate_w = download_f32(g, &store, &format!("{lp}.ffn.gate.weight")).unwrap();
        let gate_bias =
            download_f32(g, &store, &format!("{lp}.ffn.gate.e_score_correction_bias")).unwrap();
        let mc = MoeCfg {
            dim,
            inter: config.moe_intermediate_size,
            n_routed: config.num_experts,
            topk: config.num_experts_per_tok,
            gate_temp: 1.0,
            norm_topk_prob: config.norm_topk_prob,
            route_scale: config.routed_scaling_factor as f32,
            swiglu_limit: config.swiglu_limit,
        };
        let (rw_r, ri_r) = ref_gate(
            &f_in_r,
            &gate_w,
            &gate_bias,
            m,
            dim,
            mc.n_routed,
            mc.topk,
            1.0,
            mc.norm_topk_prob,
            mc.route_scale,
        );
        // dequantise the routed experts on the CPU
        let files = Arc::new(ShardFiles::open_dir(dir).unwrap());
        let slices = ExpertSliceMap::new(files).unwrap();
        let lay = slices.slot_layout();
        let q2 = GgmlType::from_id(10, 128).unwrap();
        let q3 = GgmlType::from_id(11, 128).unwrap();
        let inter = config.moe_intermediate_size;
        let mut experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> =
            vec![(Vec::new(), Vec::new(), Vec::new()); mc.n_routed];
        for &e in &ri_r {
            if !experts[e].0.is_empty() {
                continue;
            }
            let mut raw = vec![0u8; lay.bytes];
            slices.read_expert(0, e as u32, &mut raw).unwrap();
            let mut w1 = vec![0f32; inter * dim];
            let mut w3 = vec![0f32; inter * dim];
            let mut w2 = vec![0f32; dim * inter];
            dequant_to_f32(
                q2,
                &raw[lay.gate_off..lay.gate_off + lay.gate_bytes],
                inter * dim,
                &mut w1,
            )
            .unwrap();
            dequant_to_f32(
                q2,
                &raw[lay.up_off..lay.up_off + lay.up_bytes],
                inter * dim,
                &mut w3,
            )
            .unwrap();
            dequant_to_f32(
                q3,
                &raw[lay.down_off..lay.down_off + lay.down_bytes],
                dim * inter,
                &mut w2,
            )
            .unwrap();
            experts[e] = (w1, w2, w3);
        }
        let s1 = download_f32(g, &store, &format!("{lp}.ffn.shared_experts.w1")).unwrap();
        let s2 = download_f32(g, &store, &format!("{lp}.ffn.shared_experts.w2")).unwrap();
        let s3 = download_f32(g, &store, &format!("{lp}.ffn.shared_experts.w3")).unwrap();
        let mw = MoeWeights {
            gate_w: &gate_w,
            gate_bias: &gate_bias,
            experts: experts
                .iter()
                .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice()))
                .collect(),
            shared: (&s1, &s2, &s3),
        };
        let (moe_r, _, _) = ref_moe(&f_in_r, m, &mw, &mc);
        // production
        let mcfg = MoeV41Cfg {
            dim,
            inter,
            n_routed: mc.n_routed,
            topk: mc.topk,
            gate_temp: 1.0,
            norm_topk_prob: mc.norm_topk_prob,
            route_scale: mc.route_scale,
            swiglu_limit: mc.swiglu_limit,
            max_tokens: 16,
        };
        let moe = MoeV41::new(g, mcfg).unwrap();
        let mw_d = MoeV41LayerWeights {
            layer: 0,
            gate_w: bf16_ptr(&store, &format!("{lp}.ffn.gate.weight")).unwrap(),
            gate_bias: gate_bias.clone(),
            shared_w1: resident_mat(&store, &format!("{lp}.ffn.shared_experts.w1")).unwrap(),
            shared_w2: resident_mat(&store, &format!("{lp}.ffn.shared_experts.w2")).unwrap(),
            shared_w3: resident_mat(&store, &format!("{lp}.ffn.shared_experts.w3")).unwrap(),
        };
        let arena = PinnedArena::alloc(g, 64 * lay.bytes).unwrap();
        let mut lru = ExpertLru::new(arena.host(), arena.dev(), arena.bytes(), lay).unwrap();
        let f_in_d = up_bf16(g, &f_in_r);
        let (moe_out, rw_d, ri_d) = moe
            .forward(g, &mw_d, &mut lru, &slices, f_in_d, m, 4, stream)
            .unwrap();
        assert_eq!(ri_d, ri_r, "routing indices");
        report("moe routing weights", &rw_d, &rw_r);
        let moe_rel = report("moe out", &dl_bf16(g, moe_out, m * dim), &moe_r);
        println!("layer 0 real-weight oracle: attn rel {attn_rel:.2e}, moe rel {moe_rel:.2e}");
        assert!(
            attn_rel < 3e-2 && moe_rel < 6e-2,
            "layer 0 diverges from the reference"
        );
    }
}
