// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::{ModelWeightLoader, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::{DenseFfnLayer, FfnComponent, Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, MtpWeights, Nvfp4Variant, SsmWeights, dense,
    dense_auto, dense_f32_safe, dense_keep_f32, dequant_nvfp4_to_bf16, detect_nvfp4_variant,
    gpu_concat_rows, interleave_ba, load_dense_ffn, load_fp8_block_scaled_as_fp8weight,
    load_kv_scales, load_mtp, quantize_to_nvfp4, quantized_auto,
};

/// True when `{prefix}.weight` is FP8 E4M3 on disk with a 2D block scale
/// (`weight_scale_inv` or 2D `weight_scale`) — i.e. a native FP8 checkpoint
/// projection that should load as `Fp8Weight` rather than be requantized to
/// NVFP4. Mirrors `qwen35::load_layers::proj_is_native_fp8`.
fn proj_is_native_fp8(store: &WeightStore, prefix: &str) -> bool {
    let is_fp8_weight = store
        .get(&format!("{prefix}.weight"))
        .map(|w| w.dtype == WeightDtype::FP8E4M3)
        .unwrap_or(false);
    let has_block_scale = store.contains(&format!("{prefix}.weight_scale_inv"))
        || store
            .get(&format!("{prefix}.weight_scale"))
            .map(|s| s.shape.len() == 2)
            .unwrap_or(false);
    is_fp8_weight && has_block_scale
}

/// True when `{prefix}.weight` is on-disk FP8 with ANY dequant scale — block
/// (`weight_scale_inv` / 2D `weight_scale`) OR the ModelOpt per-tensor SCALAR
/// `weight_scale` (shape `[]`). `load_fp8_block_scaled_as_fp8weight` broadcasts
/// the scalar into a uniform block grid, so all three are loadable as FP8. The
/// nvidia Qwen3.6-27B-NVFP4 GDN projections use the scalar form, which
/// `proj_is_native_fp8` (2D-only) misses.
fn proj_is_fp8_any_scale(store: &WeightStore, prefix: &str) -> bool {
    let is_fp8 = store
        .get(&format!("{prefix}.weight"))
        .map(|w| w.dtype == WeightDtype::FP8E4M3)
        .unwrap_or(false);
    is_fp8
        && (store.contains(&format!("{prefix}.weight_scale_inv"))
            || store.contains(&format!("{prefix}.weight_scale")))
}

/// Concatenate two block-scaled FP8 weights along rows (dim 0):
/// `[n_a, k] ++ [n_b, k] -> [n_a+n_b, k]`. FP8 bytes (1 B/elem) and the
/// `[n/128, k/128]` FP32 block-scale grids are contiguous row-major, so each is
/// a straight device-to-device append. Requires `n_a % 128 == 0` so the
/// scale-grid rows meet at a block boundary (GDN qkv rows are a multiple of
/// 128). Produces the `[Q|K|V|Z]` sequential order the SSM layer expects.
fn concat_fp8_block_scaled(
    a: &Fp8Weight,
    b: &Fp8Weight,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    let kb = k.div_ceil(128);
    let a_w = a.n as usize * k;
    let b_w = b.n as usize * k;
    let weight = gpu.alloc(a_w + b_w)?;
    gpu.copy_d2d(a.weight, weight, a_w)?;
    gpu.copy_d2d(b.weight, weight.offset(a_w), b_w)?;
    let a_s = (a.n as usize).div_ceil(128) * kb * 4;
    let b_s = (b.n as usize).div_ceil(128) * kb * 4;
    let row_scale = gpu.alloc(a_s + b_s)?;
    gpu.copy_d2d(a.row_scale, row_scale, a_s)?;
    gpu.copy_d2d(b.row_scale, row_scale.offset(a_s), b_s)?;
    Ok(Fp8Weight {
        weight,
        row_scale,
        n: a.n + b.n,
        k: k as u32,
        scale_format: crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
    })
}

/// Opt-in gate for native dense-FP8 attention + FFN dispatch (Qwythos / dense
/// Ornith-FP8). Default OFF.
///
/// VERIFIED 2026-06-29 on Qwythos-9B-FP8 (gb10/ornith-1.0-9b): with the flag
/// on, the FP8 arms fire for all 32 FFN + 8 full-attn layers and text is
/// correct (coherence/fib/tools 3/3). BUT it is NOT a perf win — ~30 tok/s vs
/// ~40 for the NVFP4 fallback — because this target's NVFP4 W4A16 kernels
/// (fused dual-GEMV decode, transposed m128 prefill) are more optimized than
/// its FP8 W8A16 kernels (unfused per-projection GEMV, non-transposed
/// `w8a16_gemm` prefill; the attention FP8 prefill transpose also does not
/// engage). Vision prefill additionally hits a CUDA-700. Making FP8 pay off
/// here needs dedicated dense-FP8 kernels (fused FP8 dual-GEMV + fast
/// transposed FP8 prefill GEMM), not loader wiring. Until then NVFP4 autoquant
/// is the better dense runtime. `ATLAS_DENSE_FP8=1` opts in for that kernel work.
fn dense_fp8_enabled() -> bool {
    std::env::var("ATLAS_DENSE_FP8").as_deref() == Ok("1")
}

mod loaders_b;

pub struct Qwen35DenseWeightLoader;

impl ModelWeightLoader for Qwen35DenseWeightLoader {
    fn supports_tp(&self) -> bool {
        // FullAttention layers are TP-sharded (NVFP4-from-disk and BF16
        // → NVFP4 paths). LinearAttention (GDN SSM) layers run
        // full-replica per rank — see qwen35.rs for the rationale.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect::<Vec<_>>()
        } else {
            config.layer_types.clone()
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        let variant = detect_nvfp4_variant(store, config);
        let weight_format = WeightFormat::detect(store, config);
        tracing::info!(
            "Weight format: {:?}, NVFP4 variant: {:?}",
            weight_format,
            variant
        );

        // Native FP8 SSM prefill GEMM (Qwen3.6-27B-FP8 root-cause fix,
        // commit 3ebc08a). Atlas's prior SSM in_proj_qkv path was
        // FP8 → BF16 → NVFP4 → BF16 (in `w4a16_gemm` dequant) → MMA — a
        // double-quant chain whose NVFP4 hop's ~4-bit per-group precision
        // is dominated by signal at q/v but attenuated into a k-direction
        // error (HF conv-k ‖6.3‖ vs conv-v ‖117.2‖, ~18× smaller). For
        // every FP8-on-disk checkpoint we install a single-scale FP8 copy
        // of the stacked `[QKV|Z]` and `out_proj` weights for prefill,
        // bypassing the NVFP4 intermediate. Prefill dispatches via the
        // existing `fp8_gemm_n128` (BF16 act × FP8 weight) — same path
        // the MoE shared-expert FP8 prefill uses. Decode/GEMV unchanged.
        // Originally env-gated `ATLAS_FP8_SSM_PREFILL=1`; promoted to
        // unconditional 2026-05-20 after live verification (commit
        // dfb4e8a era, tokens_to_first_degeneration 1,196 → 16,968).
        // 2026-07-03 precision policy: GDN projections stay ≥FP8 — the
        // NVFP4 requant of qkvz/out_proj flattens tensors that both
        // checkpoint toolchains deliberately keep high-precision (modelopt
        // sensitivity analysis ships them FP8; unsloth ships BF16). Applies
        // to ALL NVFP4 variants now, not just Fp8Dequanted. Kill-switch
        // ATLAS_NO_GDN_FP8_PREFILL restores the pre-policy behavior for A/B.
        let fp8_ssm_prefill = std::env::var_os("ATLAS_NO_GDN_FP8_PREFILL").is_none();
        let bf16_to_fp8_k = if fp8_ssm_prefill {
            tracing::info!(
                "SSM in_proj_qkv + out_proj via native FP8 prefill GEMM \
                 (BF16 act × FP8 weight via fp8_gemm_n128); NVFP4 kept as \
                 structural fallback for decode batch paths"
            );
            Some(gpu.kernel("w4a16", "bf16_to_fp8")?)
        } else {
            None
        };

        // ATLAS_MEM_PROFILE: per-phase GPU-free trace to pin the strix/APU
        // load-time footprint (FP8-source persistence vs NVFP4 steady-state vs
        // BF16 requant transients). Gated env so it's a no-op in production.
        let mem_profile = std::env::var("ATLAS_MEM_PROFILE").is_ok();
        let log_free = |tag: &str| {
            if mem_profile && let Ok(free) = gpu.free_memory() {
                tracing::info!("MEM_PROFILE[{tag}]: {:.2} GB GPU-free", free as f64 / 1e9);
            }
        };
        log_free("dense-load-start");

        for (i, lt) in layer_types.iter().enumerate() {
            if i % 8 == 0 {
                log_free(&format!("layer-{i}"));
            }
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // Dense FFN instead of MoE. Native FP8 checkpoints (single-GPU)
            // load gate/up/down directly as block-scaled `Fp8Weight` and
            // dispatch w8a16 — no NVFP4 requant. TP>1 still uses the NVFP4
            // path (FP8 FFN sharding is a follow-up).
            let ffn_fp8 = dense_fp8_enabled()
                && config.tp_world_size.max(1) == 1
                && matches!(variant, Nvfp4Variant::Fp8Dequanted)
                && proj_is_native_fp8(store, &format!("{lp}.mlp.gate_proj"));
            // Always load the NVFP4 weights so every dispatch path (incl. the
            // batched spec-decode forward_k2/k3 paths that have no FP8 branch)
            // has a valid weight to fall back to — a null NVFP4 weight under a
            // w4a16 dispatch is the CUDA-700-at-concurrency bug. When native
            // FP8 is enabled we overlay the block-scaled FP8 weights on top;
            // the hot forward / forward_prefill paths then use FP8, the rare
            // batched paths fall back to real NVFP4.
            let ffn_weights = load_dense_ffn(
                store, &lp, gpu, variant, absmax_k, quantize_k, stream, config,
            )?;
            let mut dffn = DenseFfnLayer::new(ffn_weights, gpu)?;
            if ffn_fp8 {
                let load_ffn_fp8 = |name: &str| {
                    load_fp8_block_scaled_as_fp8weight(store, &format!("{lp}.mlp.{name}"), gpu)
                };
                dffn.set_fp8_weights(
                    load_ffn_fp8("gate_proj")?,
                    load_ffn_fp8("up_proj")?,
                    load_ffn_fp8("down_proj")?,
                );
            }
            // ATLAS_FFN_MMQ: eagerly materialize Q4_K + free the dead `_t` copies at load,
            // BEFORE KV cache sizing, so net FFN footprint == NVFP4 baseline (no decode OOM-throttle).
            dffn.finalize_q4k_load(gpu, h as u32, config.intermediate_size as u32, stream)?;
            // ATLAS_FFN_NVFP4_MMQ: same discipline for the W4A4 FP4-MMQ arm — repack
            // gate/up to block_nvfp4 + free their `_t` copies (net ~0 footprint).
            dffn.finalize_nvfp4_mmq_load(gpu, h as u32, config.intermediate_size as u32, stream)?;
            let ffn = FfnComponent::Dense(dffn);

            match lt {
                LayerType::FullAttention => {
                    let p = format!("{lp}.self_attn");
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let (attn, q_nvfp4, k_nvfp4, v_nvfp4) = match variant {
                        Nvfp4Variant::CompressedTensors => {
                            // NVFP4-from-disk path: column-parallel Q/K/V, row-parallel O.
                            let group_size = 16usize;
                            let load_nvfp4 = |name: &str,
                                              full_n: usize,
                                              full_k: usize,
                                              kind: TpShardKind|
                             -> Result<crate::weight_map::QuantizedWeight> {
                                let prefix = format!("{p}.{name}");
                                // Mixed-precision compressed-tensors checkpoints
                                // (unsloth Qwen3.6-*-NVFP4, re-quantized 2026-07-10)
                                // NVFP4-pack most of the net but keep attention
                                // q/k/v/o as FP8 (`.weight` FP8E4M3 + a per-row
                                // `.weight_scale`, no `.weight_packed`). Without the
                                // pack metadata, dequant and runtime-quantize to NVFP4
                                // instead of failing on the absent
                                // `weight_global_scale`. Mirrors the MoE loader's
                                // attention arm (weight_loader/qwen35/load_layers/
                                // attention_arms.rs).
                                let src = if store.contains(&format!("{prefix}.weight_packed")) {
                                    quantized_auto(store, &prefix, gpu, variant)?
                                } else {
                                    let dense_bf16 =
                                        dense_auto(store, &format!("{prefix}.weight"), gpu)?;
                                    quantize_to_nvfp4(
                                        &dense_bf16,
                                        full_n,
                                        full_k,
                                        gpu,
                                        absmax_k,
                                        quantize_k,
                                        stream,
                                    )?
                                };
                                if tp_size == 1 {
                                    return Ok(src);
                                }
                                let sharded = shard_quantized_nvfp4(
                                    &src, full_n, full_k, kind, tp_rank, tp_size, group_size, gpu,
                                )?;
                                gpu.free(src.weight)?;
                                gpu.free(src.weight_scale)?;
                                Ok(sharded)
                            };
                            let [q, k, v, o] = load_qkvo_tp(config, load_nvfp4)?;
                            let dummy = DenseWeight {
                                weight: spark_runtime::gpu::DevicePtr::NULL,
                            };
                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
                            let attn = AttentionWeights {
                                q_proj: dummy,
                                k_proj: dummy,
                                v_proj: dummy,
                                o_proj: o,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q), Some(k), Some(v))
                        }
                        Nvfp4Variant::Standard
                        | Nvfp4Variant::Fp8Dequanted
                        | Nvfp4Variant::Bf16Raw => {
                            // BF16 → NVFP4 path: shard BF16 then quantize per-rank.
                            let load_bf16_then_nvfp4 = |name: &str,
                                                        full_n: usize,
                                                        full_k: usize,
                                                        kind: TpShardKind|
                             -> Result<(
                                DenseWeight,
                                crate::weight_map::QuantizedWeight,
                            )> {
                                // Pre-quantized Standard NVFP4 (e.g. sakamakismile): weight is U8
                                // on disk. Load directly as QuantizedWeight without BF16 roundtrip.
                                // TP sharding of pre-quantized NVFP4 is not yet supported: enforce
                                // tp_size=1 explicitly, or every rank silently loads the full
                                // unsharded weight (duplicated, not sharded — wrong results with
                                // no error).
                                let weight_key = format!("{p}.{name}.weight");
                                if matches!(
                                    store.get(&weight_key).map(|w| w.dtype),
                                    Ok(WeightDtype::UInt8)
                                ) {
                                    anyhow::ensure!(
                                        tp_size == 1,
                                        "pre-quantized NVFP4 weight '{weight_key}' (U8 on disk) \
                                         cannot be loaded under tensor parallelism (tp_size={tp_size}): \
                                         TP sharding of pre-quantized NVFP4 checkpoints is not yet \
                                         implemented. Use tp_size=1, or dequantize this checkpoint to \
                                         BF16 first so it goes through the shard-then-requantize path."
                                    );
                                    let null_dense = DenseWeight {
                                        weight: spark_runtime::gpu::DevicePtr::NULL,
                                    };
                                    let qw = quantized_auto(
                                        store,
                                        &format!("{p}.{name}"),
                                        gpu,
                                        Nvfp4Variant::Standard,
                                    )?;
                                    return Ok((null_dense, qw));
                                }
                                let src = dense_auto(store, &weight_key, gpu)?;
                                let (sharded_ptr, local_n, local_k) = shard_dense_bf16(
                                    src.weight, full_n, full_k, kind, tp_rank, tp_size, gpu,
                                )?;
                                let sharded = DenseWeight {
                                    weight: sharded_ptr,
                                };
                                let q = quantize_to_nvfp4(
                                    &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                                )?;
                                if sharded_ptr != src.weight {
                                    gpu.free(sharded_ptr)?;
                                }
                                Ok((src, q))
                            };
                            let [
                                (q_dense, q_nvfp4),
                                (k_dense, k_nvfp4),
                                (v_dense, v_nvfp4),
                                (o_dense, o_nvfp4),
                            ] = load_qkvo_tp(config, load_bf16_then_nvfp4)?;

                            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

                            // The BF16 q/k/v/o dense tensors are only the intermediate
                            // fed to the GPU quantize_to_nvfp4 above. Prefill AND decode
                            // always dispatch the NVFP4 weights, so the BF16 copies are
                            // dead once quantized. Free them instead of retaining a full
                            // second copy of every projection (Atlas issue #A1).
                            gpu.free(q_dense.weight)?;
                            gpu.free(k_dense.weight)?;
                            gpu.free(v_dense.weight)?;
                            gpu.free(o_dense.weight)?;

                            let attn = AttentionWeights {
                                q_proj: DenseWeight {
                                    weight: spark_runtime::gpu::DevicePtr::NULL,
                                },
                                k_proj: DenseWeight {
                                    weight: spark_runtime::gpu::DevicePtr::NULL,
                                },
                                v_proj: DenseWeight {
                                    weight: spark_runtime::gpu::DevicePtr::NULL,
                                },
                                o_proj: o_nvfp4,
                                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                                q_norm_full: None,
                                k_norm_full: None,
                                k_scale,
                                v_scale,
                            };
                            (attn, Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4))
                        }
                    };

                    let mut attn_layer = Qwen3AttentionLayer::new(
                        input_norm,
                        attn,
                        post_attn_norm,
                        ffn,
                        attn_idx,
                        q_nvfp4,
                        k_nvfp4,
                        v_nvfp4,
                        gpu,
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;
                    // Fast-prefill: transposed NVFP4 copies route the 16 full-attn
                    // layers' q/k/v/o prefill GEMMs onto w4a16_gemm_t_m128 (28.8%
                    // of prefill GPU time on the base w4a16_gemm path; ~1.3x e2e).
                    // predequant_for_prefill() is deliberately NOT called: the FP8
                    // predequant route is slower for these bandwidth-bound GEMMs.
                    if let (Some(qw), Some(kw), Some(vw)) = (q_nvfp4, k_nvfp4, v_nvfp4) {
                        let (nh, hd) = (config.num_attention_heads, config.head_dim);
                        let (nkv, hh) = (config.num_key_value_heads, config.hidden_size);
                        let q_n = nh * hd * if config.attn_gated { 2 } else { 1 };
                        let qt = qw.transpose_for_gemm(gpu, q_n, hh)?;
                        let kt = kw.transpose_for_gemm(gpu, nkv * hd, hh)?;
                        let vt = vw.transpose_for_gemm(gpu, nkv * hd, hh)?;
                        let op = &attn_layer.attn.o_proj;
                        let ot = op.transpose_for_gemm(gpu, hh, nh * hd)?;
                        attn_layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
                    }
                    // Overlay native FP8 q/k/v/o on top of the NVFP4 weights when
                    // enabled (single-GPU FP8 checkpoint). Hot decode/prefill paths
                    // dispatch FP8 (w8a16); any path without an FP8 branch falls back
                    // to the real NVFP4 weights above (never a null → no CUDA-700).
                    if dense_fp8_enabled()
                        && config.tp_world_size.max(1) == 1
                        && matches!(variant, Nvfp4Variant::Fp8Dequanted)
                        && proj_is_native_fp8(store, &format!("{p}.q_proj"))
                    {
                        let load_fp8_proj = |name: &str,
                                             _n: usize,
                                             _k: usize,
                                             _kind: TpShardKind|
                         -> Result<Fp8Weight> {
                            load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.{name}"), gpu)
                        };
                        let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8_proj)?;
                        attn_layer.set_fp8_weights(
                            Some(q_fp8),
                            Some(k_fp8),
                            Some(v_fp8),
                            Some(o_fp8),
                        );
                        if let Err(e) = attn_layer.transpose_fp8_for_prefill(gpu, stream) {
                            tracing::warn!("Layer {i}: dense FP8 transpose failed: {e}");
                        }
                    }
                    layers.push(Box::new(attn_layer));
                    attn_idx += 1;
                }
                LayerType::LinearAttention => {
                    let nv = config.linear_num_value_heads;
                    let nk = config.linear_num_key_heads;
                    let qkv_rows = config.ssm_qkv_size();
                    let z_rows = config.ssm_z_size();
                    let value_dim = nv * config.linear_value_head_dim;
                    let la = format!("{lp}.linear_attn");

                    // SSM projections are loaded per-projection by on-disk dtype:
                    // each of in_proj_qkv / in_proj_z / out_proj may independently
                    // be NVFP4-packed (`weight_packed`) or plain (`weight`, routed
                    // by `dense_auto` → BF16/FP32/FP8). The unsloth NVFP4 re-quant
                    // of Qwen3.6-27B quantizes ONLY out_proj while keeping the
                    // in_proj_* in BF16; the old all-or-nothing gate (keyed on
                    // in_proj_qkv.weight_packed) then looked for a non-existent
                    // out_proj.weight and failed to build. `dense_auto` is dequant-
                    // to-BF16 for the concat pipeline regardless of source dtype.
                    let load_ssm_proj =
                        |name: &str, rows: usize, cols: usize| -> Result<DenseWeight> {
                            if store.contains(&format!("{name}.weight_packed")) {
                                dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
                            } else if matches!(
                                store.get(&format!("{name}.weight")).map(|w| w.dtype),
                                Ok(WeightDtype::UInt8)
                            ) {
                                // Standard-convention NVFP4 (packed bytes at
                                // `.weight`, not `.weight_packed`) — same dequant,
                                // different on-disk key.
                                dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
                            } else {
                                dense_auto(store, &format!("{name}.weight"), gpu)
                            }
                        };
                    // Native FP8 GDN (nvidia mixed-precision checkpoint): the
                    // in_proj_qkv / in_proj_z / out_proj projections ship as
                    // F8_E4M3 + per-tensor scale — modelopt's sensitivity
                    // analysis keeps the SSM projections high-precision. The
                    // default path (`load_ssm_proj` → `dense_auto`) dequants to
                    // BF16 then RE-quantizes to NVFP4 (4-bit), a lossy
                    // double-quant of these 48/64 layers that regressed BFCL-ST
                    // ~7pt (non_live 85.4→76.6). Load the on-disk FP8 directly
                    // (concat qkv+z on-device into [Q|K|V|Z] order) and route
                    // BOTH prefill (w8a16_gemm_pipelined) and decode
                    // (w8a16_gemv) through the fp8w fields — no requant, decode
                    // stays fast (FP8 = half BF16's weight bytes). MUST run
                    // BEFORE `load_ssm_proj` consumes the store tensors.
                    // Internal opt-out for the FP8-vs-NVFP4 GDN A/B + KL-drift
                    // gate (not a user choice; mirrors the `ATLAS_NO_*` debug
                    // levers). Default engages native FP8.
                    if std::env::var_os("ATLAS_NO_GDN_FP8").is_none()
                        && proj_is_fp8_any_scale(store, &format!("{la}.in_proj_qkv"))
                        && proj_is_fp8_any_scale(store, &format!("{la}.in_proj_z"))
                        && proj_is_fp8_any_scale(store, &format!("{la}.out_proj"))
                    {
                        let in_proj_a = dense(store, &format!("{la}.in_proj_a.weight"))?;
                        let in_proj_b = dense(store, &format!("{la}.in_proj_b.weight"))?;
                        let conv1d = dense(store, &format!("{la}.conv1d.weight"))?;
                        let a_log = dense_keep_f32(store, &format!("{la}.A_log"), gpu)?;
                        let dt_bias = dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?;
                        let norm = dense_f32_safe(store, &format!("{la}.norm.weight"), gpu)?;
                        let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;
                        let qkv_f = load_fp8_block_scaled_as_fp8weight(
                            store,
                            &format!("{la}.in_proj_qkv"),
                            gpu,
                        )?;
                        let z_f = load_fp8_block_scaled_as_fp8weight(
                            store,
                            &format!("{la}.in_proj_z"),
                            gpu,
                        )?;
                        let out_f = load_fp8_block_scaled_as_fp8weight(
                            store,
                            &format!("{la}.out_proj"),
                            gpu,
                        )?;
                        let qkvz_f = concat_fp8_block_scaled(&qkv_f, &z_f, h, gpu)?;
                        // The concat copied both grids; free the per-projection
                        // scale allocs (weight bytes are store-owned, not freed).
                        gpu.free(qkv_f.row_scale)?;
                        gpu.free(z_f.row_scale)?;
                        let ssm = SsmWeights {
                            in_proj_qkvz: DenseWeight {
                                weight: spark_runtime::gpu::DevicePtr::NULL,
                            },
                            in_proj_ba: ba_dense,
                            conv1d,
                            a_log,
                            dt_bias,
                            norm,
                            out_proj: crate::weight_map::QuantizedWeight::null(),
                        };
                        let mut layer = Qwen3SsmLayer::new_sequential(
                            input_norm,
                            ssm,
                            post_attn_norm,
                            ffn,
                            None,
                            None,
                            None,
                            config,
                            gpu,
                        )?;
                        layer.set_fp8_decode_weights(Some(qkvz_f), Some(out_f));
                        tracing::info!(
                            "SSM[{lp}] native FP8 GDN: qkvz+out_proj block-scaled FP8 \
                             (no NVFP4 requant; prefill+decode via w8a16)"
                        );
                        layers.push(Box::new(layer));
                        continue;
                    }

                    // A, B, conv1d, A_log, dt_bias, norm are independent of the
                    // qkv/z/out_proj on-disk format below — load them once up
                    // front so both the native-NVFP4 fast path and the legacy
                    // dequant/requant path can share them.
                    //
                    // A_log and dt_bias MUST be FP32 — consumer kernels in
                    // `ssm_preprocess.cu` and `mamba2_ssm_decode.cu` declare
                    // them `const float*`. Loading via `dense()` kept BF16
                    // storage, reinterpreting 48-elt BF16 (96B) as 48-elt
                    // FP32 → per-head scrambled decay gates and exponential
                    // error amplification through GDR recurrence at long
                    // context. The MoE sister loader (`ssm_qwen35.rs`)
                    // already promotes these; dense was missing the mirror.
                    //
                    // in_proj_a/b: route through `load_ssm_proj` (not the raw
                    // `dense()` byte-reinterpret) so a Standard-NVFP4 A/B
                    // (U8-packed) checkpoint dequants correctly instead of
                    // being read as BF16 garbage.
                    let in_proj_a = load_ssm_proj(&format!("{la}.in_proj_a"), nv, h)?;
                    let in_proj_b = load_ssm_proj(&format!("{la}.in_proj_b"), nv, h)?;
                    let conv1d = dense(store, &format!("{la}.conv1d.weight"))?;
                    let a_log = dense_keep_f32(store, &format!("{la}.A_log"), gpu)?;
                    let dt_bias = dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?;
                    // norm.weight: use `dense_f32_safe` (FP32-aware: detects
                    // a fp32 checkpoint and truncates to BF16 with logging;
                    // bf16 passes through). Mirrors `weight_map/ssm_qwen35.rs`
                    // MoE sister loader (backported here 2026-05-20).
                    let norm = dense_f32_safe(store, &format!("{la}.norm.weight"), gpu)?;
                    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;
                    let qkvz_size = config.ssm_qkvz_size();

                    // Native Standard-NVFP4 GDN (pre-quantized checkpoint, e.g.
                    // sakamakismile): in_proj_qkv / in_proj_z / out_proj ship
                    // U8-packed NVFP4 directly on disk (`.weight` dtype UInt8,
                    // not `.weight_packed` — that's the compressed-tensors
                    // convention `load_ssm_proj` already dequants above). Load
                    // them straight into `QuantizedWeight` and concat on GPU,
                    // skipping the BF16-dequant→re-quantize roundtrip entirely
                    // (that roundtrip is what the FP8/BF16-opt-in paths above
                    // exist to avoid for FP8-native and BF16-preferring
                    // checkpoints; here there's no lossy step to avoid in the
                    // first place — the data is already NVFP4). Requires all
                    // three projections to be U8; a partial-U8 checkpoint
                    // falls through to the legacy path below, where the
                    // `load_ssm_proj` UInt8 branch added above still dequants
                    // each U8 tensor correctly on its own.
                    let native_nvfp4 = matches!(
                        store
                            .get(&format!("{la}.in_proj_qkv.weight"))
                            .map(|w| w.dtype),
                        Ok(WeightDtype::UInt8)
                    ) && matches!(
                        store
                            .get(&format!("{la}.in_proj_z.weight"))
                            .map(|w| w.dtype),
                        Ok(WeightDtype::UInt8)
                    ) && matches!(
                        store.get(&format!("{la}.out_proj.weight")).map(|w| w.dtype),
                        Ok(WeightDtype::UInt8)
                    );
                    if native_nvfp4 {
                        let qkv_qw = quantized_auto(
                            store,
                            &format!("{la}.in_proj_qkv"),
                            gpu,
                            Nvfp4Variant::Standard,
                        )?;
                        let z_qw = quantized_auto(
                            store,
                            &format!("{la}.in_proj_z"),
                            gpu,
                            Nvfp4Variant::Standard,
                        )?;
                        let qkvz_nvfp4 = qkv_qw.concat_rows(&z_qw, qkv_rows, z_rows, h, gpu)?;
                        let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

                        let out_proj_nvfp4 = quantized_auto(
                            store,
                            &format!("{la}.out_proj"),
                            gpu,
                            Nvfp4Variant::Standard,
                        )?;
                        let out_proj_nvfp4_t =
                            out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

                        let ssm = SsmWeights {
                            in_proj_qkvz: DenseWeight {
                                weight: spark_runtime::gpu::DevicePtr::NULL,
                            },
                            in_proj_ba: ba_dense,
                            conv1d,
                            a_log,
                            dt_bias,
                            norm,
                            out_proj: out_proj_nvfp4,
                        };
                        let mut layer = Qwen3SsmLayer::new_sequential(
                            input_norm,
                            ssm,
                            post_attn_norm,
                            ffn,
                            Some(qkvz_nvfp4),
                            Some(qkvz_nvfp4_t),
                            Some(out_proj_nvfp4_t),
                            config,
                            gpu,
                        )?;
                        layer.predequant_for_prefill(gpu, config, stream)?;
                        tracing::info!(
                            "SSM[{lp}] native NVFP4 GDN: qkvz+out_proj loaded pre-quantized \
                             (U8-packed on disk; no BF16 dequant/requant roundtrip)"
                        );
                        layers.push(Box::new(layer));
                        continue;
                    }

                    let qkv_dense = load_ssm_proj(&format!("{la}.in_proj_qkv"), qkv_rows, h)?;
                    let z_dense = load_ssm_proj(&format!("{la}.in_proj_z"), z_rows, h)?;
                    let out_proj_dense = load_ssm_proj(&format!("{la}.out_proj"), h, value_dim)?;

                    let qkvz_dense =
                        gpu_concat_rows(&qkv_dense, qkv_rows, &z_dense, z_rows, h, gpu)?;
                    // qkv/z BF16 are only inputs to the concat above; free them now
                    // rather than leaking them for the layer's lifetime (Atlas issue #A1).
                    gpu.free(qkv_dense.weight)?;
                    gpu.free(z_dense.weight)?;

                    // GDN ≥FP8 precision policy (2026-07-04). The nvidia
                    // Qwen3.6-27B-NVFP4 checkpoint ships GDN in_proj_qkv /
                    // out_proj as native F8_E4M3 (modelopt sensitivity
                    // analysis deliberately keeps the SSM projections
                    // high-precision); `load_ssm_proj` dequants them to BF16,
                    // and the code below then RE-quantizes to NVFP4 (4-bit) —
                    // a lossy double-quant of the exact tensors the toolchain
                    // protected. That regressed BFCL-ST ~7pt (non_live 85.4→
                    // 76.6) vs the 06-15 reference. When enabled, keep the
                    // BF16 dequant (≥FP8) and route qkvz + out_proj through the
                    // dense_gemv / dense_gemm dispatch (in_proj_qkvz +
                    // out_proj_dense fields), mirroring the MoE sister loader's
                    // arm (qwen35/load_layers/linear_attn_arms.rs). Gated for a
                    // clean A/B + KL-drift gate before flipping the default.
                    let gdn_bf16 = matches!(
                        std::env::var("ATLAS_GDN_BF16_WEIGHTS").ok().as_deref(),
                        Some("1")
                    );
                    if gdn_bf16 {
                        let ssm = SsmWeights {
                            in_proj_qkvz: DenseWeight {
                                weight: qkvz_dense.weight,
                            },
                            in_proj_ba: ba_dense,
                            conv1d,
                            a_log,
                            dt_bias,
                            norm,
                            // Unused: out_proj_dense (set below) has higher
                            // dispatch priority in both prefill and decode.
                            out_proj: crate::weight_map::QuantizedWeight::null(),
                        };
                        let mut layer = Qwen3SsmLayer::new_sequential(
                            input_norm,
                            ssm,
                            post_attn_norm,
                            ffn,
                            None,
                            None,
                            None,
                            config,
                            gpu,
                        )?;
                        layer.out_proj_dense = Some(out_proj_dense);
                        tracing::info!(
                            "SSM[{lp}] ATLAS_GDN_BF16_WEIGHTS: qkvz + out_proj kept BF16 \
                             (≥FP8; NVFP4 requant skipped)"
                        );
                        layers.push(Box::new(layer));
                        continue;
                    }

                    let qkvz_nvfp4 = quantize_to_nvfp4(
                        &qkvz_dense,
                        qkvz_size,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;

                    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

                    let out_proj_nvfp4 = quantize_to_nvfp4(
                        &out_proj_dense,
                        h,
                        value_dim,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;

                    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

                    // Native FP8 SSM prefill GEMM: build a single-scale FP8
                    // copy of `qkvz_dense` [qkvz_size, h] and `out_proj_dense`
                    // [h, value_dim] by direct BF16→FP8 truncation. SSM weight
                    // magnitudes fit in FP8 E4M3 range (|w| ≤ 448), so no
                    // separate scalar dequant is needed at GEMM time — the
                    // `fp8_gemm_n128` kernel interprets the FP8 bytes as
                    // values directly (mirrors how `predequant_nvfp4_to_fp8`
                    // bakes `scale2` into the FP8 stream). PCND: gated.
                    let (qkvz_fp8_prefill, out_proj_fp8_prefill) =
                        if let Some(b2f_k) = bf16_to_fp8_k {
                            let qkvz_total = (qkvz_size * h) as u32;
                            let qkvz_fp8 = gpu.alloc(qkvz_size * h)?;
                            crate::layers::ops::bf16_to_fp8(
                                gpu,
                                b2f_k,
                                qkvz_dense.weight,
                                qkvz_fp8,
                                qkvz_total,
                                stream,
                            )?;
                            let out_total = (h * value_dim) as u32;
                            let out_fp8 = gpu.alloc(h * value_dim)?;
                            crate::layers::ops::bf16_to_fp8(
                                gpu,
                                b2f_k,
                                out_proj_dense.weight,
                                out_fp8,
                                out_total,
                                stream,
                            )?;
                            gpu.synchronize(stream)?;
                            (Some(qkvz_fp8), Some(out_fp8))
                        } else {
                            (None, None)
                        };

                    // SSM prefill/decode always dispatch qkvz_nvfp4/_t and the NVFP4
                    // out_proj; the BF16 qkvz_dense / out_proj_dense were only quantize
                    // inputs. Free them rather than keep a third full-precision copy of
                    // the largest SSM tensor across every layer (Atlas issue #A1).
                    gpu.free(qkvz_dense.weight)?;
                    gpu.free(out_proj_dense.weight)?;

                    let ssm = SsmWeights {
                        in_proj_qkvz: DenseWeight {
                            weight: spark_runtime::gpu::DevicePtr::NULL,
                        },
                        in_proj_ba: ba_dense,
                        conv1d,
                        a_log,
                        dt_bias,
                        norm,
                        out_proj: out_proj_nvfp4,
                    };

                    let mut layer = Qwen3SsmLayer::new_sequential(
                        input_norm,
                        ssm,
                        post_attn_norm,
                        ffn,
                        Some(qkvz_nvfp4),
                        Some(qkvz_nvfp4_t),
                        Some(out_proj_nvfp4_t),
                        config,
                        gpu,
                    )?;
                    layer.predequant_for_prefill(gpu, config, stream)?;
                    // Install the FP8 prefill weights AFTER `predequant_for_prefill`
                    // (which sets `out_proj_fp8` from NVFP4 + scale2). The
                    // native-FP8 path overrides both pointers when active,
                    // routing prefill through `fp8_gemm_n128` instead of
                    // `w4a16_gemm_t`. Decode batch paths keep their NVFP4
                    // fallback (the `qkvz_nvfp4*` fields above).
                    if qkvz_fp8_prefill.is_some() || out_proj_fp8_prefill.is_some() {
                        layer.set_fp8_prefill_only_weights(qkvz_fp8_prefill, out_proj_fp8_prefill);
                    }
                    layers.push(Box::new(layer));
                }
                LayerType::SlidingAttention => {
                    unreachable!("unexpected SlidingAttention in this loader")
                }
                LayerType::Moe => unreachable!("Qwen3.5 dense has no standalone MoE layers"),
            }

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Qwen3.5 dense weight loader: {} layers ({} attention, {} SSM, dense FFN)",
            layers.len(),
            attn_idx,
            layers.len() - attn_idx,
        );

        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loaders_b::load_embedding(store, config)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loaders_b::load_final_norm(store, config)
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        loaders_b::load_lm_head(store, config)
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "Loading dense MTP weights (variant={:?}, hidden={}, inter={})",
            variant,
            config.hidden_size,
            config.intermediate_size,
        );
        // `load_mtp` auto-detects MoE vs dense FFN by inspecting the weight
        // names. For dense Qwen3.6-27B-FP8 it returns a MtpWeights with
        // `dense_ffn = Some(...)` and NULL placeholders for the MoE fields.
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        if mtp.dense_ffn.is_some() {
            tracing::info!("Dense MTP head ready (FP8 e4m3 projections + dense gate/up/down MLP)");
        } else {
            tracing::info!(
                "MoE MTP head ready ({} experts) — dense loader sees MoE bundle",
                mtp.experts.len(),
            );
        }
        Ok(Some(mtp))
    }

    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        // Dense Qwen3.5 / Holo VL checkpoints (e.g. Holo-3.1-0.8B, Ornith-1.0-9B)
        // ship the SAME Qwen3-VL ViT tower as their MoE siblings. The MoE
        // loader's `load_vision_encoder` reads only `store` + `config.vision`
        // (no MoE-specific state), so reuse it verbatim. The shared model
        // forward (`model/trait_impl/*`, gated on `vision_encoder.is_some()`)
        // then merges image embeddings — no dense-specific forward changes.
        super::qwen35::Qwen35WeightLoader.load_vision_encoder(store, config, gpu)
    }
}
