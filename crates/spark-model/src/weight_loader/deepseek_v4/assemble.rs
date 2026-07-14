// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::layers::MoeLayer;
use crate::layers::qwen3_attention::{
    CompressorWeights, HcHeadWeights, HcSiteWeights, HcWeights, MlaWeights, Qwen3AttentionLayer,
};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense, dense_auto,
    quantized, quantized_v2,
};

/// Load one MoE expert projection, dispatching by the on-disk format so the V4
/// loader handles every DeepSeek-V4-Flash checkpoint variant. The nvidia
/// checkpoint is heterogeneous — routed experts are NVFP4 but shared experts are
/// FP8 block-scaled:
///   - `.weight_packed` + `.weight_global_scale`  → CompressedTensors (RedHat) → `quantized_v2`
///   - `.weight_scale_2` (+ `.input_scale`)        → Standard NVFP4 (nvidia routed) → `quantized`
///   - `.weight` is F8_E4M3 + `.scale` (F8_E8M0)   → FP8 block-scaled (nvidia shared experts) →
///     `quantized_from_fp8` (FP8→BF16→NVFP4 at load)
///   - `.weight` is U8/I8 + `.scale` (F8_E8M0)     → NVFP4 with E8M0 block scales (DeepSeek-V4
///     ORIGINAL format, used by the MTP module) — not yet wired to a GEMM path
fn load_expert_proj(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    qctx: crate::weight_map::QuantizeCtx,
) -> Result<QuantizedWeight> {
    use spark_runtime::weights::WeightDtype;
    if store.contains(&format!("{prefix}.weight_packed")) {
        return quantized_v2(store, prefix, gpu);
    }
    if store.contains(&format!("{prefix}.weight_scale_2")) {
        return quantized(store, prefix, gpu);
    }
    // No global scale → either FP8 block-scaled or E8M0-microscaled NVFP4;
    // distinguish by the `.weight` tensor dtype.
    let (n, shape_k, dtype) = {
        let w = store
            .get(&format!("{prefix}.weight"))
            .with_context(|| format!("{prefix}: no .weight tensor"))?;
        (w.shape[0], w.shape[1], w.dtype)
    };
    match dtype {
        // FP8 block-scaled (nvidia shared experts): weight shape is logical [n,k].
        WeightDtype::FP8E4M3 => crate::weight_map::quantized_from_fp8(
            store,
            prefix,
            n,
            shape_k,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        ),
        // NVFP4 4-bit (2 values/byte) + E8M0 `.scale`, no global — DeepSeek-V4's
        // ORIGINAL native MXFP4 routed format. Land the bytes device-resident
        // UNCHANGED (transcode-free): NO dequant, NO re-quantize. Tagged
        // `Mxfp4E8m0` at the MoE-layer level (see `detect_routed_scale_kind`);
        // the E8M0 GEMM variants (Phase-K) consume the E8M0 scales directly.
        // WAS: `dequant_nvfp4_e8m0_to_bf16 → quantize_to_nvfp4` = TWO lossy
        // 4-bit conversions at load (MXFP4→BF16→NVFP4) — the founding-scar path
        // ARM-2 removes. `n`/`shape_k`/`qctx`/`gpu` unused on this arm now.
        WeightDtype::UInt8 => {
            let qw = crate::weight_map::quantized_mxfp4_e8m0(store, prefix)?;
            maybe_dump_expert0(prefix, &qw, gpu)?;
            Ok(qw)
        }
        other => anyhow::bail!(
            "{prefix}: unsupported expert weight dtype {other:?} (expected FP8E4M3 or UInt8)"
        ),
    }
}

/// Phase-K STEP 0 (gating): one-shot device byte-check of the native-MXFP4
/// loader. When `ATLAS_DUMP_EXPERT0=1`, dumps `layers.0.ffn.experts.0.w1`'s
/// device-resident packed nibbles (`.weight`) + E8M0 scales (`.weight_scale`)
/// to `/tmp` for sha256 vs the on-disk reference (`ARM-2-LEG1-BYTE-CHECK.md`:
/// weight `177ac128…`, scale `ea4ac989…`). Dumps the PRE-transpose buffer (the
/// loader output, before any `transpose_for_gemm` swizzle) — matches the Leg-1
/// caveat. Sizes are fixed by the frozen ckpt: weight 4194304 B, scale 262144 B.
fn maybe_dump_expert0(prefix: &str, qw: &QuantizedWeight, gpu: &dyn GpuBackend) -> Result<()> {
    if std::env::var("ATLAS_DUMP_EXPERT0").as_deref() != Ok("1") {
        return Ok(());
    }
    if !prefix.ends_with("layers.0.ffn.experts.0.w1") {
        return Ok(());
    }
    let mut w = vec![0u8; 4194304];
    gpu.copy_d2h(qw.weight, &mut w)?;
    let mut s = vec![0u8; 262144];
    gpu.copy_d2h(qw.weight_scale, &mut s)?;
    std::fs::write("/tmp/atlas_expert0_w1_weight.bin", &w)?;
    std::fs::write("/tmp/atlas_expert0_w1_scale.bin", &s)?;
    eprintln!(
        "ATLAS_DUMP_EXPERT0: dumped {prefix} weight={} B scale={} B to /tmp/atlas_expert0_w1_*.bin",
        w.len(),
        s.len()
    );
    Ok(())
}

/// Detect the landed quant format of the ROUTED experts, for the E8M0 MoE-GEMM
/// dispatch tag (`MoeLayer::experts_scale_kind`). Mirrors `load_expert_proj`'s
/// format dispatch: ONLY the `UInt8 .weight` + `.scale` + no-global/packed arm
/// lands native MXFP4 (transcode-free, via `quantized_mxfp4_e8m0`); every other
/// arm produces standard NVFP4. Probes the first locally-owned routed expert
/// (EP-safe — each rank owns some); defaults to `Nvfp4` if none present.
fn detect_routed_scale_kind(
    store: &WeightStore,
    layer_prefix: &str,
    config: &ModelConfig,
    force_all_experts: bool,
) -> crate::weight_map::WeightQuantFormat {
    use crate::weight_map::WeightQuantFormat;
    use spark_runtime::weights::WeightDtype;
    for e in 0..config.num_experts {
        if force_all_experts || config.is_local_expert(e) {
            let wp = format!("{layer_prefix}.ffn.experts.{e}.w1");
            let native = !store.contains(&format!("{wp}.weight_packed"))
                && !store.contains(&format!("{wp}.weight_scale_2"))
                && store.contains(&format!("{wp}.scale"))
                && store
                    .get(&format!("{wp}.weight"))
                    .map(|w| w.dtype == WeightDtype::UInt8)
                    .unwrap_or(false);
            return if native {
                WeightQuantFormat::Mxfp4E8m0
            } else {
                WeightQuantFormat::Nvfp4
            };
        }
    }
    WeightQuantFormat::Nvfp4
}

/// Detect the SHARED expert quant format (ARM-2 Phase-K RIDER A1). Same native
/// -MXFP4 test as `detect_routed_scale_kind`, on `ffn.shared_experts.w1`. The
/// native V4 ckpt ships the shared expert FP8-block-scaled (`.weight` F8_E4M3,
/// NOT UInt8) → `Nvfp4` (it is transcoded to NVFP4 at load); a genuinely native
/// MXFP4 shared expert (UInt8 `.weight` + `.scale`) → `Mxfp4E8m0`.
fn detect_shared_scale_kind(
    store: &WeightStore,
    layer_prefix: &str,
) -> crate::weight_map::WeightQuantFormat {
    use crate::weight_map::WeightQuantFormat;
    use spark_runtime::weights::WeightDtype;
    let wp = format!("{layer_prefix}.ffn.shared_experts.w1");
    let native = !store.contains(&format!("{wp}.weight_packed"))
        && !store.contains(&format!("{wp}.weight_scale_2"))
        && store.contains(&format!("{wp}.scale"))
        && store
            .get(&format!("{wp}.weight"))
            .map(|w| w.dtype == WeightDtype::UInt8)
            .unwrap_or(false);
    if native {
        WeightQuantFormat::Mxfp4E8m0
    } else {
        WeightQuantFormat::Nvfp4
    }
}

#[allow(clippy::too_many_arguments)]
pub fn assemble_layer(
    layer_idx: usize,
    layer_prefix: &str,
    // When true, load ALL experts locally regardless of EP sharding. Used for
    // the MTP draft module, which runs only on rank 0 with no EP all-reduce, so
    // it needs every expert present.
    force_all_experts: bool,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    wq_a: DenseWeight,
    wq_a_nvfp4: Option<QuantizedWeight>,
    wq_b: DenseWeight,
    wq_b_nvfp4: Option<QuantizedWeight>,
    q_a_norm: DenseWeight,
    wkv_a: DenseWeight,
    wkv_a_nvfp4: Option<QuantizedWeight>,
    wkv_b: DenseWeight,
    kv_a_norm: DenseWeight,
    o_dense: DenseWeight,
    o_nvfp4: Option<QuantizedWeight>,
    w_uk_t: DenseWeight,
    w_uv: DenseWeight,
    wq_b_rope: DenseWeight,
    w_qk_absorbed: DenseWeight,
    w_uk_block_diag: DenseWeight,
    w_uv_block_diag: DenseWeight,
    yarn_inv_freq: DevicePtr,
    wo_a: DenseWeight,
    hc_head: Option<HcHeadWeights>,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Box<dyn TransformerLayer>> {
    // RedHatAI re-quant uses flattened naming: layers.N.* instead of model.layers.N.*
    // `layer_prefix` is the tensor-name prefix for this block: `layers.{idx}` for
    // main layers, or `mtp.0` for the MTP draft module (which reuses this exact
    // body — MLA + mHC + MoE). `layer_idx` is still used for the per-index meta
    // lookups (compress_ratios / hash-layer / kv-dtype); passing an out-of-range
    // index (= num_hidden_layers) for the MTP module makes all three fall to the
    // safe defaults: no compressor, no hash routing, bf16 KV.
    let lp = layer_prefix.to_string();
    let h = config.hidden_size;
    let kv_dtype = layer_kv_dtypes
        .get(layer_idx)
        .copied()
        .unwrap_or(KvCacheDtype::Bf16);

    // Quantize context for any FP8→NVFP4 runtime conversion (nvidia checkpoint
    // ships its shared experts as FP8 block-scaled; see `load_expert_proj`).
    let qctx = crate::weight_map::QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    };

    // ── MoE FFN ──
    // RedHatAI re-quant uses ffn.gate.weight and ffn.experts.E.w1/w2/w3 naming.
    let p = &lp;
    let gate = dense(store, &format!("{p}.ffn.gate.weight"))?;
    // ROUTER GATE MUST STAY HIGH-PRECISION. The checkpoint ships `ffn.gate.weight`
    // in BF16 (the quant recipes deliberately exclude the router from NVFP4). A
    // 4-bit router flips the low-margin top-6 expert selection on essentially
    // every token -> incoherent "token salad" output. Route through the BF16
    // `dense_gemv` path (gate_nvfp4 = None) instead of re-quantizing to NVFP4.
    let gate_nvfp4 = None;

    let mut experts = Vec::with_capacity(config.num_experts);
    for e in 0..config.num_experts {
        if force_all_experts || config.is_local_expert(e) {
            let ep = format!("{p}.ffn.experts.{e}");
            let gate_proj = load_expert_proj(store, &format!("{ep}.w1"), gpu, qctx)
                .with_context(|| format!("DeepSeek-V4 expert {e}: w1"))?;
            let up_proj = load_expert_proj(store, &format!("{ep}.w3"), gpu, qctx)
                .with_context(|| format!("DeepSeek-V4 expert {e}: w3"))?;
            let down_proj = load_expert_proj(store, &format!("{ep}.w2"), gpu, qctx)
                .with_context(|| format!("DeepSeek-V4 expert {e}: w2"))?;
            experts.push(ExpertWeight {
                gate_proj,
                up_proj,
                down_proj,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }
    // Shared expert: DeepSeek-V4 has n_shared_experts=1, always-on and UNGATED
    // (reference MoE.forward does `y += shared_experts(x)` after the routed
    // all-reduce). It is NOT EP-sharded — every rank loads the full shared
    // expert and adds it once post-all-reduce (forward_prefill handles the
    // EP-once blend; moe_batched_blend treats a NULL gate as sigmoid=1.0).
    // The weights live under `ffn.shared_experts.{w1,w2,w3}` (NVFP4), same
    // packing as the routed experts. Leaving this null caused the MoE prefill
    // to dereference null gate/up/down pointers (CUDA illegal address).
    let sep = format!("{p}.ffn.shared_experts");
    let shared_expert = ExpertWeight {
        gate_proj: load_expert_proj(store, &format!("{sep}.w1"), gpu, qctx)
            .with_context(|| "DeepSeek-V4 shared expert: w1")?,
        up_proj: load_expert_proj(store, &format!("{sep}.w3"), gpu, qctx)
            .with_context(|| "DeepSeek-V4 shared expert: w3")?,
        down_proj: load_expert_proj(store, &format!("{sep}.w2"), gpu, qctx)
            .with_context(|| "DeepSeek-V4 shared expert: w2")?,
    };

    // ── Shared-expert gate ──
    // DeepSeek-V4-Flash ships `ffn.shared_expert_gate.weight` (RedHatAI
    // re-quant) or `mlp.shared_expert_gate.weight` (original HF checkpoint).
    // The shared expert is sigmoid-gated (`output += sigmoid(gate·x) * shared`),
    // NOT ungated. Leaving it NULL makes `moe_weighted_sum_blend` treat the
    // gate as 1.0, over-adding the shared expert every layer.
    let shared_expert_gate = match store
        .get(&format!("{p}.ffn.shared_expert_gate.weight"))
        .or_else(|_| store.get(&format!("{p}.mlp.shared_expert_gate.weight")))
    {
        Ok(t) => DenseWeight { weight: t.ptr },
        Err(_) => DenseWeight {
            weight: DevicePtr::NULL,
        },
    };

    // ── MoE routing (noaux_tc, sigmoid scoring) ──
    // DeepSeek-V4 scores experts with `sigmoid(logits)` and normalizes the
    // top-k gate weights as `σ(l_i)/Σσ(l_j)` — NOT softmax. The MoE layer
    // keys the sigmoid path off `correction_bias` being `Some`, so we must
    // supply a bias even though V4-Flash ships none: a zero `[num_experts]`
    // buffer gives `topk(sigmoid(logits) + 0)` with correct weight
    // normalization. (Falling back to softmax selects right experts but
    // wrong blend weights → degraded output.)
    let correction_bias = load_correction_bias(store, &lp, config.num_experts, gpu)?;

    let moe_weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias,
    };
    // ── Hash-MoE routing (DeepSeek-V4 paper §2.1) ──
    // The first `num_hash_layers` MoE layers select experts via a static
    // `tid2eid[token_id]` table ([vocab_size, top_k] i64) instead of top-K of
    // the gate scores. The learned gate still supplies the sqrtsoftplus scores
    // that weight the selected experts. `Some(table)` here is the SSOT marking
    // this as a hash-routed layer.
    let tid2eid_dev = if layer_idx < config.num_hash_layers {
        let t = store
            .get(&format!("{lp}.ffn.gate.tid2eid"))
            .with_context(|| {
                format!(
                    "DeepSeek-V4 hash layer {layer_idx}: missing ffn.gate.tid2eid \
                 (num_hash_layers={})",
                    config.num_hash_layers
                )
            })?;
        let expected = config.vocab_size * config.num_experts_per_tok;
        anyhow::ensure!(
            t.num_elements() == expected,
            "DeepSeek-V4 tid2eid layer {layer_idx}: {} elements != vocab_size*top_k ({})",
            t.num_elements(),
            expected
        );
        Some(t.ptr)
    } else {
        None
    };
    let mut moe = MoeLayer::new_with_hash(
        moe_weights,
        config.num_experts,
        gate_nvfp4,
        tid2eid_dev,
        gpu,
        config,
    )?;
    // Tag routed-expert quant format so the Phase-K E8M0 MoE-GEMM variants
    // dispatch on native MXFP4 (transcode-free) vs the standard NVFP4 kernels.
    moe.experts_scale_kind = detect_routed_scale_kind(store, p, config, force_all_experts);
    // Tag the SHARED expert format independently (RIDER A1): the native ckpt is
    // heterogeneous — routed E8M0-MXFP4, shared FP8→NVFP4. The dual-format decode
    // kernel asserts shared==Nvfp4; a different shared format fires the `expect`.
    moe.shared_experts_scale_kind = detect_shared_scale_kind(store, p);

    // ── MLA weights ──
    // RedHatAI checkpoint: wkv_a may only contain kv_lora_rank rows (no rope).
    // Try loading a separate rope tensor; if absent, allocate a zero buffer.
    let wkv_a_rope = if let Ok(rope_w) = store.get(&format!("{lp}.attn.wkv_rope.weight")) {
        DenseWeight { weight: rope_w.ptr }
    } else {
        let rope_bytes = config.qk_rope_head_dim * h * 2;
        let rope_ptr = gpu.alloc(rope_bytes)?;
        gpu.memset(rope_ptr, 0, rope_bytes)?;
        DenseWeight { weight: rope_ptr }
    };

    // ── DeepSeek Sparse Attention compressor (compressed layers only) ──
    // compress_ratios[L]: 0 = full attention (no compressor); 4 = CSA (2*hd proj,
    // overlap window, + indexer); 128 = HCA (hd proj, single window).
    let compressor = {
        let ratio = config.compress_ratios.get(layer_idx).copied().unwrap_or(0);
        if ratio > 0 {
            let cp = format!("{lp}.attn.compressor");
            let is_csa = ratio < 128; // 4 = CSA; 128 = HCA
            let hd = config.head_dim;
            let proj_dim = if is_csa { 2 * hd } else { hd };
            let wkv = dense(store, &format!("{cp}.wkv.weight"))?;
            let wgate = dense(store, &format!("{cp}.wgate.weight"))?;
            // compressor.norm is a STANDARD RMSNorm → subtract 1 for the offset kernel.
            let norm = dense_auto(store, &format!("{cp}.norm.weight"), gpu)?;
            let ape = store.get(&format!("{cp}.ape"))?.ptr;
            // 4b: allocate the persistent flat compressed-KV pool for this layer.
            // Sized to the full context (max_position_embeddings // ratio blocks)
            // so decode never overflows; each block is one hd_mla-wide FP8-E4M3
            // comp_k. FP8 (1 byte/elem) matches the raw KV arm's dtype/scale so the
            // decode kernel reads both arms at one scale (single online softmax).
            // ponytail: sized from model max_pos, not the runtime --max-seq-len;
            // tighten to the KV budget if the ~3.3GB CSA-layer total ever matters.
            // Block width = qk_nope_head_dim + qk_rope_head_dim (= 448+64 = 512), the
            // width cache_skip_v4 builds comp_k at (rope in-place at 448-511). NOT
            // kv_lora_rank+rope (576, the RAW cache token) — the compressed pool and
            // the decode compressed-arm read must both use 512, or blocks ≥1 misalign.
            let hd_mla = config.qk_nope_head_dim + config.qk_rope_head_dim;
            let pool_blocks = config.max_position_embeddings.div_ceil(ratio);
            let pool = gpu.alloc(pool_blocks * hd_mla)?;
            gpu.memset(pool, 0, pool_blocks * hd_mla)?;
            // 4b inc-3: decode-time normed-x rings (BF16, 2 bytes/elem). `ring`
            // holds the current window's `ratio` tokens; CSA also keeps the
            // previous window (`prev_win`) + a 2×ratio concat stage for the
            // overlap. HCA has no overlap → NULL prev_win/stage. Sizes are tiny
            // (HCA r=128,h≈4096 → 1 MB ring/layer; CSA r=4 → ~kB).
            let h = config.hidden_size;
            let ring = gpu.alloc(ratio * h * 2)?;
            let (prev_win, stage) = if is_csa {
                (gpu.alloc(ratio * h * 2)?, gpu.alloc(2 * ratio * h * 2)?)
            } else {
                (
                    spark_runtime::gpu::DevicePtr::NULL,
                    spark_runtime::gpu::DevicePtr::NULL,
                )
            };
            Some(CompressorWeights {
                wkv,
                wgate,
                norm,
                ape,
                ratio,
                proj_dim,
                is_csa,
                pool,
                pool_blocks,
                ring,
                prev_win,
                stage,
            })
        } else {
            None
        }
    };

    // Per-head attention sink logit (s_aux); present on all V4 attention layers.
    let attn_sink = store
        .get(&format!("{lp}.attn.attn_sink"))
        .map(|w| w.ptr)
        .unwrap_or(DevicePtr::NULL);

    // Native block-scaled FP8 weights for the hot decode GEMVs (the checkpoint
    // ships wq_a/wq_b/wo_b as FP8-E4M3 + 128×128 block scales). The decode path
    // reads these at 1 byte/elem via w8a16_gemv instead of the BF16 dequant's
    // 2 bytes — ~2× less weight traffic on the memory-bound MLA projections,
    // lossless (in-kernel F32 dequant). `None` when the checkpoint isn't FP8.
    let load_fp8_mla = |suffix: &str| {
        crate::weight_map::load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{lp}.attn.{suffix}"),
            gpu,
        )
        .ok()
    };
    let wq_a_fp8 = load_fp8_mla("wq_a");
    let wq_b_fp8 = load_fp8_mla("wq_b");
    let wo_b_fp8 = load_fp8_mla("wo_b");
    let wo_a_fp8 = load_fp8_mla("wo_a");
    let wkv_a_fp8 = load_fp8_mla("wkv");

    let mla = MlaWeights {
        wq_a,
        wq_a_nvfp4,
        wq_a_fp8,
        wq_b,
        wq_b_nvfp4,
        wq_b_fp8,
        q_a_norm,
        wkv_a,
        wkv_a_nvfp4,
        wkv_a_fp8,
        wkv_b,
        kv_a_norm,
        wkv_a_rope,
        wkv_a_merged: DenseWeight {
            weight: wkv_a.weight,
        },
        wo: o_dense,
        wo_nvfp4: o_nvfp4,
        wo_a,
        wo_a_nvfp4: None,
        wo_a_fp8,
        wo_b: o_dense,
        wo_b_nvfp4: None,
        wo_b_fp8,
        w_uk_t,
        w_uv,
        wq_b_rope,
        w_qk_absorbed,
        w_uk_block_diag,
        w_uv_block_diag,
        yarn_inv_freq,
        main_inv_freq: super::compute::main_inv_freq(config, gpu)?,
        q_lora_rank: config.q_lora_rank,
        kv_lora_rank: config.kv_lora_rank,
        o_lora_rank: config.o_lora_rank,
        nope: config.qk_nope_head_dim,
        rope: config.qk_rope_head_dim,
        v_dim: config.v_head_dim,
        compressor,
        attn_sink,
    };

    // ── Attention dummy + layer ──
    let attn = AttentionWeights {
        q_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        k_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        v_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        o_proj: QuantizedWeight::null(),
        q_norm: DenseWeight {
            weight: DevicePtr::NULL,
        },
        k_norm: DenseWeight {
            weight: DevicePtr::NULL,
        },
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_attn_norm,
        FfnComponent::Moe(moe),
        layer_idx,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        0,
        config,
    )?;
    layer.set_mla_weights(mla);

    // ── Manifold-Constrained Hyper-Connections (mHC) ──
    // Every block keeps `hc_mult` residual streams mixed by a per-block
    // Sinkhorn matrix. Load-bearing: skipping it diverges the residual flow.
    if config.hc_mult > 0 {
        let attn = load_hc_site(store, &lp, "attn", config, gpu)?;
        let ffn = load_hc_site(store, &lp, "ffn", config, gpu)?;
        layer.set_hc_weights(HcWeights {
            attn,
            ffn,
            head: hc_head.clone(),
            hc_mult: config.hc_mult,
            sinkhorn_iters: config.hc_sinkhorn_iters,
            hc_eps: config.hc_eps,
        });
    }

    // V4 loads its norm weights EXACTLY (no -1 pre-subtraction) and normalizes
    // with `rms_norm_vanilla`. The only paths that would bypass that kernel are
    // the fused residual+norm ones, taken when the mHC highway is absent — and
    // they are offset-from-1 only, with no vanilla twin in-tree. They would
    // silently compute `x * (1 + w)` on an exact weight. Fail loudly instead.
    anyhow::ensure!(
        layer.hc.is_some(),
        "DeepSeek-V4 requires the mHC highway (hc_mult > 0, got {}): without it the fused \
         residual+norm kernels would apply the offset-from-1 convention to exactly-loaded \
         norm weights",
        config.hc_mult
    );

    Ok(Box::new(layer))
}

/// Load one HC site (`attn` or `ffn`) for a block as float32 device buffers.
/// Tries the flat `hc_<site>_*` (DeepSeek reference / RedHatAI re-quant) and
/// the HF `<site>_hc.*` naming. Fails fast if a tensor is missing — HC is
/// load-bearing, so a silent skip would produce gibberish.
fn load_hc_site(
    store: &WeightStore,
    layer_prefix: &str,
    site: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<HcSiteWeights> {
    let hc = config.hc_mult;
    let mix_hc = (2 + hc) * hc;
    let hc_dim = hc * config.hidden_size;
    let hc_fn = load_hc_f32(
        store,
        &[
            format!("{layer_prefix}.hc_{site}_fn"),
            format!("{layer_prefix}.{site}_hc.fn"),
        ],
        mix_hc * hc_dim,
        gpu,
    )
    .with_context(|| format!("DeepSeek-V4 HC {site} fn ({layer_prefix})"))?;
    let hc_base = load_hc_f32(
        store,
        &[
            format!("{layer_prefix}.hc_{site}_base"),
            format!("{layer_prefix}.{site}_hc.base"),
        ],
        mix_hc,
        gpu,
    )
    .with_context(|| format!("DeepSeek-V4 HC {site} base ({layer_prefix})"))?;
    let hc_scale = load_hc_f32(
        store,
        &[
            format!("{layer_prefix}.hc_{site}_scale"),
            format!("{layer_prefix}.{site}_hc.scale"),
        ],
        3,
        gpu,
    )
    .with_context(|| format!("DeepSeek-V4 HC {site} scale ({layer_prefix})"))?;
    Ok(HcSiteWeights {
        hc_fn,
        hc_base,
        hc_scale,
    })
}

/// Resolve the first existing tensor among `candidates`, returning an F32
/// device pointer (BF16 tensors are widened). Errors if none exist or the
/// element count mismatches `expect_n`.
pub fn load_hc_f32(
    store: &WeightStore,
    candidates: &[String],
    expect_n: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    use spark_runtime::weights::WeightDtype;
    let Some(t) = candidates.iter().find_map(|k| store.get(k).ok()) else {
        anyhow::bail!("HC tensor not found; tried {candidates:?}");
    };
    let n = t.num_elements();
    anyhow::ensure!(
        n == expect_n,
        "HC tensor length {n} != expected {expect_n} (tried {candidates:?})"
    );
    match t.dtype {
        WeightDtype::BF16 => {
            let mut bf16_buf = vec![0u8; n * 2];
            gpu.copy_d2h(t.ptr, &mut bf16_buf)?;
            let mut f32_buf = vec![0u8; n * 4];
            for i in 0..n {
                f32_buf[i * 4 + 2] = bf16_buf[i * 2];
                f32_buf[i * 4 + 3] = bf16_buf[i * 2 + 1];
            }
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(ptr)
        }
        WeightDtype::FP32 => Ok(t.ptr),
        other => anyhow::bail!(
            "load_hc_f32: unsupported dtype {:?} for HC weight (tried {candidates:?}). \
             HC kernels expect F32; BF16 is auto-widened. FP8/E8M0 weights need dequant support.",
            other
        ),
    }
}

/// Load the DeepSeek-V4 MoE `e_score_correction_bias` (noaux_tc loss-free
/// balancing bias) as an F32 `[num_experts]` tensor. The RedHatAI re-quant
/// flattens layer keys to `layers.N.ffn.gate.*`. Returns `None` (softmax
/// routing) only if the checkpoint genuinely lacks the tensor.
fn load_correction_bias(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<DenseWeight>> {
    use spark_runtime::weights::WeightDtype;
    let candidates = [
        format!("{layer_prefix}.ffn.gate.e_score_correction_bias"),
        format!("{layer_prefix}.ffn.gate.correction_bias"),
        format!("{layer_prefix}.mlp.gate.e_score_correction_bias"),
        // RedHat/HF V4-Flash names the noaux_tc selection bias `ffn.gate.bias`
        // (the router nn.Linear is bias=False, so this IS the e_score_correction
        // _bias — used ONLY for top-k selection; weights gather raw scores).
        format!("{layer_prefix}.ffn.gate.bias"),
        format!("{layer_prefix}.mlp.gate.bias"),
    ];
    let Some(bias_t) = candidates.iter().find_map(|k| store.get(k).ok()) else {
        // V4-Flash ships no correction bias (noaux_tc with bias=0). Supply a
        // zeroed F32 buffer so the MoE still routes via sigmoid scoring
        // (`σ(l_i)/Σσ(l_j)` weights) rather than softmax — `MoeLayer::new`
        // keys the sigmoid path off `correction_bias` being `Some`.
        let bytes = num_experts * 4;
        let ptr = gpu.alloc(bytes)?;
        gpu.memset(ptr, 0, bytes)?;
        return Ok(Some(DenseWeight { weight: ptr }));
    };
    let n = bias_t.num_elements();
    anyhow::ensure!(
        n == num_experts,
        "DeepSeek-V4 correction_bias length {n} != num_experts {num_experts}"
    );
    // The moe_topk_sigmoid kernel consumes an F32 bias. F32 tensors pass
    // through unchanged; BF16 must be widened (zero-extend mantissa).
    if bias_t.dtype == WeightDtype::BF16 {
        let mut bf16_buf = vec![0u8; n * 2];
        gpu.copy_d2h(bias_t.ptr, &mut bf16_buf)?;
        let mut f32_buf = vec![0u8; n * 4];
        for i in 0..n {
            f32_buf[i * 4 + 2] = bf16_buf[i * 2];
            f32_buf[i * 4 + 3] = bf16_buf[i * 2 + 1];
        }
        let ptr = gpu.alloc(f32_buf.len())?;
        gpu.copy_h2d(&f32_buf, ptr)?;
        Ok(Some(DenseWeight { weight: ptr }))
    } else {
        Ok(Some(DenseWeight { weight: bias_t.ptr }))
    }
}
