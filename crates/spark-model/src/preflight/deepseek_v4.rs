// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed checkpoint contract for native DeepSeek V4 DSpark releases.

use std::collections::BTreeSet;

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::{WeightDtype, WeightStore};

const PROJECTIONS: [&str; 3] = ["w1", "w2", "w3"];
const ATTN_PROJECTIONS: [&str; 5] = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"];

pub(super) fn check_native_dspark_checkpoint(
    store: &WeightStore,
    config: &ModelConfig,
    use_speculative: bool,
) -> Result<()> {
    if config.model_type != "deepseek_v4" || config.dspark_block_size == 0 {
        return Ok(());
    }

    ensure!(
        config.hidden_size.is_multiple_of(128) && config.moe_intermediate_size.is_multiple_of(128),
        "DeepSeek-V4 native quant contract requires hidden and MoE dimensions divisible by 128"
    );
    ensure!(
        config.dspark_block_size == 5,
        "DeepSeek-V4 native DSpark contract requires K=5, found K={}",
        config.dspark_block_size,
    );

    check_target_quant_contract(store, config)?;
    let (local_start, local_end) = config.local_expert_range();
    let local_experts = local_end - local_start;
    tracing::info!(
        "DeepSeek-V4 target quant contract: EP rank {}/{} owns experts [{}..{}); \
         routed={} MXFP4/E8M0 projections -> E8M0 MoE; \
         shared={} FP8/E8M0 projections -> NVFP4 MoE; attention={} FP8/E8M0 projections -> BF16 dense",
        config.ep_rank,
        config.ep_world_size,
        local_start,
        local_end,
        config.num_hidden_layers * local_experts * PROJECTIONS.len(),
        config.num_hidden_layers * PROJECTIONS.len(),
        config.num_hidden_layers * ATTN_PROJECTIONS.len(),
    );

    if !use_speculative {
        tracing::info!(
            "DeepSeek-V4 checkpoint declares native DSpark K={}, but speculation is off; \
             draft tensors remain independent of the qualified target path",
            config.dspark_block_size,
        );
        return Ok(());
    }

    check_dspark_quant_contract(store, config)?;
    tracing::info!(
        "Pre-flight: checkpoint-native DeepSeek-V4 DSpark K={} passed its three-stage \
         tensor and quantization contract; enabling the native distributed proposer",
        config.dspark_block_size,
    );
    Ok(())
}

fn check_target_quant_contract(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    ensure!(
        config.num_experts > 0 && config.moe_intermediate_size > 0,
        "DeepSeek-V4 native quant contract requires routed experts"
    );
    for layer in 0..config.num_hidden_layers {
        let ffn = format!("layers.{layer}.ffn");
        check_expert_group(store, &ffn, config)?;
        check_shared_expert(store, &ffn, config)?;
        let attn = format!("layers.{layer}.attn");
        for projection in ATTN_PROJECTIONS {
            check_fp8_block_scaled(store, &format!("{attn}.{projection}"))?;
        }
    }
    Ok(())
}

fn check_dspark_quant_contract(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    let draft_config = native_dspark_config(config)?;
    let stages = discover_dspark_stages(store);
    let expected = BTreeSet::from([0usize, 1, 2]);
    ensure!(
        stages == expected,
        "DeepSeek-V4 DSpark requires stages {{0,1,2}}, found {stages:?}"
    );

    for stage in 0..3 {
        let root = format!("mtp.{stage}");
        let ffn = format!("{root}.ffn");
        check_expert_group(store, &ffn, &draft_config)
            .with_context(|| format!("DSpark stage {stage} routed experts"))?;
        check_shared_expert(store, &ffn, &draft_config)
            .with_context(|| format!("DSpark stage {stage} shared expert"))?;
        for projection in ATTN_PROJECTIONS {
            check_fp8_block_scaled(store, &format!("{root}.attn.{projection}"))
                .with_context(|| format!("DSpark stage {stage} attention"))?;
        }
        check_stage_support_tensors(store, stage, &draft_config)?;
        tracing::info!(
            "DeepSeek-V4 DSpark stage {stage} quant contract: expert rank {}/{} owns routed={} \
             full-width MXFP4/E8M0 projections, shared={} FP8/E8M0, \
             attention={} FP8/E8M0 projections",
            draft_config.ep_rank,
            draft_config.ep_world_size,
            (draft_config.local_expert_range().1 - draft_config.local_expert_range().0)
                * PROJECTIONS.len(),
            PROJECTIONS.len(),
            ATTN_PROJECTIONS.len(),
        );
    }
    Ok(())
}

fn native_dspark_config(config: &ModelConfig) -> Result<ModelConfig> {
    let mut draft = config.clone();
    if config.tp_world_size > 1 && config.ep_world_size == 1 {
        ensure!(
            config.tp_rank < config.tp_world_size,
            "native DSpark TP rank {} is outside world size {}",
            config.tp_rank,
            config.tp_world_size
        );
        draft.moe_intermediate_size = config
            .moe_intermediate_size
            .checked_mul(config.tp_world_size)
            .ok_or_else(|| anyhow::anyhow!("native DSpark routed width overflow"))?;
        draft.shared_expert_intermediate_size = config
            .shared_expert_intermediate_size
            .checked_mul(config.tp_world_size)
            .ok_or_else(|| anyhow::anyhow!("native DSpark shared width overflow"))?;
        draft.ep_rank = config.tp_rank;
        draft.ep_world_size = config.tp_world_size;
    }
    Ok(draft)
}

fn check_expert_group(store: &WeightStore, ffn: &str, config: &ModelConfig) -> Result<()> {
    let (local_start, local_end) = config.local_expert_range();
    ensure!(
        local_start < local_end && local_end <= config.num_experts,
        "invalid DeepSeek-V4 EP expert range [{local_start}..{local_end}) for {} experts",
        config.num_experts,
    );
    for expert in local_start..local_end {
        for projection in PROJECTIONS {
            let prefix = format!("{ffn}.experts.{expert}.{projection}");
            let (out_dim, in_dim) = projection_dims(projection, config);
            expect_tensor(
                store,
                &format!("{prefix}.weight"),
                WeightDtype::UInt8,
                &[out_dim, in_dim / 2],
            )?;
            expect_tensor(
                store,
                &format!("{prefix}.scale"),
                WeightDtype::FP8E8M0,
                &[out_dim, in_dim / 32],
            )?;
        }
    }
    Ok(())
}

fn check_shared_expert(store: &WeightStore, ffn: &str, config: &ModelConfig) -> Result<()> {
    for projection in PROJECTIONS {
        let prefix = format!("{ffn}.shared_experts.{projection}");
        let (out_dim, in_dim) = projection_dims(projection, config);
        expect_tensor(
            store,
            &format!("{prefix}.weight"),
            WeightDtype::FP8E4M3,
            &[out_dim, in_dim],
        )?;
        expect_tensor(
            store,
            &format!("{prefix}.scale"),
            WeightDtype::FP8E8M0,
            &[out_dim / 128, in_dim / 128],
        )?;
    }
    Ok(())
}

fn projection_dims(projection: &str, config: &ModelConfig) -> (usize, usize) {
    if projection == "w2" {
        (config.hidden_size, config.moe_intermediate_size)
    } else {
        (config.moe_intermediate_size, config.hidden_size)
    }
}

fn check_fp8_block_scaled(store: &WeightStore, prefix: &str) -> Result<()> {
    let weight_name = format!("{prefix}.weight");
    let weight = store
        .get(&weight_name)
        .with_context(|| format!("missing FP8 weight `{weight_name}`"))?;
    ensure!(
        weight.dtype == WeightDtype::FP8E4M3,
        "`{weight_name}` has {:?}, expected FP8 E4M3",
        weight.dtype,
    );
    ensure!(
        weight.shape.len() == 2,
        "`{weight_name}` has shape {:?}, expected a matrix",
        weight.shape,
    );
    let scale_shape = [weight.shape[0].div_ceil(128), weight.shape[1].div_ceil(128)];
    expect_tensor(
        store,
        &format!("{prefix}.scale"),
        WeightDtype::FP8E8M0,
        &scale_shape,
    )
}

fn check_stage_support_tensors(
    store: &WeightStore,
    stage: usize,
    config: &ModelConfig,
) -> Result<()> {
    let root = format!("mtp.{stage}");
    expect_tensor(
        store,
        &format!("{root}.attn_norm.weight"),
        WeightDtype::BF16,
        &[config.hidden_size],
    )?;
    expect_tensor(
        store,
        &format!("{root}.attn.q_norm.weight"),
        WeightDtype::BF16,
        &[config.q_lora_rank],
    )?;
    expect_tensor(
        store,
        &format!("{root}.attn.kv_norm.weight"),
        WeightDtype::BF16,
        &[config.kv_lora_rank],
    )?;
    expect_tensor(
        store,
        &format!("{root}.ffn_norm.weight"),
        WeightDtype::BF16,
        &[config.hidden_size],
    )?;
    expect_tensor(
        store,
        &format!("{root}.ffn.gate.weight"),
        WeightDtype::BF16,
        &[config.num_experts, config.hidden_size],
    )?;
    expect_tensor(
        store,
        &format!("{root}.attn.attn_sink"),
        WeightDtype::FP32,
        &[config.num_attention_heads * config.tp_world_size],
    )?;
    expect_tensor(
        store,
        &format!("{root}.ffn.gate.bias"),
        WeightDtype::FP32,
        &[config.num_experts],
    )?;

    check_hc_site(store, &root, "attn", config)?;
    check_hc_site(store, &root, "ffn", config)?;

    if stage == 0 {
        let main_proj_shape = [
            config.hidden_size,
            config.hidden_size * config.dspark_target_layer_ids.len(),
        ];
        check_fp8_block_scaled_shape(store, "mtp.0.main_proj", &main_proj_shape)?;
        expect_tensor(
            store,
            "mtp.0.main_norm.weight",
            WeightDtype::BF16,
            &[config.hidden_size],
        )?;
    }
    if stage == 2 {
        expect_tensor(
            store,
            "mtp.2.confidence_head.proj.weight",
            WeightDtype::BF16,
            &[1, config.hidden_size + config.dspark_markov_rank],
        )?;
        for name in [
            "mtp.2.markov_head.markov_w1.weight",
            "mtp.2.markov_head.markov_w2.weight",
        ] {
            expect_tensor(
                store,
                name,
                WeightDtype::BF16,
                &[config.vocab_size, config.dspark_markov_rank],
            )?;
        }
        expect_tensor(
            store,
            "mtp.2.norm.weight",
            WeightDtype::BF16,
            &[config.hidden_size],
        )?;
        check_hc_head(store, "mtp.2", config)?;
    }
    Ok(())
}

fn check_fp8_block_scaled_shape(
    store: &WeightStore,
    prefix: &str,
    shape: &[usize; 2],
) -> Result<()> {
    expect_tensor(
        store,
        &format!("{prefix}.weight"),
        WeightDtype::FP8E4M3,
        shape,
    )?;
    expect_tensor(
        store,
        &format!("{prefix}.scale"),
        WeightDtype::FP8E8M0,
        &[shape[0] / 128, shape[1] / 128],
    )
}

fn check_hc_site(store: &WeightStore, root: &str, site: &str, config: &ModelConfig) -> Result<()> {
    let mix_hc = config.hc_mult * (config.hc_mult + 2);
    let hc_dim = config.hc_mult * config.hidden_size;
    expect_tensor(
        store,
        &format!("{root}.hc_{site}_base"),
        WeightDtype::FP32,
        &[mix_hc],
    )?;
    expect_tensor(
        store,
        &format!("{root}.hc_{site}_fn"),
        WeightDtype::FP32,
        &[mix_hc, hc_dim],
    )?;
    expect_tensor(
        store,
        &format!("{root}.hc_{site}_scale"),
        WeightDtype::FP32,
        &[config.hc_mult - 1],
    )
}

fn check_hc_head(store: &WeightStore, root: &str, config: &ModelConfig) -> Result<()> {
    let hc_dim = config.hc_mult * config.hidden_size;
    expect_tensor(
        store,
        &format!("{root}.hc_head_base"),
        WeightDtype::FP32,
        &[config.hc_mult],
    )?;
    expect_tensor(
        store,
        &format!("{root}.hc_head_fn"),
        WeightDtype::FP32,
        &[config.hc_mult, hc_dim],
    )?;
    expect_tensor(
        store,
        &format!("{root}.hc_head_scale"),
        WeightDtype::FP32,
        &[1],
    )
}

fn discover_dspark_stages(store: &WeightStore) -> BTreeSet<usize> {
    store
        .names()
        .filter_map(|name| {
            let tail = name.strip_prefix("mtp.")?;
            tail.split('.').next()?.parse::<usize>().ok()
        })
        .collect()
}

fn expect_tensor(
    store: &WeightStore,
    name: &str,
    dtype: WeightDtype,
    shape: &[usize],
) -> Result<()> {
    let tensor = store
        .get(name)
        .with_context(|| format!("missing required tensor `{name}`"))?;
    ensure!(
        tensor.dtype == dtype,
        "`{name}` has {:?}, expected {dtype:?}",
        tensor.dtype,
    );
    ensure!(
        tensor.shape == shape,
        "`{name}` has shape {:?}, expected {shape:?}",
        tensor.shape,
    );
    Ok(())
}

#[cfg(test)]
mod tests;
