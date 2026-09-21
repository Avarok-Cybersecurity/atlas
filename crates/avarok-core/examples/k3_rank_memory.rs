// SPDX-License-Identifier: AGPL-3.0-only
//! K3 rank memory from config estimates or pinned headers, using production TP.
//! Usage: k3_rank_memory TP [HEADER_DIRECTORY [A_LOG_PAYLOAD_DIRECTORY]].
//! Reports weight payload and known binding copies, not measured GPU high-water.
use anyhow::Result;
use avarok_core::{
    config::parse_config,
    kimi_k3::{
        K3Graph, MixerKind,
        tp::{TpAxis, plan_tensor_bytes, tensor_plan},
        weights::required_names,
        weights_geometry::expected_replicated_shape,
    },
};
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let world: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "8".into())
        .parse()?;
    let mut c = parse_config(include_str!(
        "../../../docs/k3/fixtures/moonshotai-Kimi-K3-config.json"
    ))?;
    anyhow::ensure!(
        world > 0
            && c.num_attention_heads.is_multiple_of(world)
            && c.linear_num_key_heads.is_multiple_of(world),
        "invalid TP world"
    );
    c.tp_world_size = world;
    c.tp_rank = world - 1;
    c.num_attention_heads /= world;
    c.num_key_value_heads /= world;
    c.linear_num_key_heads /= world;
    c.linear_num_value_heads /= world;
    let mut headers = BTreeMap::new();
    let header_dir = std::env::args().nth(2);
    if let Some(dir) = &header_dir {
        for file in std::fs::read_dir(dir)? {
            let data: serde_json::Value = serde_json::from_slice(&std::fs::read(file?.path())?)?;
            for (name, meta) in data.as_object().unwrap() {
                if !name.starts_with("language_model.") && !name.starts_with("lm_head.") {
                    continue;
                }
                let shape: Vec<usize> = serde_json::from_value(meta["shape"].clone())?;
                let dtype = meta["dtype"].as_str().unwrap();
                let width = match dtype {
                    "BF16" => 2,
                    "F32" => 4,
                    "U8" | "F8_E8M0" => 1,
                    _ => anyhow::bail!("unsupported header dtype {dtype}: {name}"),
                };
                anyhow::ensure!(
                    headers.insert(name.clone(), (shape, width)).is_none(),
                    "duplicate header tensor"
                );
            }
        }
    }
    let graph = K3Graph::from_config(&c);
    let mut source = 0usize;
    let mut local = 0usize;
    let mut replicated = 0usize;
    let mut staging = 0usize;
    let mut largest = String::new();
    let mut binding_extra = 0usize;
    let mut host_f32_payload = 0usize;
    let mut host_copy_temporary = 0usize;
    let mut counted_tensors = 0usize;
    let a_log_dir = std::env::args().nth(3);
    let mut verified_a_log_tails = 0usize;
    let mut groups: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for name in required_names(&c) {
        let mixer = name
            .split_once("model.layers.")
            .map(|(_, tail)| {
                graph.layers[tail.split('.').next().unwrap().parse::<usize>().unwrap()].mixer
            })
            .unwrap_or(MixerKind::Kda);
        let (axis, n, k) = tensor_plan(&name, mixer, &c);
        let routed = name.contains(".block_sparse_moe.experts.");
        let group = if routed {
            "routed_experts"
        } else if name.contains(".shared_experts.") {
            "shared_experts"
        } else if name.contains("embed_tokens") || name.ends_with("lm_head.weight") {
            "embedding_lm_head"
        } else {
            "other_text"
        };
        let tensors = if routed {
            vec![
                (format!("{}_packed", name), vec![n, k / 2], 1),
                (format!("{}_scale", name), vec![n, k / 32], 1),
            ]
        } else {
            let shape = if axis == TpAxis::Replicated {
                expected_replicated_shape(&name, &c)?
            } else {
                vec![n, k]
            };
            vec![(name, shape, 2)]
        };
        for (name, shape, bytes) in tensors {
            let (shape, bytes) = if header_dir.is_some() {
                headers
                    .remove(&name)
                    .ok_or_else(|| anyhow::anyhow!("missing header: {name}"))?
            } else {
                (shape, bytes)
            };
            anyhow::ensure!(
                if routed {
                    bytes == 1
                } else {
                    matches!(bytes, 2 | 4)
                },
                "unsupported dtype role: {name}"
            );
            let p = plan_tensor_bytes(&name, &shape, bytes, mixer, &c)?;
            if p.axis == TpAxis::Replicated {
                avarok_core::kimi_k3::weights_geometry::validate_replicated_shape(
                    &name, &shape, &c,
                )?;
            }
            counted_tensors += 1;
            binding_extra += avarok_core::kimi_k3::binding_memory::extra_gpu_bytes(
                &name,
                bytes == 4,
                p.local_bytes() / bytes,
            )?;
            if (name.contains("model.layers.") && !routed)
                || name.contains("model.output_attn_res_")
            {
                host_f32_payload += p.local_bytes() / bytes * 4;
                host_copy_temporary = host_copy_temporary.max(p.local_bytes());
            }
            if let Some(dir) = &a_log_dir
                && name.ends_with(".self_attn.A_log")
            {
                let layer = name
                    .split_once("model.layers.")
                    .unwrap()
                    .1
                    .split('.')
                    .next()
                    .unwrap();
                p.validate_source(&std::fs::read(
                    std::path::Path::new(dir).join(format!("{layer}.bin")),
                )?)?;
                verified_a_log_tails += 1;
            }
            source += p.source_bytes;
            local += p.local_bytes();
            if p.axis == TpAxis::Replicated {
                replicated += p.local_bytes();
            }
            if p.local_bytes() > staging {
                staging = p.local_bytes();
                largest = name;
            }
            let row = groups.entry(group).or_default();
            row.0 += p.source_bytes;
            row.1 += p.local_bytes();
            if p.axis == TpAxis::Replicated {
                row.2 += p.local_bytes();
            }
        }
    }
    anyhow::ensure!(
        headers.is_empty(),
        "{} text tensors unaccounted: {:?}",
        headers.len(),
        headers.keys().take(5).collect::<Vec<_>>()
    );
    let binding = (local * 3).div_ceil(10);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "basis":if header_dir.is_some() {"official f831ab66814297da540d832a5235f8e904f29d06 safetensors headers, text only"} else {"config-derived, all non-routed text BF16; routed U8 MXFP4 + E8M0; no vision/MTP; not checkpoint-header verified"},
            "tp_world":world,"rank":c.tp_rank,"source_bytes":source,"local_bytes":local,"replicated_bytes":replicated,"largest_local_tensor_bytes":staging,"largest_local_tensor":largest,"old_30pct_binding_allowance_bytes":binding,"old_admission_before_reserve_bytes":local+binding+staging,
            "known_binding_extra_gpu_bytes":binding_extra,"admission_before_reserve_bytes":local+binding_extra+staging,
            "counted_text_tensors":counted_tensors,"verified_a_log_zero_tails":verified_a_log_tails,
            "host_f32_layer_payload_bytes":host_f32_payload,"host_f32_layer_payload_all_ranks_bytes":host_f32_payload*world,
            "largest_host_bind_raw_copy_bytes":host_copy_temporary,"groups_source_local_replicated":groups
        }))?
    );
    Ok(())
}
