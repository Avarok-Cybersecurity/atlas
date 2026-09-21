// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::config::parse_config;

fn production(rank: usize, world: usize) -> ModelConfig {
    let mut c = parse_config(include_str!(
        "../../../../../docs/k3/fixtures/moonshotai-Kimi-K3-config.json"
    ))
    .unwrap();
    c.tp_rank = rank;
    c.tp_world_size = world;
    c.num_attention_heads /= world;
    c.linear_num_key_heads /= world;
    c
}

#[test]
fn official_tp8_expert_storage_shapes() {
    let c = production(3, 8);
    for (projection, packed, scales) in [
        ("w1", [384, 1792], [384, 112]),
        ("w3", [384, 1792], [384, 112]),
        ("w2", [3584, 192], [3584, 12]),
    ] {
        let prefix =
            format!("language_model.model.layers.1.block_sparse_moe.experts.11.{projection}");
        let full = if projection == "w2" {
            [3584, 1536]
        } else {
            [3072, 1792]
        };
        let p = plan_tensor_bytes(
            &format!("{prefix}.weight_packed"),
            &full,
            1,
            MixerKind::Kda,
            &c,
        )
        .unwrap();
        assert_eq!(p.local_shape, packed);
        let full_scale = [full[0], full[1] / 16];
        let s = plan_tensor_bytes(
            &format!("{prefix}.weight_scale"),
            &full_scale,
            1,
            MixerKind::Kda,
            &c,
        )
        .unwrap();
        assert_eq!(s.local_shape, scales);
        assert_eq!(p.local_bytes(), s.local_bytes() * 16);
    }
}

#[test]
fn packed_tp_rejects_partial_scale_groups_and_bad_rank() {
    let mut c = production(0, 8);
    c.moe_intermediate_size = 288;
    let key = "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight_packed";
    assert!(plan_tensor_bytes(key, &[3584, 144], 1, MixerKind::Kda, &c).is_err());
    c = production(8, 8);
    assert!(plan_tensor_bytes(key, &[3584, 1536], 1, MixerKind::Kda, &c).is_err());
    c.tp_world_size = 0;
    assert!(plan_tensor_bytes(key, &[3584, 1536], 1, MixerKind::Kda, &c).is_err());
}

fn reconstruct(name: &str, shape: &[usize], element_bytes: usize, world: usize) {
    let size = shape.iter().product::<usize>() * element_bytes;
    let source: Vec<u8> = (0..size)
        .map(|i| ((i.wrapping_mul(37) ^ (i >> 8) ^ (i >> 16)) & 255) as u8)
        .collect();
    let mut rebuilt = vec![0u8; size];
    let mut seen = vec![false; size];
    for rank in 0..world {
        let c = production(rank, world);
        let plan = plan_tensor_bytes(name, shape, element_bytes, MixerKind::Kda, &c).unwrap();
        assert_eq!(plan.source_bytes, size);
        assert_eq!(plan.local_bytes() * world, size);
        let shard = plan.copy_shard(&source).unwrap();
        let mut cursor = 0;
        for range in plan.byte_ranges() {
            let end = cursor + range.len();
            assert!(
                seen[range.clone()].iter().all(|&v| !v),
                "rank slices overlap"
            );
            rebuilt[range.clone()].copy_from_slice(&shard[cursor..end]);
            seen[range].fill(true);
            cursor = end;
        }
        assert_eq!(cursor, shard.len());
    }
    assert!(seen.iter().all(|&v| v), "rank slices leave holes");
    assert_eq!(rebuilt, source);
}

#[test]
fn official_experts_tp2_tp8_reconstruct_packed_and_scales_exactly() {
    for world in [2, 8] {
        for projection in ["w1", "w2", "w3"] {
            let shape = if projection == "w2" {
                [3584, 1536]
            } else {
                [3072, 1792]
            };
            let prefix =
                format!("language_model.model.layers.1.block_sparse_moe.experts.111.{projection}");
            reconstruct(&format!("{prefix}.weight_packed"), &shape, 1, world);
            reconstruct(
                &format!("{prefix}.weight_scale"),
                &[shape[0], shape[1] / 16],
                1,
                world,
            );
        }
    }
}

#[test]
fn malformed_packed_shape_dtype_and_source_length_refuse() {
    let c = production(0, 8);
    let key = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed";
    assert!(plan_tensor_bytes(key, &[3072, 1791], 1, MixerKind::Kda, &c).is_err());
    assert!(plan_tensor_bytes(key, &[3072, 1792], 2, MixerKind::Kda, &c).is_err());
    assert!(
        plan_tensor_bytes(
            "mystery.weight_packed",
            &[3072, 1792],
            1,
            MixerKind::Kda,
            &c
        )
        .is_err()
    );
    let plan = plan_tensor_bytes(key, &[3072, 1792], 1, MixerKind::Kda, &c).unwrap();
    assert!(plan.copy_shard(&[0; 16]).is_err());
    assert!(plan_tensor_bytes("replicated", &[usize::MAX, 2], 4, MixerKind::Kda, &c).is_err());
}

#[test]
fn dense_preupload_preserves_f32_dtype_and_conv_shape() {
    let mut c = production(1, 2);
    c.hidden_size = 4;
    c.linear_num_key_heads = 2;
    c.linear_key_head_dim = 2;
    let q = plan_tensor_bytes(
        "language_model.model.layers.1.self_attn.q_proj.weight",
        &[8, 4],
        4,
        MixerKind::Kda,
        &c,
    )
    .unwrap();
    assert_eq!(q.local_shape, [4, 4]);
    let src: Vec<u8> = (0..32u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    assert_eq!(q.copy_shard(&src).unwrap(), src[64..]);
    let conv = plan_tensor_bytes(
        "language_model.model.layers.1.self_attn.q_conv1d.weight",
        &[8, 1, 4],
        2,
        MixerKind::Kda,
        &c,
    )
    .unwrap();
    assert_eq!(conv.local_shape, [4, 1, 4]);
}

#[test]
fn packed_shards_dequantize_to_independent_logical_slices() {
    use crate::mxfp4_e8m0::dequant_nvfp4_e8m0_to_f32 as dequant;
    for world in [2, 8] {
        for projection in ["w1", "w2", "w3"] {
            let (n, k) = if projection == "w2" {
                (64, 256)
            } else {
                (256, 64)
            };
            let packed: Vec<u8> = (0..n * k / 2)
                .map(|i| ((i * 17 + i / 37) % 256) as u8)
                .collect();
            let scales: Vec<u8> = (0..n * k / 32).map(|i| 124 + (i % 7) as u8).collect();
            let full = dequant(&packed, &scales, n, k).unwrap();
            for rank in 0..world {
                let mut c = production(rank, world);
                c.moe_intermediate_size = 256;
                c.moe_latent_size = 64;
                let prefix = format!(
                    "language_model.model.layers.1.block_sparse_moe.experts.0.{projection}"
                );
                let wp = plan_tensor_bytes(
                    &format!("{prefix}.weight_packed"),
                    &[n, k / 2],
                    1,
                    MixerKind::Kda,
                    &c,
                )
                .unwrap();
                let sp = plan_tensor_bytes(
                    &format!("{prefix}.weight_scale"),
                    &[n, k / 32],
                    1,
                    MixerKind::Kda,
                    &c,
                )
                .unwrap();
                let w = wp.copy_shard(&packed).unwrap();
                let s = sp.copy_shard(&scales).unwrap();
                let ln = if projection == "w2" { n } else { n / world };
                let lk = if projection == "w2" { k / world } else { k };
                let got = dequant(&w, &s, ln, lk).unwrap();
                let mut want = Vec::new();
                for row in 0..ln {
                    let source_row = if projection == "w2" {
                        row
                    } else {
                        row + rank * ln
                    };
                    let start_col = if projection == "w2" { rank * lk } else { 0 };
                    want.extend_from_slice(
                        &full[source_row * k + start_col..source_row * k + start_col + lk],
                    );
                }
                assert_eq!(got, want, "{projection} TP{world} rank {rank}");
                let mut wrong = s.clone();
                wrong[0] += 1;
                assert_ne!(
                    dequant(&w, &wrong, ln, lk).unwrap(),
                    want,
                    "wrong scale must fail the numerical oracle"
                );
            }
        }
    }
}

#[test]
fn official_a_log_padding_slices_only_96_heads_and_rejects_real_tail() {
    let name = "language_model.model.layers.0.self_attn.A_log";
    let values: Vec<f32> = (0..96).map(|h| h as f32 + 0.25).chain([0.0; 32]).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    for world in [1, 2, 8] {
        let mut reconstructed = Vec::new();
        for rank in 0..world {
            let c = production(rank, world);
            let plan = plan_tensor_bytes(name, &[128], 4, MixerKind::Kda, &c).unwrap();
            assert_eq!(plan.source_bytes, 512);
            assert_eq!(plan.local_shape, [96 / world]);
            reconstructed.extend(plan.copy_shard(&bytes).unwrap());
            for bad in [1.0f32, f32::NAN, f32::INFINITY] {
                let mut corrupt = bytes.clone();
                corrupt[127 * 4..128 * 4].copy_from_slice(&bad.to_le_bytes());
                assert!(plan.validate_source(&corrupt).is_err());
                assert!(plan.copy_shard(&corrupt).is_err());
            }
        }
        assert_eq!(reconstructed, bytes[..96 * 4]);
    }
    let c = production(0, 8);
    assert!(plan_tensor_bytes(name, &[128], 2, MixerKind::Kda, &c).is_err());
    assert!(plan_tensor_bytes(name, &[129], 4, MixerKind::Kda, &c).is_err());
    let mut wrong = c.clone();
    wrong.linear_num_key_heads = 8;
    assert!(plan_tensor_bytes(name, &[128], 4, MixerKind::Kda, &wrong).is_err());
}
