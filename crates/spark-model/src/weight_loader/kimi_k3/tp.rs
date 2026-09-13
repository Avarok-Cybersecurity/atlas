// SPDX-License-Identifier: AGPL-3.0-only

//! K3 Megatron TP plan + `slice_for_rank` apply.
//!
//! Head counts on `config` are already per-rank (`serve_phases::topology`
//! divides them). Full sizes reconstruct as `local * tp_size`.
//!
//! Sharded: KDA q/k/v/o/g + conv / A_log / dt_bias / b_proj / f_b; MLA
//! q_b / kv_b / g / o; dense gate/up/down; expert w1/w3 col, w2 row.
//! Replicated: embed, RMSNorm, router, f_a, o_norm, q_a, kv_a, shared.

use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;
use atlas_core::kimi_k3::{MixerKind, MlpKind};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::kimi_k3::bound::WeightMeta;
use crate::tp_shard::{TpShardKind, slice_for_rank};
use crate::weight_map::DenseWeight;

use super::bf16::f32_le_to_bf16_bytes;

/// `(kind, full_out, full_in)` for one checkpoint key. `in == 1` is a 1D
/// column-parallel vector (`A_log`, `dt_bias`).
pub fn tensor_plan(
    name: &str,
    mixer: MixerKind,
    _mlp: MlpKind,
    config: &ModelConfig,
) -> (TpShardKind, usize, usize) {
    let tp = config.tp_world_size.max(1);
    let h = config.hidden_size;
    let kda_heads = config.linear_num_key_heads * tp;
    let kda_d = config.linear_key_head_dim;
    let kda_q = kda_heads * kda_d;
    let conv_k = config.linear_conv_kernel_dim.max(1);
    let mla_heads = config.num_attention_heads * tp;
    let qk = config.qk_nope_head_dim + config.qk_rope_head_dim;
    let dv = mla_heads * config.v_head_dim;
    let kv_b = mla_heads * (config.qk_nope_head_dim + config.v_head_dim);
    let inter = config.intermediate_size;
    let eh = config.moe_intermediate_size;
    let lat = config.moe_latent_size;

    if name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
    {
        return (TpShardKind::ColumnParallel, kda_q, h);
    }
    if name.ends_with(".self_attn.q_conv1d.weight")
        || name.ends_with(".self_attn.k_conv1d.weight")
        || name.ends_with(".self_attn.v_conv1d.weight")
    {
        return (TpShardKind::ColumnParallel, kda_q, conv_k);
    }
    if name.ends_with(".self_attn.g_proj.weight") {
        let n = match mixer {
            MixerKind::Kda => kda_q,
            MixerKind::Mla => dv,
        };
        return (TpShardKind::ColumnParallel, n, h);
    }
    if name.ends_with(".self_attn.o_proj.weight") {
        let inn = match mixer {
            MixerKind::Kda => kda_q,
            MixerKind::Mla => dv,
        };
        return (TpShardKind::RowParallel, h, inn);
    }
    if name.ends_with(".self_attn.b_proj.weight") {
        return (TpShardKind::ColumnParallel, kda_heads, h);
    }
    if name.ends_with(".self_attn.A_log") {
        return (TpShardKind::ColumnParallel, kda_heads, 1);
    }
    if name.ends_with(".self_attn.dt_bias") {
        return (TpShardKind::ColumnParallel, kda_q, 1);
    }
    if name.ends_with(".self_attn.f_b_proj.weight") {
        return (TpShardKind::ColumnParallel, kda_q, kda_d);
    }
    if name.ends_with(".self_attn.q_b_proj.weight") {
        return (
            TpShardKind::ColumnParallel,
            mla_heads * qk,
            config.q_lora_rank,
        );
    }
    if name.ends_with(".self_attn.kv_b_proj.weight") {
        return (TpShardKind::ColumnParallel, kv_b, config.kv_lora_rank);
    }
    if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
        return (TpShardKind::ColumnParallel, inter, h);
    }
    if name.ends_with(".mlp.down_proj.weight") {
        return (TpShardKind::RowParallel, h, inter);
    }
    if name.contains(".block_sparse_moe.experts.") {
        if name.ends_with(".w1.weight") || name.ends_with(".w3.weight") {
            return (TpShardKind::ColumnParallel, eh, lat);
        }
        if name.ends_with(".w2.weight") {
            return (TpShardKind::RowParallel, lat, eh);
        }
    }
    (TpShardKind::Replicated, 1, 1)
}

pub fn load_sharded(
    store: &WeightStore,
    name: &str,
    mixer: MixerKind,
    mlp: MlpKind,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<(DenseWeight, WeightMeta)> {
    let t = store.get(name)?;
    let (kind, full_out, full_in) = tensor_plan(name, mixer, mlp, config);
    if config.tp_world_size.max(1) <= 1 || kind == TpShardKind::Replicated {
        return Ok((
            DenseWeight { weight: t.ptr },
            WeightMeta {
                name: name.to_string(),
                dtype: t.dtype,
                numel: t.num_elements(),
            },
        ));
    }
    let want = full_out.saturating_mul(full_in);
    ensure!(
        t.num_elements() == want,
        "{name}: {} elems, TP plan [{full_out}, {full_in}] wants {want}",
        t.num_elements()
    );
    let (bf16_ptr, owned) = as_bf16(gpu, t.ptr, t.dtype, want)?;
    let (ptr, local_out, local_in) =
        slice_for_rank(bf16_ptr, full_out, full_in, kind, config, gpu)?;
    if owned && ptr != bf16_ptr {
        gpu.free(bf16_ptr)?;
    }
    Ok((
        DenseWeight { weight: ptr },
        WeightMeta {
            name: name.to_string(),
            dtype: WeightDtype::BF16,
            numel: local_out * local_in,
        },
    ))
}

fn as_bf16(
    gpu: &dyn GpuBackend,
    ptr: spark_runtime::gpu::DevicePtr,
    dtype: WeightDtype,
    numel: usize,
) -> Result<(spark_runtime::gpu::DevicePtr, bool)> {
    match dtype {
        WeightDtype::BF16 => Ok((ptr, false)),
        WeightDtype::FP32 => {
            let mut raw = vec![0u8; numel * 4];
            gpu.copy_d2h(ptr, &mut raw)?;
            let bf = f32_le_to_bf16_bytes(&raw);
            let dst = gpu.alloc(bf.len())?;
            gpu.copy_h2d(&bf, dst)?;
            Ok((dst, true))
        }
        other => bail!("K3 TP shard: unsupported dtype {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::KimiK3WeightLoader;
    use super::super::bf16::{layer_keys, text_key};
    use super::*;
    use crate::weight_loader::ModelWeightLoader;
    use atlas_core::config::parse_config;
    use atlas_core::kimi_k3::K3Graph;
    use atlas_core::kimi_k3::ops::matvec;
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::WeightTensor;
    use std::collections::HashMap;

    const TWIN: &str = include_str!("../../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json");

    fn twin() -> ModelConfig {
        parse_config(TWIN).expect("0.40B twin")
    }

    /// Same head divide as `serve_phases::topology`.
    fn divide_heads_for_tp(config: &mut ModelConfig, rank: usize, size: usize) {
        config.tp_rank = rank;
        config.tp_world_size = size;
        if size > 1 {
            config.num_attention_heads /= size;
            config.num_key_value_heads /= size;
            config.linear_num_key_heads /= size;
            config.linear_num_value_heads /= size;
        }
    }

    #[test]
    fn kda_q_is_column_o_is_row() {
        let mut c = twin();
        divide_heads_for_tp(&mut c, 0, 2);
        let q = tensor_plan(
            "language_model.model.layers.0.self_attn.q_proj.weight",
            MixerKind::Kda,
            MlpKind::Dense,
            &c,
        );
        let o = tensor_plan(
            "language_model.model.layers.0.self_attn.o_proj.weight",
            MixerKind::Kda,
            MlpKind::Dense,
            &c,
        );
        assert_eq!(q, (TpShardKind::ColumnParallel, 8 * 32, 1024));
        assert_eq!(o, (TpShardKind::RowParallel, 1024, 8 * 32));
        let g = tensor_plan(
            "language_model.model.layers.0.self_attn.g_proj.weight",
            MixerKind::Kda,
            MlpKind::Dense,
            &c,
        );
        assert_eq!(g.0, TpShardKind::ColumnParallel);
        let w1 = tensor_plan(
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight",
            MixerKind::Kda,
            MlpKind::LatentMoe,
            &c,
        );
        let w2 = tensor_plan(
            "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight",
            MixerKind::Kda,
            MlpKind::LatentMoe,
            &c,
        );
        assert_eq!(w1, (TpShardKind::ColumnParallel, 256, 512));
        assert_eq!(w2, (TpShardKind::RowParallel, 512, 256));
        let embed = tensor_plan(
            "language_model.model.embed_tokens.weight",
            MixerKind::Kda,
            MlpKind::Dense,
            &c,
        );
        assert_eq!(embed.0, TpShardKind::Replicated);
    }

    #[test]
    fn slice_for_rank_column_splits_ramp() {
        let mut c = twin();
        divide_heads_for_tp(&mut c, 1, 2);
        let gpu = MockGpuBackend::new();
        let n = 8u16;
        let bytes: Vec<u8> = (0..n).flat_map(|v| v.to_le_bytes()).collect();
        let src = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(&bytes, src).unwrap();
        let (dst, local_out, local_in) =
            slice_for_rank(src, 4, 2, TpShardKind::ColumnParallel, &c, &gpu).unwrap();
        assert_eq!((local_out, local_in), (2, 2));
        let mut got = vec![0u8; 8];
        gpu.copy_d2h(dst, &mut got).unwrap();
        assert_eq!(got, bytes[8..].to_vec());
    }

    fn put_bf16(
        gpu: &MockGpuBackend,
        map: &mut HashMap<String, WeightTensor>,
        name: String,
        n: usize,
        fill: u16,
    ) {
        // Mix high bits: a plain wrapping_add(i) repeats every 65536 elems,
        // which is exactly one K3 TP=2 q_proj shard (128*1024), so rank-0
        // and rank-1 copies compared equal while the slice was correct.
        let bytes: Vec<u8> = (0..n)
            .flat_map(|i| {
                let v = fill
                    .wrapping_add(i as u16)
                    .wrapping_add(((i >> 16) as u16).wrapping_mul(0x9E37));
                v.to_le_bytes()
            })
            .collect();
        let ptr = gpu.alloc(bytes.len().max(2)).unwrap();
        if !bytes.is_empty() {
            gpu.copy_h2d(&bytes, ptr).unwrap();
        }
        map.insert(
            name,
            WeightTensor {
                ptr,
                shape: vec![n.max(1)],
                dtype: WeightDtype::BF16,
            },
        );
    }

    fn store_for_tp(gpu: &MockGpuBackend, config: &ModelConfig) -> WeightStore {
        let graph = K3Graph::from_config(config);
        let mut map = HashMap::new();
        put_bf16(
            gpu,
            &mut map,
            text_key(config, "model.output_attn_res_proj.weight"),
            2,
            0,
        );
        put_bf16(
            gpu,
            &mut map,
            text_key(config, "model.output_attn_res_norm.weight"),
            2,
            0,
        );
        for spec in &graph.layers {
            for k in layer_keys(config, spec.index, spec.mixer, spec.mlp, config.num_experts) {
                let (kind, o, i) = tensor_plan(&k, spec.mixer, spec.mlp, config);
                let n = if kind == TpShardKind::Replicated {
                    2
                } else {
                    o * i
                };
                put_bf16(gpu, &mut map, k, n, 1);
            }
        }
        WeightStore::from_map(map)
    }

    #[test]
    fn load_tp2_shards_q_proj_half() {
        let mut c = twin();
        divide_heads_for_tp(&mut c, 0, 2);
        let gpu = MockGpuBackend::new();
        let store = store_for_tp(&gpu, &c);
        let layers = KimiK3WeightLoader
            .load_layers(&store, &c, &gpu, &[])
            .expect("tp=2 bind");
        assert_eq!(layers.len(), 8);
        // Layer 0 is KDA: q_proj is the first sharded self_attn after norms.
        // Inspect via a second rank-1 load: local numel must be half of full.
        let q_name = text_key(&c, "model.layers.0.self_attn.q_proj.weight");
        let (kind, fo, fi) = tensor_plan(&q_name, MixerKind::Kda, MlpKind::Dense, &c);
        assert_eq!(kind, TpShardKind::ColumnParallel);
        let (dw0, meta) =
            load_sharded(&store, &q_name, MixerKind::Kda, MlpKind::Dense, &c, &gpu).unwrap();
        assert_eq!(meta.numel, (fo / 2) * fi);
        assert_eq!(meta.dtype, WeightDtype::BF16);
        let mut c1 = twin();
        divide_heads_for_tp(&mut c1, 1, 2);
        let (dw1, m1) =
            load_sharded(&store, &q_name, MixerKind::Kda, MlpKind::Dense, &c1, &gpu).unwrap();
        assert_eq!(m1.numel, meta.numel);
        let mut a = vec![0u8; meta.numel * 2];
        let mut b = vec![0u8; m1.numel * 2];
        gpu.copy_d2h(dw0.weight, &mut a).unwrap();
        gpu.copy_d2h(dw1.weight, &mut b).unwrap();
        assert_ne!(a, b, "rank-0 and rank-1 q_proj shards must differ");
        let _ = layers;
    }

    #[test]
    fn drop_rank1_o_proj_shard_changes_hidden() {
        // Row-parallel o_proj: each rank keeps half the input cols; sum == full.
        let h = 4usize;
        let q = 8usize;
        let w: Vec<f32> = (0..h * q).map(|i| i as f32 * 0.25).collect();
        let x: Vec<f32> = (0..q).map(|i| (i as f32) * 0.5).collect();
        let full = matvec(&w, &x, h, q);
        let local = q / 2;
        let mut w0 = Vec::with_capacity(h * local);
        let mut w1 = Vec::with_capacity(h * local);
        for r in 0..h {
            let row = &w[r * q..(r + 1) * q];
            w0.extend_from_slice(&row[..local]);
            w1.extend_from_slice(&row[local..]);
        }
        let y0 = matvec(&w0, &x[..local], h, local);
        let y1 = matvec(&w1, &x[local..], h, local);
        let summed: Vec<f32> = y0.iter().zip(&y1).map(|(a, b)| a + b).collect();
        assert_eq!(summed, full, "TP=2 o_proj shards must allreduce to TP=1");
        assert_ne!(
            y0, full,
            "RST known-bad: drop rank-1 shard must change hidden"
        );
    }
}
