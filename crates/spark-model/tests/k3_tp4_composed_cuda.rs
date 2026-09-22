// SPDX-License-Identifier: AGPL-3.0-only
//! Official TP4 rank-local dimensions through the production host graph and
//! CUDA callbacks: KDA+dense then MLA+selected packed experts+shared MLP.
//! Synthetic weights, one local rank: no checkpoint loader, NCCL, all-layer,
//! tokenizer, or model-quality claim. Independent kernel oracles live elsewhere.
#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use avarok_core::config::parse_config;
use avarok_core::kimi_k3::cpu_weights::{
    DenseMlp, KdaWeights, MixerW, MlaWeights, MlpW, MoeWeights,
};
use avarok_core::kimi_k3::{
    Ablation, AttnResStream, K3CpuLayer, K3LayerCtx, K3LayerSpec, KdaState, LayerCache, MixerKind,
    MlaKv, MlpKind, forward_one_layer_with_cores, kda_from, mla_from, moe_from,
};
use half::bf16;
use spark_model::kimi_k3::{
    dense_cuda::launch_dense_matrices,
    kda_cuda::{K3KdaDecodeKernels, KdaDeviceState, launch_k3_kda_decode_token_on_device},
    mla_cuda::{K3MlaDecodeKernels, launch_k3_mla_decode_token},
    moe_cuda::{K3MoeGemmKernels, launch_k3_latent_moe_experts},
};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::{cell::RefCell, collections::BTreeMap, time::Instant};

struct Memory<'a> {
    gpu: &'a dyn GpuBackend,
    ptrs: Vec<DevicePtr>,
    bytes: usize,
}
impl Memory<'_> {
    fn upload(&mut self, data: &[u8]) -> Result<DevicePtr> {
        let next = self
            .bytes
            .checked_add(data.len())
            .context("fixture size overflow")?;
        ensure!(
            next <= 1024 * 1024 * 1024,
            "fixture exceeds 1 GiB persistent GPU budget"
        );
        let p = self.gpu.alloc(data.len())?;
        self.ptrs.push(p);
        self.bytes = next;
        self.gpu.copy_h2d(data, p)?;
        Ok(p)
    }
    fn dense(&mut self, w: &DenseMlp) -> Result<[DevicePtr; 3]> {
        let mut out = [DevicePtr::NULL; 3];
        for (dst, values) in out.iter_mut().zip([&w.gate, &w.up, &w.down]) {
            let raw: Vec<u8> = values
                .iter()
                .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
                .collect();
            *dst = self.upload(&raw)?;
        }
        Ok(out)
    }
}
impl Drop for Memory<'_> {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..).rev() {
            let _ = self.gpu.free(p);
        }
    }
}
fn matrix(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
    let mut a = vec![0.; rows * cols];
    for r in 0..rows {
        a[r * cols + (r * 7 + seed) % cols] = 0.125;
        a[r * cols + (r * 11 + seed + 1) % cols] = -0.0625;
    }
    a
}
fn dense(h: usize, i: usize) -> DenseMlp {
    DenseMlp {
        gate: matrix(i, h, 1),
        up: matrix(i, h, 3),
        down: matrix(h, i, 5),
    }
}
fn layer(index: usize, mixer: MixerW, mlp: MlpW, h: usize) -> K3CpuLayer {
    K3CpuLayer {
        spec: K3LayerSpec {
            index,
            mixer: if index == 0 {
                MixerKind::Kda
            } else {
                MixerKind::Mla
            },
            mlp: if index == 0 {
                MlpKind::Dense
            } else {
                MlpKind::LatentMoe
            },
        },
        input_norm: vec![1.; h],
        post_norm: vec![1.; h],
        attn_res_proj: vec![0.01; h],
        attn_res_norm: vec![1.; h],
        mlp_res_proj: vec![0.01; h],
        mlp_res_norm: vec![1.; h],
        mixer,
        mlp,
    }
}
fn timed<T>(
    stats: &RefCell<BTreeMap<&'static str, f64>>,
    label: &'static str,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let start = Instant::now();
    let result = f();
    *stats.borrow_mut().entry(label).or_default() += start.elapsed().as_secs_f64();
    result
}

#[test]
#[ignore = "requires explicit idle CUDA GPU; full TP4 local widths, approximately 2.3 GiB host fixture"]
fn k3_tp4_rank_local_composed_graph_reset_and_history() -> Result<()> {
    let ordinal = std::env::var("K3_ORACLE_GPU_ORDINAL")?.parse()?;
    let target = avarok_kernels::ptx_for_exact_target("kimi-k3", "mxfp4").context("K3 target")?;
    let gpu = AvarokCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    let mut c = parse_config(include_str!(
        "../../../docs/k3/fixtures/moonshotai-Kimi-K3-config.json"
    ))?;
    c.tp_world_size = 4;
    c.tp_rank = 0;
    c.num_attention_heads /= 4;
    c.num_key_value_heads /= 4;
    c.linear_num_key_heads /= 4;
    c.linear_num_value_heads /= 4;
    let kd = kda_from(&c);
    let ml = mla_from(&c);
    let mut mo = moe_from(&c);
    mo.expert_hidden /= 4;
    let h = c.hidden_size;
    let di = c.intermediate_size / 4;
    let si = c.shared_expert_intermediate_size;
    ensure!(
        (
            h,
            kd.heads,
            kd.head_dim,
            ml.qk_head_dim(),
            ml.v_head_dim,
            di,
            mo.latent,
            mo.expert_hidden,
            mo.top_k,
            mo.n_routed,
            si
        ) == (7168, 24, 128, 192, 128, 8448, 3584, 768, 16, 896, 6144),
        "official TP4 fixture geometry changed"
    );
    let q = kd.qkv_dim();
    let mq = ml.heads * ml.qk_head_dim();
    let mv = ml.heads * ml.v_head_dim;
    // Count full matrix storage before allocating. Sparse deterministic values
    // still occupy and exercise full-width production matrices.
    let matrix_elements = 5 * h * q
        + kd.head_dim * h
        + q * kd.head_dim
        + kd.heads * h
        + ml.q_lora_rank * h
        + mq * ml.q_lora_rank
        + (ml.kv_lora_rank + ml.qk_rope_head_dim) * h
        + ml.heads * (ml.qk_nope_head_dim + ml.v_head_dim) * ml.kv_lora_rank
        + 2 * h * mv
        + 3 * h * (di + si)
        + 2 * h * mo.latent
        + mo.n_routed * h;
    ensure!(
        matrix_elements * 4 < 3 * 1024 * 1024 * 1024,
        "fixture exceeds 3 GiB host matrix budget"
    );
    let build = Instant::now();
    let kw = KdaWeights {
        q_proj: matrix(q, h, 1),
        k_proj: matrix(q, h, 2),
        v_proj: matrix(q, h, 3),
        conv: (0..kd.conv_elems())
            .map(|i| [0.125, 0.25, 0.5, 1.0][i % 4])
            .collect(),
        f_a: matrix(kd.head_dim, h, 4),
        f_b: matrix(q, kd.head_dim, 5),
        dt_bias: vec![0.; q],
        a_log: vec![-1.; kd.heads],
        b_proj: matrix(kd.heads, h, 6),
        g_proj: matrix(q, h, 7),
        o_norm: vec![1.; kd.head_dim],
        o_proj: matrix(h, q, 8),
    };
    let mw = MlaWeights {
        q_a: matrix(ml.q_lora_rank, h, 9),
        q_a_ln: vec![1.; ml.q_lora_rank],
        q_b: matrix(mq, ml.q_lora_rank, 10),
        kv_a: matrix(ml.kv_lora_rank + ml.qk_rope_head_dim, h, 11),
        kv_a_ln: vec![1.; ml.kv_lora_rank],
        kv_b: matrix(
            ml.heads * (ml.qk_nope_head_dim + ml.v_head_dim),
            ml.kv_lora_rank,
            12,
        ),
        g_proj: matrix(mv, h, 13),
        o_proj: matrix(h, mv, 14),
    };
    let dw = dense(h, di);
    let sw = dense(h, si);
    let mut memory = Memory {
        gpu: &gpu,
        ptrs: Vec::new(),
        bytes: 0,
    };
    let dp = memory.dense(&dw)?;
    let sp = memory.dense(&sw)?;
    let moe = MoeWeights {
        down: matrix(mo.latent, h, 15),
        up: matrix(h, mo.latent, 16),
        norm: vec![1.; mo.latent],
        router: matrix(mo.n_routed, h, 17),
        bias: (0..mo.n_routed)
            .map(|e| {
                if e < mo.top_k {
                    2. + (mo.top_k - e) as f32 * 0.01
                } else {
                    -2.
                }
            })
            .collect(),
        experts: vec![(Vec::new(), Vec::new(), Vec::new()); mo.n_routed],
        shared: Some(sw),
    };
    // Keep the real 896-way router but provision only its deterministically
    // selected 16 experts. This is not a complete checkpoint admission test.
    let mut packed = Vec::new();
    for e in 0..mo.top_k {
        for (role, n, k) in [
            ("w1", mo.expert_hidden, mo.latent),
            ("w3", mo.expert_hidden, mo.latent),
            ("w2", mo.latent, mo.expert_hidden),
        ] {
            let bytes: Vec<u8> = (0..n * k / 2)
                .map(|i| if (i + e) % 2 == 0 { 0x21 } else { 0xa9 })
                .collect();
            let weight = memory.upload(&bytes)?;
            let scale = memory.upload(&vec![120; n * k / 32])?;
            packed.push((
                format!("model.layers.1.block_sparse_moe.experts.{e}.{role}"),
                QuantizedWeight {
                    weight,
                    weight_scale: scale,
                    weight_scale_2: 1.,
                    input_scale: DevicePtr::NULL,
                    weight_scale_2_vec: DevicePtr::NULL,
                },
            ));
        }
    }
    let layers = [
        layer(0, MixerW::Kda(kw), MlpW::Dense(dw), h),
        layer(1, MixerW::Mla(mw), MlpW::Moe(moe), h),
    ];
    let kk = K3KdaDecodeKernels::resolve(&gpu)?;
    let mk = K3MlaDecodeKernels::resolve(&gpu)?;
    let ek = K3MoeGemmKernels::resolve(&gpu)?;
    let stats = RefCell::new(BTreeMap::new());
    let dense_core =
        |_w: &DenseMlp, x: &[f32], hidden: usize, inter: usize, beta: f32, linear: f32| {
            let p = if inter == di { dp } else { sp };
            ensure!(inter == di || inter == si, "unexpected dense width");
            timed(&stats, if inter == di { "dense" } else { "shared" }, || {
                launch_dense_matrices(
                    &gpu, x, p[0], p[1], p[2], 1, 1, hidden, inter, beta, linear, stream,
                )
            })
        };
    let ctx = K3LayerCtx {
        kda: &kd,
        mla: &ml,
        moe: &mo,
        situ_beta: 4.,
        situ_linear_beta: 25.,
        hidden: h,
        dense_intermediate: di,
        eps: c.rms_norm_eps as f32,
        rope_theta: c.rope_theta as f32,
        reduce_hidden: None,
        dense_mlp: Some(&dense_core),
        tp_rank: 0,
        tp_world: 1,
    };
    let build_seconds = build.elapsed().as_secs_f64();
    let run = |inputs: &[usize]| -> Result<Vec<Vec<f32>>> {
        let initial = KdaState::new(&kd);
        let device = KdaDeviceState::alloc_and_upload(&gpu, &initial)?;
        let mut caches = [LayerCache::Kda(initial), LayerCache::Mla(MlaKv::default())];
        let result = (|| {
            let mut outputs = Vec::new();
            for (pos, &salt) in inputs.iter().enumerate() {
                let mut residual = AttnResStream::new(h, c.attn_res_block_size);
                residual.partial = (0..h)
                    .map(|i| (((i * 19 + salt * 13) % 127) as f32 - 63.) / 64.)
                    .collect();
                for (layer, cache) in layers.iter().zip(&mut caches) {
                    forward_one_layer_with_cores(
                        &ctx,
                        layer,
                        pos,
                        cache,
                        &mut residual,
                        Ablation::default(),
                        |x, w, g, b, cfg, _| {
                            timed(&stats, "kda", || {
                                launch_k3_kda_decode_token_on_device(
                                    &gpu, &kk, x, w, g, b, cfg, &device, stream,
                                )
                            })
                        },
                        |q, k, v, g, kv, cfg, p, theta| {
                            timed(&stats, "mla_host_kv", || {
                                launch_k3_mla_decode_token(
                                    &gpu, &mk, q, k, v, g, kv, cfg, p, theta, stream,
                                )
                            })
                        },
                        |_, x, ids, w, cfg| {
                            ensure!(
                                ids.len() == 16 && ids.iter().all(|&id| id < 16),
                                "selected experts outside provisioned set"
                            );
                            timed(&stats, "packed_experts", || {
                                launch_k3_latent_moe_experts(
                                    &gpu, &ek, &packed, x, ids, w, cfg, stream,
                                )
                            })
                        },
                    )?;
                }
                ensure!(
                    residual.partial.len() == h && residual.partial.iter().all(|x| x.is_finite()),
                    "invalid composed output"
                );
                ensure!(
                    residual.partial.iter().any(|x| x.abs() > 0.01),
                    "vacuous zero output"
                );
                outputs.push(residual.partial);
            }
            if let LayerCache::Mla(kv) = &caches[1] {
                ensure!(kv.seq_len == inputs.len(), "MLA did not advance");
            }
            Ok(outputs)
        })();
        device.free(&gpu)?;
        result
    };
    let started = Instant::now();
    let first = run(&[1, 2])?;
    let replay = run(&[1, 2])?;
    ensure!(
        first == replay,
        "fresh reset/replay changed composed outputs"
    );
    let cleared = run(&[2])?;
    let history_delta = first[1]
        .iter()
        .zip(&cleared[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    ensure!(
        history_delta > 1e-5,
        "fixture insensitive to lost causal state: {history_delta}"
    );
    ensure!(
        ["dense", "shared", "kda", "mla_host_kv", "packed_experts"]
            .iter()
            .all(|key| stats.borrow().get(key).is_some_and(|t| *t > 0.0)),
        "a required CUDA callback did not execute"
    );
    let bytes = memory.bytes;
    println!(
        "PASS TP4 rank-local composed graph: five tokens, two layers, reset exact, history_delta={history_delta}; matrix_host_bytes={}, persistent_gpu_fixture_bytes={bytes}, build_seconds={build_seconds:.3}, graph_seconds={:.3}, callback_seconds={:?}",
        matrix_elements * 4,
        started.elapsed().as_secs_f64(),
        stats.borrow()
    );
    println!(
        "Scope: production host graph + CUDA callbacks, sparse full-size synthetic matrices, selected16/896 experts; no loader/NCCL/full-model proof. Bytes exclude CUDA context/allocator overhead and transient wrapper allocations."
    );
    Ok(())
}
