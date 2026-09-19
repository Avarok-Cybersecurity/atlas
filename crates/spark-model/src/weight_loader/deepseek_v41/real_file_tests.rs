// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

use super::*;
use crate::layers::attn_v41::AttnV41LayerState;
use crate::layers::attn_v41::{AttnV41, AttnV41Cfg, AttnV41LayerWeights, LayerRole, SharedV41};
use crate::layers::deepseek_v41_ref::attn::{AttnWeights, freqs_cis};
use crate::layers::deepseek_v41_ref::compress::{
    CompAttnCfg, IndexerCfg, LayerAttnState, SharedRuntime, attention_any,
};
use crate::layers::deepseek_v41_ref::hc::{hc_mixes, hc_post, hc_pre, rms_norm};
use crate::layers::deepseek_v41_ref::moe::{MoeCfg, MoeWeights, gate as ref_gate, moe as ref_moe};
use crate::layers::moe_v41::{MoeV41, MoeV41Cfg, MoeV41LayerWeights};
use crate::layers::ops;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::GgufLoader;
use spark_runtime::weights::WeightLoader;
use spark_runtime::weights::dequant_cpu::{GgmlType, dequant_to_f32};
use spark_runtime::weights::expert_stream::{
    ExpertLru, ExpertSliceMap, ExpertSource, PinnedArena, ShardFiles,
};
use std::sync::Arc;

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
    let set =
        avarok_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4").expect("kernel target");
    let gpu =
        spark_runtime::cuda_backend::AvarokCudaBackend::new(0, &set.modules).expect("CUDA backend");
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
        .forward(g, &mw_d, &mut lru, &slices, f_in_d, m, 4, None, stream)
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
