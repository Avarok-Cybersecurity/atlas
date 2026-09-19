// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! The GPU attention against the CPU reference model, layer by layer, across
//! prefill and both decode steps: the reference forward (itself 39/39 against
//! DeepSeek's golden) supplies every layer's true `attn_in`, and the GPU layer
//! must reproduce its `q`, its attended rows, its index selection (exactly),
//! its sparse-attention output and its output projection within bf16, while
//! carrying the window rings, the compressor state, the two caches and the
//! shared slots across all three regimes exactly as the reference does.

use super::*;
use crate::layers::deepseek_v41_ref::engram::EngramTables;
use crate::layers::deepseek_v41_ref::model::{
    ModelCfg, ModelState, ModelWeights, StepTrace, forward,
};
use crate::layers::deepseek_v41_ref::{Golden, fixed_int};

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    (b.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

fn f32_of_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[cfg(feature = "cuda")]
fn backend() -> spark_runtime::cuda_backend::AvarokCudaBackend {
    let set = avarok_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
        .expect("deepseek-v4-flash/nvfp4 not in this build");
    spark_runtime::cuda_backend::AvarokCudaBackend::new(0, &set.modules).expect("CUDA backend")
}

/// Run the reference for the three regimes, keeping every trace.
fn reference_traces(g: &Golden, c: &ModelCfg, t: &EngramTables) -> Vec<(String, usize, StepTrace)> {
    let w = ModelWeights::from_golden(g, c, t);
    let mut st = ModelState::new(c, t);
    let vocab = g.fixture_u64("vocab_size");
    let prefill = g.fixture_u64("prefill_len") as usize;
    let ids: Vec<i64> = (0..(prefill + 2) as u64)
        .map(|i| fixed_int("input_ids", i, vocab) as i64)
        .collect();
    let regimes = g.regimes();
    vec![
        (
            regimes[0].clone(),
            0,
            forward(&ids[..prefill], 0, c, &w, &mut st, t),
        ),
        (
            regimes[1].clone(),
            prefill,
            forward(&ids[prefill..prefill + 1], prefill, c, &w, &mut st, t),
        ),
        (
            regimes[2].clone(),
            prefill + 1,
            forward(
                &ids[prefill + 1..prefill + 2],
                prefill + 1,
                c,
                &w,
                &mut st,
                t,
            ),
        ),
    ]
}

fn cfg_of(c: &ModelCfg) -> AttnV41Cfg {
    AttnV41Cfg {
        dim: c.dim,
        n_heads: c.n_heads,
        head_dim: c.head_dim,
        rope_dim: c.rope_dim,
        q_rank: c.q_rank,
        o_rank: c.o_rank,
        groups: c.groups,
        window: c.window,
        eps: c.eps,
        index_heads: c.index_heads,
        index_hd: c.index_hd,
        index_topk: c.index_topk,
        cand_topk_blocks: c.cand_topk_blocks,
        cand_block: c.cand_block,
        max_seq: c.max_seq,
        max_tokens: c.max_seq,
        rope_theta: c.rope_theta,
        compress_rope_theta: c.compress_rope_theta,
        rope_factor: c.rope_factor,
        orig_seq: c.orig_seq,
        beta_fast: c.beta_fast,
        beta_slow: c.beta_slow,
    }
}

fn role_of(c: &ModelCfg, l: usize) -> LayerRole {
    LayerRole {
        ratio: c.ratios[l],
        is_kv_source: c.kv_sources.contains(&l),
        is_index_source: c.index_sources.contains(&l),
        is_candidate_source: c.cand_src == l as i64,
        uses_candidates: c.cand_src >= 0 && (c.cand_src as usize) < l,
    }
}

#[cfg(feature = "cuda")]
fn up_bf16(g: &dyn spark_runtime::gpu::GpuBackend, v: &[f32]) -> DevicePtr {
    let bytes: Vec<u8> = v.iter().flat_map(|&x| bf16_bits(x).to_le_bytes()).collect();
    let p = g.alloc(bytes.len().max(4)).unwrap();
    g.copy_h2d(&bytes, p).unwrap();
    p
}

#[cfg(feature = "cuda")]
fn up_f32(g: &dyn spark_runtime::gpu::GpuBackend, v: &[f32]) -> DevicePtr {
    upload_f32(g, v).unwrap()
}

#[cfg(feature = "cuda")]
fn down_bf16(g: &dyn spark_runtime::gpu::GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(2)
        .map(|c| f32_of_bf16(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// bf16 tolerance on the reference's own values: half an ulp at the row's
/// magnitude plus a floor for the f32 reduction-order noise.
fn assert_within_bf16(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs {}",
        got.len(),
        want.len()
    );
    let max = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let tol = max * 2f32.powi(-7) + 1e-4;
    let mut worst = (0f32, 0usize);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    assert!(
        worst.0 <= tol,
        "{what}: max abs diff {} > tol {tol} at {} (got {}, want {}; max |want| {max})",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn gpu_attention_matches_the_reference_on_every_layer_and_regime() {
    use spark_runtime::gpu::GpuBackend;
    let golden = Golden::load();
    let c = ModelCfg::from_golden(&golden);
    let t = EngramTables::from_golden(&golden);
    let traces = reference_traces(&golden, &c, &t);
    let rw = ModelWeights::from_golden(&golden, &c, &t);

    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let cfg = cfg_of(&c);
    let mut attn = AttnV41::new(g, cfg.clone()).unwrap();
    let mut weights = Vec::new();
    let mut states = Vec::new();
    for l in 0..c.n_layers {
        let lw = &rw.layers[l];
        let role = role_of(&c, l);
        let comp = lw.comp_wkv.as_ref().map(|wkv| CompressorWeightsGpu {
            kv: if role.ratio > 1 {
                up_f32(g, wkv)
            } else {
                up_bf16(g, wkv)
            },
            gate: lw.comp_wgate.as_ref().map(|w| up_f32(g, w)),
            norm: up_f32(g, lw.comp_norm.as_ref().unwrap()),
        });
        let idx = lw.idx_wq_b.as_ref().map(|wq_b| IndexerWeightsGpu {
            wq_b: up_bf16(g, wq_b),
            weights_proj: up_bf16(g, lw.idx_weights_proj.as_ref().unwrap()),
            wk: lw.idx_wk.as_ref().map(|w| up_bf16(g, w)),
            k_norm: lw.idx_k_norm.as_ref().map(|w| up_f32(g, w)),
        });
        weights.push(AttnV41LayerWeights {
            role,
            sink: up_f32(g, &lw.sink),
            wq_a: AttnMat::Bf16(up_bf16(g, &lw.wq_a)),
            q_norm: up_f32(g, &lw.q_norm),
            wq_b: AttnMat::Bf16(up_bf16(g, &lw.wq_b)),
            wkv: AttnMat::Bf16(up_bf16(g, &lw.wkv)),
            kv_norm: up_f32(g, &lw.kv_norm),
            wo_a: AttnMat::Bf16(up_bf16(g, &lw.wo_a)),
            wo_b: AttnMat::Bf16(up_bf16(g, &lw.wo_b)),
            comp,
            idx,
        });
        states.push(AttnV41LayerState::new(g, &cfg, role).unwrap());
    }
    let mut shared = SharedV41::default();
    let (nh, hd) = (c.n_heads, c.head_dim);
    let mut captured_layers = 0usize;
    for (regime, start_pos, trace) in &traces {
        let m = trace.tokens;
        for l in 0..c.n_layers {
            let lt = &trace.layers[l];
            let what = format!("{regime}.L{l}");
            let x = up_bf16(g, &lt.attn_in);
            // the CUDA-graph decode step (deepseek_v41_layer's segment A holds
            // this body): the position and the -1-padded selection go to the
            // device, the body is captured once and replayed, and its output
            // must be the eager forward's below bit for bit. It runs first
            // because both write the same kv row into the same ring slot.
            let graph_out = if m == 1 && attn.decode_capturable(&weights[l]) {
                let rows_b = attn
                    .decode_prep(&weights[l], &shared, g, *start_pos, stream)
                    .unwrap();
                g.begin_capture(stream).unwrap();
                let out = attn
                    .decode_body(g, &weights[l], &states[l], x, rows_b, stream)
                    .unwrap();
                let graph = g.end_capture(stream).unwrap();
                g.launch_graph(graph, stream).unwrap();
                g.synchronize(stream).unwrap();
                let bytes = {
                    let mut b = vec![0u8; c.dim * 2];
                    g.copy_d2h(out, &mut b).unwrap();
                    b
                };
                g.destroy_graph(graph).unwrap();
                captured_layers += 1;
                Some(bytes)
            } else {
                None
            };
            let run = attn
                .forward(
                    g,
                    &weights[l],
                    &mut states[l],
                    &mut shared,
                    x,
                    m,
                    *start_pos,
                    stream,
                )
                .unwrap();
            g.free(x).unwrap();
            if let Some(graph_bytes) = graph_out {
                attn.invalidate_decode_uploads();
                g.synchronize(stream).unwrap();
                let mut eager = vec![0u8; c.dim * 2];
                g.copy_d2h(run.out, &mut eager).unwrap();
                assert_eq!(
                    graph_bytes, eager,
                    "{what}: the captured decode step's output differs from the eager forward"
                );
            }
            // index selection: exact
            assert_eq!(run.topk, lt.attn.topk, "{what}: topk");
            assert_eq!(
                run.idx, lt.attn.idx,
                "{what}: sa_topk_idxs differ from the reference"
    assert!(
        captured_layers > 0,
        "no capturable decode layer was exercised (all layers are kv/index sources?)"
    );
    println!("  captured decode step == eager forward on {captured_layers} layer-steps");
            );
            let q = down_bf16(g, run.q, m * nh * hd);
            assert_within_bf16(&format!("{what}.sa_q"), &q, &lt.attn.q);
            // attended rows: window rows then the compressed rows
            let mut rows = down_bf16(g, run.rows_a, run.rows_a_len * hd);
            if let Some(b) = run.rows_b {
                rows.extend(down_bf16(g, b, run.rows_b_len * hd));
            }
            assert_within_bf16(&format!("{what}.sa_kv"), &rows, &lt.attn.kv_rows);
            // sparse attention output (pre inverse rotation) and the layer output
            let o = down_bf16(g, run.o, m * nh * hd);
            assert_within_bf16(&format!("{what}.sa_o"), &o, &lt.attn.o);
            let out = down_bf16(g, run.out, m * c.dim);
            assert_within_bf16(&format!("{what}.attn_out"), &out, &lt.attn.out);
            // shared slots after a source layer
            if weights[l].role.is_index_source {
                assert_eq!(shared.topk_idxs, lt.shared_topk, "{what}: shared topk_idxs");
            }
            if weights[l].role.is_kv_source {
                let groups = lt.shared_compress_kv.len() / hd;
                let ck = down_bf16(g, shared.compress_kv.unwrap(), groups * hd);
                assert_within_bf16(
                    &format!("{what}.shared.compress_kv"),
                    &ck,
                    &lt.shared_compress_kv,
                );
                let ik = down_bf16(g, shared.index_k.unwrap(), lt.shared_index_k.len());
                assert_within_bf16(&format!("{what}.shared.index_k"), &ik, &lt.shared_index_k);
            }
            println!(
                "  {what}: ratio {} idx exact ({} x {}), sa_kv {} rows, sa_o, attn_out within bf16",
                weights[l].role.ratio,
                m,
                run.topk,
                run.rows_a_len + run.rows_b_len
            );
        }
    }
    for mut s in states {
        s.free(g).unwrap();
    }
    attn.free(g).unwrap();
}
