// SPDX-License-Identifier: AGPL-3.0-only

//! Launch tests for gated dense-BF16 Q/K/V in multi-seq decode
//! (`ops::attn_gated_bf16_batchm`), on the mock backend through the real
//! `ms_phase_qkv`.
//!
//! The row is a pure weight-pass change: with it on, each projection is ONE
//! `dense_gemv_batchm` for all rows instead of one `dense_gemv` per row, and
//! the per-row `deinterleave_qg` calls are the same launches at the same
//! addresses either way. Numeric identity of `dense_gemv_batchm` against
//! `dense_gemv_bf16` is the kernel's own property (its header; GPU oracle
//! `examples/dense_gemv_bf16_batchm_microtest`), not this file's.

use super::super::ctx::MultiSeqCtx;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::layers::{FfnComponent, qwen3_attention::Qwen3AttentionLayer};
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight};
use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

const GEMV_K: u64 = 0xB001;
const BATCHM_K: u64 = 0xB0B8;
const DEINTERLEAVE_K: u64 = 0xF0D1;
const WIDTH: usize = 128;

/// What one `ms_phase_qkv` run launched, per projection and for the split.
struct Run {
    /// (q, k, v): every launch that read that projection's weight.
    proj: [Vec<MockLaunch>; 3],
    /// The `deinterleave_qg` launches, by output address.
    deinterleave: Vec<MockArg>,
    rows: usize,
    per_seq_qkv: usize,
    normed: spark_runtime::gpu::DevicePtr,
}

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_ne_bytes().to_vec())
}

fn run(rows: usize, row_on: bool) -> Run {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = WIDTH;
    config.intermediate_size = 128;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = WIDTH;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let buffers = BufferArena::new(&config, 8, 16, 16, 8, &gpu).unwrap();
    let w = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    let norm = w(WIDTH * 2);
    let (q, k, v) = (
        w(2 * WIDTH * WIDTH * 2),
        w(WIDTH * WIDTH * 2),
        w(WIDTH * WIDTH * 2),
    );
    let attn = AttentionWeights {
        q_proj: q,
        k_proj: k,
        v_proj: v,
        o_proj: QuantizedWeight::null(),
        q_norm: norm,
        k_norm: norm,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new(
        norm,
        attn,
        norm,
        FfnComponent::None,
        0,
        None,
        None,
        None,
        &gpu,
        KvCacheDtype::Bf16,
        0,
        &config,
    )
    .unwrap();
    assert!(layer.gated, "the Qwen3-Next config is output-gated");
    assert!(
        layer.q_weight.is_none() && layer.k_weight.is_none() && layer.v_weight.is_none(),
        "dense BF16 attention carries no quantized sidecar"
    );
    layer.dense_gemv_k = KernelHandle(GEMV_K);
    layer.dense_gemv_batchm_k = KernelHandle(BATCHM_K);
    layer.deinterleave_qg_k = KernelHandle(DEINTERLEAVE_K);
    layer.gated_bf16_batchm = row_on;

    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let fwd = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        decode_step: false,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    let c = MultiSeqCtx::new(
        &layer,
        &fwd,
        buffers.hidden_states(),
        buffers.residual(),
        rows,
        16,
        0,
    );
    let first = gpu.launch_count();
    layer.ms_phase_qkv(&c).unwrap();
    let all: Vec<MockLaunch> = gpu.launches_snapshot()[first..].to_vec();
    let reading = |p: spark_runtime::gpu::DevicePtr| -> Vec<MockLaunch> {
        all.iter()
            .filter(|l| l.args.contains(&MockArg::Buffer(p)))
            .cloned()
            .collect()
    };
    Run {
        proj: [reading(q.weight), reading(k.weight), reading(v.weight)],
        deinterleave: all
            .iter()
            .filter(|l| l.func == DEINTERLEAVE_K)
            .map(|l| l.args[0].clone())
            .collect(),
        rows,
        per_seq_qkv: c.per_seq_qkv,
        normed: c.normed,
    }
}

/// The batched tier only covers 2..=8 rows (the kernel's MAX_M); one row is
/// the plain decode shape and always takes the per-row GEMV.
fn batched(rows: usize, row_on: bool) -> bool {
    row_on && (2..=8).contains(&rows)
}

fn check(rows: usize, row_on: bool) {
    let r = run(rows, row_on);
    for (i, launches) in r.proj.iter().enumerate() {
        if batched(rows, row_on) {
            assert_eq!(
                launches.len(),
                1,
                "proj {i}: one weight pass at {rows} rows"
            );
            let l = &launches[0];
            assert_eq!(l.func, BATCHM_K, "proj {i}");
            assert_eq!(
                l.args[0],
                MockArg::Buffer(r.normed),
                "proj {i}: all rows from row 0"
            );
            assert_eq!(l.args[3], u32_arg(rows as u32), "proj {i}: M = rows");
        } else {
            assert_eq!(
                launches.len(),
                rows,
                "proj {i}: one GEMV per row at {rows} rows"
            );
            assert!(launches.iter().all(|l| l.func == GEMV_K), "proj {i}");
        }
    }
    // The split: one per row, at each row's own Q segment, in both routes.
    let qkv_buf_row0 = match &r.proj[0][0].args[2] {
        MockArg::Buffer(p) => *p,
        other => panic!("q output is not a buffer: {other:?}"),
    };
    let expect: Vec<MockArg> = (0..r.rows)
        .map(|i| MockArg::Buffer(qkv_buf_row0.offset(i * r.per_seq_qkv)))
        .collect();
    assert_eq!(
        r.deinterleave, expect,
        "deinterleave at {rows} rows, row_on={row_on}"
    );
}

#[test]
fn gated_bf16_qkv_launches_at_1_2_4_8_streams_with_the_row_on() {
    for rows in [1, 2, 4, 8] {
        check(rows, true);
    }
}

#[test]
fn gated_bf16_qkv_launches_at_1_2_4_8_streams_with_the_row_off() {
    for rows in [1, 2, 4, 8] {
        check(rows, false);
    }
}

/// Above the kernel's MAX_M the tier declines even with the row on.
#[test]
fn gated_bf16_qkv_above_eight_rows_stays_per_row() {
    check(12, true);
}

/// The row only moves weight passes: the split runs the same launches either
/// way, so the Q/K/V bits the attention reads are the per-row GEMV's.
#[test]
fn gated_bf16_qkv_split_is_the_same_on_both_routes() {
    for rows in [2, 4, 8] {
        assert_eq!(run(rows, true).deinterleave, run(rows, false).deinterleave);
    }
}
