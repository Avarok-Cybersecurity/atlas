// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the dense FFN's W4 arm
//! (`dense_ffn/w4.rs`) against recording fakes of the `W4Linear` contract.
//!
//! What they pin:
//! * every entry point (`forward`, `forward_k2`, `forward_k3`, `forward_km`,
//!   `forward_prefill`, `forward_batched`) routes rows inside the decode
//!   window to ONE decode call per projection (gate+up through
//!   `decode_pair`), and wider row counts to chunked `prefill`;
//! * a declined prefill dequantizes the projection once and runs the BF16
//!   GEMM per declined chunk;
//! * a CPU oracle: with computing fakes, a verify row's output is byte-equal
//!   to the M = 1 decode of the same input row;
//! * with no W4 weights the hooks are inert: the BF16 and FP8 layers' small
//!   batch entry points launch exactly what `forward_prefill` launches, as
//!   before the hooks existed, and nothing is allocated;
//! * precedence (FP8 over W4), the BF16 overlay detach, the load-time scratch
//!   reservation, the capture guard, the finalizers' skip and the LoRA refusal.

use super::super::{DenseFfnLayer, DenseFfnWeights};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::port::w4::fake::{self, Call, FakeW4, Log};
use crate::port::w4::{self as w4c, W4Box, scratch};
use crate::weight_map::QuantizedWeight;
use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::sync::Arc;

const H: usize = 128;
const INTER: usize = 256;
const STREAM: u64 = 7;

fn config() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = H;
    config.intermediate_size = INTER;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = INTER;
    config.vocab_size = 128;
    config
}

fn null_nvfp4() -> DenseFfnWeights {
    DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    }
}

fn bare_layer(gpu: &MockGpuBackend) -> DenseFfnLayer {
    let mut layer = DenseFfnLayer::new(null_nvfp4(), gpu).unwrap();
    layer.act_mul = KernelHandle(0xAC7);
    layer
}

/// The three fakes of one layer, sharing one call log.
struct Fakes {
    log: Log,
    gate: Arc<FakeW4>,
    up: Arc<FakeW4>,
    down: Arc<FakeW4>,
}

fn fakes(max_rows: usize, edit: impl Fn(FakeW4) -> FakeW4) -> Fakes {
    let log = fake::log();
    let gate = Arc::new(edit(FakeW4::new("gate", INTER, H, max_rows, &log)));
    let up = Arc::new(edit(FakeW4::new("up", INTER, H, max_rows, &log)));
    let down = Arc::new(edit(FakeW4::new("down", H, INTER, max_rows, &log)));
    Fakes {
        log,
        gate,
        up,
        down,
    }
}

fn install(layer: &mut DenseFfnLayer, f: &Fakes) -> usize {
    let (g, u, d): (W4Box, W4Box, W4Box) = (f.gate.clone(), f.up.clone(), f.down.clone());
    layer.set_w4(g, u, d)
}

fn w4_layer(gpu: &MockGpuBackend, f: &Fakes) -> DenseFfnLayer {
    let mut layer = bare_layer(gpu);
    install(&mut layer, f);
    layer.finalize_w4_load(gpu, STREAM).unwrap();
    layer
}

fn with_ctx(gpu: &MockGpuBackend, capture: bool, f: impl FnOnce(&ForwardContext, &BufferArena)) {
    let config = config();
    let buffers = BufferArena::new(&config, 8, 256, 256, 8, gpu).unwrap();
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: capture,
        decode_step: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    f(&ctx, &buffers);
}

fn take(log: &Log) -> Vec<(&'static str, Call)> {
    std::mem::take(&mut *log.lock())
}

/// The decode-window call sequence for `rows` rows.
fn decode_seq(input: DevicePtr, rows: usize, b: &BufferArena) -> Vec<(&'static str, Call)> {
    let (g, u, o) = (b.expert_gate_out(), b.expert_up_out(), b.moe_output());
    vec![
        (
            "gate",
            Call::Decode {
                x: input,
                rows,
                out: g,
            },
        ),
        (
            "up",
            Call::Decode {
                x: input,
                rows,
                out: u,
            },
        ),
        ("down", Call::Decode { x: g, rows, out: o }),
    ]
}

type Key = (u64, [u32; 3], [u32; 3], u32, u64, Vec<MockArg>);

fn trace_since(gpu: &MockGpuBackend, from: usize) -> Vec<Key> {
    gpu.launches_snapshot()[from..]
        .iter()
        .map(|l| {
            (
                l.func,
                l.grid,
                l.block,
                l.shared_mem,
                l.stream,
                l.args.clone(),
            )
        })
        .collect()
}

#[test]
fn m1_decode_is_pair_then_swiglu_then_down() {
    let gpu = MockGpuBackend::new();
    let f = fakes(16, |w| w);
    let layer = w4_layer(&gpu, &f);
    with_ctx(&gpu, false, |ctx, b| {
        let input = b.norm_output();
        let start = gpu.launch_count();
        let out = layer.forward(input, ctx, STREAM).unwrap();
        assert_eq!(out, b.moe_output());
        assert_eq!(take(&f.log), decode_seq(input, 1, b));
        // Exactly one launch of our own between the projections: SwiGLU in
        // place over gate_out.
        let t = trace_since(&gpu, start);
        assert_eq!(t.len(), 1, "{t:?}");
        let g = MockArg::Buffer(b.expert_gate_out());
        let u = MockArg::Buffer(b.expert_up_out());
        assert!(t[0].5.contains(&g) && t[0].5.contains(&u));
    });
}

#[test]
fn a_fusing_pair_is_one_call() {
    let gpu = MockGpuBackend::new();
    let f = fakes(16, |w| w.fusing_pair());
    let layer = w4_layer(&gpu, &f);
    with_ctx(&gpu, false, |ctx, b| {
        let input = b.norm_output();
        layer.forward_km(input, 5, ctx, STREAM).unwrap();
        let (g, u, o) = (b.expert_gate_out(), b.expert_up_out(), b.moe_output());
        let pair = Call::Pair {
            x: input,
            rows: 5,
            out_a: g,
            out_b: u,
        };
        let down = Call::Decode {
            x: g,
            rows: 5,
            out: o,
        };
        assert_eq!(take(&f.log), vec![("gate", pair), ("down", down)]);
    });
}

#[test]
fn every_row_count_inside_the_window_is_one_decode_per_projection() {
    let gpu = MockGpuBackend::new();
    let f = fakes(16, |w| w);
    let layer = w4_layer(&gpu, &f);
    with_ctx(&gpu, false, |ctx, b| {
        let input = b.norm_output();
        layer.forward_k2(input, ctx, STREAM).unwrap();
        assert_eq!(take(&f.log), decode_seq(input, 2, b));
        layer.forward_k3(input, ctx, STREAM).unwrap();
        assert_eq!(take(&f.log), decode_seq(input, 3, b));
        for m in 4..=8u32 {
            assert!(
                layer.can_forward_km(m),
                "a W4 layer serves the {m}-row verify arm"
            );
            layer.forward_km(input, m, ctx, STREAM).unwrap();
            assert_eq!(take(&f.log), decode_seq(input, m as usize, b), "km m={m}");
        }
        for m in 1..=16usize {
            layer.forward_batched(input, m, ctx, STREAM).unwrap();
            assert_eq!(take(&f.log), decode_seq(input, m, b), "batched m={m}");
            layer.forward_prefill(input, m, ctx, STREAM).unwrap();
            assert_eq!(take(&f.log), decode_seq(input, m, b), "prefill m={m}");
        }
    });
}

#[test]
fn wider_row_counts_prefill_each_projection() {
    let gpu = MockGpuBackend::new();
    let f = fakes(16, |w| w);
    let layer = w4_layer(&gpu, &f);
    with_ctx(&gpu, false, |ctx, b| {
        let input = b.norm_output();
        let (g, u, o) = (b.expert_gate_out(), b.expert_up_out(), b.moe_output());
        for m in [17usize, 64, 2048] {
            layer.forward_prefill(input, m, ctx, STREAM).unwrap();
            assert_eq!(
                take(&f.log),
                vec![
                    (
                        "gate",
                        Call::Prefill {
                            x: input,
                            m,
                            out: g
                        }
                    ),
                    (
                        "up",
                        Call::Prefill {
                            x: input,
                            m,
                            out: u
                        }
                    ),
                    ("down", Call::Prefill { x: g, m, out: o }),
                ],
                "m={m}"
            );
        }
        // A narrow window: rows 9..=16 now prefill, and the pair splits.
        let f8 = fakes(8, |w| w);
        let narrow = w4_layer(&gpu, &f8);
        narrow.forward_batched(input, 12, ctx, STREAM).unwrap();
        let names: Vec<_> = take(&f8.log)
            .into_iter()
            .map(|(n, c)| (n, matches!(c, Call::Prefill { m: 12, .. })))
            .collect();
        assert_eq!(names, vec![("gate", true), ("up", true), ("down", true)]);
    });
}

#[test]
fn a_long_prefill_is_chunked_at_the_right_row_offsets() {
    let gpu = MockGpuBackend::new();
    let f = fakes(16, |w| w);
    let layer = w4_layer(&gpu, &f);
    with_ctx(&gpu, false, |ctx, b| {
        let input = b.norm_output();
        let (g, u, o) = (b.expert_gate_out(), b.expert_up_out(), b.moe_output());
        layer.forward_prefill(input, 5000, ctx, STREAM).unwrap();
        let chunks = [(0usize, 1667usize), (1667, 1667), (3334, 1666)];
        let mut want = Vec::new();
        for (name, x, out, k, n) in [
            ("gate", input, g, H, INTER),
            ("up", input, u, H, INTER),
            ("down", g, o, INTER, H),
        ] {
            for &(first, m) in &chunks {
                let c = Call::Prefill {
                    x: x.offset(first * k * 2),
                    m,
                    out: out.offset(first * n * 2),
                };
                want.push((name, c));
            }
        }
        assert_eq!(take(&f.log), want);
    });
}

#[test]
fn a_declined_chunk_dequantizes_once_and_runs_the_bf16_gemm() {
    let gpu = MockGpuBackend::new();
    // gate: chunk 1 accepted, chunks 2 and 3 declined; up and down accept.
    let log = fake::log();
    let gate = Arc::new(FakeW4::new("gate", INTER, H, 16, &log).answering(&[true, false, false]));
    let up = Arc::new(FakeW4::new("up", INTER, H, 16, &log));
    let down = Arc::new(FakeW4::new("down", H, INTER, 16, &log));
    let f = Fakes {
        log,
        gate,
        up,
        down,
    };
    let layer = w4_layer(&gpu, &f);
    let spec = layer.w4().unwrap().scratch_spec();
    let deq = scratch::bf16_dequant(&gpu, &spec).unwrap();
    with_ctx(&gpu, false, |ctx, b| {
        let input = b.norm_output();
        let g = b.expert_gate_out();
        let before = w4c::prefill_declines();
        let start = gpu.launch_count();
        layer
            .forward_w4_rows_with(input, 5000, ctx, false, STREAM)
            .unwrap();
        let gate_calls: Vec<Call> = take(&f.log)
            .into_iter()
            .filter(|(n, _)| *n == "gate")
            .map(|(_, c)| c)
            .collect();
        let x = |r: usize| input.offset(r * H * 2);
        let o = |r: usize| g.offset(r * INTER * 2);
        assert_eq!(
            gate_calls,
            vec![
                Call::Prefill {
                    x: x(0),
                    m: 1667,
                    out: o(0)
                },
                Call::Prefill {
                    x: x(1667),
                    m: 1667,
                    out: o(1667)
                },
                Call::Dequant { out: deq },
                Call::Prefill {
                    x: x(3334),
                    m: 1666,
                    out: o(3334)
                },
            ]
        );
        assert!(w4c::prefill_declines() >= before + 2);
        // One BF16 GEMM per declined chunk, reading the dequantized weight.
        let gemms: Vec<_> = trace_since(&gpu, start)
            .into_iter()
            .filter(|k| k.5.contains(&MockArg::Buffer(deq)))
            .collect();
        assert_eq!(gemms.len(), 2, "{gemms:?}");
        for (k, r) in gemms.iter().zip([1667usize, 3334]) {
            assert!(k.5.contains(&MockArg::Buffer(x(r))) && k.5.contains(&MockArg::Buffer(o(r))));
        }
    });
}

/// CPU oracle: computing fakes, a verify row's output bytes against the M=1
/// decode of the same row, through every small-batch entry point.
#[test]
fn verify_rows_are_byte_equal_to_m1_decode() {
    let gpu = MockGpuBackend::new();
    let f = fakes(16, |w| w.computing(0x5EED));
    let layer = w4_layer(&gpu, &f);
    with_ctx(&gpu, false, |ctx, b| {
        let input = b.norm_output();
        let rows: Vec<Vec<u8>> = (0..4)
            .map(|r| {
                (0..H)
                    .flat_map(|i| {
                        let v = ((r * 37 + i * 11) % 29) as f32 / 16.0 - 0.9;
                        half::bf16::from_f32(v).to_le_bytes()
                    })
                    .collect()
            })
            .collect();
        let read_out = |n_rows: usize| -> Vec<u8> {
            let mut v = vec![0u8; n_rows * H * 2];
            gpu.copy_d2h(b.moe_output(), &mut v).unwrap();
            v
        };
        let m1: Vec<Vec<u8>> = rows
            .iter()
            .map(|row| {
                gpu.copy_h2d(row, input).unwrap();
                layer.forward(input, ctx, STREAM).unwrap();
                read_out(1)
            })
            .collect();
        assert!(
            m1.iter().any(|o| o.iter().any(|&x| x != 0)),
            "the oracle must compute"
        );
        let stack = |m: usize| -> Vec<u8> { rows[..m].concat() };
        type Entry<'e> = Box<dyn Fn(usize) + 'e>;
        let entries: [(usize, Entry); 3] = [
            (
                2,
                Box::new(|_| layer.forward_k2(input, ctx, STREAM).unwrap()),
            ),
            (
                3,
                Box::new(|_| layer.forward_k3(input, ctx, STREAM).unwrap()),
            ),
            (
                4,
                Box::new(|m| layer.forward_km(input, m as u32, ctx, STREAM).unwrap()),
            ),
        ];
        for (m, run) in &entries {
            gpu.copy_h2d(&stack(*m), input).unwrap();
            run(*m);
            assert_eq!(read_out(*m), m1[..*m].concat(), "rows={m}");
        }
        take(&f.log);
    });
}

/// The load-time and precedence tests, split for the 500-line cap.
#[path = "w4_tests_load.rs"]
mod load;
