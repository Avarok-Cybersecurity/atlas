// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the batch-M FP8 verify forward: the host model of the
//! kernel's loop order against the exact M=1 model (the bit-identity
//! argument, on the host), the route choice, and the launches it makes. The
//! device byte compare is `examples/fp8_ada_batchm_microtest.rs`.

use super::*;
use crate::layer::MoeLoraRoute;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::QuantizedWeight;
use crate::weight_map::fp8_blockscale_ref::{
    BlockScaledFp8, gemv_w8a16_exact, quantize_blockscaled, round_bf16,
};
use avarok_core::config::ModelConfig;
use avarok_core::numeric::fp8_e4m3_to_f32;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

// ── Host model of `fp8_gemv_ada_batchm_<M>` ─────────────────────────────

const LANES: usize = 64;
const CHUNK: usize = 16;

/// The batch-M kernel's arithmetic before the BF16 round, structured as the
/// kernel is: per output row, per lane, per chunk, per weight word: decode
/// once, then every activation row in turn. `a` is `[m, k]`.
fn batchm_model(a: &[f32], m: usize, w: &BlockScaledFp8) -> Vec<f32> {
    let k = w.k;
    let chunks = k / CHUNK;
    let mut out = vec![0f32; m * w.n];
    for n in 0..w.n {
        let mut lane_acc = vec![[0f32; LANES]; m];
        for lane in 0..LANES {
            let mut acc = vec![0f32; m];
            let mut c = lane;
            while c < chunks {
                let scale = w.scale_at(n, c * CHUNK);
                for j in 0..4 {
                    let col = c * CHUNK + 4 * j;
                    let wv: [f32; 4] =
                        std::array::from_fn(|i| fp8_e4m3_to_f32(w.bytes[n * k + col + i]) * scale);
                    for (t, acc_t) in acc.iter_mut().enumerate() {
                        for (i, wi) in wv.iter().enumerate() {
                            *acc_t += a[t * k + col + i] * wi;
                        }
                    }
                }
                c += LANES;
            }
            for (dst, v) in lane_acc.iter_mut().zip(&acc) {
                dst[lane] = *v;
            }
        }
        for (t, lanes) in lane_acc.iter().enumerate() {
            out[t * w.n + n] = warp_sum(&lanes[..32]) + warp_sum(&lanes[32..]);
        }
    }
    out
}

/// The 5-step `shfl.down` butterfly of one warp, lane 0's result.
fn warp_sum(half: &[f32]) -> f32 {
    let mut v = [0f32; 32];
    v.copy_from_slice(half);
    for offset in [16usize, 8, 4, 2, 1] {
        let prev = v;
        for i in 0..32 {
            let src = if i + offset < 32 {
                prev[i + offset]
            } else {
                prev[i]
            };
            v[i] = prev[i] + src;
        }
    }
    v[0]
}

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn bf16s(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| round_bf16(self.unit() * scale)).collect()
    }
}

#[test]
fn the_batchm_loop_order_is_m_launches_of_the_m1_kernel_bit_for_bit() {
    // Shapes with a partial scale block in K (272 = 2 x 128 + 16), a row
    // count that is not a multiple of the CTA's 4 or the scale block's 128,
    // and one deep K that wraps every lane several times.
    for (seed, (n, k)) in [(260usize, 272usize), (9, 2560), (5, 9216)]
        .into_iter()
        .enumerate()
    {
        let mut rng = Lcg(0x1515_0000 + seed as u64);
        let w = quantize_blockscaled(&rng.bf16s(n * k, 3.0), n, k);
        for m in MIN_ROWS..=MAX_ROWS {
            let a = rng.bf16s(m * k, 1.5);
            let batched = batchm_model(&a, m, &w);
            for (t, (row, got)) in a.chunks_exact(k).zip(batched.chunks_exact(n)).enumerate() {
                let single = gemv_w8a16_exact(row, &w);
                for (col, (b, s)) in got.iter().zip(&single).enumerate() {
                    assert_eq!(
                        b.to_bits(),
                        s.to_bits(),
                        "[{n}x{k}] m={m} row {t} col {col}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_planted_difference_in_one_row_moves_only_that_row() {
    // The comparator above must be able to fail: change one activation of
    // row 1 and exactly row 1 of the batched result moves.
    let (n, k, m) = (8usize, 256usize, 3usize);
    let mut rng = Lcg(0x1515_00FF);
    let w = quantize_blockscaled(&rng.bf16s(n * k, 3.0), n, k);
    let a = rng.bf16s(m * k, 1.0);
    let mut planted = a.clone();
    planted[k + 7] = round_bf16(planted[k + 7] + 0.5);
    let (x, y) = (batchm_model(&a, m, &w), batchm_model(&planted, m, &w));
    let moved = |t: usize| (0..n).any(|c| x[t * n + c].to_bits() != y[t * n + c].to_bits());
    assert!(!moved(0) && moved(1) && !moved(2));
}

// ── Route choice and launches ───────────────────────────────────────────

const H: u32 = 256;
const INTER: u32 = 512;
const DUAL_K: KernelHandle = KernelHandle(0xB2D);
const PLAIN_K: KernelHandle = KernelHandle(0xB21);
const SILU_K: KernelHandle = KernelHandle(0xB25);
const ACT_MUL: KernelHandle = KernelHandle(0xAC7);

fn kernels() -> RowKernels {
    RowKernels {
        plain: PLAIN_K,
        dual: DUAL_K,
        silu_input: SILU_K,
    }
}

fn fp8(gpu: &MockGpuBackend, n: u32, k: u32, format: WeightQuantFormat) -> Fp8Weight {
    Fp8Weight {
        weight: gpu.alloc((n * k) as usize).unwrap(),
        row_scale: gpu
            .alloc(4 * n.div_ceil(128) as usize * k.div_ceil(128) as usize)
            .unwrap(),
        n,
        k,
        scale_format: format,
    }
}

/// What `qwen35_dense.rs` builds on the native-FP8 route: NULL NVFP4
/// fallbacks and an FP8 overlay.
fn fp8_layer(
    gpu: &MockGpuBackend,
    activation: FfnActivation,
    format: WeightQuantFormat,
) -> DenseFfnLayer {
    let mut layer = DenseFfnLayer::new_with_activation(
        DenseFfnWeights {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        activation,
        gpu,
    )
    .unwrap();
    layer.act_mul = ACT_MUL;
    let (g, u, d) = (
        fp8(gpu, INTER, H, format),
        fp8(gpu, INTER, H, format),
        fp8(gpu, H, INTER, format),
    );
    layer.set_fp8_weights(g, u, d);
    layer
}

fn with_ctx<R>(
    gpu: &MockGpuBackend,
    split_silu: bool,
    f: impl FnOnce(&ForwardContext, &BufferArena) -> R,
) -> R {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = H as usize;
    config.intermediate_size = INTER as usize;
    config.num_experts = 0;
    config.num_experts_per_tok = 0;
    config.moe_intermediate_size = INTER as usize;
    config.vocab_size = 256;
    let buffers = BufferArena::new(&config, 64, 256, 256, 8, gpu).unwrap();
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let mut levers = ModelLevers::defaults();
    levers.decode_split_silu = split_silu;
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
        graph_capture: false,
        decode_step: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    f(&ctx, &buffers)
}

const BS: WeightQuantFormat = WeightQuantFormat::Fp8BlockScaled;

#[test]
fn the_route_is_the_decode_forwards_own_arm() {
    let gpu = MockGpuBackend::new();
    let silu = fp8_layer(&gpu, FfnActivation::SiLU, BS);
    let (h, i) = (H as usize, INTER as usize);
    assert_eq!(
        plan(&silu, &kernels(), 4, h, i, true),
        Some(VerifyRoute::SplitSilu)
    );
    assert_eq!(
        plan(&silu, &kernels(), 4, h, i, false),
        Some(VerifyRoute::FusedSilu)
    );
    let gelu = fp8_layer(&gpu, FfnActivation::GeLU, BS);
    assert_eq!(
        plan(&gelu, &kernels(), 4, h, i, true),
        Some(VerifyRoute::PerProjection)
    );
}

#[test]
fn every_row_count_from_two_to_eight_and_no_other() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu, FfnActivation::SiLU, BS);
    let (h, i) = (H as usize, INTER as usize);
    for m in 0..=12 {
        let ok = plan(&layer, &kernels(), m, h, i, true).is_some();
        assert_eq!(ok, (MIN_ROWS..=MAX_ROWS).contains(&m), "m={m}");
        assert_eq!(RowKernels::names(m, false).is_some(), ok, "names m={m}");
    }
}

#[test]
fn a_missing_twin_declines_rather_than_mixing_kernels() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu, FfnActivation::SiLU, BS);
    let (h, i) = (H as usize, INTER as usize);
    let no_dual = RowKernels {
        dual: KernelHandle(0),
        ..kernels()
    };
    assert_eq!(plan(&layer, &no_dual, 4, h, i, true), None);
    let no_silu = RowKernels {
        silu_input: KernelHandle(0),
        ..kernels()
    };
    // The split arm does not need the SiLU-input twin; the fused arm does.
    assert_eq!(
        plan(&layer, &no_silu, 4, h, i, true),
        Some(VerifyRoute::SplitSilu)
    );
    assert_eq!(plan(&layer, &no_silu, 4, h, i, false), None);
}

#[test]
fn per_row_scales_and_foreign_shapes_decline() {
    let gpu = MockGpuBackend::new();
    let rows = fp8_layer(&gpu, FfnActivation::SiLU, WeightQuantFormat::Fp8PerRow);
    let (h, i) = (H as usize, INTER as usize);
    assert_eq!(plan(&rows, &kernels(), 4, h, i, true), None);
    let layer = fp8_layer(&gpu, FfnActivation::SiLU, BS);
    assert_eq!(
        plan(&layer, &kernels(), 4, h + 16, i, true),
        None,
        "hidden mismatch"
    );
    assert_eq!(plan(&layer, &kernels(), 4, h, i + 8, true), None, "K % 16");
}

#[test]
fn a_layer_without_fp8_weights_is_not_eligible() {
    let gpu = MockGpuBackend::new();
    let layer = DenseFfnLayer::new(
        DenseFfnWeights {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        &gpu,
    )
    .unwrap();
    assert_eq!(
        plan(&layer, &kernels(), 4, H as usize, INTER as usize, true),
        None
    );
}

fn funcs(gpu: &MockGpuBackend, from: usize) -> Vec<u64> {
    gpu.launches_snapshot()[from..]
        .iter()
        .map(|l| l.func)
        .collect()
}

#[test]
fn the_split_route_is_three_launches_one_pass_per_weight() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu, FfnActivation::SiLU, BS);
    with_ctx(&gpu, true, |ctx, buffers| {
        let (x, out) = (buffers.norm_output(), buffers.moe_output());
        let start = gpu.launch_count();
        let allocs = gpu.alloc_count();
        assert!(forward_km_fp8_with(&layer, &kernels(), x, 5, out, ctx, 7).unwrap());
        assert_eq!(funcs(&gpu, start), vec![DUAL_K.0, ACT_MUL.0, PLAIN_K.0]);
        assert_eq!(
            gpu.alloc_count(),
            allocs,
            "no allocation on the verify path"
        );
        let l = gpu.launches_snapshot();
        assert_eq!(
            l[start].grid,
            [INTER.div_ceil(4), 1, 2],
            "dual: both projections"
        );
        assert_eq!(
            l[start + 1].grid[0],
            (5 * INTER).div_ceil(256),
            "silu over all rows"
        );
        assert_eq!(l[start + 2].grid, [H.div_ceil(4), 1, 1]);
        assert!(l[start..].iter().all(|l| l.stream == 7));
        // The down projection writes the caller's `out`.
        assert!(l[start + 2].args.contains(&MockArg::Buffer(out)));
        assert!(l[start].args.contains(&MockArg::Buffer(x)));
    });
}

#[test]
fn the_fused_route_is_two_launches() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu, FfnActivation::SiLU, BS);
    with_ctx(&gpu, false, |ctx, buffers| {
        let start = gpu.launch_count();
        let out = buffers.moe_output();
        assert!(
            forward_km_fp8_with(&layer, &kernels(), buffers.norm_output(), 8, out, ctx, 0).unwrap()
        );
        assert_eq!(funcs(&gpu, start), vec![DUAL_K.0, SILU_K.0]);
    });
}

#[test]
fn the_per_projection_route_mirrors_the_gelu_decode_arm() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu, FfnActivation::GeLU, BS);
    with_ctx(&gpu, true, |ctx, buffers| {
        let start = gpu.launch_count();
        let out = buffers.moe_output();
        assert!(
            forward_km_fp8_with(&layer, &kernels(), buffers.norm_output(), 3, out, ctx, 0).unwrap()
        );
        assert_eq!(
            funcs(&gpu, start),
            vec![PLAIN_K.0, PLAIN_K.0, ACT_MUL.0, PLAIN_K.0]
        );
    });
}

#[test]
fn a_decline_launches_nothing() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu, FfnActivation::SiLU, BS);
    with_ctx(&gpu, true, |ctx, buffers| {
        let start = gpu.launch_count();
        let out = buffers.moe_output();
        assert!(
            !forward_km_fp8_with(&layer, &kernels(), buffers.norm_output(), 9, out, ctx, 0)
                .unwrap()
        );
        assert_eq!(gpu.launch_count(), start);
    });
}

#[test]
fn with_the_row_off_the_entry_point_declines_without_a_lookup() {
    // The default environment resolves every port row off; this test is
    // meaningful only then (an operator running the suite with the row
    // exported is testing the other polarity on purpose).
    if ops::port_rows::ffn_fp8_verify_gemv::row() {
        return;
    }
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu, FfnActivation::SiLU, BS);
    with_ctx(&gpu, true, |ctx, buffers| {
        let lookups = gpu.kernel_lookups_snapshot().len();
        let start = gpu.launch_count();
        let out = buffers.moe_output();
        assert!(!forward_km_fp8(&layer, buffers.norm_output(), 4, out, ctx, 0).unwrap());
        assert_eq!(gpu.launch_count(), start);
        assert_eq!(gpu.kernel_lookups_snapshot().len(), lookups);
    });
}
