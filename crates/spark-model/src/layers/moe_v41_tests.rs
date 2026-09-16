// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! The streamed MoE against two oracles on synthetic Q2_K / Q3_K experts held in
//! a real pinned, device-visible arena:
//!   * routing: the reference `gate` on the same logits, indices exact;
//!   * output: the production numerics emulated on the CPU (the loader's own
//!     decoders for the weights, q8_1 emulation for the activations, bf16 GEMMs,
//!     the reference's SwiGLU and accumulation), within the K-quant tolerance;
//!     and the reference's expert math on the dequantised weights, loosely
//!     (the q8_1 activation quantisation is the only difference).

use std::sync::Mutex;

use anyhow::Result;

use super::*;
use crate::layers::deepseek_v41_ref::moe::{MoeCfg, MoeWeights, gate as ref_gate, moe as ref_moe};
use spark_runtime::weights::dequant_cpu::{GgmlType, dequant_to_f32};
use spark_runtime::weights::expert_stream::SlotLayout;

const DIM: usize = 512;
const INTER: usize = 256;
const N_ROUTED: usize = 8;
const TOPK: usize = 2;
const TOKENS: usize = 3;
const SCALES_F16: [u16; 4] = [0x2C00, 0x2800, 0x3000, 0x2400]; // 2^-4, 2^-5, 2^-3, 2^-6

fn lcg_bytes(n: usize, mut state: u32) -> Vec<u8> {
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        })
        .collect()
}

/// `[n][k]` raw super-blocks with sane per-block fp16 scales.
fn build_blocks(
    n: usize,
    k: usize,
    block_bytes: usize,
    f16_offsets: &[usize],
    seed: u32,
) -> Vec<u8> {
    let n_blocks = n * (k / 256);
    let mut raw = lcg_bytes(n_blocks * block_bytes, seed);
    for b in 0..n_blocks {
        for (i, &off) in f16_offsets.iter().enumerate() {
            let s = SCALES_F16[(b + i) % SCALES_F16.len()];
            raw[b * block_bytes + off] = (s & 0xFF) as u8;
            raw[b * block_bytes + off + 1] = (s >> 8) as u8;
        }
    }
    raw
}

fn bf16r(x: f32) -> f32 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    f32::from_bits(b.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000)
}

fn rand_bf16(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    lcg_bytes(n * 2, seed)
        .chunks_exact(2)
        .map(|c| bf16r((u16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0 - 1.0) * scale))
        .collect()
}

fn q8_emulate_block(x: &[f32], block: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (i, chunk) in x.chunks(block).enumerate() {
        let amax = chunk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        for (j, &v) in chunk.iter().enumerate() {
            out[i * block + j] = if amax == 0.0 {
                0.0
            } else {
                (v / d).round() * d
            };
        }
    }
    out
}

/// bf16 GEMM: `y[i][o] = bf16(sum x[i][d] * w[o][d])`.
fn gemm_bf16(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * n];
    for i in 0..rows {
        for o in 0..n {
            y[i * n + o] = bf16r(
                x[i * k..(i + 1) * k]
                    .iter()
                    .zip(&w[o * k..(o + 1) * k])
                    .map(|(a, b)| a * b)
                    .sum(),
            );
        }
    }
    y
}

struct Experts {
    gate: Vec<Vec<u8>>,
    up: Vec<Vec<u8>>,
    down: Vec<Vec<u8>>,
    reads: Mutex<usize>,
}

impl ExpertSource for Experts {
    fn slot_layout(&self) -> SlotLayout {
        SlotLayout::new(self.gate[0].len(), self.up[0].len(), self.down[0].len())
    }
    fn num_experts(&self) -> usize {
        self.gate.len()
    }
    fn read_expert(&self, _layer: u32, e: u32, dst: &mut [u8]) -> Result<()> {
        let l = self.slot_layout();
        let e = e as usize;
        dst[l.gate_off..l.gate_off + l.gate_bytes].copy_from_slice(&self.gate[e]);
        dst[l.up_off..l.up_off + l.up_bytes].copy_from_slice(&self.up[e]);
        dst[l.down_off..l.down_off + l.down_bytes].copy_from_slice(&self.down[e]);
        *self.reads.lock().unwrap() += 1;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn backend() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    let set = atlas_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
        .expect("deepseek-v4-flash/nvfp4 not in this build");
    spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend")
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn streamed_moe_matches_the_emulated_numerics_and_the_reference() {
    run_case(TOKENS);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn streamed_moe_prefill_groups_take_the_mmq_arm() {
    run_case(40);
}

#[cfg(feature = "cuda")]
fn run_case(tokens: usize) {
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::weights::expert_stream::PinnedArena;
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let cfg = MoeV41Cfg {
        dim: DIM,
        inter: INTER,
        n_routed: N_ROUTED,
        topk: TOPK,
        gate_temp: 1.0,
        norm_topk_prob: true,
        route_scale: 1.5,
        swiglu_limit: 10.0,
        max_tokens: tokens,
    };
    // experts: gate/up Q2_K [INTER, DIM], down Q3_K [DIM, INTER]
    let ex = Experts {
        gate: (0..N_ROUTED)
            .map(|e| build_blocks(INTER, DIM, 84, &[80, 82], 0x1000 + e as u32))
            .collect(),
        up: (0..N_ROUTED)
            .map(|e| build_blocks(INTER, DIM, 84, &[80, 82], 0x2000 + e as u32))
            .collect(),
        down: (0..N_ROUTED)
            .map(|e| build_blocks(DIM, INTER, 110, &[108], 0x3000 + e as u32))
            .collect(),
        reads: Mutex::new(0),
    };
    let q2 = GgmlType::from_id(10, 128).unwrap();
    let q3 = GgmlType::from_id(11, 128).unwrap();
    let deq = |t: GgmlType, raw: &[u8], n: usize| {
        let mut w = vec![0f32; n];
        dequant_to_f32(t, raw, n, &mut w).unwrap();
        w
    };
    let w1: Vec<Vec<f32>> = ex.gate.iter().map(|r| deq(q2, r, INTER * DIM)).collect();
    let w3: Vec<Vec<f32>> = ex.up.iter().map(|r| deq(q2, r, INTER * DIM)).collect();
    let w2: Vec<Vec<f32>> = ex.down.iter().map(|r| deq(q3, r, DIM * INTER)).collect();
    // router, shared expert, input
    let gate_w = rand_bf16(N_ROUTED * DIM, 0x77, 0.05);
    let gate_bias: Vec<f32> = (0..N_ROUTED).map(|e| (e as f32 - 3.5) * 0.01).collect();
    let s1 = rand_bf16(INTER * DIM, 0x81, 0.05);
    let s2 = rand_bf16(DIM * INTER, 0x82, 0.05);
    let s3 = rand_bf16(INTER * DIM, 0x83, 0.05);
    let x = rand_bf16(tokens * DIM, 0x99, 1.0);

    let up = |v: &[f32]| {
        let b = bf16_bytes(v);
        let p = g.alloc(b.len()).unwrap();
        g.copy_h2d(&b, p).unwrap();
        p
    };
    let lw = MoeV41LayerWeights {
        layer: 7,
        gate_w: up(&gate_w),
        gate_bias: gate_bias.clone(),
        shared_w1: ResidentMat::Bf16(up(&s1)),
        shared_w2: ResidentMat::Bf16(up(&s2)),
        shared_w3: ResidentMat::Bf16(up(&s3)),
    };
    let x_dev = up(&x);
    let moe = MoeV41::new(g, cfg.clone()).unwrap();
    let layout = ex.slot_layout();
    let arena = PinnedArena::alloc(g, N_ROUTED * layout.bytes).unwrap();
    let mut lru = ExpertLru::new(arena.host(), arena.dev(), arena.bytes(), layout).unwrap();

    let (out_dev, weights, indices) = moe
        .forward(g, &lw, &mut lru, &ex, x_dev, tokens, 2, stream)
        .unwrap();
    let mut ob = vec![0u8; tokens * DIM * 2];
    g.copy_d2h(out_dev, &mut ob).unwrap();
    let got: Vec<f32> = ob
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();

    // oracle 1: routing = the reference gate on f32 logits
    let (ref_w, ref_i) = ref_gate(
        &x, &gate_w, &gate_bias, tokens, DIM, N_ROUTED, TOPK, 1.0, true, 1.5,
    );
    assert_eq!(
        indices, ref_i,
        "routing indices differ from the reference gate"
    );
    for (a, b) in weights.iter().zip(&ref_w) {
        assert!(
            (a - b).abs() <= 1e-5 * b.abs().max(1.0),
            "routing weight {a} vs {b}"
        );
    }
    let uniq: std::collections::HashSet<usize> = indices.iter().copied().collect();
    assert_eq!(
        *ex.reads.lock().unwrap(),
        uniq.len(),
        "one read per distinct expert"
    );

    // oracle 2: the production numerics emulated on the CPU
    // group sizes decide the arm: <= 8 rows = the decode GEMV (q8 block 32
    // everywhere), larger = the MMQ arm (D2S6: block 64 for the Q2_K gate/up,
    // D4: block 32 for the Q3_K down)
    let mut group_size = vec![0usize; N_ROUTED];
    for &e in &indices {
        group_size[e] += 1;
    }
    let mut acc = vec![0f32; tokens * DIM];
    for t in 0..tokens {
        for kk in 0..TOPK {
            let e = indices[t * TOPK + kk];
            let rw = weights[t * TOPK + kk];
            let mmq = group_size[e] > 8;
            let xq = q8_emulate_block(&x[t * DIM..(t + 1) * DIM], if mmq { 64 } else { 32 });
            let gt = gemm_bf16(&xq, &w1[e], 1, DIM, INTER);
            let ut = gemm_bf16(&xq, &w3[e], 1, DIM, INTER);
            let h: Vec<f32> = (0..INTER)
                .map(|j| {
                    let g_ = gt[j].min(10.0);
                    let u = ut[j].clamp(-10.0, 10.0);
                    bf16r((g_ / (1.0 + (-g_).exp())) * u * rw)
                })
                .collect();
            let hq = q8_emulate_block(&h, 32);
            let d = gemm_bf16(&hq, &w2[e], 1, INTER, DIM);
            for j in 0..DIM {
                acc[t * DIM + j] += d[j];
            }
        }
    }
    let shared = crate::layers::deepseek_v41_ref::moe::expert(
        &x, &s1, &s2, &s3, tokens, DIM, INTER, 10.0, None,
    );
    let want: Vec<f32> = acc.iter().zip(&shared).map(|(a, s)| bf16r(a + s)).collect();
    let max = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let tol = max * 2f32.powi(-7) + 1e-3;
    let mut worst = 0f32;
    for (a, b) in got.iter().zip(&want) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst <= tol,
        "emulated numerics: max abs diff {worst} > tol {tol} (max |want| {max})"
    );
    println!(
        "  streamed MoE: {} values within {tol:.4} of the emulated numerics (worst {worst:.5})",
        got.len()
    );

    // oracle 3: the reference's own expert math on the dequantised weights, loosely
    let experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..N_ROUTED)
        .map(|e| (w1[e].clone(), w2[e].clone(), w3[e].clone()))
        .collect();
    let rw = MoeWeights {
        gate_w: &gate_w,
        gate_bias: &gate_bias,
        experts: experts
            .iter()
            .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice()))
            .collect(),
        shared: (&s1, &s2, &s3),
    };
    let rc = MoeCfg {
        dim: DIM,
        inter: INTER,
        n_routed: N_ROUTED,
        topk: TOPK,
        gate_temp: 1.0,
        norm_topk_prob: true,
        route_scale: 1.5,
        swiglu_limit: 10.0,
    };
    let (ref_y, _, _) = ref_moe(&x, tokens, &rw, &rc);
    let mut worst_ref = 0f32;
    for (a, b) in got.iter().zip(&ref_y) {
        worst_ref = worst_ref.max((a - b).abs());
    }
    let loose = max * 2f32.powi(-5) + 1e-2;
    assert!(
        worst_ref <= loose,
        "reference math: max abs diff {worst_ref} > {loose}"
    );
    println!(
        "  vs the reference's bf16 expert math (no q8 activations): worst {worst_ref:.5} (loose bound {loose:.4})"
    );

    moe.free(g).unwrap();
    arena.free(g).unwrap();
}
