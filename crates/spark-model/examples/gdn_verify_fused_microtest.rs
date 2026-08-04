// SPDX-License-Identifier: AGPL-3.0-only
//! Golden oracle (STAGE 0) + fused conv+norm equivalence (STAGE 1) for the
//! fused K=2 GDN verify kernel (`gdn_verify_fused_k2`).
//!
//! ## Why
//! K=2 MTP verify runs the projection epilogue (conv1d+L2norm, GDN gates,
//! gated-RMS-norm) as PER-TOKEN loops — each launched/computed TWICE instead
//! of once-fused like the single-token decode path. The fused
//! activation-replay kernel folds those scalar GDN epilogue ops into a single
//! launch. This oracle is the losslessness foundation:
//!
//!   STAGE 0 (golden): run the CURRENT K=2 path op-for-op — the
//!     `causal_conv1d_update_l2norm` ×2 + `gated_delta_rule_wy2` +
//!     `gated_rms_norm` ×2 sequence exactly as `decode_batched_conv_gdn`
//!     calls them today — and capture the golden per-token gated-norm
//!     outputs, the committed H (after token 1), the rollback H_inter (after
//!     token 0), and the committed / intermediate conv-state. Deterministic
//!     across two runs of the same seed.
//!
//!   STAGE 1 (fused conv+norm): run `gdn_verify_fused_k2`, which fuses ONLY
//!     the conv1d+L2norm and the gated-RMS-norm for BOTH K=2 positions into
//!     one launch each (BA-projection/gates and the WY2 recurrence stay as
//!     their existing separate launches — Stage 2 folds BA in). GATE: fused
//!     per-token gated-norm output cos ≥ 0.99999 vs golden AND any conv-state
//!     it touches (committed + position-0 rollback) cos ≥ 0.99999.
//!
//!   cargo run -p spark-model --release --example gdn_verify_fused_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

// Qwen3-Next GDN head config (matches production / gdn_wy_verify_microtest).
const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const D_CONV: usize = 4;

// Channel dims (production: key_dim=2048, value_dim=4096, conv_dim=8192).
const KEY_DIM: usize = NK * KD; // 2048
const VALUE_DIM: usize = NV * VD; // 4096
const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM; // 8192 (Q|K|V)
const QK_CH: usize = KEY_DIM * 2; // 4096 (Q+K get L2 norm)
const QKVZ_SIZE: usize = CONV_DIM + VALUE_DIM; // 12288 (Q|K|V|Z)

const L2_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-6;
const PASS_COS: f64 = 0.99999;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Random K=2 layer inputs. Mirrors the production `decode_batched` layout:
///   - `deinterleaved`: [K, QKVZ_SIZE] BF16  (Q|K|V|Z per token)
///   - `conv_state`:    [CONV_DIM, D_CONV] FP32 (sliding window, in/out)
///   - `conv_weight`:   [CONV_DIM, D_CONV] BF16
///   - `h_state`:       [NV, KD, VD] FP32 (GDN recurrent state, in/out)
///   - `gates`:         [K, gate(NV)+beta(NV)] FP32   (gb_stride = 2*NV)
struct Inputs {
    deinterleaved: Vec<bf16>, // K*QKVZ_SIZE
    conv_state0: Vec<f32>,    // CONV_DIM*D_CONV (initial)
    conv_weight: Vec<bf16>,   // CONV_DIM*D_CONV
    h0: Vec<f32>,             // NV*KD*VD
    gates: Vec<f32>,          // K*2*NV  (gate then beta, per token)
    norm_weight: Vec<bf16>,   // VD
}

fn gen_inputs(seed: u64) -> Inputs {
    let mut r = Lcg(seed);
    let deinterleaved: Vec<bf16> = (0..2 * QKVZ_SIZE)
        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
        .collect();
    let conv_state0: Vec<f32> = (0..CONV_DIM * D_CONV)
        .map(|_| r.r(-0.3, 0.3) as f32)
        .collect();
    let conv_weight: Vec<bf16> = (0..CONV_DIM * D_CONV)
        .map(|_| bf16::from_f64(r.r(-0.3, 0.3)))
        .collect();
    let h0: Vec<f32> = (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect();
    // gates: per token [gate(NV), beta(NV)]; gate in (0.80,0.999), beta (0,1).
    let mut gates = Vec::with_capacity(2 * 2 * NV);
    for _ in 0..2 {
        for _ in 0..NV {
            gates.push(r.r(0.80, 0.999) as f32);
        }
        for _ in 0..NV {
            gates.push(r.r(0.0, 1.0) as f32);
        }
    }
    let norm_weight: Vec<bf16> = (0..VD).map(|_| bf16::from_f64(r.r(0.5, 1.5))).collect();
    Inputs {
        deinterleaved,
        conv_state0,
        conv_weight,
        h0,
        gates,
        norm_weight,
    }
}

/// Captured golden state from the current per-token path.
struct Golden {
    norm_out: Vec<f32>,       // K*VALUE_DIM (per-token gated-norm output, BF16→f32)
    h_committed: Vec<f32>,    // NV*KD*VD  (H after token 1)
    h_inter: Vec<f32>,        // NV*KD*VD  (H after token 0 — rollback)
    conv_committed: Vec<f32>, // CONV_DIM*D_CONV (after token 1)
    conv_inter: Vec<f32>,     // CONV_DIM*D_CONV (after token 0 — rollback)
}

/// Launch `causal_conv1d_update_l2norm` for a single token (batch=1), exactly
/// as `conv1d_update_l2norm` (ops::ssm_mamba) wraps it.
fn launch_conv1d(
    g: &dyn GpuBackend,
    k: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
) -> Result<()> {
    let bias = DevicePtr::NULL;
    KernelLaunch::new(g, k)
        .grid([CONV_DIM as u32 / 256, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_u32(1) // batch
        .arg_u32(CONV_DIM as u32) // dim
        .arg_u32(D_CONV as u32)
        .arg_u32(QK_CH as u32)
        .arg_u32(KD as u32) // head_dim (L2 group)
        .arg_f32(L2_EPS)
        .launch(0)
}

/// Launch `gated_delta_rule_wy2` (the K=2 WY recurrence), exactly as
/// `gdn_decode_wy2` (ops::ssm_gdn_b) wraps it. H/H_inter mutated in place.
#[allow(clippy::too_many_arguments)]
fn launch_wy2(
    g: &dyn GpuBackend,
    k: KernelHandle,
    h_state: DevicePtr,
    q: DevicePtr,
    key: DevicePtr,
    val: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    out: DevicePtr,
    h_inter: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(q)
        .arg_ptr(key)
        .arg_ptr(val)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(out)
        .arg_ptr(h_inter)
        .arg_u32(1) // batch_size
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(CONV_DIM as u32) // qk_stride
        .arg_u32(CONV_DIM as u32) // v_stride
        .arg_u32((NV * 2) as u32) // gb_stride
        .launch(0)
}

/// Launch `gated_rms_norm` for a single token, exactly as `gated_rms_norm`
/// (ops::norm) wraps it (num_tokens=NV, hidden_size=VD per-head).
fn launch_norm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, 1, 1]) // one block per head row
        .block([VD as u32, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(VD as u32) // hidden_size (per-head norm group)
        .arg_f32(RMS_EPS)
        .arg_u32(VD as u32) // gate_stride
        .arg_u32(VD as u32) // group_size (unused)
        .launch(0)
}

/// STAGE 0 — run the CURRENT K=2 path op-for-op and capture golden outputs.
///
/// Sequence mirrors `decode_batched_conv_gdn` (num_tokens==2 arm):
///   conv(t0) → snapshot conv_inter
///   conv(t1) → snapshot conv_committed
///   wy2 (writes H_inter after t0, H after t1)
///   norm(t0), norm(t1)
fn run_golden(g: &dyn GpuBackend, ins: &Inputs) -> Result<Golden> {
    let conv_k = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?;
    let wy2_k = g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?;
    let norm_k = g.kernel("norm", "gated_rms_norm")?;

    // Device buffers.
    let conv_state = up_f32(g, &ins.conv_state0)?;
    let conv_weight = up_bf16(g, &ins.conv_weight)?;
    let deint = up_bf16(g, &ins.deinterleaved)?;
    let h_state = up_f32(g, &ins.h0)?;
    let h_inter = g.alloc(NV * KD * VD * 4)?;
    let gates = up_f32(g, &ins.gates)?;
    let norm_w = up_bf16(g, &ins.norm_weight)?;

    // conv output: [K, CONV_DIM] BF16. GDN output: [K, VALUE_DIM] BF16.
    let conv_out = g.alloc(2 * CONV_DIM * 2)?;
    let gdn_out = g.alloc(2 * VALUE_DIM * 2)?;
    let norm_out = g.alloc(2 * VALUE_DIM * 2)?;
    // rollback snapshots of conv-state.
    let conv_inter = g.alloc(CONV_DIM * D_CONV * 4)?;
    let conv_committed = g.alloc(CONV_DIM * D_CONV * 4)?;

    // ── conv(t0) ──
    launch_conv1d(g, conv_k, conv_state, deint, conv_weight, conv_out)?;
    g.copy_d2d_async(conv_state, conv_inter, CONV_DIM * D_CONV * 4, 0)?;
    // ── conv(t1) ──
    let deint_1 = deint.offset(QKVZ_SIZE * 2);
    let conv_out_1 = conv_out.offset(CONV_DIM * 2);
    launch_conv1d(g, conv_k, conv_state, deint_1, conv_weight, conv_out_1)?;
    g.copy_d2d_async(conv_state, conv_committed, CONV_DIM * D_CONV * 4, 0)?;

    // ── WY2 GDN ──
    let q_ptr = conv_out;
    let k_ptr = conv_out.offset(KEY_DIM * 2);
    let v_ptr = conv_out.offset(KEY_DIM * 2 * 2);
    let gate_ptr = gates;
    let beta_ptr = gates.offset(NV * 4);
    launch_wy2(
        g, wy2_k, h_state, q_ptr, k_ptr, v_ptr, gate_ptr, beta_ptr, gdn_out, h_inter,
    )?;

    // ── gated-RMS-norm ×2 ──
    // Z gate is at offset [Q|K|V] within the deinterleaved buffer, per token.
    for t in 0..2usize {
        let gdn_t = gdn_out.offset(t * VALUE_DIM * 2);
        let z_t = deint.offset(t * QKVZ_SIZE * 2 + CONV_DIM * 2);
        let norm_t = norm_out.offset(t * VALUE_DIM * 2);
        launch_norm(g, norm_k, gdn_t, z_t, norm_w, norm_t)?;
    }
    g.synchronize(0)?;

    let golden = Golden {
        norm_out: dn_bf16(g, norm_out, 2 * VALUE_DIM)?,
        h_committed: dn_f32(g, h_state, NV * KD * VD)?,
        h_inter: dn_f32(g, h_inter, NV * KD * VD)?,
        conv_committed: dn_f32(g, conv_committed, CONV_DIM * D_CONV)?,
        conv_inter: dn_f32(g, conv_inter, CONV_DIM * D_CONV)?,
    };

    for p in [
        conv_state,
        conv_weight,
        deint,
        h_state,
        h_inter,
        gates,
        norm_w,
        conv_out,
        gdn_out,
        norm_out,
        conv_inter,
        conv_committed,
    ] {
        let _ = g.free(p);
    }
    Ok(golden)
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    let mut all_ok = true;
    for layer in 0..6u64 {
        let ins = gen_inputs(0xF02E_D000 ^ layer);

        // GATE: golden capture is deterministic — run twice on the same seed
        // and assert byte-identical across every captured tensor.
        let a = run_golden(g, &ins)?;
        let b = run_golden(g, &ins)?;
        let repro = a.norm_out == b.norm_out
            && a.h_committed == b.h_committed
            && a.h_inter == b.h_inter
            && a.conv_committed == b.conv_committed
            && a.conv_inter == b.conv_inter;
        all_ok &= repro;
        eprintln!(
            "STAGE0 layer={layer}  golden reproducible={}  \
             norm_out[0..4]={:?}  h_committed[0]={:.6}  conv_inter[0]={:.6}",
            if repro { "YES" } else { "NO" },
            &a.norm_out[0..4],
            a.h_committed[0],
            a.conv_inter[0],
        );

        // STAGE 1: fused conv+norm kernel vs golden.
        let fused = run_fused_stage1(g, &ins)?;
        let mut min_out = 1.0f64;
        for t in 0..2 {
            let fa = &fused.norm_out[t * VALUE_DIM..(t + 1) * VALUE_DIM];
            let ga = &a.norm_out[t * VALUE_DIM..(t + 1) * VALUE_DIM];
            min_out = min_out.min(cos(fa, ga));
        }
        let conv_committed_cos = cos(&fused.conv_committed, &a.conv_committed);
        let conv_inter_cos = cos(&fused.conv_inter, &a.conv_inter);
        let state_cos = conv_committed_cos.min(conv_inter_cos);
        let ok = min_out >= PASS_COS && state_cos >= PASS_COS;
        all_ok &= ok;
        eprintln!(
            "STAGE1 layer={layer}  norm_out_cos={min_out:.7} \
             conv_committed_cos={conv_committed_cos:.7} \
             conv_inter_cos={conv_inter_cos:.7}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    eprintln!(
        "\nFused-verify GATE (STAGE0 repro + STAGE1 cos≥{PASS_COS}): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}

/// Captured outputs from the Stage-1 fused conv+norm kernel.
struct FusedStage1 {
    norm_out: Vec<f32>,       // K*VALUE_DIM
    conv_committed: Vec<f32>, // CONV_DIM*D_CONV (after token 1)
    conv_inter: Vec<f32>,     // CONV_DIM*D_CONV (after token 0)
}

/// STAGE 1 — fused conv1d+L2norm ×2 in one launch, then WY2 (existing
/// separate launch), then fused gated-RMS-norm ×2 in one launch.
///
/// Drives `gdn_verify_fused_k2` the same way `decode_batched_conv_gdn` does
/// under `ATLAS_GDN_FUSED_VERIFY`: the conv phase advances state 0→1 in
/// registers and writes the position-0 conv-state snapshot once; WY2 consumes
/// its conv output (unchanged); the norm phase produces the gated-norm output.
fn run_fused_stage1(g: &dyn GpuBackend, ins: &Inputs) -> Result<FusedStage1> {
    let conv_k = g.kernel("gdn_verify_fused_k2", "gdn_verify_fused_conv_k2")?;
    let norm_k = g.kernel("gdn_verify_fused_k2", "gdn_verify_fused_norm_k2")?;
    let wy2_k = g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?;

    let conv_state = up_f32(g, &ins.conv_state0)?;
    let conv_weight = up_bf16(g, &ins.conv_weight)?;
    let deint = up_bf16(g, &ins.deinterleaved)?;
    let h_state = up_f32(g, &ins.h0)?;
    let h_inter = g.alloc(NV * KD * VD * 4)?;
    let gates = up_f32(g, &ins.gates)?;
    let norm_w = up_bf16(g, &ins.norm_weight)?;

    let conv_out = g.alloc(2 * CONV_DIM * 2)?;
    let gdn_out = g.alloc(2 * VALUE_DIM * 2)?;
    let norm_out = g.alloc(2 * VALUE_DIM * 2)?;
    let conv_inter = g.alloc(CONV_DIM * D_CONV * 4)?;

    // ── Fused conv phase: both positions, one launch ──
    KernelLaunch::new(g, conv_k)
        .grid([CONV_DIM as u32 / 256, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(deint)
        .arg_ptr(conv_weight)
        .arg_ptr(conv_out)
        .arg_ptr(conv_inter)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(QK_CH as u32)
        .arg_u32(KD as u32)
        .arg_u32(QKVZ_SIZE as u32) // input stride (BF16 elems between positions)
        .arg_u32(CONV_DIM as u32) // output stride
        .arg_f32(L2_EPS)
        .launch(0)?;

    // ── WY2 GDN (unchanged separate launch) ──
    let q_ptr = conv_out;
    let k_ptr = conv_out.offset(KEY_DIM * 2);
    let v_ptr = conv_out.offset(KEY_DIM * 2 * 2);
    launch_wy2(
        g,
        wy2_k,
        h_state,
        q_ptr,
        k_ptr,
        v_ptr,
        gates,
        gates.offset(NV * 4),
        gdn_out,
        h_inter,
    )?;

    // ── Fused norm phase: both positions, one launch ──
    KernelLaunch::new(g, norm_k)
        .grid([NV as u32, 2, 1])
        .block([VD as u32, 1, 1])
        .arg_ptr(gdn_out)
        .arg_ptr(deint)
        .arg_ptr(norm_w)
        .arg_ptr(norm_out)
        .arg_u32(VD as u32) // hidden_size (per-head group)
        .arg_f32(RMS_EPS)
        .arg_u32(QKVZ_SIZE as u32) // deint position stride (BF16 elems)
        .arg_u32(CONV_DIM as u32) // z offset within a position
        .arg_u32(VALUE_DIM as u32) // gdn/out position stride
        .launch(0)?;

    g.synchronize(0)?;

    let out = FusedStage1 {
        norm_out: dn_bf16(g, norm_out, 2 * VALUE_DIM)?,
        conv_committed: dn_f32(g, conv_state, CONV_DIM * D_CONV)?,
        conv_inter: dn_f32(g, conv_inter, CONV_DIM * D_CONV)?,
    };

    for p in [
        conv_state,
        conv_weight,
        deint,
        h_state,
        h_inter,
        gates,
        norm_w,
        conv_out,
        gdn_out,
        norm_out,
        conv_inter,
    ] {
        let _ = g.free(p);
    }
    Ok(out)
}
