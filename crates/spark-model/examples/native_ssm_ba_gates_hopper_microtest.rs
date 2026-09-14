// SPDX-License-Identifier: AGPL-3.0-only

//! Oracle for the Hopper SSM BA-gates twin (#928).
//!
//! THE GATE IS BITWISE. `kernels/hopper/common/ssm_ba_gates_hopper.cu` changes
//! the launch geometry — one CTA per token instead of `ceil(N/4)` — and hoists
//! A's (exact) bf16 -> f32 widening out of the output loop. It changes nothing
//! else: the same lane-strided `kv` sweep, the same 5-step shuffle butterfly,
//! the same `warp_even + warp_odd` cross-warp sum, the same transforms. So
//! "close enough" is the wrong answer here: `gate` is a multiplicative decay
//! applied to the GDN state at every one of 48 layers and every chunk of the
//! scan, and `beta` is its write gate. A differing byte means an expression or
//! a reduction order moved. `max_abs` is printed so a failure is diagnosable,
//! not because a non-zero value would be accepted.
//!
//! It runs the four token counts round 13's nsys actually observed —
//! `M in {17, 25, 1168, 4576}` — and runs the twin at ALL of them, including
//! the two the dispatcher's token-count guard would refuse
//! (`ops::ba_gates_pick`). That is deliberate: bit-identity is a property of
//! the arithmetic and must hold at every M, while the guard is about filling
//! 132 SMs. Conflating the two would leave the tail-chunk shapes untested.
//!
//! Guard bytes bracket every device buffer either kernel writes, because the
//! twin computes its own output indices from a tiled group sweep and an
//! off-by-one there writes PAST the row rather than producing wrong numbers
//! inside it.
//!
//! Timing is reported per arm as microseconds, as COMPULSORY GB/s (one read of
//! the activation block, one read of the BA weight, one write of the gate
//! block) and as the activation-row READS each arm issues — 96 for the parent
//! at N=96 (one per BA output), 12 for the twin (one per 8-output tile per
//! 64-lane group). It is a single-harness observation on
//! whatever GPU runs it; on a non-Hopper device the twin still RUNS, it simply
//! has fewer SMs to leave idle, so treat the numbers as engagement evidence and
//! not as the H100 A/B. `SSM-BA-GATES-ATTRIBUTION.md` holds the receipt.
//!
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example native_ssm_ba_gates_hopper_microtest

use anyhow::Result;
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Qwen3.8-27B: `num_v_heads = 48`, `vheads_per_group = 2`, `hidden = 5120`,
/// so `ssm_ba_size = 2 * nv = 96` (`kernels/hopper/qwen3.8-27b/MODEL.toml`).
const NV: usize = 48;
const N: usize = 2 * NV;
const K: usize = 5120;
const VPG: usize = 2;
const GATE_STRIDE: usize = 2 * NV;

const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;
const WARMUP: u32 = 3;
const REPS: u32 = 10;

/// Deterministic LCG — the same inputs must reach both kernels, and a
/// thread-rng would make a failure unreproducible.
struct Lcg(u32);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) & 0xFF_FFFF) as f32 / 8_388_608.0 - 1.0
    }
}

fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
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

/// Allocate `bytes` of payload with `GUARD` sentinel bytes on each side and
/// return (base, payload).
fn guarded(g: &dyn GpuBackend, bytes: usize) -> Result<(DevicePtr, DevicePtr)> {
    let base = g.alloc(bytes + 2 * GUARD)?;
    g.copy_h2d(&vec![SENTINEL; bytes + 2 * GUARD], base)?;
    Ok((base, base.offset(GUARD)))
}

fn guards_intact(g: &dyn GpuBackend, base: DevicePtr, bytes: usize) -> Result<bool> {
    let mut raw = vec![0u8; bytes + 2 * GUARD];
    g.copy_d2h(base, &mut raw)?;
    Ok(raw[..GUARD].iter().all(|b| *b == SENTINEL)
        && raw[GUARD + bytes..].iter().all(|b| *b == SENTINEL))
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// (differing 4-byte lanes, largest absolute difference).
fn diff(a: &[u8], b: &[u8]) -> (usize, f32) {
    let mut n = 0usize;
    let mut worst = 0.0f32;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if x != y {
            n += 1;
            let (xv, yv) = (
                f32::from_le_bytes(x.try_into().unwrap()),
                f32::from_le_bytes(y.try_into().unwrap()),
            );
            worst = worst.max((xv - yv).abs());
        }
    }
    (n, worst)
}

fn time_us(g: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS))
}

/// Device-side inputs shared by both arms of one leg.
struct Leg {
    a: DevicePtr,
    b: DenseWeight,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
}

fn inputs(g: &dyn GpuBackend, m: usize) -> Result<Leg> {
    let mut r = Lcg(0x5EED_1234);
    // Activations and BA weights at the scale a real normed hidden state and a
    // trained projection sit at; the gate transform saturates for large |x|,
    // and a saturated gate hides a reduction-order difference.
    let a: Vec<f32> = (0..m * K).map(|_| r.next_f32() * 0.5).collect();
    let b: Vec<f32> = (0..N * K).map(|_| r.next_f32() * 0.02).collect();
    let a_log: Vec<f32> = (0..NV).map(|_| r.next_f32() * 2.0).collect();
    let dt_bias: Vec<f32> = (0..NV).map(|_| r.next_f32()).collect();
    Ok(Leg {
        a: up_bf16(g, &a)?,
        b: DenseWeight {
            weight: up_bf16(g, &b)?,
        },
        a_log: up_f32(g, &a_log)?,
        dt_bias: up_f32(g, &dt_bias)?,
    })
}

/// One token count: both arms, byte comparison, guards, timings.
fn leg(g: &dyn GpuBackend, parent: KernelHandle, twin: KernelHandle, m: usize) -> Result<bool> {
    let inp = inputs(g, m)?;
    let out_bytes = m * GATE_STRIDE * 4;
    let (ref_base, ref_p) = guarded(g, out_bytes)?;
    let (new_base, new_p) = guarded(g, out_bytes)?;

    let m32 = m as u32;
    // KernelHandle(0) for the twin forces `ba_gates_pick` down the "kernel
    // absent" arm, so this is the gb10 parent at every M regardless of what
    // `[defaults] ssm_ba_gates_hopper` says on the host this runs on.
    let run_parent = |dst: DevicePtr| -> Result<()> {
        ops::dense_gemm_ba_gates_prefill(
            g,
            parent,
            KernelHandle(0),
            inp.a,
            &inp.b,
            inp.a_log,
            inp.dt_bias,
            dst,
            m32,
            N as u32,
            K as u32,
            K as u32,
            GATE_STRIDE as u32,
            NV as u32,
            VPG as u32,
            0,
        )
    };
    // The twin's launcher DIRECTLY, bypassing the token-count guard: the guard
    // is a speed rule and this file is the correctness one.
    let run_twin = |dst: DevicePtr| -> Result<()> {
        ops::dense_gemm_ba_gates_prefill_hopper(
            g,
            twin,
            inp.a,
            &inp.b,
            inp.a_log,
            inp.dt_bias,
            dst,
            m32,
            N as u32,
            K as u32,
            K as u32,
            GATE_STRIDE as u32,
            NV as u32,
            VPG as u32,
            0,
        )
    };

    run_parent(ref_p)?;
    run_twin(new_p)?;
    g.synchronize(0)?;

    let ref_bytes = dn(g, ref_p, out_bytes)?;
    let new_bytes = dn(g, new_p, out_bytes)?;
    let (nd, worst) = diff(&ref_bytes, &new_bytes);
    let guards = guards_intact(g, ref_base, out_bytes)? && guards_intact(g, new_base, out_bytes)?;

    // KNOWN_BAD: a harness that has never rejected is not evidence. One ulp on
    // one gate is the SMALLEST defect this gate claims to catch, so that is
    // what the control injects.
    let mut bad = ref_bytes.clone();
    let poisoned = f32::from_le_bytes(bad[..4].try_into().unwrap());
    let nudged = f32::from_bits(poisoned.to_bits() ^ 1);
    bad[..4].copy_from_slice(&nudged.to_le_bytes());
    let (bad_n, _) = diff(&bad, &new_bytes);
    let control_ok = bad_n == nd + 1;

    let t_parent = time_us(g, || run_parent(ref_p))?;
    let t_twin = time_us(g, || run_twin(new_p))?;
    // Compulsory: the activation block read once, the BA weight read once, the
    // gate block written once. Everything above that is amplification.
    let compulsory = (m * K * 2 + N * K * 2 + out_bytes) as f64;
    let gbps = |us: f64| compulsory / (us * 1e-6) / 1e9;
    // Activation-row reads per token: the parent does one per BA output; the
    // twin does one per (64-lane group, output-group tile).
    let n_groups = N.div_ceil(ops::BA_GATES_OUTS as usize);
    let parent_amp = N;
    let twin_amp = n_groups.div_ceil(ops::BA_GATES_GROUPS as usize) * ops::BA_GATES_OUTS as usize;

    eprintln!(
        "  M={m:<5} out_diff={nd} max_abs={worst:.3e} guards={} control={} | \
         parent {t_parent:9.2} us {:7.1} GB/s (A reads {parent_amp}) | \
         twin {t_twin:9.2} us {:7.1} GB/s (A reads {twin_amp}) | {:.2}x",
        if guards { "ok" } else { "CLOBBERED" },
        if control_ok { "refused" } else { "BLIND" },
        gbps(t_parent),
        gbps(t_twin),
        t_parent / t_twin,
    );

    for p in [
        inp.a,
        inp.b.weight,
        inp.a_log,
        inp.dt_bias,
        ref_base,
        new_base,
    ] {
        g.free(p).ok();
    }
    anyhow::ensure!(guards, "M={m}: a kernel wrote outside its output buffer");
    anyhow::ensure!(
        control_ok,
        "M={m}: the KNOWN_BAD control did not trip — a one-ulp poison in the \
         reference produced {bad_n} differing lanes against a clean {nd}, so \
         this leg proves nothing"
    );
    anyhow::ensure!(
        nd == 0,
        "M={m}: {nd} gate/beta lanes differ from the gb10 parent (max_abs \
         {worst:.3e}); the twin must be BIT-identical"
    );
    Ok(true)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let parent = g.kernel("ssm_preprocess", "dense_gemm_ba_gates_prefill")?;
    let twin = g.kernel("ssm_ba_gates_hopper", "dense_gemm_ba_gates_prefill_hopper")?;

    let sms = ops::ba_gates_sm_count(g);
    eprintln!(
        "native_ssm_ba_gates_hopper_microtest: N={N} K={K} nv={NV} vpg={VPG} \
         sm_count={sms} twin_floor={} tokens",
        ops::ba_gates_min_tokens(sms)
    );
    eprintln!(
        "  block={} lanes/out={} outs/tile-step={} groups/tile={}",
        ops::BA_GATES_BLOCK,
        ops::BA_GATES_LANES,
        ops::BA_GATES_OUTS,
        ops::BA_GATES_GROUPS,
    );

    // The four token counts round 13's nsys observed: the 4593-token prefill's
    // two chunks (4576 + 17), the 1193-token prefix-cache replay tail (25), and
    // the 1168-token head chunk.
    for m in [17usize, 25, 1168, 4576] {
        leg(g, parent, twin, m)?;
    }
    eprintln!("  ALL LEGS BIT-IDENTICAL");
    Ok(())
}
