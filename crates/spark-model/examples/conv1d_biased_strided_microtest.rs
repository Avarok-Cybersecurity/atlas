// SPDX-License-Identifier: AGPL-3.0-only
//! Bit-exactness gate for the strided BIASED multi-sequence conv1d
//! (`causal_conv1d_update_strided`) against the per-sequence loop it replaces
//! in the Nemotron-H concurrent-decode path.
//!
//! ## Why
//! The Nemotron-H (Mamba-2) multi-seq decode runs the conv as an N-launch loop
//! with pre-offset pointers, because `causal_conv1d_update` hardcodes BOTH its
//! input and output row strides as `dim` (= d_xbc = 6144), while the batched
//! path feeds it from the batched `in_proj` output whose rows are
//! `in_proj_size` = 10304 apart. A `batch=n` launch of the plain kernel reads
//! sequence b>=1 from `b*6144` instead of `b*10304` — landing inside the
//! PREVIOUS sequence's B/C/dt region and feeding garbage into the SSM scan.
//! Correct at n=1, silently corrupt at n>=2: the nastiest failure mode there
//! is, because every single-sequence test still passes.
//!
//! The existing strided twin (`causal_conv1d_update_l2norm_f32_strided`) is
//! NOT usable here: it hardcodes `bias = NULL`, applies a per-head L2 norm,
//! and writes FP32. Nemotron needs a REAL conv1d bias, NO L2, and BF16 out.
//!
//! ## Legs
//!   GOLDEN:  `causal_conv1d_update` xN, batch=1, pre-offset pointers —
//!            exactly what the multi-seq path calls today.
//!   STRIDED: one `causal_conv1d_update_strided` launch, batch=N.
//!   CONTROL: `causal_conv1d_update` at batch=N — the bug this fix removes.
//!            It MUST MISMATCH. If it ever matches, the harness is not
//!            exercising the stride and the positive result is VOID.
//!
//! GATE: conv `output` AND the committed `conv_state` byte-identical. Both,
//! because a state-only bug is the dangerous one — it survives the step that
//! produced it and poisons the next token.
//!
//! Rows are laid out at the PRODUCTION input stride (10304) so a stride bug
//! pulls in real neighbouring data rather than zeros.
//!
//! Also reports isolated wall-clock for the N-launch loop vs the single
//! strided launch at n in {2,4,8}.
//!
//! Exit: 0 pass / 1 fail / 2 kernels absent from this build's PTX set.
//!
//!   cargo run -p spark-model --release --example conv1d_biased_strided_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

// Nemotron-3.5 Lightning-30B mamba-2 conv shapes.
const D_XBC: usize = 6144; // conv dim  (x | B | C)
const D_CONV: usize = 4;
const IN_PROJ_SIZE: usize = 10304; // x | B | C | dt | z  -> the input row stride

const SEEDS: [u64; 3] = [1, 99, 12345];
const RUNGS: [usize; 3] = [2, 4, 8];
const N_PARITY: usize = 5; // odd, > any tile count, and not a rung

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
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn as_bf16(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}
fn max_abs_err(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

/// Host-side inputs for one seed, at the PRODUCTION layout.
struct Inputs {
    n: usize,
    /// n rows of `IN_PROJ_SIZE` BF16 — the conv reads only the first `D_XBC`
    /// of each row; the rest is the dt/z region a stride bug would pull in.
    in_proj: Vec<bf16>,
    conv_state0: Vec<f32>,  // n * D_XBC * D_CONV, contiguous pool slots
    conv_weight: Vec<bf16>, // D_XBC * D_CONV
    conv_bias: Vec<f32>,    // D_XBC — REAL bias, not NULL
}

fn gen_inputs(seed: u64, n: usize) -> Inputs {
    let mut r = Lcg(seed);
    Inputs {
        n,
        in_proj: (0..n * IN_PROJ_SIZE)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect(),
        conv_state0: (0..n * D_XBC * D_CONV)
            .map(|_| r.r(-0.3, 0.3) as f32)
            .collect(),
        conv_weight: (0..D_XBC * D_CONV)
            .map(|_| bf16::from_f64(r.r(-0.3, 0.3)))
            .collect(),
        conv_bias: (0..D_XBC).map(|_| r.r(-0.2, 0.2) as f32).collect(),
    }
}

/// Device-resident copy of one `Inputs`, plus a fresh output buffer.
struct Dev {
    state: DevicePtr,
    input: DevicePtr,
    weight: DevicePtr,
    bias: DevicePtr,
    out: DevicePtr,
    n: usize,
}

fn upload(g: &dyn GpuBackend, inp: &Inputs) -> Result<Dev> {
    Ok(Dev {
        state: up_f32(g, &inp.conv_state0)?,
        input: up_bf16(g, &inp.in_proj)?,
        weight: up_bf16(g, &inp.conv_weight)?,
        bias: up_f32(g, &inp.conv_bias)?,
        out: g.alloc(inp.n * D_XBC * 2)?,
        n: inp.n,
    })
}

struct Captured {
    out: Vec<u8>,       // n * D_XBC BF16
    committed: Vec<u8>, // n * D_XBC * D_CONV FP32
}

fn capture(g: &dyn GpuBackend, d: &Dev) -> Result<Captured> {
    Ok(Captured {
        out: dn_bytes(g, d.out, d.n * D_XBC * 2)?,
        committed: dn_bytes(g, d.state, d.n * D_XBC * D_CONV * 4)?,
    })
}

/// One launch of the PLAIN (non-strided) kernel. `batch` + explicit pointers
/// let this serve BOTH the golden per-row loop (batch=1, pre-offset) and the
/// negative control (batch=N, which mis-strides the input).
fn launch_plain(
    g: &dyn GpuBackend,
    k: KernelHandle,
    state: DevicePtr,
    input: DevicePtr,
    weight: DevicePtr,
    bias: DevicePtr,
    out: DevicePtr,
    batch: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([D_XBC.div_ceil(256) as u32, batch, 1])
        .block([256, 1, 1])
        .arg_ptr(state)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(bias)
        .arg_ptr(out)
        .arg_u32(batch)
        .arg_u32(D_XBC as u32)
        .arg_u32(D_CONV as u32)
        .launch(0)
}

/// GOLDEN: the N-launch loop with pre-offset pointers (today's path).
fn enqueue_golden(g: &dyn GpuBackend, k: KernelHandle, d: &Dev) -> Result<()> {
    for i in 0..d.n {
        launch_plain(
            g,
            k,
            d.state.offset(i * D_XBC * D_CONV * 4),
            d.input.offset(i * IN_PROJ_SIZE * 2),
            d.weight,
            d.bias,
            d.out.offset(i * D_XBC * 2),
            1,
        )?;
    }
    Ok(())
}

/// STRIDED: one launch for all N rows.
fn enqueue_strided(g: &dyn GpuBackend, k: KernelHandle, d: &Dev) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([D_XBC.div_ceil(256) as u32, d.n as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(d.state)
        .arg_ptr(d.input)
        .arg_ptr(d.weight)
        .arg_ptr(d.bias)
        .arg_ptr(d.out)
        .arg_u32(d.n as u32)
        .arg_u32(D_XBC as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(IN_PROJ_SIZE as u32) // input row stride
        .arg_u32(D_XBC as u32) // output row stride
        .launch(0)
}

/// NEGATIVE CONTROL: the plain kernel at batch=N. Reads row b>=1 from
/// `b*D_XBC` inside a `IN_PROJ_SIZE`-strided buffer, so it MUST differ.
fn enqueue_plain_batched(g: &dyn GpuBackend, k: KernelHandle, d: &Dev) -> Result<()> {
    launch_plain(g, k, d.state, d.input, d.weight, d.bias, d.out, d.n as u32)
}

fn run<F>(g: &dyn GpuBackend, inp: &Inputs, f: F) -> Result<Captured>
where
    F: Fn(&dyn GpuBackend, &Dev) -> Result<()>,
{
    let d = upload(g, inp)?;
    f(g, &d)?;
    g.synchronize(0)?;
    capture(g, &d)
}

fn time_ms<F>(g: &dyn GpuBackend, d: &Dev, iters: usize, f: F) -> Result<f64>
where
    F: Fn(&dyn GpuBackend, &Dev) -> Result<()>,
{
    for _ in 0..20 {
        f(g, d)?;
    }
    g.synchronize(0)?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f(g, d)?;
    }
    g.synchronize(0)?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn main() -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;

    let opt = |m: &str, f: &str| g.kernel(m, f).unwrap_or(KernelHandle(0));
    let k_plain = opt("causal_conv1d", "causal_conv1d_update");
    let k_strided = opt("causal_conv1d", "causal_conv1d_update_strided");
    if k_plain.0 == 0 || k_strided.0 == 0 {
        println!(
            "SKIPPED: causal_conv1d_update{{,_strided}} not in this build's PTX set \
             (build with ATLAS_TARGET_MODEL=\"*\"). The bitwise gate did NOT run."
        );
        std::process::exit(2);
    }

    let mut all_ok = true;
    for seed in SEEDS {
        let inp = gen_inputs(seed, N_PARITY);
        let golden = run(g, &inp, |g, d| enqueue_golden(g, k_plain, d))?;
        let strided = run(g, &inp, |g, d| enqueue_strided(g, k_strided, d))?;
        let control = run(g, &inp, |g, d| enqueue_plain_batched(g, k_plain, d))?;

        let out_id = golden.out == strided.out;
        let st_id = golden.committed == strided.committed;
        let err = max_abs_err(&as_bf16(&golden.out), &as_bf16(&strided.out));
        // The control must NOT match, or this harness proves nothing.
        let ctrl_out_differs = golden.out != control.out;
        let ctrl_st_differs = golden.committed != control.committed;
        let ctrl_err = max_abs_err(&as_bf16(&golden.out), &as_bf16(&control.out));

        println!(
            "seed {seed:>5} n={N_PARITY}: output identical={out_id}  conv_state identical={st_id}  \
             max|err|={err:.3e}   [neg-control differs: out={ctrl_out_differs} \
             state={ctrl_st_differs} max|err|={ctrl_err:.3e}]"
        );
        if !out_id || !st_id {
            println!("  FAIL: strided batch=N diverges from the per-row loop");
            all_ok = false;
        }
        if !ctrl_out_differs || !ctrl_st_differs {
            println!(
                "  FAIL: the NON-strided kernel at batch=N matched golden — this harness \
                 is NOT exercising the stride, so the positive result is VOID"
            );
            all_ok = false;
        }
    }

    println!("\nisolated timing (Lightning conv shapes, 1000 iters):");
    for n in RUNGS {
        let inp = gen_inputs(7, n);
        let d = upload(g, &inp)?;
        let loop_ms = time_ms(g, &d, 1000, |g, d| enqueue_golden(g, k_plain, d))?;
        let one_ms = time_ms(g, &d, 1000, |g, d| enqueue_strided(g, k_strided, d))?;
        println!(
            "  n={n}: {n}-launch loop {loop_ms:.4} ms   1-launch strided {one_ms:.4} ms   \
             speedup {:.2}x",
            loop_ms / one_ms
        );
    }

    println!(
        "\n{}",
        if all_ok {
            "PASS — strided batch=N is byte-identical to the per-row loop (output AND \
             conv_state), and the unstrided batch=N control is provably wrong."
        } else {
            "FAIL"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
