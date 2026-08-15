// SPDX-License-Identifier: AGPL-3.0-only
//! Bit-exactness gate for the strided multi-sequence Mamba-2 SSM decode
//! (`mamba2_ssm_decode_strided`) against the per-sequence loop it replaces in
//! the concurrent-decode path.
//!
//! ## Why
//! `mamba2_ssm_decode` hardcodes every batch stride as the DENSE one
//! (`num_heads*head_dim` for x/output, `n_groups*state_size` for B/C,
//! `num_heads` for dt). In concurrent decode those tensors are slices of much
//! wider rows: x/B/C live inside the `d_xbc`-wide conv output and dt lives
//! inside the `in_proj_size`-wide projection row. A `batch=n` launch of the
//! plain kernel therefore reads sequence b>=1 from the wrong offset — landing
//! in a real neighbouring sequence's data — so the multi-seq path had to loop
//! at `batch=1` with pre-offset pointers, one launch per sequence per layer.
//!
//! The strided kernel takes the four strides explicitly so the whole batch
//! goes in ONE launch. This oracle proves it is numerically IDENTICAL:
//!
//!   GOLDEN:  `mamba2_ssm_decode` xN, batch=1, pre-offset pointers — exactly
//!            what the multi-seq path calls today.
//!   STRIDED: one `mamba2_ssm_decode_strided` launch, batch=N.
//!   GATE:    BF16 `output` AND the mutated FP32 `h_state` byte-identical.
//!            `h_state` is the dangerous one: a wrong-row state write is
//!            silent and poisons every future token of that sequence.
//!
//! A deliberate NEGATIVE control also runs: the plain kernel at batch=N (the
//! bug this kernel removes) MUST mismatch. If it ever matches, the test is not
//! exercising the stride difference and the positive result is void.
//!
//! Two shapes run: Lightning (state_size=128) and Puzzle (state_size=96). The
//! second exists specifically to cover the `n_warps = ceil(state_size/32)`
//! epilogue guard — a hard-coded 4 would sum unwritten smem garbage there.
//! A third variant uses a non-dense `y_stride` so the output stride is
//! exercised independently of the input strides.
//!
//! Also reports isolated timing: the N-launch loop vs the single strided
//! launch at N in {2,4,8}.
//!
//!   cargo run -p spark-model --release --example mamba2_strided_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

/// One SSM geometry + the production row strides its activations arrive with.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    num_heads: usize,
    head_dim: usize,
    state_size: usize,
    n_groups: usize,
    /// BF16 elems between sequences in the conv output (x, and B/C inside it).
    xbc_stride: usize,
    /// BF16 elems between sequences in the in_proj row (where dt lives).
    dt_stride: usize,
    /// BF16 elems between sequences in the SSM output.
    y_stride: usize,
}

impl Shape {
    /// x occupies [0, d_inner) of the conv row; B then C follow it.
    fn d_inner(&self) -> usize {
        self.num_heads * self.head_dim
    }
    fn b_off(&self) -> usize {
        self.d_inner()
    }
    fn c_off(&self) -> usize {
        self.d_inner() + self.n_groups * self.state_size
    }
    /// dt sits after z|xBC in the in_proj row.
    fn dt_off(&self) -> usize {
        self.d_inner() + self.xbc_stride
    }
    fn h_elems(&self) -> usize {
        self.num_heads * self.head_dim * self.state_size
    }
}

/// Lightning-30B: 64 heads x 64 head_dim x 128 state, 8 groups.
/// d_inner 4096, d_xbc = 4096 + 2*8*128 = 6144, in_proj = 4096+6144+64 = 10304.
const LIGHTNING: Shape = Shape {
    name: "lightning s128",
    num_heads: 64,
    head_dim: 64,
    state_size: 128,
    n_groups: 8,
    xbc_stride: 6144,
    dt_stride: 10304,
    y_stride: 4096,
};

/// Same, but with a deliberately non-dense output stride so `y_stride` is
/// exercised on its own (production y_stride == d_inner happens to be dense).
const LIGHTNING_WIDE_Y: Shape = Shape {
    name: "lightning wide-y",
    y_stride: 4608,
    ..LIGHTNING
};

/// Puzzle-75B geometry: state_size=96 -> n_warps=3, the epilogue guard case.
/// d_xbc = 4096 + 2*8*96 = 5632, in_proj = 4096+5632+64 = 9792.
const PUZZLE: Shape = Shape {
    name: "puzzle  s96",
    num_heads: 64,
    head_dim: 64,
    state_size: 96,
    n_groups: 8,
    xbc_stride: 5632,
    dt_stride: 9792,
    y_stride: 4096,
};

const NS: [usize; 3] = [2, 4, 8];
// No-op clamp — matches the production caller (`ssm_decode`).
const DT_MIN: f32 = 1e-9;
const DT_MAX: f32 = 1e9;

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
fn max_abs_bf16(a: &[u8], b: &[u8]) -> f64 {
    a.chunks_exact(2)
        .zip(b.chunks_exact(2))
        .map(|(x, y)| {
            let xv = bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f64();
            let yv = bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f64();
            (xv - yv).abs()
        })
        .fold(0.0f64, f64::max)
}
fn max_abs_f32(a: &[u8], b: &[u8]) -> f64 {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .map(|(x, y)| {
            let xv = f32::from_le_bytes([x[0], x[1], x[2], x[3]]) as f64;
            let yv = f32::from_le_bytes([y[0], y[1], y[2], y[3]]) as f64;
            (xv - yv).abs()
        })
        .fold(0.0f64, f64::max)
}

/// Host-side inputs, laid out at the PRODUCTION row strides so any wrong
/// stride lands inside a real neighbouring sequence's data.
struct Inputs {
    conv_out: Vec<bf16>, // n * xbc_stride  (x | B | C, plus slack)
    proj: Vec<bf16>,     // n * dt_stride   (z | xBC | dt)
    h0: Vec<f32>,        // n * h_elems     (contiguous pool slots)
    a_log: Vec<f32>,
    d_param: Vec<f32>,
    dt_bias: Vec<f32>,
}

fn gen_inputs(s: &Shape, n: usize, seed: u64) -> Inputs {
    let mut r = Lcg(seed);
    Inputs {
        conv_out: (0..n * s.xbc_stride)
            .map(|_| bf16::from_f64(r.r(-0.6, 0.6)))
            .collect(),
        proj: (0..n * s.dt_stride)
            .map(|_| bf16::from_f64(r.r(-1.5, 1.5)))
            .collect(),
        h0: (0..n * s.h_elems())
            .map(|_| r.r(-0.4, 0.4) as f32)
            .collect(),
        // A_log stores log|A|; keep dA in a sane range.
        a_log: (0..s.num_heads).map(|_| r.r(-1.5, 1.0) as f32).collect(),
        d_param: (0..s.num_heads).map(|_| r.r(-1.0, 1.0) as f32).collect(),
        dt_bias: (0..s.num_heads).map(|_| r.r(-1.0, 0.5) as f32).collect(),
    }
}

/// Device-resident copy of one `Inputs` (fresh state each leg).
struct Dev {
    h: DevicePtr,
    conv: DevicePtr,
    proj: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    out: DevicePtr,
}

impl Dev {
    fn new(g: &dyn GpuBackend, s: &Shape, n: usize, inp: &Inputs) -> Result<Self> {
        Ok(Dev {
            h: up_f32(g, &inp.h0)?,
            conv: up_bf16(g, &inp.conv_out)?,
            proj: up_bf16(g, &inp.proj)?,
            a_log: up_f32(g, &inp.a_log)?,
            d_param: up_f32(g, &inp.d_param)?,
            dt_bias: up_f32(g, &inp.dt_bias)?,
            out: g.alloc(n * s.y_stride * 2)?,
        })
    }
    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.h,
            self.conv,
            self.proj,
            self.a_log,
            self.d_param,
            self.dt_bias,
            self.out,
        ] {
            g.free(p)?;
        }
        Ok(())
    }
}

struct Captured {
    out: Vec<u8>,     // n * y_stride BF16
    h_state: Vec<u8>, // n * h_elems FP32
}

impl Captured {
    fn read(g: &dyn GpuBackend, s: &Shape, n: usize, d: &Dev) -> Result<Self> {
        Ok(Captured {
            out: dn_bytes(g, d.out, n * s.y_stride * 2)?,
            h_state: dn_bytes(g, d.h, n * s.h_elems() * 4)?,
        })
    }
}

/// One launch of the plain (non-strided) kernel. `batch` + the pointers let
/// this serve BOTH the golden per-seq loop (batch=1, pre-offset) and the
/// negative control (batch=N, which mis-strides everything).
#[allow(clippy::too_many_arguments)]
fn launch_plain(
    g: &dyn GpuBackend,
    k: KernelHandle,
    s: &Shape,
    d: &Dev,
    row: usize,
    batch: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([s.num_heads as u32, batch, 1])
        .block([s.state_size as u32, 1, 1])
        .arg_ptr(d.h.offset(row * s.h_elems() * 4))
        .arg_ptr(d.conv.offset(row * s.xbc_stride * 2))
        .arg_ptr(d.conv.offset((row * s.xbc_stride + s.b_off()) * 2))
        .arg_ptr(d.conv.offset((row * s.xbc_stride + s.c_off()) * 2))
        .arg_ptr(d.proj.offset((row * s.dt_stride + s.dt_off()) * 2))
        .arg_ptr(d.a_log)
        .arg_ptr(d.d_param)
        .arg_ptr(d.dt_bias)
        .arg_ptr(d.out.offset(row * s.y_stride * 2))
        .arg_u32(batch)
        .arg_u32(s.num_heads as u32)
        .arg_u32(s.head_dim as u32)
        .arg_u32(s.state_size as u32)
        .arg_u32(s.n_groups as u32)
        .arg_f32(DT_MIN)
        .arg_f32(DT_MAX)
        .launch(0)
}

/// One strided launch covering all `n` rows, through the production wrapper.
fn launch_strided(g: &dyn GpuBackend, k: KernelHandle, s: &Shape, d: &Dev, n: usize) -> Result<()> {
    ops::mamba2_ssm_decode_strided(
        g,
        k,
        d.h,
        d.conv,
        d.conv.offset(s.b_off() * 2),
        d.conv.offset(s.c_off() * 2),
        d.proj.offset(s.dt_off() * 2),
        d.a_log,
        d.d_param,
        d.dt_bias,
        d.out,
        n as u32,
        s.num_heads as u32,
        s.head_dim as u32,
        s.state_size as u32,
        s.n_groups as u32,
        DT_MIN,
        DT_MAX,
        s.xbc_stride as u32,
        s.xbc_stride as u32,
        s.dt_stride as u32,
        s.y_stride as u32,
        0,
    )
}

/// GOLDEN: the per-row batch=1 loop with pre-offset pointers.
fn run_golden(
    g: &dyn GpuBackend,
    k: KernelHandle,
    s: &Shape,
    n: usize,
    inp: &Inputs,
) -> Result<Captured> {
    let d = Dev::new(g, s, n, inp)?;
    g.memset(d.out, 0, n * s.y_stride * 2)?;
    for i in 0..n {
        launch_plain(g, k, s, &d, i, 1)?;
    }
    g.synchronize(0)?;
    let c = Captured::read(g, s, n, &d)?;
    d.free(g)?;
    Ok(c)
}

fn run_strided(
    g: &dyn GpuBackend,
    k: KernelHandle,
    s: &Shape,
    n: usize,
    inp: &Inputs,
) -> Result<Captured> {
    let d = Dev::new(g, s, n, inp)?;
    g.memset(d.out, 0, n * s.y_stride * 2)?;
    launch_strided(g, k, s, &d, n)?;
    g.synchronize(0)?;
    let c = Captured::read(g, s, n, &d)?;
    d.free(g)?;
    Ok(c)
}

/// NEGATIVE CONTROL: the plain kernel at batch=N — the bug this kernel
/// removes. It infers dense strides, so it MUST differ from golden.
fn run_plain_batched(
    g: &dyn GpuBackend,
    k: KernelHandle,
    s: &Shape,
    n: usize,
    inp: &Inputs,
) -> Result<Captured> {
    let d = Dev::new(g, s, n, inp)?;
    g.memset(d.out, 0, n * s.y_stride * 2)?;
    launch_plain(g, k, s, &d, 0, n as u32)?;
    g.synchronize(0)?;
    let c = Captured::read(g, s, n, &d)?;
    d.free(g)?;
    Ok(c)
}

/// Isolated timing: N batch=1 launches vs ONE strided launch, same work.
fn time_legs(
    g: &dyn GpuBackend,
    k_plain: KernelHandle,
    k_strided: KernelHandle,
    s: &Shape,
    n: usize,
    inp: &Inputs,
) -> Result<(f64, f64)> {
    const WARMUP: usize = 20;
    const ITERS: usize = 200;
    let d = Dev::new(g, s, n, inp)?;

    for _ in 0..WARMUP {
        for i in 0..n {
            launch_plain(g, k_plain, s, &d, i, 1)?;
        }
    }
    g.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        for i in 0..n {
            launch_plain(g, k_plain, s, &d, i, 1)?;
        }
    }
    g.synchronize(0)?;
    let per_row = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    for _ in 0..WARMUP {
        launch_strided(g, k_strided, s, &d, n)?;
    }
    g.synchronize(0)?;
    let t1 = Instant::now();
    for _ in 0..ITERS {
        launch_strided(g, k_strided, s, &d, n)?;
    }
    g.synchronize(0)?;
    let strided = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    d.free(g)?;
    Ok((per_row, strided))
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;

    let k_plain = g.kernel("mamba2_ssm", "mamba2_ssm_decode")?;
    let k_strided = g.kernel("mamba2_ssm", "mamba2_ssm_decode_strided")?;

    let mut all_ok = true;
    println!("== bit-parity: strided batch=N vs per-row batch=1 loop ==");
    for s in [LIGHTNING, LIGHTNING_WIDE_Y, PUZZLE] {
        for n in NS {
            for seed in [1u64, 12345] {
                let inp = gen_inputs(&s, n, seed);
                let golden = run_golden(g, k_plain, &s, n, &inp)?;
                let strided = run_strided(g, k_strided, &s, n, &inp)?;
                let bad = run_plain_batched(g, k_plain, &s, n, &inp)?;

                let out_id = golden.out == strided.out;
                let h_id = golden.h_state == strided.h_state;
                let d_out = max_abs_bf16(&golden.out, &strided.out);
                let d_h = max_abs_f32(&golden.h_state, &strided.h_state);
                // The control must NOT match, or this test proves nothing.
                let ctrl_differs = golden.out != bad.out || golden.h_state != bad.h_state;

                println!(
                    "{:<17} n={n} seed={seed:<6} out identical={out_id} \
                     h_state identical={h_id}  max|d out|={d_out:.3e} \
                     max|d h|={d_h:.3e}  [neg-control differs={ctrl_differs}]",
                    s.name
                );
                if !out_id || !h_id {
                    println!("  FAIL: strided diverges from the per-row loop");
                    all_ok = false;
                }
                if !ctrl_differs {
                    println!(
                        "  FAIL: plain kernel at batch=N matched golden — the test is NOT \
                         exercising the stride difference, so the positive result is void"
                    );
                    all_ok = false;
                }
            }
        }
    }

    println!("\n== isolated timing (us/step, {} iters) ==", 200);
    for s in [LIGHTNING, PUZZLE] {
        for n in NS {
            let inp = gen_inputs(&s, n, 7);
            let (per_row, strided) = time_legs(g, k_plain, k_strided, &s, n, &inp)?;
            println!(
                "{:<17} n={n}: per-row loop {per_row:8.2} us   strided {strided:8.2} us   \
                 speedup {:.2}x",
                s.name,
                per_row / strided
            );
        }
    }

    println!(
        "\n{}",
        if all_ok {
            "PASS — strided batch=N is byte-identical to the per-row loop \
             (output AND h_state), and the unstrided batch=N control is provably wrong."
        } else {
            "FAIL"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
