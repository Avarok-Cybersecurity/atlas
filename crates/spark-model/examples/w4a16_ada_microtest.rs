// SPDX-License-Identifier: AGPL-3.0-only
//! The never-measured Ada number: Atlas's NVFP4 W4A16 GEMV family against
//! `fp8_gemv_ada`, DRAM-true, M = 1..32, with row-invariance proved bit for bit.
//!
//! The question: if NVFP4 reaches
//! at least 85% of the measured ceiling at M=1 and stays a single weight pass at
//! M=8, the Marlin port is limited to GPTQ/AWQ compatibility and M = 16 to 64.
//! This example prints that receipt; it never flips a row.
//!
//! Ada has no FP4 datapath (`kernels/ada/HARDWARE.toml`): the kernels decode
//! E2M1 through a shared-memory table, so whether they reach the bandwidth bar
//! is open. The batch tiers are checked, not trusted:
//!  * `oracle`: `w4a16_gemv` against a CPU dequantize-and-dot of the same bytes
//!    (E2M1 table, E4M3 group scale, tensor scale), within 2 BF16 ulps plus a
//!    term-magnitude bound. Controls: a doubled tensor scale and swapped nibbles.
//!  * `parity`: every tier (`sw`, `batch2` .. `batch32`, the rt2 twin) is
//!    byte-identical, row for row, to M separate `w4a16_gemv` calls, at every M it
//!    serves. Controls: a 1-ULP perturbation of the reference activations and a
//!    reference shifted by one row.
//!  * `cold`: every ring is more than twice the L2; an L2-resident ring is the
//!    control.
//! Timing runs only when every gate passed. Method credit: TensorFold's rotated
//! distinct weight copies and its planted-control gates; Marlin's M sweep.
//!
//! Run on the Ada card (needs no model):
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example w4a16_ada_microtest --features cuda,gpu-examples -- \
//!       [--peak-gbs 250] [--m 1..32] [--json out.json]
use anyhow::{Result, anyhow};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::GpuBackend;
use std::collections::HashMap;

#[path = "common/bench_args.rs"]
mod bench_args;
#[path = "common/cold_l2.rs"]
mod cold_l2;
#[path = "common/nvfp4_oracle.rs"]
mod nvfp4_oracle;
#[path = "common/planted.rs"]
mod planted;
#[path = "common/profiler.rs"]
mod profiler;
#[path = "common/sweep_alloc.rs"]
mod sweep_alloc;
#[path = "common/sweep_extra.rs"]
mod sweep_extra;
#[path = "common/sweep_report.rs"]
mod sweep_report;
#[path = "common/sweep_routes.rs"]
mod sweep_routes;
#[path = "common/weight_fill.rs"]
mod weight_fill;

use bench_args::{arg_value, bench_cfg};
use cold_l2::{bench_arms, gbs, plan_hot, temperature_label};
use planted::{Gates, Verdict, diff_bf16_bytes, flip_one_ulp_bf16};
use sweep_alloc::{AllocCfg, Io, alloc_io, alloc_weights, free_io, free_weights};
use sweep_report::{Family, Row, SHAPES, weight_bytes};
use sweep_routes::{Ctx, Handles, Route, SCALE2, make_arm};

/// The receipt bars.
const PCT_BAR: f64 = 85.0;
/// "Single weight pass at M=8": the time at M=8 may exceed the time at M=1 by at
/// most this factor (one pass is about 1.0; two passes are 2.0).
const SINGLE_PASS_RATIO: f64 = 1.25;
const MAX_M: u32 = 32;

fn fp8_comparator(m: u32) -> Option<&'static str> {
    match m {
        1 => Some("fp8_gemv_ada"),
        2..=4 => Some("fp8_batch4"),
        5..=16 => Some("fp8_batch16"),
        _ => None,
    }
}

fn download(
    gpu: &dyn GpuBackend,
    p: spark_runtime::gpu::DevicePtr,
    bytes: usize,
) -> Result<Vec<u8>> {
    let mut host = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut host)?;
    Ok(host)
}

/// The CPU oracle gate on a small shape.
fn oracle_gate(
    gpu: &dyn GpuBackend,
    h: &Handles,
    gemv: &Route,
    stream: u64,
    gates: &mut Gates,
) -> Result<()> {
    let (n, k) = (256u32, 1024u32);
    let hot = AllocCfg {
        l2_bytes: 32 << 20,
        cap_bytes: 1 << 30,
        hot: true,
    };
    let w = alloc_weights(gpu, &[Family::Nvfp4], n, k, &hot)?;
    let io = alloc_io(gpu, n, k, 1)?;
    let ctx = Ctx {
        gpu,
        h,
        w: &w,
        io: &io,
        n,
        k,
        stream,
    };
    let (packed, scales) = w
        .fp4
        .as_ref()
        .map(|r| r.all()[0])
        .ok_or_else(|| anyhow!("no nvfp4 weights"))?;
    let (packed_h, scales_h) = (
        download(gpu, packed, (n * k / 2) as usize)?,
        download(gpu, scales, (n * k / 16) as usize)?,
    );
    let a_h = nvfp4_oracle::bf16_bytes_to_f32(&download(gpu, io.a, k as usize * 2)?);
    gpu.memset(io.c, 0, n as usize * 2)?;
    let mut arm = make_arm(&ctx, gemv, 1)?;
    arm.call()?;
    gpu.synchronize(stream)?;
    let got = download(gpu, io.c, n as usize * 2)?;
    drop(arm);
    gates.gate(
        "oracle: w4a16_gemv vs CPU dequantize-and-dot",
        &["scale2_doubled", "nibbles_swapped"],
        |v| {
            let (s2, swap) = match v {
                Some("scale2_doubled") => (2.0 * f64::from(SCALE2), false),
                Some("nibbles_swapped") => (f64::from(SCALE2), true),
                _ => (f64::from(SCALE2), false),
            };
            let (want, mag) =
                nvfp4_oracle::oracle(&a_h, &packed_h, &scales_h, s2, n as usize, k as usize, swap);
            let (bad, worst) = nvfp4_oracle::outside_bound(&got, &want, &mag);
            Ok(Verdict::new(
                bad == 0,
                format!("{bad} of {n} outputs outside the bound, worst {worst:.3} of it"),
            ))
        },
    );
    free_io(gpu, &io)?;
    free_weights(gpu, &w)
}

/// Row-invariance gates for every NVFP4 tier on one shape.
fn parity_gates(
    gpu: &dyn GpuBackend,
    h: &Handles,
    routes: &[Route],
    stream: u64,
    gates: &mut Gates,
) -> Result<()> {
    let s = SHAPES
        .iter()
        .find(|s| s.id == "ffn_up")
        .ok_or_else(|| anyhow!("shape table"))?;
    let (n, k) = (s.n, s.k);
    let hot = AllocCfg {
        l2_bytes: 32 << 20,
        cap_bytes: 1 << 30,
        hot: true,
    };
    let w = alloc_weights(gpu, &[Family::Nvfp4], n, k, &hot)?;
    let io = alloc_io(gpu, n, k, MAX_M + 2)?;
    let (c_ref, a_pert) = (
        gpu.alloc((MAX_M as usize + 2) * n as usize * 2)?,
        gpu.alloc((MAX_M as usize + 2) * k as usize * 2)?,
    );
    // The perturbed activations: one ULP flipped in every 61st element of row 1.
    let mut a_bytes = download(gpu, io.a, (MAX_M as usize + 2) * k as usize * 2)?;
    for e in (7..k as usize).step_by(61) {
        flip_one_ulp_bf16(&mut a_bytes, k as usize + e);
    }
    gpu.copy_h2d(&a_bytes, a_pert)?;
    let ctx = Ctx {
        gpu,
        h,
        w: &w,
        io: &io,
        n,
        k,
        stream,
    };
    let gemv = routes
        .iter()
        .find(|r| r.id == "nvfp4_gemv")
        .ok_or_else(|| anyhow!("nvfp4_gemv missing"))?;
    let (nn, kk) = (n as usize, k as usize);
    // M separate w4a16_gemv calls, one per row, under an optional planted defect.
    let reference = |rows: usize, control: Option<&str>| -> Result<Vec<u8>> {
        for r in 0..rows {
            let src_row = if control == Some("row_shifted") {
                r + 1
            } else {
                r
            };
            let base = if control == Some("ulp_row1") {
                a_pert
            } else {
                io.a
            };
            let io_r = Io {
                a: base.offset(src_row * kk * 2),
                c: c_ref.offset(r * nn * 2),
                ..io
            };
            let ctx_r = Ctx { io: &io_r, ..ctx };
            let mut arm = make_arm(&ctx_r, gemv, 1)?;
            arm.call()?;
        }
        gpu.synchronize(stream)?;
        download(gpu, c_ref, rows * nn * 2)
    };
    let batch = |route: &Route, m: u32| -> Result<Vec<u8>> {
        gpu.memset(io.c, 0, m as usize * nn * 2)?;
        let mut arm = make_arm(&ctx, route, m)?;
        arm.call()?;
        gpu.synchronize(stream)?;
        download(gpu, io.c, m as usize * nn * 2)
    };
    for r in routes
        .iter()
        .filter(|r| r.family == Family::Nvfp4 && r.id != "nvfp4_gemv")
    {
        gates.gate(
            &format!(
                "parity: {} == M x w4a16_gemv, M={}..={}",
                r.id, r.t_min, r.t_max
            ),
            // ulp_row1 perturbs row 1 only, so a route whose largest M is 1 cannot see it: planting it there makes
            // the gate vacuous by construction, not by a blind kernel. row_shifted is visible at M = 1.
            if r.t_max >= 2 {
                &["ulp_row1", "row_shifted"]
            } else {
                &["row_shifted"]
            },
            |v| {
                let want = reference(r.t_max as usize, v)?;
                let mut bad = Vec::new();
                for m in r.t_min..=r.t_max {
                    let got = batch(r, m)?;
                    let d = diff_bf16_bytes(&got, &want[..m as usize * nn * 2]);
                    if d.n_diff != 0 {
                        bad.push(format!("M={m}: {} elements differ", d.n_diff));
                    }
                }
                Ok(Verdict::new(
                    bad.is_empty(),
                    if bad.is_empty() {
                        "byte-identical at every M".to_string()
                    } else {
                        bad.join("; ")
                    },
                ))
            },
        );
    }
    for p in [c_ref, a_pert] {
        gpu.free(p)?;
    }
    free_io(gpu, &io)?;
    free_weights(gpu, &w)
}

fn main() -> Result<()> {
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let info = profiler::device_info();
    let l2 = if info.l2_bytes > 0 {
        info.l2_bytes
    } else {
        cold_l2::FALLBACK_L2_BYTES
    };
    let h = Handles::resolve(gpu);
    let mut routes = sweep_routes::catalog(&h, &[]);
    routes.retain(|r| r.family != Family::Bf16);
    if !routes.iter().any(|r| r.id == "nvfp4_gemv")
        || !routes.iter().any(|r| r.id == "fp8_gemv_ada")
    {
        println!(
            "w4a16_gemv or fp8_gemv_ada is absent from this target (build with AVAROK_TARGET_HW=ada): SKIP"
        );
        std::process::exit(2);
    }
    let gemv = routes
        .iter()
        .find(|r| r.id == "nvfp4_gemv")
        .cloned()
        .expect("checked above");
    let (peak, peak_src) = match arg_value("--peak-gbs") {
        Some(s) => (s.parse::<f64>()?, "measured (--peak-gbs)"),
        None => (
            info.spec_gbs().unwrap_or(288.0),
            "SPEC SHEET (run hbm_bandwidth_probe and pass --peak-gbs)",
        ),
    };
    let mlist = sweep_report::parse_t_list(&arg_value("--m").unwrap_or_else(|| "1..32".into()))
        .map_err(anyhow::Error::msg)?;
    let mut gates = Gates::new("w4a16_ada_microtest");
    oracle_gate(gpu, &h, &gemv, stream, &mut gates)?;
    parity_gates(gpu, &h, &routes, stream, &mut gates)?;
    if gates.failures() > 0 {
        println!(
            "timing skipped: a correctness gate failed, so the numbers would describe a broken kernel"
        );
        return gates.finish();
    }

    let cfg = bench_cfg(stream)?;
    println!(
        "device: {} SMs, L2 {} MiB | peak {peak:.1} GB/s [{peak_src}]",
        info.sm_count,
        l2 >> 20
    );
    println!("{}", sweep_report::table_header());
    let mut medians: HashMap<(String, String, u32), f64> = HashMap::new();
    let mut all_rows: Vec<Row> = Vec::new();
    for shape in SHAPES
        .iter()
        .filter(|s| s.id == "ffn_up" || s.id == "ffn_down")
    {
        let acfg = AllocCfg {
            l2_bytes: l2,
            cap_bytes: cold_l2::DEFAULT_RING_CAP,
            hot: false,
        };
        let w = alloc_weights(gpu, &[Family::Fp8, Family::Nvfp4], shape.n, shape.k, &acfg)?;
        let io = alloc_io(gpu, shape.n, shape.k, MAX_M)?;
        let ctx = Ctx {
            gpu,
            h: &h,
            w: &w,
            io: &io,
            n: shape.n,
            k: shape.k,
            stream,
        };
        let plan = w
            .plans
            .iter()
            .find(|(f, _)| *f == Family::Nvfp4)
            .map(|(_, p)| *p)
            .expect("nvfp4 plan");
        gates.gate(
            &format!("cold: {} nvfp4 ring exceeds 2x L2", shape.id),
            &["l2_resident_ring"],
            |v| {
                let p = if v.is_some() {
                    plan_hot(plan.per_copy_bytes, l2)
                } else {
                    plan
                };
                Ok(Verdict::new(
                    cold_l2::guard_cold(&p).is_ok(),
                    format!(
                        "{} copies, {} MiB total, L2 {} MiB",
                        p.copies,
                        p.total_bytes() >> 20,
                        l2 >> 20
                    ),
                ))
            },
        );
        let temp = temperature_label(&plan, false)?;
        for &m in mlist.iter().filter(|m| **m <= MAX_M) {
            let mut want: Vec<&str> = Vec::new();
            let ladder = sweep_report::ladder_route(Family::Nvfp4, m);
            want.extend(ladder.as_deref());
            if m == 1 {
                want.push("nvfp4_sw");
            }
            want.extend(fp8_comparator(m));
            let valid: Vec<&Route> = want
                .iter()
                .filter_map(|id| routes.iter().find(|r| r.id == *id && r.valid_at(m)))
                .collect();
            let mut arms = Vec::new();
            for r in &valid {
                arms.push(make_arm(&ctx, r, m)?);
            }
            let results = bench_arms(gpu, &mut arms, &cfg)?;
            drop(arms);
            for (r, res) in valid.iter().zip(&results) {
                let g = gbs(
                    weight_bytes(r.family, shape.n, shape.k) as f64,
                    res.median_us,
                );
                medians.insert((shape.id.to_string(), r.id.clone(), m), res.median_us);
                let m1 = medians
                    .get(&(shape.id.to_string(), r.id.clone(), 1))
                    .copied();
                let row = Row {
                    shape: shape.id.to_string(),
                    n: shape.n,
                    k: shape.k,
                    t: m,
                    route: r.id.clone(),
                    family: r.family,
                    median_us: res.median_us,
                    spread_pct: res.spread_pct,
                    gbs: g,
                    pct_peak: 100.0 * g / peak,
                    t1_lin_x: m1
                        .map_or(f64::NAN, |m1| sweep_report::t1_lin_x(m, m1, res.median_us)),
                    temp: temp.to_string(),
                };
                println!("{}", sweep_report::table_row(&row));
                all_rows.push(row);
            }
        }
        let get = |route: &str, m: u32| {
            medians
                .get(&(shape.id.to_string(), route.to_string(), m))
                .copied()
        };
        if let (Some(t1), Some(fp8)) = (get("nvfp4_gemv", 1), get("fp8_gemv_ada", 1)) {
            let g1 = gbs(weight_bytes(Family::Nvfp4, shape.n, shape.k) as f64, t1);
            let pct = 100.0 * g1 / peak;
            let verdict = |ok: bool| if ok { "MET" } else { "NOT_MET" };
            print!(
                "RECEIPT shape={} nvfp4_m1_gbs={g1:.1} pct_of_peak={pct:.1} bar={PCT_BAR} verdict={} fp8_over_nvfp4_m1={:.2}x",
                shape.id,
                verdict(pct >= PCT_BAR),
                fp8 / t1
            );
            match sweep_report::ladder_route(Family::Nvfp4, 8).and_then(|id| get(&id, 8)) {
                Some(t8) => println!(
                    " m8_over_m1={:.2} bar={SINGLE_PASS_RATIO} verdict={}",
                    t8 / t1,
                    verdict(t8 / t1 <= SINGLE_PASS_RATIO)
                ),
                None => println!(" m8_over_m1=n/a"),
            }
        }
        free_io(gpu, &io)?;
        free_weights(gpu, &w)?;
    }
    if let Some(path) = arg_value("--json") {
        let doc = serde_json::json!({ "tool": "w4a16_ada_microtest", "peak_gbs": peak, "peak_source": peak_src,
            "pct_bar": PCT_BAR, "single_pass_ratio": SINGLE_PASS_RATIO, "rows": all_rows.iter().map(sweep_report::row_json).collect::<Vec<_>>() });
        std::fs::write(&path, serde_json::to_string_pretty(&doc)?)?;
    }
    gates.finish()
}
