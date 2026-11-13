// SPDX-License-Identifier: AGPL-3.0-only
//! Per-T linear sweep: time every linear route at T = 1..128 rows for the 4B
//! projection shapes, DRAM-true, and report where each route stops sharing its
//! weight pass.
//!
//! The question it answers (queue row Q7): is C=8 MTP bandwidth- or
//! compute-limited on Ada? A route that reads its weights once for T rows keeps
//! `T1_lin_x = T * median(T=1) / median(T)` near T; one that is compute- or
//! launch-limited falls away. The output also names, per T, the fastest route
//! and how far the route Atlas dispatches today (the "ladder") sits from it.
//!
//! Method credit: Cinference's linear-op benchmark and its `T1_lin_x` metric
//! (`bench/README.md`, `docs/maintainer/linear-tuning.md` sections 1 to 4); the
//! rotating weight copies are TensorFold's `QWEN27_DISTINCT` method.
//!
//! Shapes (N x K): ffn_up 9216x2560, ffn_down 2560x9216, gdn_in 12288x2560,
//! out_proj 2560x4096, lm_head 248320x2560. Routes: BF16 `dense_gemv_bf16`,
//! `dense_gemv_bf16_batchm`, cuBLASLt; FP8 `fp8_gemv_ada`, the gb10
//! `w8a16_gemv`, the W8A16 batch4 and batch16 tiers, the W8A8 GEMM; NVFP4
//! `w4a16_gemv` and its tiers (see `common/sweep_routes.rs`). Kernels of a later
//! lane (affine, Marlin, IQ4) join by name with `--route-kernel`.
//!
//! Every weight read is cold: each family keeps a ring of distinct copies whose
//! total exceeds twice the L2, and the run refuses to report otherwise
//! (`--hot` prints the L2-resident contrast, marked HOT). Percent-of-peak uses
//! the MEASURED ceiling: pass the `read_gbs` that `hbm_bandwidth_probe` printed
//! as `--peak-gbs`, or the spec sheet is used and labelled as such.
//!
//! Gates (each with a planted control): every route's output is finite and not
//! all zero, so a mis-specified launch cannot post a time for a kernel that did
//! nothing. Bit parity of NVFP4 rows across M is `w4a16_ada_microtest`.
//!
//! Run on the Ada card (needs no model):
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example linear_t_sweep --features cuda,gpu-examples -- \
//!       [--shapes ffn_up,lm_head] [--t all|priority|1..8,16] [--families bf16,fp8,nvfp4] \
//!       [--peak-gbs 250] [--cap-mib 1536] [--hot] [--json out.json] \
//!       [--route-kernel label=module:entry:abi[:grid=D][:tmax=T]]
use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::GpuBackend;
use std::collections::HashMap;

#[path = "common/bench_args.rs"]
mod bench_args;
#[path = "common/cold_l2.rs"]
mod cold_l2;
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

use bench_args::{all_values, arg_parse, arg_value, bench_cfg, has_flag};
use cold_l2::{bench_arms, gbs, temperature_label};
use planted::{Gates, Verdict, all_finite_bf16, count_nonzero_bf16};
use sweep_alloc::{AllocCfg, alloc_io, alloc_weights, free_io, free_weights};
use sweep_report::{Family, MAX_T, Row, SHAPES, Shape, weight_bytes};
use sweep_routes::{Ctx, Handles, Route, make_arm};

/// The shape the output-sanity gates run on: the smallest, so the D2H is small.
const SANITY_SHAPE: &str = "out_proj";

fn pick_shapes() -> Result<Vec<Shape>> {
    let want = bench_args::list_value_in(&std::env::args().skip(1).collect::<Vec<_>>(), "--shapes");
    match want {
        None => Ok(SHAPES.to_vec()),
        Some(ids) => ids
            .iter()
            .map(|id| {
                SHAPES.iter().find(|s| s.id == id).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown shape {id}; known: ffn_up, ffn_down, gdn_in, out_proj, lm_head"
                    )
                })
            })
            .collect(),
    }
}

fn pick_families() -> Result<Vec<Family>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match bench_args::list_value_in(&args, "--families") {
        None => Ok(vec![Family::Bf16, Family::Fp8, Family::Nvfp4]),
        Some(v) => v
            .iter()
            .map(|s| Family::parse(s).ok_or_else(|| anyhow::anyhow!("unknown family {s}")))
            .collect(),
    }
}

/// Run `route` once at `t` rows and return the output BF16 bytes.
fn run_once(gpu: &dyn GpuBackend, ctx: &Ctx<'_>, r: &Route, t: u32) -> Result<Vec<u8>> {
    let len = t as usize * ctx.n as usize * 2;
    gpu.memset(ctx.io.c, 0, len)?;
    let mut arm = make_arm(ctx, r, t)?;
    arm.call()?;
    gpu.synchronize(ctx.stream)?;
    let mut host = vec![0u8; len];
    gpu.copy_d2h(ctx.io.c, &mut host)?;
    Ok(host)
}

/// The output-sanity gate for every route that can run at `t <= 8`.
fn sanity_gates(gpu: &dyn GpuBackend, ctx: &Ctx<'_>, routes: &[Route], gates: &mut Gates) {
    for r in routes {
        let t = r.t_max.min(8);
        gates.gate(
            &format!("{} output is finite and non-zero (T={t})", r.id),
            &["zeroed_output", "nan_output"],
            |v| {
                let mut bytes = match v {
                    Some("zeroed_output") => vec![0u8; t as usize * ctx.n as usize * 2],
                    _ => run_once(gpu, ctx, r, t)?,
                };
                if v == Some("nan_output") {
                    bytes[..2].copy_from_slice(&half::bf16::NAN.to_bits().to_le_bytes());
                }
                let nz = count_nonzero_bf16(&bytes);
                Ok(Verdict::new(
                    all_finite_bf16(&bytes) && nz > bytes.len() / 4,
                    format!("{nz} of {} elements non-zero", bytes.len() / 2),
                ))
            },
        );
    }
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
    let shapes = pick_shapes()?;
    let fams = pick_families()?;
    let tlist = sweep_report::parse_t_list(&arg_value("--t").unwrap_or_else(|| "all".into()))
        .map_err(anyhow::Error::msg)?;
    let hot = has_flag("--hot");
    let cap = arg_parse("--cap-mib", 1536usize)? << 20;
    let (peak, peak_src) = match arg_value("--peak-gbs") {
        Some(s) => (s.parse::<f64>()?, "measured (--peak-gbs)"),
        None => match info.spec_gbs() {
            Some(s) => (
                s,
                "SPEC SHEET (run hbm_bandwidth_probe and pass --peak-gbs)",
            ),
            None => (288.0, "SPEC SHEET 288 (unverified)"),
        },
    };
    let handles = Handles::resolve(gpu);
    let mut extras = Vec::new();
    for spec in all_values("--route-kernel") {
        let e = sweep_extra::parse_extra(&spec)?;
        match sweep_routes::resolve_extra(gpu, &e) {
            Some(r) => extras.push(r),
            None => println!(
                "route x:{} skipped: {}::{} is absent from this target",
                e.label, e.module, e.entry
            ),
        }
    }
    let mut routes = sweep_routes::catalog(&handles, &extras);
    routes.retain(|r| fams.contains(&r.family));
    if let Some(only) = arg_value("--routes") {
        let keys: Vec<&str> = only.split(',').collect();
        routes.retain(|r| keys.iter().any(|k| r.id.contains(k)));
    }
    if routes.is_empty() {
        bail!("no route is available for the chosen families on this target");
    }
    println!(
        "device: {} SMs, L2 {} MiB | peak {peak:.1} GB/s [{peak_src}] | {} routes, {} T values, rotated x{} rounds",
        info.sm_count,
        l2 >> 20,
        routes.len(),
        tlist.len(),
        bench_cfg(stream)?.rounds
    );
    for r in &routes {
        println!(
            "  route {:<18} {:<6} T {}..{}",
            r.id,
            r.family.name(),
            r.t_min,
            r.t_max
        );
    }
    let cfg = bench_cfg(stream)?;
    // The sanity gates run routes at up to 8 rows whatever the T list says.
    let max_t = tlist.last().copied().unwrap_or(1).clamp(8, MAX_T);
    let mut gates = Gates::new("linear_t_sweep");
    let mut t1: HashMap<(String, String), f64> = HashMap::new();
    let mut all_rows: Vec<Row> = Vec::new();
    println!("{}", sweep_report::table_header());

    for shape in &shapes {
        let in_use: Vec<Family> = fams
            .iter()
            .copied()
            .filter(|f| routes.iter().any(|r| r.family == *f))
            .collect();
        let acfg = AllocCfg {
            l2_bytes: l2,
            cap_bytes: cap,
            hot,
        };
        let w = match alloc_weights(gpu, &in_use, shape.n, shape.k, &acfg) {
            Ok(w) => w,
            Err(e) => {
                println!("shape {} skipped: {e:#}", shape.id);
                continue;
            }
        };
        let io = alloc_io(gpu, shape.n, shape.k, max_t)?;
        let ctx = Ctx {
            gpu,
            h: &handles,
            w: &w,
            io: &io,
            n: shape.n,
            k: shape.k,
            stream,
        };
        let temp_of = |f: Family| -> Result<&'static str> {
            let plan = w.plans.iter().find(|(pf, _)| *pf == f).map(|(_, p)| p);
            temperature_label(plan.expect("plan exists for every family in use"), hot)
        };
        for &t in &tlist {
            let valid: Vec<&Route> = routes.iter().filter(|r| r.valid_at(t)).collect();
            if valid.is_empty() {
                continue;
            }
            let mut arms = Vec::new();
            for r in &valid {
                arms.push(make_arm(&ctx, r, t)?);
            }
            let results = bench_arms(gpu, &mut arms, &cfg)?;
            drop(arms);
            let mut rows: Vec<Row> = Vec::new();
            for (r, res) in valid.iter().zip(&results) {
                let g = gbs(
                    weight_bytes(r.family, shape.n, shape.k) as f64,
                    res.median_us,
                );
                let key = (shape.id.to_string(), r.id.clone());
                if t == 1 {
                    t1.insert(key.clone(), res.median_us);
                }
                let lin = t1
                    .get(&key)
                    .map_or(f64::NAN, |m1| sweep_report::t1_lin_x(t, *m1, res.median_us));
                rows.push(Row {
                    shape: shape.id.to_string(),
                    n: shape.n,
                    k: shape.k,
                    t,
                    route: r.id.clone(),
                    family: r.family,
                    median_us: res.median_us,
                    spread_pct: res.spread_pct,
                    gbs: g,
                    pct_peak: 100.0 * g / peak,
                    t1_lin_x: lin,
                    temp: temp_of(r.family)?.to_string(),
                });
            }
            for row in &rows {
                println!("{}", sweep_report::table_row(row));
            }
            for &f in &in_use {
                let fr: Vec<Row> = rows.iter().filter(|r| r.family == f).cloned().collect();
                if let Some(s) = sweep_report::summarize(&fr, f) {
                    let gap = s.ladder_us.map_or("n/a".to_string(), |l| {
                        format!("{:+.1}%", sweep_report::ladder_gap_pct(l, s.best_us))
                    });
                    println!(
                        "# {} T={t} {}: best {} ({:.1} us), ladder {} gap {gap}",
                        shape.id,
                        f.name(),
                        s.best,
                        s.best_us,
                        s.ladder.as_deref().unwrap_or("none")
                    );
                }
            }
            all_rows.extend(rows);
        }
        if shape.id == SANITY_SHAPE {
            sanity_gates(gpu, &ctx, &routes, &mut gates);
        }
        free_io(gpu, &io)?;
        free_weights(gpu, &w)?;
    }
    if let Some(path) = arg_value("--json") {
        let doc = serde_json::json!({
            "tool": "linear_t_sweep", "peak_gbs": peak, "peak_source": peak_src,
            "l2_bytes": l2, "hot": hot,
            "rows": all_rows.iter().map(sweep_report::row_json).collect::<Vec<_>>(),
        });
        std::fs::write(&path, serde_json::to_string_pretty(&doc)?)?;
        println!("wrote {path}");
    }
    if !shapes.iter().any(|s| s.id == SANITY_SHAPE) {
        gates.skip(
            "output sanity",
            "the out_proj shape was not selected (add it to --shapes)",
        );
    }
    gates.finish()
}
