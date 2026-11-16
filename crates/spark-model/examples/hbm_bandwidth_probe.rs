// SPDX-License-Identifier: AGPL-3.0-only
//! Measured memory-bandwidth ceiling for the card: sustained read, write and
//! copy GB/s, from buffers far larger than the L2.
//!
//! Atlas reports used to divide by the 288 GB/s spec figure. A real kernel
//! never reaches the spec, and a hot-L2 microtest exceeds it (the guide records
//! 927 GB/s for the 4B down projection). Every "percent of peak" in the L07
//! examples and in the later kernel lanes is a percent of the number this probe
//! prints on its `CEILING` line.
//!
//! Method credit: Cinference `tools/hbm_bandwidth_probe.cu` (a sustained
//! read/write/copy probe), re-written for Atlas. The kernels are three small PTX
//! functions (`common/hbm_ptx.rs`) loaded with `cuModuleLoadData`, so the probe
//! needs no new `.cu` file and no change to the kernel build.
//!
//! What it measures, each as the median of five rotated rounds (3 warm-up calls,
//! 20 timed calls per sample, CUDA events), cold except where marked:
//!  * `read_cold`   PTX streaming read (four 16-byte loads in flight per thread)
//!  * `write_cold`  PTX streaming write
//!  * `copy_cold`   PTX read plus write (GB/s counts both directions)
//!  * `memset_cold` `cuMemsetD8Async`, the library's write path (cross-check)
//!  * `dtod_cold`   `cuMemcpyDtoDAsync`, the library's copy path (cross-check)
//!  * `read_hot`    the same read kernel over an L2-resident region, repeated
//!                  inside one launch: the contrast that shows the cache number
//!
//! Gates (each with a planted control; a control that passes fails the run):
//!  * the cold ring is accepted and an L2-resident one is refused;
//!  * the cold read is within the spec sheet (the hot number is the control);
//!  * the read is repeatable across rounds (a doubled sample is the control);
//!  * the PTX write agrees with `cuMemsetD8Async` (the hot read is the control).
//!
//! Run on the Ada card (needs no model):
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example hbm_bandwidth_probe --features cuda,gpu-examples \
//!       -- [--mib 1024] [--spec-gbs 288] [--json out.json] [--profile-measured]
use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::GpuBackend;

#[path = "common/bench_args.rs"]
mod bench_args;
#[path = "common/cold_l2.rs"]
mod cold_l2;
#[path = "common/hbm_ptx.rs"]
mod hbm_ptx;
#[path = "common/planted.rs"]
mod planted;
#[path = "common/profiler.rs"]
mod profiler;

use bench_args::{arg_parse, arg_value, bench_cfg};
use cold_l2::{Arm, ArmResult, bench_arms, gbs, guard_cold};
use hbm_ptx::{Geom, Probe};
use planted::{Gates, Verdict};

/// Blocks per SM and threads per block of the probe launches.
const BLOCKS_PER_SM: u32 = 8;
const BLOCK: u32 = 256;
/// The cold read may not exceed the spec sheet by more than this factor, and
/// must reach at least [`MIN_SPEC_FRACTION`] of it.
const SPEC_SLACK: f64 = 1.03;
const MIN_SPEC_FRACTION: f64 = 0.50;

/// Largest whole number of trips not above `bytes`, and at least one trip.
fn round_to_trips(bytes: usize, trip: usize) -> usize {
    (bytes / trip).max(1) * trip
}

/// Bytes crossing the memory interface per call, for each probe kind.
fn traffic_bytes(kind: &str, bytes: usize, reps: u32) -> f64 {
    let b = bytes as f64;
    match kind {
        "copy_cold" | "dtod_cold" => 2.0 * b,
        "read_hot" => b * f64::from(reps),
        _ => b,
    }
}

fn find<'a>(rs: &'a [ArmResult], name: &str) -> Result<&'a ArmResult> {
    rs.iter()
        .find(|r| r.name == name)
        .ok_or_else(|| anyhow::anyhow!("no result named {name}"))
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
    let sms = if info.sm_count > 0 { info.sm_count } else { 34 };
    let spec: Option<f64> = arg_value("--spec-gbs")
        .map(|s| s.parse::<f64>())
        .transpose()?
        .or_else(|| info.spec_gbs());
    let geom = Geom {
        grid: sms * BLOCKS_PER_SM,
        block: BLOCK,
    };
    let want_mib: usize = arg_parse("--mib", 1024usize)?;
    let bytes = round_to_trips(want_mib << 20, geom.trip_bytes());
    let hot_bytes = round_to_trips(l2 / 4, geom.trip_bytes());
    ensure!(
        hot_bytes <= l2 / 2,
        "one probe trip ({} MiB) does not fit in half the {} MiB L2",
        hot_bytes >> 20,
        l2 >> 20
    );
    let hot_reps = (bytes / hot_bytes).max(1) as u32;
    println!(
        "device: {} SMs, L2 {} MiB, cc {}.{}, spec {} GB/s | buffers {} MiB, hot region {:.1} MiB x {} reps",
        info.sm_count,
        l2 >> 20,
        info.cc_major,
        info.cc_minor,
        spec.map_or("unknown".to_string(), |s| format!("{s:.0}")),
        bytes >> 20,
        hot_bytes as f64 / 1_048_576.0,
        hot_reps
    );
    ensure!(
        gpu.free_memory()? > 2 * bytes + (512 << 20),
        "not enough free device memory for two {} MiB buffers",
        bytes >> 20
    );
    let probe = Probe::load()?;
    let (src, dst, sink) = (gpu.alloc(bytes)?, gpu.alloc(bytes)?, gpu.alloc(256)?);
    gpu.memset(src, 0x3C, bytes)?;
    gpu.memset(dst, 0x00, bytes)?;
    gpu.memset(sink, 0x00, 256)?;
    let (n_vec, n_hot) = ((bytes / 16) as u64, (hot_bytes / 16) as u64);

    let cfg = bench_cfg(stream)?;
    let mut arms = vec![
        Arm::new("read_cold", || {
            probe.read(geom, stream, src, n_vec, 1, sink)
        }),
        Arm::new("write_cold", || probe.write(geom, stream, dst, n_vec, 1)),
        Arm::new("copy_cold", || probe.copy(geom, stream, src, dst, n_vec, 1)),
        Arm::new("memset_cold", || gpu.memset_async(dst, 0x5A, bytes, stream)),
        Arm::new("dtod_cold", || gpu.copy_d2d_async(src, dst, bytes, stream)),
        Arm::new("read_hot", || {
            probe.read(geom, stream, src, n_hot, hot_reps, sink)
        }),
    ];
    let results = bench_arms(gpu, &mut arms, &cfg)?;
    drop(arms);

    println!(
        "{:<12} {:>12} {:>10} {:>8} {:>9} {:>8}",
        "probe", "MiB/call", "median us", "spread%", "GB/s", "%spec"
    );
    let mut gb = std::collections::BTreeMap::new();
    for r in &results {
        let t = if r.name == "read_hot" {
            traffic_bytes(&r.name, hot_bytes, hot_reps)
        } else {
            traffic_bytes(&r.name, bytes, 1)
        };
        let g = gbs(t, r.median_us);
        gb.insert(r.name.clone(), g);
        let pct = spec.map_or("n/a".to_string(), |s| format!("{:.0}", 100.0 * g / s));
        let temp = if r.name == "read_hot" {
            "HOT: L2-resident, not a bandwidth number"
        } else {
            "cold"
        };
        println!(
            "{:<12} {:>12.0} {:>10.1} {:>8.1} {:>9.1} {:>8}   {}",
            r.name,
            t / 1_048_576.0,
            r.median_us,
            r.spread_pct,
            g,
            pct,
            temp
        );
    }
    let (read, write, copy) = (gb["read_cold"], gb["write_cold"], gb["copy_cold"]);
    println!(
        "CEILING read_gbs={read:.1} write_gbs={write:.1} copy_gbs={copy:.1} spec_gbs={}",
        spec.map_or("unknown".to_string(), |s| format!("{s:.1}"))
    );
    if let Some(path) = arg_value("--json") {
        let rows: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "probe": r.name, "median_us": r.median_us, "spread_pct": r.spread_pct,
                    "samples_us": r.samples_us, "gbs": gb[&r.name],
                })
            })
            .collect();
        let doc = serde_json::json!({
            "tool": "hbm_bandwidth_probe", "l2_bytes": l2, "buffer_bytes": bytes,
            "read_gbs": read, "write_gbs": write, "copy_gbs": copy, "spec_gbs": spec,
            "rows": rows,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&doc)?)?;
        println!("wrote {path}");
    }

    let spread_max: f64 = arg_parse("--spread-max", 5.0)?;
    let mut gates = Gates::new("hbm_bandwidth_probe");
    let cold = cold_l2::RingPlan {
        copies: 1,
        per_copy_bytes: bytes,
        l2_bytes: l2,
    };
    let hot = cold_l2::plan_hot(hot_bytes, l2);
    gates.gate("cold ring accepted", &["l2_resident_ring"], |v| {
        let plan = if v.is_some() { &hot } else { &cold };
        Ok(Verdict::new(
            guard_cold(plan).is_ok(),
            format!(
                "{} MiB read per call, L2 {} MiB",
                plan.total_bytes() >> 20,
                l2 >> 20
            ),
        ))
    });
    match spec {
        Some(s) => {
            let hot_gbs = gb["read_hot"];
            gates.gate(
                "cold read within the spec sheet",
                &["hot_read_reported_as_cold"],
                |v| {
                    let g = if v.is_some() { hot_gbs } else { read };
                    Ok(Verdict::new(
                        g <= s * SPEC_SLACK && g >= s * MIN_SPEC_FRACTION,
                        format!(
                            "{g:.0} GB/s against spec {s:.0} (allowed {:.0}% to {:.0}%)",
                            MIN_SPEC_FRACTION * 100.0,
                            SPEC_SLACK * 100.0
                        ),
                    ))
                },
            );
        }
        None => gates.skip(
            "cold read within the spec sheet",
            "no spec figure (pass --spec-gbs)",
        ),
    }
    let read_samples = find(&results, "read_cold")?.samples_us.clone();
    gates.gate("read repeatable across rounds", &["doubled_sample"], |v| {
        let mut s = read_samples.clone();
        if v.is_some() {
            s[0] *= 2.0;
        }
        let sp = cold_l2::spread_pct(&s);
        Ok(Verdict::new(
            sp <= spread_max,
            format!("spread {sp:.1}% (max {spread_max}%)"),
        ))
    });
    let memset_gbs = gb["memset_cold"];
    let hot_gbs = gb["read_hot"];
    gates.gate(
        "PTX write agrees with cuMemsetD8Async",
        &["hot_read_as_library_figure"],
        |v| {
            let lib = if v.is_some() { hot_gbs } else { memset_gbs };
            let ratio = write / lib;
            Ok(Verdict::new(
                (0.75..=1.25).contains(&ratio),
                format!("PTX write {write:.0} GB/s vs library {lib:.0} GB/s (ratio {ratio:.2})"),
            ))
        },
    );
    for p in [src, dst, sink] {
        gpu.free(p)?;
    }
    gates.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trips_round_down_but_never_to_zero() {
        let g = Geom {
            grid: 272,
            block: 256,
        };
        let trip = g.trip_bytes();
        assert_eq!(round_to_trips(1 << 30, trip) % trip, 0);
        assert!(round_to_trips(1 << 30, trip) <= 1 << 30);
        assert_eq!(round_to_trips(10, trip), trip);
    }

    #[test]
    fn copy_counts_both_directions() {
        assert_eq!(traffic_bytes("read_cold", 100, 1), 100.0);
        assert_eq!(traffic_bytes("copy_cold", 100, 1), 200.0);
        assert_eq!(traffic_bytes("dtod_cold", 100, 1), 200.0);
        assert_eq!(traffic_bytes("read_hot", 100, 7), 700.0);
    }

    #[test]
    fn the_hot_region_fits_in_a_quarter_to_half_of_the_l2() {
        let g = Geom {
            grid: 34 * BLOCKS_PER_SM,
            block: BLOCK,
        };
        let l2 = 32usize << 20;
        let hot = round_to_trips(l2 / 4, g.trip_bytes());
        assert!(hot <= l2 / 2 && hot > 0);
    }
}
