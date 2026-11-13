// SPDX-License-Identifier: AGPL-3.0-only
//! Pinned host memory and host-device copy rates under WSL2: queue row Q5, the
//! viability gate for the host-RAM KV tier (L25, L26a, L26b, L27).
//!
//! Two measurements.
//!  1. Pinning. 1, 4 and 16 GiB pinned as one call and as 64 MiB chunks, by
//!     `cuMemAllocHost` and by `cuMemHostRegister`, each arm in a FRESH PROCESS
//!     from a cold context (vLLM's `benchmark_pin_memory.py` pattern), so the
//!     first pin never rides on a warm context or pages already faulted in. WSL2
//!     may refuse large pins; a refusal is a result (`rc` and its name), and a size
//!     that does not fit `MemAvailable` with a 4 GiB margin is skipped, not tried.
//!  2. Copy rates. H2D and D2H GB/s from pinned memory at 32 KiB x N descriptors
//!     for each method: one big copy, an N-call loop, and `copy_batch_*` (L08's
//!     `cuMemcpyBatchAsync`, only when `supports_copy_batch()`), plus pinned
//!     against registered against pageable sources. Rotated rounds, CUDA events.
//!
//! Gates (planted controls, see `common/planted.rs`): every method round-trips
//! 32 MiB byte for byte (a corrupted byte is the control), and the result lines
//! the children print parse (a garbage line is the control).
//!
//! Method credit: LMCache brief P (pin 1, 4, 16 GiB one-shot against 64 MB
//! chunks, whether WSL2 refuses) and the vLLM `cuMemcpyBatchAsync` path.
//!
//! Run on the Ada card (needs no model):
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example pinned_probe --features cuda,gpu-examples -- \
//!       [--gib 1,4,16] [--timeout-s 180] [--skip-pin] [--skip-copy] [--json out.json]
use anyhow::{Result, anyhow, bail};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{CopyEntry, DevicePtr, GpuBackend};
use std::time::Duration;

#[path = "common/bench_args.rs"]
mod bench_args;
#[path = "common/cold_l2.rs"]
mod cold_l2;
#[path = "common/pinned_arms.rs"]
mod pinned_arms;
#[path = "common/planted.rs"]
mod planted;
#[path = "common/profiler.rs"]
mod profiler;
#[path = "common/weight_fill.rs"]
mod weight_fill;

use bench_args::{arg_parse, arg_value, bench_cfg, has_flag};
use cold_l2::{Arm, bench_arms, gbs};
use pinned_arms::{PinArm, PinOutcome, format_line, parse_line, run_arm};
use planted::{Gates, Verdict};

/// One descriptor, the size of a KV block's K or V slab at 4B.
const DESC: usize = 32 << 10;
/// Descriptor counts: 2 MiB, 8 MiB, 32 MiB and 128 MiB per call.
const COUNTS: [usize; 4] = [64, 256, 1024, 4096];
/// The host-tier viability bar of the plan: the row stays off below this.
const VIABLE_GBS: f64 = 6.0;
const GIB: f64 = 1_073_741_824.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    Single,
    Loop,
    Batch,
    BatchAnyOrder,
}

/// `n` descriptors of `d` bytes over `dev` and `host`, optionally reversed (the
/// "sources in any order" case).
fn entries(dev: DevicePtr, host: usize, n: usize, d: usize, reversed: bool) -> Vec<CopyEntry> {
    let mut v: Vec<CopyEntry> = (0..n)
        .map(|i| CopyEntry {
            dev: dev.offset(i * d),
            host: host + i * d,
            bytes: d,
        })
        .collect();
    if reversed {
        v.reverse();
    }
    v
}

/// SAFETY contract: `addr` is a live pinned region of at least `len` bytes that
/// no other reference aliases for the slice's lifetime.
unsafe fn host_slice<'a>(addr: usize, len: usize) -> &'a mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, len) }
}

fn issue_h2d(
    gpu: &dyn GpuBackend,
    s: u64,
    m: Method,
    host: usize,
    dev: DevicePtr,
    n: usize,
) -> Result<()> {
    // SAFETY: `host` is a pinned staging region of n * DESC bytes owned by main.
    let all = unsafe { host_slice(host, n * DESC) };
    match m {
        Method::Single => gpu.copy_h2d_async_retained(all, dev, s),
        Method::Loop => (0..n).try_for_each(|i| {
            gpu.copy_h2d_async_retained(&all[i * DESC..(i + 1) * DESC], dev.offset(i * DESC), s)
        }),
        Method::Batch => gpu.copy_batch_h2d_async(&entries(dev, host, n, DESC, false), s, false),
        Method::BatchAnyOrder => {
            gpu.copy_batch_h2d_async(&entries(dev, host, n, DESC, true), s, true)
        }
    }
}

fn issue_d2h(
    gpu: &dyn GpuBackend,
    s: u64,
    m: Method,
    host: usize,
    dev: DevicePtr,
    n: usize,
) -> Result<()> {
    // SAFETY: as in `issue_h2d`.
    let all = unsafe { host_slice(host, n * DESC) };
    match m {
        Method::Single => gpu.copy_d2h_async(dev, all, s),
        Method::Loop => all
            .chunks_mut(DESC)
            .enumerate()
            .try_for_each(|(i, c)| gpu.copy_d2h_async(dev.offset(i * DESC), c, s)),
        Method::Batch => gpu.copy_batch_d2h_async(&entries(dev, host, n, DESC, false), s),
        Method::BatchAnyOrder => gpu.copy_batch_d2h_async(&entries(dev, host, n, DESC, true), s),
    }
}

/// Copy `n * DESC` bytes host A -> device -> host B with the named methods and
/// report whether B equals A. `corrupt` flips one byte of B (the planted control).
#[allow(clippy::too_many_arguments)]
fn roundtrip(
    gpu: &dyn GpuBackend,
    s: u64,
    m: Method,
    a: usize,
    b: usize,
    dev: DevicePtr,
    n: usize,
    corrupt: bool,
) -> Result<bool> {
    let total = n * DESC;
    {
        // SAFETY: both regions are pinned staging buffers of at least `total`
        // bytes; no other slice over them is live in this block.
        let (ha, hb) = unsafe { (host_slice(a, total), host_slice(b, total)) };
        ha.copy_from_slice(&weight_fill::pattern(
            weight_fill::Fill::Bytes,
            total,
            0x7157,
        ));
        hb.fill(0);
    }
    gpu.memset(dev, 0, total)?;
    issue_h2d(gpu, s, m, a, dev, n)?;
    gpu.synchronize(s)?;
    issue_d2h(gpu, s, m, b, dev, n)?;
    gpu.synchronize(s)?;
    // SAFETY: as above; the copies have completed, so nothing writes the regions.
    let (ha, hb) = unsafe { (host_slice(a, total), host_slice(b, total)) };
    if corrupt {
        hb[total / 2] ^= 0xFF;
    }
    Ok(ha == hb)
}

fn pin_phase(gates: &mut Gates) -> Result<Vec<PinOutcome>> {
    let gibs: Vec<f64> = arg_value("--gib")
        .unwrap_or_else(|| "1,4,16".into())
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|e| anyhow!("--gib {s:?}: {e}"))
        })
        .collect::<Result<_>>()?;
    let timeout = Duration::from_secs(arg_parse("--timeout-s", 180u64)?);
    let plans = pinned_arms::plan_sizes(pinned_arms::mem_available_kib(), &gibs, 4.0);
    println!(
        "{:<17} {:>7} {:>4} {:>11} {:>9} {:>5}  note",
        "pin arm", "GiB", "ok", "wall_ms", "GiB/s", "rc"
    );
    let mut out = Vec::new();
    for p in &plans {
        for arm in PinArm::ALL {
            if !p.run {
                println!("{:<17} {:>7.1} skipped: {}", arm.name(), p.gib, p.reason);
                continue;
            }
            let o = pinned_arms::run_child(arm, p.gib, timeout);
            println!(
                "{:<17} {:>7.1} {:>4} {:>11.1} {:>9} {:>5}  {}",
                o.arm,
                o.gib,
                u8::from(o.ok),
                o.wall_ms,
                o.gib_per_s().map_or("-".to_string(), |g| format!("{g:.2}")),
                o.rc,
                o.note
            );
            out.push(o);
        }
    }
    let line = format_line(&PinOutcome {
        arm: "alloc_once".into(),
        gib: 1.0,
        ok: true,
        wall_ms: 12.5,
        rc: 0,
        note: String::new(),
    });
    gates.gate("child result lines parse", &["garbage_line"], |v| {
        let text = if v.is_some() {
            "PINARM arm=alloc_once gib=x".to_string()
        } else {
            line.clone()
        };
        Ok(Verdict::new(
            parse_line(&text).is_some_and(|o| o.arm == "alloc_once"),
            "format and parse agree",
        ))
    });
    Ok(out)
}

fn copy_phase(gates: &mut Gates, json: &mut Vec<serde_json::Value>) -> Result<()> {
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_copy_stream().or_else(|_| gpu.create_stream())?;
    let batch = gpu.supports_copy_batch();
    println!("copy_batch available (cuMemcpyBatchAsync resolved): {batch}");
    let max_total = COUNTS[COUNTS.len() - 1] * DESC;
    let (a, b) = (
        gpu.alloc_host_pinned(max_total)? as usize,
        gpu.alloc_host_pinned(max_total)? as usize,
    );
    let dev = gpu.alloc(max_total)?;

    let mut methods = vec![Method::Single, Method::Loop];
    if batch {
        methods.extend([Method::Batch, Method::BatchAnyOrder]);
    }
    for m in methods.iter().copied().filter(|m| *m != Method::Single) {
        gates.gate(
            &format!("round trip {m:?} over 1024 x 32 KiB"),
            &["corrupt_byte"],
            |v| {
                let same = roundtrip(gpu, stream, m, a, b, dev, 1024, v.is_some())?;
                Ok(Verdict::new(same, "host B equals host A after H2D and D2H"))
            },
        );
    }
    if !batch {
        gates.skip(
            "round trip batch methods",
            "the backend has no cuMemcpyBatchAsync (supports_copy_batch() is false)",
        );
    }

    let cfg = bench_cfg(stream)?;
    println!(
        "{:<8} {:>6} {:>10} {:>14} {:>8} {:>9}",
        "dir", "N", "MiB/call", "method", "spr%", "GB/s"
    );
    let mut rates: Vec<(String, f64)> = Vec::new();
    for n in COUNTS {
        let bytes = n * DESC;
        let mut arms: Vec<Arm> = Vec::new();
        for m in methods.iter().copied() {
            arms.push(Arm::new(format!("h2d_{m:?}"), move || {
                issue_h2d(gpu, stream, m, a, dev, n)
            }));
            arms.push(Arm::new(format!("d2h_{m:?}"), move || {
                issue_d2h(gpu, stream, m, b, dev, n)
            }));
        }
        for r in bench_arms(gpu, &mut arms, &cfg)? {
            let g = gbs(bytes as f64, r.median_us);
            println!(
                "{:<8} {:>6} {:>10.1} {:>14} {:>8.1} {:>9.2}",
                &r.name[..3],
                n,
                bytes as f64 / 1_048_576.0,
                &r.name[4..],
                r.spread_pct,
                g
            );
            json.push(serde_json::json!({"arm": r.name, "n": n, "bytes": bytes, "median_us": r.median_us, "gbs": g}));
            rates.push((format!("{}@{n}", r.name), g));
        }
    }
    // Pinned against registered against pageable, one big H2D copy.
    let n = 1024;
    let pageable = vec![0x5Au8; n * DESC];
    let layout = std::alloc::Layout::from_size_align(n * DESC, 4096)?;
    // SAFETY: non-zero size; freed below with the same layout.
    let reg = unsafe { std::alloc::alloc_zeroed(layout) };
    let registered = gpu.host_register(reg as usize, n * DESC).is_ok();
    println!(
        "host_register on an ordinary allocation: {}",
        if registered {
            "ok"
        } else {
            "refused (arm skipped)"
        }
    );
    let mut arms: Vec<Arm> = vec![
        Arm::new("h2d_pinned", || {
            issue_h2d(gpu, stream, Method::Single, a, dev, n)
        }),
        Arm::new("h2d_pageable", || {
            gpu.copy_h2d_async_retained(&pageable, dev, stream)
        }),
    ];
    if registered {
        arms.push(Arm::new("h2d_registered", || {
            issue_h2d(gpu, stream, Method::Single, reg as usize, dev, n)
        }));
    }
    for r in bench_arms(gpu, &mut arms, &cfg)? {
        println!(
            "{:<17} one {:.0} MiB copy: {:>8.2} GB/s (spread {:.1}%)",
            r.name,
            (n * DESC) as f64 / 1_048_576.0,
            gbs((n * DESC) as f64, r.median_us),
            r.spread_pct
        );
    }
    drop(arms);
    let rate = |key: &str| {
        rates
            .iter()
            .find(|(k, _)| k == key)
            .map_or(f64::NAN, |(_, g)| *g)
    };
    let last = COUNTS[COUNTS.len() - 1];
    let (h2d, d2h) = (
        rate(&format!("h2d_Single@{last}")),
        rate(&format!("d2h_Single@{last}")),
    );
    let verdict = if h2d >= VIABLE_GBS && d2h >= VIABLE_GBS {
        "MET"
    } else {
        "NOT_MET"
    };
    println!(
        "RECEIPT pinned_h2d_single_gbs={h2d:.2} pinned_d2h_single_gbs={d2h:.2} bar={VIABLE_GBS} verdict={verdict}"
    );
    if registered {
        gpu.host_unregister(reg as usize)?;
    }
    // SAFETY: allocated above with this layout.
    unsafe { std::alloc::dealloc(reg, layout) };
    for p in [a, b] {
        gpu.free_host_pinned(p as *mut u8, max_total)?;
    }
    gpu.free(dev)
}

fn main() -> Result<()> {
    if let Some(name) = arg_value("--arm") {
        // Child mode: one pin in a cold context, one result line.
        let arm = PinArm::parse(&name).ok_or_else(|| anyhow!("unknown --arm {name}"))?;
        let gib: f64 = arg_parse("--gib", 1.0)?;
        let _context = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
        println!("{}", format_line(&run_arm(arm, (gib * GIB) as usize)));
        return Ok(());
    }
    let mut gates = Gates::new("pinned_probe");
    let mut json: Vec<serde_json::Value> = Vec::new();
    if !has_flag("--skip-pin") {
        let outcomes = pin_phase(&mut gates)?;
        json.extend(outcomes.iter().map(|o| serde_json::json!({"pin": o.arm, "gib": o.gib, "ok": o.ok, "wall_ms": o.wall_ms, "rc": o.rc, "note": o.note})));
        let big = outcomes
            .iter()
            .filter(|o| o.arm == "alloc_once" && o.ok)
            .map(|o| o.gib)
            .fold(0.0, f64::max);
        println!(
            "RECEIPT largest_one_shot_pin_gib={big:.0} refusals={}",
            outcomes.iter().filter(|o| !o.ok).count()
        );
    }
    if !has_flag("--skip-copy") {
        copy_phase(&mut gates, &mut json)?;
    }
    if let Some(path) = arg_value("--json") {
        std::fs::write(&path, serde_json::to_string_pretty(&json)?)?;
    }
    if json.is_empty() {
        bail!("nothing was run (both phases skipped)");
    }
    gates.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_contiguous_and_reversible() {
        let e = entries(DevicePtr(0x1000), 0x9000, 4, 100, false);
        assert_eq!(e.len(), 4);
        assert_eq!((e[0].dev.0, e[0].host, e[0].bytes), (0x1000, 0x9000, 100));
        assert_eq!((e[3].dev.0, e[3].host), (0x1000 + 300, 0x9000 + 300));
        let r = entries(DevicePtr(0x1000), 0x9000, 4, 100, true);
        assert_eq!((r[0].dev.0, r[3].dev.0), (0x1000 + 300, 0x1000));
        // Reversal permutes the order only: the device and host offsets stay paired.
        assert!(
            r.iter()
                .all(|x| x.host - 0x9000 == (x.dev.0 - 0x1000) as usize)
        );
    }

    #[test]
    fn counts_span_two_to_128_mib() {
        assert_eq!(COUNTS[0] * DESC, 2 << 20);
        assert_eq!(COUNTS[3] * DESC, 128 << 20);
        assert_eq!(DESC, 32 * 1024);
    }
}
