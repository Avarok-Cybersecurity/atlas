// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle for the port sprint's `GpuBackend` extensions (lane L08, contract 3.7.1) on
//! the real driver: graph templates, topology signatures, in-place exec update, graph
//! upload, batched async copies, event polling and host registration. The CPU tests in
//! `spark-runtime` cover the trait defaults and the mock; this is the receipt that the
//! driver entry points resolved through `cuGetProcAddress` have the layouts and ABIs the
//! Rust side assumes.
//!
//!   AVAROK_TARGET_HW=ada cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example gpu_ext_selftest
//!
//! Gates (each prints PASS or FAIL; exit code 1 on any FAIL):
//!
//! * G1 a template's exec replays (memset + D2D capture, bytes read back);
//! * G2 the topology signature is stable across two captures of one sequence and differs
//!   for another sequence; with the `bf16_add` module present, a kernel node is hashed
//!   (kernel versus memcpy at the same position differ);
//! * G3 `graph_exec_update` succeeds for the same topology and the exec then replays the
//!   NEW values; it refuses a different topology; a fresh instantiate replays that one;
//! * G4 batched H2D scatter + D2H gather of 64 x 32 KiB entries round-trips byte for
//!   byte (both through `cuMemcpyBatchAsync` when `supports_copy_batch()`, else the
//!   per-copy fallback), with the per-copy primitives timed beside it (information);
//! * G5 `event_query` turns true once the event completed, and a poll loop terminates;
//!   two timing events around a D2D report a positive elapsed time without blocking;
//! * G6 `host_register` page-locks a caller buffer, `pinned_hosts` reports it tier-owned,
//!   a batch with `src_any_order` round-trips through it, and `host_unregister` undoes it.
//!
//! Every byte gate also runs a planted control (an expectation that is deliberately
//! wrong) that must be rejected; a control that matches fails the run.

use std::time::Instant;

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{CopyEntry, DevicePtr, GpuBackend, GraphTemplate, KernelArg};

struct Gates {
    failed: usize,
}

impl Gates {
    fn check(&mut self, name: &str, ok: bool, detail: &str) {
        println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            self.failed += 1;
        }
    }

    /// `wrong_matches`: did a deliberately wrong expectation match? It must not.
    fn control(&mut self, name: &str, wrong_matches: bool) {
        let detail = if wrong_matches {
            "a wrong expectation matched, so the comparison cannot fail"
        } else {
            "a wrong expectation was rejected"
        };
        self.check(&format!("{name} planted control"), !wrong_matches, detail);
    }
}

fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn all_eq(v: &[u8], b: u8) -> bool {
    v.iter().all(|&x| x == b)
}

/// One capture on `s`: a memset per value into `buf`, then a D2D of `buf` into `out`.
fn capture_fill(
    g: &dyn GpuBackend,
    s: u64,
    buf: DevicePtr,
    out: DevicePtr,
    n: usize,
    vals: &[u8],
) -> Result<()> {
    g.begin_capture(s)?;
    let body = (|| {
        for &v in vals {
            g.memset_async(buf, v, n, s)?;
        }
        g.copy_d2d_async(buf, out, n, s)
    })();
    if let Err(e) = body {
        g.abort_capture_if_active(s);
        return Err(e);
    }
    Ok(())
}

fn replay(g: &dyn GpuBackend, exec: spark_runtime::gpu::GraphHandle, s: u64) -> Result<()> {
    g.launch_graph(exec, s)?;
    g.synchronize(s)
}

fn gate_graphs(g: &dyn GpuBackend, gates: &mut Gates) -> Result<()> {
    const N: usize = 1 << 16;
    let s = g.create_stream()?;
    let buf = g.alloc(N)?;
    let out = g.alloc(N)?;

    capture_fill(g, s, buf, out, N, &[0x11])?;
    let (exec, t1) = g.end_capture_template(s)?;
    g.graph_upload(exec, s)?;
    replay(g, exec, s)?;
    let got = read(g, out, N)?;
    gates.check(
        "G1 template exec replays",
        all_eq(&got, 0x11),
        "64 KiB of 0x11",
    );
    gates.control("G1", all_eq(&got, 0x12));

    if !g.supports_graph_exec_update() {
        gates.check(
            "G2/G3 exec update",
            false,
            "supports_graph_exec_update() is false",
        );
        g.graph_template_destroy(t1)?;
        g.destroy_graph(exec)?;
        g.free(buf)?;
        return g.free(out);
    }
    let sig1 = g.graph_topology_signature(&t1)?;
    capture_fill(g, s, buf, out, N, &[0x22])?;
    let t2 = g.end_capture_graph(s)?;
    let sig2 = g.graph_topology_signature(&t2)?;
    gates.check(
        "G2 signature stable",
        sig1 == sig2,
        &format!("{sig1:#018x} vs {sig2:#018x}"),
    );
    capture_fill(g, s, buf, out, N, &[0x33, 0x44])?;
    let t3 = g.end_capture_graph(s)?;
    let sig3 = g.graph_topology_signature(&t3)?;
    gates.check(
        "G2 signature separates topologies",
        sig3 != sig1,
        "extra memset node",
    );
    gate_kernel_node(g, s, buf, out, gates)?;

    let updated = g.graph_exec_update(exec, &t2)?;
    gates.check(
        "G3 update, same topology",
        updated,
        "cuGraphExecUpdate accepted",
    );
    replay(g, exec, s)?;
    let got = read(g, out, N)?;
    gates.check(
        "G3 updated exec replays new values",
        all_eq(&got, 0x22),
        "0x22",
    );
    gates.control("G3", all_eq(&got, 0x11));
    let refused = !g.graph_exec_update(exec, &t3)?;
    gates.check("G3 update refused, new topology", refused, "Ok(false)");
    // A refused update leaves the exec stale by contract: rebuild, as a cache would.
    g.destroy_graph(exec)?;
    let exec3 = g.graph_instantiate(&t3)?;
    replay(g, exec3, s)?;
    let got = read(g, out, N)?;
    gates.check(
        "G3 fresh instantiate replays",
        all_eq(&got, 0x44),
        "0x44 (last memset)",
    );
    for t in [t1, t2, t3] {
        g.graph_template_destroy(t)?;
    }
    g.destroy_graph(exec3)?;
    g.free(buf)?;
    g.free(out)
}

/// With the `bf16_add` module present: a kernel node hashes differently from a memcpy
/// node in the same position, and two captures with the kernel agree.
fn gate_kernel_node(
    g: &dyn GpuBackend,
    s: u64,
    a: DevicePtr,
    b: DevicePtr,
    gates: &mut Gates,
) -> Result<()> {
    if !g.has_module("bf16_add") {
        println!("SKIP G2 kernel node: module bf16_add not in this build");
        return Ok(());
    }
    let k = g.kernel("bf16_add", "bf16_add_inplace")?;
    let n: i32 = 1024;
    let n_bytes = n.to_le_bytes();
    let with_kernel = |g: &dyn GpuBackend| -> Result<GraphTemplate> {
        g.begin_capture(s)?;
        let body = (|| {
            g.memset_async(a, 0, 2 * n as usize, s)?;
            let args = [
                KernelArg::Buffer(a),
                KernelArg::Buffer(b),
                KernelArg::Bytes(&n_bytes),
            ];
            g.launch_typed(
                k,
                [(n as u32).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                s,
                &args,
            )
        })();
        if let Err(e) = body {
            g.abort_capture_if_active(s);
            return Err(e);
        }
        g.end_capture_graph(s)
    };
    let k1 = with_kernel(g)?;
    let k2 = with_kernel(g)?;
    capture_fill(g, s, a, b, 2 * n as usize, &[0])?;
    let m = g.end_capture_graph(s)?;
    let (sk1, sk2, sm) = (
        g.graph_topology_signature(&k1)?,
        g.graph_topology_signature(&k2)?,
        g.graph_topology_signature(&m)?,
    );
    gates.check(
        "G2 kernel node stable",
        sk1 == sk2,
        "two captures with bf16_add_inplace",
    );
    gates.check(
        "G2 kernel node hashed",
        sk1 != sm,
        "kernel versus memcpy node",
    );
    for t in [k1, k2, m] {
        g.graph_template_destroy(t)?;
    }
    Ok(())
}

fn gate_batch(g: &dyn GpuBackend, gates: &mut Gates) -> Result<()> {
    const E: usize = 64;
    const B: usize = 32 * 1024;
    let s = g.create_copy_stream()?;
    let dev = g.alloc(2 * E * B)?;
    let up = g.alloc_host_pinned(E * B)?;
    let down = g.alloc_host_pinned(E * B)?;
    // SAFETY: two fresh page-locked regions of E * B bytes each, owned by this function.
    let (up_s, down_s) = unsafe {
        (
            std::slice::from_raw_parts_mut(up, E * B),
            std::slice::from_raw_parts_mut(down, E * B),
        )
    };
    for (i, x) in up_s.iter_mut().enumerate() {
        *x = (i.wrapping_mul(2_654_435_761) >> 13) as u8;
    }
    // Device slots permuted and spaced so a src/dst or offset mix-up cannot pass.
    let slot = |i: usize| dev.offset(((i * 37) % E) * 2 * B + B / 2);
    let entries = |host: *mut u8| -> Vec<CopyEntry> {
        (0..E)
            .map(|i| CopyEntry {
                dev: slot(i),
                host: host as usize + i * B,
                bytes: B,
            })
            .collect()
    };
    let (scatter, gather) = (entries(up), entries(down));
    let batched = g.supports_copy_batch();
    let t0 = Instant::now();
    g.copy_batch_h2d_async(&scatter, s, false)?;
    g.copy_batch_d2h_async(&gather, s)?;
    g.synchronize(s)?;
    let batch_us = t0.elapsed().as_micros();
    let path = if batched {
        "cuMemcpyBatchAsync"
    } else {
        "per-copy fallback"
    };
    let same = down_s[..] == up_s[..];
    gates.check(
        "G4 batched round trip",
        same,
        &format!("{E} x {B} B via {path}"),
    );
    gates.control("G4", down_s[B..] == up_s[..(E - 1) * B]);

    down_s.fill(0);
    let t0 = Instant::now();
    for e in &scatter {
        // SAFETY: as above; `up` stays alive and unchanged until the sync below.
        let src = unsafe { std::slice::from_raw_parts(e.host as *const u8, e.bytes) };
        g.copy_h2d_async_retained(src, e.dev, s)?;
    }
    for e in &gather {
        // SAFETY: `down` is not read until the sync below.
        let dst = unsafe { std::slice::from_raw_parts_mut(e.host as *mut u8, e.bytes) };
        g.copy_d2h_async(e.dev, dst, s)?;
    }
    g.synchronize(s)?;
    let loop_us = t0.elapsed().as_micros();
    let same = down_s[..] == up_s[..];
    gates.check(
        "G4 per-copy round trip",
        same,
        "same entries, single copies",
    );
    println!(
        "info G4: one cold round trip of {} MiB: batch path {batch_us} us, per-copy {loop_us} us \
         (bandwidth receipts come from pinned_probe, queue Q5)",
        (2 * E * B) >> 20
    );
    g.free_host_pinned(up, E * B)?;
    g.free_host_pinned(down, E * B)?;
    g.free(dev)
}

fn gate_events(g: &dyn GpuBackend, gates: &mut Gates) -> Result<()> {
    const BIG: usize = 128 << 20;
    let s = g.create_stream()?;
    let (a, b) = (g.alloc(BIG)?, g.alloc(BIG)?);
    let ev = g.create_event()?;
    g.copy_d2d_async(a, b, BIG, s)?;
    g.record_event(ev, s)?;
    let early = g.event_query(ev)?;
    let t0 = Instant::now();
    let mut polls = 0u64;
    while !g.event_query(ev)? {
        polls += 1;
        if t0.elapsed().as_secs() > 10 {
            bail!("event_query never turned true within 10 s");
        }
        std::hint::spin_loop();
    }
    gates.check(
        "G5 poll loop terminates",
        true,
        &format!("{polls} polls after enqueue"),
    );
    g.event_synchronize(ev)?;
    gates.check(
        "G5 true after completion",
        g.event_query(ev)?,
        "event_synchronize then query",
    );
    println!("info G5: query right after a 128 MiB D2D enqueue returned {early}");

    // Timing events (L06's device phase timer): elapsed is None or Some, never a wait.
    let (t0e, t1e) = (g.create_timing_event()?, g.create_timing_event()?);
    g.record_event(t0e, s)?;
    g.copy_d2d_async(b, a, BIG, s)?;
    g.record_event(t1e, s)?;
    let t0 = Instant::now();
    let ms = loop {
        if let Some(ms) = g.event_elapsed_ms(t0e, t1e)? {
            break ms;
        }
        if t0.elapsed().as_secs() > 10 {
            bail!("event_elapsed_ms stayed None for 10 s");
        }
        std::hint::spin_loop();
    };
    let detail = format!("{ms:.3} ms for a 128 MiB D2D");
    gates.check(
        "G5 timing events measure",
        ms > 0.0 && ms.is_finite(),
        &detail,
    );
    g.destroy_event(t0e)?;
    g.destroy_event(t1e)?;
    g.destroy_event(ev)?;
    g.free(a)?;
    g.free(b)
}

fn gate_host_register(g: &dyn GpuBackend, gates: &mut Gates) -> Result<()> {
    const BYTES: usize = 8 << 20;
    let layout = std::alloc::Layout::from_size_align(BYTES, 4096)?;
    // SAFETY: non-zero size; freed with the same layout below.
    let p = unsafe { std::alloc::alloc_zeroed(layout) };
    if p.is_null() {
        bail!("host allocation of {BYTES} bytes failed");
    }
    let result = register_round_trip(g, p, BYTES, gates);
    // SAFETY: allocated above with `layout`; unregistered (or never registered) by now.
    unsafe { std::alloc::dealloc(p, layout) };
    result
}

fn register_round_trip(
    g: &dyn GpuBackend,
    p: *mut u8,
    bytes: usize,
    gates: &mut Gates,
) -> Result<()> {
    if let Err(e) = g.host_register(p as usize, bytes) {
        gates.check("G6 host_register", false, &format!("{e:#}"));
        return Ok(());
    }
    // SAFETY: `p` is a live allocation of `bytes` bytes owned by the caller.
    let all = unsafe { std::slice::from_raw_parts_mut(p, bytes) };
    let tier = spark_runtime::pinned_hosts::is_tier_owned(all);
    gates.check(
        "G6 registry marks it tier-owned",
        tier,
        "pinned_hosts::is_tier_owned",
    );
    let half = bytes / 2;
    for (i, x) in all[..half].iter_mut().enumerate() {
        *x = (i % 251) as u8;
    }
    let s = g.create_copy_stream()?;
    let dev = g.alloc(half)?;
    let base = p as usize;
    let up = [CopyEntry {
        dev,
        host: base,
        bytes: half,
    }];
    let down = [CopyEntry {
        dev,
        host: base + half,
        bytes: half,
    }];
    g.copy_batch_h2d_async(&up, s, true)?;
    g.copy_batch_d2h_async(&down, s)?;
    g.synchronize(s)?;
    let (lo, hi) = all.split_at(half);
    gates.check(
        "G6 round trip through a registered buffer",
        lo == hi,
        "src_any_order H2D",
    );
    gates.control("G6", lo[1..] == hi[..half - 1]);
    g.free(dev)?;
    g.host_unregister(base)?;
    let gone = !spark_runtime::pinned_hosts::is_pinned(all);
    gates.check(
        "G6 host_unregister clears the registry",
        gone,
        "pinned_hosts::is_pinned",
    );
    Ok(())
}

fn main() -> Result<()> {
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut gates = Gates { failed: 0 };
    println!(
        "gpu_ext_selftest: supports_graph_exec_update={} supports_copy_batch={}",
        g.supports_graph_exec_update(),
        g.supports_copy_batch()
    );
    gate_graphs(g, &mut gates)?;
    gate_batch(g, &mut gates)?;
    gate_events(g, &mut gates)?;
    gate_host_register(g, &mut gates)?;
    if gates.failed > 0 {
        println!("gpu_ext_selftest: {} gate(s) FAILED", gates.failed);
        std::process::exit(1);
    }
    println!("gpu_ext_selftest: all gates PASS");
    Ok(())
}
