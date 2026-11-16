// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the pinning half of `pinned_probe`: four ways to page-lock
//! host memory, each run in a FRESH PROCESS from a cold context, the result
//! line they print, and the guards that keep a 16 GiB pin from taking the WSL2
//! VM down.
//!
//! Why a process per arm: vLLM's pinned-memory regression gate
//! (`benchmarks/benchmark_pin_memory.py`) runs each arm in its own process,
//! because the first pin in a process pays for context setup and a second pin
//! of the same range is served from pages the first already faulted in. The
//! measured question (as in LMCache): does pinning tens of GiB
//! block startup, and does WSL2 refuse it?
//!
//! The four arms: `alloc_once` (one `cuMemAllocHost`), `alloc_chunked` (64 MiB
//! `cuMemAllocHost` calls), `register_once` (one `cuMemHostRegister` over an
//! ordinary allocation) and `register_chunked` (64 MiB registers over one
//! allocation). Only the pin calls are timed; allocation of the backing memory
//! and cleanup are outside the clock.
#![allow(dead_code)]

use std::ffi::c_void;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

unsafe extern "C" {
    fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> i32;
    fn cuMemFreeHost(p: *mut c_void) -> i32;
    fn cuMemHostRegister_v2(p: *mut c_void, bytesize: usize, flags: u32) -> i32;
    fn cuMemHostUnregister(p: *mut c_void) -> i32;
}

/// LMCache pins 64 MB chunks (`LazyMemoryAllocator`).
pub const CHUNK_BYTES: usize = 64 << 20;
const GIB: f64 = 1_073_741_824.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinArm {
    AllocOnce,
    AllocChunked,
    RegisterOnce,
    RegisterChunked,
}

impl PinArm {
    pub const ALL: [PinArm; 4] = [
        PinArm::AllocOnce,
        PinArm::AllocChunked,
        PinArm::RegisterOnce,
        PinArm::RegisterChunked,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::AllocOnce => "alloc_once",
            Self::AllocChunked => "alloc_chunked",
            Self::RegisterOnce => "register_once",
            Self::RegisterChunked => "register_chunked",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.name() == s)
    }
}

/// The CUDA driver errors a pin can return, by name.
pub fn cu_err_name(rc: i32) -> &'static str {
    match rc {
        0 => "SUCCESS",
        1 => "INVALID_VALUE",
        2 => "OUT_OF_MEMORY",
        3 => "NOT_INITIALIZED",
        201 => "INVALID_CONTEXT",
        712 => "HOST_MEMORY_ALREADY_REGISTERED",
        713 => "HOST_MEMORY_NOT_REGISTERED",
        801 => "NOT_SUPPORTED",
        _ => "OTHER",
    }
}

/// One arm's result.
#[derive(Clone, Debug, PartialEq)]
pub struct PinOutcome {
    pub arm: String,
    pub gib: f64,
    pub ok: bool,
    pub wall_ms: f64,
    pub rc: i32,
    pub note: String,
}

impl PinOutcome {
    fn failed(arm: &str, gib: f64, note: &str) -> Self {
        Self {
            arm: arm.to_string(),
            gib,
            ok: false,
            wall_ms: 0.0,
            rc: -1,
            note: note.to_string(),
        }
    }

    /// GiB pinned per second, when the arm succeeded.
    pub fn gib_per_s(&self) -> Option<f64> {
        (self.ok && self.wall_ms > 0.0).then(|| self.gib / (self.wall_ms / 1e3))
    }
}

/// The line a child prints and the driver parses.
pub fn format_line(o: &PinOutcome) -> String {
    let note = if o.note.is_empty() {
        "-".to_string()
    } else {
        o.note.replace(' ', "_")
    };
    format!(
        "PINARM arm={} gib={:.3} ok={} wall_ms={:.3} rc={} note={}",
        o.arm,
        o.gib,
        u8::from(o.ok),
        o.wall_ms,
        o.rc,
        note
    )
}

/// Parse the first `PINARM` line in `text`.
pub fn parse_line(text: &str) -> Option<PinOutcome> {
    let line = text.lines().find(|l| l.starts_with("PINARM "))?;
    let get = |key: &str| {
        line.split_whitespace()
            .find_map(|t| t.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
    };
    let note = get("note")?;
    Some(PinOutcome {
        arm: get("arm")?.to_string(),
        gib: get("gib")?.parse().ok()?,
        ok: get("ok")? == "1",
        wall_ms: get("wall_ms")?.parse().ok()?,
        rc: get("rc")?.parse().ok()?,
        note: if note == "-" {
            String::new()
        } else {
            note.to_string()
        },
    })
}

fn time_pins(mut pin: impl FnMut() -> i32) -> (i32, f64) {
    let t0 = Instant::now();
    let rc = pin();
    (rc, t0.elapsed().as_secs_f64() * 1e3)
}

fn alloc_pins(bytes: usize, chunk: usize) -> (i32, f64) {
    let mut ptrs: Vec<*mut c_void> = Vec::new();
    let (rc, ms) = time_pins(|| {
        let mut left = bytes;
        while left > 0 {
            let n = left.min(chunk);
            let mut p: *mut c_void = std::ptr::null_mut();
            // SAFETY: valid out-pointer; a context exists (the backend made one).
            let rc = unsafe { cuMemAllocHost_v2(&mut p, n) };
            if rc != 0 {
                return rc;
            }
            ptrs.push(p);
            left -= n;
        }
        0
    });
    for p in ptrs {
        // SAFETY: each pointer came from cuMemAllocHost_v2 and is freed once.
        unsafe { cuMemFreeHost(p) };
    }
    (rc, ms)
}

fn register_pins(bytes: usize, chunk: usize) -> (i32, f64, &'static str) {
    let Ok(layout) = std::alloc::Layout::from_size_align(bytes, 4096) else {
        return (-1, 0.0, "bad_layout");
    };
    // SAFETY: non-zero size; the region is only registered, never dereferenced.
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        return (-1, 0.0, "host_alloc_failed");
    }
    let mut registered: Vec<(usize, usize)> = Vec::new();
    let (rc, ms) = time_pins(|| {
        let mut off = 0;
        while off < bytes {
            let n = (bytes - off).min(chunk);
            // SAFETY: [base+off, base+off+n) lies inside the allocation.
            let rc = unsafe { cuMemHostRegister_v2(base.add(off).cast(), n, 0) };
            if rc != 0 {
                return rc;
            }
            registered.push((off, n));
            off += n;
        }
        0
    });
    for (off, _) in registered {
        // SAFETY: registered above at exactly this address.
        unsafe { cuMemHostUnregister(base.add(off).cast()) };
    }
    // SAFETY: allocated above with this layout.
    unsafe { std::alloc::dealloc(base, layout) };
    (rc, ms, "")
}

/// Run one arm in this process. Never panics; failure is an outcome.
pub fn run_arm(arm: PinArm, bytes: usize) -> PinOutcome {
    let gib = bytes as f64 / GIB;
    let (rc, wall_ms, extra) = match arm {
        PinArm::AllocOnce => {
            let (rc, ms) = alloc_pins(bytes, bytes.max(1));
            (rc, ms, "")
        }
        PinArm::AllocChunked => {
            let (rc, ms) = alloc_pins(bytes, CHUNK_BYTES);
            (rc, ms, "")
        }
        PinArm::RegisterOnce => register_pins(bytes, bytes.max(1)),
        PinArm::RegisterChunked => register_pins(bytes, CHUNK_BYTES),
    };
    let note = if !extra.is_empty() {
        extra.to_string()
    } else if rc == 0 {
        String::new()
    } else {
        cu_err_name(rc).to_string()
    };
    PinOutcome {
        arm: arm.name().to_string(),
        gib,
        ok: rc == 0,
        wall_ms,
        rc,
        note,
    }
}

/// `MemAvailable` from `/proc/meminfo` text, in kiB.
pub fn parse_mem_available_kib(text: &str) -> Option<u64> {
    text.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

pub fn mem_available_kib() -> Option<u64> {
    parse_mem_available_kib(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// Whether a pin of `gib` may run, and why not.
#[derive(Clone, Debug, PartialEq)]
pub struct SizePlan {
    pub gib: f64,
    pub run: bool,
    pub reason: String,
}

/// A pin needs its bytes plus `margin_gib` free in `MemAvailable`; larger sizes
/// are skipped so the probe cannot push the VM into swap or the OOM killer.
pub fn plan_sizes(avail_kib: Option<u64>, gibs: &[f64], margin_gib: f64) -> Vec<SizePlan> {
    gibs.iter()
        .map(|&gib| match avail_kib {
            None => SizePlan {
                gib,
                run: true,
                reason: "MemAvailable unknown".to_string(),
            },
            Some(kib) => {
                let avail = kib as f64 / (1024.0 * 1024.0);
                let need = gib + margin_gib;
                SizePlan {
                    gib,
                    run: need <= avail,
                    reason: format!("needs {need:.1} GiB with margin, MemAvailable {avail:.1} GiB"),
                }
            }
        })
        .collect()
}

/// Run one arm in a fresh child process and parse its line. A timeout kills the
/// child; any failure to get a line is an outcome with a note, not an error.
pub fn run_child(arm: PinArm, gib: f64, timeout: Duration) -> PinOutcome {
    let name = arm.name();
    let Ok(exe) = std::env::current_exe() else {
        return PinOutcome::failed(name, gib, "no_current_exe");
    };
    let spawn = Command::new(exe)
        .args(["--arm", name, "--gib", &format!("{gib}")])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let Ok(mut child) = spawn else {
        return PinOutcome::failed(name, gib, "spawn_failed");
    };
    let start = Instant::now();
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                timed_out = true;
                break;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => break,
        }
    }
    let Ok(out) = child.wait_with_output() else {
        return PinOutcome::failed(name, gib, "wait_failed");
    };
    if timed_out {
        return PinOutcome::failed(name, gib, "timeout");
    }
    parse_line(&String::from_utf8_lossy(&out.stdout))
        .unwrap_or_else(|| PinOutcome::failed(name, gib, "no_result_line"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PinOutcome {
        PinOutcome {
            arm: "register_once".into(),
            gib: 4.0,
            ok: false,
            wall_ms: 1234.5,
            rc: 801,
            note: "NOT SUPPORTED".into(),
        }
    }

    #[test]
    fn result_lines_round_trip() {
        let line = format_line(&sample());
        assert!(
            line.starts_with("PINARM arm=register_once gib=4.000 ok=0"),
            "{line}"
        );
        let back = parse_line(&format!("noise\n{line}\ntrailer")).unwrap();
        assert_eq!(back.note, "NOT_SUPPORTED");
        assert_eq!(
            (back.arm.as_str(), back.rc, back.ok),
            ("register_once", 801, false)
        );
        assert!((back.wall_ms - 1234.5).abs() < 1e-9);
        // An empty note round-trips as empty.
        let mut ok = sample();
        ok.note.clear();
        ok.ok = true;
        assert_eq!(parse_line(&format_line(&ok)).unwrap().note, "");
    }

    #[test]
    fn garbage_is_not_a_result() {
        assert!(parse_line("").is_none());
        assert!(parse_line("PINARM arm=x").is_none());
        assert!(parse_line("PINARM arm=x gib=abc ok=1 wall_ms=1 rc=0 note=-").is_none());
        assert!(parse_line("some other PINARM line").is_none());
    }

    #[test]
    fn arm_names_round_trip() {
        for a in PinArm::ALL {
            assert_eq!(PinArm::parse(a.name()), Some(a));
        }
        assert_eq!(PinArm::parse("nope"), None);
    }

    #[test]
    fn meminfo_parses() {
        let t = "MemTotal:       65536000 kB\nMemFree: 1 kB\nMemAvailable:   33554432 kB\nBuffers: 2 kB\n";
        assert_eq!(parse_mem_available_kib(t), Some(33_554_432));
        assert_eq!(parse_mem_available_kib("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn big_pins_are_skipped_when_memory_is_short() {
        // 32 GiB available: 16 GiB + 4 GiB margin fits, 32 GiB does not.
        let plans = plan_sizes(Some(32 * 1024 * 1024), &[1.0, 16.0, 32.0], 4.0);
        assert_eq!(
            plans.iter().map(|p| p.run).collect::<Vec<_>>(),
            vec![true, true, false]
        );
        assert!(plan_sizes(None, &[64.0], 4.0)[0].run);
    }

    #[test]
    fn throughput_needs_a_success() {
        assert_eq!(sample().gib_per_s(), None);
        let mut ok = sample();
        ok.ok = true;
        ok.wall_ms = 2000.0;
        assert_eq!(ok.gib_per_s(), Some(2.0));
    }

    #[test]
    fn error_names() {
        assert_eq!(cu_err_name(801), "NOT_SUPPORTED");
        assert_eq!(cu_err_name(2), "OUT_OF_MEMORY");
        assert_eq!(cu_err_name(9999), "OTHER");
    }
}
