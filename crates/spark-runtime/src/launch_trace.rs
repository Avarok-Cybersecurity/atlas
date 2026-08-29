// SPDX-License-Identifier: AGPL-3.0-only

//! ANOMALIES A56 diagnostic: record every GPU op a step enqueues, so two
//! consecutive steps can be diffed.
//!
//! A CUDA graph BAKES the grid, block, shared-mem and argument BYTES of every
//! launch at capture time. So a captured region is replayable if and only if
//! two consecutive executions of it enqueue byte-identical ops. Anything that
//! differs between step N and step N+1 is a host value the graph froze — which
//! is exactly the class of bug where the capture pass is byte-exact but the
//! replay is not.
//!
//! This turns "which host scalar leaked into a kernel argument?" from a code
//! read into a mechanical diff. Off unless `begin()` is called.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// One enqueued op. `kind` separates kernels from memsets/copies so an op
/// appearing or vanishing shows up as a kind mismatch rather than arg noise.
#[derive(Clone, PartialEq, Eq)]
pub struct Entry {
    pub kind: &'static str,
    pub func: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
    /// Args as u64 words: buffers are the raw address, scalars are LE-packed.
    pub args: Vec<u64>,
}

static ON: AtomicBool = AtomicBool::new(false);
static TRACE: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
static PREV: Mutex<Option<Vec<Entry>>> = Mutex::new(None);
static NAMES: Mutex<Option<HashMap<u64, String>>> = Mutex::new(None);

#[inline(always)]
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Remember a kernel handle's name. Called from the backend's `kernel()`
/// lookup, which runs at init only.
pub fn name_kernel(handle: u64, module: &str, func: &str) {
    let mut g = NAMES.lock().unwrap();
    g.get_or_insert_with(HashMap::new)
        .insert(handle, format!("{module}::{func}"));
}

fn name_of(handle: u64) -> String {
    NAMES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&handle).cloned())
        .unwrap_or_else(|| format!("fn@{handle:#x}"))
}

pub fn begin() {
    TRACE.lock().unwrap().clear();
    ON.store(true, Ordering::Relaxed);
}

#[inline(always)]
pub fn record(e: Entry) {
    if on() {
        TRACE.lock().unwrap().push(e);
    }
}

/// Stop recording and diff this trace against the previous one. Returns
/// `None` on the first call (nothing to compare against yet), else a report
/// naming every op that differs.
pub fn end_and_diff(max_report: usize) -> Option<String> {
    ON.store(false, Ordering::Relaxed);
    let cur = std::mem::take(&mut *TRACE.lock().unwrap());
    let prev = PREV.lock().unwrap().replace(cur.clone())?;

    let mut out = String::new();
    let mut n = 0usize;
    if prev.len() != cur.len() {
        out.push_str(&format!(
            "OP COUNT differs: prev {} vs cur {}\n",
            prev.len(),
            cur.len()
        ));
    }
    for (i, (p, c)) in prev.iter().zip(cur.iter()).enumerate() {
        if p == c {
            continue;
        }
        n += 1;
        if n > max_report {
            continue;
        }
        let mut what = Vec::new();
        if p.kind != c.kind || p.func != c.func {
            what.push(format!("op {} -> {}", name_of(p.func), name_of(c.func)));
        }
        if p.grid != c.grid {
            what.push(format!("grid {:?} -> {:?}", p.grid, c.grid));
        }
        if p.block != c.block {
            what.push(format!("block {:?} -> {:?}", p.block, c.block));
        }
        if p.smem != c.smem {
            what.push(format!("smem {} -> {}", p.smem, c.smem));
        }
        for (a, (pv, cv)) in p.args.iter().zip(c.args.iter()).enumerate() {
            if pv != cv {
                what.push(format!("arg{a} {pv:#x} -> {cv:#x} (Δ {})", *cv as i64 - *pv as i64));
            }
        }
        if p.args.len() != c.args.len() {
            what.push(format!("argc {} -> {}", p.args.len(), c.args.len()));
        }
        out.push_str(&format!(
            "#{i} {} [{}] {}\n",
            name_of(c.func),
            c.kind,
            what.join("; ")
        ));
    }
    Some(format!("{n} differing op(s) of {}\n{out}", cur.len().min(prev.len())))
}
