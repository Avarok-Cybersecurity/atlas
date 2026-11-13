// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`mtp_device_timing`) against a scripted clock: events
//! are stamped with a hand-advanced "now", so every interval is known exactly.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use super::*;

struct Fake {
    now: Cell<f64>,
    at: RefCell<HashMap<u64, f64>>,
    next: Cell<u64>,
    creates: Cell<u32>,
    ready: Cell<bool>,
    fail_create: Cell<bool>,
    fail_elapsed: Cell<bool>,
}

impl Fake {
    fn new() -> Self {
        Self {
            now: Cell::new(0.0),
            at: RefCell::new(HashMap::new()),
            next: Cell::new(0),
            creates: Cell::new(0),
            ready: Cell::new(true),
            fail_create: Cell::new(false),
            fail_elapsed: Cell::new(false),
        }
    }

    fn advance(&self, ms: f64) {
        self.now.set(self.now.get() + ms);
    }
}

impl DeviceClock for Fake {
    fn timing_event_create(&self) -> anyhow::Result<u64> {
        self.creates.set(self.creates.get() + 1);
        if self.fail_create.get() {
            anyhow::bail!("timing events unsupported");
        }
        self.next.set(self.next.get() + 1);
        Ok(self.next.get())
    }

    fn timing_event_record(&self, event: u64, _stream: u64) -> anyhow::Result<()> {
        self.at.borrow_mut().insert(event, self.now.get());
        Ok(())
    }

    fn timing_event_elapsed_ms(&self, start: u64, end: u64) -> anyhow::Result<Option<f32>> {
        if self.fail_elapsed.get() {
            anyhow::bail!("elapsed failed");
        }
        if !self.ready.get() {
            return Ok(None);
        }
        let at = self.at.borrow();
        Ok(Some((at[&end] - at[&start]) as f32))
    }
}

/// One step: 0.5 ms gap, Start, 1 ms propose, 2 ms verify, 0.5 ms commit.
fn steps(t: &DeviceTimer, f: &Fake, n: usize) {
    for _ in 0..n {
        f.advance(0.5);
        t.mark(Phase::Start, f, 0);
        f.advance(1.0);
        t.mark(Phase::Propose, f, 0);
        f.advance(2.0);
        t.mark(Phase::Verify, f, 0);
        f.advance(0.5);
        t.mark(Phase::Commit, f, 0);
    }
}

// gap, propose, verify, readback, pipeline, commit, bootstrap, other
const ONE_WINDOW: [u64; BUCKETS] = [12_500, 25_000, 50_000, 0, 0, 12_500, 0, 0];

#[test]
fn a_window_charges_each_interval_to_the_phase_that_closed_it() {
    let (t, f) = (DeviceTimer::new(), Fake::new());
    // 26 Starts: the first is the origin, the 26th closes the window; the 27th
    // polls it.
    steps(&t, &f, 27);
    assert_eq!(t.totals_us(), ONE_WINDOW);
    assert_eq!(t.steps.load(Ordering::Relaxed), u64::from(SUMMARY_PERIOD));
    assert_eq!(t.windows.load(Ordering::Relaxed), 1);
}

#[test]
fn an_incomplete_window_waits_and_nothing_syncs() {
    let (t, f) = (DeviceTimer::new(), Fake::new());
    f.ready.set(false);
    steps(&t, &f, 30);
    assert_eq!(
        t.totals_us(),
        [0; BUCKETS],
        "nothing may be read before it completes"
    );
    assert_eq!(t.inner.lock().pending.len(), 1);
    f.ready.set(true);
    steps(&t, &f, 1);
    assert_eq!(t.totals_us(), ONE_WINDOW);
    assert!(t.inner.lock().pending.is_empty());
}

#[test]
fn events_are_recycled_after_a_window_is_read() {
    let (t, f) = (DeviceTimer::new(), Fake::new());
    steps(&t, &f, 25 * 8 + 2);
    assert_eq!(t.windows.load(Ordering::Relaxed), 8);
    // Eight windows of about 102 events each would be ~816 creates without
    // recycling; recycling keeps it near two windows' worth.
    assert!(f.creates.get() < 260, "creates={}", f.creates.get());
    // And the totals still scale linearly.
    assert_eq!(t.totals_us()[Phase::Verify as usize], 8 * 50_000);
}

#[test]
fn a_lagging_stream_cannot_grow_the_backlog_without_bound() {
    let (t, f) = (DeviceTimer::new(), Fake::new());
    f.ready.set(false);
    steps(&t, &f, 25 * 40);
    assert_eq!(t.inner.lock().pending.len(), MAX_PENDING);
    // The dropped windows' events went back to the pool, not out of the books.
    assert!(!t.inner.lock().pool.is_empty());
}

#[test]
fn a_backend_that_cannot_time_events_disables_the_timer_once() {
    let (t, f) = (DeviceTimer::new(), Fake::new());
    f.fail_create.set(true);
    steps(&t, &f, 40);
    assert!(t.inner.lock().dead);
    assert_eq!(f.creates.get(), 1, "no create is retried after the failure");
    assert_eq!(t.totals_us(), [0; BUCKETS]);
}

#[test]
fn an_elapsed_error_disables_the_timer() {
    let (t, f) = (DeviceTimer::new(), Fake::new());
    steps(&t, &f, 26);
    f.fail_elapsed.set(true);
    steps(&t, &f, 1);
    assert!(t.inner.lock().dead);
    assert_eq!(t.totals_us(), [0; BUCKETS]);
}

#[test]
fn a_path_without_start_marks_still_records() {
    let (t, f) = (DeviceTimer::new(), Fake::new());
    // The first mark is the origin; later ones are intervals.
    t.mark(Phase::Verify, &f, 0);
    f.advance(3.0);
    t.mark(Phase::Verify, &f, 0);
    let g = t.inner.lock();
    assert_eq!(g.seg.len(), 2);
    assert_eq!(g.starts, 0);
}

#[test]
fn the_phase_table_is_aligned() {
    assert_eq!(NAMES.len(), BUCKETS);
    assert_eq!(Phase::Start as usize, 0);
    assert_eq!(NAMES[Phase::Start as usize], "gap");
    assert_eq!(NAMES[Phase::Propose as usize], "propose");
    assert_eq!(NAMES[Phase::Verify as usize], "verify");
    assert_eq!(NAMES[Phase::Readback as usize], "readback");
    assert_eq!(NAMES[Phase::Pipeline as usize], "pipeline");
    assert_eq!(NAMES[Phase::Commit as usize], "commit");
    assert_eq!(NAMES[Phase::Bootstrap as usize], "bootstrap");
    assert_eq!(Phase::Other as usize, BUCKETS - 1);
}

#[test]
fn the_text_has_every_phase_and_the_step_count() {
    let (t, f) = (DeviceTimer::new(), Fake::new());
    steps(&t, &f, 27);
    let mut out = String::new();
    t.render(&mut out);
    assert!(out.contains("# TYPE atlas_mtp_device_phase_microseconds_total counter"));
    assert!(out.contains("atlas_mtp_device_phase_microseconds_total{phase=\"verify\"} 50000\n"));
    assert!(out.contains("atlas_mtp_device_phase_microseconds_total{phase=\"gap\"} 12500\n"));
    assert!(out.contains("atlas_mtp_device_phase_microseconds_total{phase=\"bootstrap\"} 0\n"));
    assert!(out.contains("atlas_mtp_device_phase_steps_total 25\n"));
}

#[test]
fn off_by_default_the_process_timer_is_never_touched() {
    if enabled() {
        return; // the environment armed level 2
    }
    let f = Fake::new();
    mark(Phase::Start, &f, 0);
    mark(Phase::Verify, &f, 0);
    assert_eq!(
        f.creates.get(),
        0,
        "an unarmed mark must not create an event"
    );
    assert!(TIMER.inner.lock().seg.is_empty());
}
