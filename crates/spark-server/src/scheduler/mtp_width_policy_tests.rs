// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the `mtp_width_policy` row: the pacing arithmetic and the
//! per-width memory on their own, then the gate driven exactly as the
//! scheduler drives it (`next_step`, run that step, `record_*`,
//! `take_fresh_decision`) through the tier-0 shape that motivated the row:
//! at 8 streams an MTP step costs 99 ms for 13 tokens and a decode step 40 ms
//! for 8 (RTX 4060 Ti 16 GB, Qwen3.5-4B, FP8 FFN, one draft, 2026-11-12).
//!
//! The gate reads its base intervals from `AVAROK_MTP_GATE_REPROBE` and
//! `AVAROK_MTP_GATE_REFRESH`; these tests assume the defaults (256 and 1024),
//! as the gate's own tests do.

use super::*;
use crate::scheduler::mtp_gate::{GateDecision, GateStep, MtpGate};
use std::time::Duration;

// ── The arithmetic and the memory ────────────────────────────────────────

#[test]
fn a_width_regime_maps_to_its_power_of_two_bucket() {
    assert_eq!(bucket(1), 0);
    assert_eq!(bucket(2), 1);
    assert_eq!(bucket(4), 2);
    assert_eq!(bucket(8), 3);
    assert_eq!(bucket(1 << 15), 15);
    // Past the last bucket every width shares it; 0 (nothing recorded) is 1.
    assert_eq!(bucket(1 << 20), BUCKETS - 1);
    assert_eq!(bucket(0), 0);
}

#[test]
fn an_interval_counts_tokens_per_stream_and_doubles_per_confirming_probe() {
    // One stream, no backoff: the base interval, which is the pre-row value.
    assert_eq!(interval(256, 1, 0), 256);
    assert_eq!(interval(256, 0, 0), 256, "nothing recorded yet");
    // Eight streams: 256 tokens per stream.
    assert_eq!(interval(256, 8, 0), 2048);
    assert_eq!(interval(256, 8, 1), 4096);
    assert_eq!(interval(256, 8, 2), 8192);
    assert_eq!(interval(256, 8, MAX_BACKOFF_LOG2), 16384);
    assert_eq!(interval(256, 8, 40), 16384, "capped at 8x");
    assert_eq!(interval(usize::MAX, 8, 3), usize::MAX, "saturates");
}

#[test]
fn the_memory_keeps_one_snapshot_per_bucket_and_hands_it_back_once() {
    let mut m: WidthMemory<u32> = WidthMemory::new();
    assert_eq!(m.take(8), None);
    m.save(8, 80);
    m.save(1, 10);
    m.save(8, 81); // the latest save wins
    assert_eq!(m.take(8), Some(81));
    assert_eq!(m.take(8), None, "taken out while it is live");
    assert_eq!(m.take(1), Some(10));
    assert_eq!(m.take(4), None);
}

#[test]
fn confirming_probes_back_off_and_anything_else_resets() {
    let mut m: WidthMemory<()> = WidthMemory::new();
    assert_eq!(m.confirms(), 0);
    for want in [1, 2, 3, 3, 3] {
        m.note_probe(true);
        assert_eq!(m.confirms(), want);
    }
    m.note_probe(false);
    assert_eq!(m.confirms(), 0, "the other mode looked better");
    m.note_probe(true);
    m.note_probe(true);
    m.reset_backoff();
    assert_eq!(m.confirms(), 0, "a switch or a new regime");
    // A resumed width keeps its backoff only up to the resume cap.
    m.resume(MAX_BACKOFF_LOG2);
    assert_eq!(m.confirms(), RESUME_BACKOFF_LOG2);
    m.resume(1);
    assert_eq!(m.confirms(), 1);
}

// ── The gate, driven as the scheduler drives it ──────────────────────────

/// One workload shape at a batch width: tokens and milliseconds of an MTP
/// step, and milliseconds of a decode step (which emits `width` tokens).
#[derive(Clone, Copy)]
struct Shape {
    width: usize,
    mtp_tokens: usize,
    mtp_ms: u64,
    dec_ms: u64,
}

/// Tier 0 at one stream: MTP wins (2 tokens in 38 ms against 1 in 27 ms).
const ONE: Shape = Shape {
    width: 1,
    mtp_tokens: 2,
    mtp_ms: 38,
    dec_ms: 27,
};
/// Tier 0 at eight streams: MTP loses (13 tokens in 99 ms against 8 in 40 ms).
const EIGHT: Shape = Shape {
    width: 8,
    mtp_tokens: 13,
    mtp_ms: 99,
    dec_ms: 40,
};

#[derive(Default, Debug)]
struct Tally {
    mtp_steps: usize,
    dec_steps: usize,
    tokens: usize,
    wall_ms: u64,
    /// `DisableMtp` hand-offs the scheduler would have acted on.
    disables: usize,
}

impl Tally {
    fn steps(&self) -> usize {
        self.mtp_steps + self.dec_steps
    }
    fn tok_per_s(&self) -> f64 {
        self.tokens as f64 * 1000.0 / self.wall_ms as f64
    }
}

/// Run `steps` scheduler steps of `shape`: whatever the gate asks for runs,
/// is recorded at the shape's width, and the fresh decision is drained.
fn run(g: &mut MtpGate, steps: usize, shape: Shape) -> Tally {
    let mut t = Tally::default();
    for _ in 0..steps {
        match g.next_step() {
            GateStep::MeasureVerify => {
                g.record_verify_step(
                    Duration::from_millis(shape.mtp_ms),
                    shape.mtp_tokens,
                    shape.width,
                );
                t.mtp_steps += 1;
                t.tokens += shape.mtp_tokens;
                t.wall_ms += shape.mtp_ms;
            }
            GateStep::MeasureDecode => {
                g.record_decode(Duration::from_millis(shape.dec_ms), shape.width);
                t.dec_steps += 1;
                t.tokens += shape.width;
                t.wall_ms += shape.dec_ms;
            }
        }
        if g.take_fresh_decision() == Some(GateDecision::DisableMtp) {
            t.disables += 1;
        }
    }
    t
}

/// A gate that has served one stream long enough to stand in Mtp there.
fn warmed(width_policy: bool) -> MtpGate {
    let mut g = MtpGate::with_width_policy(1, width_policy);
    let t = run(&mut g, 400, ONE);
    assert_eq!(t.dec_steps, 0, "no serial refresh inside 800 tokens");
    assert!(!g.in_serial_mode());
    g
}

#[test]
fn without_the_row_a_third_of_the_steps_at_eight_streams_probe_the_losing_mode() {
    // The measured failure, reproduced from the gate's own rules: re-probe
    // every 256 AGGREGATE tokens (32 decode steps at 8 streams) for a
    // 16-step window.
    let mut g = warmed(false);
    let t = run(&mut g, 4800, EIGHT);
    assert!(g.in_serial_mode(), "the gate does choose Serial");
    assert!(
        t.mtp_steps * 4 > t.steps(),
        "expected about 1 step in 3 on MTP, got {t:?}"
    );
    // 256 tokens in 1.28 s plus 208 in 1.58 s: what tier 0 delivered.
    let tps = t.tok_per_s();
    assert!((155.0..170.0).contains(&tps), "{tps:.1} tok/s, {t:?}");
}

#[test]
fn with_the_row_the_same_traffic_keeps_the_probes_rare() {
    let mut g = warmed(true);
    let t = run(&mut g, 4800, EIGHT);
    assert!(g.in_serial_mode());
    // The first look (one MTP step in flight, the dwell window) plus probes
    // at 256, 512, 1024 and 2048 steps: under 2% of the steps.
    assert!(
        t.mtp_steps * 20 < t.steps(),
        "expected under 5% of the steps on MTP, got {t:?}"
    );
    let tps = t.tok_per_s();
    assert!(
        tps > 190.0,
        "{tps:.1} tok/s against 200 for decode alone, {t:?}"
    );
}

#[test]
fn the_row_recovers_what_the_probes_cost() {
    let (mut off, mut on) = (warmed(false), warmed(true));
    let (a, b) = (run(&mut off, 4800, EIGHT), run(&mut on, 4800, EIGHT));
    assert!(
        b.tok_per_s() > 1.15 * a.tok_per_s(),
        "row on {:.1} tok/s, row off {:.1} tok/s",
        b.tok_per_s(),
        a.tok_per_s()
    );
}

#[test]
fn a_width_seen_before_resumes_its_decision_without_a_probe() {
    let mut g = warmed(true);
    run(&mut g, 600, EIGHT);
    assert!(g.in_serial_mode());
    // Back to one stream: that bucket resumes Mtp, where it stood. (100
    // steps stay inside its 1024-token serial refresh interval.)
    let one = run(&mut g, 100, ONE);
    assert!(!g.in_serial_mode());
    assert_eq!(
        one.dec_steps, 1,
        "only the step already dispatched as decode, got {one:?}"
    );
    // And back to eight: the first step was dispatched under the one-stream
    // decision (MTP); the bucket then resumes Serial and tells the scheduler
    // to drop its drafts, with no probe.
    let eight = run(&mut g, 150, EIGHT);
    assert!(g.in_serial_mode());
    assert_eq!(eight.mtp_steps, 1, "{eight:?}");
    assert_eq!(eight.disables, 1, "the scheduler is told once, {eight:?}");
    assert!(!g.is_probing());
}

#[test]
fn without_the_row_coming_back_to_a_width_costs_a_fresh_measurement() {
    let mut g = warmed(false);
    run(&mut g, 600, EIGHT);
    run(&mut g, 200, ONE);
    let eight = run(&mut g, 200, EIGHT);
    // Stale baselines, a probe pulled forward, the dwell, then the 32:16
    // cycle again: many times the one step the row needs.
    assert!(eight.mtp_steps >= 32, "{eight:?}");
}

#[test]
fn a_width_never_seen_is_measured_before_it_is_trusted() {
    // The row keeps the pre-row rule for a first visit: the inherited mode
    // is probed at once and the switch needs the dwell, so the first 33
    // steps at 8 streams are the same with the row on and off.
    let (mut off, mut on) = (warmed(false), warmed(true));
    let (a, b) = (run(&mut off, 33, EIGHT), run(&mut on, 33, EIGHT));
    assert_eq!(a.mtp_steps, b.mtp_steps);
    assert_eq!(a.dec_steps, b.dec_steps);
    assert_eq!(a.dec_steps, 16, "one serial probe window");
    assert!(off.in_serial_mode() && on.in_serial_mode());
}

#[test]
fn a_probe_that_favors_the_other_mode_switches_and_resets_the_backoff() {
    let mut g = warmed(true);
    run(&mut g, 4800, EIGHT);
    assert!(g.in_serial_mode());
    // The verify gets cheap (say the GDN verify GEMM is fixed): 13 tokens in
    // 45 ms beats 8 in 40. The standing backoff delays the next probe, it
    // does not prevent it.
    let fast = Shape {
        mtp_ms: 45,
        ..EIGHT
    };
    let mut waited = 0;
    while g.in_serial_mode() {
        run(&mut g, 1, fast);
        waited += 1;
        assert!(waited < 2600, "the backed-off probe never came");
    }
    // Once in Mtp the serial refresh starts from the base interval again:
    // 1024 tokens per stream at 8 streams is 8192 tokens, which is 631 MTP
    // steps of 13 tokens. A backoff carried over would make it 8 times that.
    let mut mtp_run = 0;
    while g.next_step() == GateStep::MeasureVerify {
        run(&mut g, 1, fast);
        mtp_run += 1;
        assert!(mtp_run < 6000, "the serial refresh never came");
    }
    assert!(
        (620..=640).contains(&mtp_run),
        "{mtp_run} MTP steps to the refresh"
    );
}

#[test]
fn the_row_off_constructor_is_the_plain_gate() {
    // `MtpGate::new` reads the process row; this comparison only means
    // something with it off (the default).
    if row() {
        return;
    }
    let (mut a, mut b) = (MtpGate::new(1), MtpGate::with_width_policy(1, false));
    for shape in [ONE, EIGHT, ONE, EIGHT] {
        for _ in 0..700 {
            assert_eq!(a.next_step(), b.next_step());
            match a.next_step() {
                GateStep::MeasureVerify => {
                    let wall = Duration::from_millis(shape.mtp_ms);
                    a.record_verify_step(wall, shape.mtp_tokens, shape.width);
                    b.record_verify_step(wall, shape.mtp_tokens, shape.width);
                }
                GateStep::MeasureDecode => {
                    let wall = Duration::from_millis(shape.dec_ms);
                    a.record_decode(wall, shape.width);
                    b.record_decode(wall, shape.width);
                }
            }
            assert_eq!(a.take_fresh_decision(), b.take_fresh_decision());
            assert_eq!(a.in_serial_mode(), b.in_serial_mode());
        }
    }
}

// ── One stream: the row is a no-op ───────────────────────────────────────

#[test]
fn an_interval_at_one_stream_is_the_base_whatever_the_backoff() {
    for confirms in 0..=40 {
        assert_eq!(interval(256, 1, confirms), 256);
        assert_eq!(interval(1024, 1, confirms), 1024);
        assert_eq!(interval(256, 0, confirms), 256, "nothing recorded yet");
    }
    // Two streams is the first width the row paces per stream.
    assert_eq!(interval(256, 2, 0), 512);
}

/// Drive a row-on and a row-off gate through the same single-stream trace and
/// compare everything the scheduler can observe after every step: which step
/// runs next, the fresh decision, the standing mode and whether a probe is
/// running. The trace is long enough for several serial refreshes (1024
/// tokens apart) that each CONFIRM the standing mode, which is what the
/// backoff reacts to at wider widths.
fn assert_one_stream_identical(shape: Shape, steps: usize) -> (Tally, Tally) {
    assert_eq!(shape.width, 1);
    let (mut off, mut on) = (
        MtpGate::with_width_policy(1, false),
        MtpGate::with_width_policy(1, true),
    );
    let (mut a, mut b) = (Tally::default(), Tally::default());
    for i in 0..steps {
        assert_eq!(off.next_step(), on.next_step(), "step {i}");
        assert_eq!(off.is_probing(), on.is_probing(), "step {i}");
        assert_eq!(off.in_serial_mode(), on.in_serial_mode(), "step {i}");
        let (ta, tb) = (run(&mut off, 1, shape), run(&mut on, 1, shape));
        // `run` drains the fresh decision; the hand-off counts must match.
        assert_eq!(ta.disables, tb.disables, "step {i}");
        assert_eq!(ta.mtp_steps, tb.mtp_steps, "step {i}");
        assert_eq!(ta.dec_steps, tb.dec_steps, "step {i}");
        for (t, d) in [(&mut a, ta), (&mut b, tb)] {
            t.mtp_steps += d.mtp_steps;
            t.dec_steps += d.dec_steps;
            t.tokens += d.tokens;
            t.wall_ms += d.wall_ms;
            t.disables += d.disables;
        }
    }
    (a, b)
}

#[test]
fn one_stream_takes_the_row_off_path_step_for_step() {
    // MTP wins at one stream: the gate stays in Mtp and refreshes the serial
    // baseline every 1024 tokens, 8000 steps being about 16 refreshes, each
    // of which holds the standing mode.
    let (a, b) = assert_one_stream_identical(ONE, 8000);
    assert!(
        a.dec_steps > 0,
        "the trace must include serial probes: {a:?}"
    );
    assert_eq!(a.dec_steps, b.dec_steps);
    assert_eq!(a.mtp_steps, b.mtp_steps);
    assert_eq!(a.wall_ms, b.wall_ms);
    assert_eq!(b.disables, 0);
}

#[test]
fn one_stream_where_decode_wins_takes_the_row_off_path_step_for_step() {
    // The mirror: MTP loses at one stream (acceptance collapsed), the gate
    // switches to Serial and re-probes MTP every 256 tokens, each probe
    // confirming Serial. The row must not stretch those intervals at C=1.
    let losing = Shape {
        mtp_tokens: 1,
        mtp_ms: 45,
        ..ONE
    };
    let (a, b) = assert_one_stream_identical(losing, 8000);
    assert_eq!(a.mtp_steps, b.mtp_steps);
    assert_eq!(a.disables, b.disables);
    assert!(a.disables >= 1, "the gate should have gone Serial: {a:?}");
}

#[test]
fn the_screen_trace_never_leaves_mtp_or_probes_with_the_row_on() {
    // The bench C=1 phase: three requests of 256 tokens, 153 verify passes
    // each, 1.66 tokens a pass. 768 tokens stay inside the 1024-token refresh
    // interval, so no serial step may run, with the row on or off.
    for row in [false, true] {
        let mut g = MtpGate::with_width_policy(1, row);
        let t = run(&mut g, 3 * 153, ONE);
        assert_eq!(t.dec_steps, 0, "row {row}: {t:?}");
        assert_eq!(t.disables, 0);
        assert!(!g.is_probing() && !g.in_serial_mode());
    }
}

/// Per-step cost of the gate at one stream, row off against row on. Timing
/// is machine dependent, so this prints and asserts only a loose bound:
/// `cargo test -p spark-server -- --ignored one_stream_gate_cost --nocapture`.
#[test]
#[ignore = "timing probe, run by hand"]
fn one_stream_gate_cost_per_step() {
    const N: usize = 2_000_000;
    let mut ns = [0.0f64; 2];
    for (i, row) in [false, true].into_iter().enumerate() {
        let mut g = MtpGate::with_width_policy(1, row);
        let wall = Duration::from_millis(38);
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            std::hint::black_box(g.next_step());
            g.record_verify_step(std::hint::black_box(wall), 2, 1);
            std::hint::black_box(g.take_fresh_decision());
        }
        ns[i] = t0.elapsed().as_nanos() as f64 / N as f64;
    }
    println!(
        "gate per step at one stream: row off {:.1} ns, row on {:.1} ns",
        ns[0], ns[1]
    );
    // Both are far below a 33 ms step; a regression to microseconds would show.
    assert!(ns[1] < ns[0] + 200.0, "{ns:?}");
}
