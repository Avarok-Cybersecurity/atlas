// SPDX-License-Identifier: AGPL-3.0-only

//! One wave per tick, and why a burst engages the lever at all.
//!
//! H100 round 15 §3.7 measured the VARLEN lever as a THROUGHPUT loss on the
//! short shape even with #1002's slot-aliasing fix in: whenever it engaged,
//! `16 streams -> 3 wave(s)` ran back-to-back inside one tick, no stream was
//! promoted until all three had run, and TTFT p50 collapsed onto p99
//! (1 359.4 -> 4 141.2 ms) while aggregate fell 513.86 -> 427.53 tok/s
//! (-16.8%). TPOT moved the other way (25.90 -> 21.28 ms), which is what says
//! the batching itself works and the SCHEDULING of it did not.
//!
//! Split from `prefill_waves_tests.rs` for the 500-LoC cap.

use super::{WaveGeom, plan_prefill_waves, varlen_admission, waves_this_tick};

/// The round-13/15 short shape: 1193 tokens pre-split to a 1168-token head and
/// a 25-token tail at KV block size 16.
const HEAD: usize = 1168;
const TAIL: usize = 25;
const CAP: usize = 8192;

fn head() -> WaveGeom {
    WaveGeom {
        chunk_start: 0,
        chunk_len: HEAD,
        is_last: false,
    }
}

fn tail() -> WaveGeom {
    WaveGeom {
        chunk_start: HEAD,
        chunk_len: TAIL,
        is_last: true,
    }
}

/// THE fix, on THE shape. Sixteen fresh 1193-token streams against an
/// 8192-token budget plan 7 / 7 / 2 — and only the first seven prefill on tick
/// 1. The other nine keep their `chunk_offset` and re-plan next tick, so wave 2
/// is never issued before wave 1's streams have been through
/// `promote_completed_prefills` and the tick's decode step.
#[test]
fn only_the_first_wave_prefills_on_tick_one() {
    let geoms = vec![head(); 16];
    let planned = plan_prefill_waves(&geoms, true, CAP);
    assert_eq!(
        planned.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![7, 7, 2],
        "7 x 1168 = 8176 <= 8192; an eighth would be 9344"
    );

    let dispatched = waves_this_tick(planned.clone(), true);
    assert_eq!(dispatched, vec![(0..7).collect::<Vec<_>>()]);
    let ran: usize = dispatched.iter().map(Vec::len).sum();
    assert_eq!(ran, 7, "only 7 of 16 streams advance on tick 1");
    assert_eq!(
        geoms.len() - ran,
        9,
        "the other nine are deferred to the next tick, not dropped"
    );
}

/// Tick by tick, the whole burst — the property the round-15 collapse violated:
/// a stream reaches its LAST chunk (and therefore its first token, and
/// therefore promotion) before a later wave's forward is ever issued.
///
/// Tick 1: streams 0..6 take their heads. Tick 2: the planner is re-run from
/// the live geometry, so those seven are now at `chunk_start = 1168` and stand
/// FIRST in FIFO order — their tails are wave 1 and they complete. Only then
/// does the next set of heads go. Under the old all-waves-per-tick rule all
/// sixteen heads and then all sixteen tails ran inside two ticks with nobody
/// promoted in between, which is the p50 = p99 signature.
#[test]
fn the_first_waves_streams_reach_decode_before_the_second_waves_forward() {
    // Tick 1 — sixteen fresh heads.
    let t1 = vec![head(); 16];
    let w1 = waves_this_tick(plan_prefill_waves(&t1, true, CAP), true);
    assert_eq!(w1.len(), 1);
    assert_eq!(w1[0], (0..7).collect::<Vec<_>>());
    assert!(
        !t1[w1[0][0]].is_last,
        "the heads are not last chunks; nobody is promoted on tick 1"
    );

    // Tick 2 — the seven that ran are at their tails, nine are still fresh.
    let mut t2 = vec![tail(); 7];
    t2.extend(std::iter::repeat_n(head(), 9));
    let w2 = waves_this_tick(plan_prefill_waves(&t2, true, CAP), true);
    assert_eq!(w2, vec![(0..7).collect::<Vec<_>>()], "the seven TAILS");
    for &i in &w2[0] {
        assert!(
            t2[i].is_last,
            "stream {i} finishes its prompt on tick 2 and is promoted at the \
             end of THIS tick — before any further head wave is issued"
        );
    }
    let m: usize = w2[0].iter().map(|&i| t2[i].chunk_len).sum();
    assert_eq!(
        m,
        7 * TAIL,
        "one 175-token forward, not seven 25-token ones"
    );
}

/// The deferred streams are not starved and they still BATCH: the wave the
/// planner gives them next tick is the same first-fit pack, taken among
/// themselves. Wave 1 always contains stream 0 because the planner is first-fit
/// in FIFO order, so the head of the queue advances one chunk every tick —
/// exactly the guarantee the single-stream `prefilling.first_mut()` path gives.
#[test]
fn deferred_streams_batch_among_themselves_next_tick() {
    let nine = vec![head(); 9];
    let w = waves_this_tick(plan_prefill_waves(&nine, true, CAP), true);
    assert_eq!(w[0].len(), 7, "seven of the nine still pack one wave");
    assert_eq!(w[0][0], 0, "the FIFO head is always in the dispatched wave");
}

/// Flag OFF is untouched: one wave holding every stream, dispatched whole. The
/// pre-wave scheduler made ONE `prefill_batch_chunk` call with all streams and
/// this lever is default OFF, so that path must be byte-identical.
#[test]
fn flag_off_still_dispatches_every_stream_in_one_tick() {
    let geoms = vec![head(); 16];
    let planned = plan_prefill_waves(&geoms, false, CAP);
    assert_eq!(planned, vec![(0..16).collect::<Vec<_>>()]);
    assert_eq!(waves_this_tick(planned.clone(), false), planned);
    // And the degenerate cases.
    assert!(waves_this_tick(Vec::new(), true).is_empty());
    assert!(waves_this_tick(Vec::new(), false).is_empty());
    let one = vec![vec![0usize]];
    assert_eq!(waves_this_tick(one.clone(), true), one);
}

// ── the admission verdict, and WHY a burst engages (round 15 anomaly 8) ──

/// The long shape still skips the deferral, for the reason it always did — the
/// `deferral SKIPPED` serve line must keep firing on `4096x512` C=16, where
/// round 15 measured V15 landing on A15's numbers exactly (398.34 vs 401.28
/// tok/s, 4 115.2 vs 4 118.8 ms TTFT).
#[test]
fn the_long_shape_deferral_is_still_skipped_for_the_same_reason() {
    let long = vec![4576usize; 16];
    let a = varlen_admission(true, true, false, 0, 16, 0, &long, CAP);
    assert!(!a.defer);
    assert_eq!(a.reason, "no two chunk-0s fit one wave");

    // …and the short shape still defers.
    let short = vec![1193usize; 16];
    let b = varlen_admission(true, true, false, 0, 16, 0, &short, CAP);
    assert!(b.defer);
    assert_eq!(b.reason, "two smallest chunk-0s share a wave");
}

/// WHY engagement varied burst to burst. Same sixteen short prompts, same
/// budget, same binary — the only difference is whether a decode was already
/// running when the tick admitted them, which is arrival timing against the
/// scheduler's tick period and not something the planner can pin. The verdict
/// now NAMES that input, so a serve log says which burst was which instead of
/// leaving it to be inferred from a 17.52% rep spread.
#[test]
fn an_active_decode_is_what_flips_engagement_between_bursts() {
    let short = vec![1193usize; 16];
    let engaged = varlen_admission(true, true, false, 0, 16, 0, &short, CAP);
    let not = varlen_admission(true, true, false, 1, 16, 0, &short, CAP);
    assert!(engaged.defer && !not.defer);
    assert_eq!(
        not.reason,
        "decode already active this tick (arrival timing)"
    );
    assert_ne!(
        engaged.reason, not.reason,
        "the log distinguishes the bursts"
    );
}

/// Every other rung refuses with its own name, so the line is never ambiguous
/// about which gate closed — the same posture `ssm_ba_gates_hopper_reject`
/// takes, and for the same reason: a lever that asks to be enabled and silently
/// is not measures as "no effect".
#[test]
fn every_admission_rung_names_itself() {
    let short = vec![1193usize; 16];
    let reasons = [
        varlen_admission(false, true, false, 0, 16, 0, &short, CAP).reason,
        varlen_admission(true, false, false, 0, 16, 0, &short, CAP).reason,
        varlen_admission(true, true, true, 0, 16, 0, &short, CAP).reason,
        varlen_admission(true, true, false, 3, 16, 0, &short, CAP).reason,
        varlen_admission(true, true, false, 0, 1, 0, &[1193], CAP).reason,
        varlen_admission(true, true, false, 0, 16, 0, &[4576; 16], CAP).reason,
    ];
    let mut unique = reasons.to_vec();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), reasons.len(), "two rungs share a reason");
    for r in reasons {
        assert!(!r.is_empty());
    }
    // A lone arrival with a stream already in flight still defers: the late
    // request joins the next wave, which is what `!prefilling.is_empty()`
    // bought and must keep buying.
    let late = varlen_admission(true, true, false, 0, 1, 1, &[1193, 1193], CAP);
    assert!(late.defer);
}
