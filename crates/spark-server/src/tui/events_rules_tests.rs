// SPDX-License-Identifier: AGPL-3.0-only

//! The event loop's rules, driven directly — the loop itself needs a real tty.

use super::*;
use std::sync::mpsc::channel;

// ── the key-release filter ───────────────────────────────────────────────────

#[test]
fn a_key_release_is_not_an_action() {
    // Windows emits a record for both edges, and the kitty protocol can be
    // asked to. Without this, one keystroke is applied twice.
    assert!(!key_is_actionable(KeyEventKind::Release));
}

#[test]
fn a_held_key_still_acts() {
    // `Repeat`, not `Press`, is what a held arrow key sends — so the filter
    // must name what it drops rather than what it keeps.
    assert!(key_is_actionable(KeyEventKind::Press));
    assert!(key_is_actionable(KeyEventKind::Repeat));
}

// ── hot-swap coalescing ──────────────────────────────────────────────────────

#[test]
fn only_the_newest_publication_survives() {
    // A swap replaces the run: the older handles name a model that is no
    // longer loaded, so acting on them would toggle a lever nobody is reading.
    let (tx, rx) = channel();
    for i in 0..4 {
        tx.send(i).expect("live receiver");
    }
    assert_eq!(newest(&rx), Some(3));
    assert_eq!(newest(&rx), None, "and the queue is drained, not stepped");
}

#[test]
fn an_empty_or_dead_channel_yields_nothing() {
    let (tx, rx) = channel::<u8>();
    assert_eq!(newest(&rx), None);
    drop(tx);
    assert_eq!(
        newest(&rx),
        None,
        "a run that never published is not an error"
    );
}

#[test]
fn a_dead_channel_still_yields_what_it_carried() {
    // `serve()` can publish and return before the tick next runs.
    let (tx, rx) = channel();
    tx.send(7).expect("live receiver");
    drop(tx);
    assert_eq!(newest(&rx), Some(7));
}

// ── sampling cadence ─────────────────────────────────────────────────────────

#[test]
fn metrics_are_sampled_once_a_second_at_the_ten_hertz_tick() {
    let sampled: Vec<u32> = (1..=25).filter(|t| samples_metrics(*t)).collect();
    assert_eq!(sampled, vec![10, 20]);
}

#[test]
fn the_first_sample_waits_a_full_interval() {
    // The counter is incremented BEFORE the question, so tick 1 is 100ms in
    // and the first sample is a delta measured over a whole second rather than
    // over nothing.
    assert!(!samples_metrics(1));
    assert!(samples_metrics(SAMPLE_EVERY));
}

#[test]
fn the_wrap_costs_one_short_interval_and_nothing_else() {
    // 13.6 years of uptime at 10Hz. Pinned because the alternative — a
    // saturating counter — parks on u32::MAX, which is not a multiple of ten,
    // and stops sampling for good.
    assert!(samples_metrics(4_294_967_290));
    assert!(!samples_metrics(u32::MAX));
    assert!(samples_metrics(0), "zero is a multiple of everything");
    let after_wrap = (4_294_967_291u32..=u32::MAX)
        .filter(|t| samples_metrics(*t))
        .count();
    assert_eq!(after_wrap, 0, "so the short interval is six ticks, not two");
}

// ── the lazy tick ────────────────────────────────────────────────────────────

fn idle() -> LibraryPhase {
    LibraryPhase::default()
}

#[test]
fn nothing_lazy_happens_off_the_section_that_needs_it() {
    for s in [Section::Main, Section::Stats, Section::Network] {
        let w = tick_work(
            s,
            LibraryPhase {
                dirty: true,
                ..idle()
            },
        );
        assert_eq!(w, TickWork::default(), "{}", s.label());
    }
}

#[test]
fn entering_the_library_attaches_the_recipes_and_scans_the_cache() {
    let w = tick_work(
        Section::Library,
        LibraryPhase {
            dirty: true,
            ..idle()
        },
    );
    assert!(w.start_scan);
    assert!(w.attach_recipes);
    assert!(w.poll_library);
    assert!(!w.load_history);
}

#[test]
fn the_recipe_fetch_is_not_repeated_once_it_has_run() {
    // It is rate-limited at 60 calls an hour and has nothing to do with what
    // changed on disk, so a rescan must not re-trigger it.
    let w = tick_work(
        Section::Library,
        LibraryPhase {
            dirty: true,
            recipes_attached: true,
            ..idle()
        },
    );
    assert!(!w.attach_recipes);
    assert!(w.start_scan, "the local scan is the half that does repeat");
}

#[test]
fn a_store_that_cannot_exist_is_not_asked_for_again() {
    // ★ THE FIX. `attached()` stays false when `ArtifactStore::discover()`
    // fails, and it fails only on facts fixed for the life of the process —
    // so `in_library && !attached` retried it every 100ms, warning each time.
    // Ten thousand identical lines evict the whole log ring, startup included.
    let w = tick_work(
        Section::Library,
        LibraryPhase {
            recipes_unavailable: true,
            ..idle()
        },
    );
    assert!(!w.attach_recipes);
    assert!(w.poll_library, "and the local half still renders");
}

#[test]
fn a_clean_library_starts_no_scan() {
    assert!(!tick_work(Section::Library, idle()).start_scan);
}

#[test]
fn a_dirty_library_waits_for_the_running_scan_to_finish() {
    // ★ THE FIX. `start_scan` is idempotent, but the caller cleared
    // `library_dirty` regardless — so a download that finished DURING a scan
    // lost its request: the scan in flight was started before that checkpoint
    // existed and its results never mention it. The model then did not appear
    // until something unrelated dirtied the list again.
    let mid_scan = LibraryPhase {
        dirty: true,
        scan_in_flight: true,
        ..idle()
    };
    assert!(
        !tick_work(Section::Library, mid_scan).start_scan,
        "so the caller leaves the flag set"
    );
    // And the moment it lands, the scan that CAN see the change starts.
    let settled = LibraryPhase {
        scan_in_flight: false,
        ..mid_scan
    };
    assert!(tick_work(Section::Library, settled).start_scan);
}

#[test]
fn the_benchmarks_section_is_the_only_one_that_reads_history() {
    let w = tick_work(
        Section::Benchmarks,
        LibraryPhase {
            dirty: true,
            ..idle()
        },
    );
    assert!(w.load_history);
    assert!(!w.start_scan, "and it does not do the Library's work");
    assert!(!w.attach_recipes);
    assert!(!w.poll_library);
}

// ── exit classification ──────────────────────────────────────────────────────

#[test]
fn an_idle_loop_keeps_going() {
    assert_eq!(exit_kind(false, false, false), None);
}

#[test]
fn q_and_ctrl_c_stop_the_server() {
    assert_eq!(exit_kind(true, false, false), Some(Exit::Quit));
}

#[test]
fn detach_leaves_the_server_running() {
    assert_eq!(exit_kind(false, true, false), Some(Exit::Detach));
}

#[test]
fn a_shutdown_from_elsewhere_is_not_a_detach() {
    // ★ THE FIX. A SIGTERM — `docker stop`, or `tui::stop_and_join` on the way
    // out of `main` — sets neither flag, and the old two-flag reading fell
    // through to "TUI detached — plain logs resume", which tells the operator
    // the server is still up. It was already draining.
    assert_eq!(exit_kind(false, false, true), Some(Exit::ShuttingDown));
}

#[test]
fn the_users_intent_outranks_the_flag_it_raised() {
    // Ctrl+C and `/quit` set `should_quit` AND request shutdown; both must
    // still classify as the deliberate quit they are.
    assert_eq!(exit_kind(true, false, true), Some(Exit::Quit));
    // `/detach` cannot set `should_quit`, but a signal racing it can arrive
    // first. Leaving the dashboard is still what the user asked for.
    assert_eq!(exit_kind(false, true, true), Some(Exit::Detach));
}
