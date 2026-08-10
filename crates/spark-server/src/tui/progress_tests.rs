// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// A second load must render as a LOAD, not as one that already finished.
///
/// This is the defect `reset` exists for: without it `enter_phase` only
/// advances `Pending → Running`, so every phase stays `Done` from the first
/// load and the checklist shows a completed run while the model is still
/// loading.
#[test]
fn a_second_load_starts_from_a_clean_checklist() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::Phase {
        phase: 0,
        name: "banner".into(),
    });
    m.apply(ProgressEvent::Ready { port: 8888 });
    assert!(m.ready);
    assert_eq!(m.phases[0].state, PhaseState::Done);

    m.reset();

    assert!(!m.ready, "a new load has not finished");
    assert_eq!(m.port, 0, "and has not bound yet");
    assert!(
        m.phases.iter().all(|p| p.state == PhaseState::Pending),
        "every phase is pending again"
    );

    // And a re-entered phase now actually enters, rather than staying Done.
    m.apply(ProgressEvent::Phase {
        phase: 0,
        name: "banner".into(),
    });
    assert_eq!(m.phases[0].state, PhaseState::Running);
}

/// The load-rate window must reopen, or the second load's GB/s is the
/// first load's number.
#[test]
fn reset_reopens_the_frozen_load_window() {
    let mut m = ProgressModel::default();
    // The window opens on the FIRST shard, not at process start, so a load
    // has to actually begin before there is anything to freeze.
    m.apply(ProgressEvent::ShardStart {
        shard: 1,
        total: 2,
        name: "shard-1".into(),
    });
    m.apply(ProgressEvent::Ready { port: 8888 });
    assert!(
        m.load_secs.is_some(),
        "the window should be frozen after a load completes"
    );

    m.reset();
    assert!(
        m.load_secs.is_none(),
        "the frozen window survived a reset: the second load would report \
         the first load's rate, since freeze_load_window keeps the FIRST close"
    );
    assert!(m.load_started.is_none(), "and the window is not yet open");
}

#[test]
fn phase_entry_closes_earlier_phases() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::Phase {
        phase: 0,
        name: "banner".into(),
    });
    m.apply(ProgressEvent::Phase {
        phase: 3,
        name: "gpu init".into(),
    });
    assert_eq!(m.phases[0].state, PhaseState::Done);
    assert_eq!(m.phases[1].state, PhaseState::Done);
    assert_eq!(m.phases[3].state, PhaseState::Running);
    assert_eq!(m.phases[4].state, PhaseState::Pending);
}

#[test]
fn ready_completes_everything() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::Phase {
        phase: 5,
        name: "weight load".into(),
    });
    m.apply(ProgressEvent::Ready { port: 8888 });
    assert!(m.ready);
    assert!(m.phases.iter().all(|p| p.state == PhaseState::Done));
}

#[test]
fn shard_rollover_snaps_shard_bar() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::ShardDone {
        shard: 1,
        total: 4,
        used_gb: 2.0,
        free_gb: 100.0,
    });
    assert_eq!(m.shard_target(), 1.0);
    m.apply(ProgressEvent::ShardStart {
        shard: 2,
        total: 4,
        name: "s2".into(),
    });
    assert_eq!(m.shard_target(), 0.0);
}

/// GB/s must be measured over the weight-load window, not since process start,
/// and must stop moving once the load is over. Previously it divided a constant
/// number of bytes by an ever-growing elapsed time, so a finished load's rate
/// decayed toward zero for as long as the server ran -- and the pre-load time
/// (CUDA init, model resolution, preflight) was in the divisor throughout.
#[test]
fn load_rate_is_windowed_and_freezes_at_the_last_shard() {
    let ms = |n| std::thread::sleep(std::time::Duration::from_millis(n));
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::Preflight {
        disk_gb: 10.0,
        free_gb: 100.0,
    });
    ms(120); // pre-load work that must NOT count against the rate
    for shard in 1..=2u64 {
        m.apply(ProgressEvent::ShardStart {
            shard,
            total: 2,
            name: "s".into(),
        });
        ms(60);
        m.apply(ProgressEvent::ShardDone {
            shard,
            total: 2,
            used_gb: 1.0,
            free_gb: 9.0,
        });
    }
    let secs = m.load_secs().expect("last shard closes the window");
    assert!(
        secs < m.started_at.elapsed().as_secs_f64() - 0.1,
        "window {secs}s must exclude the 120ms of pre-load work"
    );
    let (rate, eta) = m.rate_eta().expect("rate known once shards are in");
    assert_eq!(eta, 0.0, "nothing left to load");
    assert!(rate > 0.0);

    ms(120);
    let (rate_later, _) = m.rate_eta().unwrap();
    assert_eq!(rate, rate_later, "a finished load's rate must not drift");
}

/// The overall bar's denominator is a shard count that is zero until the first
/// shard event lands, so every read of it is a division that has to be guarded.
#[test]
fn the_overall_fraction_is_zero_at_the_start_one_at_the_end_and_never_a_nan() {
    let mut m = ProgressModel::default();
    assert_eq!(m.shard_total, 0);
    assert_eq!(m.overall_target(), 0.0, "nothing known yet");
    assert!(m.overall_target().is_finite(), "0/0 must not reach the bar");

    m.apply(ProgressEvent::ShardDone {
        shard: 0,
        total: 4,
        used_gb: 0.0,
        free_gb: 1.0,
    });
    assert_eq!(m.overall_target(), 0.0, "no shard done is 0%");

    m.apply(ProgressEvent::ShardDone {
        shard: 4,
        total: 4,
        used_gb: 1.0,
        free_gb: 1.0,
    });
    assert_eq!(m.overall_target(), 1.0, "the last shard is 100%");
}

/// A load with no shards at all (a cached or tiny checkpoint) still reads 100%
/// once the server is up, rather than a bar frozen at zero.
#[test]
fn a_load_with_no_shards_reads_full_only_once_ready() {
    let mut m = ProgressModel::default();
    assert_eq!(m.overall_target(), 0.0);
    m.apply(ProgressEvent::Ready { port: 8888 });
    assert_eq!(m.overall_target(), 1.0);
}

/// A shard count past its own total is a loader bug, not a reason to draw a bar
/// past the end of its box.
#[test]
fn an_overshooting_shard_count_is_clamped_to_full() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::ShardDone {
        shard: 9,
        total: 4,
        used_gb: 1.0,
        free_gb: 1.0,
    });
    assert_eq!(m.overall_target(), 1.0);
}

#[test]
fn the_rate_is_unknown_rather_than_infinite_before_anything_is_measured() {
    let mut m = ProgressModel::default();
    assert_eq!(m.rate_eta(), None, "no denominator, no rate");
    m.apply(ProgressEvent::Preflight {
        disk_gb: 0.0,
        free_gb: 100.0,
    });
    m.apply(ProgressEvent::ShardStart {
        shard: 1,
        total: 2,
        name: "s".into(),
    });
    assert_eq!(m.rate_eta(), None, "a zero-GB checkpoint has no rate");
    m.disk_gb = 10.0;
    m.shard_total = 0;
    assert_eq!(
        m.rate_eta(),
        None,
        "and a zero shard total is not a division"
    );
}

#[test]
fn a_ready_before_any_shard_leaves_the_load_window_unopened() {
    // The backstop freeze must not stamp a window that never opened, or the
    // panel would report a load time for a load it never saw.
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::Ready { port: 8888 });
    assert_eq!(m.load_secs(), None);
    assert_eq!(m.rate_eta(), None);
}

#[test]
fn easing_converges_on_the_target_and_snaps_rather_than_creeping() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::ShardDone {
        shard: 4,
        total: 4,
        used_gb: 1.0,
        free_gb: 1.0,
    });
    assert_eq!(m.displayed_overall(), 0.0, "the bar starts where it was");
    for _ in 0..20 {
        m.ease_tick();
        assert!(m.displayed_overall().is_finite());
    }
    assert_eq!(
        m.displayed_overall(),
        1.0,
        "the last fraction of a percent snaps; an asymptote never fills the bar"
    );
}

#[test]
fn easing_a_model_that_knows_nothing_stays_at_zero() {
    let mut m = ProgressModel::default();
    for _ in 0..5 {
        m.ease_tick();
    }
    assert_eq!(m.displayed_overall(), 0.0);
}

#[test]
fn the_memory_sparkline_is_bounded() {
    let mut m = ProgressModel::default();
    for shard in 1..=200u64 {
        m.apply(ProgressEvent::ShardDone {
            shard,
            total: 500,
            used_gb: shard as f64,
            free_gb: 0.0,
        });
    }
    assert_eq!(
        m.mem_history.len(),
        64,
        "the history is a window, not a log"
    );
    assert_eq!(
        m.mem_history.last().copied(),
        Some(2000),
        "and it keeps the NEWEST samples"
    );
}

#[test]
fn a_phase_index_past_the_checklist_completes_it_instead_of_panicking() {
    // The index arrives over a wire from an instrumented call site; a build
    // that adds a phase must not index out of a shorter table.
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::Phase {
        phase: 99,
        name: "from a newer build".into(),
    });
    assert!(m.phases.iter().all(|p| p.state == PhaseState::Done));
    let (done, total, secs) = m.phase_counts();
    assert_eq!((done, total), (PHASE_NAMES.len(), PHASE_NAMES.len()));
    assert!(secs >= 0.0 && secs.is_finite());
}

#[test]
fn re_entering_the_phase_that_is_running_does_not_restart_its_clock() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::Phase {
        phase: 5,
        name: "weight load".into(),
    });
    let started = m.phases[5].started.expect("running phases are stamped");
    m.apply(ProgressEvent::Phase {
        phase: 5,
        name: "weight load".into(),
    });
    assert_eq!(m.phases[5].started, Some(started));
    assert_eq!(m.phases[5].state, PhaseState::Running);
}

#[test]
fn phase_counts_start_at_none_done() {
    let m = ProgressModel::default();
    let (done, total, _) = m.phase_counts();
    assert_eq!(done, 0);
    assert_eq!(total, PHASE_NAMES.len());
    assert_eq!(total, 12, "the checklist the Main tab draws");
}

#[test]
fn preflight_and_layer_events_land_where_the_panel_reads_them() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::Preflight {
        disk_gb: 42.5,
        free_gb: 96.0,
    });
    assert_eq!(m.disk_gb, 42.5);
    assert_eq!(m.gpu_free_gb, 96.0);
    m.apply(ProgressEvent::Layer {
        layer: 39,
        total: 40,
    });
    assert_eq!((m.layer, m.layer_total), (39, 40));
}

#[test]
fn a_shard_done_with_no_total_does_not_close_the_load_window() {
    // A zero total means the loader does not know how many shards there are;
    // treating that as "the last one" would stop the clock on the first shard.
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::ShardStart {
        shard: 1,
        total: 0,
        name: "s".into(),
    });
    m.apply(ProgressEvent::ShardDone {
        shard: 1,
        total: 0,
        used_gb: 1.0,
        free_gb: 1.0,
    });
    assert_eq!(m.load_secs(), None, "the load is still running");
}

#[test]
fn the_shard_bar_fills_on_done_and_only_resets_on_a_new_shard() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::ShardStart {
        shard: 1,
        total: 4,
        name: "s1".into(),
    });
    assert_eq!(m.shard_target(), 0.0);
    m.apply(ProgressEvent::ShardDone {
        shard: 1,
        total: 4,
        used_gb: 1.0,
        free_gb: 1.0,
    });
    assert_eq!(m.shard_target(), 1.0);
    // A repeat of the same shard_start must not blank a filled bar.
    m.apply(ProgressEvent::ShardStart {
        shard: 1,
        total: 4,
        name: "s1".into(),
    });
    assert_eq!(m.shard_target(), 1.0);
}

#[test]
fn ready_records_the_port_and_the_time_it_took() {
    let mut m = ProgressModel::default();
    m.apply(ProgressEvent::Ready { port: 8890 });
    assert!(m.ready);
    assert_eq!(m.port, 8890);
    assert!(
        m.ready_in_secs >= 0.0 && m.ready_in_secs.is_finite(),
        "{}",
        m.ready_in_secs
    );
}
