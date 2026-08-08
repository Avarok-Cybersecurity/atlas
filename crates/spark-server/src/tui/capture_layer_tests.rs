// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use tracing_subscriber::layer::SubscriberExt;

/// Run `f` with the layer installed and collect everything it decoded.
fn captured(f: impl FnOnce()) -> Vec<ProgressEvent> {
    let (tx, rx) = std::sync::mpsc::channel();
    let sub = tracing_subscriber::registry().with(ProgressCaptureLayer::new(tx));
    tracing::subscriber::with_default(sub, f);
    rx.try_iter().collect()
}

#[test]
fn decodes_progress_events_and_ignores_others() {
    let got = captured(|| {
        spark_runtime::progress::phase(3, "gpu init");
        spark_runtime::progress::shard_start(2, 26, "model-00002.safetensors");
        tracing::info!("an ordinary log line");
        spark_runtime::progress::ready(8888);
    });
    assert_eq!(
        got,
        vec![
            ProgressEvent::Phase {
                phase: 3,
                name: "gpu init".into()
            },
            ProgressEvent::ShardStart {
                shard: 2,
                total: 26,
                name: "model-00002.safetensors".into()
            },
            ProgressEvent::Ready { port: 8888 },
        ]
    );
}

#[test]
fn every_event_the_loader_emits_round_trips() {
    // The Main tab's whole state machine is fed from here; a variant that
    // decoded to `None` would silently stall a bar rather than fail.
    let got = captured(|| {
        spark_runtime::progress::preflight(42.5, 96.25);
        spark_runtime::progress::shard_done(1, 4, 12.5, 83.75);
        spark_runtime::progress::layer(17, 40);
    });
    assert_eq!(
        got,
        vec![
            ProgressEvent::Preflight {
                disk_gb: 42.5,
                free_gb: 96.25
            },
            ProgressEvent::ShardDone {
                shard: 1,
                total: 4,
                used_gb: 12.5,
                free_gb: 83.75
            },
            ProgressEvent::Layer {
                layer: 17,
                total: 40
            },
        ]
    );
}

#[test]
fn a_long_run_of_events_arrives_in_the_order_it_was_emitted() {
    // The pane replays these into a state machine where order IS the meaning:
    // a shard_done overtaking its shard_start would reset the bar it just
    // filled.
    let n = 500usize;
    let got = captured(|| {
        for shard in 1..=n {
            spark_runtime::progress::shard_start(shard, n, "s");
            spark_runtime::progress::shard_done(shard, n, shard as f64, 0.0);
        }
    });
    assert_eq!(got.len(), n * 2);
    for (i, chunk) in got.chunks(2).enumerate() {
        let shard = i as u64 + 1;
        assert_eq!(
            chunk[0],
            ProgressEvent::ShardStart {
                shard,
                total: n as u64,
                name: "s".into()
            }
        );
        assert_eq!(
            chunk[1],
            ProgressEvent::ShardDone {
                shard,
                total: n as u64,
                used_gb: shard as f64,
                free_gb: 0.0
            }
        );
    }
}

#[test]
fn an_unknown_event_kind_is_dropped_rather_than_guessed() {
    let got = captured(|| {
        tracing::debug!(target: spark_runtime::progress::TARGET, ev = "from-a-newer-build", phase = 1u64);
    });
    assert!(got.is_empty(), "{got:?}");
}

#[test]
fn a_field_wider_than_its_type_saturates_instead_of_wrapping() {
    // `phase` is a u8 and `port` a u16 in the typed event. A wrap would point
    // the checklist at phase 0 and the READY banner at a port nothing is on.
    let got = captured(|| {
        tracing::debug!(target: spark_runtime::progress::TARGET, ev = "phase", phase = 300u64, name = "x");
        tracing::debug!(target: spark_runtime::progress::TARGET, ev = "ready", port = 70_000u64);
    });
    assert_eq!(
        got,
        vec![
            ProgressEvent::Phase {
                phase: u8::MAX,
                name: "x".into()
            },
            ProgressEvent::Ready { port: u16::MAX },
        ]
    );
}

#[test]
fn a_negative_count_is_clamped_to_zero_rather_than_becoming_enormous() {
    // An i64 field reaches the u64 bag through `record_i64`; casting -1 would
    // make a shard total of 18 quintillion.
    let got = captured(|| {
        tracing::debug!(target: spark_runtime::progress::TARGET, ev = "layer", layer = -1i64, total = 40i64);
    });
    assert_eq!(
        got,
        vec![ProgressEvent::Layer {
            layer: 0,
            total: 40
        }]
    );
}

#[test]
fn a_missing_field_takes_its_zero_rather_than_dropping_the_event() {
    let got = captured(|| {
        tracing::debug!(target: spark_runtime::progress::TARGET, ev = "shard_start");
    });
    assert_eq!(
        got,
        vec![ProgressEvent::ShardStart {
            shard: 0,
            total: 0,
            name: String::new()
        }]
    );
}

#[test]
fn emitting_after_the_dashboard_is_gone_neither_blocks_nor_panics() {
    // The TUI thread owns the receiving end and may exit at any point; the
    // server has to keep serving, so a send into a dead channel is ignored.
    let (tx, rx) = std::sync::mpsc::channel();
    drop(rx);
    let sub = tracing_subscriber::registry().with(ProgressCaptureLayer::new(tx));
    tracing::subscriber::with_default(sub, || {
        for shard in 1..=1000usize {
            spark_runtime::progress::shard_start(shard, 1000, "s");
        }
        spark_runtime::progress::ready(8888);
    });
}

#[test]
fn an_undrained_channel_does_not_stall_the_loader() {
    // The channel is unbounded on purpose: the drain happens on the UI tick, so
    // a paused or slow dashboard must never apply backpressure to weight
    // loading. Every event is still there when it does drain.
    let (tx, rx) = std::sync::mpsc::channel();
    let sub = tracing_subscriber::registry().with(ProgressCaptureLayer::new(tx));
    tracing::subscriber::with_default(sub, || {
        for layer in 0..10_000usize {
            spark_runtime::progress::layer(layer, 10_000);
        }
    });
    assert_eq!(rx.try_iter().count(), 10_000);
}
