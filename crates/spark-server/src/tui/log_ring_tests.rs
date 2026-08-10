// SPDX-License-Identifier: AGPL-3.0-only

//! The ring is process-global and every test in this binary shares it, so each
//! case takes `SERIAL` before touching it: two tests interleaving pushes would
//! make any assertion about ORDER or about what was evicted meaningless.

use super::*;

use parking_lot::Mutex;

/// STATIC, DELIBERATELY — the thing under test is itself a process-global.
static SERIAL: Mutex<()> = Mutex::new(());

fn line(message: &str) -> LogLine {
    LogLine {
        at: SystemTime::now(),
        level: Level::INFO,
        target: "t".into(),
        message: message.into(),
    }
}

fn push_all(messages: &[String]) {
    for m in messages {
        push(line(m));
    }
}

fn tagged(tag: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{tag}-{i}")).collect()
}

fn messages(n: usize) -> Vec<String> {
    tail(n).into_iter().map(|l| l.message).collect()
}

#[test]
fn ring_caps_and_tails() {
    let _serial = SERIAL.lock();
    for i in 0..(CAP + 10) {
        push(LogLine {
            at: SystemTime::now(),
            level: Level::INFO,
            target: "t".into(),
            message: format!("m{i}"),
        });
    }
    let t = tail(5);
    assert_eq!(t.len(), 5);
    assert_eq!(t.last().unwrap().message, format!("m{}", CAP + 9));
    assert!(seq() >= (CAP + 10) as u64);
}

#[test]
fn lines_come_back_oldest_first() {
    let _serial = SERIAL.lock();
    let want = tagged("order", 6);
    push_all(&want);
    assert_eq!(messages(want.len()), want);
}

#[test]
fn the_oldest_line_is_the_one_dropped_at_capacity() {
    let _serial = SERIAL.lock();
    push(line("evict-me"));
    push_all(&tagged("filler", CAP));
    // CAP younger lines have arrived since, so the marker is exactly one line
    // past the end of the ring.
    assert_eq!(tail(CAP + 10).len(), CAP, "the ring never grows past CAP");
    assert!(
        !tail(CAP).iter().any(|l| l.message == "evict-me"),
        "the oldest line goes first"
    );
}

#[test]
fn a_tail_of_zero_is_empty_and_a_tail_past_the_end_is_not_padded() {
    let _serial = SERIAL.lock();
    let want = tagged("edge", 3);
    push_all(&want);
    assert!(tail(0).is_empty());
    assert_eq!(
        messages(1),
        vec!["edge-2".to_string()],
        "a tail of one is newest"
    );
    let everything = tail(usize::MAX);
    assert!(
        everything.len() <= CAP,
        "asking for more cannot invent lines"
    );
    assert_eq!(
        everything.last().map(|l| l.message.clone()),
        Some("edge-2".to_string())
    );
}

#[test]
fn a_line_far_larger_than_the_pane_does_not_corrupt_its_neighbours() {
    // Nothing truncates on the way in, so an enormous message must stay one
    // entry rather than spilling into the lines around it.
    let _serial = SERIAL.lock();
    let huge = "x".repeat(1 << 20);
    push(line("before-huge"));
    push(line(&huge));
    push(line("after-huge"));
    let got = messages(3);
    assert_eq!(got[0], "before-huge");
    assert_eq!(got[1].len(), huge.len());
    assert_eq!(got[2], "after-huge");

    let mut dump = Vec::new();
    dump_to(&mut dump, 3);
    let text = String::from_utf8(dump).expect("the dump stays valid utf-8");
    assert_eq!(text.lines().count(), 3, "one line in, one line out");
}

#[test]
fn unicode_survives_the_ring_and_the_panic_dump() {
    // The dump is what a panicking process prints; a byte-slicing bug here
    // would panic inside the panic hook.
    let _serial = SERIAL.lock();
    let odd = [
        "loaded 27B — 4.2 GB/s ✓",
        "日本語のログ行",
        "e\u{0301}\u{0301}\u{0301} combining",
        "\u{1F680}\u{1F9EA}\u{200D}\u{1F525}",
        "\u{202E}rtl-override",
        "nul\u{0}embedded",
    ];
    for m in odd {
        push(line(m));
    }
    let got = messages(odd.len());
    assert_eq!(got, odd);

    let mut dump = Vec::new();
    dump_to(&mut dump, odd.len());
    let text = String::from_utf8(dump).expect("the dump stays valid utf-8");
    for m in odd {
        assert!(text.contains(m), "{m:?} missing from the dump");
    }
}

#[test]
fn seq_counts_every_line_ever_pushed_not_the_ring_length() {
    // The pane's "n new lines while scrolled up" badge is this difference; a
    // count that stalled at CAP would report no new lines on a busy load.
    let _serial = SERIAL.lock();
    let before = seq();
    push_all(&tagged("seq", CAP + 5));
    assert_eq!(seq() - before, (CAP + 5) as u64);
}

#[test]
fn the_layer_captures_events_and_leaves_the_progress_channel_alone() {
    use tracing_subscriber::layer::SubscriberExt;

    let _serial = SERIAL.lock();
    let before = seq();
    let sub = tracing_subscriber::registry().with(LogRingLayer);
    tracing::subscriber::with_default(sub, || {
        tracing::warn!("captured by the pane");
        // Progress has its own typed channel; duplicating it here would fill
        // the pane with the shard events the Main tab already draws.
        spark_runtime::progress::ready(8888);
    });
    assert_eq!(seq() - before, 1, "only the ordinary event was captured");
    let last = tail(1).pop().expect("one line");
    assert_eq!(last.message, "captured by the pane");
    assert_eq!(last.level, Level::WARN);
    assert!(
        last.target.contains("log_ring"),
        "the target is kept for the pane's dim column, got {:?}",
        last.target
    );
}
