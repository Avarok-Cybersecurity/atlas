// SPDX-License-Identifier: AGPL-3.0-only

//! Loop-watchdog cases that need a realistic stream to reproduce.
//!
//! `stream_guards`' own inline tests cover the trivial paths (already
//! triggered, empty text, four identical adjacent lines, buffer trim).
//! The two failures the watchdog was actually rebuilt for — a repeat
//! separated by kilobytes of unique prose, and a repeat whose last
//! instance begins mid-line — need a stream long enough to exercise the
//! window and the substring fallback, so they live here.

use crate::api::stream_guards::check_loop_watchdog;

const PHRASE: &str = "I'll create the project files and verify everything works:";

/// Feed `text` in `chunk` byte slices, as SSE deltas arrive, and report
/// whether the watchdog fired at any point.
fn fires(text: &str, chunk: usize) -> bool {
    let mut scan = String::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + chunk).min(text.len());
        while !text.is_char_boundary(end) {
            end += 1;
        }
        if check_loop_watchdog(&text[start..end], &mut scan, false) {
            return true;
        }
        start = end;
    }
    false
}

/// `n` copies of `PHRASE`, each followed by ~1.5 KB of filler that is
/// unique per repetition — so only the phrase itself repeats and the
/// watchdog cannot fire on the interstitial.
fn phrase_with_unique_interstitials(n: usize) -> String {
    let mut feed = String::new();
    for i in 0..n {
        feed.push_str(PHRASE);
        feed.push('\n');
        for j in 0..40 {
            feed.push_str(&format!(
                "segment {i}.{j} of the source dump under review\n"
            ));
        }
    }
    feed
}

/// Token-sized SSE deltas. The watchdog only ever inspects the LAST line
/// in its buffer, so what it sees depends on where a chunk boundary
/// happens to land; sweeping the realistic sizes keeps a passing result
/// from being one lucky alignment.
const TOKEN_SIZES: std::ops::RangeInclusive<usize> = 1..=16;

#[test]
fn a_repeat_separated_by_kilobytes_of_unique_prose_still_fires() {
    // The claude-export.txt failure: the phrase recurred four times but
    // each instance sat behind ~3 KB of source-dump prose, so under the
    // old 3 KB window only one copy was ever in view. The window is 8 KB
    // (trimmed from 10 KB) precisely so this stays visible.
    //
    // The filler is unique per repetition on purpose. The version of this
    // test that predated the rewrite reused the SAME filler every time, so
    // the watchdog could fire on the repeated FILLER and the test passed
    // without the phrase ever mattering.
    let feed = phrase_with_unique_interstitials(5);
    assert!(
        feed.len() > 3_000,
        "the interstitials must exceed the old 3 KB window to be a regression test; got {}",
        feed.len()
    );
    for chunk in TOKEN_SIZES {
        assert!(
            fires(&feed, chunk),
            "5 repeats across large unique interstitials must fire at chunk size {chunk}"
        );
    }
}

#[test]
fn three_repeats_do_not_fire() {
    // The threshold is 4. Firing at 3 truncates legitimate output —
    // "build / fix / build" cycles repeat their intro line innocently —
    // and a truncated response is a worse failure than a long one.
    let feed = phrase_with_unique_interstitials(3);
    for chunk in TOKEN_SIZES {
        assert!(
            !fires(&feed, chunk),
            "three repeats are below the threshold and must not stop the stream \
             (chunk size {chunk})"
        );
    }
}

#[test]
fn a_repeat_whose_last_instance_starts_mid_line_still_fires() {
    // export.txt line 919: the 4th instance was glued to other text
    // ("…everything works:        let body ="), which defeats exact-line
    // equality. The substring scan over >=30-char candidates is what
    // catches it.
    let mut feed = String::new();
    for _ in 0..3 {
        feed.push_str(PHRASE);
        feed.push('\n');
        feed.push_str("intermediate prose\n");
    }
    feed.push_str(PHRASE);
    feed.push_str("        let body = vec![];");
    for chunk in TOKEN_SIZES {
        assert!(
            fires(&feed, chunk),
            "substring scan must catch a repeated phrase whose last instance \
             is mid-line (chunk size {chunk})"
        );
    }
}

#[test]
fn ordinary_varied_prose_never_fires() {
    // The watchdog stops generation, so a false positive silently
    // truncates a good answer. Long, non-repeating output must survive.
    let mut feed = String::new();
    for i in 0..200 {
        feed.push_str(&format!(
            "Step {i}: inspect the {i}th module and record what it exports.\n"
        ));
    }
    assert!(!fires(&feed, 64), "varied prose must not trip the watchdog");
}

#[test]
fn a_repeating_short_line_does_not_fire() {
    // Lines under 16 trimmed chars are ignored on purpose: source code
    // legitimately repeats short lines (`}`, `self.buf.clear();`) and
    // stopping on those would truncate every code-heavy answer.
    let feed = "}\n".repeat(50);
    assert!(!fires(&feed, 8), "short repeated lines must be ignored");
}
