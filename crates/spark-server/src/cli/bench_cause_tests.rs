// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The block the 2026-09-22 `bfcl-subset[0/6]` failure left at the end of
/// serve-lease.log, verbatim in shape.
const KV_OOM: &str = "native FP8 dense residency: weights 23.42 GB ...\n\
Error: Failed to build model\n\
\n\
Caused by:\n    \
No memory left for KV cache: total GPU = 121.7 GB, --gpu-memory-utilization 70% \
→ budget 85.2 GB, but 69.9 GB already consumed + 23.7 GB inference reserve = \
93.7 GB committed.\n";

#[test]
fn the_final_block_starts_at_the_last_column_zero_error_line() {
    let block = final_error_block(KV_OOM).expect("the log ends in an error");
    assert!(block.starts_with("Error: Failed to build model"), "{block}");
    assert!(block.contains("Caused by:"), "{block}");
    assert!(block.contains("No memory left for KV cache"), "{block}");
    assert!(!block.contains("native FP8 dense residency"), "{block}");
    // Two earlier boots in the same append-only log: only the LAST counts.
    let twice = format!("{KV_OOM}\n...a later boot...\nError: something else\n");
    assert_eq!(
        final_error_block(&twice).as_deref(),
        Some("Error: something else")
    );
    // A mid-line "Error:" is prose, not an anchor; a log that never errored
    // yields nothing to quote.
    assert_eq!(
        final_error_block("all good\nlogged Error: in prose\n"),
        None
    );
    assert_eq!(final_error_block(""), None);
    assert_eq!(
        final_error_block("Error: at byte zero"),
        Some("Error: at byte zero".into())
    );
}

#[test]
fn a_long_block_is_capped_and_says_how_much_stayed_in_the_log() {
    let long = format!("Error: {}", "é".repeat(CAUSE_CAP));
    let block = final_error_block(&long).unwrap();
    assert!(block.len() < long.len());
    assert!(block.contains("… (+"), "{block}");
    assert!(block.contains("bytes in the log)"), "{block}");
    // Cut on a char boundary, never inside one.
    assert!(std::str::from_utf8(block.as_bytes()).is_ok());
}

#[test]
fn the_tail_of_a_file_reads_whole_lines_and_a_short_file_whole() {
    let dir = std::env::temp_dir().join(format!("bench-cause-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("serve-lease.log");
    // Ten 5-byte lines, then a 9-byte error line.
    let text = format!("{}Error: x\n", "aaaa\n".repeat(10));
    std::fs::write(&p, &text).unwrap();
    assert_eq!(
        tail_of_file(&p, 1 << 20).unwrap(),
        text,
        "shorter than the window: whole"
    );
    // A 12-byte window opens mid-line ("aa\nError: x\n"): the partial line
    // is dropped and the read starts at the boundary.
    assert_eq!(tail_of_file(&p, 12).unwrap(), "Error: x\n");
    // A window that holds only the end of the last line returns that end
    // rather than an empty string — the earlier draft did the latter.
    assert_eq!(tail_of_file(&p, 5).unwrap(), "r: x\n");
    assert!(tail_of_file(&dir.join("absent"), 10).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_embedded_block_is_indented_so_the_outer_error_stays_the_anchor() {
    let inner = final_error_block(KV_OOM).unwrap();
    let outer = format!(
        "Error: the leased server exited — serve-lease.log ends with:\n{}",
        indented(&inner)
    );
    let found = final_error_block(&outer).unwrap();
    assert!(
        found.starts_with("Error: the leased server exited"),
        "{found}"
    );
    assert!(found.contains("No memory left for KV cache"), "{found}");
}
