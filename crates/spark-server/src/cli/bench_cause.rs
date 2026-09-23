// SPDX-License-Identifier: AGPL-3.0-only

//! The last thing a failed child said, for an error that has to outlive its
//! log.
//!
//! Two logs sit between a gate failure and whoever reads it. A leased serve
//! writes to `serve-lease.log` on the node, append-only across a campaign
//! (126 MB on 2026-09-22); the gate child writes to `child.log`, which the
//! agent hands back. When the serve dies at boot the whole diagnosis is the
//! `Error:` / `Caused by:` block at the end of the first, and until #1242 the
//! certify orchestrator printed "the node returned no record" instead. Both
//! readers use this: the lease embeds the serve's block in its own error, and
//! placement embeds the child's block in its refusal.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// How much of a log's tail is read to find its final block. A boot failure's
/// block is a few hundred bytes; a serve that logged a weight-loading progress
/// line per shard before dying is still well inside this.
pub(crate) const TAIL_BYTES: u64 = 64 * 1024;

/// The longest block an error carries. A `Caused by:` chain is a few lines;
/// anything past this is a log, not a cause, and stays in the file.
pub(crate) const CAUSE_CAP: usize = 4096;

/// The last `Error:` block in `text`: from the final line that starts with
/// `Error:` — the column-0 line anyhow's `Debug` rendering opens with, the
/// same anchor a reader scans for — to the end, trimmed and capped. `None`
/// when the text has no such line, so the caller says "see the log" rather
/// than quoting something that is not an error.
pub(crate) fn final_error_block(text: &str) -> Option<String> {
    let start = text
        .match_indices("Error:")
        .filter(|(i, _)| *i == 0 || text.as_bytes()[i - 1] == b'\n')
        .map(|(i, _)| i)
        .next_back()?;
    let block = text[start..].trim();
    if block.len() <= CAUSE_CAP {
        return Some(block.to_string());
    }
    let mut cut = CAUSE_CAP;
    while !block.is_char_boundary(cut) {
        cut -= 1;
    }
    Some(format!(
        "{}… (+{} bytes in the log)",
        &block[..cut],
        block.len() - cut
    ))
}

/// The last `bytes` of a file, lossily decoded, starting at a line boundary
/// when the read began mid-line and a whole line fits — a window that holds
/// only the tail of one line returns that tail rather than nothing. A file
/// shorter than `bytes` is read whole.
pub(crate) fn tail_of_file(path: &Path, bytes: u64) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let mut text = String::new();
    if len > bytes {
        f.seek(SeekFrom::Start(len - bytes))?;
        let mut raw = Vec::new();
        f.read_to_end(&mut raw)?;
        let cut = match raw.iter().position(|b| *b == b'\n') {
            Some(i) if i + 1 < raw.len() => i + 1,
            _ => 0,
        };
        text = String::from_utf8_lossy(&raw[cut..]).into_owned();
    } else {
        f.read_to_string(&mut text)?;
    }
    Ok(text)
}

/// `block` with every line indented by four spaces, so a block embedded in
/// another error keeps that error's own `Error:` line as the outermost one
/// [`final_error_block`] finds.
pub(crate) fn indented(block: &str) -> String {
    block
        .lines()
        .map(|l| format!("    {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
#[path = "bench_cause_tests.rs"]
mod tests;
