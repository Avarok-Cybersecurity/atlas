// SPDX-License-Identifier: AGPL-3.0-only

//! Copying to the clipboard from inside the dashboard, over OSC 52.
//!
//! # Why OSC 52 and not a clipboard crate
//!
//! The dashboard is very often driven over SSH — that is the normal way these
//! boxes are used. A clipboard crate (`arboard`, `copypasta`) talks to the
//! X11/Wayland server on the machine the PROCESS runs on, which is the wrong
//! machine: it would put text on dgx1's clipboard while the human is sitting at
//! a laptop. It also drags in C-built X11 bindings.
//!
//! OSC 52 asks the TERMINAL to set the clipboard, so the text lands where the
//! keyboard is, works over SSH and tmux, and needs nothing but a write to
//! stdout.
//!
//! # What can go wrong, and why this is still worth doing
//!
//! The terminal has to support it, and some require opting in — tmux needs
//! `set -g set-clipboard on`, and a few terminals disable OSC 52 by default as
//! a security measure (a program that can write to your clipboard can stage a
//! paste). There is no reply to parse, so we cannot know whether it worked;
//! the toast therefore says what was SENT, not what was received.

use base64::Engine as _;

/// Terminals commonly cap the escape sequence; beyond this the write is
/// silently dropped or, worse, truncated into the clipboard.
///
/// 74994 is the documented practical ceiling for the widely-used
/// implementations. Sending less and saying so beats sending garbage.
const MAX_BYTES: usize = 74_994;

/// The bytes to write to the terminal to set its clipboard to `text`.
///
/// Split from the write so the encoding is testable without a terminal, which
/// is the only part that can be wrong in an interesting way.
pub fn osc52(text: &str) -> Option<Vec<u8>> {
    if text.is_empty() {
        return None;
    }
    let b64 = base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
    if b64.len() > MAX_BYTES {
        return None;
    }
    // `52;c;` — `c` is the CLIPBOARD selection (as opposed to `p`, primary).
    // BEL-terminated rather than ST: both are legal, BEL is accepted by more
    // terminals in practice.
    Some(format!("\x1b]52;c;{b64}\x07").into_bytes())
}

/// Would this text be rejected for being too large?
pub fn too_large(text: &str) -> bool {
    !text.is_empty() && osc52(text).is_none()
}

/// Put `text` on the terminal's clipboard.
///
/// Returns what to tell the user. `Ok` does NOT mean the terminal accepted it —
/// OSC 52 has no reply — only that it was written.
pub fn copy(text: &str) -> Result<usize, String> {
    if text.is_empty() {
        return Err("nothing selected".into());
    }
    let Some(seq) = osc52(text) else {
        return Err(format!(
            "selection is too large to copy ({} chars; the limit is about {} )",
            text.chars().count(),
            MAX_BYTES / 4 * 3
        ));
    };
    write_raw(&seq).map_err(|e| format!("could not write to the terminal: {e}"))?;
    Ok(text.chars().count())
}

/// Write straight to stdout, bypassing the ratatui backend.
///
/// The sequence must reach the terminal unmodified and immediately; going
/// through the frame buffer would either escape it or defer it to the next
/// draw.
fn write_raw(bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    out.write_all(bytes)?;
    out.flush()
}

#[cfg(test)]
#[path = "clipboard_tests.rs"]
mod tests;
