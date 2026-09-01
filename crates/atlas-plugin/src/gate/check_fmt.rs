// SPDX-License-Identifier: AGPL-3.0-only

//! Message rendering for `check.rs`, split out at the 500-LoC cap. Exact
//! piecewise move — no logic changed.
//!
//! ★ Deliberately NOT a boundary file, and this is the reason it was the block
//! chosen to move. `hardening_tests` pins every verdict-DECIDING symbol inside
//! `BOUNDARY_FILES`; this decides nothing — it turns a list of paths into one
//! readable line. Moving `record_covers` or `invalidating_paths` here instead
//! would have put a verdict outside the boundary, which is the exact shape
//! `coverage.rs` calls "a lock whose key is kept inside it".

/// Render invalidating paths for a one-line message: a few names, then a count.
///
/// A refactor can touch hundreds of files, and pasting all of them buries the
/// verdict it is attached to.
pub(super) fn summarize_paths(paths: &[String]) -> String {
    const SHOWN: usize = 3;
    if paths.len() <= SHOWN {
        return paths.join(", ");
    }
    format!(
        "{} and {} more",
        paths[..SHOWN].join(", "),
        paths.len() - SHOWN
    )
}
