// SPDX-License-Identifier: AGPL-3.0-only
//! Whether the prefill-end ("exact leaf") SSM snapshot earns a pool slot.
//!
//! Every prefill on an SSM model used to save THREE snapshots: the tail-split
//! checkpoint one block below the last full block, the exact leaf at the
//! prompt's end, and later the finish leaf. The exact leaf is unusable for an
//! identical repeated prompt by construction (`prefix_lookup` bypasses the
//! exact full-prompt shortcut by default — computing the last token needs
//! state@(N-1), the snapshot holds state@N, and the recurrence is not
//! invertible), and for a prompt that EXTENDS this one it saves at most
//! `2 * block_size` tokens of replay over the tail checkpoint, which the
//! suffix pass folds in anyway. What it costs is a third of the pool: measured
//! 2026-09-13 on the concurrency sweep's 8-slot pool, the tail checkpoint —
//! the only restorable anchor for the sweep's repeat pattern — was the oldest
//! of the three and was LRU-evicted before the two unusable leaves, so from
//! two concurrent prompts upward every warm request recomputed its SSM state
//! (`matched=657 snapshot_tokens=657`: the exact leaf survived, the tail did
//! not). Skipping the redundant leaf is what lets a small pool hold what a
//! warm request can actually use.

/// The cases where the exact leaf still earns its slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactLeaf {
    /// Save it: nothing else anchors this prompt's end.
    Save,
    /// Save it: the exact shortcut is enabled and reads the stashed hidden
    /// only this snapshot carries.
    SaveForShortcut,
    /// Skip it: the tail checkpoint at `tail` covers every restore this leaf
    /// could serve, within `replay` tokens of SSM work.
    Redundant { tail: usize, replay: usize },
}

/// Decide, from what this prefill already saved.
///
/// `tail_checkpoint` is the token count of the tail-split checkpoint saved
/// during THIS prefill (`None` when the split did not fire — vision prompts,
/// a prompt under two blocks, or a save that lost the pool race).
pub fn exact_leaf(
    tail_checkpoint: Option<usize>,
    total: usize,
    block_size: usize,
    exact_shortcut_enabled: bool,
) -> ExactLeaf {
    if exact_shortcut_enabled {
        return ExactLeaf::SaveForShortcut;
    }
    match tail_checkpoint {
        // A tail more than two blocks back is not the tail split (the cut is
        // exactly one block below the last full block); do not trust it to
        // stand in for the leaf.
        Some(tail) if tail < total && total - tail <= 2 * block_size => ExactLeaf::Redundant {
            tail,
            replay: total - tail,
        },
        _ => ExactLeaf::Save,
    }
}

/// `AVAROK_MARCONI_EXACT=1` re-enables the exact full-prompt shortcut. Read
/// here and nowhere else.
pub fn marconi_exact_enabled() -> bool {
    std::env::var("AVAROK_MARCONI_EXACT").as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_leaf_is_redundant_exactly_when_the_tail_split_fired_for_this_prompt() {
        // 657 tokens, bs 16: tail cut at 640, replay 17.
        assert_eq!(
            exact_leaf(Some(640), 657, 16, false),
            ExactLeaf::Redundant {
                tail: 640,
                replay: 17
            }
        );
        // Block-aligned prompt: cut at total - 2bs, replay 32 — still inside.
        assert_eq!(
            exact_leaf(Some(608), 640, 16, false),
            ExactLeaf::Redundant {
                tail: 608,
                replay: 32
            }
        );
        // NEGATIVE CONTROLS: no tail this prefill; a stale tail far below the
        // prompt end; a tail at or past the end (impossible, refused anyway).
        assert_eq!(exact_leaf(None, 657, 16, false), ExactLeaf::Save);
        assert_eq!(exact_leaf(Some(256), 657, 16, false), ExactLeaf::Save);
        assert_eq!(exact_leaf(Some(657), 657, 16, false), ExactLeaf::Save);
        // The shortcut needs the leaf's hidden: always saved when enabled.
        assert_eq!(
            exact_leaf(Some(640), 657, 16, true),
            ExactLeaf::SaveForShortcut
        );
    }
}
