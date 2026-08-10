// SPDX-License-Identifier: AGPL-3.0-only

//! Per-token classification masks, derived from the resolved tokenizer.
//!
//! Each mask is **indexed by token id**, which is what makes it dangerous to
//! keep anywhere the vocabulary that produced it cannot reach. Index `N` means
//! one token under one tokenizer and a different token under another; a mask
//! that outlives its vocabulary does not fail, it silently classifies the wrong
//! ids and the logit processors suppress the wrong tokens.
//!
//! These used to live in three process-wide `OnceLock`s beside a `set_*` for
//! each — the only tokenizer-derived values in `TokenizerRuntime` that were not
//! simply returned to the caller like their siblings (`think_end_token`,
//! `tool_call_start_token`, `grammar_engine`, …). They are returned now.
//!
//! Every field is optional and every reader must treat `None` as "this guard is
//! inert". That is the pre-existing fail-open contract: a mask that could not be
//! built disables its feature rather than blocking generation.

use std::sync::Arc;

/// The three token-classification masks for one vocabulary.
///
/// Cheap to clone — three `Arc`s — so it is passed by value where that reads
/// better and by reference on the hot decode path.
#[derive(Clone, Default)]
pub struct VocabMasks {
    /// `mask[id]` iff token `id` decodes to a pure ASCII-digit run (optionally
    /// one leading space). Drives the digit-normalized content-loop path.
    /// `None` → that path is inert; the exact detector is unaffected.
    pub numeric: Option<Arc<[bool]>>,
    /// `mask[id]` iff token `id` decodes to text ending in a well-formed
    /// generation boundary — a newline, or sentence-ending punctuation
    /// optionally trailed by a closing quote or whitespace. Drives
    /// rollback-to-boundary. `None` → rollback finds no boundary and the
    /// watchdog falls back to its hard stop.
    pub boundary: Option<Arc<[bool]>>,
    /// `mask[id]` iff token `id` decodes to text whose last character is
    /// alphanumeric, i.e. emitting `</think>` right after it would split a
    /// word. `None` → the suppression is skipped.
    pub mid_word: Option<Arc<[bool]>>,
}

impl VocabMasks {
    /// True when a token id is classified by the numeric mask. Folds the
    /// "mask absent" and "id out of range" cases into the fail-open answer, so
    /// callers do not each re-derive it.
    pub fn is_numeric(&self, id: u32) -> bool {
        Self::at(&self.numeric, id)
    }

    pub fn is_boundary(&self, id: u32) -> bool {
        Self::at(&self.boundary, id)
    }

    pub fn is_mid_word(&self, id: u32) -> bool {
        Self::at(&self.mid_word, id)
    }

    fn at(mask: &Option<Arc<[bool]>>, id: u32) -> bool {
        mask.as_deref()
            .and_then(|m| m.get(id as usize))
            .copied()
            .unwrap_or(false)
    }
}

impl std::fmt::Debug for VocabMasks {
    /// Prints the population of each mask rather than several thousand bools.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = |m: &Option<Arc<[bool]>>| match m.as_deref() {
            Some(m) => format!("{}/{}", m.iter().filter(|b| **b).count(), m.len()),
            None => "none".to_string(),
        };
        f.debug_struct("VocabMasks")
            .field("numeric", &count(&self.numeric))
            .field("boundary", &count(&self.boundary))
            .field("mid_word", &count(&self.mid_word))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(bits: &[bool]) -> Option<Arc<[bool]>> {
        Some(Arc::from(bits.to_vec()))
    }

    #[test]
    fn an_absent_mask_classifies_nothing() {
        let m = VocabMasks::default();
        assert!(!m.is_numeric(0) && !m.is_boundary(7) && !m.is_mid_word(u32::MAX));
    }

    #[test]
    fn an_id_past_the_end_is_unclassified_rather_than_a_panic() {
        // The failure mode this guards: a mask built for a SMALLER vocabulary.
        // Fail-open is the pre-existing contract — the guard goes inert, it
        // does not index out of bounds.
        let m = VocabMasks {
            numeric: mask(&[true, false]),
            ..Default::default()
        };
        assert!(m.is_numeric(0));
        assert!(!m.is_numeric(1));
        assert!(!m.is_numeric(2), "past the end of a 2-token vocabulary");
        assert!(!m.is_numeric(50_000));
    }

    #[test]
    fn the_three_masks_are_independent() {
        let m = VocabMasks {
            numeric: mask(&[true, false]),
            boundary: mask(&[false, true]),
            mid_word: None,
        };
        assert!(m.is_numeric(0) && !m.is_boundary(0) && !m.is_mid_word(0));
        assert!(!m.is_numeric(1) && m.is_boundary(1) && !m.is_mid_word(1));
    }

    #[test]
    fn debug_reports_populations_not_contents() {
        let m = VocabMasks {
            numeric: mask(&[true, false, true]),
            ..Default::default()
        };
        let s = format!("{m:?}");
        assert!(s.contains("2/3"), "{s}");
        assert!(s.contains("none"), "{s}");
        assert!(!s.contains("true"), "must not print the bools: {s}");
    }
}
