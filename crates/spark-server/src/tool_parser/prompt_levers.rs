// SPDX-License-Identifier: AGPL-3.0-only

//! [`PromptLevers`] — the per-model decisions that change how a prompt is
//! *rendered*, carried to the renderers rather than read from a global.
//!
//! These come from MODEL.toml `[behavior]`, so they are model-scoped by
//! definition: TSCG's TAS operator picks its delimiters from the model's BPE
//! merges, and a gain on one tokenizer implies nothing about another. Held in
//! a process global, a hot-swap would render the *next* model's tool schemas
//! under the *previous* model's setting — silently, since both paths produce a
//! valid prompt and the difference only shows up as tool-call accuracy.
//!
//! It is a struct rather than a bare `bool` because the prompt-rendering seam
//! is the natural home for the next such lever, and widening a struct that is
//! already threaded costs one line instead of another signature sweep.

/// Prompt-rendering decisions for one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PromptLevers {
    /// `[behavior].tscg` — compile tool schemas into compact function
    /// signatures instead of embedding the raw JSON. Default `false`, which is
    /// the byte-identical pre-TSCG path.
    pub tscg: bool,
}

impl PromptLevers {
    /// Every lever off — the unmodified JSON path. Used by the parser unit
    /// tests, which would otherwise need a `[behavior]` table to render a
    /// prompt.
    pub const OFF: Self = Self { tscg: false };

    pub fn new(tscg: bool) -> Self {
        Self { tscg }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_the_default_and_the_json_path() {
        assert_eq!(PromptLevers::default(), PromptLevers::OFF);
        const { assert!(!PromptLevers::OFF.tscg) };
    }

    #[test]
    fn two_models_can_disagree_within_one_process() {
        // The property the `OnceLock<bool>` could not have: `set_tscg_enabled`
        // was idempotent, so the second model to load kept the first's answer.
        let a = PromptLevers::new(true);
        let b = PromptLevers::new(false);
        assert!(a.tscg && !b.tscg);
    }
}
