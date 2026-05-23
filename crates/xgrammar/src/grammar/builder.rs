// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarBuilder — the mutable construction API for the BNF AST.
// Port of `GrammarBuilder` from xgrammar `cpp/grammar_builder.h`.
//
// The builder accumulates rules and expressions, writing expressions
// into the CSR `expr_data`/`expr_indptr` vectors, and finally emits a
// finished [`GrammarData`] via [`GrammarData::from_parts`].
//
// Layout reminder: each CSR slot is `[type, data_len, data0, data1, …]`.
//
// This file holds the core struct + expression-handling methods. Rule
// handling and the error type live in `builder_rules.rs`; tests in
// `builder_tests.rs` — split to keep each file under the 250-line cap.

use std::cell::RefCell;
use std::collections::HashMap;

use super::data::{GrammarData, Rule};
use super::expr::{GrammarExpr, GrammarExprType};

#[path = "builder_rules.rs"]
mod rules;
#[cfg(test)]
#[path = "builder_tests.rs"]
mod tests;

pub use rules::BuilderError;

/// One element of a character class: an inclusive `[lower, upper]`
/// codepoint range. Port of `GrammarBuilder::CharacterClassElement`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterClassElement {
    /// Inclusive lower bound (unicode codepoint).
    pub lower: i32,
    /// Inclusive upper bound (unicode codepoint).
    pub upper: i32,
}

impl CharacterClassElement {
    /// A range `[lower, upper]`.
    pub fn new(lower: i32, upper: i32) -> Self {
        Self { lower, upper }
    }
}

/// Decoded tag-dispatch description handed to [`GrammarBuilder::add_tag_dispatch`].
/// Mirrors [`super::data::TagDispatch`] but is the *input* form: rule ids
/// (not tag byte-string expr ids) are stored directly.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TagDispatchSpec {
    /// `(tag, rule_id)` pairs.
    pub tag_rule_pairs: Vec<(String, i32)>,
    /// If true, EOS may stop the dispatch.
    pub stop_eos: bool,
    /// Strings that stop the dispatch.
    pub stop_str: Vec<String>,
    /// If true, the dispatch loops after dispatching.
    pub loop_after_dispatch: bool,
    /// Strings excluded by the tag dispatch.
    pub excluded_str: Vec<String>,
}

/// Helper class to build a BNF grammar.
///
/// Equivalent to xgrammar's `GrammarBuilder`. Holds the in-progress
/// rule list and CSR expression store, plus a name→id index.
#[derive(Debug, Default)]
pub struct GrammarBuilder {
    rules: Vec<Rule>,
    expr_data: Vec<i32>,
    expr_indptr: Vec<i32>,
    rule_name_to_id: HashMap<String, i32>,
    /// Cache of the next suffix index to probe for each `name_hint`,
    /// used by [`GrammarBuilder::get_new_rule_name`]. Port of upstream
    /// `next_cnt_per_hint_` (xgrammar commit 96ae88b). Without this the
    /// per-call probe restarts from `_1` every time, making repeated
    /// calls with the same hint O(N) each (O(N^2) overall at high tool
    /// counts). The cache is purely an optimization: every candidate is
    /// still validated against `rule_name_to_id`, so the chosen name is
    /// identical to the un-cached probe. `RefCell` keeps the public
    /// `&self` signature stable.
    next_cnt_per_hint: RefCell<HashMap<String, i32>>,
}

impl GrammarBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /* ---- crate-internal field access for the `rules` submodule ---- */

    pub(super) fn rules_vec(&self) -> &[Rule] {
        &self.rules
    }
    pub(super) fn rules_vec_mut(&mut self) -> &mut Vec<Rule> {
        &mut self.rules
    }
    pub(super) fn rule_name_to_id(&self) -> &HashMap<String, i32> {
        &self.rule_name_to_id
    }
    pub(super) fn rule_name_to_id_mut(&mut self) -> &mut HashMap<String, i32> {
        &mut self.rule_name_to_id
    }
    pub(super) fn next_cnt_per_hint(&self) -> &RefCell<HashMap<String, i32>> {
        &self.next_cnt_per_hint
    }

    /// Finish building and return the [`GrammarData`], setting the root
    /// rule to the rule with `root_rule_name`.
    ///
    /// Returns `Err` if the named rule is not present.
    pub fn get(self, root_rule_name: &str) -> Result<GrammarData, BuilderError> {
        let root_rule_id = self.get_rule_id(root_rule_name);
        if root_rule_id == -1 {
            return Err(BuilderError::RootRuleNotFound(root_rule_name.to_string()));
        }
        self.get_by_id(root_rule_id)
    }

    /// Finish building and return the [`GrammarData`], setting the root
    /// rule to `root_rule_id`.
    ///
    /// Returns `Err` if `root_rule_id` is out of bounds.
    pub fn get_by_id(self, root_rule_id: i32) -> Result<GrammarData, BuilderError> {
        if root_rule_id < 0 || root_rule_id >= self.rules.len() as i32 {
            return Err(BuilderError::RootRuleOutOfBounds(root_rule_id));
        }
        Ok(GrammarData::from_parts(
            self.rules,
            self.expr_data,
            self.expr_indptr,
            root_rule_id,
        ))
    }

    /* ****************** GrammarExpr handling ****************** */

    /// Append a raw `(type, data[])` expression to the CSR store and
    /// return its expr id.
    pub fn add_grammar_expr(&mut self, kind: GrammarExprType, data: &[i32]) -> i32 {
        self.expr_indptr.push(self.expr_data.len() as i32);
        self.expr_data.push(kind as i32);
        self.expr_data.push(data.len() as i32);
        self.expr_data.extend_from_slice(data);
        self.expr_indptr.len() as i32 - 1
    }

    /// Add a `ByteString` expr from raw bytes (each `0..=255`).
    pub fn add_byte_string_bytes(&mut self, bytes: &[i32]) -> i32 {
        self.add_grammar_expr(GrammarExprType::ByteString, bytes)
    }

    /// Add a `ByteString` expr from a Rust string (UTF-8 bytes).
    pub fn add_byte_string(&mut self, s: &str) -> i32 {
        let bytes: Vec<i32> = s.bytes().map(|b| b as i32).collect();
        self.add_grammar_expr(GrammarExprType::ByteString, &bytes)
    }

    /// Build the CSR payload of a character class:
    /// `[is_negative, lower0, upper0, lower1, upper1, …]`.
    fn char_class_data(elements: &[CharacterClassElement], is_negative: bool) -> Vec<i32> {
        let mut data = Vec::with_capacity(1 + elements.len() * 2);
        data.push(is_negative as i32);
        for r in elements {
            data.push(r.lower);
            data.push(r.upper);
        }
        data
    }

    /// Add a `CharacterClass` expr, e.g. `[a-z]` / `[^a-z]`.
    pub fn add_character_class(
        &mut self,
        elements: &[CharacterClassElement],
        is_negative: bool,
    ) -> i32 {
        let data = Self::char_class_data(elements, is_negative);
        self.add_grammar_expr(GrammarExprType::CharacterClass, &data)
    }

    /// Add a `CharacterClassStar` expr, e.g. `[a-z]*`.
    pub fn add_character_class_star(
        &mut self,
        elements: &[CharacterClassElement],
        is_negative: bool,
    ) -> i32 {
        let data = Self::char_class_data(elements, is_negative);
        self.add_grammar_expr(GrammarExprType::CharacterClassStar, &data)
    }

    /// Add an `EmptyStr` expr (the empty string `""`).
    pub fn add_empty_str(&mut self) -> i32 {
        self.add_grammar_expr(GrammarExprType::EmptyStr, &[])
    }

    /// Add a `RuleRef` expr pointing at `rule_id`.
    pub fn add_rule_ref(&mut self, rule_id: i32) -> i32 {
        self.add_grammar_expr(GrammarExprType::RuleRef, &[rule_id])
    }

    /// Add a `Sequence` expr (concatenation of sub-expr ids).
    pub fn add_sequence(&mut self, elements: &[i32]) -> i32 {
        self.add_grammar_expr(GrammarExprType::Sequence, elements)
    }

    /// Add a `Choices` expr (alternation of sub-expr ids).
    pub fn add_choices(&mut self, choices: &[i32]) -> i32 {
        self.add_grammar_expr(GrammarExprType::Choices, choices)
    }

    /// Add a `Repeat` expr `[rule_id, min, max]`.
    pub fn add_repeat(&mut self, ref_rule_id: i32, min_repeat: i32, max_repeat: i32) -> i32 {
        self.add_grammar_expr(
            GrammarExprType::Repeat,
            &[ref_rule_id, min_repeat, max_repeat],
        )
    }

    /// Add a `TagDispatch` expr. Tag strings and stop/exclude string
    /// lists are interned as `ByteString`/`Choices` sub-exprs first,
    /// matching the CSR layout decoded by [`GrammarData::tag_dispatch`].
    pub fn add_tag_dispatch(&mut self, spec: &TagDispatchSpec) -> i32 {
        let mut data: Vec<i32> = Vec::with_capacity(spec.tag_rule_pairs.len() * 2 + 4);
        for (tag, rule_id) in &spec.tag_rule_pairs {
            let tag_expr = self.add_byte_string(tag);
            data.push(tag_expr);
            data.push(*rule_id);
        }
        data.push(spec.stop_eos as i32);
        let stop_ids: Vec<i32> = spec
            .stop_str
            .iter()
            .map(|s| self.add_byte_string(s))
            .collect();
        data.push(self.add_choices(&stop_ids));
        data.push(spec.loop_after_dispatch as i32);
        let excl_ids: Vec<i32> = spec
            .excluded_str
            .iter()
            .map(|s| self.add_byte_string(s))
            .collect();
        data.push(self.add_choices(&excl_ids));
        self.add_grammar_expr(GrammarExprType::TagDispatch, &data)
    }

    /// Number of expressions stored so far.
    pub fn num_grammar_exprs(&self) -> i32 {
        self.expr_indptr.len() as i32
    }

    /// Borrowed view of expr `expr_id`. Panics if out of bounds — the
    /// builder is internal and ids it returns are always valid.
    pub fn get_grammar_expr(&self, expr_id: i32) -> GrammarExpr<'_> {
        let start = self.expr_indptr[expr_id as usize] as usize;
        let kind = GrammarExprType::from_i32(self.expr_data[start])
            .expect("corrupt builder state: unknown GrammarExprType tag");
        let data_len = self.expr_data[start + 1] as usize;
        let data = &self.expr_data[start + 2..start + 2 + data_len];
        GrammarExpr { kind, data }
    }
}
