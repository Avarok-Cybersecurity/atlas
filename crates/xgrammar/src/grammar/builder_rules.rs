// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarBuilder — rule-handling methods and the builder error type.
// Split out of `builder.rs` to keep each file under the 250-line cap.
// Port of the rule-handling half of `GrammarBuilder`
// (xgrammar `cpp/grammar_builder.h`).

use super::GrammarBuilder;
use crate::grammar::data::Rule;

impl GrammarBuilder {
    /// Add a fully-formed rule and return its id.
    ///
    /// Returns `Err` if a rule of the same name already exists.
    pub fn add_rule(&mut self, rule: Rule) -> Result<i32, BuilderError> {
        if self.rule_name_to_id().contains_key(&rule.name) {
            return Err(BuilderError::DuplicateRule(rule.name.clone()));
        }
        let id = self.rules_vec().len() as i32;
        let name = rule.name.clone();
        self.rules_vec_mut().push(rule);
        self.rule_name_to_id_mut().insert(name, id);
        Ok(id)
    }

    /// Add a rule with the given `name` and `body_expr_id`.
    pub fn add_rule_named(
        &mut self,
        name: impl Into<String>,
        body_expr_id: i32,
    ) -> Result<i32, BuilderError> {
        self.add_rule(Rule::new(name, body_expr_id))
    }

    /// Add a rule with a body, deriving a unique name from `name_hint`.
    pub fn add_rule_with_hint(
        &mut self,
        name_hint: &str,
        body_expr_id: i32,
    ) -> Result<i32, BuilderError> {
        let name = self.get_new_rule_name(name_hint);
        self.add_rule(Rule::new(name, body_expr_id))
    }

    /// Number of rules.
    pub fn num_rules(&self) -> i32 {
        self.rules_vec().len() as i32
    }

    /// The rule with the given id.
    pub fn get_rule(&self, rule_id: i32) -> &Rule {
        &self.rules_vec()[rule_id as usize]
    }

    /// Add a body-less rule (`body_expr_id = -1`). The body must be set
    /// later with [`GrammarBuilder::update_rule_body`]. Useful when the
    /// rule id is needed to build the body (recursive rules).
    pub fn add_empty_rule(&mut self, name: impl Into<String>) -> Result<i32, BuilderError> {
        self.add_rule(Rule::new(name, -1))
    }

    /// Add a body-less rule with a unique name derived from `name_hint`.
    pub fn add_empty_rule_with_hint(&mut self, name_hint: &str) -> Result<i32, BuilderError> {
        let name = self.get_new_rule_name(name_hint);
        self.add_rule(Rule::new(name, -1))
    }

    /// Set the body of rule `rule_id`.
    ///
    /// Returns `Err` if `rule_id` is out of range.
    pub fn update_rule_body(
        &mut self,
        rule_id: i32,
        body_expr_id: i32,
    ) -> Result<(), BuilderError> {
        if rule_id < 0 || rule_id >= self.rules_vec().len() as i32 {
            return Err(BuilderError::RuleIdOutOfRange(rule_id));
        }
        self.rules_vec_mut()[rule_id as usize].body_expr_id = body_expr_id;
        Ok(())
    }

    /// Set the body of the rule named `rule_name`.
    pub fn update_rule_body_named(
        &mut self,
        rule_name: &str,
        body_expr_id: i32,
    ) -> Result<(), BuilderError> {
        let rule_id = self.get_rule_id(rule_name);
        if rule_id == -1 {
            return Err(BuilderError::RuleNameNotFound(rule_name.to_string()));
        }
        self.update_rule_body(rule_id, body_expr_id)
    }

    /// Set the lookahead-assertion expr id of rule `rule_id`. An id of
    /// `-1` means no assertion.
    pub fn update_lookahead_assertion(
        &mut self,
        rule_id: i32,
        lookahead_assertion_id: i32,
    ) -> Result<(), BuilderError> {
        if rule_id < 0 || rule_id >= self.rules_vec().len() as i32 {
            return Err(BuilderError::RuleIdOutOfRange(rule_id));
        }
        self.rules_vec_mut()[rule_id as usize].lookahead_assertion_id = lookahead_assertion_id;
        Ok(())
    }

    /// Set the lookahead-assertion expr id of the rule named `rule_name`.
    pub fn update_lookahead_assertion_named(
        &mut self,
        rule_name: &str,
        lookahead_assertion_id: i32,
    ) -> Result<(), BuilderError> {
        let rule_id = self.get_rule_id(rule_name);
        if rule_id == -1 {
            return Err(BuilderError::RuleNameNotFound(rule_name.to_string()));
        }
        self.update_lookahead_assertion(rule_id, lookahead_assertion_id)
    }

    /// Mark the lookahead assertion of rule `rule_id` as exact.
    pub fn update_lookahead_exact(
        &mut self,
        rule_id: i32,
        is_exact: bool,
    ) -> Result<(), BuilderError> {
        if rule_id < 0 || rule_id >= self.rules_vec().len() as i32 {
            return Err(BuilderError::RuleIdOutOfRange(rule_id));
        }
        self.rules_vec_mut()[rule_id as usize].is_exact_lookahead = is_exact;
        Ok(())
    }

    /// Find an unused rule name starting from `name_hint`, appending an
    /// integer suffix (`_1`, `_2`, …) on collision.
    ///
    /// The first probed suffix is read from `next_cnt_per_hint` (a cache
    /// of the last suffix reached for this hint) rather than restarting
    /// from `1` every call. This makes repeated calls with the same hint
    /// amortized O(1) instead of O(N) — see the field doc on
    /// [`GrammarBuilder`]. Behavior is unchanged: every candidate is
    /// still verified absent from `rule_name_to_id`, and the cache only
    /// advances (never skips a still-free lower suffix, because suffixes
    /// for a given hint are only ever consumed in increasing order).
    pub fn get_new_rule_name(&self, name_hint: &str) -> String {
        if !self.rule_name_to_id().contains_key(name_hint) {
            return name_hint.to_string();
        }
        let mut cache = self.next_cnt_per_hint().borrow_mut();
        let cnt = cache.entry(name_hint.to_string()).or_insert(1);
        loop {
            let candidate = format!("{name_hint}_{cnt}");
            if !self.rule_name_to_id().contains_key(&candidate) {
                return candidate;
            }
            *cnt += 1;
        }
    }

    /// Rule id of the rule named `name`, or `-1` if absent.
    pub fn get_rule_id(&self, name: &str) -> i32 {
        self.rule_name_to_id().get(name).copied().unwrap_or(-1)
    }
}

/// Errors emitted by [`GrammarBuilder`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BuilderError {
    /// A rule with this name was added twice.
    #[error("rule \"{0}\" is added multiple times")]
    DuplicateRule(String),
    /// The requested root rule name does not exist.
    #[error("the root rule with name \"{0}\" is not found")]
    RootRuleNotFound(String),
    /// The requested root rule id is out of bounds.
    #[error("the root rule id {0} is out of bound")]
    RootRuleOutOfBounds(i32),
    /// A rule id passed to an update method is out of range.
    #[error("rule id {0} is out of range")]
    RuleIdOutOfRange(i32),
    /// A rule name passed to an update method is not found.
    #[error("rule \"{0}\" is not found")]
    RuleNameNotFound(String),
}
