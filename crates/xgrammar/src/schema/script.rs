// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNF script builder — port of `class EBNFScriptCreator` from
// `cpp/ebnf_script_creator.h`.
//
// Accumulates `(rule_name, rule_body)` pairs in insertion order and
// renders them as an EBNF script. Rule names are de-duplicated by
// appending a numeric suffix.

use crate::support::escape::print_str_as_escaped;
use std::collections::HashSet;

/// Maximum numeric suffix tried when de-duplicating a rule name —
/// matches the C++ `NAME_SUFFIX_MAXIMUM`.
const NAME_SUFFIX_MAXIMUM: i32 = 10_000;

/// Builds an EBNF grammar script rule-by-rule.
#[derive(Debug, Default)]
pub struct EbnfScriptCreator {
    rules: Vec<(String, String)>,
    rule_names: HashSet<String>,
}

impl EbnfScriptCreator {
    /// Fresh, empty script builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a unique rule name based on `hint`, returning the name
    /// actually assigned. Port of `AllocateRuleName`.
    pub fn allocate_rule_name(&mut self, hint: &str) -> String {
        if !self.rule_names.contains(hint) {
            self.rule_names.insert(hint.to_string());
            return hint.to_string();
        }
        for i in 0..NAME_SUFFIX_MAXIMUM {
            let candidate = format!("{hint}_{i}");
            if !self.rule_names.contains(&candidate) {
                self.rule_names.insert(candidate.clone());
                return candidate;
            }
        }
        // Practically unreachable — fall back to a counter-derived
        // name so we never panic (C++ aborts here).
        let fallback = format!("{hint}_{}", self.rule_names.len());
        self.rule_names.insert(fallback.clone());
        fallback
    }

    /// Add a rule under a name allocated via [`Self::allocate_rule_name`].
    /// Returns the rule name. Port of `AddRuleWithAllocatedName`.
    pub fn add_rule_with_allocated_name(&mut self, name: &str, body: &str) -> String {
        debug_assert!(
            self.rule_names.contains(name),
            "rule name {name} was not allocated"
        );
        self.rules.push((name.to_string(), body.to_string()));
        name.to_string()
    }

    /// Allocate a name from `hint` and add the rule in one step.
    /// Returns the actual rule name. Port of `AddRule`.
    pub fn add_rule(&mut self, hint: &str, body: &str) -> String {
        let name = self.allocate_rule_name(hint);
        self.add_rule_with_allocated_name(&name, body)
    }

    /// Render the full EBNF script — one `name ::= body` line per
    /// rule. Port of `GetScript`.
    pub fn get_script(&self) -> String {
        let mut script = String::new();
        for (name, body) in &self.rules {
            script.push_str(name);
            script.push_str(" ::= ");
            script.push_str(body);
            script.push('\n');
        }
        script
    }

    /// Concatenate `items` with single-space separators, wrapped in
    /// parentheses. Port of static `Concat`.
    pub fn concat(items: &[String]) -> String {
        let mut s = String::from("(");
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(item);
        }
        s.push(')');
        s
    }

    /// Join `items` with ` | `, wrapped in parentheses. Port of
    /// static `Or`.
    pub fn or(items: &[String]) -> String {
        let mut s = String::from("(");
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                s.push_str(" | ");
            }
            s.push_str(item);
        }
        s.push(')');
        s
    }

    /// Escape and double-quote `str`. Port of static `Str`.
    pub fn str_lit(s: &str) -> String {
        format!("\"{}\"", print_str_as_escaped(s.as_bytes()))
    }

    /// Apply an EBNF repetition suffix to `item`. Port of static
    /// `Repeat`. `max == -1` means unbounded.
    pub fn repeat(item: &str, min: i32, max: i32) -> String {
        if min == 0 && max == 1 {
            return format!("{item}?");
        }
        if min == 0 && max == -1 {
            return format!("{item}*");
        }
        if min == 1 && max == -1 {
            return format!("{item}+");
        }
        if min == 0 && max == 0 {
            return String::new();
        }
        if min == max {
            return format!("{item}{{{min}}}");
        }
        if max == -1 {
            return format!("{item}{{{min},}}");
        }
        format!("{item}{{{min},{max}}}")
    }
}
