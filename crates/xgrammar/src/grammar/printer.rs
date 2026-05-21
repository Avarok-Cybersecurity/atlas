// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarPrinter — prints a `GrammarData` back to EBNF text.
// Port of `cpp/grammar_printer.{h,cc}`.
//
// Used by tests and debugging to round-trip a grammar: parse -> pass ->
// print. The output format mirrors xgrammar exactly so functor passes
// can be verified against the C++ reference strings.

use std::collections::HashMap;

use super::data::{GrammarData, Rule};
use super::expr::{GrammarExpr, GrammarExprType};
use crate::support::escape::print_str_as_escaped;
use crate::support::{char_to_utf8, print_as_escaped};

/// Prints the BNF AST in xgrammar's standard EBNF format.
pub struct GrammarPrinter<'g> {
    grammar: &'g GrammarData,
}

impl<'g> GrammarPrinter<'g> {
    /// Wrap a grammar for printing.
    pub fn new(grammar: &'g GrammarData) -> Self {
        Self { grammar }
    }

    /// Render the complete grammar to EBNF text — one rule per line.
    pub fn render(&self) -> String {
        let mut result = String::new();
        for i in 0..self.grammar.num_rules() {
            result.push_str(&self.print_rule(self.grammar.rule(i)));
            result.push('\n');
        }
        result
    }

    /// Print one rule: `name ::= body [(=lookahead)]`.
    pub fn print_rule(&self, rule: &Rule) -> String {
        let mut res = format!(
            "{} ::= {}",
            rule.name,
            self.print_expr_id(rule.body_expr_id)
        );
        if rule.lookahead_assertion_id != -1 {
            res.push_str(" (=");
            res.push_str(&self.print_expr_id(rule.lookahead_assertion_id));
            res.push(')');
        }
        res
    }

    /// Print the rule with the given id.
    pub fn print_rule_id(&self, rule_id: i32) -> String {
        self.print_rule(self.grammar.rule(rule_id))
    }

    /// Print the expression with the given id.
    pub fn print_expr_id(&self, expr_id: i32) -> String {
        self.print_expr(&self.grammar.expr(expr_id))
    }

    /// Print a `GrammarExpr`, dispatching on its type.
    pub fn print_expr(&self, e: &GrammarExpr<'_>) -> String {
        match e.kind {
            GrammarExprType::ByteString => Self::print_byte_string(e),
            GrammarExprType::CharacterClass => Self::print_character_class(e),
            GrammarExprType::CharacterClassStar => {
                format!("{}*", Self::print_character_class(e))
            }
            GrammarExprType::EmptyStr => "\"\"".to_string(),
            GrammarExprType::RuleRef => self.print_rule_ref(e),
            GrammarExprType::Sequence => self.print_sequence(e),
            GrammarExprType::Choices => self.print_choices(e),
            GrammarExprType::TagDispatch => self.print_tag_dispatch(e),
            GrammarExprType::Repeat => self.print_repeat(e),
        }
    }

    fn print_byte_string(e: &GrammarExpr<'_>) -> String {
        let bytes: Vec<u8> = e.data.iter().map(|&b| b as u8).collect();
        format!("\"{}\"", print_str_as_escaped(&bytes))
    }

    fn print_character_class(e: &GrammarExpr<'_>) -> String {
        let custom: HashMap<i32, String> = HashMap::from([
            ('-' as i32, "\\-".to_string()),
            (']' as i32, "\\]".to_string()),
        ]);
        let mut result = String::from("[");
        if e[0] != 0 {
            result.push('^');
        }
        let mut i = 1;
        while i < e.len() {
            result.push_str(&print_as_escaped(e[i], &custom));
            if e[i] != e[i + 1] {
                result.push('-');
                result.push_str(&print_as_escaped(e[i + 1], &custom));
            }
            i += 2;
        }
        result.push(']');
        result
    }

    fn print_rule_ref(&self, e: &GrammarExpr<'_>) -> String {
        self.grammar.rule(e[0]).name.clone()
    }

    fn print_sequence(&self, e: &GrammarExpr<'_>) -> String {
        let parts: Vec<String> = e.data.iter().map(|&id| self.print_expr_id(id)).collect();
        format!("({})", parts.join(" "))
    }

    fn print_choices(&self, e: &GrammarExpr<'_>) -> String {
        let parts: Vec<String> = e.data.iter().map(|&id| self.print_expr_id(id)).collect();
        format!("({})", parts.join(" | "))
    }

    fn print_repeat(&self, e: &GrammarExpr<'_>) -> String {
        format!("{}{{{}, {}}}", self.grammar.rule(e[0]).name, e[1], e[2])
    }

    fn print_string(s: &str) -> String {
        format!("\"{}\"", print_str_as_escaped(s.as_bytes()))
    }

    fn print_tag_dispatch(&self, e: &GrammarExpr<'_>) -> String {
        let td = self.grammar.tag_dispatch(self.expr_id_of(e));
        let mut result = String::from("TagDispatch(\n");
        let indent = "  ";
        for (tag, rule_id) in &td.tag_rule_pairs {
            result.push_str(&format!(
                "{indent}({}, {}),\n",
                Self::print_string(tag),
                self.grammar.rule(*rule_id).name
            ));
        }
        result.push_str(&format!("{indent}stop_eos={},\n", td.stop_eos));
        let stop: Vec<String> = td.stop_str.iter().map(|s| Self::print_string(s)).collect();
        result.push_str(&format!("{indent}stop_str=({}),\n", stop.join(", ")));
        result.push_str(&format!(
            "{indent}loop_after_dispatch={},\n",
            td.loop_after_dispatch
        ));
        let excl: Vec<String> = td
            .excluded_str
            .iter()
            .map(|s| Self::print_string(s))
            .collect();
        result.push_str(&format!("{indent}excludes=({})\n)", excl.join(", ")));
        result
    }

    /// `tag_dispatch` decoding needs the expr id; recover it by matching
    /// the data slice's start offset against the grammar's CSR layout.
    fn expr_id_of(&self, e: &GrammarExpr<'_>) -> i32 {
        for id in 0..self.grammar.num_exprs() {
            let cand = self.grammar.expr(id);
            if cand.kind == e.kind && std::ptr::eq(cand.data.as_ptr(), e.data.as_ptr()) {
                return id;
            }
        }
        panic!("printer: tag dispatch expr not found in grammar");
    }
}

/// Convenience: render `grammar` as EBNF text.
pub fn print_grammar(grammar: &GrammarData) -> String {
    GrammarPrinter::new(grammar).render()
}

/// Convert a unicode codepoint to its UTF-8 byte values as `i32`s.
/// Helper shared with `functor` (single-element char-class -> byte string).
pub(crate) fn codepoint_to_bytes(cp: i32) -> Vec<i32> {
    char_to_utf8(cp).into_iter().map(|b| b as i32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::parse_ebnf_default;

    fn roundtrip(ebnf: &str) -> String {
        let g = parse_ebnf_default(ebnf).expect("parse");
        print_grammar(&g)
    }

    #[test]
    fn prints_byte_string_rule() {
        let out = roundtrip("root ::= \"abc\"\n");
        assert!(out.contains("root ::="));
        assert!(out.contains("\"abc\""));
    }

    #[test]
    fn prints_character_class() {
        let out = roundtrip("root ::= [a-z]\n");
        assert!(out.contains("[a-z]"));
    }

    #[test]
    fn prints_character_class_star() {
        let out = roundtrip("root ::= [a-z]*\n");
        assert!(out.contains("[a-z]*"));
    }

    #[test]
    fn prints_negated_class() {
        let out = roundtrip("root ::= [^a-z]\n");
        assert!(out.contains("[^a-z]"));
    }

    #[test]
    fn prints_choices() {
        let out = roundtrip("root ::= \"a\" | \"b\"\n");
        assert!(out.contains(" | "));
    }

    #[test]
    fn prints_empty_string() {
        let out = roundtrip("root ::= \"\"\n");
        assert!(out.contains("\"\""));
    }
}
