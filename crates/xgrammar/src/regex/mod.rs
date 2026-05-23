// SPDX-License-Identifier: AGPL-3.0-only
//
// Regex converter — port of `cpp/regex_converter.{h,cc}`.
//
// Converts a JavaScript-flavoured regex pattern string into an EBNF
// grammar string (the `RegexToEBNF` entry point) and, as a
// convenience, into a parsed `GrammarData` AST by feeding that EBNF
// through the existing `grammar::parse_ebnf` parser.
//
// RELATION TO THE FSM REGEX BUILDER
// ---------------------------------
// `fsm::builder::RegexFsmBuilder` (ported by the W2 FSM agent) takes a
// regex and builds an NFA/FSM directly. This module is the *other*
// path: it lowers a regex to the EBNF grammar surface syntax, which
// the EBNF parser then turns into a `GrammarData` AST. The two share
// no code — they are independent lowerings of the same input, exactly
// as in upstream xgrammar where `regex_converter.cc` and
// `fsm_builder.cc` are separate translation units.
//
// The implementation refers to the regex described in
// <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions>.
//
// FAITHFULNESS / SIMPLIFICATIONS vs C++
// -------------------------------------
//  * C++ aborts the process (`XGRAMMAR_LOG(FATAL)`) on a malformed
//    regex; we return `Err(RegexError)` instead — no panics, no
//    `unsafe`.
//  * C++ emits warnings to stderr for stray `^`/`$` and unknown
//    escapes; we collect them into `ConvertOutput::warnings` rather
//    than performing I/O (keeps the converter pure).
//  * Backreferences, lookahead/lookbehind, named-group *flags*,
//    unicode-property `\p{…}` and word boundaries are unsupported and
//    reported as errors — identical to upstream.

mod converter;
mod cursor;
mod escape_handlers;
mod sub_handlers;

pub use converter::{ConvertOutput, RegexConverter};

use crate::grammar::{GrammarData, parse_ebnf};

/// Error raised while converting a regex to EBNF.
///
/// `position` is 1-based, matching the C++ `RaiseError` message
/// (`current_ - start_ + 1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexError {
    /// 1-based codepoint offset into the regex where the error was
    /// detected.
    pub position: usize,
    /// Human-readable description of the problem.
    pub message: String,
}

impl std::fmt::Display for RegexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Regex parsing error at position {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for RegexError {}

/// Convert a regex string to an EBNF grammar string.
///
/// When `with_rule_name` is `true` the result is wrapped as
/// `root ::= <body>\n`; otherwise just the rule body is returned.
/// Faithful port of C++ `RegexToEBNF`.
pub fn regex_to_ebnf(regex: &str, with_rule_name: bool) -> Result<String, RegexError> {
    let body = RegexConverter::new(regex)?.convert()?.ebnf;
    if with_rule_name {
        Ok(format!("root ::= {body}\n"))
    } else {
        Ok(body)
    }
}

/// Error from [`regex_to_grammar`]: either the regex itself was
/// malformed, or the EBNF it lowered to failed to parse.
#[derive(Debug)]
pub enum RegexToGrammarError {
    /// The regex pattern was malformed.
    Regex(RegexError),
    /// The lowered EBNF failed to parse (should not happen for a
    /// well-formed regex; indicates an internal converter bug).
    Ebnf(crate::grammar::ParseError),
}

impl std::fmt::Display for RegexToGrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegexToGrammarError::Regex(e) => write!(f, "{e}"),
            RegexToGrammarError::Ebnf(e) => write!(f, "lowered EBNF failed to parse: {e:?}"),
        }
    }
}

impl std::error::Error for RegexToGrammarError {}

/// Convert a regex string straight to a parsed [`GrammarData`] AST.
///
/// This is the regex→grammar-AST path: it lowers the regex to EBNF
/// (via [`regex_to_ebnf`]) and parses that with the existing EBNF
/// parser, so the result is a fully-formed grammar usable by the rest
/// of the crate. Upstream xgrammar's `Grammar::FromRegex` does the
/// same composition.
pub fn regex_to_grammar(regex: &str) -> Result<GrammarData, RegexToGrammarError> {
    let ebnf = regex_to_ebnf(regex, true).map_err(RegexToGrammarError::Regex)?;
    parse_ebnf(&ebnf, "root").map_err(RegexToGrammarError::Ebnf)
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_patterns;
