// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar AST — port wave W1 (foundation).
//
// Ported so far:
//   expr.rs  — GrammarExprType, GrammarExpr      (cpp/grammar_impl.h)
//   data.rs  — Rule, TagDispatch, GrammarData    (cpp/grammar_impl.h)
//
// Pending waves (see ../../PORT_PLAN.md):
//   builder  — GrammarBuilder        (cpp/grammar_builder.h)   W2
//   parser   — EBNF parser           (cpp/grammar_parser.cc)   W2
//   functor  — normalization passes  (cpp/grammar_functor.cc)  W3
//   printer  — EBNF printer          (cpp/grammar_printer.cc)  W3

pub mod builder;
pub mod data;
pub mod expr;
pub mod functor;
pub mod lexer;
pub mod parser;
pub mod printer;

pub use builder::GrammarBuilder;
pub use data::{GrammarData, Rule, TagDispatch};
pub use expr::{GrammarExpr, GrammarExprType};
pub use functor::{
    GrammarConcat, GrammarFsmBuilder, GrammarFsmHasher, GrammarNormalizer, GrammarOptimizer,
    GrammarUnion,
};
pub use parser::{ParseError, parse_ebnf, parse_ebnf_default};
pub use printer::{GrammarPrinter, print_grammar};
