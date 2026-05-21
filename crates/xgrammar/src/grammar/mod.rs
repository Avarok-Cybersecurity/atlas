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

pub mod data;
pub mod expr;

pub use data::{GrammarData, Rule, TagDispatch};
pub use expr::{GrammarExpr, GrammarExprType};
