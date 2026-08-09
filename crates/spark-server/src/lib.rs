// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! Atlas Spark — shared modules for integration tests.

pub mod tokenizer;

// Reasoning-parser format registry (DeepSeek-R1 `<think>`/`</think>` contract
// used by DS4F opt-in reasoning mode). Exposed here so the tokenizer unit tests
// that assert `[reasoning]` registration compile under `--lib`, mirroring the
// binary crate root. Its only crate dependency is `tokenizer`, already public above.
pub mod reasoning_parser;

// The three pure modules added in PR 4 (OpenAI compat remaining items) are
// public here so `cargo test -p spark-server --lib` can exercise their
// unit tests without needing to build the full binary.
#[path = "auth.rs"]
pub mod auth;
#[path = "rate_limiter.rs"]
pub mod rate_limiter;
#[path = "refusal.rs"]
pub mod refusal;
