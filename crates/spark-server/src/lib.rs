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

// The three pure modules added in PR 4 (OpenAI compat remaining items) are
// public here so `cargo test -p spark-server --lib` can exercise their
// unit tests without needing to build the full binary.
#[path = "auth.rs"]
pub mod auth;
#[path = "rate_limiter.rs"]
pub mod rate_limiter;
#[path = "refusal.rs"]
pub mod refusal;
