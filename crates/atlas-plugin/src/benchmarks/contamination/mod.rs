// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-contamination detector: concurrent requests must not change each
//! other's output. See `score.rs` for the four-leg design and why it is four.

pub mod score;
pub mod transcript;
