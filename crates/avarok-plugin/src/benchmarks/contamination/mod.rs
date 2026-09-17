// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-contamination detector: concurrent requests must not change each
//! other's output. See `score.rs` for the four-leg design and why it is four,
//! `prompts.rs` for how the probe corpus makes leakage detectable, and
//! `driver.rs` for the state machine that runs the legs.

pub mod driver;
pub mod prompts;
pub mod report;
pub mod score;
pub mod transcript;

pub use driver::{DESCRIPTOR, METADATA};
