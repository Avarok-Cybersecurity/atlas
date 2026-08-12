// SPDX-License-Identifier: AGPL-3.0-only

//! SSM state poisoning detector: replay a fixed conversation script and
//! require byte-identical transcripts. See `driver.rs` for the invariant and
//! the incident that motivated the gate, `probe.rs` for the pinned script,
//! `compare.rs` for the per-round comparison, and `score.rs` for the
//! zero-tolerance decision rule.

pub mod compare;
pub mod driver;
pub mod probe;
pub mod report;
pub mod score;

pub use driver::{DESCRIPTOR, METADATA};
