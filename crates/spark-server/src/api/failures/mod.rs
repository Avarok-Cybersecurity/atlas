// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-turn failure-recovery sub-systems extracted from `api.rs`.
//! Each sibling file groups one cluster of `"F<n>"` tagged helpers used by
//! `chat_completions` to detect and mitigate stalled / looping agents.

mod circuit;
mod circuit_f60;
#[cfg(test)]
mod circuit_tests;
mod classification;
mod duplicate;
mod duplicate_helpers;
#[cfg(test)]
mod duplicate_tests;
mod stall;

pub(super) use circuit::*;
pub(super) use circuit_f60::*;
pub(super) use classification::*;
pub(super) use duplicate::*;
pub(super) use duplicate_helpers::*;
pub(super) use stall::*;
