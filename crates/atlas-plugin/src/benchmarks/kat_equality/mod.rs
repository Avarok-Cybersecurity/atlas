// SPDX-License-Identifier: AGPL-3.0-only

//! KAT equality: the same sample, in different request orders, byte-for-byte.
//!
//! ## What it asserts
//!
//! A known-answer test asserts that an input produces an output. That is only
//! a claim about the input if the output does not also depend on what ran
//! BEFORE it. This gate issues one sample set against ONE server in two or
//! more orders and requires every `sample_id`'s reply to be byte-identical.
//!
//! ## Why the existing gates cannot do this
//!
//! `ssm-state-poisoning-gate` is the closest, and it cannot:
//!
//!   - it records `RoundVerdict::Jittered` — a reply that came back reworded —
//!     as a PASS **by design**, because for its purposes alternating restore
//!     anchors are a healthy engine property. Here that IS the finding;
//!   - it replays one fixed 4-turn script varying only WARMTH, never ORDER,
//!     across ~48 generations. The effect being hunted moved 12 samples in
//!     995. Forty-eight generations of one script has no power to see it.
//!
//! Extending it would mean inverting its jitter rule, which would break what
//! it does measure. Two gates, two invariants.
//!
//! ## The measured prior
//!
//! At one commit, one model, temperature 0, the golden 995-sample draw scored
//! whole vs its own four shards and disagreed on **12** samples;
//! `ATLAS_NO_TAIL_SPLIT=1` took it to 4 and `ATLAS_MARCONI_MIN_TOKENS=1e8` to
//! 2. Those are the numbers this gate has to be able to see, and they are why
//! the sample count matters: a handful of prompts would not have caught it.

pub mod compare;
pub mod driver;
pub mod report;

pub use compare::{
    Observation, OrderRun, SampleVerdict, Score, permutation, score, verdict, verdict_for,
};

pub use driver::{DESCRIPTOR, METADATA};

#[cfg(test)]
#[path = "compare_tests.rs"]
mod compare_tests;

#[cfg(test)]
#[path = "driver_tests.rs"]
mod driver_tests;
