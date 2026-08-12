// SPDX-License-Identifier: AGPL-3.0-only

//! Re-export shim: the transcript type moved to `benchmarks::transcript` so
//! every driver that reduces a reply to its comparable part shares ONE
//! definition (the SSM state poisoning gate compares transcripts too, and a
//! second copy of this type would be a second, diverging equality contract).
//! All contamination-internal imports keep working through this module.

pub use crate::benchmarks::transcript::{RequestOutcome, Transcript};
