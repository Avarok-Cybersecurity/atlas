// SPDX-License-Identifier: AGPL-3.0-only

//! What a number or an enum looks like once it is on the screen.
//!
//! Two things live here because both were, before this file, decided
//! independently at each call site — and both were visibly disagreeing with
//! themselves in one UI.
//!
//! * **Byte counts.** The Library card said a checkpoint was `18.6 GB` while
//!   the download progress line for the same file said `20.0 GB`, because one
//!   divided by 1024³ and the other by 10⁹. Whichever is "right" in the
//!   abstract, a user watching a download finish and become a Library entry
//!   sees the size change for no reason. [`bytes`] is now the only place that
//!   decides.
//! * **Scheduler enums.** `{:?}` is a debugging tool that reached the screen:
//!   the Stats tab and `/status` both printed `MTP gate Mtp`. Debug output is
//!   not a rendering — it is the type's field names, it changes when someone
//!   renames a variant, and it says nothing to a reader who does not have the
//!   enum open. [`mtp_mode_label`] says what the state MEANS.

use crate::scheduler::snapshot::MtpModeSnap;

/// One binary GiB. The whole UI's divisor, and the same one `nvidia-smi`,
/// `free` and the HF cache report in — an Atlas number a user cross-checks
/// against those must not differ by 7%.
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: u64 = 1024 * 1024;

/// A byte count for a human: `18.6 GB` at or above a gibibyte, `812 MB` below.
///
/// ★ **One function, because three of them disagreed.** The label stays `GB`
/// rather than `GiB` — every other tool on the box does the same, and matching
/// them is worth more here than matching SI, since the value a user compares
/// this against is `nvidia-smi`'s, not a disk vendor's. Sub-gibibyte truncates
/// rather than rounds, so a value one byte short of a gibibyte never reads
/// `1024 MB` next to a `1.0 GB` that is larger than it.
pub fn bytes(n: u64) -> String {
    let g = n as f64 / GIB;
    if g >= 1.0 {
        format!("{g:.1} GB")
    } else {
        format!("{} MB", n / MIB)
    }
}

/// What the scheduler's MTP gate is doing, in words.
///
/// Deliberately not `{:?}`: the variant names are an implementation detail
/// (`Mtp` tells a reader nothing), and a rename would silently change what the
/// dashboard says.
pub fn mtp_mode_label(mode: MtpModeSnap) -> &'static str {
    match mode {
        MtpModeSnap::Mtp => "speculative",
        MtpModeSnap::Serial => "serial",
        MtpModeSnap::Probing => "probing",
        MtpModeSnap::Off => "off",
    }
}

#[cfg(test)]
#[path = "format_tests.rs"]
mod tests;
