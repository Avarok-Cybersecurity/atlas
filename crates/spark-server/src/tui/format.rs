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
//! * **Byte RATES.** Same disagreement, one field to the right and left
//!   standing when the sizes were unified. [`rate`] settles it; the argument
//!   is on the function.
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
const KIB: f64 = 1024.0;

/// A byte count for a human: `18.6 GB` at or above a gibibyte, `812 MB` below.
///
/// ★ **One function, because three of them disagreed.** Sub-gibibyte truncates
/// rather than rounds, so a value one byte short of a gibibyte never reads
/// `1024 MB` next to a `1.0 GB` that is larger than it.
///
/// ## `GB` for 1024³ — SETTLED, do not re-litigate
///
/// The label is wrong by IEC 80000-13, which reserves `GB` for 10⁹ and calls
/// this a `GiB`. It stays anyway, and the reason is that a unit label is not a
/// standards citation — it is what lets a reader match this number to another
/// number they are already looking at. Every source an Atlas operator
/// cross-checks against is 1024-based and labels it `GB`: `nvidia-smi`,
/// `free -g`, `df -h`, `htop`, and the HF cache's own reporting. Switching to
/// `GiB` would make Atlas the only correct thing on a screen full of `GB`, and
/// the reader's first conclusion would be that the two disagree by 7%.
///
/// Nothing in the tree divides by 10⁹ any more, so there is no second reading
/// to confuse it with: `data/metrics_poll` (GPU + host memory), the download
/// row, the Library card, the load-rate `GB/s` on Main and [`rate`] below all
/// use 1024. The cost of the wrong label is a pedantic reader; the cost of the
/// right one is a reader who thinks a number is wrong. This picks the first,
/// once, here.
pub fn bytes(n: u64) -> String {
    let g = n as f64 / GIB;
    if g >= 1.0 {
        format!("{g:.1} GB")
    } else {
        format!("{} MB", n / MIB)
    }
}

/// A byte-per-second rate for a human: `92 MB/s`, `3.0 MB/s`, `2.0 KB/s`,
/// `0 B/s`.
///
/// ★ **The second half of the same unification, and the base is the whole
/// point.** [`bytes`] settled sizes and left rates alone: the download row
/// divided by 10⁶ and the Stats tile by 1024, so one download line read
/// `3.7 GB / 32.5 GB  ·  96 MB/s` with the two figures on 7% different scales.
/// Anyone who divided one by the other to get a time remaining — which is the
/// only reason both numbers are on that line — got an answer 7% wrong, and
/// nothing on screen said why.
///
/// **Binary, 1024-based**, for three reasons in ascending order of weight:
/// `iftop`, `nload` and `bmon` all scale their byte counters by 1024 too, so
/// the external cross-check argument does not favour decimal; the Stats tile
/// was already binary, so binary is the majority of the sites being merged;
/// and decisively, the sizes it sits beside are binary, and a rate that does
/// not divide into the size printed next to it is the defect being fixed.
///
/// **The unit is spelled out**, which the Stats tile did not do — it rendered
/// `↓2K/s ↑3.0M/s`, naming a magnitude and no unit at all. `M/s` of what, on a
/// tile whose other two figures are request counts? The terseness bought a
/// breakpoint a few columns lower on a tile that is already truncating at 100
/// columns, and cost the one word that says these are bytes.
///
/// One decimal below ten, none above: enough to see a rate move without a
/// digit that flickers every frame on a figure that jitters by percent.
pub fn rate(bytes_per_sec: f64) -> String {
    let (n, unit) = match bytes_per_sec {
        b if b >= GIB => (b / GIB, "GB"),
        b if b >= MIB as f64 => (b / MIB as f64, "MB"),
        b if b >= KIB => (b / KIB, "KB"),
        b => (b.max(0.0), "B"),
    };
    if n < 10.0 && unit != "B" {
        format!("{n:.1} {unit}/s")
    } else {
        format!("{n:.0} {unit}/s")
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
