// SPDX-License-Identifier: AGPL-3.0-only

//! How the radix walk may match a NON-block-aligned tail: the single reader
//! of `AVAROK_PREFIX_SUBBLOCK`, and the spark-runtime switch of the
//! `prefix_cow_tail` row.
//!
//! # Why sharing ships OFF (issue #1193, diagnosed in #692)
//!
//! Both sub-block arms in [`super::inner::RadixTreeInner::walk`] end a match
//! *inside* a KV block and hand that block's physical index to the requester,
//! which pushes it onto its own `block_table`. `PagedKvCache::inc_ref` is a
//! bare counter bump, so the block the requester received is also its own
//! writable tail:
//!
//! * **partial-suffix arm** — the donor is a sequence that published its
//!   partially-filled tail block at END OF PREFILL and is still decoding into
//!   it. Donor and requester then write different K/V into the same physical
//!   slots from `matched_tokens` onward.
//! * **child-key arm** — the donor block is full and committed, and other
//!   sequences map it read-only as prompt K/V; the requester's tail writes
//!   land in offsets `[remainder, block_size)` of that committed block.
//!
//! Two writers in byte-identical lockstep (identical prompts, temperature 0)
//! write the same bytes, so the tear is benign — which is why serial decode
//! and every single-stream test looked clean. Any desync between the writers
//! makes the writes differ and every reader then attends over a torn block;
//! measured 2026-08-21 on qwen3.8-27B + DFlash2, C=4 went 0/4 with three
//! sequences emitting identical wrong text.
//!
//! # Cost of OFF, and the copy-on-write tail that wins it back
//!
//! Up to `block_size - 1` tokens of prefill recompute per hit, and
//! non-block-aligned exact-repeat prompts no longer reach the Marconi
//! exact-leaf shortcut (`matched < total`). The copy-on-write tail
//! ([`SubBlockMode::Cow`]: row `prefix_cow_tail`, or
//! `AVAROK_PREFIX_SUBBLOCK=cow`) wins those tokens back soundly: the walk
//! (`inner/cow_walk.rs`) finds the same sub-block hits but returns the source
//! block outside `matched_blocks`, and the prefill
//! (`spark-model .../prefill_b/prefix_cow.rs`) allocates a private block and
//! copies every attention layer's K and V into it on the compute stream before
//! the block table is used. Only the CoW-aware lookup
//! (`PrefixCache::lookup_cow`) returns such a match; plain `lookup` stays
//! block-aligned for every other caller. Measured counterpoint for the old
//! lever (`docs/ROBUSTNESS.md`, #936 arm ladder): with the lever armed to `0`
//! the agentic shard's output was byte-identical to baseline, 0 of 251
//! samples moved.
//!
//! # What is deliberately NOT changed
//!
//! The INSERT side still publishes the partial tail block into
//! `RadixNode::partial_suffix` and still holds the cache's KV ref on it, so
//! `AVAROK_PREFIX_SUBBLOCK=1` restores the pre-#1193 behaviour completely
//! rather than half of it. With CoW on, that published block is put back to
//! work as a copy source.
//!
//! # Why the sharing lever stays
//!
//! It is one of the #936 arm-ladder levers, so the unsound path must remain
//! reachable for A/B against the copy-on-write implementation.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// How `walk` treats a prompt tail that ends inside a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SubBlockMode {
    /// Block-aligned matches only (the shipped default).
    Off,
    /// The pre-#1193 aliasing arms (`AVAROK_PREFIX_SUBBLOCK=1`; unsound).
    Share,
    /// Copy-on-write tails through `lookup_cow`.
    Cow,
}

/// What `AVAROK_PREFIX_SUBBLOCK` said, parsed once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnvSetting {
    Unset,
    Off,
    Share,
    Cow,
    Refused,
}

/// The `prefix_cow_tail` row, published once at startup by `port_apply`
/// (`spark_model::layers::ops::port_rows::prefix_cow_tail::row()`). Default
/// off.
static COW_ROW: AtomicBool = AtomicBool::new(false);

/// Publish the resolved `prefix_cow_tail` row. Read once per tree, at
/// construction; call it before the model (and its prefix cache) is built.
pub fn set_cow_tail_enabled(on: bool) {
    COW_ROW.store(on, Ordering::Relaxed);
}

/// The mode a new tree takes: the env lever when set, else the row.
///
/// The env read is cached: `walk` runs on every prefix lookup, and nothing
/// mutates the environment after start. The row is read per call (once per
/// tree).
pub(super) fn resolved_mode() -> SubBlockMode {
    static ENV: OnceLock<EnvSetting> = OnceLock::new();
    let env = *ENV.get_or_init(|| parse(std::env::var("AVAROK_PREFIX_SUBBLOCK").ok().as_deref()));
    mode_for(env, COW_ROW.load(Ordering::Relaxed))
}

/// The lever wins over the row; an unset lever leaves the choice to the row.
/// A refused value stays OFF whatever the row says, as it always has.
fn mode_for(env: EnvSetting, cow_row: bool) -> SubBlockMode {
    match env {
        EnvSetting::Unset if cow_row => SubBlockMode::Cow,
        EnvSetting::Unset | EnvSetting::Off | EnvSetting::Refused => SubBlockMode::Off,
        EnvSetting::Share => SubBlockMode::Share,
        EnvSetting::Cow => SubBlockMode::Cow,
    }
}

/// The lever's parse, split out so the contract is testable without touching
/// process-global state.
///
/// Unset is the shipped default and is OFF. An unrecognised value is NOT
/// silently coerced: it is reported and refused, because the only values that
/// can turn an unsound path on must be unambiguous — `AVAROK_PREFIX_SUBBLOCK=
/// true` silently reading as ON under the pre-#1193 `!= "0"` rule is exactly
/// the shape of accident this refuses.
fn parse(raw: Option<&str>) -> EnvSetting {
    match raw {
        None => EnvSetting::Unset,
        Some("1") => EnvSetting::Share,
        Some("0") => EnvSetting::Off,
        Some("cow") => EnvSetting::Cow,
        Some(other) => {
            tracing::error!(
                "AVAROK_PREFIX_SUBBLOCK={other:?} is not a recognised value (expected \"0\", \
                 \"1\" or \"cow\"); refusing it and leaving sub-block tail matching OFF (see \
                 issue #1193)"
            );
            EnvSetting::Refused
        }
    }
}

/// Whether the lever alone arms the legacy sharing arms (the pre-CoW
/// contract, kept for its tests).
#[cfg(test)]
fn enabled_from(raw: Option<&str>) -> bool {
    mode_for(parse(raw), false) == SubBlockMode::Share
}

#[cfg(test)]
mod tests {
    use super::{EnvSetting, SubBlockMode, enabled_from, mode_for, parse};

    /// The shipped default is the whole point of #1193: an operator who sets
    /// nothing must not get the aliasing path.
    #[test]
    fn unset_is_off_and_only_an_explicit_one_turns_it_on() {
        assert!(!enabled_from(None), "unset must be OFF");
        assert!(enabled_from(Some("1")), "an explicit 1 is the only ON");
        assert!(!enabled_from(Some("0")));
    }

    /// Under the pre-#1193 rule (`!= "0"`) every one of these read as ON. They
    /// must now all be OFF — a typo must never arm an unsound path.
    #[test]
    fn an_unrecognised_value_is_refused_rather_than_coerced_on() {
        for raw in ["true", "yes", "on", "", "1 ", "01", "ON", "COW", "cow "] {
            assert!(
                !enabled_from(Some(raw)),
                "{raw:?} must not arm sub-block tail matching"
            );
            assert_eq!(parse(Some(raw)), EnvSetting::Refused, "{raw:?}");
            assert_eq!(
                mode_for(parse(Some(raw)), true),
                SubBlockMode::Off,
                "{raw:?}: a refused lever stays OFF even with the CoW row on"
            );
        }
    }

    /// With the row off, an unset lever and `0` parse exactly as before.
    #[test]
    fn the_row_off_parse_is_unchanged() {
        assert_eq!(mode_for(parse(None), false), SubBlockMode::Off);
        assert_eq!(mode_for(parse(Some("0")), false), SubBlockMode::Off);
        assert_eq!(mode_for(parse(Some("1")), false), SubBlockMode::Share);
    }

    /// `cow` is the one new value, and the row turns CoW on only when the
    /// lever is unset: an explicit `0` or `1` is an A/B instrument and wins.
    #[test]
    fn cow_comes_from_the_lever_or_from_the_row_when_the_lever_is_unset() {
        assert_eq!(mode_for(parse(Some("cow")), false), SubBlockMode::Cow);
        assert_eq!(mode_for(parse(Some("cow")), true), SubBlockMode::Cow);
        assert_eq!(mode_for(parse(None), true), SubBlockMode::Cow);
        assert_eq!(mode_for(parse(Some("0")), true), SubBlockMode::Off);
        assert_eq!(mode_for(parse(Some("1")), true), SubBlockMode::Share);
        assert!(
            !enabled_from(Some("cow")),
            "cow never arms the sharing arms"
        );
    }
}
