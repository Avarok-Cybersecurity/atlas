// SPDX-License-Identifier: AGPL-3.0-only

//! Whether the radix walk may match a NON-block-aligned tail — the single
//! reader of `AVAROK_PREFIX_SUBBLOCK`.
//!
//! # Why this ships OFF (issue #1193, diagnosed in #692)
//!
//! Both sub-block arms in [`super::inner::RadixTreeInner::walk`] end a match
//! *inside* a KV block and hand that block's physical index to the requester,
//! which pushes it onto its own `block_table`. There is no copy-on-write
//! anywhere in the paged KV path — `PagedKvCache::inc_ref` is a bare counter
//! bump — so the block the requester received is also its own writable tail:
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
//! # Cost of OFF
//!
//! Up to `block_size - 1` tokens of prefill recompute per hit, and
//! non-block-aligned exact-repeat prompts no longer reach the Marconi
//! exact-leaf shortcut (`matched < total`). The way to win those tokens back
//! is a copy-on-write tail — allocate a fresh block, per-layer D2D of the
//! matched rows, event-ordered against the donor's stream — not this lever.
//! Measured counterpoint (`docs/ROBUSTNESS.md`, #936 arm ladder): with the
//! lever armed to `0` the agentic shard's output was byte-identical to
//! baseline, 0 of 251 samples moved.
//!
//! # What is deliberately NOT changed
//!
//! The INSERT side still publishes the partial tail block into
//! `RadixNode::partial_suffix` and still holds the cache's KV ref on it, so
//! `AVAROK_PREFIX_SUBBLOCK=1` restores the pre-#1193 behaviour completely
//! rather than half of it. The cost of that is one pinned-until-evicted block
//! per cached prompt whose length is not block-aligned — unchanged from before
//! this fix, and the thing a copy-on-write tail would put back to work.
//!
//! # Why the lever stays
//!
//! It is one of the #936 arm-ladder levers, so the unsound path must remain
//! reachable for A/B against a future copy-on-write implementation.

/// `AVAROK_PREFIX_SUBBLOCK=1` re-enables sub-block tail matching (unsound; see
/// the module docs). Read here and nowhere else.
///
/// The env read is cached: `walk` runs on every prefix lookup, and nothing
/// mutates the environment after start.
pub(super) fn partial_tail_sharing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| enabled_from(std::env::var("AVAROK_PREFIX_SUBBLOCK").ok().as_deref()))
}

/// The lever's parse, split out so the contract is testable without touching
/// process-global state.
///
/// Unset is the shipped default and is OFF. An unrecognised value is NOT
/// silently coerced: it is reported and refused, because the only values that
/// can turn an unsound path on must be unambiguous — `AVAROK_PREFIX_SUBBLOCK=
/// true` silently reading as ON under the pre-#1193 `!= "0"` rule is exactly
/// the shape of accident this refuses.
fn enabled_from(raw: Option<&str>) -> bool {
    match raw {
        None => false,
        Some("1") => true,
        Some("0") => false,
        Some(other) => {
            tracing::error!(
                "AVAROK_PREFIX_SUBBLOCK={other:?} is not a recognised value (expected \"0\" or \
                 \"1\") — refusing it and leaving sub-block tail matching OFF (see issue #1193)"
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::enabled_from;

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
        for raw in ["true", "yes", "on", "", "1 ", "01", "ON"] {
            assert!(
                !enabled_from(Some(raw)),
                "{raw:?} must not arm sub-block tail matching"
            );
        }
    }
}
