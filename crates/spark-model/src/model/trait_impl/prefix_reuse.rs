// SPDX-License-Identifier: AGPL-3.0-only

//! SSOT for what `usage.prompt_tokens_details.cached_tokens` reports.
//!
//! Atlas #919 (part 2): a chat completion reported `cached_tokens: 48` while
//! the server log for the same request showed a full prefill and no reuse.
//! The field was stamped from `PrefixMatch::matched_tokens` — the LOOKUP
//! result — at `prefix_lookup.rs` / `prefill_a.rs` / `prefill_c.rs`, before the
//! code decided whether to actually use the match. Three paths find a match,
//! inc_ref its blocks onto the block table, and then recompute all of its KV
//! anyway:
//!
//! * a hybrid-SSM model with no usable SSM snapshot
//!   (`"…but no SSM snapshot — recomputing all KV"`),
//! * the exact-leaf snapshot shortcut bypass (default ON,
//!   `ATLAS_MARCONI_EXACT=1` re-enables the shortcut),
//! * a Marconi restore declined below `marconi_min_tokens()` or by the session
//!   gate.
//!
//! 48 is 3 x the default `--block-size 16`, exactly the shape a block-aligned
//! lookup produces. So the log was right and the usage field was wrong.
//!
//! `matched_tokens` itself keeps its meaning — `cached_prefix_tokens` is
//! load-bearing for block-ref accounting (`sequence.rs` release, `cache_sequence`
//! double-bump avoidance, `forward_layers.rs` KV write floor). The reported
//! count is a SECOND, narrower number: the tokens whose KV this request really
//! read out of the cache instead of recomputing.

/// Prompt tokens actually served from the prefix cache.
///
/// `matched`: `PrefixMatch::matched_tokens` from the lookup.
/// `kv_write_start`: the position the prefill actually starts writing KV at —
///   0 whenever the match was found but discarded.
/// `skip`: whether the prefill took the skip path at all.
///
/// Never exceeds `matched`: the SSM paths can set `kv_write_start` from a
/// snapshot depth, and a snapshot deeper than the matched prefix still only
/// reuses the matched prefix.
pub(crate) fn reused_prefix_tokens(matched: usize, kv_write_start: usize, skip: bool) -> usize {
    if !skip && kv_write_start == 0 {
        return 0;
    }
    kv_write_start.min(matched)
}

#[cfg(test)]
mod tests {
    use super::reused_prefix_tokens;

    /// The reported #919 case: a 48-token block-aligned match that the SSM arm
    /// throws away (`kv_write_start = 0`) must report 0, not 48.
    #[test]
    fn a_discarded_match_reports_zero_not_the_lookup_length() {
        assert_eq!(reused_prefix_tokens(48, 0, false), 0);
    }

    /// The exact-leaf bypass finds a FULL-prompt hit and then recomputes
    /// everything — the worst version of the bug, `cached_tokens == prompt_tokens`
    /// on a 100% full prefill.
    #[test]
    fn the_exact_leaf_bypass_reports_zero() {
        assert_eq!(reused_prefix_tokens(4593, 0, false), 0);
    }

    /// A genuine non-SSM skip reports the whole reused prefix.
    #[test]
    fn a_real_skip_reports_the_reused_prefix() {
        assert_eq!(reused_prefix_tokens(48, 48, true), 48);
    }

    /// An intermediate SSM snapshot reuses only up to the snapshot depth, even
    /// though the lookup matched further.
    #[test]
    fn an_intermediate_snapshot_reports_only_what_it_skipped() {
        assert_eq!(reused_prefix_tokens(512, 256, true), 256);
    }

    /// A snapshot deeper than the matched prefix cannot inflate the count.
    #[test]
    fn the_report_never_exceeds_the_matched_prefix() {
        assert_eq!(reused_prefix_tokens(256, 512, true), 256);
    }

    /// No match at all.
    #[test]
    fn no_match_reports_zero() {
        assert_eq!(reused_prefix_tokens(0, 0, false), 0);
    }
}
