// SPDX-License-Identifier: AGPL-3.0-only
//
// Pure stop-string accumulator + hold-back algorithm extracted from
// `handle_token`. Self-contained so the algorithm can be unit-tested
// without spinning up a full `StreamCtx`.

/// Pure stop-string accumulator + hold-back algorithm. Returns the
/// bytes that should be forwarded to the client this delta; the
/// remainder (≤ `buffer_len` bytes) stays withheld inside
/// `accumulated_content` until the next call or until `handle_done`
/// flushes the tail at stream close.
///
/// Mirrors vLLM's `IncrementalDetokenizer.update`
/// (`vllm/v1/engine/detokenizer.py`):
/// 1. Append `new_chars` to the accumulator.
/// 2. Search the accumulator for any stop string.
/// 3a. On hit, truncate the accumulator AND the emittable delta at
///     the match position (Atlas never echoes the stop literal).
/// 3b. On miss, hold back the last `buffer_len` bytes; emit
///     everything between the previously emitted offset and the
///     hold-back boundary, snapped to a valid UTF-8 char boundary.
///
/// Pre/postconditions:
/// - `*triggered` must be `false` on entry (callers gate on this).
/// - On match, `*triggered` is flipped to `true` and the accumulator
///   is truncated to the prefix that precedes the stop string.
/// - On miss, `*triggered` stays `false`.
pub(super) fn apply_stop_string_holdback(
    new_chars: &str,
    stop_strings: &[String],
    buffer_len: usize,
    accumulated_content: &mut String,
    emitted_len: &mut usize,
    triggered: &mut bool,
) -> String {
    debug_assert!(!*triggered, "caller must gate on !triggered");
    accumulated_content.push_str(new_chars);

    // Bounded search window: vLLM only scans the suffix that could
    // contain a stop string straddling the new chars. Atlas keeps the
    // simpler full-string scan here because Atlas accumulators are
    // already bounded by the per-request token budget and the inner
    // memchr-driven `str::find` is O(n) anyway.
    let matched_pos = stop_strings
        .iter()
        .filter_map(|s| accumulated_content.find(s.as_str()))
        .min();

    if let Some(pos) = matched_pos {
        accumulated_content.truncate(pos);
        let emit_start = (*emitted_len).min(pos);
        let out = accumulated_content[emit_start..pos].to_string();
        *emitted_len = pos;
        *triggered = true;
        return out;
    }

    // No match: hold back the last `buffer_len` bytes. Snap to a UTF-8
    // boundary so the emitted prefix is always valid Rust `str` and
    // the held-back tail never contains a partial codepoint.
    let acc_len = accumulated_content.len();
    let raw_emit_end = acc_len.saturating_sub(buffer_len);
    let emit_end = accumulated_content.floor_char_boundary(raw_emit_end);
    let emit_start = (*emitted_len).min(emit_end);
    let out = accumulated_content[emit_start..emit_end].to_string();
    *emitted_len = emit_end;
    out
}

#[cfg(test)]
mod stop_string_holdback_tests {
    use super::apply_stop_string_holdback;

    /// Stop string spanning a chunk boundary must not leak the
    /// partial prefix in the first delta. When the suffix arrives in
    /// the next chunk the full output up to (but excluding) the stop
    /// string is emitted; the stop literal itself is consumed.
    #[test]
    fn stop_string_spanning_chunk_boundary_does_not_leak() {
        let stops = vec!["<|im_start|>".to_string()];
        let buffer_len = "<|im_start|>".len() - 1; // 11
        let mut acc = String::new();
        let mut emitted = 0usize;
        let mut triggered = false;

        // Delta 1: "hello " — entirely inside the hold-back window
        // (6 bytes < buffer_len=11). Nothing emitted.
        let out = apply_stop_string_holdback(
            "hello ",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "");
        assert_eq!(acc, "hello ");
        assert!(!triggered);

        // Delta 2: "<|im_st" — partial stop string. acc="hello <|im_st"
        // (len=13). raw_emit_end=13-11=2, so we emit "he".
        // Crucially, "<|im_st" is HELD BACK — never sent to client.
        let out = apply_stop_string_holdback(
            "<|im_st",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "he");
        assert!(!out.contains("<|im_st"), "partial stop leaked to client");
        assert!(!triggered);

        // Delta 3: "art|>" completes the stop string. We match at
        // pos=6, truncate acc to "hello ", and emit "llo " (bytes
        // 2..6 of acc).
        let out = apply_stop_string_holdback(
            "art|>",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "llo ");
        assert_eq!(acc, "hello ");
        assert!(triggered);

        // Concatenating all emitted deltas yields the pre-stop output
        // ("hello ") with the stop literal consumed. No partial leak.
        let total = String::new() + "" + "he" + "llo ";
        assert_eq!(total, "hello ");
        assert!(!total.contains("<|im_st"));
        assert!(!total.contains("<|im_start|>"));
    }

    /// When no stop strings are configured, `buffer_len=0` and the
    /// hold-back collapses to a pass-through: every byte of every
    /// delta is emitted immediately.
    #[test]
    fn no_stop_strings_is_zero_behavior_change() {
        let stops: Vec<String> = Vec::new();
        let buffer_len = 0usize;
        let mut acc = String::new();
        let mut emitted = 0usize;
        let mut triggered = false;

        let out = apply_stop_string_holdback(
            "hello ",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "hello ");
        assert!(!triggered);

        let out = apply_stop_string_holdback(
            "world",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "world");
        assert!(!triggered);

        // Even a string that LOOKS like a stop marker is forwarded
        // verbatim because no stop strings are configured.
        let out = apply_stop_string_holdback(
            "<|im_start|>",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "<|im_start|>");
        assert!(!triggered);
    }

    /// Multi-byte UTF-8 inside the hold-back window must never be
    /// sliced mid-codepoint. `floor_char_boundary` snaps the cut to a
    /// valid boundary so the emitted prefix is always valid `str`.
    #[test]
    fn utf8_boundary_safety_in_holdback() {
        // "é" is 2 bytes (0xC3 0xA9). Build an accumulator whose
        // raw cut would land inside the codepoint and verify
        // floor_char_boundary saves us.
        let stops = vec!["STOP".to_string()];
        let buffer_len = 3usize; // > 0 to exercise the hold-back
        let mut acc = String::new();
        let mut emitted = 0usize;
        let mut triggered = false;

        // acc becomes "aébc" (5 bytes). raw_emit_end = 5-3 = 2 lands
        // mid-codepoint of 'é' (1..3). floor_char_boundary snaps to
        // 1, so we emit "a" only.
        let out = apply_stop_string_holdback(
            "aébc",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "a");
        assert!(out.is_char_boundary(out.len()));
        assert!(!triggered);
    }
}
