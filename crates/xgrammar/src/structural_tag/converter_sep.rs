// SPDX-License-Identifier: AGPL-3.0-only
//
// Structural-tag → grammar converter — tags-with-separator assembly.
// Split from `converter_tags.rs` to keep each file under the 250-line
// cap. Ports steps 2 and 3 of `VisitSub(TagsWithSeparatorFormat)` from
// `cpp/structural_tag.cc`.
//
// Grammar shape (separator `sep`, end strings `end`):
//   stop_after_first || sep matches an end string:
//     at_least_one  : root ::= tags end1 | tags end2 | ...
//     otherwise     : root ::= tags end1 | ... | end1 | ... | ""
//   normal (looping):
//     root      ::= tags sub  (| end  when !at_least_one)
//     sub       ::= sep tags sub | end

use super::converter::StructuralTagConverter;
use super::error::StructuralTagResult;

/// Run steps 2/3 of the tags-with-separator conversion. `all_tags_rule`
/// is the already-built rule matching any single tag.
pub(super) fn convert(
    conv: &mut StructuralTagConverter,
    all_tags_rule: i32,
    separator: &str,
    at_least_one: bool,
    stop_after_first: bool,
    detected_end_strs: &[String],
) -> StructuralTagResult<i32> {
    // Non-empty end strings, as byte-string expr ids.
    let end_str_expr_ids: Vec<i32> = detected_end_strs
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| conv.builder.add_byte_string(s))
        .collect();
    let has_end_strs = !end_str_expr_ids.is_empty();
    let separator_matches_end = detected_end_strs.iter().any(|e| e == separator);

    // Step 2: stop_after_first OR separator collides with an end string.
    if stop_after_first || (has_end_strs && separator_matches_end) {
        let body = build_stop_body(
            conv,
            all_tags_rule,
            &end_str_expr_ids,
            has_end_strs,
            at_least_one,
        );
        return conv.add_rule("tags_with_separator", body);
    }

    // Step 3: normal looping handling.
    let sub_rule = conv
        .builder
        .add_empty_rule_with_hint("tags_with_separator_sub")
        .map_err(|e| super::error::StructuralTagError::invalid(e.to_string()))?;

    let end_str_sequence = build_end_sequence(conv, &end_str_expr_ids);

    // Recursive case body: (sep) all_tags sub.
    let mut sub_seq_elems = Vec::new();
    if !separator.is_empty() {
        sub_seq_elems.push(conv.builder.add_byte_string(separator));
    }
    sub_seq_elems.push(conv.builder.add_rule_ref(all_tags_rule));
    sub_seq_elems.push(conv.builder.add_rule_ref(sub_rule));
    let sub_seq = conv.builder.add_sequence(&sub_seq_elems);
    let sub_body = conv.builder.add_choices(&[sub_seq, end_str_sequence]);
    conv.builder
        .update_rule_body(sub_rule, sub_body)
        .map_err(|e| super::error::StructuralTagError::invalid(e.to_string()))?;

    // Root rule: all_tags sub  (| end).
    let all_tags_ref = conv.builder.add_rule_ref(all_tags_rule);
    let sub_ref = conv.builder.add_rule_ref(sub_rule);
    let main_seq = conv.builder.add_sequence(&[all_tags_ref, sub_ref]);
    let mut choices = vec![main_seq];
    if !at_least_one {
        choices.push(end_str_sequence);
    }
    let root_body = conv.builder.add_choices(&choices);
    conv.add_rule("tags_with_separator", root_body)
}

/// Build the rule body for the stop-after-first / collision case.
fn build_stop_body(
    conv: &mut StructuralTagConverter,
    all_tags_rule: i32,
    end_str_expr_ids: &[i32],
    has_end_strs: bool,
    at_least_one: bool,
) -> i32 {
    let all_tags_ref = conv.builder.add_rule_ref(all_tags_rule);
    if at_least_one {
        if !has_end_strs {
            let seq = conv.builder.add_sequence(&[all_tags_ref]);
            return conv.builder.add_choices(&[seq]);
        }
        let choices: Vec<i32> = end_str_expr_ids
            .iter()
            .map(|&e| conv.builder.add_sequence(&[all_tags_ref, e]))
            .collect();
        return conv.builder.add_choices(&choices);
    }
    if !has_end_strs {
        let seq = conv.builder.add_sequence(&[all_tags_ref]);
        let empty = conv.builder.add_empty_str();
        return conv.builder.add_choices(&[seq, empty]);
    }
    let mut choices: Vec<i32> = end_str_expr_ids
        .iter()
        .map(|&e| conv.builder.add_sequence(&[all_tags_ref, e]))
        .collect();
    for &e in end_str_expr_ids {
        let s = conv.builder.add_sequence(&[e]);
        choices.push(s);
    }
    conv.builder.add_choices(&choices)
}

/// Build the end-string expr: `EmptyStr` if no ends, a single sequence
/// if one, otherwise a `Choices` of single-element sequences.
fn build_end_sequence(conv: &mut StructuralTagConverter, end_str_expr_ids: &[i32]) -> i32 {
    match end_str_expr_ids.len() {
        0 => conv.builder.add_empty_str(),
        1 => conv.builder.add_sequence(&[end_str_expr_ids[0]]),
        _ => {
            let seqs: Vec<i32> = end_str_expr_ids
                .iter()
                .map(|&e| conv.builder.add_sequence(&[e]))
                .collect();
            conv.builder.add_choices(&seqs)
        }
    }
}
