// SPDX-License-Identifier: AGPL-3.0-only
//
// Structural-tag analyzer — port of `StructuralTagAnalyzer` from
// `cpp/structural_tag.cc`.
//
// Walks the [`Format`] tree and fills in the analyzer-only fields:
//  * `AnyText`/`TriggeredTags`/`TagsWithSeparator`: `detected_end_strs`
//    — the end strings of the nearest enclosing tag, so an unbounded
//    inner format knows where to stop.
//  * `Sequence`/`Or`: `is_unlimited` — whether the format can match
//    arbitrarily long input.
//  * `Tag`: clears `end` when its content is unlimited (the end string
//    is taken over by the inner format's `detected_end_strs`).
//
// The C++ keeps a stack of `FormatPtrVariant` to find the enclosing
// tag's end strings. Here we pass the enclosing tag's end strings down
// the recursion explicitly — equivalent, and simpler without raw
// pointers (zero `unsafe`).

use super::error::{StructuralTagError, StructuralTagResult};
use super::format::{Format, StructuralTag, TagFormat};

/// Run the analyzer over a structural tag, mutating its [`Format`] tree
/// in place. Port of `StructuralTagAnalyzer::Analyze`.
pub(super) fn analyze(structural_tag: &mut StructuralTag) -> StructuralTagResult<()> {
    visit(&mut structural_tag.format, &[], 0)
}

/// Whether `format` can match arbitrarily long input.
/// Port of `StructuralTagAnalyzer::IsUnlimited`.
fn format_is_unlimited(format: &Format) -> bool {
    match format {
        Format::AnyText { .. }
        | Format::TriggeredTags { .. }
        | Format::TagsWithSeparator { .. } => true,
        Format::Sequence { is_unlimited, .. } | Format::Or { is_unlimited, .. } => *is_unlimited,
        _ => false,
    }
}

/// Whether `format` carries non-empty excludes.
/// Port of `StructuralTagAnalyzer::IsExcluded`.
fn is_excluded(format: &Format) -> bool {
    match format {
        Format::AnyText { excludes, .. } => !excludes.is_empty(),
        Format::TriggeredTags { excludes, .. } => !excludes.is_empty(),
        _ => false,
    }
}

/// Visit a format. `enclosing_end` is the end-strings of the nearest
/// enclosing tag. `depth` guards against pathological nesting.
fn visit(format: &mut Format, enclosing_end: &[String], depth: u32) -> StructuralTagResult<()> {
    if depth > super::parser::MAX_FORMAT_DEPTH {
        return Err(StructuralTagError::invalid("Format nesting too deep"));
    }
    match format {
        Format::ConstString(_)
        | Format::JsonSchema { .. }
        | Format::Grammar(_)
        | Format::Regex(_) => Ok(()),

        Format::AnyText {
            detected_end_strs, ..
        } => {
            *detected_end_strs = enclosing_end.to_vec();
            Ok(())
        }

        Format::Sequence {
            elements,
            is_unlimited,
        } => visit_sequence(elements, is_unlimited, enclosing_end, depth),

        Format::Or {
            elements,
            is_unlimited,
        } => visit_or(elements, is_unlimited, enclosing_end, depth),

        Format::Tag(tag) => visit_tag(tag, depth),

        Format::TriggeredTags {
            tags,
            detected_end_strs,
            ..
        } => {
            for tag in tags.iter_mut() {
                visit_tag(tag, depth + 1)?;
            }
            *detected_end_strs = enclosing_end.to_vec();
            Ok(())
        }

        Format::TagsWithSeparator {
            tags,
            detected_end_strs,
            ..
        } => {
            for tag in tags.iter_mut() {
                visit_tag(tag, depth + 1)?;
            }
            *detected_end_strs = enclosing_end.to_vec();
            Ok(())
        }
    }
}

/// Port of `StructuralTagAnalyzer::VisitSub(SequenceFormat*)`.
fn visit_sequence(
    elements: &mut [Format],
    is_unlimited: &mut bool,
    enclosing_end: &[String],
    depth: u32,
) -> StructuralTagResult<()> {
    let n = elements.len();
    for (i, element) in elements.iter_mut().enumerate() {
        visit(element, enclosing_end, depth + 1)?;
        if i + 1 < n && format_is_unlimited(element) && !is_excluded(element) {
            return Err(StructuralTagError::invalid(format!(
                "Only the last element in a sequence can be unlimited, but the {i}th \
                 element of sequence format is unlimited"
            )));
        }
    }
    if let Some(last) = elements.last() {
        *is_unlimited = format_is_unlimited(last) && !is_excluded(last);
    }
    Ok(())
}

/// Port of `StructuralTagAnalyzer::VisitSub(OrFormat*)`.
fn visit_or(
    elements: &mut [Format],
    is_unlimited: &mut bool,
    enclosing_end: &[String],
    depth: u32,
) -> StructuralTagResult<()> {
    let mut any_unlimited = false;
    let mut all_unlimited = true;
    for element in elements.iter_mut() {
        visit(element, enclosing_end, depth + 1)?;
        let unlimited = format_is_unlimited(element) && !is_excluded(element);
        any_unlimited |= unlimited;
        all_unlimited &= unlimited;
    }
    if any_unlimited && !all_unlimited {
        return Err(StructuralTagError::invalid(
            "Now we only support all elements in an or format to be unlimited or all \
             limited, but the or format has both unlimited and limited elements",
        ));
    }
    *is_unlimited = any_unlimited;
    Ok(())
}

/// Port of `StructuralTagAnalyzer::VisitSub(TagFormat*)`.
fn visit_tag(tag: &mut TagFormat, depth: u32) -> StructuralTagResult<()> {
    // The content's enclosing-tag end strings are this tag's `end`.
    visit(&mut tag.content, &tag.end, depth + 1)?;
    if format_is_unlimited(&tag.content) {
        let has_non_empty = tag.end.iter().any(|s| !s.is_empty());
        if !has_non_empty {
            if is_excluded(&tag.content) {
                return Ok(());
            }
            return Err(StructuralTagError::invalid(
                "When the content is unlimited, at least one end string must be non-empty",
            ));
        }
        // End strings are now carried by the content's detected_end_strs.
        tag.end.clear();
    }
    Ok(())
}
