// SPDX-License-Identifier: AGPL-3.0-only
//
// Structural-tag → grammar converter — the tag family.
// Split from `converter.rs` to keep each file under the 250-line cap.
// Ports `VisitSub(TagFormat)`, `VisitSub(TriggeredTagsFormat)` and
// `VisitSub(TagsWithSeparatorFormat)` from `cpp/structural_tag.cc`.
//
// This is the tool-call envelope: a tag is `begin content end`; a
// triggered-tags format dispatches on a trigger prefix, generating
// free text until a trigger appears, then the matched tag body, then
// loops. That is precisely `<tool_call>{json}</tool_call>` enforcement.

use super::converter::StructuralTagConverter;
use super::error::{StructuralTagError, StructuralTagResult};
use super::format::TagFormat;

impl StructuralTagConverter {
    /// Build an expr that matches any one of `ends` (`begin`-relative
    /// view excluded). Empty strings become `EmptyStr`. A single end is
    /// returned directly; multiple ends become a `tag_end` rule ref.
    fn end_expr(&mut self, ends: &[String]) -> StructuralTagResult<i32> {
        if ends.len() == 1 {
            return Ok(self.empty_or_bytes(&ends[0]));
        }
        let mut seqs = Vec::with_capacity(ends.len());
        for end in ends {
            let e = self.empty_or_bytes(end);
            seqs.push(self.builder.add_sequence(&[e]));
        }
        let choices = self.builder.add_choices(&seqs);
        let rule = self.add_rule("tag_end", choices)?;
        Ok(self.builder.add_rule_ref(rule))
    }

    /// `EmptyStr` for empty input, otherwise a `ByteString`.
    pub(super) fn empty_or_bytes(&mut self, s: &str) -> i32 {
        if s.is_empty() {
            self.builder.add_empty_str()
        } else {
            self.builder.add_byte_string(s)
        }
    }

    /// Build a `Sequence(begin content [end])` expr for one tag, given
    /// the already-built content rule id and a `begin` byte string.
    fn tag_sequence(
        &mut self,
        begin_expr: i32,
        content_rule_id: i32,
        end: &[String],
    ) -> StructuralTagResult<i32> {
        let content_ref = self.builder.add_rule_ref(content_rule_id);
        let elems = if end.is_empty() {
            vec![begin_expr, content_ref]
        } else {
            let end_expr = self.end_expr(end)?;
            vec![begin_expr, content_ref, end_expr]
        };
        Ok(self.builder.add_sequence(&elems))
    }

    /// Port of `VisitSub(TagFormat)`.
    pub(super) fn visit_tag(&mut self, tag: &TagFormat) -> StructuralTagResult<i32> {
        let content_rule = self.visit(&tag.content)?;
        let begin_expr = self.builder.add_byte_string(&tag.begin);
        let seq = self.tag_sequence(begin_expr, content_rule, &tag.end)?;
        let choices = self.builder.add_choices(&[seq]);
        self.add_rule("tag", choices)
    }

    /// Convert every tag's content rule and validate trigger matching.
    /// Returns `(content_rule_ids, trigger_to_tag_ids)`.
    fn build_tag_contents(
        &mut self,
        triggers: &[String],
        tags: &[TagFormat],
    ) -> StructuralTagResult<(Vec<i32>, Vec<Vec<usize>>)> {
        let mut trigger_to_tag_ids: Vec<Vec<usize>> = vec![Vec::new(); triggers.len()];
        let mut content_rule_ids = Vec::with_capacity(tags.len());
        for (tag_idx, tag) in tags.iter().enumerate() {
            let mut matched: Option<usize> = None;
            for (trig_idx, trigger) in triggers.iter().enumerate() {
                if tag.begin.starts_with(trigger.as_str()) {
                    if matched.is_some() {
                        return Err(StructuralTagError::invalid(
                            "One tag matches multiple triggers in a triggered tags format",
                        ));
                    }
                    matched = Some(trig_idx);
                }
            }
            let trig_idx = matched.ok_or_else(|| {
                StructuralTagError::invalid(
                    "One tag does not match any trigger in a triggered tags format",
                )
            })?;
            trigger_to_tag_ids[trig_idx].push(tag_idx);
            content_rule_ids.push(self.visit(&tag.content)?);
        }
        Ok((content_rule_ids, trigger_to_tag_ids))
    }

    /// Build a `Choices` of `begin content [end]` over `tags`, using
    /// `begin_of` to derive each tag's begin byte string.
    fn tag_choices<F: Fn(&TagFormat) -> String>(
        &mut self,
        tags: &[TagFormat],
        content_rule_ids: &[i32],
        indices: &[usize],
        begin_of: F,
    ) -> StructuralTagResult<i32> {
        let mut choices = Vec::with_capacity(indices.len());
        for &i in indices {
            let begin_expr = self.builder.add_byte_string(&begin_of(&tags[i]));
            choices.push(self.tag_sequence(begin_expr, content_rule_ids[i], &tags[i].end)?);
        }
        Ok(self.builder.add_choices(&choices))
    }

    /// Port of `VisitSub(TriggeredTagsFormat)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn visit_triggered_tags(
        &mut self,
        triggers: &[String],
        tags: &[TagFormat],
        excludes: &[String],
        at_least_one: bool,
        stop_after_first: bool,
        detected_end_strs: &[String],
    ) -> StructuralTagResult<i32> {
        let (content_rule_ids, trigger_to_tag_ids) = self.build_tag_contents(triggers, tags)?;
        let all_indices: Vec<usize> = (0..tags.len()).collect();

        // Special case: at_least_one && stop_after_first -> one tag.
        if at_least_one && stop_after_first {
            let mut body =
                self.tag_choices(tags, &content_rule_ids, &all_indices, |t| t.begin.clone())?;
            if !detected_end_strs.is_empty() {
                let sub = self.add_rule("triggered_tags_sub", body)?;
                let sub_ref = self.builder.add_rule_ref(sub);
                let end_expr = self.detected_end_expr(detected_end_strs)?;
                let seq = self.builder.add_sequence(&[sub_ref, end_expr]);
                body = self.builder.add_choices(&[seq]);
            }
            return self.add_rule("triggered_tags", body);
        }

        // Normal case: text + triggered tags via TagDispatch.
        let mut tag_rule_pairs = Vec::with_capacity(triggers.len());
        for (trig_idx, trigger) in triggers.iter().enumerate() {
            let choices = self.tag_choices(
                tags,
                &content_rule_ids,
                &trigger_to_tag_ids[trig_idx],
                |t| t.begin[trigger.len()..].to_string(),
            )?;
            let sub = self.add_rule("triggered_tags_group", choices)?;
            tag_rule_pairs.push((trigger.clone(), sub));
        }

        let non_empty_ends: Vec<String> = detected_end_strs
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let loop_after = !stop_after_first;
        let mut rule_expr = if non_empty_ends.is_empty() {
            self.builder.add_tag_dispatch(&super::tag_dispatch_spec(
                tag_rule_pairs,
                true,
                Vec::new(),
                loop_after,
                excludes.to_vec(),
            ))
        } else {
            self.builder.add_tag_dispatch(&super::tag_dispatch_spec(
                tag_rule_pairs,
                false,
                non_empty_ends,
                loop_after,
                excludes.to_vec(),
            ))
        };

        if at_least_one {
            let first_choices =
                self.tag_choices(tags, &content_rule_ids, &all_indices, |t| t.begin.clone())?;
            let first_rule = self.add_rule("triggered_tags_first", first_choices)?;
            let dispatch_rule = self.add_rule("triggered_tags_sub", rule_expr)?;
            let first_ref = self.builder.add_rule_ref(first_rule);
            let dispatch_ref = self.builder.add_rule_ref(dispatch_rule);
            let seq = self.builder.add_sequence(&[first_ref, dispatch_ref]);
            rule_expr = self.builder.add_choices(&[seq]);
        }
        self.add_rule("triggered_tags", rule_expr)
    }

    /// Build the detected-end-strings expr for a non-empty list.
    fn detected_end_expr(&mut self, ends: &[String]) -> StructuralTagResult<i32> {
        if ends.len() == 1 {
            return Ok(self.empty_or_bytes(&ends[0]));
        }
        let mut seqs = Vec::with_capacity(ends.len());
        for end in ends {
            let e = self.empty_or_bytes(end);
            seqs.push(self.builder.add_sequence(&[e]));
        }
        let choices = self.builder.add_choices(&seqs);
        let rule = self.add_rule("end_choices", choices)?;
        Ok(self.builder.add_rule_ref(rule))
    }

    /// Port of `VisitSub(TagsWithSeparatorFormat)`.
    pub(super) fn visit_tags_with_separator(
        &mut self,
        tags: &[TagFormat],
        separator: &str,
        at_least_one: bool,
        stop_after_first: bool,
        detected_end_strs: &[String],
    ) -> StructuralTagResult<i32> {
        // Rule matching any tag.
        let mut choice_ids = Vec::with_capacity(tags.len());
        for tag in tags {
            let tag_rule = self.visit_tag(tag)?;
            let tag_ref = self.builder.add_rule_ref(tag_rule);
            choice_ids.push(self.builder.add_sequence(&[tag_ref]));
        }
        let all_tags_choices = self.builder.add_choices(&choice_ids);
        let all_tags_rule = self.add_rule("tags_with_separator_tags", all_tags_choices)?;

        super::converter_sep::convert(
            self,
            all_tags_rule,
            separator,
            at_least_one,
            stop_after_first,
            detected_end_strs,
        )
    }
}
