// SPDX-License-Identifier: AGPL-3.0-only
//
// Structural-tag JSON parser — combinatorial formats.
// Split from `parser.rs` to keep each file under the 250-line cap.
// Ports `ParseSequenceFormat`, `ParseOrFormat`, `ParseTagFormat`,
// `ParseTriggeredTagsFormat` and `ParseTagsWithSeparatorFormat` from
// `cpp/structural_tag.cc`.

use super::error::{StructuralTagError, StructuralTagResult};
use super::format::{Format, TagFormat};
use super::parser::{StructuralTagParser, find};
use crate::schema::JsonValue;

impl StructuralTagParser {
    /// Port of `ParseSequenceFormat`.
    pub(super) fn parse_sequence(
        &mut self,
        obj: &[(String, JsonValue)],
    ) -> StructuralTagResult<Format> {
        let elements = self.parse_elements(obj, "Sequence")?;
        Ok(Format::Sequence {
            elements,
            is_unlimited: false,
        })
    }

    /// Port of `ParseOrFormat`.
    pub(super) fn parse_or(&mut self, obj: &[(String, JsonValue)]) -> StructuralTagResult<Format> {
        let elements = self.parse_elements(obj, "Or")?;
        Ok(Format::Or {
            elements,
            is_unlimited: false,
        })
    }

    /// Shared `elements`-array parsing for sequence/or formats.
    fn parse_elements(
        &mut self,
        obj: &[(String, JsonValue)],
        kind: &str,
    ) -> StructuralTagResult<Vec<Format>> {
        let array = find(obj, "elements")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                StructuralTagError::invalid(format!(
                    "{kind} format must have an elements field with an array"
                ))
            })?;
        let mut elements = Vec::with_capacity(array.len());
        for element in array {
            elements.push(self.parse_format(element)?);
        }
        if elements.is_empty() {
            return Err(StructuralTagError::invalid(format!(
                "{kind} format must have at least one element"
            )));
        }
        Ok(elements)
    }

    /// Parse a tag-format value, checking the `type` field if present.
    /// Port of the `picojson::value` overload of `ParseTagFormat`.
    pub(super) fn parse_tag_value(&mut self, value: &JsonValue) -> StructuralTagResult<TagFormat> {
        let obj = value
            .as_object()
            .ok_or_else(|| StructuralTagError::invalid("Tag format must be an object"))?;
        if let Some(t) = find(obj, "type")
            && t.as_str() != Some("tag")
        {
            return Err(StructuralTagError::invalid(
                "Tag format's type must be a string \"tag\"",
            ));
        }
        self.parse_tag_obj(obj)
    }

    /// Parse a tag-format object. Port of the `picojson::object`
    /// overload of `ParseTagFormat`.
    pub(super) fn parse_tag_obj(
        &mut self,
        obj: &[(String, JsonValue)],
    ) -> StructuralTagResult<TagFormat> {
        let begin = find(obj, "begin")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                StructuralTagError::invalid("Tag format's begin field must be a string")
            })?;
        let content_val = find(obj, "content")
            .ok_or_else(|| StructuralTagError::invalid("Tag format must have a content field"))?;
        let content = self.parse_format(content_val)?;
        let end_val = find(obj, "end")
            .ok_or_else(|| StructuralTagError::invalid("Tag format must have an end field"))?;

        let end_strings = parse_end_field(end_val)?;
        Ok(TagFormat {
            begin: begin.to_string(),
            content: Box::new(content),
            end: end_strings,
        })
    }

    /// Parse a list of tags. Shared by triggered/separator formats.
    fn parse_tags(
        &mut self,
        obj: &[(String, JsonValue)],
        kind: &str,
    ) -> StructuralTagResult<Vec<TagFormat>> {
        let array = find(obj, "tags")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                StructuralTagError::invalid(format!(
                    "{kind} format must have a tags field with an array"
                ))
            })?;
        let mut tags = Vec::with_capacity(array.len());
        for tag in array {
            tags.push(self.parse_tag_value(tag)?);
        }
        if tags.is_empty() {
            return Err(StructuralTagError::invalid(format!(
                "{kind} format's tags must be non-empty"
            )));
        }
        Ok(tags)
    }

    /// Port of `ParseTriggeredTagsFormat`.
    pub(super) fn parse_triggered_tags(
        &mut self,
        obj: &[(String, JsonValue)],
    ) -> StructuralTagResult<Format> {
        let triggers_arr = find(obj, "triggers")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                StructuralTagError::invalid(
                    "Triggered tags format must have a triggers field with an array",
                )
            })?;
        let mut triggers = Vec::with_capacity(triggers_arr.len());
        for trigger in triggers_arr {
            match trigger.as_str() {
                Some(s) if !s.is_empty() => triggers.push(s.to_string()),
                _ => {
                    return Err(StructuralTagError::invalid(
                        "Triggered tags format's triggers must be non-empty strings",
                    ));
                }
            }
        }
        if triggers.is_empty() {
            return Err(StructuralTagError::invalid(
                "Triggered tags format's triggers must be non-empty",
            ));
        }
        let tags = self.parse_tags(obj, "Triggered tags")?;
        let excludes = parse_excludes(obj)?;
        let at_least_one = parse_bool_field(obj, "at_least_one")?;
        let stop_after_first = parse_bool_field(obj, "stop_after_first")?;
        Ok(Format::TriggeredTags {
            triggers,
            tags,
            excludes,
            at_least_one,
            stop_after_first,
            detected_end_strs: Vec::new(),
        })
    }

    /// Port of `ParseTagsWithSeparatorFormat`.
    pub(super) fn parse_tags_with_separator(
        &mut self,
        obj: &[(String, JsonValue)],
    ) -> StructuralTagResult<Format> {
        let tags = self.parse_tags(obj, "Tags with separator")?;
        let separator = find(obj, "separator")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                StructuralTagError::invalid(
                    "Tags with separator format's separator field must be a string",
                )
            })?;
        let at_least_one = parse_bool_field(obj, "at_least_one")?;
        let stop_after_first = parse_bool_field(obj, "stop_after_first")?;
        Ok(Format::TagsWithSeparator {
            tags,
            separator: separator.to_string(),
            at_least_one,
            stop_after_first,
            detected_end_strs: Vec::new(),
        })
    }
}

/// Parse the `end` field: a string, or a non-empty array of strings.
fn parse_end_field(end_val: &JsonValue) -> StructuralTagResult<Vec<String>> {
    if let Some(s) = end_val.as_str() {
        return Ok(vec![s.to_string()]);
    }
    if let Some(array) = end_val.as_array() {
        if array.is_empty() {
            return Err(StructuralTagError::invalid(
                "Tag format's end array cannot be empty",
            ));
        }
        let mut end_strings = Vec::with_capacity(array.len());
        for item in array {
            let s = item.as_str().ok_or_else(|| {
                StructuralTagError::invalid("Tag format's end array must contain only strings")
            })?;
            end_strings.push(s.to_string());
        }
        return Ok(end_strings);
    }
    Err(StructuralTagError::invalid(
        "Tag format's end field must be a string or array of strings",
    ))
}

/// Parse the optional `excludes` array (non-empty strings).
fn parse_excludes(obj: &[(String, JsonValue)]) -> StructuralTagResult<Vec<String>> {
    let array = match find(obj, "excludes") {
        None => return Ok(Vec::new()),
        Some(v) => v.as_array().ok_or_else(|| {
            StructuralTagError::invalid(
                "Triggered tags format should have a excludes field with an array",
            )
        })?,
    };
    let mut excludes = Vec::with_capacity(array.len());
    for item in array {
        match item.as_str() {
            Some(s) if !s.is_empty() => excludes.push(s.to_string()),
            _ => {
                return Err(StructuralTagError::invalid(
                    "Triggered tags format's excluded_strs must be non-empty strings",
                ));
            }
        }
    }
    Ok(excludes)
}

/// Parse an optional boolean field, defaulting to `false`.
fn parse_bool_field(obj: &[(String, JsonValue)], key: &str) -> StructuralTagResult<bool> {
    match find(obj, key) {
        None => Ok(false),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| StructuralTagError::invalid(format!("{key} must be a boolean"))),
    }
}
