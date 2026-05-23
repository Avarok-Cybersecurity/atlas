// SPDX-License-Identifier: AGPL-3.0-only
//
// Composite-keyword parsers — port of `SchemaParser::Parse{Const,
// Enum,Ref,AnyOf,AllOf,TypeArray}` and `SchemaParser::ResolveRef`
// from `cpp/json_schema_converter.cc`.

use super::cache_key::serialize_compact;
use super::error::{SchemaError, SchemaResult};
use super::json_value::JsonValue;
use super::parser::SchemaParser;
use super::spec::{SchemaSpec, SchemaSpecPtr, SpecKind};

impl SchemaParser {
    /// Parse a `const` schema into a [`SpecKind::Const`].
    pub(super) fn parse_const(&self, schema: &JsonValue) -> SchemaResult<SpecKind> {
        let value = schema
            .get("const")
            .ok_or_else(|| SchemaError::invalid("const keyword missing"))?;
        Ok(SpecKind::Const(serialize_compact(value)))
    }

    /// Parse an `enum` schema into a [`SpecKind::Enum`].
    pub(super) fn parse_enum(&self, schema: &JsonValue) -> SchemaResult<SpecKind> {
        let arr = schema
            .get("enum")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| SchemaError::invalid("enum must be an array"))?;
        let values = arr.iter().map(serialize_compact).collect();
        Ok(SpecKind::Enum(values))
    }

    /// Extract the `$ref` URI string. Port of `ParseRef`.
    pub(super) fn parse_ref(&self, schema: &JsonValue) -> SchemaResult<String> {
        match schema.get("$ref") {
            Some(JsonValue::String(s)) => Ok(s.clone()),
            _ => Err(SchemaError::invalid("$ref must be a string")),
        }
    }

    /// Parse `anyOf` / `oneOf`. Port of `ParseAnyOf`.
    pub(super) fn parse_any_of(&mut self, schema: &JsonValue) -> SchemaResult<SpecKind> {
        let key = if schema.contains_key("anyOf") {
            "anyOf"
        } else {
            "oneOf"
        };
        let arr = schema
            .get(key)
            .and_then(JsonValue::as_array)
            .ok_or_else(|| SchemaError::invalid(format!("{key} must be an array")))?
            .to_vec();
        let mut options = Vec::with_capacity(arr.len());
        for (idx, option) in arr.iter().enumerate() {
            options.push(self.parse(option, &format!("case_{idx}"), None)?);
        }
        Ok(SpecKind::AnyOf(options))
    }

    /// Parse `allOf`. Port of `ParseAllOf`.
    pub(super) fn parse_all_of(&mut self, schema: &JsonValue) -> SchemaResult<SpecKind> {
        let arr = schema
            .get("allOf")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| SchemaError::invalid("allOf must be an array"))?
            .to_vec();
        let mut schemas = Vec::with_capacity(arr.len());
        for (idx, sub) in arr.iter().enumerate() {
            schemas.push(self.parse(sub, &format!("all_{idx}"), None)?);
        }
        Ok(SpecKind::AllOf(schemas))
    }

    /// Parse a `"type": [...]` array. Port of `ParseTypeArray`.
    pub(super) fn parse_type_array(
        &mut self,
        schema: &JsonValue,
        hint: &str,
    ) -> SchemaResult<SpecKind> {
        let type_arr = schema
            .get("type")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| SchemaError::invalid("type must be an array"))?
            .to_vec();
        let base = schema
            .as_object()
            .ok_or_else(|| SchemaError::invalid("schema must be an object"))?;

        let mut type_schemas = Vec::new();
        if type_arr.is_empty() {
            // Empty type array: drop `type` and re-parse as "any".
            let copy: Vec<(String, JsonValue)> =
                base.iter().filter(|(k, _)| k != "type").cloned().collect();
            let parsed = self.parse(&JsonValue::Object(copy), hint, None)?;
            type_schemas.push(parsed);
            return Ok(SpecKind::TypeArray(type_schemas));
        }
        for type_val in &type_arr {
            let type_name = type_val
                .as_str()
                .ok_or_else(|| {
                    SchemaError::invalid("type must be a string or an array of strings")
                })?
                .to_string();
            let copy: Vec<(String, JsonValue)> = base
                .iter()
                .map(|(k, v)| {
                    if k == "type" {
                        (k.clone(), type_val.clone())
                    } else {
                        (k.clone(), v.clone())
                    }
                })
                .collect();
            let sub_hint = format!("{hint}_{type_name}");
            let parsed = self.parse(&JsonValue::Object(copy), &sub_hint, None)?;
            type_schemas.push(parsed);
        }
        Ok(SpecKind::TypeArray(type_schemas))
    }

    /// Resolve a `$ref` URI to a [`SchemaSpec`]. Handles `#`
    /// (whole-document), `#/path/to/def` JSON-pointer style refs, and
    /// circular references via a placeholder cache entry. Port of
    /// `ResolveRef`.
    pub fn resolve_ref(&mut self, uri: &str, _rule_name_hint: &str) -> SchemaResult<SchemaSpecPtr> {
        if let Some(cached) = self.ref_cache.get(uri) {
            return Ok(cached.clone());
        }

        if uri == "#" {
            // Insert a placeholder to break direct self-recursion.
            let placeholder = SchemaSpec::make(SpecKind::Any, "", "root");
            self.ref_cache.insert(uri.to_string(), placeholder);
            let root = self.root_schema.clone();
            let resolved = self.parse(&root, "root", None)?;
            self.ref_cache.insert(uri.to_string(), resolved.clone());
            return Ok(resolved);
        }

        if uri.len() < 2 || !uri.starts_with("#/") {
            // C++ warns and falls back to "any" for malformed URIs.
            return Ok(SchemaSpec::make(SpecKind::Any, "", "any"));
        }

        let mut parts: Vec<String> = Vec::new();
        let mut rule_prefix = String::new();
        for part in uri[2..].split('/') {
            if !part.is_empty() {
                parts.push(part.to_string());
            }
            if !rule_prefix.is_empty() {
                rule_prefix.push('_');
            }
            for c in part.chars() {
                if c.is_alphabetic() || c == '_' || c == '-' || c == '.' {
                    rule_prefix.push(c);
                }
            }
        }

        let mut current = self.root_schema.clone();
        for p in &parts {
            let next = match current.get(p) {
                Some(v) => v.clone(),
                None => {
                    return Err(SchemaError::ref_error(format!(
                        "Cannot find field {p} in {uri}"
                    )));
                }
            };
            current = next;
        }

        let hint = if rule_prefix.is_empty() {
            "ref".to_string()
        } else {
            rule_prefix
        };
        let resolved = self.parse(&current, &hint, None)?;
        self.ref_cache.insert(uri.to_string(), resolved.clone());
        Ok(resolved)
    }
}
