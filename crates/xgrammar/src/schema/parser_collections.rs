// SPDX-License-Identifier: AGPL-3.0-only
//
// Collection-type parsers — port of `SchemaParser::ParseArray` and
// `SchemaParser::ParseObject` from `cpp/json_schema_converter.cc`.

use super::error::{SchemaError, SchemaResult};
use super::json_value::JsonValue;
use super::parser::SchemaParser;
use super::spec::{ArraySpec, ObjectSpec, PatternProperty, Property, SchemaSpec, SpecKind};

impl SchemaParser {
    /// Parse an `array`-typed schema. Port of `ParseArray`.
    pub(super) fn parse_array(&mut self, schema: &JsonValue) -> SchemaResult<SpecKind> {
        let mut spec = ArraySpec::default();

        if let Some(prefix) = schema.get("prefixItems") {
            let arr = prefix
                .as_array()
                .ok_or_else(|| SchemaError::invalid("prefixItems must be an array"))?;
            for item in arr {
                if let Some(false) = item.as_bool() {
                    return Err(SchemaError::unsatisfiable("prefixItems contains false"));
                }
                if !item.is_object() && item.as_bool().is_none() {
                    return Err(SchemaError::invalid(
                        "prefixItems must be an array of objects or booleans",
                    ));
                }
                let parsed = self.parse(item, "prefix_item", None)?;
                spec.prefix_items.push(parsed);
            }
        }

        if let Some(items) = schema.get("items") {
            if items.as_bool().is_none() && !items.is_object() {
                return Err(SchemaError::invalid("items must be a boolean or an object"));
            }
            if let Some(false) = items.as_bool() {
                spec.allow_additional_items = false;
            } else {
                spec.allow_additional_items = true;
                spec.additional_items = Some(self.parse(items, "item", None)?);
            }
        } else if let Some(uneval) = schema.get("unevaluatedItems") {
            if uneval.as_bool().is_none() && !uneval.is_object() {
                return Err(SchemaError::invalid(
                    "unevaluatedItems must be a boolean or an object",
                ));
            }
            if let Some(false) = uneval.as_bool() {
                spec.allow_additional_items = false;
            } else {
                spec.allow_additional_items = true;
                spec.additional_items = Some(self.parse(uneval, "unevaluated_item", None)?);
            }
        } else if !self.strict_mode {
            spec.allow_additional_items = true;
            spec.additional_items = Some(SchemaSpec::make(SpecKind::Any, "", "any"));
        } else {
            spec.allow_additional_items = false;
        }

        if let Some(v) = schema.get("minItems") {
            let i = Self::int_keyword(v, "minItems")?;
            spec.min_items = i.max(0);
        }
        if let Some(v) = schema.get("minContains") {
            let i = Self::int_keyword(v, "minContains")?;
            spec.min_items = spec.min_items.max(i);
        }
        if let Some(v) = schema.get("maxItems") {
            let i = Self::int_keyword(v, "maxItems")?;
            if i < 0 {
                return Err(SchemaError::invalid(
                    "maxItems must be a non-negative integer",
                ));
            }
            spec.max_items = i;
        }

        let prefix_size = spec.prefix_items.len() as i64;
        if spec.max_items != -1 && spec.min_items > spec.max_items {
            return Err(SchemaError::unsatisfiable(format!(
                "minItems is greater than maxItems: {} > {}",
                spec.min_items, spec.max_items
            )));
        }
        if spec.max_items != -1 && spec.max_items < prefix_size {
            return Err(SchemaError::unsatisfiable(format!(
                "maxItems is less than the number of prefixItems: {} < {}",
                spec.max_items, prefix_size
            )));
        }
        if !spec.allow_additional_items {
            if prefix_size < spec.min_items {
                return Err(SchemaError::unsatisfiable(
                    "minItems is greater than the number of prefixItems, but additional \
                     items are not allowed",
                ));
            }
            if spec.max_items != -1 && prefix_size > spec.max_items {
                return Err(SchemaError::unsatisfiable(
                    "maxItems is less than the number of prefixItems, but additional \
                     items are not allowed",
                ));
            }
        }
        Ok(SpecKind::Array(spec))
    }

    /// Parse an `object`-typed schema. Port of `ParseObject`.
    pub(super) fn parse_object(&mut self, schema: &JsonValue) -> SchemaResult<SpecKind> {
        let mut spec = ObjectSpec::default();

        if let Some(props) = schema.get("properties") {
            let entries = props
                .as_object()
                .ok_or_else(|| SchemaError::invalid("properties must be an object"))?;
            for (key, value) in entries {
                let parsed = self.parse(value, key, None)?;
                spec.properties.push(Property {
                    name: key.clone(),
                    schema: parsed,
                });
            }
        }

        if let Some(req) = schema.get("required") {
            let arr = req
                .as_array()
                .ok_or_else(|| SchemaError::invalid("required must be an array"))?;
            for r in arr {
                let name = r
                    .as_str()
                    .ok_or_else(|| SchemaError::invalid("required entries must be strings"))?;
                if !spec.required.iter().any(|x| x == name) {
                    spec.required.push(name.to_string());
                }
            }
        }

        if let Some(pp) = schema.get("patternProperties") {
            let entries = pp
                .as_object()
                .ok_or_else(|| SchemaError::invalid("patternProperties must be an object"))?;
            for (key, value) in entries {
                let parsed = self.parse(value, "pattern_prop", None)?;
                spec.pattern_properties.push(PatternProperty {
                    pattern: key.clone(),
                    schema: parsed,
                });
            }
        }

        if let Some(prop_names) = schema.get("propertyNames") {
            if !prop_names.is_object() {
                return Err(SchemaError::invalid("propertyNames must be an object"));
            }
            if let Some(JsonValue::String(t)) = prop_names.get("type")
                && t != "string"
            {
                return Err(SchemaError::unsatisfiable(
                    "propertyNames must be an object that validates string",
                ));
            }
            spec.property_names = Some(self.parse(prop_names, "property_name", Some("string"))?);
        }

        spec.allow_additional_properties = !self.strict_mode;
        if let Some(add) = schema.get("additionalProperties") {
            match add.as_bool() {
                Some(b) => spec.allow_additional_properties = b,
                None => {
                    spec.allow_additional_properties = true;
                    spec.additional_properties_schema =
                        Some(self.parse(add, "additional", None)?);
                }
            }
        }

        spec.allow_unevaluated_properties = true;
        if schema.contains_key("additionalProperties") {
            spec.allow_unevaluated_properties = spec.allow_additional_properties;
        } else if let Some(uneval) = schema.get("unevaluatedProperties") {
            match uneval.as_bool() {
                Some(b) => spec.allow_unevaluated_properties = b,
                None => {
                    spec.allow_unevaluated_properties = true;
                    spec.unevaluated_properties_schema =
                        Some(self.parse(uneval, "unevaluated", None)?);
                }
            }
        } else if self.strict_mode {
            spec.allow_unevaluated_properties = false;
        }

        if let Some(v) = schema.get("minProperties") {
            let i = Self::int_keyword(v, "minProperties")?;
            if i < 0 {
                return Err(SchemaError::unsatisfiable(
                    "minProperties must be a non-negative integer",
                ));
            }
            spec.min_properties = i;
        }
        if let Some(v) = schema.get("maxProperties") {
            let i = Self::int_keyword(v, "maxProperties")?;
            if i < 0 {
                return Err(SchemaError::unsatisfiable(
                    "maxProperties must be a non-negative integer",
                ));
            }
            spec.max_properties = i;
        }

        if spec.max_properties != -1 && spec.min_properties > spec.max_properties {
            return Err(SchemaError::unsatisfiable(format!(
                "minProperties is greater than maxProperties: {} > {}",
                spec.min_properties, spec.max_properties
            )));
        }
        if spec.max_properties != -1 && spec.required.len() as i64 > spec.max_properties {
            return Err(SchemaError::unsatisfiable(
                "maxProperties is less than the number of required properties",
            ));
        }
        if spec.pattern_properties.is_empty()
            && spec.property_names.is_none()
            && !spec.allow_additional_properties
            && !spec.allow_unevaluated_properties
            && spec.min_properties > spec.properties.len() as i64
        {
            return Err(SchemaError::unsatisfiable(
                "minProperties is greater than the number of properties, but additional \
                 properties aren't allowed",
            ));
        }
        Ok(SpecKind::Object(spec))
    }

    /// Read an integer-valued keyword, erroring on non-integers.
    fn int_keyword(value: &JsonValue, name: &str) -> SchemaResult<i64> {
        match value {
            JsonValue::Number {
                as_i64: Some(i), ..
            } => Ok(*i),
            _ => Err(SchemaError::invalid(format!("{name} must be an integer"))),
        }
    }
}
