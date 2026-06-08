// SPDX-License-Identifier: AGPL-3.0-only
//
// Cache-key computation and compact JSON serialization — port of
// `SchemaParser::ComputeCacheKey` and the picojson `serialize()`
// calls from `cpp/json_schema_converter.cc`.
//
// `compute_cache_key` produces a canonical, order-independent string
// for a schema fragment so identical sub-schemas share one EBNF rule.
// `serialize_compact` reproduces picojson's `serialize(false)` so the
// integer/number cache keys (e.g. `{"type":"integer"}`) match.

use super::json_value::JsonValue;

/// Keys carrying no semantic meaning for grammar generation, excluded
/// from the cache key. Matches the C++ `kSkippedKeys` set.
const SKIPPED_KEYS: &[&str] = &[
    "title",
    "default",
    "description",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
    "$schema",
];

/// Compute a canonical, order-independent cache key for `schema`.
/// Object keys are sorted; annotation-only keys are dropped. Mirrors
/// `SchemaParser::ComputeCacheKey`.
pub fn compute_cache_key(schema: &JsonValue) -> String {
    match schema {
        JsonValue::Object(entries) => {
            let mut sorted: Vec<&(String, JsonValue)> = entries
                .iter()
                .filter(|(k, _)| !SKIPPED_KEYS.contains(&k.as_str()))
                .collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            let mut result = String::from("{");
            for (idx, (key, value)) in sorted.iter().enumerate() {
                if idx != 0 {
                    result.push(',');
                }
                result.push('"');
                result.push_str(key);
                result.push_str("\":");
                result.push_str(&compute_cache_key(value));
            }
            result.push('}');
            result
        }
        JsonValue::Array(items) => {
            let mut result = String::from("[");
            for (idx, item) in items.iter().enumerate() {
                if idx != 0 {
                    result.push(',');
                }
                result.push_str(&compute_cache_key(item));
            }
            result.push(']');
            result
        }
        scalar => serialize_compact(scalar),
    }
}

/// Serialize a [`JsonValue`] with no extra whitespace — the analogue
/// of picojson's `serialize(false)`. Object keys keep insertion order
/// (picojson serializes in stored order).
pub fn serialize_compact(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number { as_f64, as_i64 } => match as_i64 {
            Some(i) => i.to_string(),
            None => format_number(*as_f64),
        },
        JsonValue::String(s) => escape_json_string(s),
        JsonValue::Array(items) => {
            let mut out = String::from("[");
            for (i, item) in items.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                out.push_str(&serialize_compact(item));
            }
            out.push(']');
            out
        }
        JsonValue::Object(entries) => {
            let mut out = String::from("{");
            for (i, (k, v)) in entries.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                out.push_str(&escape_json_string(k));
                out.push(':');
                out.push_str(&serialize_compact(v));
            }
            out.push('}');
            out
        }
    }
}

/// Render a non-integer double the way picojson does (shortest
/// round-trippable form).
fn format_number(v: f64) -> String {
    if v == v.trunc() && v.is_finite() {
        // Whole-valued double still printed without fraction.
        format!("{}", v as i64)
    } else {
        let s = format!("{v}");
        s
    }
}

/// Escape a string as a JSON string literal (including the quotes).
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_sorts_object_keys() {
        let v = JsonValue::parse(r#"{"b":1,"a":2}"#).unwrap();
        assert_eq!(compute_cache_key(&v), r#"{"a":2,"b":1}"#);
    }

    #[test]
    fn cache_key_skips_annotations() {
        let v = JsonValue::parse(r#"{"type":"integer","title":"x","description":"y"}"#).unwrap();
        assert_eq!(compute_cache_key(&v), r#"{"type":"integer"}"#);
    }

    #[test]
    fn integer_type_cache_key_matches_basic() {
        let v = JsonValue::parse(r#"{"type": "integer"}"#).unwrap();
        assert_eq!(compute_cache_key(&v), "{\"type\":\"integer\"}");
    }

    #[test]
    fn serialize_compact_roundtrips() {
        let v = JsonValue::parse(r#"{ "a" : [1, 2.5, "x"] }"#).unwrap();
        assert_eq!(serialize_compact(&v), r#"{"a":[1,2.5,"x"]}"#);
    }
}
