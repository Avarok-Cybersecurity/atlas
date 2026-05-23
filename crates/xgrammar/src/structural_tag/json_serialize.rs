// SPDX-License-Identifier: AGPL-3.0-only
//
// Compact, order-preserving JSON serializer for [`JsonValue`].
//
// WHY THIS EXISTS
// ---------------
// The C++ `StructuralTagParser` stores the embedded `json_schema`
// sub-document by calling picojson's `value.serialize(false)` — a
// compact re-serialization. The schema converter then re-parses that
// string. Object key order must survive the round-trip because JSON
// Schema conversion is order-sensitive (`properties` declaration
// order). `serde_json::Map` is not order-preserving in this crate's
// feature set, so we serialize the crate's own ordered `JsonValue`
// directly here. This mirrors picojson `serialize(false)`.

use crate::schema::JsonValue;

/// Compactly serialize a [`JsonValue`] to a JSON string, preserving
/// object key order. Equivalent to picojson `serialize(false)`.
pub fn serialize_compact(value: &JsonValue) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

fn write_value(value: &JsonValue, out: &mut String) {
    match value {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Number { as_f64, as_i64 } => match as_i64 {
            Some(i) => out.push_str(&i.to_string()),
            None => out.push_str(&format_f64(*as_f64)),
        },
        JsonValue::String(s) => write_string(s, out),
        JsonValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        JsonValue::Object(entries) => {
            out.push('{');
            for (i, (key, val)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_value(val, out);
            }
            out.push('}');
        }
    }
}

/// Render an `f64` without a trailing `.0` for integral values, the
/// same shape picojson produces for non-integer-flagged numbers.
fn format_f64(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v}");
        s
    }
}

/// Write `s` as a JSON string literal with the standard escapes.
fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::JsonValue;

    fn roundtrip(text: &str) -> String {
        serialize_compact(&JsonValue::parse(text).expect("parse"))
    }

    #[test]
    fn preserves_object_key_order() {
        assert_eq!(roundtrip(r#"{"b":1,"a":2}"#), r#"{"b":1,"a":2}"#);
    }

    #[test]
    fn serializes_scalars() {
        assert_eq!(roundtrip("true"), "true");
        assert_eq!(roundtrip("null"), "null");
        assert_eq!(roundtrip("42"), "42");
        assert_eq!(roundtrip(r#""hi""#), r#""hi""#);
    }

    #[test]
    fn serializes_nested() {
        let s = r#"{"type":"object","properties":{"a":{"type":"string"}}}"#;
        assert_eq!(roundtrip(s), s);
    }

    #[test]
    fn escapes_strings() {
        assert_eq!(roundtrip(r#""a\nb""#), r#""a\nb""#);
    }
}
