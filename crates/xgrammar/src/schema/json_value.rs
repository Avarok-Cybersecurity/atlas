// SPDX-License-Identifier: AGPL-3.0-only
//
// Order-preserving JSON value — a thin layer over `serde_json`.
//
// WHY THIS EXISTS
// ---------------
// JSON Schema conversion is order-sensitive: object `properties` must
// be emitted in declaration order (the C++ uses picojson's
// `ordered_keys()`). `serde_json::Map` only preserves insertion order
// when the crate's `preserve_order` feature is enabled, and this
// crate's `Cargo.toml` may not be modified. So we parse the document
// once into our own `JsonValue` whose objects keep insertion order.
//
// The parser is a straightforward recursive-descent JSON reader. It
// is not a general-purpose JSON library — it exists only to feed the
// schema converter with ordered objects. Numbers are kept as both an
// `f64` and an exact-integer flag so integer-bound checks stay exact.

/// An order-preserving JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    /// A JSON number. `is_int` is set when the literal had no
    /// fraction/exponent and fits in `i64`.
    Number {
        as_f64: f64,
        as_i64: Option<i64>,
    },
    String(String),
    Array(Vec<JsonValue>),
    /// Object with insertion-ordered entries.
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Parse `text` as JSON, preserving object key order.
    pub fn parse(text: &str) -> Result<JsonValue, String> {
        let chars: Vec<char> = text.chars().collect();
        let mut p = Parser { chars, pos: 0 };
        p.skip_ws();
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.chars.len() {
            return Err(format!("trailing characters at position {}", p.pos));
        }
        Ok(v)
    }

    /// Whether this value is a JSON object.
    pub fn is_object(&self) -> bool {
        matches!(self, JsonValue::Object(_))
    }

    /// Borrow as an object's entry slice, if it is one.
    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow as an array slice, if it is one.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Borrow as a string, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow as a bool, if it is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Look up a key in an object.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Whether an object contains `key`.
    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() {
            match self.chars[self.pos] {
                ' ' | '\t' | '\n' | '\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(JsonValue::String(self.parse_string()?)),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!("unexpected character '{c}' at {}", self.pos)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected '{c}' at position {}", self.pos))
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.expect('{')?;
        let mut entries: Vec<(String, JsonValue)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':')?;
            let value = self.parse_value()?;
            // Last-key-wins, matching JSON object semantics.
            if let Some(slot) = entries.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = value;
            } else {
                entries.push((key, value));
            }
            self.skip_ws();
            match self.peek() {
                Some(',') => self.pos += 1,
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
        Ok(JsonValue::Object(entries))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(',') => self.pos += 1,
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
        Ok(JsonValue::Array(items))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".to_string()),
                Some('"') => {
                    self.pos += 1;
                    break;
                }
                Some('\\') => {
                    self.pos += 1;
                    let esc = self.peek().ok_or("dangling escape")?;
                    self.pos += 1;
                    match esc {
                        '"' => s.push('"'),
                        '\\' => s.push('\\'),
                        '/' => s.push('/'),
                        'b' => s.push('\u{0008}'),
                        'f' => s.push('\u{000C}'),
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        'u' => {
                            let cp = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&cp) {
                                // High surrogate: expect a low surrogate.
                                if self.peek() == Some('\\') {
                                    self.pos += 1;
                                    self.expect('u')?;
                                    let lo = self.parse_hex4()?;
                                    let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                    s.push(char::from_u32(c).ok_or("invalid surrogate pair")?);
                                } else {
                                    return Err("lone high surrogate".to_string());
                                }
                            } else {
                                s.push(char::from_u32(cp).ok_or("invalid \\u escape")?);
                            }
                        }
                        other => return Err(format!("bad escape '\\{other}'")),
                    }
                }
                Some(c) => {
                    self.pos += 1;
                    s.push(c);
                }
            }
        }
        Ok(s)
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.peek().ok_or("short \\u escape")?;
            let d = c.to_digit(16).ok_or("bad hex digit")?;
            v = v * 16 + d;
            self.pos += 1;
        }
        Ok(v)
    }

    fn parse_bool(&mut self) -> Result<JsonValue, String> {
        if self.matches_literal("true") {
            Ok(JsonValue::Bool(true))
        } else if self.matches_literal("false") {
            Ok(JsonValue::Bool(false))
        } else {
            Err(format!("invalid literal at {}", self.pos))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, String> {
        if self.matches_literal("null") {
            Ok(JsonValue::Null)
        } else {
            Err(format!("invalid literal at {}", self.pos))
        }
    }

    fn matches_literal(&mut self, lit: &str) -> bool {
        let lit_chars: Vec<char> = lit.chars().collect();
        if self.pos + lit_chars.len() > self.chars.len() {
            return false;
        }
        if self.chars[self.pos..self.pos + lit_chars.len()] == lit_chars[..] {
            self.pos += lit_chars.len();
            true
        } else {
            false
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        let mut is_int = true;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            match c {
                '0'..='9' => self.pos += 1,
                '.' | 'e' | 'E' | '+' | '-' => {
                    is_int = false;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        let as_f64: f64 = text
            .parse()
            .map_err(|_| format!("invalid number '{text}'"))?;
        let as_i64 = if is_int {
            text.parse::<i64>().ok()
        } else {
            None
        };
        Ok(JsonValue::Number { as_f64, as_i64 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ordered_object() {
        let v = JsonValue::parse(r#"{"b":1,"a":2,"c":3}"#).unwrap();
        let keys: Vec<&str> = v
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(keys, ["b", "a", "c"]);
    }

    #[test]
    fn parses_scalars_and_nesting() {
        let v = JsonValue::parse(r#"{"x":[true,null,-3.5,"hi"]}"#).unwrap();
        let arr = v.get("x").unwrap().as_array().unwrap();
        assert_eq!(arr[0], JsonValue::Bool(true));
        assert_eq!(arr[1], JsonValue::Null);
        assert_eq!(arr[3].as_str(), Some("hi"));
    }

    #[test]
    fn integer_flag_is_exact() {
        let v = JsonValue::parse("42").unwrap();
        match v {
            JsonValue::Number { as_i64, .. } => assert_eq!(as_i64, Some(42)),
            _ => panic!("expected number"),
        }
        let f = JsonValue::parse("42.0").unwrap();
        match f {
            JsonValue::Number { as_i64, .. } => assert_eq!(as_i64, None),
            _ => panic!("expected number"),
        }
    }

    #[test]
    fn rejects_trailing_junk() {
        assert!(JsonValue::parse("{} extra").is_err());
        assert!(JsonValue::parse("{").is_err());
    }

    #[test]
    fn parses_escapes() {
        let v = JsonValue::parse(r#""a\nbA""#).unwrap();
        assert_eq!(v.as_str(), Some("a\nbA"));
    }
}
