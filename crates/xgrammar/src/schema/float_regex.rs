// SPDX-License-Identifier: AGPL-3.0-only
//
// Float range -> regex generation — port of
// `JSONSchemaConverter::{FormatFloat,GenerateFloatRangeRegex}` from
// `cpp/json_schema_converter.cc`.

use super::range_regex::generate_range_regex;

/// Format a double to at most `precision` fractional digits, trimming
/// trailing zeros. Port of `FormatFloat`.
fn format_float(value: f64, precision: i32) -> String {
    if value == (value as i64) as f64 {
        return (value as i64).to_string();
    }
    let mut result = format!("{value:.*}", precision as usize);
    if let Some(dot) = result.find('.') {
        let trimmed_len = result.trim_end_matches('0').len();
        if trimmed_len > dot {
            result.truncate(trimmed_len);
        } else if trimmed_len == dot {
            result.truncate(dot);
        }
    }
    result
}

/// Escape the regex-significant `.` in a formatted-float boundary
/// string so it is treated as a literal decimal point rather than the
/// wildcard. Port of `EscapeDotForRegex` (upstream commit c4cf39f, #642).
fn escape_dot_for_regex(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        if c == '.' {
            result.push_str("\\.");
        } else {
            result.push(c);
        }
    }
    result
}

/// Strip the leading `^(` and trailing `)$` from an integer range
/// regex so it can be embedded.
fn strip_anchors(s: &str) -> String {
    if s.len() >= 2 {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Generate a regex matching floats in `[start, end]` inclusive, each
/// bound `None` meaning unbounded, with at most `precision`
/// fractional digits. Port of `GenerateFloatRangeRegex`.
pub fn generate_float_range_regex(start: Option<f64>, end: Option<f64>, precision: i32) -> String {
    if let (Some(s), Some(e)) = (start, end)
        && s > e
    {
        return "^()$".to_string();
    }
    if start.is_none() && end.is_none() {
        return format!("^-?\\d+(\\.\\d{{1,{precision}}})?$");
    }

    let mut parts: Vec<String> = Vec::new();

    let mut start_int = 0i64;
    let mut end_int = 0i64;
    let mut start_frac = 0.0;
    let mut end_frac = 0.0;
    let mut is_start_neg = false;
    let mut is_end_neg = false;

    if let Some(s) = start {
        is_start_neg = s < 0.0;
        start_int = s.floor() as i64;
        start_frac = s - start_int as f64;
    }
    if let Some(e) = end {
        is_end_neg = e < 0.0;
        end_int = e.floor() as i64;
        end_frac = e - end_int as f64;
    }

    match (start, end) {
        (Some(s), None) => {
            let s_str = format_float(s, precision);
            // Escape the literal `.` so it is not the regex wildcard
            // (upstream commit c4cf39f, #642).
            parts.push(escape_dot_for_regex(&s_str));
            if start_frac > 0.0 {
                emit_frac_above(&mut parts, &s_str, is_start_neg, precision);
            }
            if start_int < i64::MAX - 1 {
                let ir = strip_anchors(&generate_range_regex(Some(start_int + 1), None));
                parts.push(format!("{ir}(\\.\\d{{1,{precision}}})?"));
            }
        }
        (None, Some(e)) => {
            let e_str = format_float(e, precision);
            // Escape the literal `.` (upstream commit c4cf39f, #642).
            parts.push(escape_dot_for_regex(&e_str));
            if end_frac > 0.0 {
                emit_frac_below(&mut parts, &e_str, is_end_neg, precision);
            }
            if end_int > i64::MIN + 1 {
                let ir = strip_anchors(&generate_range_regex(None, Some(end_int - 1)));
                parts.push(format!("{ir}(\\.\\d{{1,{precision}}})?"));
            }
        }
        (Some(s), Some(e)) => {
            let s_str = format_float(s, precision);
            let e_str = format_float(e, precision);
            if start_int == end_int {
                if start_frac == 0.0 && end_frac == 0.0 {
                    parts.push(start_int.to_string());
                } else {
                    // Escape literal `.` in float bounds (upstream c4cf39f, #642).
                    parts.push(escape_dot_for_regex(&s_str));
                    if s_str != e_str {
                        parts.push(escape_dot_for_regex(&e_str));
                    }
                }
            } else {
                // Escape literal `.` in float bounds (upstream c4cf39f, #642).
                parts.push(escape_dot_for_regex(&s_str));
                if s_str != e_str {
                    parts.push(escape_dot_for_regex(&e_str));
                }
                if end_int > start_int + 1 {
                    let ir = strip_anchors(&generate_range_regex(
                        Some(start_int + 1),
                        Some(end_int - 1),
                    ));
                    parts.push(format!("{ir}(\\.\\d{{1,{precision}}})?"));
                }
                if start_frac > 0.0 {
                    emit_frac_above(&mut parts, &s_str, is_start_neg, precision);
                } else {
                    parts.push(format!("{start_int}\\.\\d{{1,{precision}}}"));
                }
                if end_frac > 0.0 {
                    emit_frac_below(&mut parts, &e_str, is_end_neg, precision);
                } else {
                    parts.push(format!("{end_int}\\.\\d{{1,{precision}}}"));
                }
            }
        }
        (None, None) => unreachable!(),
    }

    format!("^({})$", parts.join("|"))
}

/// Emit the patterns covering fractional values strictly greater than
/// the lower bound's fraction (mirrors the C++ start-side loop).
fn emit_frac_above(parts: &mut Vec<String>, num_str: &str, is_neg: bool, precision: i32) {
    let dot = match num_str.find('.') {
        Some(d) => d,
        None => return,
    };
    let int_part = &num_str[..dot];
    let frac: Vec<char> = num_str[dot + 1..].chars().collect();
    for (i, &fc) in frac.iter().enumerate() {
        if i == 0 {
            if is_neg {
                for d in '0'..fc {
                    parts.push(format!("{int_part}\\.{d}\\d{{0,{}}}", precision - 1));
                }
            } else {
                let mut d = (fc as u8 + 1) as char;
                while d <= '9' {
                    parts.push(format!("{int_part}\\.{d}\\d{{0,{}}}", precision - 1));
                    d = (d as u8 + 1) as char;
                }
            }
        } else {
            let pref: String = frac[..i].iter().collect();
            if is_neg {
                if fc > '0' {
                    for d in '0'..fc {
                        parts.push(format!(
                            "{int_part}\\.{pref}{d}\\d{{0,{}}}",
                            precision - i as i32 - 1
                        ));
                    }
                }
            } else {
                let mut d = (fc as u8 + 1) as char;
                while d <= '9' {
                    parts.push(format!(
                        "{int_part}\\.{pref}{d}\\d{{0,{}}}",
                        precision - i as i32 - 1
                    ));
                    d = (d as u8 + 1) as char;
                }
            }
        }
    }
}

/// Emit the patterns covering fractional values strictly less than
/// the upper bound's fraction (mirrors the C++ end-side loop).
fn emit_frac_below(parts: &mut Vec<String>, num_str: &str, is_neg: bool, precision: i32) {
    let dot = match num_str.find('.') {
        Some(d) => d,
        None => return,
    };
    let int_part = &num_str[..dot];
    let frac: Vec<char> = num_str[dot + 1..].chars().collect();
    for (i, &fc) in frac.iter().enumerate() {
        if i == 0 {
            if is_neg {
                let mut d = (fc as u8 + 1) as char;
                while d <= '9' {
                    parts.push(format!("{int_part}\\.{d}\\d{{0,{}}}", precision - 1));
                    d = (d as u8 + 1) as char;
                }
            } else {
                for d in '0'..fc {
                    parts.push(format!("{int_part}\\.{d}\\d{{0,{}}}", precision - 1));
                }
            }
        } else if is_neg {
            let pref: String = frac[..i].iter().collect();
            let mut d = (fc as u8 + 1) as char;
            while d <= '9' {
                parts.push(format!(
                    "{int_part}\\.{pref}{d}\\d{{0,{}}}",
                    precision - i as i32 - 1
                ));
                d = (d as u8 + 1) as char;
            }
        } else if fc > '0' {
            let pref: String = frac[..i].iter().collect();
            for d in '0'..fc {
                parts.push(format!(
                    "{int_part}\\.{pref}{d}\\d{{0,{}}}",
                    precision - i as i32 - 1
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbounded_float() {
        assert_eq!(
            generate_float_range_regex(None, None, 6),
            r"^-?\d+(\.\d{1,6})?$"
        );
    }

    #[test]
    fn inverted_range_is_empty() {
        assert_eq!(generate_float_range_regex(Some(5.0), Some(1.0), 6), "^()$");
    }

    #[test]
    fn whole_number_format() {
        assert_eq!(format_float(3.0, 6), "3");
        assert_eq!(format_float(3.5, 6), "3.5");
        assert_eq!(format_float(3.120000, 6), "3.12");
    }

    #[test]
    fn bounded_produces_anchored_regex() {
        let r = generate_float_range_regex(Some(1.0), Some(10.0), 6);
        assert!(r.starts_with("^(") && r.ends_with(")$"));
    }

    #[test]
    fn escape_dot_helper() {
        assert_eq!(escape_dot_for_regex("0.5"), r"0\.5");
        assert_eq!(escape_dot_for_regex("-3.125"), r"-3\.125");
        assert_eq!(escape_dot_for_regex("42"), "42");
    }

    #[test]
    fn float_boundary_dot_is_escaped_not_wildcard() {
        // Regression for upstream c4cf39f (#642): the decimal point of a
        // float boundary must be a literal `\.`, never an unescaped `.`
        // wildcard that would accept `0,5` etc.
        let r = generate_float_range_regex(Some(0.5), None, 6);
        assert!(r.contains(r"0\.5"), "expected escaped boundary in {r}");
        // No bare `.` may appear except as part of an escaped `\.`.
        let bytes: Vec<char> = r.chars().collect();
        for (i, &c) in bytes.iter().enumerate() {
            if c == '.' {
                assert!(i > 0 && bytes[i - 1] == '\\', "unescaped dot in {r}");
            }
        }
    }

    #[test]
    fn float_boundary_dot_escaped_both_bounds() {
        let r = generate_float_range_regex(Some(0.25), Some(0.75), 6);
        assert!(r.contains(r"0\.25"), "expected escaped start in {r}");
        assert!(r.contains(r"0\.75"), "expected escaped end in {r}");
    }
}
