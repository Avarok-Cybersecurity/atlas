// SPDX-License-Identifier: AGPL-3.0-only
//
// Integer range -> regex generation — port of
// `JSONSchemaConverter::{MakePatternForDigitRange,GenerateNumberPatterns,
// GenerateSubRangeRegex,GenerateRangeRegex}` from
// `cpp/json_schema_converter.cc`.
//
// Produces a regex (anchored with `^...$`) matching all integers in a
// closed `[start, end]` range, where `None` means unbounded.

/// A digit-range pattern for one fixed-length slot: either a literal
/// digit or a `[a-b]` class, optionally followed by `\d{n}`.
fn make_pattern_for_digit_range(start: char, end: char, remaining: i32) -> String {
    let mut out = String::new();
    if start == end {
        out.push(start);
    } else {
        out.push('[');
        out.push(start);
        out.push('-');
        out.push(end);
        out.push(']');
    }
    if remaining > 0 {
        out.push_str(&format!("\\d{{{remaining}}}"));
    }
    out
}

/// Generate the alternation of fixed-length digit patterns covering
/// the positive range `[lower, upper]`. Port of `GenerateNumberPatterns`.
fn generate_number_patterns(lower: i64, upper: i64) -> Vec<String> {
    let mut patterns = Vec::new();
    let lower_len = lower.to_string().len() as i32;
    let upper_len = upper.to_string().len() as i32;

    for len in lower_len..=upper_len {
        let digit_min = 10i64.pow((len - 1) as u32);
        let digit_max = 10i64.pow(len as u32) - 1;
        let start = if len == lower_len { lower } else { digit_min };
        let end = if len == upper_len { upper } else { digit_max };
        let start_str = start.to_string();
        let end_str = end.to_string();
        let sb: Vec<char> = start_str.chars().collect();
        let eb: Vec<char> = end_str.chars().collect();

        if len == 1 {
            patterns.push(make_pattern_for_digit_range(sb[0], eb[0], 0));
            continue;
        }

        let mut prefix = 0usize;
        while prefix < len as usize && sb[prefix] == eb[prefix] {
            prefix += 1;
        }
        if prefix == len as usize {
            patterns.push(start_str.clone());
            continue;
        }
        if prefix > 0 && prefix >= (len as usize).saturating_sub(2) {
            let common: String = sb[..prefix].iter().collect();
            patterns.push(format!(
                "{common}{}",
                make_pattern_for_digit_range(sb[prefix], eb[prefix], len - prefix as i32 - 1)
            ));
            continue;
        }

        if len == lower_len && len == upper_len {
            push_full_len_patterns(
                &mut patterns,
                start,
                end,
                digit_min,
                digit_max,
                &sb,
                &eb,
                len,
            );
        } else if len == lower_len && len != upper_len {
            if start == digit_min {
                patterns.push(format!("[1-9]\\d{{{}}}", len - 1));
            } else {
                push_start_open(&mut patterns, &sb, len);
                patterns.push(start_str.clone());
            }
        } else if len != lower_len && len == upper_len {
            if end == digit_max {
                patterns.push(format!("[1-9]\\d{{{}}}", len - 1));
            } else {
                push_end_open(&mut patterns, &eb, len);
                patterns.push(end_str.clone());
            }
        } else {
            patterns.push(format!("[1-9]\\d{{{}}}", len - 1));
        }
    }
    patterns
}

/// Append the patterns generated when the length covers both the
/// lower and upper bound exactly.
#[allow(clippy::too_many_arguments)]
fn push_full_len_patterns(
    patterns: &mut Vec<String>,
    start: i64,
    end: i64,
    digit_min: i64,
    digit_max: i64,
    sb: &[char],
    eb: &[char],
    len: i32,
) {
    let start_str: String = sb.iter().collect();
    let end_str: String = eb.iter().collect();
    if start == digit_max {
        patterns.push(start_str);
    } else if start == digit_min {
        if end == digit_max {
            patterns.push(format!("[1-9]\\d{{{}}}", len - 1));
        } else {
            for (i, &ec) in eb.iter().enumerate() {
                if i == 0 {
                    if ec > '1' {
                        patterns.push(make_pattern_for_digit_range('1', prev_char(ec), len - 1));
                    }
                } else {
                    let pref: String = eb[..i].iter().collect();
                    if ec > '0' {
                        patterns.push(format!(
                            "{pref}{}",
                            make_pattern_for_digit_range('0', prev_char(ec), len - i as i32 - 1)
                        ));
                    }
                }
            }
            patterns.push(end_str);
        }
    } else if end == digit_max {
        for (i, &sc) in sb.iter().enumerate() {
            if i == 0 {
                if sc < '9' {
                    patterns.push(make_pattern_for_digit_range(next_char(sc), '9', len - 1));
                }
            } else {
                let pref: String = sb[..i].iter().collect();
                if sc < '9' {
                    patterns.push(format!(
                        "{pref}{}",
                        make_pattern_for_digit_range(next_char(sc), '9', len - i as i32 - 1)
                    ));
                }
            }
        }
        patterns.push(start_str);
    } else {
        push_full_distinct_first(patterns, sb, eb, len);
    }
}

/// The hardest sub-case: first digits differ and both bounds are
/// strictly inside the length band.
fn push_full_distinct_first(patterns: &mut Vec<String>, sb: &[char], eb: &[char], len: i32) {
    let start_str: String = sb.iter().collect();
    let end_str: String = eb.iter().collect();
    let sf = sb[0];
    let ef = eb[0];
    if (ef as i32) - (sf as i32) > 1 {
        patterns.push(make_pattern_for_digit_range(
            next_char(sf),
            prev_char(ef),
            len - 1,
        ));
    }
    for (i, &sc) in sb.iter().enumerate() {
        if i == 0 {
            let pref: String = sb[..1].iter().collect();
            if sb[1] < '9' {
                patterns.push(format!(
                    "{pref}{}",
                    make_pattern_for_digit_range(next_char(sb[1]), '9', len - 2)
                ));
            }
        } else {
            let pref: String = sb[..i].iter().collect();
            if sc < '9' {
                patterns.push(format!(
                    "{pref}{}",
                    make_pattern_for_digit_range(next_char(sc), '9', len - i as i32 - 1)
                ));
            }
        }
    }
    patterns.push(start_str);
    for (i, &ec) in eb.iter().enumerate() {
        if i == 0 {
            let pref: String = eb[..1].iter().collect();
            if eb[1] > '0' {
                patterns.push(format!(
                    "{pref}{}",
                    make_pattern_for_digit_range('0', prev_char(eb[1]), len - 2)
                ));
            }
        } else {
            let pref: String = eb[..i].iter().collect();
            if ec > '0' {
                patterns.push(format!(
                    "{pref}{}",
                    make_pattern_for_digit_range('0', prev_char(ec), len - i as i32 - 1)
                ));
            }
        }
    }
    patterns.push(end_str);
}

/// Open-ended-above patterns starting from `sb` of length `len`.
fn push_start_open(patterns: &mut Vec<String>, sb: &[char], len: i32) {
    for (i, &sc) in sb.iter().enumerate() {
        if i == 0 {
            if sc < '9' {
                patterns.push(make_pattern_for_digit_range(next_char(sc), '9', len - 1));
            }
        } else {
            let pref: String = sb[..i].iter().collect();
            if sc < '9' {
                patterns.push(format!(
                    "{pref}{}",
                    make_pattern_for_digit_range(next_char(sc), '9', len - i as i32 - 1)
                ));
            }
        }
    }
}

/// Open-ended-below patterns ending at `eb` of length `len`.
fn push_end_open(patterns: &mut Vec<String>, eb: &[char], len: i32) {
    for (i, &ec) in eb.iter().enumerate() {
        if i == 0 {
            if ec > '1' {
                patterns.push(make_pattern_for_digit_range('1', prev_char(ec), len - 1));
            }
        } else {
            let pref: String = eb[..i].iter().collect();
            if ec > '0' {
                patterns.push(format!(
                    "{pref}{}",
                    make_pattern_for_digit_range('0', prev_char(ec), len - i as i32 - 1)
                ));
            }
        }
    }
}

fn next_char(c: char) -> char {
    char::from(c as u8 + 1)
}
fn prev_char(c: char) -> char {
    char::from(c as u8 - 1)
}

/// Generate the parenthesized alternation for the positive range
/// `[lower, upper]`. Port of `GenerateSubRangeRegex`.
fn generate_sub_range_regex(lower: i64, upper: i64) -> String {
    let patterns = generate_number_patterns(lower, upper);
    format!("({})", patterns.join("|"))
}

/// Generate a regex matching integers in `[start, end]` (inclusive),
/// each bound `None` meaning unbounded. Port of `GenerateRangeRegex`.
pub fn generate_range_regex(start: Option<i64>, end: Option<i64>) -> String {
    let mut parts: Vec<String> = Vec::new();

    match (start, end) {
        (None, None) => return "^-?\\d+$".to_string(),
        (Some(s), None) => {
            if s <= 0 {
                if s < 0 {
                    parts.push(format!("-{}", generate_sub_range_regex(s.abs(), 1)));
                }
                parts.push("0".to_string());
                parts.push("[1-9]\\d*".to_string());
            } else {
                let start_str = s.to_string();
                let sb: Vec<char> = start_str.chars().collect();
                let len = sb.len() as i32;
                if len == 1 {
                    parts.push(make_pattern_for_digit_range(sb[0], '9', 0));
                    parts.push("[1-9]\\d*".to_string());
                } else {
                    parts.push(start_str.clone());
                    push_start_open(&mut parts, &sb, len);
                    parts.push(format!("[1-9]\\d{{{len},}}"));
                }
            }
        }
        (None, Some(e)) => {
            if e >= 0 {
                parts.push("-[1-9]\\d*".to_string());
                parts.push("0".to_string());
                if e > 0 {
                    parts.push(generate_sub_range_regex(1, e));
                }
            } else {
                let end_str = (-e).to_string();
                let eb: Vec<char> = end_str.chars().collect();
                let len = eb.len() as i32;
                if len == 1 {
                    parts.push(format!("-{}", make_pattern_for_digit_range(eb[0], '9', 0)));
                    parts.push("-[1-9]\\d*".to_string());
                } else {
                    parts.push(e.to_string());
                    for (i, &ec) in eb.iter().enumerate() {
                        if i == 0 {
                            if ec > '1' {
                                parts.push(format!(
                                    "-{}",
                                    make_pattern_for_digit_range('1', prev_char(ec), len - 1)
                                ));
                            }
                        } else {
                            let pref: String = eb[..i].iter().collect();
                            if ec > '0' {
                                parts.push(format!(
                                    "-{pref}{}",
                                    make_pattern_for_digit_range(
                                        '0',
                                        prev_char(ec),
                                        len - i as i32 - 1
                                    )
                                ));
                            }
                        }
                    }
                    parts.push(format!("-[1-9]\\d{{{len},}}"));
                }
            }
        }
        (Some(s), Some(e)) => {
            if s > e {
                return "^()$".to_string();
            }
            if s < 0 {
                let neg_end = (-1i64).min(e);
                parts.push(format!("-{}", generate_sub_range_regex(-neg_end, -s)));
            }
            if s <= 0 && e >= 0 {
                parts.push("0".to_string());
            }
            if e > 0 {
                let pos_start = 1i64.max(s);
                parts.push(generate_sub_range_regex(pos_start, e));
            }
        }
    }

    format!("^({})$", parts.join("|"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpp_examples() {
        assert_eq!(generate_range_regex(Some(12), Some(16)), r"^((1[2-6]))$");
        assert_eq!(generate_range_regex(Some(1), Some(10)), r"^(([1-9]|10))$");
        assert_eq!(
            generate_range_regex(Some(-5), Some(10)),
            r"^(-([1-5])|0|([1-9]|10))$"
        );
        assert_eq!(generate_range_regex(Some(-15), Some(-10)), r"^(-(1[0-5]))$");
        assert_eq!(generate_range_regex(None, None), r"^-?\d+$");
        assert_eq!(generate_range_regex(Some(5), None), r"^([5-9]|[1-9]\d*)$");
        assert_eq!(generate_range_regex(None, Some(0)), r"^(-[1-9]\d*|0)$");
        assert_eq!(generate_range_regex(Some(5), Some(5)), r"^((5))$");
        assert_eq!(
            generate_range_regex(Some(-10), Some(0)),
            r"^(-([1-9]|10)|0)$"
        );
        assert_eq!(generate_range_regex(Some(0), Some(10)), r"^(0|([1-9]|10))$");
    }

    #[test]
    fn matches_multi_length_examples() {
        assert_eq!(
            generate_range_regex(Some(1), Some(9999)),
            r"^(([1-9]|[1-9]\d{1}|[1-9]\d{2}|[1-9]\d{3}))$"
        );
        assert_eq!(
            generate_range_regex(Some(s_int()), None),
            r"^([5-9]|[1-9]\d*)$"
        );
    }

    fn s_int() -> i64 {
        5
    }
}
