// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. `\boxed{}` extraction and MATH answer normalization, written from scratch for this lane.
//! Responsibility: the content of the last `\boxed{...}`, a light normalization of math answers, and equality.

use std::sync::LazyLock;

use regex::Regex;

use super::Grade;
use super::numeric::parse_number;

static TEXT_WRAP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\(?:text|mathrm|mbox)\{([^{}]*)\}").expect("static regex"));
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("static regex"));
static FRAC_DD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\frac(\d)(\d)").expect("static regex"));
static FRAC_D_BRACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\frac(\d)\{").expect("static regex"));
static SQRT_D: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\sqrt(\d)").expect("static regex"));
static VAR_EQ: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A[a-zA-Z]=").expect("static regex"));
static TRAILING_ZEROS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A(\d+)\.0+\z").expect("static regex"));
static COMMA_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A-?\d{1,3}(,\d{3})+(\.\d+)?\z").expect("static regex"));

/// Content of the last `\boxed{...}` (brace matched), or `None`.
pub fn last_boxed(text: &str) -> Option<String> {
    let i = text.rfind("\\boxed")?;
    let j = i + text[i..].find('{')?;
    let mut depth = 0i64;
    for (off, b) in text.as_bytes()[j..].iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[j + 1..j + off].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Strips formatting that does not change a math answer (units, `\left`, spacing, `x =`, thousands commas).
pub fn normalize_math(s: &str) -> String {
    let mut s = TEXT_WRAP.replace_all(s.trim(), "${1}").into_owned();
    for a in [
        "\\!",
        "\\,",
        "\\;",
        "\\:",
        "\\ ",
        "\\left",
        "\\right",
        "^\\circ",
        "^{\\circ}",
        "\\%",
        "%",
        "\\$",
        "$",
    ] {
        s = s.replace(a, "");
    }
    s = s.replace("dfrac", "frac").replace("tfrac", "frac");
    s = SPACES.replace_all(&s, "").trim_end_matches('.').to_string();
    s = FRAC_DD.replace_all(&s, "\\frac{${1}}{${2}}").into_owned();
    s = FRAC_D_BRACE.replace_all(&s, "\\frac{${1}}{").into_owned();
    s = SQRT_D.replace_all(&s, "\\sqrt{${1}}").into_owned();
    if VAR_EQ.is_match(&s)
        && let Some((_, rest)) = s.clone().split_once('=')
    {
        s = rest.to_string();
    }
    s = TRAILING_ZEROS.replace_all(&s, "${1}").into_owned();
    if COMMA_NUMBER.is_match(&s) {
        s = s.replace(',', "");
    }
    s
}

/// Equal after normalization, or numerically equal within a relative `1e-6`.
pub fn math_equal(a: &str, b: &str) -> bool {
    let (na, nb) = (normalize_math(a), normalize_math(b));
    if na == nb {
        return true;
    }
    match (parse_number(&na), parse_number(&nb)) {
        (Some(fa), Some(fb)) => (fa - fb).abs() <= 1e-6 * fb.abs().max(1.0),
        _ => false,
    }
}

/// Grades a reply whose answer is the last `\boxed{}` against the gold string.
pub fn grade_boxed(text: &str, gold: &str) -> Grade {
    match last_boxed(text) {
        None => Grade::no_answer(),
        Some(got) => Grade::new(math_equal(&got, gold), Some(got), "ok"),
    }
}
