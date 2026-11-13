// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Evaluation design after 80-routing.md section 5 item 4 (the B0 gate) and section 8.
//! Responsibility: the length and keyword features of the baseline arm, and the synthetic data of the selftest.

use std::sync::LazyLock;

use regex::Regex;

use super::Record;
use crate::route_gate::mat::Mat;
use crate::route_gate::mt::NumpyRandom;

pub const KEYWORDS: [&str; 40] = [
    "prove",
    "show that",
    "integral",
    "derivative",
    "probability",
    "expected value",
    "maximum",
    "minimum",
    "how many",
    "find all",
    "compute",
    "evaluate",
    "write a function",
    "implement",
    "return",
    "assert",
    "class ",
    "largest",
    "smallest",
    "sum of",
    "remainder",
    "real numbers",
    "triangle",
    "circle",
    "polynomial",
    "which of the following",
    " not ",
    "except",
    "calculate",
    "what is the value",
    "determine",
    "sequence",
    "matrix",
    "prime",
    "modulo",
    "percent",
    "fraction",
    "graph",
    "vector",
    "step by step",
];

const BASE_FEATURES: [&str; 12] = [
    "log_chars",
    "log_words",
    "log_lines",
    "digit_frac",
    "upper_frac",
    "log_dollar",
    "log_backslash",
    "log_longest_number",
    "n_options",
    "has_def",
    "has_code_fence",
    "question_marks",
];

/// Names of the features, in column order (`kw:` plus the keyword, stripped).
pub fn feature_names() -> Vec<String> {
    let mut v: Vec<String> = BASE_FEATURES.iter().map(|s| (*s).to_string()).collect();
    v.extend(KEYWORDS.iter().map(|k| format!("kw:{}", k.trim())));
    v
}

static DIGITS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").expect("static regex"));
static OPTION_LINES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[A-J]\. ").expect("static regex"));

/// Length and keyword features of the prompt text alone. Deliberately generic, and it does see the answer-format
/// instructions (so it can tell a multiple-choice prompt from a code prompt): a router in front of Atlas sees them too.
pub fn text_features(prompt: &str) -> Vec<f64> {
    let low = prompt.to_lowercase();
    let chars = prompt.chars().count();
    let n = chars.max(1) as f64;
    let longest = DIGITS
        .find_iter(prompt)
        .map(|m| m.as_str().chars().count())
        .max()
        .unwrap_or(0);
    let count = |c: char| prompt.chars().filter(|&x| x == c).count() as f64;
    let mut f = vec![
        (chars as f64).ln_1p(),
        (prompt.split_whitespace().count() as f64).ln_1p(),
        (count('\n') + 1.0).ln_1p(),
        prompt.chars().filter(char::is_ascii_digit).count() as f64 / n,
        prompt.chars().filter(|c| c.is_uppercase()).count() as f64 / n,
        count('$').ln_1p(),
        count('\\').ln_1p(),
        (longest as f64).ln_1p(),
        OPTION_LINES.find_iter(prompt).count() as f64,
        f64::from(u8::from(low.contains("def "))),
        f64::from(u8::from(prompt.contains("```"))),
        count('?'),
    ];
    f.extend(
        KEYWORDS
            .iter()
            .map(|k| f64::from(u8::from(low.contains(k)))),
    );
    f
}

/// Records and readouts where failure depends on a hidden direction of the readout (plus dataset effects).
pub fn make_synthetic(n: usize, d: usize, seed: u64, signal: f64) -> (Vec<Record>, Mat) {
    let mut rng = NumpyRandom::new(seed);
    let mut direction = rng.randn_vec(d);
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    direction.iter_mut().for_each(|v| *v /= norm);
    let names = ["gsm8k", "math", "mmlu"];
    let sources: Vec<&str> = rng
        .choice_indices(3, n)
        .into_iter()
        .map(|i| names[i])
        .collect();
    let base = |s: &str| match s {
        "gsm8k" => -0.6,
        "math" => 0.6,
        _ => 0.0,
    };
    let noise = rng.randn_vec(n);
    let latent: Vec<f64> = sources
        .iter()
        .zip(&noise)
        .map(|(s, z)| base(s) + z * 0.8)
        .collect();
    let noise2 = rng.randn_vec(n);
    let fails: Vec<bool> = latent
        .iter()
        .zip(&noise2)
        .map(|(l, z)| l + z * 0.4 > 0.0)
        .collect();
    let mut x = rng.randn_vec(n * d);
    for i in 0..n {
        for j in 0..d {
            x[i * d + j] += latent[i] * signal * direction[j];
        }
    }
    let recs = (0..n)
        .map(|i| {
            let reps = (5.0 + if latent[i] > 0.0 { 4.0 } else { 0.0 } + rng.randint(0, 12) as f64)
                as usize;
            let tail = if latent[i] > 1.0 { "prove" } else { "?" };
            Record {
                id: format!("s{i}"),
                source: sources[i].to_string(),
                passed: !fails[i],
                prompt: format!("Solve {}{tail}", "x ".repeat(reps)),
            }
        })
        .collect();
    (
        recs,
        Mat {
            rows: n,
            cols: d,
            data: x,
        },
    )
}
