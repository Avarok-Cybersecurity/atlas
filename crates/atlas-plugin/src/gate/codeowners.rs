// SPDX-License-Identifier: AGPL-3.0-only

//! Who owns a changed path, per `.github/CODEOWNERS`.
//!
//! # Scope, and why it is narrow on purpose
//!
//! GitHub's CODEOWNERS matching is gitignore-flavoured and has corners. This
//! implements the subset the file actually uses — leading `/`, trailing `/`,
//! and `*` within a segment — and treats anything else, `**` included, as a
//! **non-match**.
//!
//! That direction is deliberate but it is NOT the usual fail-closed choice, so
//! it is worth being plain about: an unrecognised pattern means someone is not
//! mentioned on a PR they own. It does not let a change through a gate — the
//! mentions are advisory, and nothing here blocks a merge. Guessing an owner
//! wrongly is worse than staying quiet, because a wrong mention teaches people
//! to ignore the bot.
//!
//! `codeowners_tests.rs` pins the real file: every pattern in
//! `.github/CODEOWNERS` must be one this understands, so an unsupported pattern
//! fails a unit test instead of silently un-mentioning its owner.

use std::path::Path;

/// One CODEOWNERS rule, in file order.
#[derive(Debug, Clone)]
pub struct Rule {
    pub pattern: String,
    pub owners: Vec<String>,
}

/// Parse `.github/CODEOWNERS`. Missing file yields no rules.
pub fn load(root: &Path) -> Vec<Rule> {
    let Ok(text) = std::fs::read_to_string(root.join(".github/CODEOWNERS")) else {
        return Vec::new();
    };
    parse(&text)
}

pub fn parse(text: &str) -> Vec<Rule> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .filter_map(|l| {
            let mut parts = l.split_whitespace();
            let pattern = parts.next()?.to_string();
            let owners: Vec<String> = parts
                .filter(|o| o.starts_with('@') || o.contains('@'))
                .map(str::to_string)
                .collect();
            // A pattern with no owners is CODEOWNERS' way of REMOVING ownership
            // for that path. Kept as a rule with an empty owner list so
            // last-match-wins can honour it, rather than dropped.
            Some(Rule { pattern, owners })
        })
        .collect()
}

/// Owners of `path`, or empty when nothing matches.
///
/// Last matching rule wins, as GitHub does — so a specific rule placed after a
/// broad one overrides it.
pub fn owners_of<'a>(rules: &'a [Rule], path: &str) -> &'a [String] {
    rules
        .iter()
        .rfind(|r| matches(&r.pattern, path))
        .map(|r| r.owners.as_slice())
        .unwrap_or(&[])
}

/// Every owner mentioned for any of `paths`, deduplicated and sorted.
pub fn owners_for_paths(rules: &[Rule], paths: &[String]) -> Vec<String> {
    let mut out: Vec<String> = paths
        .iter()
        .flat_map(|p| owners_of(rules, p).iter().cloned())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Whether this module understands `pattern` well enough to answer for it.
///
/// Separator-spanning `**`, `?`, character classes and escapes are left out:
/// no rule in `.github/CODEOWNERS` uses them, and partially implementing those
/// constructs would be worse than reporting them unsupported.
/// `every_pattern_in_the_real_file_is_supported` fails if one appears, which
/// is the signal to implement it rather than to let an owner go unmentioned.
pub fn is_supported(pattern: &str) -> bool {
    !pattern.contains("**")
        && !pattern.contains('?')
        && !pattern.contains('[')
        && !pattern.contains(']')
        && !pattern.contains('\\')
}

/// Whether a CODEOWNERS pattern matches a repo-relative path.
///
/// gitignore semantics, for the subset in [`is_supported`]:
/// - a trailing `/` means the directory and everything under it;
/// - a pattern containing a `/` anywhere but the end is anchored to the root;
/// - a pattern with no `/` matches that basename at ANY depth, which is why
///   `Cargo.toml` covers `crates/spark-server/Cargo.toml`.
fn matches(pattern: &str, path: &str) -> bool {
    if !is_supported(pattern) {
        return false;
    }
    let (body, dir_only) = match pattern.strip_suffix('/') {
        Some(p) => (p, true),
        None => (pattern, false),
    };
    let anchored = pattern.starts_with('/') || body.contains('/');
    let body = body.strip_prefix('/').unwrap_or(body);
    if body.is_empty() {
        return false;
    }

    let under = |base: &str| path.starts_with(&format!("{base}/"));

    if anchored {
        // A directory pattern covers its subtree; a file pattern is exact
        // (modulo `*` within a segment).
        return glob_segment(body, path) || under(body);
    }
    // Unanchored: the basename at any depth, or — for a directory pattern — any
    // directory of that name at any depth.
    if path.rsplit('/').next().is_some_and(|n| star(body, n)) {
        return true;
    }
    if dir_only || !body.contains('*') {
        return path
            .split('/')
            .any(|segment| star(body, segment) && path != segment);
    }
    false
}

/// `*` matches within one path segment only.
fn glob_segment(pattern: &str, path: &str) -> bool {
    let pat: Vec<&str> = pattern.split('/').collect();
    let seg: Vec<&str> = path.split('/').collect();
    if pat.len() != seg.len() {
        return false;
    }
    pat.iter().zip(&seg).all(|(p, s)| star(p, s))
}

/// `*` within a single segment.
fn star(pattern: &str, s: &str) -> bool {
    let text: Vec<char> = s.chars().collect();
    let mut matched = vec![false; text.len() + 1];
    matched[0] = true;
    for token in pattern.chars() {
        let mut next = vec![false; text.len() + 1];
        if token == '*' {
            next[0] = matched[0];
            for index in 1..=text.len() {
                next[index] = matched[index] || next[index - 1];
            }
        } else {
            for index in 1..=text.len() {
                next[index] = matched[index - 1] && token == text[index - 1];
            }
        }
        matched = next;
    }
    matched[text.len()]
}

#[cfg(test)]
#[path = "codeowners_tests.rs"]
mod codeowners_tests;
