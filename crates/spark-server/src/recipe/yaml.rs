// SPDX-License-Identifier: AGPL-3.0-only

//! A deliberately small YAML reader for `atlas-recipes`.
//!
//! Atlas has no YAML dependency and this is not a reason to add one: the
//! recipes use six constructs, and a general parser would accept a great deal
//! more than the format actually is.
//!
//! **The failure mode here is not a crash, it is a wrong serve config.** A key
//! silently skipped because its shape was unfamiliar changes what gets served
//! and surfaces later as a mysterious coherence or throughput regression. So
//! this reader **errors on anything it does not recognise** rather than
//! skipping it. Being unable to read a recipe is a good outcome; reading three
//! quarters of one is not.
//!
//! Supported, because these are what the 25 real recipes contain:
//! `key: scalar`, nested maps by two-space indent, `key: |` literal blocks,
//! `- item` sequences, `{}` empty maps, `#` comments and blank lines.

use anyhow::{Result, bail};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Yaml {
    Scalar(String),
    List(Vec<Yaml>),
    Map(BTreeMap<String, Yaml>),
}

impl Yaml {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Yaml::Scalar(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&BTreeMap<String, Yaml>> {
        match self {
            Yaml::Map(m) => Some(m),
            _ => None,
        }
    }
}

/// One significant line: its indent and its content, comments already gone.
struct Line {
    no: usize,
    indent: usize,
    text: String,
}

/// Parse a whole document. The top level must be a map — every recipe is one.
pub fn parse(input: &str) -> Result<Yaml> {
    let lines = significant(input)?;
    let (value, end) = parse_block(&lines, 0, 0)?;
    if end != lines.len() {
        let l = &lines[end];
        bail!("line {}: unexpected indentation at {:?}", l.no, l.text);
    }
    match value {
        Yaml::Map(_) => Ok(value),
        _ => bail!("the document must be a mapping"),
    }
}

/// Strip comments and blanks, but never inside a literal block — a `#` there is
/// content, and the recipes' `command:` blocks are full of them.
fn significant(input: &str) -> Result<Vec<Line>> {
    let mut out = Vec::new();
    let mut raw = input.lines().enumerate().peekable();
    while let Some((i, line)) = raw.next() {
        let no = i + 1;
        let indent = line.len() - line.trim_start().len();
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let text = strip_comment(trimmed);
        if text.is_empty() {
            continue;
        }
        let is_literal = text.ends_with(": |") || text.ends_with(": |-");
        if !is_literal {
            out.push(Line { no, indent, text });
            continue;
        }
        // Consume the block verbatim: every following line indented deeper than
        // the key, comments and blanks included.
        let mut body: Vec<String> = Vec::new();
        let mut block_indent = None;
        while let Some((_, peek)) = raw.peek() {
            let peek_indent = peek.len() - peek.trim_start().len();
            if !peek.trim().is_empty() && peek_indent <= indent {
                break;
            }
            let (_, l) = raw.next().expect("peeked");
            if l.trim().is_empty() {
                body.push(String::new());
                continue;
            }
            let base = *block_indent.get_or_insert(peek_indent);
            body.push(l.chars().skip(base).collect());
        }
        while body.last().is_some_and(|l| l.is_empty()) {
            body.pop();
        }
        // Re-emit the key with its folded value, marked so parse_block knows.
        let key = text.trim_end_matches(['|', '-', ' ']).trim_end_matches(':');
        out.push(Line {
            no,
            indent,
            text: format!("{key}:\u{0}{}", body.join("\n")),
        });
    }
    Ok(out)
}

/// Remove a trailing `# comment`, respecting double quotes.
fn strip_comment(s: &str) -> String {
    let mut in_quotes = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes && (i == 0 || s.as_bytes()[i - 1] == b' ') => {
                return s[..i].trim_end().to_string();
            }
            _ => {}
        }
    }
    s.trim_end().to_string()
}

/// Parse the run of lines at `indent`, returning the value and the next index.
fn parse_block(lines: &[Line], start: usize, indent: usize) -> Result<(Yaml, usize)> {
    if lines.get(start).is_some_and(|l| l.text.starts_with("- ")) {
        return parse_list(lines, start, indent);
    }
    let mut map = BTreeMap::new();
    let mut i = start;
    while i < lines.len() {
        let line = &lines[i];
        if line.indent < indent {
            break;
        }
        if line.indent > indent {
            bail!(
                "line {}: unexpected indentation at {:?}",
                line.no,
                line.text
            );
        }
        let Some((key, rest)) = line.text.split_once(':') else {
            bail!(
                "line {}: expected `key: value`, found {:?}",
                line.no,
                line.text
            );
        };
        let key = key.trim().to_string();
        if key.is_empty() {
            bail!("line {}: empty key", line.no);
        }
        // A folded literal block, marked by `significant`.
        if let Some(body) = rest.strip_prefix('\u{0}') {
            map.insert(key, Yaml::Scalar(body.to_string()));
            i += 1;
            continue;
        }
        let rest = rest.trim();
        if rest == "{}" {
            map.insert(key, Yaml::Map(BTreeMap::new()));
            i += 1;
            continue;
        }
        if !rest.is_empty() {
            map.insert(key, Yaml::Scalar(unquote(rest)));
            i += 1;
            continue;
        }
        // Nested block: whatever is indented deeper.
        let next_indent = lines.get(i + 1).map(|l| l.indent).unwrap_or(0);
        if next_indent <= indent {
            bail!(
                "line {}: {key:?} has no value and no indented block",
                line.no
            );
        }
        let (value, next) = parse_block(lines, i + 1, next_indent)?;
        map.insert(key, value);
        i = next;
    }
    Ok((Yaml::Map(map), i))
}

fn parse_list(lines: &[Line], start: usize, indent: usize) -> Result<(Yaml, usize)> {
    let mut items = Vec::new();
    let mut i = start;
    while i < lines.len() {
        let line = &lines[i];
        if line.indent < indent {
            break;
        }
        let Some(item) = line.text.strip_prefix("- ") else {
            bail!(
                "line {}: expected a `- ` item, found {:?}",
                line.no,
                line.text
            );
        };
        items.push(Yaml::Scalar(unquote(item.trim())));
        i += 1;
    }
    Ok((Yaml::List(items), i))
}

/// `"2"` and `'x'` become `2` and `x`. Recipes quote to force a string type,
/// which this reader does not model — every scalar is text until asked for.
fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    if b.len() >= 2
        && (b[0] == b'"' && b[b.len() - 1] == b'"' || b[0] == b'\'' && b[b.len() - 1] == b'\'')
    {
        return s[1..s.len() - 1].to_string();
    }
    s.to_string()
}

#[cfg(test)]
#[path = "yaml_tests.rs"]
mod tests;
