// SPDX-License-Identifier: AGPL-3.0-only

//! Citation extraction from assistant content — provider-neutral.
//!
//! Two extractors work together:
//!
//! * [`extract_url_citations`] finds bare + markdown-link URLs (skipping
//!   code spans so `curl https://…` examples don't become citations).
//! * [`extract`] recognizes three common "model-emitted" structured
//!   citation patterns (parsers live in [`crate::citation_structured`]):
//!
//!   1. Markdown footnotes
//!      ```text
//!      See the source[^1] for details.
//!      ...
//!      [^1]: https://example.com/source The title text
//!      ```
//!      Emits a citation at the `[^1]` reference site with the URL
//!      from the definition and the title text as `title`.
//!
//!   2. Numeric bracket refs
//!      ```text
//!      See [1] for details.
//!      ...
//!      [1] https://example.com/source
//!      ```
//!      Same shape, without the `^` sigil.
//!
//!   3. Fenced sources sections
//!      ```text
//!      Sources:
//!      - https://a.example.com
//!      - https://b.example.com
//!      ```
//!      Each bullet → one citation at the bullet's URL span.
//!
//! The parser is conservative: it only fires when the definition /
//! bullet contains an http(s) URL. [`merged_citations`] runs both and
//! dedupes on URL. Output is the neutral [`Citation`] struct — each API
//! surface converts it to its own wire annotation shape (e.g.
//! `openai::Annotation`).
//!
//! This is still post-hoc parsing — Atlas has no web-search tool, so
//! "model-sourced" here means "the model emitted a structured citation
//! pattern we recognize". The shape clients receive is identical to
//! what a real web-search backend would produce.

use crate::citation_structured::{
    footnote_citations, numeric_ref_citations, sources_block_citations,
};

/// A URL citation found in assistant text, with byte offsets into the
/// content it was extracted from. Provider-neutral.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Citation {
    pub start_index: usize,
    pub end_index: usize,
    pub url: String,
    pub title: String,
}

/// Convenience: run the bare-URL extractor + the structured citation
/// extractor and return the deduped citations. `None` when nothing
/// matched, so wire fields can be serde-skipped.
pub fn merged_citations(content: &str) -> Option<Vec<Citation>> {
    let bare = extract_url_citations(content);
    let structured = extract(content);
    let merged = merge_dedupe(bare, structured);
    if merged.is_empty() {
        None
    } else {
        Some(merged)
    }
}

/// Find structured citations in `content` and return one entry per
/// reference site. Returns an empty vec when nothing matched.
pub fn extract(content: &str) -> Vec<Citation> {
    let mut out = Vec::new();
    out.extend(footnote_citations(content));
    out.extend(numeric_ref_citations(content));
    out.extend(sources_block_citations(content));
    // Sort by start position so consumers see document order.
    out.sort_by_key(|c| c.start_index);
    out
}

/// Scan `content` for http(s) URLs and emit a citation per hit.
///
/// The extractor handles three shapes:
/// - Markdown links `[title](url)` — title from the `[...]` text.
/// - Bare URLs — title is the URL itself.
/// - URLs inside fenced code blocks (triple backticks) or inline code
///   (`` `...` ``) are **skipped** — illustrative code, not citations.
///   This prevents false positives on model output like
///   `curl https://example.com`.
pub fn extract_url_citations(content: &str) -> Vec<Citation> {
    let mut out: Vec<Citation> = Vec::new();
    let masked = mask_code_spans(content);
    // First pass: markdown links. We scan the masked copy so the
    // start/end indices we record match the original `content` exactly.
    let mut scan = 0usize;
    while scan < masked.len() {
        let rest = &masked[scan..];
        let Some(lb_rel) = rest.find('[') else { break };
        let lb = scan + lb_rel;
        // Find the matching `]` then immediate `(`.
        let after_lb = &masked[lb + 1..];
        let Some(rb_rel) = after_lb.find(']') else {
            scan = lb + 1;
            continue;
        };
        let rb = lb + 1 + rb_rel;
        if masked.as_bytes().get(rb + 1) != Some(&b'(') {
            scan = rb + 1;
            continue;
        }
        // Find the matching close paren that respects nested `()` pairs
        // inside the URL — Wikipedia and other URLs commonly contain
        // parentheses (e.g. `https://en.wikipedia.org/wiki/Foo_(bar)`).
        // The bare-find shortcut would terminate at the first `)`,
        // truncating the URL.
        let after_paren = &masked[rb + 2..];
        let Some(rparen_rel) = balanced_paren_close(after_paren) else {
            scan = rb + 2;
            continue;
        };
        let rparen = rb + 2 + rparen_rel;
        let title = &content[lb + 1..rb];
        let target = content[rb + 2..rparen].trim();
        if (target.starts_with("http://") || target.starts_with("https://"))
            && target.len() > "https://".len()
        {
            out.push(Citation {
                start_index: rb + 2,
                end_index: rparen,
                url: target.to_string(),
                title: title.to_string(),
            });
        }
        scan = rparen + 1;
    }

    // Second pass: bare URLs in regions that aren't masked AND aren't
    // already covered by a markdown link.
    let covered: Vec<(usize, usize)> = out.iter().map(|c| (c.start_index, c.end_index)).collect();
    let mut i = 0usize;
    while i < masked.len() {
        let rest = &masked[i..];
        let Some(off) = rest.find("http") else { break };
        let abs_start = i + off;
        let tail_masked = &masked[abs_start..];
        let is_url = tail_masked.starts_with("http://") || tail_masked.starts_with("https://");
        if !is_url {
            i = abs_start + 4;
            continue;
        }
        let tail = &content[abs_start..];
        let end_rel = tail
            .find(|c: char| {
                c.is_whitespace() || matches!(c, ']' | '}' | '"' | '<' | '>' | '`' | '\\')
            })
            .unwrap_or(tail.len());
        let mut raw = &tail[..end_rel];
        // Strip trailing sentence punctuation and unmatched close-parens /
        // markdown emphasis markers. Parens match pairs so URLs like
        // Wikipedia's `https://en.wikipedia.org/wiki/Foo_(bar)` survive.
        while let Some(last) = raw.chars().last() {
            let strip = match last {
                '.' | ',' | ';' | ':' | '!' | '?' | '*' | '_' => true,
                ')' => {
                    let opens = raw.matches('(').count();
                    let closes = raw.matches(')').count();
                    closes > opens
                }
                _ => false,
            };
            if !strip {
                break;
            }
            raw = &raw[..raw.len() - last.len_utf8()];
        }
        if raw.len() > "https://".len() {
            let start = abs_start;
            let end = abs_start + raw.len();
            let overlaps = covered.iter().any(|(s, e)| start < *e && end > *s);
            if !overlaps {
                out.push(Citation {
                    start_index: start,
                    end_index: end,
                    url: raw.to_string(),
                    title: raw.to_string(),
                });
            }
        }
        i = abs_start + end_rel.max(1);
    }
    // Sort by start index so downstream consumers see citations in
    // document order regardless of which pass emitted them.
    out.sort_by_key(|c| c.start_index);
    out
}

/// Merge two citation lists, deduping by URL (keeping the first hit
/// in document order). Used to combine the bare-URL extractor's output
/// with structured citations without emitting the same URL twice.
pub fn merge_dedupe(mut primary: Vec<Citation>, secondary: Vec<Citation>) -> Vec<Citation> {
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    for c in &primary {
        seen.insert(c.url.clone());
    }
    for c in secondary {
        if seen.insert(c.url.clone()) {
            primary.push(c);
        }
    }
    primary.sort_by_key(|c| c.start_index);
    primary
}

/// Return the byte offset of the `)` that balances the implicit `(`
/// before the slice (i.e. the URL's matching close), or None if no
/// balanced close exists. Handles nested `()` pairs that appear in
/// real URLs (Wikipedia article slugs, GitHub anchors, etc).
fn balanced_paren_close(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// Return a copy of `content` where the insides of fenced code blocks
/// (```) and inline code spans (`) are replaced with ASCII spaces while
/// preserving byte offsets and UTF-8 validity. Used so the URL scan can
/// skip over code regions without rebuilding indices.
///
/// We walk char-by-char to keep multi-byte characters intact: each non-
/// newline char inside a code region becomes N ASCII spaces where N is
/// the char's UTF-8 byte length. Newlines are kept so line-based parsers
/// still see the structure.
fn mask_code_spans(content: &str) -> String {
    let bytes = content.as_bytes();
    let mut out: Vec<u8> = bytes.to_vec();

    // Blank bytes [start, end) by char: non-newline chars → ASCII spaces
    // of the same byte length, newlines preserved. Char boundaries taken
    // from the original content (which is guaranteed UTF-8).
    fn blank(out: &mut [u8], content: &str, start: usize, end: usize) {
        let region = &content[start..end];
        let mut cursor = start;
        for ch in region.chars() {
            let len = ch.len_utf8();
            if ch != '\n' {
                for b in out.iter_mut().skip(cursor).take(len) {
                    *b = b' ';
                }
            }
            cursor += len;
        }
    }

    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"```") {
            let after = i + 3;
            let rest = &content[after..];
            let end = match rest.find("```") {
                Some(r) => after + r + 3,
                None => bytes.len(),
            };
            blank(&mut out, content, i, end);
            i = end;
            continue;
        }
        if bytes[i] == b'`' {
            let after = i + 1;
            let rest = &content[after..];
            let end = match rest.find('`') {
                Some(r) => after + r + 1,
                None => bytes.len(),
            };
            blank(&mut out, content, i, end);
            i = end;
            continue;
        }
        // Step one whole UTF-8 codepoint.
        let step = match bytes[i] {
            0x00..=0x7f => 1,
            0xc0..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf7 => 4,
            _ => 1,
        };
        i += step;
    }
    String::from_utf8(out).expect("mask preserves UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_citations_returns_empty() {
        let got = extract("plain text with no citations");
        assert!(got.is_empty());
    }

    #[test]
    fn merge_dedupe_by_url() {
        let a = vec![Citation {
            start_index: 0,
            end_index: 20,
            url: "https://example.com".into(),
            title: "first".into(),
        }];
        let b = vec![Citation {
            start_index: 50,
            end_index: 70,
            url: "https://example.com".into(),
            title: "second".into(),
        }];
        let merged = merge_dedupe(a, b);
        assert_eq!(merged.len(), 1);
    }

    // ── bare/markdown URL scanner (moved from openai/annotations.rs) ──

    #[test]
    fn bare_url_extracted() {
        let got = extract_url_citations("see https://example.com/foo for more");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].url, "https://example.com/foo");
        assert_eq!(got[0].title, "https://example.com/foo");
    }

    #[test]
    fn trailing_punctuation_stripped() {
        let got = extract_url_citations("go to https://example.com.");
        assert_eq!(got[0].url, "https://example.com");
    }

    #[test]
    fn wikipedia_parens_survive() {
        let got = extract_url_citations("see https://en.wikipedia.org/wiki/Foo_(bar) now");
        assert_eq!(got[0].url, "https://en.wikipedia.org/wiki/Foo_(bar)");
    }

    #[test]
    fn markdown_link_title_used() {
        let got = extract_url_citations("read [the docs](https://example.com/api) today");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].url, "https://example.com/api");
        assert_eq!(got[0].title, "the docs");
    }

    #[test]
    fn markdown_link_with_parens_in_url() {
        let got =
            extract_url_citations("see [Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar)) here");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].url, "https://en.wikipedia.org/wiki/Foo_(bar)");
    }

    #[test]
    fn code_spans_skipped() {
        assert!(extract_url_citations("run `curl https://example.com` locally").is_empty());
        assert!(extract_url_citations("```\ncurl https://example.com\n```\nno cite").is_empty());
    }

    #[test]
    fn non_http_schemes_ignored() {
        assert!(extract_url_citations("ftp://example.com not a citation").is_empty());
    }

    #[test]
    fn empty_and_plain_content() {
        assert!(extract_url_citations("").is_empty());
        assert!(extract_url_citations("no URLs here").is_empty());
    }

    #[test]
    fn query_and_fragment_kept() {
        let got = extract_url_citations("see https://example.com/p?q=1&r=2#frag here");
        assert_eq!(got[0].url, "https://example.com/p?q=1&r=2#frag");
    }

    #[test]
    fn merged_citations_combines_and_dedupes() {
        let input = "Intro https://a.example.com text.\n\nSources:\n- https://a.example.com\n- https://b.example.com\n";
        let got = merged_citations(input).unwrap();
        let mut u: Vec<&str> = got.iter().map(|c| c.url.as_str()).collect();
        u.dedup();
        assert_eq!(u.len(), 2);
    }
}
