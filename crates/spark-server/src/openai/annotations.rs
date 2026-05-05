// SPDX-License-Identifier: AGPL-3.0-only

#[allow(unused_imports)]
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::*;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    /// Model reasoning trace (from <think>...</think> tags).
    /// Only populated when enable_thinking=true. Both Cline and Roo Code
    /// check for this field. DeepSeek-originated, vLLM/LiteLLM standard.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Forward-compatible alias: mirrors `reasoning_content` for clients that
    /// use the shorter `reasoning` field name (e.g. some OpenAI SDK versions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tool_parser::ToolCall>>,
    /// OpenAI-compatible annotations (URL citations, etc.).
    /// Populated post-hoc from URLs found in `content` so web-search /
    /// retrieval clients see a familiar shape. Omitted when empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<Annotation>>,
    /// Assistant refusal message. When set, `content` should be treated
    /// as null by the client. Atlas does not currently emit refusals; the
    /// field is present so safety-aware clients stay compatible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// OpenAI `message.annotations[]` entry. Only `url_citation` is populated
/// today — the tagged variant keeps the wire format forward-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    UrlCitation { url_citation: UrlCitation },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlCitation {
    pub start_index: usize,
    pub end_index: usize,
    pub url: String,
    pub title: String,
}

/// Scan `content` for http(s) URLs and emit OpenAI-compatible
/// `url_citation` annotations.
///
/// The extractor handles three shapes:
/// - Markdown links `[title](url)` — title from the `[...]` text.
/// - Bare URLs — title is the URL itself.
/// - URLs inside fenced code blocks (triple backticks) or inline code (`` `...` ``)
///   are **skipped** — illustrative code, not citations. This prevents
///   false positives on model output like `curl https://example.com`.
///
/// Returns `None` when no URLs remain so the wire format stays identical
/// for non-web-search responses.
pub fn extract_url_annotations(content: &str) -> Option<Vec<Annotation>> {
    let mut out: Vec<Annotation> = Vec::new();
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
            out.push(Annotation::UrlCitation {
                url_citation: UrlCitation {
                    start_index: rb + 2,
                    end_index: rparen,
                    url: target.to_string(),
                    title: title.to_string(),
                },
            });
        }
        scan = rparen + 1;
    }

    // Second pass: bare URLs in regions that aren't masked AND aren't
    // already covered by a markdown link.
    let covered: Vec<(usize, usize)> = out
        .iter()
        .map(|a| match a {
            Annotation::UrlCitation {
                url_citation:
                    UrlCitation {
                        start_index,
                        end_index,
                        ..
                    },
            } => (*start_index, *end_index),
        })
        .collect();
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
        loop {
            let Some(last) = raw.chars().last() else {
                break;
            };
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
                out.push(Annotation::UrlCitation {
                    url_citation: UrlCitation {
                        start_index: start,
                        end_index: end,
                        url: raw.to_string(),
                        title: raw.to_string(),
                    },
                });
            }
        }
        i = abs_start + end_rel.max(1);
    }
    // Sort by start index so downstream consumers see annotations in
    // document order regardless of which pass emitted them.
    out.sort_by_key(|a| match a {
        Annotation::UrlCitation {
            url_citation: UrlCitation { start_index, .. },
        } => *start_index,
    });
    if out.is_empty() { None } else { Some(out) }
}

/// Return a copy of `content` where the insides of fenced code blocks
/// (```) and inline code spans (`) are replaced with ASCII spaces while
/// preserving byte offsets and UTF-8 validity. Used so the URL scan can
/// skip over code regions without rebuilding indices.
///
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
