// SPDX-License-Identifier: AGPL-3.0-only

//! Structured citation parser for assistant content.
//!
//! Complements [`crate::openai::extract_url_annotations`] (which finds
//! bare + markdown-link URLs) by recognizing three common
//! "model-emitted" citation patterns that would otherwise miss the
//! structured `url_citation` shape OpenAI clients expect:
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
//! bullet contains an http(s) URL. Output is a `Vec<Annotation>` that
//! the handler merges with the bare-URL extractor's output (dedupe on
//! URL to avoid double-counting).
//!
//! This is still post-hoc parsing — Atlas has no web-search tool, so
//! "model-sourced" here means "the model emitted a structured citation
//! pattern we recognize". The shape clients receive is identical to
//! what a real web-search backend would produce.

use crate::openai::{Annotation, UrlCitation};

/// Convenience: run the bare-URL extractor + the structured citation
/// extractor and return the deduped annotations. `None` when nothing
/// matched, so the field gets serde-skipped on the wire.
pub fn merged_annotations(content: &str) -> Option<Vec<Annotation>> {
    let bare = crate::openai::extract_url_annotations(content).unwrap_or_default();
    let structured = extract(content);
    let merged = merge_dedupe(bare, structured);
    if merged.is_empty() {
        None
    } else {
        Some(merged)
    }
}

/// Find citations in `content` and return an annotation for each
/// reference site. Returns an empty vec when nothing matched.
pub fn extract(content: &str) -> Vec<Annotation> {
    let mut out = Vec::new();
    out.extend(footnote_citations(content));
    out.extend(numeric_ref_citations(content));
    out.extend(sources_block_citations(content));
    // Sort by start position so consumers see document order.
    out.sort_by_key(|a| match a {
        Annotation::UrlCitation { url_citation } => url_citation.start_index,
    });
    out
}

/// Parse markdown footnote references: every `[^label]` in text paired
/// with a `[^label]: url [title]` definition.
fn footnote_citations(content: &str) -> Vec<Annotation> {
    let defs = collect_footnote_defs(content);
    if defs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut i = 0;
    let bytes = content.as_bytes();
    while i < bytes.len() {
        if let Some(rel) = content[i..].find("[^") {
            let start = i + rel;
            // Skip footnote definitions (`[^x]:` at line start).
            let after_br = &content[start..];
            if let Some(close_rel) = after_br.find(']') {
                let close = start + close_rel;
                // Reject if this is a definition line (next char is ':').
                if content.as_bytes().get(close + 1) == Some(&b':') {
                    i = close + 1;
                    continue;
                }
                let label = &content[start + 2..close];
                if let Some((url, title)) = defs.get(label) {
                    out.push(Annotation::UrlCitation {
                        url_citation: UrlCitation {
                            start_index: start,
                            end_index: close + 1,
                            url: url.clone(),
                            title: title.clone().unwrap_or_else(|| label.to_string()),
                        },
                    });
                }
                i = close + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    out
}

/// Walk each line; return `{label: (url, optional_title)}` for lines
/// matching `[^label]: url [title]`.
fn collect_footnote_defs(
    content: &str,
) -> std::collections::HashMap<String, (String, Option<String>)> {
    let mut map = std::collections::HashMap::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("[^") else {
            continue;
        };
        let Some(close_rel) = rest.find(']') else {
            continue;
        };
        let label = &rest[..close_rel];
        let after = &rest[close_rel + 1..];
        let Some(body) = after.strip_prefix(':') else {
            continue;
        };
        let body = body.trim_start();
        let Some((url, rest)) = split_url(body) else {
            continue;
        };
        let title = rest.trim();
        let title_opt = if title.is_empty() {
            None
        } else {
            Some(title.to_string())
        };
        map.insert(label.to_string(), (url, title_opt));
    }
    map
}

/// Parse `[N]` refs paired with a `[N] url` definition line.
fn numeric_ref_citations(content: &str) -> Vec<Annotation> {
    let defs = collect_numeric_defs(content);
    if defs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < content.len() {
        let Some(rel) = content[i..].find('[') else {
            break;
        };
        let start = i + rel;
        let after = &content[start + 1..];
        let Some(close_rel) = after.find(']') else {
            break;
        };
        let label = &after[..close_rel];
        if label.bytes().all(|b| b.is_ascii_digit()) && !label.is_empty() {
            let close = start + 1 + close_rel;
            // Skip definition sites.
            let at_line_start = start == 0 || content.as_bytes().get(start - 1) == Some(&b'\n');
            let next_is_url_hint = {
                let next = content.get(close + 1..).unwrap_or("");
                next.trim_start().starts_with("http")
            };
            let is_definition = at_line_start && next_is_url_hint;
            if !is_definition && let Some((url, title)) = defs.get(label) {
                out.push(Annotation::UrlCitation {
                    url_citation: UrlCitation {
                        start_index: start,
                        end_index: close + 1,
                        url: url.clone(),
                        title: title.clone().unwrap_or_else(|| label.to_string()),
                    },
                });
            }
            i = close + 1;
        } else {
            i = start + 1;
        }
    }
    out
}

fn collect_numeric_defs(
    content: &str,
) -> std::collections::HashMap<String, (String, Option<String>)> {
    let mut map = std::collections::HashMap::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix('[') else {
            continue;
        };
        let Some(close_rel) = rest.find(']') else {
            continue;
        };
        let label = &rest[..close_rel];
        if label.is_empty() || !label.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let mut after = &rest[close_rel + 1..];
        if let Some(stripped) = after.strip_prefix(':') {
            after = stripped;
        }
        let body = after.trim_start();
        let Some((url, rest)) = split_url(body) else {
            continue;
        };
        let title = rest.trim();
        let title_opt = if title.is_empty() {
            None
        } else {
            Some(title.to_string())
        };
        map.insert(label.to_string(), (url, title_opt));
    }
    map
}

/// Find `Sources:` or `References:` heading and emit a citation for
/// each bullet line containing an http(s) URL until a blank line or a
/// non-bullet line breaks the section.
fn sources_block_citations(content: &str) -> Vec<Annotation> {
    let lower = content.to_ascii_lowercase();
    let markers = ["sources:", "references:", "citations:"];
    let mut out = Vec::new();
    for marker in &markers {
        let mut search_from = 0usize;
        while let Some(rel) = lower[search_from..].find(marker) {
            let heading_start = search_from + rel;
            // Must be at a line start (previous char is \n or BOF).
            let at_line_start =
                heading_start == 0 || content.as_bytes().get(heading_start - 1) == Some(&b'\n');
            if !at_line_start {
                search_from = heading_start + marker.len();
                continue;
            }
            let after = heading_start + marker.len();
            let rest = &content[after..];
            // Walk lines after the heading. Skip the (typically empty)
            // line fragment between the heading's colon and the first
            // newline so the first real line is the first bullet.
            let mut cursor = after;
            let mut seen_content = false;
            for line in rest.lines() {
                let line_start = cursor;
                cursor += line.len();
                if content.as_bytes().get(cursor) == Some(&b'\n') {
                    cursor += 1;
                }
                let trimmed = line.trim_start();
                if trimmed.is_empty() {
                    if seen_content {
                        break;
                    }
                    continue;
                }
                seen_content = true;
                // Accept `- `, `* `, `• `, or a bare line as bullet.
                let body = trimmed
                    .strip_prefix("- ")
                    .or_else(|| trimmed.strip_prefix("* "))
                    .or_else(|| trimmed.strip_prefix("• "))
                    .unwrap_or(trimmed);
                let Some((url, rest_after)) = split_url(body) else {
                    break;
                };
                // Compute absolute offset of the URL in `content`.
                let body_offset_in_line = line.len() - body.len();
                let url_abs_start = line_start + body_offset_in_line;
                let url_abs_end = url_abs_start + url.len();
                let title = rest_after
                    .trim()
                    .trim_matches(|c: char| c == '-' || c.is_whitespace());
                let title_out = if title.is_empty() {
                    url.clone()
                } else {
                    title.to_string()
                };
                out.push(Annotation::UrlCitation {
                    url_citation: UrlCitation {
                        start_index: url_abs_start,
                        end_index: url_abs_end,
                        url,
                        title: title_out,
                    },
                });
            }
            search_from = after;
        }
    }
    out
}

/// Split `s` into `(url, rest)` where `url` is the first http(s) token.
/// Returns `None` when the string doesn't begin with a URL.
fn split_url(s: &str) -> Option<(String, &str)> {
    if !(s.starts_with("http://") || s.starts_with("https://")) {
        return None;
    }
    let end = s
        .find(|c: char| c.is_whitespace() || matches!(c, ')' | ']' | '>' | '`'))
        .unwrap_or(s.len());
    let url = s[..end].trim_end_matches(['.', ',', ';', ':', '!', '?']);
    if url.len() <= "https://".len() {
        return None;
    }
    Some((url.to_string(), &s[end..]))
}

/// Merge two annotation lists, deduping by URL (keeping the first hit
/// in document order). Used to combine the bare-URL extractor's output
/// with structured citations without emitting the same URL twice.
pub fn merge_dedupe(mut primary: Vec<Annotation>, secondary: Vec<Annotation>) -> Vec<Annotation> {
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    for a in &primary {
        let Annotation::UrlCitation { url_citation } = a;
        seen.insert(url_citation.url.clone());
    }
    for a in secondary {
        let url = match &a {
            Annotation::UrlCitation { url_citation } => url_citation.url.clone(),
        };
        if seen.insert(url) {
            primary.push(a);
        }
    }
    primary.sort_by_key(|a| match a {
        Annotation::UrlCitation { url_citation } => url_citation.start_index,
    });
    primary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn urls(anns: &[Annotation]) -> Vec<(&str, &str, usize, usize)> {
        anns.iter()
            .map(|a| match a {
                Annotation::UrlCitation {
                    url_citation:
                        UrlCitation {
                            url,
                            title,
                            start_index,
                            end_index,
                        },
                } => (url.as_str(), title.as_str(), *start_index, *end_index),
            })
            .collect()
    }

    #[test]
    fn footnote_ref_resolved() {
        let input = "See the docs[^1] for more.\n\n[^1]: https://example.com/api The API reference";
        let got = extract(input);
        let u = urls(&got);
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].0, "https://example.com/api");
        assert_eq!(u[0].1, "The API reference");
    }

    #[test]
    fn footnote_without_title_falls_back_to_label() {
        let input = "Cite me[^src].\n[^src]: https://example.com";
        let got = extract(input);
        let u = urls(&got);
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].0, "https://example.com");
        assert_eq!(u[0].1, "src");
    }

    #[test]
    fn numeric_refs_resolved() {
        let input =
            "See [1] and [2].\n\n[1] https://a.example.com\n[2] https://b.example.com Example B";
        let got = extract(input);
        let u = urls(&got);
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].0, "https://a.example.com");
        assert_eq!(u[1].0, "https://b.example.com");
        assert_eq!(u[1].1, "Example B");
    }

    #[test]
    fn numeric_ref_ignores_non_refs() {
        let input = "array[0] is [abc].\nNo definitions here.";
        let got = extract(input);
        assert!(got.is_empty());
    }

    #[test]
    fn sources_block_extracted() {
        let input = "Some answer.\n\nSources:\n- https://a.example.com\n- https://b.example.com short description\n\nOther text.";
        let got = extract(input);
        let u = urls(&got);
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].0, "https://a.example.com");
        assert_eq!(u[1].0, "https://b.example.com");
        assert_eq!(u[1].1, "short description");
    }

    #[test]
    fn references_heading_also_matches() {
        let input = "Stuff.\n\nReferences:\n- https://x.example.com\n";
        let got = extract(input);
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn no_citations_returns_empty() {
        let got = extract("plain text with no citations");
        assert!(got.is_empty());
    }

    #[test]
    fn merge_dedupe_by_url() {
        let a = vec![Annotation::UrlCitation {
            url_citation: UrlCitation {
                start_index: 0,
                end_index: 20,
                url: "https://example.com".into(),
                title: "first".into(),
            },
        }];
        let b = vec![Annotation::UrlCitation {
            url_citation: UrlCitation {
                start_index: 50,
                end_index: 70,
                url: "https://example.com".into(),
                title: "second".into(),
            },
        }];
        let merged = merge_dedupe(a, b);
        assert_eq!(merged.len(), 1);
    }
}
