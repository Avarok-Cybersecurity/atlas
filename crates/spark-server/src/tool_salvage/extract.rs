// SPDX-License-Identifier: AGPL-3.0-only

// ── Shape extractors ───────────────────────────────────────────

use super::extract_more::{looks_like_path, synthesise};
use super::shape::ToolShape;
use crate::tool_parser::ToolCall;

/// `<NAME>...<KEY>VAL</KEY>...</NAME>` — generalised XML form.
pub(super) fn extract_xml(content: &str, matchers: &[ToolShape]) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let lower = content.to_ascii_lowercase();
    for m in matchers {
        if m.name().len() < 3 {
            continue; // too short — false-positive prone
        }
        let open = format!("<{}>", m.name_lower());
        let close = format!("</{}>", m.name_lower());
        let mut search = 0usize;
        while let Some(rel) = lower[search..].find(&open) {
            let start = search + rel;
            let body_start = start + open.len();
            let Some(rel_close) = lower[body_start..].find(&close) else {
                break;
            };
            let body_end = body_start + rel_close;
            let body = &content[body_start..body_end];
            if let Some(args) = parse_xml_kv(body, m)
                && let Some(tc) = synthesise(m.name(), &args, "xml")
            {
                out.push(tc);
            }
            search = body_end + close.len();
        }
    }
    out
}

/// Body of an XML block parsed as `<KEY>VAL</KEY>` pairs. Keys are
/// matched case-insensitively against the tool's declared
/// properties; values are JSON-stringified.
pub(super) fn parse_xml_kv(
    body: &str,
    shape: &ToolShape,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut out = serde_json::Map::new();
    let bytes = body.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        // Skip whitespace.
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] != b'<' {
            pos += 1;
            continue;
        }
        let tag_start = pos + 1;
        let Some(gt_rel) = body[tag_start..].find('>') else {
            break;
        };
        let tag = &body[tag_start..tag_start + gt_rel];
        if tag.is_empty()
            || tag.starts_with('/')
            || !tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            pos = tag_start + gt_rel + 1;
            continue;
        }
        let val_start = tag_start + gt_rel + 1;
        let close = format!("</{tag}>");
        let Some(close_rel) = body[val_start..].find(&close) else {
            break;
        };
        let value = body[val_start..val_start + close_rel].trim().to_string();
        // Map case-insensitively to schema property name.
        let lower = tag.to_ascii_lowercase();
        if let Some(prop) = shape.original_property(&lower) {
            out.insert(prop, serde_json::Value::String(value));
        }
        pos = val_start + close_rel + close.len();
    }
    if out.is_empty() { None } else { Some(out) }
}

/// ```` ```LANG\n<body>\n``` ```` — fenced-code form. Salvages when
/// LANG matches a tool name (or "shell"→bash) AND the tool has
/// exactly one required string parameter.
pub(super) fn extract_fenced(content: &str, matchers: &[ToolShape]) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let mut search = 0usize;
    while let Some(rel) = content[search..].find("```") {
        let fence_start = search + rel;
        let after = fence_start + 3;
        // Read lang up to newline.
        let info_end = content[after..]
            .find('\n')
            .map(|i| after + i)
            .unwrap_or(content.len());
        let info = content[after..info_end].trim();
        let body_start = info_end + 1;
        // Find closing fence.
        let Some(close_rel) = content[body_start..].find("\n```") else {
            break;
        };
        let body_end = body_start + close_rel;
        let body = content[body_start..body_end].trim_end();
        // info-string may include `LANG:filename` — split on `:`
        let lang = info
            .split(':')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let path_in_info = info.split_once(':').map(|(_, p)| p.trim().to_string());
        let lang_aliases = [
            lang.as_str(),
            match lang.as_str() {
                "shell" | "sh" => "bash",
                "powershell" | "ps1" => "bash",
                other => other,
            },
        ];

        for m in matchers {
            let mname = m.name_lower();
            if !lang_aliases.contains(&mname.as_str()) {
                // Try the file-write shape: lang denotes file type,
                // info-string carries `:path`.
                if let (Some(path), Some((path_prop, content_prop))) =
                    (path_in_info.as_ref(), m.path_and_content())
                {
                    let mut args = serde_json::Map::new();
                    args.insert(path_prop, serde_json::Value::String(path.clone()));
                    args.insert(content_prop, serde_json::Value::String(body.to_string()));
                    if let Some(tc) = synthesise(m.name(), &args, "fenced") {
                        out.push(tc);
                    }
                }
                continue;
            }
            if let Some(prop) = m.single_required_string() {
                let mut args = serde_json::Map::new();
                args.insert(prop, serde_json::Value::String(body.to_string()));
                if let Some(tc) = synthesise(m.name(), &args, "fenced") {
                    out.push(tc);
                }
            }
        }
        search = body_end + 3;
    }
    out
}

/// `cat > /path << 'EOF' ... EOF` — heredoc-style file writes that
/// the model sometimes emits as bash prose. Maps to the file-write
/// shape (path-prop + content-prop).
pub(super) fn extract_heredoc(content: &str, matchers: &[ToolShape]) -> Vec<ToolCall> {
    let mut out = Vec::new();
    // Look for "cat > <PATH> << 'EOF'" (or "<<EOF" without quotes).
    let mut search = 0usize;
    while let Some(rel) = content[search..].find("cat >") {
        let head_start = search + rel;
        // Read up to end-of-line for the cat command.
        let line_end = content[head_start..]
            .find('\n')
            .map(|i| head_start + i)
            .unwrap_or(content.len());
        let line = &content[head_start..line_end];
        // Parse "cat > <path> << 'EOF'" (loose).
        let after_gt = line.find('>').map(|i| i + 1).unwrap_or(line.len());
        let rest = line[after_gt..].trim_start();
        let path = rest.split_whitespace().next().unwrap_or("");
        let path = path.trim_matches(|c: char| c == '"' || c == '\'');
        if path.is_empty() {
            search = line_end + 1;
            continue;
        }
        // Find the heredoc terminator (typically EOF). Accept any
        // identifier after "<<" optionally quoted.
        let Some(ll_rel) = line.find("<<") else {
            search = line_end + 1;
            continue;
        };
        let ll = line[ll_rel + 2..].trim_start();
        let term = ll
            .trim_start_matches(['\'', '"', '-'])
            .split(|c: char| c == '\'' || c == '"' || c.is_whitespace())
            .next()
            .unwrap_or("EOF");
        if term.is_empty() {
            search = line_end + 1;
            continue;
        }
        // Body is from line_end+1 until a line that is exactly `term`.
        let body_start = line_end + 1;
        let mut body_end = body_start;
        let mut found = false;
        for (offset, ln) in content[body_start..].split_inclusive('\n').enumerate() {
            let _ = offset;
            let trim = ln.trim_end_matches('\n');
            if trim.trim() == term {
                found = true;
                break;
            }
            body_end += ln.len();
        }
        if !found {
            search = line_end + 1;
            continue;
        }
        let body = &content[body_start..body_end];
        for m in matchers {
            if let Some((path_prop, content_prop)) = m.path_and_content() {
                let mut args = serde_json::Map::new();
                args.insert(path_prop, serde_json::Value::String(path.to_string()));
                args.insert(
                    content_prop,
                    serde_json::Value::String(body.trim_end().to_string()),
                );
                if let Some(tc) = synthesise(m.name(), &args, "heredoc") {
                    out.push(tc);
                }
                break; // first matching write tool wins
            }
        }
        search = body_end + term.len();
    }
    out
}

/// `<filename>\n\n<body>` — header+body form. Fires when the line
/// is recognisable as a file path (has a known extension or is a
/// known exact filename like `Dockerfile`) and there is a non-trivial
/// body before the next header or end-of-content.
pub(super) fn extract_header_body(content: &str, matchers: &[ToolShape]) -> Vec<ToolCall> {
    // Find the first matcher that has the file-write shape.
    let writer = matchers
        .iter()
        .find_map(|m| m.path_and_content().map(|(p, c)| (m, p, c)));
    let Some((write_shape, path_prop, content_prop)) = writer else {
        return Vec::new();
    };

    let lines: Vec<&str> = content.lines().collect();
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let path = match looks_like_path(lines[i]) {
            Some(p) => p,
            None => {
                i += 1;
                continue;
            }
        };
        let mut j = i + 1;
        while j < lines.len() && lines[j].trim().is_empty() {
            j += 1;
        }
        let body_start = j;
        while j < lines.len() && looks_like_path(lines[j]).is_none() {
            j += 1;
        }
        let body = lines[body_start..j].join("\n").trim_end().to_string();
        if body.len() >= 40 {
            blocks.push((path, body));
        }
        i = j;
    }
    blocks
        .into_iter()
        .filter_map(|(path, body)| {
            let mut args = serde_json::Map::new();
            args.insert(path_prop.clone(), serde_json::Value::String(path));
            args.insert(content_prop.clone(), serde_json::Value::String(body));
            synthesise(write_shape.name(), &args, "header_body")
        })
        .collect()
}
