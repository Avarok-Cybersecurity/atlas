// SPDX-License-Identifier: AGPL-3.0-only

//! The tool surface the agent loop presents, and its implementations.
//!
//! These are the six tools `~/.config/opencode/agents/atlas.md` enables in its
//! frontmatter — `read`, `glob`, `grep`, `bash`, `write`, `edit`. That file is
//! the agent `bench/fp8_dgx2_drift/harness/run_tier.sh` drives via
//! `default_agent: atlas`, so it is the surface the recorded 10/10 tier was
//! measured against. `fetch`, `todoread` and `todowrite` are `false` there and
//! are absent here for the same reason.
//!
//! Split out of `agent.rs` only for the 500-LoC cap; the containment rules
//! documented there apply to every path resolved below.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use super::{AgentConfig, resolve, run_shell};

/// opencode's `read`: 2000 lines, each truncated at 2000 characters.
const READ_LINES: usize = 2000;
const READ_LINE_CHARS: usize = 2000;

/// opencode's `grep` stops at 100 matches and says so.
const GREP_LIMIT: usize = 100;

/// Descriptions are opencode's own, trimmed to the clauses that hold here (no
/// persistent shell, no `workdir`, no LSP diagnostics, no `task` subagent) —
/// a tool doc that promises behaviour the tool does not have is the same class
/// of scaffolding bug as having no doc at all.
pub fn tool_schema() -> Value {
    let s = |d: &str| json!({"type": "string", "description": d});
    let n = |d: &str| json!({"type": "integer", "description": d});
    let p = s("The absolute path to the file. A relative one is also accepted.");
    json!([
        {"type": "function", "function": {"name": "bash",
            "description": "Executes a given bash command, ensuring proper handling and security measures.\n\nUsage notes:\n- The command argument is required.\n- You can specify an optional timeout in milliseconds.\n- A long-running process MUST be started detached with its output redirected to a file (`setsid cmd > /tmp/log 2>&1 &`); a background process that inherits this tool's pipes holds them open and the command is reported as timed out.\n- Long output is truncated in the middle and the number of elided characters is reported.",
            "parameters": {"type": "object", "required": ["command"], "properties": {
                "command": s("The command to execute"), "timeout": n("Optional timeout in milliseconds")}}}},
        {"type": "function", "function": {"name": "read",
            "description": "Reads a file, or lists a directory.\n\n- The offset parameter is the line number to start from (1-indexed).\n- Contents are returned with each line prefixed by its line number as `<line>: <content>`.\n- For directories, entries are returned one per line (without line numbers) with a trailing `/` for subdirectories.\n- Any line longer than 2000 characters is truncated.",
            "parameters": {"type": "object", "required": ["filePath"], "properties": {
                "filePath": p, "offset": n("The line number to start reading from (1-indexed)"),
                "limit": n("The maximum number of lines to read (defaults to 2000)")}}}},
        {"type": "function", "function": {"name": "write",
            "description": "Writes a file to the local filesystem.\n\nUsage:\n- This tool will overwrite the existing file if there is one at the provided path.\n- Parent directories are created for you.\n- NEVER proactively create documentation files (*.md) or README files.",
            "parameters": {"type": "object", "required": ["filePath", "content"], "properties": {
                "filePath": p, "content": s("The content to write to the file")}}}},
        {"type": "function", "function": {"name": "edit",
            "description": "Performs exact string replacements in files.\n\nUsage:\n- When editing text from read tool output, preserve the exact indentation as it appears AFTER the line number prefix. The prefix format is `1: `; never include any part of it in oldString.\n- The edit will FAIL if `oldString` is not found in the file.\n- The edit will FAIL if `oldString` is found multiple times. Either provide a larger string with more surrounding context to make it unique, or use `replaceAll` to change every instance.",
            "parameters": {"type": "object", "required": ["filePath", "oldString", "newString"], "properties": {
                "filePath": p, "oldString": s("The text to replace"), "newString": s("The text to replace it with"),
                "replaceAll": {"type": "boolean", "description": "Replace all occurrences (default false)"}}}}},
        {"type": "function", "function": {"name": "glob",
            "description": "- Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like \"**/*.rs\" or \"src/**/*.rs\"\n- Returns matching file paths\n- Use this tool when you need to find files by name patterns",
            "parameters": {"type": "object", "required": ["pattern"], "properties": {
                "pattern": s("The glob pattern to match files against")}}}},
        {"type": "function", "function": {"name": "grep",
            "description": "- Searches file contents and returns file paths and line numbers with matching lines\n- `pattern` is matched as a literal substring, NOT a regular expression; for regex use the bash tool with `grep -E`\n- Stops after 100 matches",
            "parameters": {"type": "object", "required": ["pattern"], "properties": {
                "pattern": s("The literal text to search for in file contents"),
                "include": s("File pattern to include in the search (e.g. \"*.rs\")")}}}}
    ])
}

/// Dispatch one model-issued tool call. Shell commands are appended to
/// `commands`, which is the evidence `score::followed_directions` reads.
pub async fn execute(
    cfg: &AgentConfig,
    call: &crate::http::ToolCall,
    commands: &mut Vec<String>,
) -> Result<String> {
    let raw = match call.arguments.is_empty() {
        true => "{}",
        false => &call.arguments,
    };
    let args: Value =
        serde_json::from_str(raw).map_err(|e| anyhow!("arguments were not valid JSON: {e}"))?;
    let arg = |k: &str| args.get(k).and_then(Value::as_str).map(str::to_string);
    let need = |k: &str| arg(k).ok_or_else(|| anyhow!("{} needs a `{k}`", call.name));
    // opencode names it `filePath`; a model that learned another surface reaches
    // for `path`. Accepting both costs nothing and saves a turn.
    let file = || match arg("filePath").or_else(|| arg("path")) {
        Some(rel) => resolve(&cfg.sandbox, &rel),
        None => bail!("{} needs a `filePath`", call.name),
    };
    match call.name.as_str() {
        "bash" => {
            let cmd = need("command")?;
            commands.push(cmd.clone());
            let limit = args
                .get("timeout")
                .and_then(Value::as_u64)
                .map(Duration::from_millis)
                // A model-supplied timeout may shorten the ceiling, never raise
                // it: the ceiling is what bounds the benchmark's wall time.
                .map_or(cfg.command_timeout, |t| t.min(cfg.command_timeout));
            run_shell(cfg, &cmd, limit).await
        }
        "write" => {
            let (path, content) = (file()?, arg("content").unwrap_or_default());
            std::fs::create_dir_all(path.parent().unwrap_or(&cfg.sandbox))?;
            std::fs::write(&path, &content)?;
            Ok(format!(
                "wrote {} ({} bytes)",
                rel(cfg, &path),
                content.len()
            ))
        }
        "read" => read_tool(&args, file()?),
        "edit" => {
            let path = file()?;
            let hits = edit_tool(&args, &path, &need("oldString")?, &need("newString")?)?;
            // Sandbox-relative, like `write`: the absolute form carries `$HOME`
            // and the run index into the model's context and into the
            // trajectory trace, whose whole job is to `diff` clean between two
            // runs — including two runs on different boxes.
            Ok(format!(
                "replaced {hits} occurrence(s) in {}",
                rel(cfg, &path)
            ))
        }
        "glob" => {
            let hits: Vec<String> = matching(cfg, Some(&need("pattern")?)).collect();
            Ok(or_empty(hits.join("\n")))
        }
        "grep" => Ok(grep_tool(cfg, &need("pattern")?, arg("include").as_deref())),
        other => bail!("unknown tool {other}; available: bash, read, write, edit, glob, grep"),
    }
}

/// opencode's `read`, including its directory mode — the cheapest way for the
/// agent to notice it wrote a `Cargo.toml` and no `src/`.
fn read_tool(args: &Value, path: PathBuf) -> Result<String> {
    if path.is_dir() {
        let mut names: Vec<String> = std::fs::read_dir(&path)?
            .flatten()
            .map(|e| {
                let slash = if e.path().is_dir() { "/" } else { "" };
                e.file_name().to_string_lossy().into_owned() + slash
            })
            .collect();
        names.sort();
        return Ok(or_empty(names.join("\n")));
    }
    let text = read_capped(&path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let num = |k: &str| args.get(k).and_then(Value::as_u64).map(|n| n as usize);
    let offset = num("offset").unwrap_or(1).max(1);
    let body: Vec<String> = text
        .lines()
        .enumerate()
        .skip(offset - 1)
        .take(num("limit").unwrap_or(READ_LINES))
        .map(|(i, l)| match l.chars().count() > READ_LINE_CHARS {
            true => format!(
                "{}: {}... (line truncated)",
                i + 1,
                head(l, READ_LINE_CHARS)
            ),
            false => format!("{}: {l}", i + 1),
        })
        .collect();
    Ok(or_empty(body.join("\n")))
}

/// Exact string replacement — the tool the agent had no way to reach for when
/// the only defect in its `main.rs` was a missing `.await`. The failure text is
/// opencode's word for word: it tells the model *how* to retry, which a bare
/// "no match" does not.
fn edit_tool(args: &Value, path: &Path, old: &str, new: &str) -> Result<usize> {
    let all = args.get("replaceAll").and_then(Value::as_bool) == Some(true);
    // Not capped like [`read_capped`]: this text is written back, so reading a
    // prefix of the file would truncate it on disk.
    let text = std::fs::read_to_string(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let hits = text.matches(old).count();
    if hits == 0 {
        bail!("oldString not found in content");
    }
    if hits > 1 && !all {
        bail!(
            "Found multiple matches for oldString ({hits}). Provide more surrounding lines in \
             oldString to identify the correct match, or set replaceAll to change every instance."
        );
    }
    std::fs::write(path, text.replacen(old, new, if all { hits } else { 1 }))?;
    Ok(hits)
}

/// Load a file the model wrote, refusing to be the thing that runs the box out
/// of memory.
///
/// Nothing bounds what ends up in the sandbox: the prompt's example redirects a
/// server to `/tmp/server.log`, and a model that redirects to `./server.log`
/// instead — or a `dd`, or a looping `println!` — leaves a file that `read` and
/// `grep` used to load whole. The cap is the most `read` can return anyway
/// (2000 lines × 2000 characters), so no byte that could have reached the model
/// is lost, and it is stated in the result rather than silently applied.
fn read_capped(path: &Path) -> std::io::Result<String> {
    use std::io::{Error, ErrorKind, Read};
    const CAP: usize = READ_LINES * READ_LINE_CHARS;
    let mut buf = Vec::new();
    std::fs::File::open(path)?
        .take(CAP as u64)
        .read_to_end(&mut buf)?;
    let capped = buf.len() == CAP;
    let mut text = match String::from_utf8(buf) {
        Ok(text) => text,
        // Non-UTF-8 is what `read_to_string` refused, and refusing is right: a
        // `read` of a binary should say so, and `grep` should skip the file
        // rather than paste mojibake into the model's context. The one
        // exception is a multi-byte character the cap itself cut in half.
        Err(e) if capped && e.as_bytes().len() - e.utf8_error().valid_up_to() < 4 => {
            let whole = e.utf8_error().valid_up_to();
            String::from_utf8_lossy(&e.as_bytes()[..whole]).into_owned()
        }
        Err(_) => {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "stream did not contain valid UTF-8",
            ));
        }
    };
    if capped {
        text.push_str("\n… (file truncated at this point) …");
    }
    Ok(text)
}

fn grep_tool(cfg: &AgentConfig, pattern: &str, include: Option<&str>) -> String {
    let mut out = Vec::new();
    for rel in matching(cfg, include) {
        let Ok(text) = read_capped(&cfg.sandbox.join(&rel)) else {
            continue;
        };
        for (i, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| l.contains(pattern))
        {
            out.push(format!("{rel}:{}: {}", i + 1, head(line.trim(), 300)));
            if out.len() >= GREP_LIMIT {
                return format!("{}\n(stopped at {GREP_LIMIT} matches)", out.join("\n"));
            }
        }
    }
    match out.is_empty() {
        true => "No files found".into(),
        false => format!("Found {} matches\n{}", out.len(), out.join("\n")),
    }
}

/// Sandbox-relative paths of every file, optionally filtered by a glob. Build
/// output and VCS metadata are skipped — a `target/` tree holds thousands of
/// files and none of them are the model's work.
///
/// **Symlinks are skipped entirely**, by [`entry.file_type()`][std::fs::DirEntry::file_type],
/// which reports the link rather than what it points at. `is_dir()` resolves the
/// link, and `ln -s . a` — three of them, one bash call — turns this walk into
/// 3^40 paths: the kernel's own `ELOOP` ceiling is the only thing that bounds
/// the depth, and it does not bound the breadth. Nothing times out a tool call,
/// so `glob` never returns and the iteration is lost. Skipping is also the
/// honest answer for a link that leaves the sandbox: `grep` reads what it finds
/// with no path check at all.
fn matching(cfg: &AgentConfig, pattern: Option<&str>) -> impl Iterator<Item = String> {
    let (mut stack, mut files) = (vec![cfg.sandbox.clone()], Vec::new());
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if matches!(entry.file_name().to_str(), Some("target") | Some(".git")) {
                continue;
            }
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            match (kind.is_dir(), kind.is_file()) {
                (true, _) => stack.push(path),
                (_, true) => files.push(rel(cfg, &path)),
                _ => {}
            }
        }
    }
    files.sort();
    let pattern = pattern.map(str::to_string);
    files.into_iter().filter(move |f| {
        pattern
            .as_ref()
            .is_none_or(|p| glob_match(p.as_bytes(), f.as_bytes()))
    })
}

fn rel(cfg: &AgentConfig, path: &Path) -> String {
    let path = path.strip_prefix(&cfg.sandbox).unwrap_or(path);
    path.to_string_lossy().into_owned()
}

/// An empty tool result tells the model nothing, and "there is nothing there"
/// is a fact worth stating — opencode says "No files found" rather than "".
fn or_empty(s: String) -> String {
    match s.is_empty() {
        true => "No files found".into(),
        false => s,
    }
}

fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// `*` (within one path segment), `**` (across segments), `?`. A pattern with
/// no separator matches the basename, which is how `*.rs` is meant.
pub fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
    if !pattern.contains(&b'/')
        && let Some(cut) = text.iter().rposition(|c| *c == b'/')
    {
        return glob_match(pattern, &text[cut + 1..]);
    }
    matches_from(pattern, text)
}

fn matches_from(pattern: &[u8], text: &[u8]) -> bool {
    let Some(&head) = pattern.first() else {
        return text.is_empty();
    };
    if head == b'*' {
        let deep = pattern.get(1) == Some(&b'*');
        let mut rest = &pattern[if deep { 2 } else { 1 }..];
        // `**/` also matches zero directories, so `**/*.rs` finds `main.rs`.
        if deep && rest.first() == Some(&b'/') {
            rest = &rest[1..];
        }
        return matches_from(rest, text)
            || (0..text.len())
                .take_while(|i| deep || text[*i] != b'/')
                .any(|i| matches_from(rest, &text[i + 1..]));
    }
    match text.first() {
        Some(c) if head == b'?' || head == *c => matches_from(&pattern[1..], &text[1..]),
        _ => false,
    }
}

/// Moved from `agent.rs` (500-LoC cap).
///
/// ★ Deliberately narrow. `<tool_call>` and `<function=` are the qwen3_coder
/// wire forms, not English — prose that merely discusses calling a tool does
/// not contain them, and a model that writes one in a fenced code block to
/// explain itself gets one extra turn, not a failed run. It is checked only
/// after `tool_calls.is_empty()`, so a turn whose call parsed correctly never
/// reaches it however much syntax the prose quotes.
pub(super) fn emitted_unparsed_call(outcome: &crate::http::ChatOutcome) -> bool {
    outcome.tool_calls.is_empty()
        && ["<tool_call>", "<function="]
            .iter()
            .any(|m| outcome.text.contains(m))
}

#[cfg(test)]
#[path = "agent_tools_tests.rs"]
mod tests;
