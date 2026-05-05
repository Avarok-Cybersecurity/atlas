// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::tool_parser::FunctionDefinition;

fn write_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "Write".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            })),
        },
    }
}

fn bash_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "bash".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            })),
        },
    }
}

#[test]
fn xml_block_round_trips_via_schema() {
    let content = "I'll write the file.\n\
                       <write><file_path>/tmp/x.rs</file_path><content>fn main() {}</content></write>";
    let calls = salvage(content, &[write_tool()]);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "Write");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["file_path"], "/tmp/x.rs");
    assert_eq!(args["content"], "fn main() {}");
}

#[test]
fn fenced_bash_block_is_extracted_when_lang_matches() {
    let content = "Let me run this:\n```bash\nls -la /tmp\n```\nDone.";
    let calls = salvage(content, &[bash_tool()]);
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["command"], "ls -la /tmp");
}

#[test]
fn fenced_block_with_file_info_string_writes_to_path() {
    // Some clients render fences as ```rust:src/main.rs
    let content = "Here it is:\n\n```rust:src/main.rs\nfn main() {}\n```\n";
    let calls = salvage(content, &[write_tool()]);
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["file_path"], "src/main.rs");
    assert_eq!(args["content"], "fn main() {}");
}

#[test]
fn header_body_extracts_two_files() {
    let content = "Now I'll create the project files:\n\n\
                       Cargo.toml\n\n[package]\nname = \"x\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
                       src/main.rs\n\nfn main() { println!(\"hi\"); }\nlet x = 1;\nlet y = 2;\n";
    let calls = salvage(content, &[write_tool()]);
    assert_eq!(calls.len(), 2, "should salvage two file blocks");
    let a: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(a["file_path"], "Cargo.toml");
    let b: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(b["file_path"], "src/main.rs");
}

#[test]
fn heredoc_form_extracts_path_and_body() {
    let content = "Setting up:\n\
                       cat > /tmp/x.toml << 'EOF'\n\
                       [package]\nname = \"y\"\nversion = \"0.1\"\n\
                       EOF\n";
    let calls = salvage(content, &[write_tool()]);
    assert_eq!(calls.len(), 1, "heredoc must salvage exactly once");
    let a: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(a["file_path"], "/tmp/x.toml");
    assert!(a["content"].as_str().unwrap().contains("name = \"y\""));
}

#[test]
fn no_tools_yields_empty() {
    let calls = salvage("anything", &[]);
    assert!(calls.is_empty());
}

#[test]
fn fenced_block_with_unmatched_lang_is_ignored() {
    let content = "Here:\n```python\nprint('hi')\n```\n";
    let calls = salvage(content, &[bash_tool()]);
    assert!(calls.is_empty(), "python lang must not salvage as bash");
}

#[test]
fn xml_inline_text_only_does_not_match_tool_name_below_3_chars() {
    let mut tool = bash_tool();
    tool.function.name = "ls".into();
    let content = "use <ls> example";
    let calls = salvage(content, &[tool]);
    assert!(calls.is_empty(), "tool name <3 chars must skip xml shape");
}

#[test]
fn salvage_dedupes_identical_args_extracted_twice() {
    // Same (path, content) appearing in BOTH XML and header+body
    // form must produce a single tool call. The body must be
    // ≥ 40 chars (the header+body gate) for both extractors to
    // hit, so we use a longer file body in both.
    let body = "fn main() { println!(\"hello\"); let x = 1; let y = 2; }";
    let content = format!(
        "<write><file_path>/tmp/x.rs</file_path><content>{body}</content></write>\n\
             /tmp/x.rs\n\n{body}\n"
    );
    let calls = salvage(&content, &[write_tool()]);
    assert_eq!(
        calls.len(),
        1,
        "same (name, args) must dedupe — got {calls:#?}"
    );
}

// ── Bare-tool invocation (#3, 2026-04-25) ──

#[test]
fn bare_invocation_write_path_then_body() {
    // Failure-mode pattern from claude-export.txt: "Write /tmp/x"
    // on its own line, followed by file body in plain prose.
    let content = "I'll create the project files now.\n\
                       Write /tmp/x.toml\n\n\
                       [package]\nname = \"x\"\nversion = \"0.1.0\"\nedition = \"2021\"\n";
    let calls = salvage(content, &[write_tool()]);
    assert_eq!(calls.len(), 1, "bare 'Write <path>' + body must salvage");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["file_path"], "/tmp/x.toml");
    assert!(args["content"].as_str().unwrap().contains("[package]"));
}

#[test]
fn bare_invocation_bash_command_inline() {
    // "Bash ls -la /tmp" as a standalone line.
    let content = "Let me check.\nBash ls -la /tmp\nDone.";
    let calls = salvage(content, &[bash_tool()]);
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["command"], "ls -la /tmp");
}

#[test]
fn bare_invocation_paren_form() {
    // "Bash(ls -la /tmp)" — Claude Code's display shape some
    // models echo back into prose.
    let content = "Bash(ls -la /tmp)\n";
    let calls = salvage(content, &[bash_tool()]);
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["command"], "ls -la /tmp");
}

#[test]
fn bare_invocation_inline_mention_does_not_fire() {
    // "I will Write the file" — Write is in the middle of prose,
    // not at line-start. Must NOT trigger.
    let content = "I will Write the file when ready.\nThen we will Bash a command.";
    let calls = salvage(content, &[write_tool(), bash_tool()]);
    assert!(
        calls.is_empty(),
        "inline mentions must not salvage: got {calls:#?}"
    );
}

#[test]
fn bare_invocation_too_short_arg_skipped() {
    // "Bash a" — arg too short / not command-like.
    let content = "Bash a\n";
    let calls = salvage(content, &[bash_tool()]);
    assert!(calls.is_empty());
}

// ── <file>PATH</file><content>BODY</content> pair (opencode
//    task-envelope mimicry, 2026-04-25 dump seq=104..111) ──

#[test]
fn file_content_pair_inside_task_envelope() {
    // Verbatim shape from dump seq=111 (truncated for the test).
    let content = r#"
I need to fix the GUI code. Let me edit the file:

<task>
<description>
Fix GUI code type mismatch
</description>
<prompt>
Fix the GUI code to use correct egui Context API
</prompt>

<file>
/tmp/calc-test60/src/gui/src/lib.rs
</file>

<content>
//! GUI Calculator using egui framework
use egui::{CentralPanel};

pub struct Calculator {
    display: String,
}
</content>
</task>
"#;
    let calls = salvage(content, &[write_tool()]);
    assert_eq!(calls.len(), 1, "must extract the inner file/content pair");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["file_path"], "/tmp/calc-test60/src/gui/src/lib.rs");
    assert!(
        args["content"].as_str().unwrap().contains("CentralPanel"),
        "content must include the body: {}",
        args["content"]
    );
}

#[test]
fn file_content_pair_skipped_when_no_write_tool() {
    let content = "<file>/tmp/x</file><content>body</content>";
    let calls = salvage(content, &[bash_tool()]);
    assert!(calls.is_empty(), "no write tool declared → no salvage");
}

#[test]
fn file_content_pair_two_pairs_yield_two_calls() {
    let content = r#"
<file>/tmp/a.rs</file>
<content>fn a() {}</content>
some prose
<file>/tmp/b.rs</file>
<content>fn b() {}</content>
"#;
    let calls = salvage(content, &[write_tool()]);
    assert_eq!(calls.len(), 2, "two pairs → two tool calls");
    let a: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    let b: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(a["file_path"], "/tmp/a.rs");
    assert_eq!(b["file_path"], "/tmp/b.rs");
}

#[test]
fn file_content_pair_skipped_when_distance_too_far() {
    // <content> appears > 256 bytes after </file> — too far,
    // probably not the matching pair. We err on the side of
    // not-firing.
    let filler = "x".repeat(300);
    let content = format!("<file>/tmp/x.rs</file>{filler}<content>body</content>");
    let calls = salvage(&content, &[write_tool()]);
    assert!(
        calls.is_empty(),
        "distance > 256B between </file> and <content> rejects: got {calls:#?}"
    );
}

#[test]
fn file_content_pair_empty_body_skipped() {
    let content = "<file>/tmp/x.rs</file><content>   </content>";
    let calls = salvage(content, &[write_tool()]);
    assert!(calls.is_empty(), "empty body rejects");
}

#[test]
fn file_content_pair_truncated_close_tag_aborts_safely() {
    // The model emitted <file>...</file><content>...  but never
    // closed </content>. Must not panic; must not emit a
    // garbage tool call.
    let content = "<file>/tmp/x.rs</file><content>body never closed";
    let calls = salvage(content, &[write_tool()]);
    assert!(calls.is_empty(), "truncated <content> rejects");
}
