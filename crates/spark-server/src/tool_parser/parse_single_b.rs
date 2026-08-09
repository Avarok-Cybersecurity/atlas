// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

pub(super) fn parse_qwen3_coder_call(text: &str, _idx: u32) -> Option<ToolCall> {
    // Handle both <function=name> (correct) and <function name> (common model error).
    // Handle NVFP4 quantization variants: <function=name>, <function name>,
    // <|function=name>, <|function name> (pipe prefix is a common NVFP4 artifact).
    let (func_start, prefix_len) = if let Some(pos) = text.find("<function=") {
        (pos, "<function=".len())
    } else if let Some(pos) = text.find("<|function=") {
        (pos, "<|function=".len())
    } else if let Some(pos) = text.find("<function ") {
        (pos, "<function ".len())
    } else if let Some(pos) = text.find("<|function ") {
        (pos, "<|function ".len())
    } else {
        return None;
    };
    let name_start = func_start + prefix_len;
    // Name ends at '>', '\n', or '<' — whichever comes first.
    // NVFP4 models sometimes emit <function=bash\n<parameter=...> without
    // closing the function tag, so we stop at newline or next tag start.
    let name_end = name_start
        + text[name_start..]
            .find(['>', '\n', '<'])
            .unwrap_or(text[name_start..].len());
    let func_name = normalize_tool_name(&text[name_start..name_end]);
    if func_name.is_empty() {
        return None;
    }

    let mut args = serde_json::Map::new();
    let after_name = if name_end < text.len() {
        &text[name_end + 1..]
    } else {
        ""
    };
    let mut rest = after_name;

    while let Some(p) = rest.find("<parameter=") {
        // Hard terminator: when `</function>` appears BEFORE the next
        // `<parameter=`, the current function block has ended and the
        // upcoming `<parameter=` belongs to a SIBLING `<function=...>`
        // block. Without this break the loop would continue to harvest
        // parameters past the `</function>` boundary and merge them
        // into THIS call's args dict — producing a single tool_call
        // whose arguments mix fields from multiple tools' schemas
        // (opencode session 2026-04-25: a `Write` call with `command`
        // / `description` / `timeout` fields from the next `bash`
        // call). This fix is independent of the recovery branches
        // below: those handle "missing `</parameter>` close inside
        // the same function"; this guard handles "consecutive
        // function blocks".
        if let Some(fc) = rest.find("</function>")
            && fc < p
        {
            break;
        }
        rest = &rest[p + "<parameter=".len()..];
        let key_end = rest.find('>')?;
        let param_name = rest[..key_end].trim().to_string();
        rest = &rest[key_end + 1..];

        // Stop at the LEAST of: `</parameter>` (proper close),
        // `<parameter=` (next param — recovery for a missing close),
        // or `</function>` (function ended without closing this param).
        // Without the recovery cases, a missing `</parameter>` swallows
        // every subsequent param into one giant value (vllm #38158
        // regression "streaming_missing_closing_tag").
        let proper = rest.find("</parameter>");
        let next_param = rest.find("<parameter=");
        let func_close = rest.find("</function>");
        let mut val_end = rest.len();
        let mut consumed_close = false;
        if let Some(p) = proper {
            val_end = p;
            consumed_close = true;
        }
        for cand in [next_param, func_close].into_iter().flatten() {
            if cand < val_end {
                val_end = cand;
                consumed_close = false;
            }
        }
        let raw_value = rest[..val_end].trim();
        // P0-2 (2026-07-09): when the value ended at a RECOVERY boundary
        // (next `<parameter=` / `</function>`), a garbled close-reopen
        // (`</parameter<parameter=` — the model dropped the `>`) leaves the
        // orphan `</parameter` as the value's tail. Strip it; it is the
        // model's intended close, not content.
        let raw_value = if !consumed_close {
            raw_value
                .strip_suffix("</parameter")
                .map(str::trim_end)
                .unwrap_or(raw_value)
        } else {
            raw_value
        };
        let advanced_to_func_close =
            !consumed_close && val_end < rest.len() && rest[val_end..].starts_with("</function>");
        rest = if consumed_close {
            &rest[val_end + "</parameter>".len()..]
        } else if val_end < rest.len() {
            // Recovery path: leave the next `<parameter=` / `</function>`
            // in place so the surrounding loop sees it.
            &rest[val_end..]
        } else {
            ""
        };

        // Always treat values as strings — the OpenAI tool calling spec
        // requires arguments to match the tool's JSON schema, and the model
        // emits values as raw text inside XML tags. If the schema declares
        // "content": {"type": "string"}, passing a parsed JSON object would
        // cause schema validation errors in clients (e.g. OpenCode).
        args.insert(param_name, serde_json::Value::String(raw_value.to_string()));

        // If the value was terminated by `</function>` (param had no
        // proper `</parameter>` close), we've reached the end of this
        // function's parameter list. Stop here so the next iteration
        // doesn't fall through to a sibling `<function=...>` block.
        if advanced_to_func_close {
            break;
        }
    }

    // Fallback: if no <parameter> tags found, try JSON between the function
    // tag and </function>. Grammar-constrained decoding emits:
    //   <function=Bash>{"command":"which cargo"}</function>
    if args.is_empty() {
        let body = after_name
            .find("</function>")
            .map(|end| after_name[..end].trim())
            .unwrap_or(after_name.trim());
        if body.starts_with('{')
            && let Ok(json_args) =
                serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(body)
        {
            args = json_args;
        }
    }

    // TAFC strip (A.4, 2026-04-25): the optional `_think` scratchpad
    // field (added to schemas via grammar::augment_schema_with_tafc_think
    // when [behavior].enable_tafc=true) carries the model's rationale
    // for selecting this tool. It must NOT reach the client tool
    // implementation — those expect schema-strict args. Strip
    // unconditionally; the field is only ever present when TAFC is
    // active in MODEL.toml.
    args.remove("_think");

    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall {
            name: func_name,
            arguments: serde_json::to_string(&serde_json::Value::Object(args))
                .unwrap_or_else(|_| "{}".into()),
        },
    })
}

/// Parse tag-style function call: `<function>NAME</function>` with optional `<parameters>` block.
///
/// Fallback for models that output XML tag format instead of qwen3_coder attribute format.
/// Handles: `<function>NAME</function><parameters>...<value>V</value>...</parameters>`
pub(super) fn parse_tag_style_call(text: &str, _idx: u32) -> Option<ToolCall> {
    let func_start = text.find("<function>")?;
    let after_tag = &text[func_start + "<function>".len()..];
    let close = after_tag.find("</function>")?;
    let raw_name = after_tag[..close].trim();

    // If the name region contains nested tags, extract <name> from within
    let func_name = if raw_name.contains('<') {
        normalize_tool_name(&extract_tag_value(raw_name, "name")?)
    } else if raw_name.is_empty() {
        return None;
    } else {
        normalize_tool_name(raw_name)
    };

    let mut args = serde_json::Map::new();
    let rest = &after_tag[close + "</function>".len()..];

    // Extract from <parameters> block if present
    if let Some(ps) = rest.find("<parameters>") {
        let params_inner = &rest[ps + "<parameters>".len()..];
        let pe = params_inner
            .find("</parameters>")
            .unwrap_or(params_inner.len());
        extract_params_from_block(&params_inner[..pe], &mut args);
    }

    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall {
            name: func_name,
            arguments: serde_json::to_string(&serde_json::Value::Object(args))
                .unwrap_or_else(|_| "{}".into()),
        },
    })
}

/// Extract text content between `<tag>` and `</tag>`.
fn extract_tag_value(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = start + text[start..].find(&close)?;
    let val = text[start..end].trim();
    if val.is_empty() {
        None
    } else {
        Some(val.to_string())
    }
}

/// Extract parameter name-value pairs from a `<parameters>` block.
///
/// Handles two layouts:
/// - Direct: `<name>N</name>...<value>V</value>` (single param at top level)
/// - Nested: `<parameter><name>N</name><value>V</value></parameter>` (multiple params)
fn extract_params_from_block(block: &str, args: &mut serde_json::Map<String, serde_json::Value>) {
    // Case 1: <parameter> children, each with <name> and <value>
    let mut rest = block;
    let mut found_any = false;
    while let Some(ps) = rest.find("<parameter>") {
        rest = &rest[ps + "<parameter>".len()..];
        let pe = rest.find("</parameter>").unwrap_or(rest.len());
        let param_block = &rest[..pe];
        if let (Some(name), Some(value)) = (
            extract_tag_value(param_block, "name"),
            extract_tag_value(param_block, "value"),
        ) {
            let json_val = serde_json::from_str::<serde_json::Value>(&value)
                .unwrap_or(serde_json::Value::String(value));
            args.insert(name, json_val);
            found_any = true;
        }
        rest = if pe < rest.len() {
            &rest[pe + "</parameter>".len()..]
        } else {
            ""
        };
    }
    if found_any {
        return;
    }

    // Case 2: Direct <name>/<value> pair at top level (single parameter)
    if let (Some(name), Some(value)) = (
        extract_tag_value(block, "name"),
        extract_tag_value(block, "value"),
    ) {
        let json_val = serde_json::from_str::<serde_json::Value>(&value)
            .unwrap_or(serde_json::Value::String(value));
        args.insert(name, json_val);
    }
}

/// Find the end of a bare function block for streaming detection.
/// Returns byte offset past the last closing tag, or `None` if incomplete.
pub(super) fn bare_function_end(text: &str) -> Option<usize> {
    if text.starts_with("<function>") {
        // Tag-style: <function>NAME</function>...<parameters>...</parameters>
        // End marker is </parameters> if present, else last </function>
        if let Some(p) = text.find("</parameters>") {
            return Some(p + "</parameters>".len());
        }
        // No parameters block — just <function>NAME</function>
        if let Some(p) = text.find("</function>") {
            let after = p + "</function>".len();
            // Check if there's a trailing </function> (some models emit an extra one)
            if let Some(p2) = text[after..].find("</function>") {
                return Some(after + p2 + "</function>".len());
            }
            return Some(after);
        }
    } else if text.starts_with("<function=") || text.starts_with("<function ") {
        // Attribute-style: <function=NAME> or <function NAME> (lenient)
        if let Some(p) = text.find("</function>") {
            return Some(p + "</function>".len());
        }
    }
    None
}
