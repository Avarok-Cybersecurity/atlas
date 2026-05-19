// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// GLM-4.7 native tool call format.
///
/// Outer tag: `<tool_call>...</tool_call>`.
///
/// Inner format — function name immediately after the open tag, then
/// alternating `<arg_key>`/`<arg_value>` pairs:
///
/// ```text
/// <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value>
/// <arg_key>key2</arg_key><arg_value>value2</arg_value></tool_call>
/// ```
///
/// This matches the format emitted by the official GLM jinja template
/// (used by both the upstream vLLM `glm47` parser and GLM's own
/// tokenizer_config.json).
pub struct Glm4Parser;

impl ToolCallParser for Glm4Parser {
    fn name(&self) -> &str {
        "glm4"
    }

    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
        let mut prompt = String::from(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        );
        for tool in tools {
            let json = serde_json::to_string(tool).unwrap_or_default();
            prompt.push_str(&json);
            prompt.push('\n');
        }
        prompt.push_str("</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n");
        prompt.push_str("<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str("<tool_call>");
            out.push_str(&tc.function.name);
            // GLM-4.7 tokenizer requires a newline between function name and
            // first arg_key — matches the jinja template:
            //   {{ '<tool_call>' + tc.name }}\n{% for k,v %}<arg_key>k</arg_key>...
            out.push('\n');
            if let Some(obj) = args.as_object() {
                for (key, val) in obj {
                    let val_str = match val {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    out.push_str("<arg_key>");
                    out.push_str(key);
                    out.push_str("</arg_key><arg_value>");
                    out.push_str(&val_str);
                    out.push_str("</arg_value>");
                }
            }
            out.push_str("</tool_call>");
        }
        out
    }

    fn leak_markers(&self) -> LeakMarkers {
        const MARKERS: LeakMarkers = LeakMarkers {
            orphan_open: &["<arg_key>", "<arg_value>"],
            close: &["</arg_key>", "</arg_value>", "</tool_call>"],
            envelope_open: &["<tool_call>"],
            envelope_close: &["</tool_call>"],
        };
        MARKERS
    }
}

/// Parse GLM-4.7 inner tool call content:
/// `function_name<arg_key>key</arg_key><arg_value>value</arg_value>...`
///
/// The function name is everything before the first `<arg_key>` (or the
/// entire string if there are no argument tags, indicating a zero-arg call).
pub(super) fn parse_glm4_call(text: &str) -> Option<ToolCall> {
    let (func_name, rest) = match text.find("<arg_key>") {
        Some(pos) => {
            let name = text[..pos].trim();
            if name.is_empty() {
                return None;
            }
            (name.to_string(), &text[pos..])
        }
        None => {
            // No arguments — whole text is the function name.
            let name = text.trim();
            if name.is_empty() {
                return None;
            }
            return Some(ToolCall {
                id: next_tool_call_id(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: "{}".into(),
                },
            });
        }
    };

    let mut args = serde_json::Map::new();
    let mut cursor = rest;
    while let Some(key_start) = cursor.find("<arg_key>") {
        cursor = &cursor[key_start + "<arg_key>".len()..];
        let key_end = cursor.find("</arg_key>").unwrap_or(cursor.len());
        let key = cursor[..key_end].trim().to_string();
        cursor = if key_end + "</arg_key>".len() <= cursor.len() {
            &cursor[key_end + "</arg_key>".len()..]
        } else {
            ""
        };

        // Find the matching <arg_value>...</arg_value> immediately after.
        let val_str = if let Some(v_start) = cursor.find("<arg_value>") {
            cursor = &cursor[v_start + "<arg_value>".len()..];
            let v_end = cursor.find("</arg_value>").unwrap_or(cursor.len());
            let v = cursor[..v_end].trim().to_string();
            cursor = if v_end + "</arg_value>".len() <= cursor.len() {
                &cursor[v_end + "</arg_value>".len()..]
            } else {
                ""
            };
            v
        } else {
            String::new()
        };

        if !key.is_empty() {
            args.insert(key, serde_json::Value::String(val_str));
        }
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
