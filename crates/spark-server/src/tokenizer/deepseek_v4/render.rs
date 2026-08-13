// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::{
    ASSISTANT, BOS, DSML, EOS, LATEST_REMINDER, ReasoningEffort, THINK_END, THINK_START, USER,
};

const TOOL_INSTRUCTIONS: &str = r##"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete reasoning inside {think_start}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"##;

pub(super) fn render_messages(
    messages: &[Value],
    thinking: bool,
    drop_thinking: bool,
    effort: ReasoningEffort,
) -> Result<String> {
    let last_user = messages
        .iter()
        .rposition(|m| matches!(role(m), Some("user" | "developer")))
        .unwrap_or(usize::MAX);
    let mut prompt = String::from(BOS);
    for index in 0..messages.len() {
        prompt.push_str(&render_message(
            index,
            messages,
            thinking,
            drop_thinking,
            effort,
            last_user,
        )?);
    }
    Ok(prompt)
}

fn render_message(
    index: usize,
    messages: &[Value],
    thinking: bool,
    drop_thinking: bool,
    effort: ReasoningEffort,
    last_user: usize,
) -> Result<String> {
    let message = &messages[index];
    let mut out = String::new();
    if index == 0 && thinking {
        out.push_str(effort.prefix());
    }

    let content = content_text(message.get("content"))?;
    match role(message).context("DeepSeek-V4 message is missing role")? {
        "system" => {
            out.push_str(&content);
            append_tools_and_format(&mut out, message)?;
        }
        "developer" => {
            if content.is_empty() {
                bail!("DeepSeek-V4 developer message requires content");
            }
            out.push_str(USER);
            out.push_str(&content);
            append_tools_and_format(&mut out, message)?;
        }
        "user" => {
            out.push_str(USER);
            if let Some(blocks) = message.get("content_blocks").and_then(Value::as_array) {
                let rendered = blocks
                    .iter()
                    .map(render_content_block)
                    .collect::<Result<Vec<_>>>()?;
                out.push_str(&rendered.join("\n\n"));
            } else {
                out.push_str(&content);
            }
        }
        "latest_reminder" => {
            out.push_str(LATEST_REMINDER);
            out.push_str(&content);
        }
        "tool" => bail!("DeepSeek-V4 tool messages must be merged before rendering"),
        "assistant" => {
            let previous_has_task = index > 0 && messages[index - 1].get("task").is_some();
            if thinking && !previous_has_task && (!drop_thinking || index > last_user) {
                out.push_str(
                    message
                        .get("reasoning_content")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
                out.push_str(THINK_END);
            }
            out.push_str(&content);
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array)
                && !calls.is_empty()
            {
                out.push_str("\n\n");
                out.push_str(&render_tool_calls(calls)?);
            }
            if !message
                .get("wo_eos")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                out.push_str(EOS);
            }
        }
        other => bail!("unsupported DeepSeek-V4 message role '{other}'"),
    }

    let next_role = messages.get(index + 1).and_then(role);
    if index + 1 < messages.len() && !matches!(next_role, Some("assistant" | "latest_reminder")) {
        return Ok(out);
    }

    if let Some(task) = message.get("task").and_then(Value::as_str) {
        let task_token = match task {
            "action" => "<｜action｜>",
            "query" => "<｜query｜>",
            "authority" => "<｜authority｜>",
            "domain" => "<｜domain｜>",
            "title" => "<｜title｜>",
            "read_url" => "<｜read_url｜>",
            other => bail!("invalid DeepSeek-V4 task '{other}'"),
        };
        if task == "action" {
            out.push_str(ASSISTANT);
            out.push_str(if thinking { THINK_START } else { THINK_END });
        }
        out.push_str(task_token);
    } else if matches!(role(message), Some("user" | "developer")) {
        out.push_str(ASSISTANT);
        out.push_str(if thinking && (!drop_thinking || index >= last_user) {
            THINK_START
        } else {
            THINK_END
        });
    }
    Ok(out)
}

fn append_tools_and_format(out: &mut String, message: &Value) -> Result<()> {
    if let Some(tools) = message.get("tools").and_then(Value::as_array)
        && !tools.is_empty()
    {
        out.push_str("\n\n");
        out.push_str(&render_tools(tools)?);
    }
    if let Some(response_format) = message.get("response_format") {
        out.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
        out.push_str(&python_json(response_format)?);
    }
    Ok(())
}

fn render_tools(tools: &[Value]) -> Result<String> {
    let schemas = tools
        .iter()
        .map(|tool| {
            let function = tool
                .get("function")
                .context("DeepSeek-V4 tools must use OpenAI function format")?;
            python_json(function)
        })
        .collect::<Result<Vec<_>>>()?
        .join("\n");
    Ok(TOOL_INSTRUCTIONS
        .replace("{dsml}", DSML)
        .replace("{think_start}", THINK_START)
        .replace("{think_end}", THINK_END)
        .replace("{schemas}", &schemas))
}

fn render_tool_calls(calls: &[Value]) -> Result<String> {
    let mut invokes = Vec::with_capacity(calls.len());
    for call in calls {
        let function = call.get("function").unwrap_or(call);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .context("DeepSeek-V4 tool call is missing function.name")?;
        let arguments = function
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));
        let arguments = match arguments {
            Value::String(raw) => {
                serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({"arguments": raw}))
            }
            other => other,
        };
        let object = arguments
            .as_object()
            .context("DeepSeek-V4 tool arguments must be a JSON object")?;
        let mut params = Vec::with_capacity(object.len());
        for (key, value) in object {
            let (is_string, rendered) = match value {
                Value::String(s) => ("true", s.clone()),
                other => ("false", python_json(other)?),
            };
            params.push(format!(
                "<{DSML}parameter name=\"{key}\" string=\"{is_string}\">{rendered}</{DSML}parameter>"
            ));
        }
        invokes.push(format!(
            "<{DSML}invoke name=\"{name}\">\n{}\n</{DSML}invoke>",
            params.join("\n")
        ));
    }
    Ok(format!(
        "<{DSML}tool_calls>\n{}\n</{DSML}tool_calls>",
        invokes.join("\n")
    ))
}

fn render_content_block(block: &Value) -> Result<String> {
    match block.get("type").and_then(Value::as_str) {
        Some("text") => content_text(block.get("text")),
        Some("tool_result") => {
            let content = match block.get("content") {
                Some(Value::Array(parts)) => parts
                    .iter()
                    .map(|part| match part.get("type").and_then(Value::as_str) {
                        Some("text") => Ok(part
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string()),
                        Some(kind) => Ok(format!("[Unsupported {kind}]")),
                        None => Ok("[Unsupported unknown]".to_string()),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .join("\n\n"),
                other => content_text(other)?,
            };
            Ok(format!("<tool_result>{content}</tool_result>"))
        }
        Some(kind) => Ok(format!("[Unsupported {kind}]")),
        None => Ok("[Unsupported unknown]".to_string()),
    }
}

fn content_text(content: Option<&Value>) -> Result<String> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| {
                        (part.get("image").is_some() || part.get("image_url").is_some())
                            .then(|| "[Image]".to_string())
                    })
                    .context("unexpected DeepSeek-V4 content item")
            })
            .collect::<Result<Vec<_>>>()
            .map(|parts| parts.join("")),
        Some(_) => bail!("unexpected DeepSeek-V4 content type"),
    }
}

fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

pub(super) fn python_json(value: &Value) -> Result<String> {
    Ok(match value {
        Value::Null => "null".to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => serde_json::to_string(v)?,
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_json)
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| Ok(format!(
                    "{}: {}",
                    serde_json::to_string(key)?,
                    python_json(value)?
                )))
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        ),
    })
}
