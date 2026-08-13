// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

pub(super) const DSML_OPEN: &str = "<｜DSML｜tool_calls>";
pub(super) const DSML_CLOSE: &str = "</｜DSML｜tool_calls>";
const INVOKE_OPEN: &str = "<｜DSML｜invoke name=\"";
const INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
const PARAM_OPEN: &str = "<｜DSML｜parameter name=\"";
const PARAM_CLOSE: &str = "</｜DSML｜parameter>";

pub struct DeepseekV4DsmlParser;

impl ToolCallParser for DeepseekV4DsmlParser {
    fn name(&self) -> &str {
        "deepseek_v4"
    }

    fn system_prompt(
        &self,
        _tools: &[ToolDefinition],
        _tool_choice: &ToolChoice,
        _levers: &PromptLevers,
    ) -> String {
        // The checkpoint-native encoder renders the canonical DSML instructions
        // and schemas. A second generic parser prompt would change prompt bytes.
        String::new()
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut invokes = Vec::with_capacity(calls.len());
        for call in calls {
            let args = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
            let mut params = Vec::new();
            if let Some(object) = args.as_object() {
                for (key, value) in object {
                    let (string, value) = match value {
                        serde_json::Value::String(value) => ("true", value.clone()),
                        value => (
                            "false",
                            serde_json::to_string(value).unwrap_or_else(|_| "null".into()),
                        ),
                    };
                    params.push(format!(
                        "<{DSML}parameter name=\"{key}\" string=\"{string}\">{value}</{DSML}parameter>",
                        DSML = "｜DSML｜",
                    ));
                }
            }
            invokes.push(format!(
                "<｜DSML｜invoke name=\"{}\">\n{}\n</｜DSML｜invoke>",
                call.function.name,
                params.join("\n")
            ));
        }
        format!("{DSML_OPEN}\n{}\n{DSML_CLOSE}", invokes.join("\n"))
    }

    fn format_tool_response(&self, content: &str) -> String {
        format!("<tool_result>{content}</tool_result>")
    }

    fn leak_markers(&self) -> LeakMarkers {
        LeakMarkers {
            orphan_open: &[INVOKE_OPEN, PARAM_OPEN],
            close: &[PARAM_CLOSE, INVOKE_CLOSE, DSML_CLOSE],
            envelope_open: &[DSML_OPEN],
            envelope_close: &[DSML_CLOSE],
        }
    }

    fn wants_typed_arguments(&self) -> bool {
        true
    }
}

pub(super) fn parse_dsml_tool_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut content = Vec::new();
    let mut calls = Vec::new();
    let mut rest = text;
    loop {
        let Some(start) = rest.find(DSML_OPEN) else {
            let tail = rest.trim();
            if !tail.is_empty() {
                content.push(tail.to_string());
            }
            break;
        };
        let before = rest[..start].trim();
        if !before.is_empty() {
            content.push(before.to_string());
        }
        let body = &rest[start + DSML_OPEN.len()..];
        let Some(end) = body.find(DSML_CLOSE) else {
            content.push(rest[start..].to_string());
            break;
        };
        calls.extend(parse_dsml_invokes(&body[..end]));
        rest = &body[end + DSML_CLOSE.len()..];
    }
    let content = (!content.is_empty()).then(|| content.join("\n"));
    (content, calls)
}

pub(super) fn parse_dsml_invokes(body: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find(INVOKE_OPEN) {
        rest = &rest[start + INVOKE_OPEN.len()..];
        let Some(name_end) = rest.find("\">\n") else {
            break;
        };
        let name = &rest[..name_end];
        let after_header = &rest[name_end + 3..];
        let Some(invoke_end) = after_header.find(INVOKE_CLOSE) else {
            break;
        };
        let params = after_header[..invoke_end].trim_matches('\n');
        if let Some(arguments) = parse_dsml_parameters(params) {
            calls.push(ToolCall {
                id: next_tool_call_id(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".into()),
                },
            });
        }
        rest = &after_header[invoke_end + INVOKE_CLOSE.len()..];
    }
    calls
}

fn parse_dsml_parameters(params: &str) -> Option<serde_json::Value> {
    let mut object = serde_json::Map::new();
    let mut rest = params.trim();
    while !rest.is_empty() {
        rest = rest.strip_prefix(PARAM_OPEN)?;
        let name_end = rest.find("\" string=\"")?;
        let name = &rest[..name_end];
        rest = &rest[name_end + "\" string=\"".len()..];
        let string_end = rest.find("\">")?;
        let is_string = &rest[..string_end];
        rest = &rest[string_end + 2..];
        let value_end = rest.find(PARAM_CLOSE)?;
        let raw = &rest[..value_end];
        let value = match is_string {
            "true" => serde_json::Value::String(raw.to_string()),
            "false" => serde_json::from_str(raw).ok()?,
            _ => return None,
        };
        if object.insert(name.to_string(), value).is_some() {
            return None;
        }
        rest = rest[value_end + PARAM_CLOSE.len()..].trim_start_matches('\n');
    }
    Some(serde_json::Value::Object(object))
}

pub(super) enum DsmlStreamAction {
    NotDsml,
    Wait,
    Continue,
}

impl StreamingToolDetector {
    pub(super) fn process_dsml(&mut self, outputs: &mut Vec<DetectorOutput>) -> DsmlStreamAction {
        if self.inside_dsml {
            let Some(end) = self.buffer.find(DSML_CLOSE) else {
                return DsmlStreamAction::Wait;
            };
            let body = self.buffer[..end].to_string();
            self.buffer = self.buffer[end + DSML_CLOSE.len()..].to_string();
            self.inside_dsml = false;
            self.emit_dsml_calls(&body, outputs);
            return DsmlStreamAction::Continue;
        }
        let Some(start) = self.buffer.find(DSML_OPEN) else {
            return DsmlStreamAction::NotDsml;
        };
        if start > 0 {
            let content = &self.buffer[..start];
            if !content.trim().is_empty() {
                outputs.push(DetectorOutput::Content(content.to_string()));
            }
        }
        self.buffer = self.buffer[start + DSML_OPEN.len()..].to_string();
        self.inside_dsml = true;
        DsmlStreamAction::Continue
    }

    pub(super) fn flush_dsml(&mut self) -> Option<Vec<DetectorOutput>> {
        if !self.inside_dsml {
            return None;
        }
        let body = std::mem::take(&mut self.buffer);
        self.inside_dsml = false;
        let mut outputs = Vec::new();
        self.emit_dsml_calls(&body, &mut outputs);
        (!outputs.is_empty()).then_some(outputs)
    }

    fn emit_dsml_calls(&mut self, body: &str, outputs: &mut Vec<DetectorOutput>) {
        for call in parse_dsml_invokes(body) {
            let index = self.call_counter as usize;
            self.call_counter += 1;
            self.emitted_tool_calls = true;
            outputs.push(DetectorOutput::ToolCall(call, index));
        }
    }
}
