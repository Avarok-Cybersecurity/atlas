// SPDX-License-Identifier: AGPL-3.0-only

use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct IncomingMessage {
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_message_content")]
    pub content: ParsedContent,
    /// Tool calls from a previous assistant message (multi-turn tool conversations).
    #[serde(default)]
    pub tool_calls: Option<Vec<crate::tool_parser::IncomingToolCall>>,
    /// ID of the tool call this message is responding to (role="tool").
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Function name for tool response messages.
    #[serde(default)]
    pub name: Option<String>,
}

/// Content extracted from a message — text and any base64-encoded images.
#[derive(Debug, Clone, Default)]
pub struct ParsedContent {
    pub text: String,
    /// Base64 data URIs: `"data:image/jpeg;base64,..."` or raw base64 strings.
    pub images: Vec<String>,
}

impl IncomingMessage {
    /// Build a synthetic system message (used by the Responses adapter to
    /// carry `instructions` into the chat-completions pipeline).
    pub fn synthetic_system(text: String) -> Self {
        Self {
            role: "system".to_string(),
            content: ParsedContent {
                text,
                images: Vec::new(),
            },
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Build a synthetic user message (used by the Responses adapter when
    /// `input` is a plain string).
    pub fn synthetic_user_text(text: String) -> Self {
        Self {
            role: "user".to_string(),
            content: ParsedContent {
                text,
                images: Vec::new(),
            },
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Translate a Responses-API `input` array item into a chat-completions
    /// message. Returns `None` for items the adapter doesn't understand (they
    /// are silently skipped so the request still runs).
    pub fn from_responses_input_item(v: &serde_json::Value) -> Option<Self> {
        let obj = v.as_object()?;
        let kind = obj
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("message");
        match kind {
            "message" => {
                let role = obj
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("user")
                    .to_string();
                let content_val = obj.get("content")?;
                let mut text = String::new();
                match content_val {
                    serde_json::Value::String(s) => text.push_str(s),
                    serde_json::Value::Array(parts) => {
                        for part in parts {
                            if let Some(po) = part.as_object() {
                                let part_kind =
                                    po.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                if matches!(part_kind, "input_text" | "output_text" | "text")
                                    && let Some(t) = po.get("text").and_then(|t| t.as_str())
                                {
                                    text.push_str(t);
                                }
                            }
                        }
                    }
                    _ => {}
                }
                Some(Self {
                    role,
                    content: ParsedContent {
                        text,
                        images: Vec::new(),
                    },
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                })
            }
            // Replay of a prior assistant function_call in the input chain.
            // Surface as an `assistant`-role message carrying the
            // structured tool_calls so the chat template can re-emit it
            // and the model sees its own prior call when paired with
            // the matching function_call_output below.
            "function_call" => {
                let name = obj.get("name").and_then(|v| v.as_str())?.to_string();
                let arguments = obj
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}")
                    .to_string();
                let call_id = obj
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| obj.get("id").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();
                Some(Self {
                    role: "assistant".to_string(),
                    content: ParsedContent::default(),
                    tool_calls: Some(vec![crate::tool_parser::IncomingToolCall {
                        id: Some(call_id),
                        function: crate::tool_parser::IncomingFunction { name, arguments },
                    }]),
                    tool_call_id: None,
                    name: None,
                })
            }
            // Tool-execution result the client sends back so the model
            // sees what its prior function_call returned. Without this
            // case multi-turn tool conversations fail: the model never
            // sees its tool's output and re-issues the same call.
            "function_call_output" => {
                let call_id = obj
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let output_text = match obj.get("output") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                    None => String::new(),
                };
                let name = obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Some(Self {
                    role: "tool".to_string(),
                    content: ParsedContent {
                        text: output_text,
                        images: Vec::new(),
                    },
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                    name: if name.is_empty() { None } else { Some(name) },
                })
            }
            // Reasoning items (Responses-API `type:"reasoning"`) are
            // intentionally NOT re-fed to the model — OpenAI's spec
            // treats `reasoning.encrypted_content` as opaque and
            // re-feeding poisons the next turn with stale internal
            // thoughts. Drop silently.
            "reasoning" => None,
            _ => None,
        }
    }
}

fn deserialize_message_content<'de, D>(d: D) -> Result<ParsedContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RawContent {
        Str(String),
        Parts(Vec<ContentPart>),
        Null(()),
    }

    #[derive(Deserialize)]
    struct ContentPart {
        #[serde(rename = "type")]
        kind: String,
        text: Option<String>,
        image_url: Option<ImageUrl>,
    }

    #[derive(Deserialize)]
    struct ImageUrl {
        url: String,
    }

    let mut out = ParsedContent::default();
    match RawContent::deserialize(d)? {
        RawContent::Str(s) => out.text = s,
        RawContent::Null(()) => {}
        RawContent::Parts(parts) => {
            let mut text_parts = Vec::new();
            for p in parts {
                match p.kind.as_str() {
                    "text" => {
                        if let Some(t) = p.text {
                            text_parts.push(t);
                        }
                    }
                    "image_url" => {
                        if let Some(iu) = p.image_url {
                            out.images.push(iu.url);
                        }
                    }
                    _ => {} // ignore unknown part types
                }
            }
            out.text = text_parts.join("");
        }
    }
    Ok(out)
}
