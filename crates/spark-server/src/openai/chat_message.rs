// SPDX-License-Identifier: AGPL-3.0-only

use serde::Deserialize;

// The modality tag is the IR's, not a wire-local copy: the whole point of
// carrying media as one tagged sequence is that the tag and the order
// survive unchanged from the wire to the rendered prompt, and two parallel
// enums would be two places for that meaning to drift.
pub use crate::ir::MediaKind;

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
    /// Historical reasoning trace from a prior assistant turn (Qwen3
    /// `<think>...</think>` body). Clients (vLLM/SGLang/opencode) round-trip
    /// this field so the chat template can rehydrate the historical
    /// `<think>` block. Without it the template emits empty
    /// `<think>\n\n</think>\n\n` wrappers for every historical assistant
    /// turn → empty-think poisoning → premature `<|im_end|>` abort.
    /// Accepts both `reasoning_content` (DeepSeek/vLLM/LiteLLM standard)
    /// and the shorter `reasoning` alias used by some OpenAI SDK versions.
    #[serde(default, alias = "reasoning")]
    pub reasoning_content: Option<String>,
}

/// Content extracted from a message — the flattened text, plus every media
/// item **in the order the client sent it**.
#[derive(Debug, Clone, Default)]
pub struct ParsedContent {
    pub text: String,
    /// Images and videos as ONE tagged sequence. They were previously two
    /// lists, which silently discarded their relative order: a request
    /// sending `video_url` then `image_url` reached the template as
    /// image-then-video, so the model was shown the items in an order the
    /// caller never wrote and any prompt referring to "the first" one
    /// described something else. Nothing errored, because the pad runs and
    /// the encoder rows still agreed with each other.
    pub media: Vec<MediaRef>,
}

/// One media item from the wire: what it is, and where its bytes come from
/// (a `data:` URI, a raw base64 string, or a remote URL resolved later).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaRef {
    pub kind: MediaKind,
    pub uri: String,
}

impl ParsedContent {
    /// A message carrying only text — the shape every synthetic and
    /// replayed message has.
    pub fn text_only(text: String) -> Self {
        ParsedContent {
            text,
            media: Vec::new(),
        }
    }

    /// The image URIs, in order. The stored-conversation writers replay
    /// images only (videos are not persisted), so they read this view
    /// rather than filtering `media` themselves.
    pub fn images(&self) -> impl Iterator<Item = &String> {
        self.media
            .iter()
            .filter(|m| m.kind == MediaKind::Image)
            .map(|m| &m.uri)
    }

    /// True when this message carries at least one image.
    pub fn has_images(&self) -> bool {
        self.images().next().is_some()
    }
}

impl IncomingMessage {
    /// Build a synthetic system message (used by the Responses adapter to
    /// carry `instructions` into the chat-completions pipeline).
    pub fn synthetic_system(text: String) -> Self {
        Self {
            role: "system".to_string(),
            content: ParsedContent::text_only(text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    /// Build a synthetic user message (used by the Responses adapter when
    /// `input` is a plain string).
    pub fn synthetic_user_text(text: String) -> Self {
        Self {
            role: "user".to_string(),
            content: ParsedContent::text_only(text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    /// Convert a stored conversation item into a message for pipeline
    /// replay. Items we don't recognize (tool outputs in exotic shapes)
    /// are silently dropped — they wouldn't contribute to the text
    /// context anyway. `reasoning_content` (written by the Responses
    /// surfaces alongside the assistant text) is rehydrated so the
    /// template can restore the prior turn's think block (F1).
    pub fn from_conversation_item(item: &serde_json::Value) -> Option<Self> {
        let role = item.get("role").and_then(|v| v.as_str())?;
        let content = item.get("content");
        let text = match content {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| {
                    p.get("text")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        let reasoning_content = item
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        Some(Self {
            role: role.to_string(),
            content: ParsedContent::text_only(text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content,
        })
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
                let role = match obj.get("role").and_then(|r| r.as_str()).unwrap_or("user") {
                    // OpenAI Responses API may use `developer`, but most
                    // model chat templates only understand system/user/assistant/tool.
                    // Map `developer` to `user` instead of `system`, because some
                    // templates require system messages to appear only at the beginning
                    // of the conversation.
                    "developer" => "user",
                    other => other,
                }
                .to_string();
                let content_val = obj.get("content")?;
                Some(Self {
                    role,
                    content: ParsedContent::from_responses_content(content_val),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    reasoning_content: None,
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
                    reasoning_content: None,
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
                let output = match obj.get("output") {
                    Some(serde_json::Value::String(s)) => ParsedContent::text_only(s.clone()),
                    // Structured output parts (`output_text` / `input_image` /
                    // …): carry text AND images so a screenshot returned by a
                    // tool reaches the vision encoder — parity with the
                    // Anthropic tool_result path and chat-completions
                    // `role:"tool"` array content (#165). Arrays that contain
                    // no recognizable parts keep the old stringified-JSON
                    // behavior so out-of-spec payloads still reach the model.
                    Some(arr @ serde_json::Value::Array(_)) => {
                        let parsed = ParsedContent::from_responses_content(arr);
                        if parsed.text.is_empty() && parsed.media.is_empty() {
                            ParsedContent::text_only(arr.to_string())
                        } else {
                            parsed
                        }
                    }
                    Some(other) => ParsedContent::text_only(other.to_string()),
                    None => ParsedContent::default(),
                };
                let name = obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Some(Self {
                    role: "tool".to_string(),
                    content: output,
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                    name: if name.is_empty() { None } else { Some(name) },
                    reasoning_content: None,
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

impl ParsedContent {
    /// Flatten a Responses-API content value (string, or array of
    /// `input_text`/`output_text`/`input_image`/… parts) into text + an
    /// ordered media list. Shared by `message` items and
    /// `function_call_output` items so images are carried on both — the
    /// pipeline collects them into the vision encoder.
    ///
    /// Media parts append to ONE list in the order they appear, so a
    /// video sent before an image stays before it all the way to the
    /// rendered prompt.
    fn from_responses_content(v: &serde_json::Value) -> Self {
        let mut text = String::new();
        let mut media: Vec<MediaRef> = Vec::new();
        match v {
            serde_json::Value::String(s) => text.push_str(s),
            serde_json::Value::Array(parts) => {
                for part in parts {
                    if let Some(po) = part.as_object() {
                        let part_kind = po.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if matches!(part_kind, "input_text" | "output_text" | "text")
                            && let Some(t) = po.get("text").and_then(|t| t.as_str())
                        {
                            text.push_str(t);
                        } else if matches!(part_kind, "input_image" | "image_url" | "image")
                            && let Some(url) = responses_image_url(po)
                        {
                            media.push(MediaRef {
                                kind: MediaKind::Image,
                                uri: url,
                            });
                        } else if matches!(part_kind, "input_video" | "video_url" | "video")
                            && let Some(url) = responses_video_url(po)
                        {
                            media.push(MediaRef {
                                kind: MediaKind::Video,
                                uri: url,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
        ParsedContent { text, media }
    }
}

/// Extract the image URL / data-URI from a Responses `input_image`
/// content part. Accepts both the flat string form
/// (`"image_url": "..."`) and the nested object form
/// (`"image_url": {"url": "..."}`).
fn responses_image_url(po: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    match po.get("image_url") {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Object(o)) => {
            o.get("url").and_then(|v| v.as_str()).map(|s| s.to_string())
        }
        _ => None,
    }
}

/// Extract the video URL / data-URI from a Responses `input_video` part.
/// Accepts the flat string form and the nested object form, exactly like
/// [`responses_image_url`] — the shapes are the same and clients reuse them.
fn responses_video_url(po: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    for key in ["video_url", "video"] {
        match po.get(key) {
            Some(serde_json::Value::String(s)) => return Some(s.clone()),
            Some(serde_json::Value::Object(o)) => {
                if let Some(u) = o.get("url").and_then(|v| v.as_str()) {
                    return Some(u.to_string());
                }
            }
            _ => {}
        }
    }
    None
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
        /// `{"type": "video_url", "video_url": {"url": "..."}}` — the shape
        /// vLLM and Qwen's own examples use. `Url` is reused because the
        /// wire shape is identical to `image_url`.
        video_url: Option<Url>,
        /// `{"type": "video", "video": "..."}`, the flat spelling some
        /// clients emit. Untagged so either form deserialises.
        video: Option<UrlOrString>,
    }

    #[derive(Deserialize)]
    struct ImageUrl {
        url: String,
    }

    #[derive(Deserialize)]
    struct Url {
        url: String,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum UrlOrString {
        Obj { url: String },
        Str(String),
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
                            out.media.push(MediaRef {
                                kind: MediaKind::Image,
                                uri: iu.url,
                            });
                        }
                    }
                    // Both spellings, because both are in the wild: OpenAI-style
                    // `video_url` objects and Qwen's flat `video`. Until this
                    // existed, a video part matched the catch-all below and was
                    // DROPPED without a word, so the model answered from the
                    // surrounding text as though no video had been sent.
                    "video_url" | "video" | "input_video" => {
                        let uri = if let Some(v) = p.video_url {
                            Some(v.url)
                        } else {
                            p.video.map(|v| match v {
                                UrlOrString::Obj { url } => url,
                                UrlOrString::Str(s) => s,
                            })
                        };
                        // Appended to the SAME list as images, at the
                        // position it arrived in — the ordering the whole
                        // vision path now preserves.
                        if let Some(uri) = uri {
                            out.media.push(MediaRef {
                                kind: MediaKind::Video,
                                uri,
                            });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_item_round_trips_reasoning_content() {
        let item = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer"}],
            "reasoning_content": "thought",
        });
        let m = IncomingMessage::from_conversation_item(&item).expect("assistant item");
        assert_eq!(m.content.text, "answer");
        assert_eq!(m.reasoning_content.as_deref(), Some("thought"));
    }

    #[test]
    fn conversation_item_without_reasoning_stays_none() {
        let item = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": "answer",
        });
        let m = IncomingMessage::from_conversation_item(&item).expect("assistant item");
        assert_eq!(m.content.text, "answer");
        assert_eq!(m.reasoning_content, None);
    }
}

#[cfg(test)]
#[path = "chat_message_video_tests.rs"]
mod video_wire_tests;
