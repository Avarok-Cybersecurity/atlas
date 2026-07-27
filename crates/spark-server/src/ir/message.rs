// SPDX-License-Identifier: AGPL-3.0-only
//
// Canonical chat message IR (request direction). Each API surface
// converts its wire messages into `Vec<Message>`; `build_msg_entries`
// and the template renderer read only these types.

/// Message author role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    /// Any other wire role string (e.g. `"developer"`, `"function"`),
    /// preserved verbatim. The canonical roles above are matched by the
    /// pipeline; unknown roles fall through to the template unchanged
    /// (which is where today's "Unexpected message role" handling lives),
    /// so introducing the IR does not change their behavior.
    Other(String),
}

impl Role {
    /// Wire string used by the OpenAI/Anthropic envelopes and the Jinja
    /// templates (`"system" | "user" | "assistant" | "tool"`, or the
    /// preserved string for [`Role::Other`]).
    pub fn as_wire(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Other(s) => s,
        }
    }

    /// Parse a known wire role. Returns `None` for anything unknown —
    /// callers decide the fallback explicitly (PCND: no silent default).
    /// Use the [`From<&str>`] impl when an unknown role must be
    /// preserved rather than rejected.
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

impl From<&str> for Role {
    /// Map any wire role string to a `Role`, preserving unknown roles as
    /// [`Role::Other`] (lossless — the template still decides what to do
    /// with them).
    fn from(s: &str) -> Self {
        Self::from_wire(s).unwrap_or_else(|| Role::Other(s.to_string()))
    }
}

/// One canonical chat message. `content` is ALWAYS a list of parts for
/// every role, so text and images interleave uniformly and an image on
/// a tool result is not a special case.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    /// Ordered content parts. Text and images interleave.
    pub content: Vec<ContentPart>,
    /// Assistant-emitted tool calls (empty for other roles).
    pub tool_calls: Vec<ToolCall>,
    /// Links a `Tool` message back to the `ToolCall.id` it answers.
    pub tool_call_id: Option<String>,
    /// Tool/function name (tool messages; some surfaces echo it).
    pub name: Option<String>,
    /// First-class historical reasoning trace (a prior assistant
    /// `<think>` body) instead of a `<think>\n\n</think>` string prefix.
    pub reasoning: Option<Reasoning>,
    /// `Tool` role only: the tool result reported an error. The intended
    /// first-class home for Anthropic's `is_error` flag. NOT wired yet — the
    /// Anthropic path still emits the `[tool error]\n` text prefix; dropping
    /// that in favor of this field is gated on flipping
    /// `ChatCompletionRequest.messages` to the IR (issue #165 follow-up), so
    /// no adapter sets it to `true` today.
    pub tool_error: bool,
}

impl Message {
    /// A synthetic system message carrying only `text`. Used for
    /// server-injected prompts (tool-parser behavioral prompt).
    pub fn synthetic_system(text: String) -> Self {
        Message {
            role: Role::System,
            content: vec![ContentPart::Text(text)],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        }
    }

    /// Prepend `prefix` to the first text part (appending a new text
    /// part when none exists — matching the canonical `[image*, text]`
    /// part order). The flattened `text()` output equals
    /// `prefix + old_text()` either way. Used for system-prompt
    /// injection.
    pub fn prepend_text(&mut self, prefix: &str) {
        for part in &mut self.content {
            if let ContentPart::Text(t) = part {
                *t = format!("{prefix}{t}");
                return;
            }
        }
        self.content.push(ContentPart::Text(prefix.to_string()));
    }

    /// Concatenate the text parts in order (images skipped). Mirrors the
    /// historical `text_parts.join("")` flattening so downstream
    /// text-scanning logic (cwd hint, vacuous-system, error detection)
    /// is unchanged.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for part in &self.content {
            if let ContentPart::Text(t) = part {
                out.push_str(t);
            }
        }
        out
    }

    /// Number of image parts on this message (drives the Jinja
    /// vision-marker expansion).
    pub fn image_count(&self) -> usize {
        self.content
            .iter()
            .filter(|p| matches!(p, ContentPart::Image(_)))
            .count()
    }
}

/// A single piece of message content. Open for future modalities
/// (audio, video, file).
#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    Text(String),
    Image(ImageSource),
}

/// Where an image comes from. The encoder consumes the inner string
/// directly (see `spark_model::vision_preprocess::preprocess_image`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSource {
    pub data: ImageData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageData {
    /// A `data:image/...;base64,...` URI (or raw base64) — the exact
    /// string `preprocess_image` decodes.
    Base64(String),
    /// Remote URL. The encoder does not fetch URLs; the pipeline rejects
    /// these with an explicit 400 (`collect_message_images`) instead of
    /// feeding them to the base64 decoder and surfacing a confusing
    /// "base64 decode failed".
    Url(String),
}

impl ImageData {
    /// Classify a wire image string. http(s) URLs → [`ImageData::Url`];
    /// everything else (data: URIs, raw base64) → [`ImageData::Base64`].
    /// Every adapter uses this so a remote URL can never masquerade as
    /// decodable base64.
    pub fn from_uri(s: String) -> Self {
        if s.starts_with("http://") || s.starts_with("https://") {
            ImageData::Url(s)
        } else {
            ImageData::Base64(s)
        }
    }
}

/// Assistant-emitted tool call. `arguments` is structured JSON (already
/// parsed), not a string.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// First-class reasoning/thinking trace.
#[derive(Debug, Clone, PartialEq)]
pub struct Reasoning {
    pub text: String,
}
