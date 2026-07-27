// SPDX-License-Identifier: AGPL-3.0-only
//
// Echo-only response context. These wire fields never influence
// generation — they are read back verbatim when the response is
// encoded (service_tier/metadata echo, `store: true` persistence,
// include_usage stream framing). They ride beside the IR envelope
// instead of inside it so the IR carries exactly what the pipeline
// computes with (PCND). Absorbed into the per-surface encoders once
// the response direction is fully IR-based.

/// Encode-time echoes captured from the surface's wire request.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResponseEcho {
    pub(crate) service_tier: Option<String>,
    pub(crate) metadata: Option<std::collections::HashMap<String, String>>,
    /// Persist the completion for later retrieval (`store: true`).
    pub(crate) store: bool,
    /// Emit the final usage-only chunk before `[DONE]`
    /// (`stream_options.include_usage`).
    pub(crate) include_usage: bool,
}

impl<'a> From<&'a crate::openai::ChatCompletionRequest> for ResponseEcho {
    /// Capture the echo fields from a parsed OpenAI wire request.
    fn from(req: &'a crate::openai::ChatCompletionRequest) -> Self {
        ResponseEcho {
            service_tier: req.service_tier.clone(),
            metadata: req.metadata.clone(),
            store: req.store.unwrap_or(false),
            include_usage: req.stream_options.map(|o| o.include_usage).unwrap_or(false),
        }
    }
}
