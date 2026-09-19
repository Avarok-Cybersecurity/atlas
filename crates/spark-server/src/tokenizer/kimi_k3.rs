// SPDX-License-Identifier: AGPL-3.0-only

//! Official K3 ships a segmented XTML encoder, not the twin's Jinja contract.
//! Until that encoder and its output parsers are integrated, raw completions
//! remain available but chat must never silently substitute generic ChatML.

use anyhow::{Context, Result, bail};
use std::path::Path;

pub(super) fn uses_xtml(model_dir: &Path, model_type: &str) -> Result<bool> {
    if model_type != "kimi_k3" {
        return Ok(false);
    }
    if model_dir.join("tiktoken.model").exists() {
        return Ok(true);
    }
    let config = model_dir.join("tokenizer_config.json");
    if !config.exists() {
        return Ok(false);
    }
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(config).context("Read Kimi tokenizer config")?)
            .context("Parse Kimi tokenizer config")?;
    Ok(value["tokenizer_class"].as_str() == Some("TikTokenTokenizer"))
}

pub(super) fn require_chat_support(encoding: super::ChatEncoding) -> Result<()> {
    if encoding == super::ChatEncoding::KimiK3XtmlUnsupported {
        bail!(
            "Kimi K3 XTML chat/reasoning/tool parsing is not implemented; \
             use /v1/completions with independently prepared token IDs for bring-up. \
             Generic ChatML and K2 tool parsers are not compatible."
        );
    }
    Ok(())
}
