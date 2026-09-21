// SPDX-License-Identifier: AGPL-3.0-only

//! Chat template from GGUF metadata (`tokenizer.chat_template`).

use anyhow::Result;
use std::path::Path;

use super::jinja_helpers::convert_python_jinja_to_minijinja;

pub(super) fn load(model_dir: &Path) -> Result<Option<String>> {
    match spark_runtime::weights::gguf_chat_template(model_dir) {
        Ok(Some(raw)) => {
            let converted = convert_python_jinja_to_minijinja(&raw);
            tracing::info!(
                "Loaded Jinja chat template from GGUF tokenizer.chat_template ({} chars)",
                converted.len()
            );
            Ok(Some(converted))
        }
        Ok(None) => Ok(None),
        Err(e) => {
            tracing::warn!("GGUF tokenizer.chat_template unreadable: {e:#}");
            Ok(None)
        }
    }
}
