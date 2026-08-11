// SPDX-License-Identifier: AGPL-3.0-only

//! `FromStr` for [`ToolCallFormat`] - the CLI/MODEL.toml parser-name mapping.
//! Split out of `tool_parser.rs` (<=500 LoC cap).

use super::ToolCallFormat;

impl std::str::FromStr for ToolCallFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hermes" => Ok(Self::Hermes),
            "qwen3_coder" => Ok(Self::Qwen3Coder),
            "qwen3_xml" => Ok(Self::Qwen3Xml),
            "gemma4" => Ok(Self::Gemma4),
            "mistral" => Ok(Self::Mistral),
            "minimax_xml" => Ok(Self::MinimaxXml),
            "deepseek_v4" | "dsml" => Ok(Self::DeepseekV4),
            "bare_json" => Ok(Self::BareJson),
            "poolside_v1" => Ok(Self::PoolsideV1),
            other => Err(format!(
                "Unknown tool call parser '{other}'. Supported: hermes, qwen3_coder, qwen3_xml, gemma4, mistral, minimax_xml, deepseek_v4, bare_json, poolside_v1",
            )),
        }
    }
}
