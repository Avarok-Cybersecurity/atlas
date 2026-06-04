// SPDX-License-Identifier: AGPL-3.0-only
//
// TSCG — Tool-Schema Compilation for Generation.
//
// Deterministic compiler that rewrites OpenAI-style JSON tool schemas
// into a compact function-signature text form before they are placed in
// the prompt. Port of "TSCG: Deterministic Tool-Schema Compilation for
// Agentic LLM Deployments" (arXiv:2605.04107, May 2026).
//
// The paper defines eight composable operators. This port implements the
// six with measured cross-model benefit and deliberately omits two, per
// the paper's own ablation:
//
//   SDM   Semantic Density Maximization — strip filler from prose      [sdm.rs]
//   TAS   Tokenizer-Aligned Syntax — single-BPE-token delimiters       [render.rs, by construction]
//   DRO   Delimiter-Role Optimization — type/phrase compaction         [render.rs]
//   CFO   Causal-Forward Ordering — required params before optional    [render.rs]
//   CAS   Causal Access Score — high-fragility atoms front-loaded      [render.rs, name+required first]
//   SAD-F Selective Anchor Duplication — recap required atoms          [render.rs, conditional]
//   CFL   Constraint-First Layout — OMITTED: paper shows it is
//         counterproductive at >=43 tools, and agentic-coding sessions
//         routinely exceed that.
//   CCP   Causal Closure Principle — OMITTED: paper measures
//         ~85-306 tokens of overhead with no accuracy benefit on any
//         model except Opus 4.7.
//
// TSCG never touches `req.tools`: XGrammar still compiles its
// output-constraining grammar from the original JSON schema. TSCG only
// changes the *prompt copy* the model reads (SSOT preserved).
//
// Gated per-model by MODEL.toml `[behavior].tscg` (default false). The
// TAS operator picks delimiters from the model's BPE merges, so a gain
// on one tokenizer does not imply one on another — enable + verify with
// `tool-eval-bench` per model.

mod render;
mod sdm;

use crate::tool_parser::ToolDefinition;

/// Boot-time TSCG enable flag, set once from the resolved
/// `ModelBehavior.tscg`. Mirrors the `enable_loop_watchdog` pattern.
static TSCG_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Set once at startup from MODEL.toml `[behavior].tscg`. Idempotent.
pub fn set_tscg_enabled(enabled: bool) {
    let _ = TSCG_ENABLED.set(enabled);
}

/// Whether tool schemas should be TSCG-compiled for this model. Defaults
/// to `false` until `set_tscg_enabled` runs, so any pre-boot caller and
/// the unit tests see the unmodified JSON path.
pub fn tscg_enabled() -> bool {
    *TSCG_ENABLED.get().unwrap_or(&false)
}

/// Compile a tool list into the compact TSCG block. The result replaces
/// the JSON `<tools>` body inside a parser's `system_prompt()`.
///
/// Output shape (one stanza per tool):
///
/// ```text
/// search_files(query:str path?:str)
/// |Search files by content or pattern
///   query: search text
/// ```
///
/// Line 1 is the signature: `name(<params>)` with `?` marking optional
/// params (those absent from the schema's `required` list). Line 2 is
/// the `|`-prefixed, SDM-compressed tool description. Indented lines
/// carry SDM-compressed per-parameter docs, emitted only when the doc
/// adds information beyond the parameter name.
pub fn compile_tools(tools: &[ToolDefinition]) -> String {
    let mut out = String::new();
    for (i, tool) in tools.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&render::compile_one(tool));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_parser::{FunctionDefinition, ToolDefinition};

    fn tool(name: &str, desc: &str, params: serde_json::Value) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: Some(desc.to_string()),
                parameters: Some(params),
            },
        }
    }

    #[test]
    fn compiles_basic_signature() {
        let t = tool(
            "search_files",
            "Search project files by content or filename pattern",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string"},
                    "path": {"type": "string", "description": "Optional directory path"}
                },
                "required": ["query"]
            }),
        );
        let out = compile_tools(std::slice::from_ref(&t));
        // Signature line: required `query` before optional `path?`.
        assert!(
            out.starts_with("search_files(query:str path?:str)"),
            "got: {out}"
        );
        // Description is on a `|` line and far shorter than the JSON.
        assert!(out.contains("\n|"), "got: {out}");
        assert!(out.len() < serde_json::to_string(&t).unwrap().len());
    }

    #[test]
    fn empty_tool_list_is_empty() {
        assert_eq!(compile_tools(&[]), "");
    }

    #[test]
    fn disabled_by_default() {
        // No `set_tscg_enabled` call in this test → flag is false.
        assert!(!tscg_enabled());
    }
}
