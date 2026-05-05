// SPDX-License-Identifier: AGPL-3.0-only

//! Tokenizer-derived runtime: vocab cap, reasoning parser, think tokens,
//! ChatML im_start hard-stop, reflection suppression, tool-call open/close
//! tokens, and the XGrammar engine.

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) struct TokenizerRuntime {
    pub(crate) reasoning_parser_box: Option<Box<dyn crate::reasoning_parser::ReasoningParser>>,
    pub(crate) think_end_token: Option<u32>,
    pub(crate) think_start_token: Option<u32>,
    pub(crate) reflection_suppress_ids: Vec<u32>,
    pub(crate) tool_call_start_token: Option<u32>,
    pub(crate) tool_call_end_token: Option<u32>,
    pub(crate) grammar_engine: Option<crate::grammar::GrammarEngine>,
}

pub(crate) fn resolve_tokenizer_runtime(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
    tokenizer: &crate::tokenizer::ChatTokenizer,
    eos_tokens: &mut Vec<u32>,
    supports_thinking: bool,
) -> TokenizerRuntime {
    use crate::{grammar, reasoning_parser};

    let tokenizer_vocab = tokenizer.inner().get_vocab_size(true);
    if tokenizer_vocab > 0 && tokenizer_vocab < config.vocab_size {
        tracing::info!(
            "Capping vocab_size from {} (config) to {} (tokenizer incl. special tokens)",
            config.vocab_size,
            tokenizer_vocab,
        );
        config.vocab_size = tokenizer_vocab;
    }

    let reasoning_parser_box: Option<Box<dyn reasoning_parser::ReasoningParser>> = {
        let defaults_toml = include_str!("../../../tool_defaults.toml");
        let defaults: toml::Value =
            toml::from_str(defaults_toml).unwrap_or(toml::Value::Table(Default::default()));
        let auto_format = defaults
            .get("reasoning")
            .and_then(|t| t.get(config.model_type.as_str()))
            .and_then(|s| s.as_str())
            .and_then(|s| s.parse::<reasoning_parser::ReasoningFormat>().ok());
        if let Some(fmt) = auto_format {
            let p = fmt.into_parser();
            tracing::info!(
                "Reasoning parser: {} (auto-detected from model_type '{}')",
                p.name(),
                config.model_type
            );
            Some(p)
        } else if supports_thinking {
            let p = reasoning_parser::ReasoningFormat::Qwen.into_parser();
            tracing::info!(
                "Reasoning parser: {} (default for thinking-capable model)",
                p.name()
            );
            Some(p)
        } else {
            None
        }
    };
    let think_end_token = reasoning_parser_box
        .as_ref()
        .and_then(|p| p.end_token_id(tokenizer));
    let think_start_token: Option<u32> = tokenizer
        .encode("<think>")
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(tid) = think_end_token {
        tracing::info!(
            "Thinking end token: {} ({})",
            tid,
            reasoning_parser_box.as_ref().unwrap().end_tag()
        );
    }
    if let Some(tid) = think_start_token {
        tracing::info!("Thinking start token: {tid} (<think>)");
    }

    let im_start_id: Option<u32> = tokenizer
        .encode("<|im_start|>")
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(id) = im_start_id {
        if !eos_tokens.contains(&id) {
            eos_tokens.push(id);
        }
        crate::scheduler::set_im_start_hard_stop(id);
        tracing::info!("ChatML role-boundary hard stop: <|im_start|> (id {id}) registered");
    }

    let reflection_words = [
        "wait", "Wait", "however", "However", "actually", "Actually", "hmm", "Hmm",
    ];
    let reflection_suppress_ids: Vec<u32> = reflection_words
        .iter()
        .filter_map(|word| tokenizer.encode(word).ok())
        .filter(|ids| ids.len() == 1)
        .map(|ids| ids[0])
        .collect();
    if !reflection_suppress_ids.is_empty() {
        tracing::info!(
            "Reflection suppression tokens: {} IDs resolved",
            reflection_suppress_ids.len()
        );
    }

    let tool_call_format_name: Option<String> = args.tool_call_parser.clone().or_else(|| {
        let defaults: toml::Table = toml::from_str(include_str!("../../../tool_defaults.toml"))
            .expect("invalid tool_defaults.toml");
        defaults
            .get("model_type")
            .and_then(|t| t.as_table())
            .and_then(|t| t.get(config.model_type.as_str()))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    });
    let (tc_start_str, tc_end_str): (&str, &str) = match tool_call_format_name.as_deref() {
        Some("minimax_xml") => ("<minimax:tool_call>", "</minimax:tool_call>"),
        _ => ("<tool_call>", "</tool_call>"),
    };
    let tool_call_start_token = tokenizer
        .encode(tc_start_str)
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(tid) = tool_call_start_token {
        tracing::info!("Tool call start token: {} ({})", tid, tc_start_str);
    } else {
        tracing::warn!(
            "Tool call start token unresolved for {tc_start_str} — \
             require_tool_call / suppress / force-emit-after-think will be no-ops"
        );
    }
    let tool_call_end_token = tokenizer
        .encode(tc_end_str)
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(tid) = tool_call_end_token {
        tracing::info!("Tool call end token: {} ({})", tid, tc_end_str);
    }

    let grammar_engine = {
        let stop_ids: Vec<i32> = eos_tokens.iter().map(|&id| id as i32).collect();
        let model_vocab_size = Some(config.vocab_size);
        match grammar::GrammarEngine::from_tokenizer(tokenizer.inner(), model_vocab_size, &stop_ids)
        {
            Ok(engine) => {
                tracing::info!(
                    "Grammar engine initialized (vocab_size={}, vocab_type=auto-detected from tokenizer)",
                    engine.vocab_size()
                );
                Some(engine)
            }
            Err(e) => {
                tracing::warn!("Grammar engine init failed (constrained decoding disabled): {e}");
                None
            }
        }
    };

    TokenizerRuntime {
        reasoning_parser_box,
        think_end_token,
        think_start_token,
        reflection_suppress_ids,
        tool_call_start_token,
        tool_call_end_token,
        grammar_engine,
    }
}
