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
    /// Token ID for a markdown code fence (```). Used to suppress the
    /// confidence-based thinking early-stop (F2) while the model is
    /// productively emitting a fenced code block inside `<think>`:
    /// code tokens are near-deterministic (top-1 prob ≥ 0.95 for long
    /// runs) but that is NOT a "done reasoning" signal. `None` =
    /// tokenizer has no atomic fence token → guard disabled (fail-open,
    /// F2 keeps its prior behaviour).
    pub(crate) code_fence_token: Option<u32>,
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
    // Markdown code fence (```) as a single atomic token (Qwen3.x BPE:
    // id 71093). Resolved the same way as think_start_token; `None` if
    // the tokenizer splits it (guard fails open — see struct doc).
    let code_fence_token: Option<u32> = tokenizer
        .encode("```")
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(fid) = code_fence_token {
        tracing::info!("Code-fence token: {} (``` — F2 fence guard active)", fid);
    }

    // Digit-normalized content-loop watchdog mask (Qwen3.6-27B greedy
    // template degeneration: `- B(46) = N\n- B(47) = M\n …` to the cap).
    // `mask[id] == true` iff the token decodes to a pure ASCII-digit run
    // with at most one leading space. `decode_with_special` drives the
    // byte-level decoder so a leading space is ' ' (NOT the raw `Ġ` BPE
    // marker that `id_to_token` would yield). Built unconditionally
    // (cheap, model-agnostic, one-time); only *consumed* under the
    // per-model `enable_loop_watchdog()` gate. Fail-open: any decode
    // error leaves that id `false`.
    {
        let vocab_size = tokenizer.inner().get_vocab_size(true);
        let mut mask: Vec<bool> = vec![false; vocab_size];
        let mut numeric_count = 0usize;
        for (id, slot) in mask.iter_mut().enumerate() {
            if let Ok(s) = tokenizer.decode_with_special(&[id as u32]) {
                let body = s.strip_prefix(' ').unwrap_or(&s);
                if !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit()) {
                    *slot = true;
                    numeric_count += 1;
                }
            }
        }
        crate::scheduler::set_numeric_token_mask(std::sync::Arc::from(mask));
        tracing::info!(
            "Numeric-token mask: {numeric_count}/{vocab_size} ids classified \
             as digit-runs (digit-normalized content-loop path active)"
        );
    }

    // Phase-C boundary-token mask (drives rollback-to-boundary). `mask[id]`
    // is true iff the token decodes to text *ending* in a well-formed
    // generation boundary: a newline, or sentence-ending punctuation
    // (`.`/`!`/`?`) optionally trailed by a closing quote / bracket /
    // whitespace. Built unconditionally (cheap, model-agnostic, one-time);
    // consumed only when a watchdog fires under `rollback_resteer`.
    // Fail-open: any decode error leaves that id `false`.
    {
        let vocab_size = tokenizer.inner().get_vocab_size(true);
        let mut mask: Vec<bool> = vec![false; vocab_size];
        let mut boundary_count = 0usize;
        let is_boundary = |s: &str| -> bool {
            // Trim trailing closing quotes / brackets / whitespace, then
            // check the last meaningful byte.
            let trimmed = s.trim_end_matches([' ', '\t', '"', '\'', ')', ']', '}', '\r']);
            match trimmed.chars().last() {
                Some('\n') => true,
                Some('.') | Some('!') | Some('?') => true,
                _ => s.ends_with('\n'),
            }
        };
        for (id, slot) in mask.iter_mut().enumerate() {
            if let Ok(s) = tokenizer.decode_with_special(&[id as u32])
                && !s.is_empty()
                && is_boundary(&s)
            {
                *slot = true;
                boundary_count += 1;
            }
        }
        crate::scheduler::set_boundary_token_mask(std::sync::Arc::from(mask));
        tracing::info!(
            "Boundary-token mask: {boundary_count}/{vocab_size} ids end in a \
             newline / sentence boundary (Phase-C rollback-to-boundary active)"
        );
    }

    // Mid-word token mask (2026-05-24): `mask[id]` is true iff the
    // token decodes to text whose last character is alphanumeric — i.e.
    // emitting `</think>` (or other sentence-end punctuation) right
    // after this token would split a word. Drives the
    // `decode_logits_seq::mid_word_</think>_defer` guard. Built once
    // unconditionally (same one-time decode loop cost as the boundary
    // mask, cheap and model-agnostic). Fail-open: any decode error
    // leaves the id `false`, so the suppression simply doesn't fire
    // for that token.
    {
        let vocab_size = tokenizer.inner().get_vocab_size(true);
        let mut mask: Vec<bool> = vec![false; vocab_size];
        let mut mid_word_count = 0usize;
        for (id, slot) in mask.iter_mut().enumerate() {
            if let Ok(s) = tokenizer.decode_with_special(&[id as u32])
                && !s.is_empty()
                && let Some(last_ch) = s.chars().last()
                && last_ch.is_alphanumeric()
            {
                *slot = true;
                mid_word_count += 1;
            }
        }
        crate::scheduler::set_mid_word_token_mask(std::sync::Arc::from(mask));
        tracing::info!(
            "Mid-word token mask: {mid_word_count}/{vocab_size} ids end in alphanumeric \
             (mid-word </think> defer active during thinking)"
        );
    }

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

    // Fix B (2026-06-05): mirror the <|im_start|> hard-stop resolution for the
    // <tool_response> control token. Only registered when it resolves to a
    // single token id. NOT added to `eos_tokens` — that would alter behavior
    // even with the kill-switch OFF (the id would be treated as a stop token on
    // the always-on EOS path); registration stays inert until the decode-time
    // gate `tool_response_stop_enabled()` (ATLAS_TOOL_RESPONSE_STOP=1, default
    // OFF) consults `tool_response_hard_stop()`.
    let tool_response_id: Option<u32> = tokenizer
        .encode("<tool_response>")
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(id) = tool_response_id {
        crate::scheduler::set_tool_response_hard_stop(id);
        tracing::info!("Tool-response hard stop: <tool_response> (id {id}) registered");
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
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        grammar_engine,
    }
}
