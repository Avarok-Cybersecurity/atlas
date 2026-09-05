// SPDX-License-Identifier: AGPL-3.0-only

//! The scheduler-service TTFT clock must include grammar preparation.
//! Exercise the real deferred-prefill entry point without GPU work. A parser
//! records the instant its real grammar compile begins; request_start must
//! precede that instant, including a cache hit. No wall-time threshold/sleep.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::api::InferenceRequest;
use crate::api::inference_types::GrammarSpec;
use crate::grammar::{GrammarEngine, GrammarError};
use crate::tool_parser::{
    IncomingToolCall, PromptLevers, ToolCallParser, ToolChoice, ToolDefinition,
};
use xgrammar::CompiledGrammar;

struct ObservedParser(Arc<Mutex<Option<Instant>>>);

impl ToolCallParser for ObservedParser {
    fn name(&self) -> &str {
        "observed-grammar"
    }
    fn system_prompt(&self, _: &[ToolDefinition], _: &ToolChoice, _: &PromptLevers) -> String {
        unreachable!("request is already tokenized")
    }
    fn format_tool_calls(&self, _: &[IncomingToolCall]) -> String {
        unreachable!("request is already tokenized")
    }
    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        _: &[ToolDefinition],
        _: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        *self.0.lock().unwrap() = Some(Instant::now());
        Some(engine.compile_ebnf("root ::= \"x\"", "root"))
    }
}

fn request(grammar_spec: Option<GrammarSpec>) -> InferenceRequest {
    let (response_tx, _rx) = tokio::sync::oneshot::channel();
    InferenceRequest::Blocking {
        prompt_tokens: Arc::new(vec![0]),
        session_hash: 0,
        adapter_slot: -1,
        src_lang_id: 0,
        tgt_lang_id: 0,
        num_beams: 1,
        length_penalty: 1.0,
        early_stopping: false,
        image_pixels: vec![],
        max_tokens: 2,
        min_tokens: 0,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        lz_penalty: 0.0,
        logit_bias: vec![],
        stop_tokens: vec![],
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        require_tool_call: false,
        tools_present: grammar_spec.is_some(),
        suppress_tool_call: false,
        disable_mtp: true,
        grammar_spec,
        seed: Some(42),
        top_logprobs: None,
        prompt_logprobs: None,
        echo: false,
        timeout_at: None,
        response_tx,
    }
}

#[test]
fn deferred_prefill_clock_includes_cold_and_cached_grammar_preparation() {
    let marker = Arc::new(Mutex::new(None));
    let mut engine = Some(GrammarEngine::new(&["x".to_string()], &[]).unwrap());
    for cached in [false, true] {
        let spec = GrammarSpec::ToolCall {
            tools: vec![],
            parser: Arc::new(ObservedParser(Arc::clone(&marker))),
            use_triggers: true,
        };
        let result = super::prefill_a_step::start_chunked_prefill(
            &super::sched_ctx::SchedCtx::for_test(),
            None,
            None,
            None,
            None,
            &super::lifecycle_tests::StubModel,
            request(Some(spec)),
            &[],
            1,
            0,
            0,
            &mut engine,
            0,
            true,
            None,
            None,
        )
        .unwrap();
        let super::emit_step::StartPrefillResult::InProgress(prefill) = result else {
            panic!("deferred prefill unexpectedly performed inference");
        };
        assert!(
            prefill.grammar_state.is_some(),
            "real grammar must be prepared"
        );
        let compiled_at = marker.lock().unwrap().unwrap();
        assert!(
            prefill.request_start <= compiled_at,
            "TTFT origin excluded grammar preparation (cached={cached})"
        );
    }
}
