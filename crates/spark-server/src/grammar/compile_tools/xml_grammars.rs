// SPDX-License-Identifier: AGPL-3.0-only

//! XML-style tool-call grammars (Qwen3-Coder, MiniMax M2.7).

use xgrammar::CompiledGrammar;

use crate::tool_parser::ToolDefinition;

use super::super::engine::{GrammarEngine, GrammarError};
use super::super::schema::{enforce_min_length_on_required_strings, sanitize_schema_for_grammar};
use super::ebnf::{short_tool_trigger_enabled, xml_param_value_body_ebnf};

impl GrammarEngine {
    /// Compile a grammar for Qwen3-Coder XML tool calls.
    ///
    /// Format: `<tool_call>\n<function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>\n</tool_call>`
    ///
    /// Uses XGrammar's `qwen_xml_parameter` content type for native XML parameter support.
    /// Falls back to `json_schema` if `qwen_xml_parameter` is not available in this XGrammar build.
    /// `value_close` is the literal delimiter that terminates a parameter
    /// VALUE region, supplied by the caller (the format's `ToolCallParser` via
    /// `param_value_close_delim()`) so the value-content rule is not hard-coded.
    pub fn compile_qwen3_coder_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
        value_close: &str,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());

        // Pre-sanitize all schemas so the fallback path can reuse them.
        struct SanitizedTool {
            name: String,
            schema: serde_json::Value,
        }
        let mut sanitized_tools = Vec::with_capacity(tools.len());
        for tool in tools {
            let name = &tool.function.name;
            let raw_schema = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            let raw_schema = match sanitize_schema_for_grammar(&raw_schema) {
                Some(s) => s,
                None => {
                    tracing::warn!("Skipping tool '{name}' in grammar — schema unsanitizable");
                    continue;
                }
            };
            if raw_schema.get("properties").is_none() && raw_schema.get("type").is_none() {
                tracing::warn!(
                    "Skipping tool '{name}' in grammar — schema has no properties or type"
                );
                continue;
            }
            let schema = enforce_min_length_on_required_strings(&raw_schema);
            sanitized_tools.push(SanitizedTool {
                name: name.clone(),
                schema,
            });
        }

        for st in &sanitized_tools {
            let begin = format!("<tool_call>\n<function={}>\n", st.name);
            let end = "\n</function>\n</tool_call>";
            // Tier-0 (Epoch 3, 2026-05-26): switch to RAW EBNF
            // (`grammar` content type) for the qwen3_coder body.
            // Previous attempts (regex `\S` Kleene-sandwich, regex `+`
            // quantifier with `[^ \t\r\n<][^<]*`, json_schema style
            // qwen_xml with minLength:1) ALL failed to enforce
            // non-empty parameter values under live opencode load
            // because xgrammar's regex-to-FSM lowering treats `*`/`+`
            // as quantifier edges with ε-transitions — the sole `\S`
            // anchor is skipped — and the json_schema converter has
            // a separate ε-edge bug for `[^]{1,}` minLength.
            //
            // EBNF rule INLINING (per B5's analysis of llama.cpp's
            // GBNF: rule body for `min,max` quantifiers is inlined
            // verbatim into the parent rule at compile time, so no
            // ε-transition can skip the first occurrence) is the
            // correct primitive for structural non-empty. Writing
            // the value rule as `first-char rest` where `first-char`
            // is a SINGLE TERMINAL CLASS (no quantifier) forces the
            // FSM to consume one matching byte before continuing.
            //
            // EBNF below:
            // - root      = one or more <parameter=KEY>VALUE</parameter> blocks separated by \n
            // - paramname = [a-zA-Z_] [a-zA-Z_0-9]*
            // - value     = MUST start with non-WS non-`<` byte, then any non-`<` bytes
            //
            // Param-name regex covers all valid Qwen3-Coder param names.
            // Value rule rejects empty AND whitespace-only AND
            // `<`-starting values, which structurally eliminates the
            // close-tag-as-first-body-token failure mode without
            // requiring sampler-level intervention.
            //
            // F2-revert (2026-05-26): F2 had relaxed the grammar to allow
            // `<` mid-value (`rest_part ::= [^<] | "<" [^/]`) to admit
            // Rust generics / shell redirects / HTML in tool args. Live
            // Wave-3 opencode testing showed the relaxation let the
            // model fall into XML-attribute syntax (emitting
            // `filePath="..." content="..."` inside what was supposed to
            // be a `<parameter=filePath>` body), creating a worse drift
            // mode than the original "1-char garbage" Epoch-3 failure.
            // SUPERSEDED by BUG#2 (2026-06-02): the strict `[^<]*` revert was
            // live until now (the F2-revert comment above is historical). It
            // refused `>`/`><`/`</X` content tokens (esp. via under-masked MTP
            // drafts) and emit_step turned each refusal into a lost turn — the
            // dominant opencode webserver_ok gap. Replaced by the
            // QWEN3_CODER_VALUE_BODY_EBNF negative-prefix ladder (allows `<`
            // content up to the literal `</parameter>` close). The F2
            // XML-attribute-drift risk is re-introduced in principle; BUG#1
            // graceful disengage keeps any residual refusal non-fatal, and the
            // live N=10 A/B is the gate for whether the ladder must be reverted.
            // Parser-side recovery (`tool_parser/parse_single_b.rs`, Tier-5c
            // re-roll) remains as defense in depth.
            // BUG#2 (2026-06-02): value EBNF built dynamically from the
            // trait-supplied `value_close` via ebnf_until_close_ladder() (SSOT,
            // no hard-coded per-model ladder). Allows `<`/`</X` value content
            // (real code) up to the literal close, replacing the strict
            // `[^<]`/`"<" [^/]` rule that refused `>`,`><`,`</X` mid-value.
            // Value body via the negative-prefix-ladder EBNF (bug#2). NOTE
            // (2026-06-02): `any_text` was trialled to remove the grammar
            // "alignment tax" on content but it let the model FREELANCE/ramble
            // without completing the tool call (finish=length) — the exact
            // failure the strict structure prevents. Kept the EBNF; the
            // content-glue it can induce is largely tolerated (Rust is
            // whitespace-insensitive; SC1 repairs TOML). The real webserver_ok
            // gap is being re-measured via the harness aggregate, not probes.
            let body_ebnf = xml_param_value_body_ebnf(value_close);
            let _ = &st.schema;
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "grammar", "grammar": body_ebnf},
                "end": end,
            }));
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        // Trigger selection depends on `use_triggers` (i.e. tool_choice mode):
        //
        // * tool_choice="auto" (use_triggers=true): per-tool LATE triggers
        //   `<tool_call>\n<function=NAME`. The model is free to emit a
        //   `<tool_call>` token and then *not* commit (e.g. by emitting
        //   plain prose afterwards), which is the ergonomic behaviour
        //   most pass-not-fail scenarios depend on (TC-11 mental math,
        //   TC-39 restraint, TC-43 ask-for-missing-arg, TC-48 multi-turn
        //   email composition). Late triggers preserve that freedom.
        //
        // * tool_choice="required"/specific (use_triggers=false): SHORT
        //   shared trigger `<tool_call>`. Without it, the model can — and
        //   does — emit `<tool_call><tool_call>…` indefinitely under
        //   required mode (`at_least_one=true` only suppresses EOS, it
        //   does not constrain content); LATE triggers stay in
        //   free-preamble forever because the `<tool_call>` special
        //   token never extends to the required `\n<function=` prefix.
        //   The SHORT trigger engages the moment the open token is
        //   sampled, locking xgrammar's `triggered_tags` alternation onto
        //   one of `\n<function=NAME>` for each registered tool — the
        //   `<tool_call><tool_call>…` lockup is unreachable by
        //   construction. Mirrors compile_minimax_xml_tool_grammar's F67
        //   fix for the same xgrammar behaviour pattern.
        // ATLAS_TOOL_SHORT_TRIGGER=1 (kill-switch, default OFF): under auto mode,
        // engage the grammar the moment `<tool_call>` is sampled (SHORT shared
        // trigger) instead of the LATE per-tool `<tool_call>\n<function=NAME`
        // prefix — same xgrammar behaviour the F67 note above relies on. OFF ⇒
        // byte-identical to the historical per-tool LATE triggers.
        let triggers: Vec<String> = if use_triggers && !short_tool_trigger_enabled() {
            sanitized_tools
                .iter()
                .map(|st| format!("<tool_call>\n<function={}", st.name))
                .collect()
        } else {
            vec!["<tool_call>".to_string()]
        };

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        match self.compile_structural_tag_raw(
            &triggers,
            &tag_entries,
            at_least_one,
            stop_after_first,
        ) {
            Ok(compiled) => Ok(compiled),
            Err(e) => {
                // Fall back to json_schema content type if qwen_xml_parameter
                // EBNF generation hits an edge case for one of these tool
                // schemas. The fallback path is fully functional — accuracy
                // is comparable, just the grammar is JSON-shaped instead of
                // XML-parameter-shaped under the hood. Emit at INFO with the
                // tool list so a follow-up bug report has the context to
                // narrow down which schema triggered xgrammar's EBNF parser
                // (Discord 2026-05-07 a1vadfs report on
                // mmangkad/Qwen3.6-27B-NVFP4: "EBNF parser error at line N").
                let tool_names: Vec<&str> =
                    sanitized_tools.iter().map(|st| st.name.as_str()).collect();
                tracing::info!(
                    "qwen_xml_parameter grammar fell back to json_schema ({e:?}). \
                     Functional but slightly looser tool-call grammar. Tools in \
                     this batch: [{}]. If you want to help narrow this down, \
                     set RUST_LOG=trace and re-run — the rejected schema is \
                     emitted at trace level by xgrammar.",
                    tool_names.join(", "),
                );
                let body_ebnf = xml_param_value_body_ebnf(value_close);
                let tag_entries_fallback: Vec<serde_json::Value> = sanitized_tools
                    .iter()
                    .map(|st| {
                        let _ = &st.schema;
                        serde_json::json!({
                            "type": "tag",
                            "begin": format!("<tool_call>\n<function={}>\n", st.name),
                            "content": {"type": "grammar", "grammar": body_ebnf},
                            "end": "\n</function>\n</tool_call>",
                        })
                    })
                    .collect();
                self.compile_structural_tag_raw(
                    &triggers,
                    &tag_entries_fallback,
                    at_least_one,
                    stop_after_first,
                )
            }
        }
    }

    /// F66 (2026-04-29): MiniMax M2.7 XML tool-call grammar.
    ///
    /// Native MiniMax format:
    /// ```xml
    /// <minimax:tool_call>
    /// <invoke name="tool_name">
    /// <parameter name="key1">value1</parameter>
    /// <parameter name="key2">value2</parameter>
    /// </invoke>
    /// </minimax:tool_call>
    /// ```
    ///
    /// Without this grammar, fix39 testing showed the model emit doubled
    /// tokens (`<invokeinvoke`, `<parameterparameter`, repeated phrases)
    /// when invoked through `--tool-call-parser minimax_xml` — XGrammar
    /// was warning "unknown parser format 'minimax_xml', skipping
    /// constrained decoding" and the unconstrained model freelanced
    /// into degenerate token loops at the tool-call boundary.
    ///
    /// Strategy: per-tool structural_tag with the OUTER frame fixed
    /// (`<minimax:tool_call>\n<invoke name="X">` and the closing
    /// `</invoke>\n</minimax:tool_call>`) and `any_text` for the body.
    /// This forces the wrapper structure to be exactly right (eliminates
    /// the `<invokeinvoke` corruption) while letting the model emit any
    /// `<parameter name="K">V</parameter>` sequence inside — the
    /// MinimaxXmlParser at parse time extracts those parameters from the
    /// body.
    ///
    /// The looser `any_text` body content was chosen over a strict
    /// per-parameter schema (which would require a custom EBNF or
    /// nested triggered_tags) because:
    ///   1. The OUTER frame doubling is the actual corruption source —
    ///      eliminating it stops the loop class.
    ///   2. MiniMax M2.7 is well-trained on the inner format and emits
    ///      it cleanly when the outer framing is constrained.
    ///   3. The output-side MinimaxXmlParser performs the strict
    ///      structural validation when extracting parameters.
    pub fn compile_minimax_xml_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());

        for tool in tools {
            let name = &tool.function.name;
            // Schema sanitization (kept consistent with other parsers
            // even though we don't use the schema for body constraint
            // — this still catches malformed schemas at compile time
            // so they're reported uniformly).
            let raw_schema = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            if sanitize_schema_for_grammar(&raw_schema).is_none() {
                tracing::warn!("Skipping tool '{name}' in grammar — schema unsanitizable");
                continue;
            }

            let begin = format!("<minimax:tool_call>\n<invoke name=\"{name}\">");
            let end = "</invoke>\n</minimax:tool_call>";
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "any_text"},
                "end": end,
            }));
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        // F67 (2026-04-29): SHORT shared trigger. xgrammar's
        // `triggered_tags` matcher is fully unconstrained until a
        // complete trigger string has been emitted; only after that
        // does it lock subsequent tokens to one of the registered
        // `tag.begin` continuations. With per-tool LATE triggers like
        // `<minimax:tool_call>\n<invoke name="bash"`, the model could
        // emit `<minimax:tool_call></minimax:tool_call>` (no `\n<invoke
        // …>` ever appears), the trigger never fired, and `at_least_one`
        // only blocked EOS — producing the
        // `<minimax:tool_call></minimax:tool_call>...` envelope loop
        // observed in fix40 live testing. The SHORT trigger
        // `<minimax:tool_call>` engages the moment the model opens the
        // envelope, after which xgrammar's TagDispatch alternation
        // forces one of `\n<invoke name="<TOOL>">` for each registered
        // tool — making the close-immediate degenerate output
        // unreachable by construction (proved by
        // `test_minimax_xml_grammar_rejects_degenerate`).
        let triggers = vec!["<minimax:tool_call>".to_string()];

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }
}
