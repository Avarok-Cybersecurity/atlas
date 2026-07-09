// SPDX-License-Identifier: AGPL-3.0-only

//! Tool-call grammar compilation (Hermes, BareJson, Qwen3Coder, Gemma4, MiniMaxXml).

use std::collections::HashMap;

use xgrammar::CompiledGrammar;

use crate::tool_parser::ToolDefinition;

use super::engine::{GrammarEngine, GrammarError};
use super::schema::{enforce_min_length_on_required_strings, sanitize_schema_for_grammar};

/// Escape a single char for use inside an EBNF char class `[^ … ]`.
fn ebnf_class_escape(c: char) -> String {
    match c {
        ']' | '\\' | '^' | '-' => format!("\\{c}"),
        _ => c.to_string(),
    }
}

/// Escape a single char for use inside an EBNF double-quoted string literal.
fn ebnf_literal_escape(c: char) -> String {
    match c {
        '"' | '\\' => format!("\\{c}"),
        _ => c.to_string(),
    }
}

/// Generic "match any run of bytes up to (but not including) the literal
/// `close` delimiter", emitted as a negative-prefix ladder. This is the
/// REUSABLE primitive — each grammar/format supplies its own close delimiter
/// via dynamic dispatch (qwen3_coder `</parameter>`, MiniMax XML close, …);
/// there is no hard-coded per-model ladder.
///
/// For `close = c0 c1 … c{n-1}` it produces the alternation
///   `[^c0] | "c0" [^c1] | "c0c1" [^c2] | … | "c0…c{n-2}" [^c{n-1}]`
/// so any byte is legal, and any prefix of the close tag is legal UNLESS the
/// run completes the exact close sequence (each `[^x]` forbids the next close
/// char). The enclosing rule then consumes the literal close itself.
///
/// BUG#2 (2026-06-02): replaces the prior strict `[^<] | "<" [^/]` value rule
/// that refused `>`,`><`,`</X` content tokens (esp. via under-masked MTP
/// drafts), which `emit_step` turned into truncated turns — the dominant
/// opencode webserver_ok gap. NOTE this re-permits `<`-content; BUG#1 graceful
/// disengage keeps any residual refusal non-fatal, and the live N=10 A/B is the
/// gate for whether the prior F2 XML-attribute-drift mode returns.
/// P1-2 (2026-07-09, env `ATLAS_GRAMMAR_FORCE_CLOSE=1`): with `force_close`,
/// the DEEPEST arm (`"</parameter" [^>]` for qwen3_coder) is omitted, so once
/// the value contains the full close-prefix the ONLY legal continuation is the
/// literal final char — xgrammar's mask deflects a dropped `>` at sample time
/// (the 45k garble `</parameter<parameter=` becomes unrepresentable). Cost:
/// value content containing `</parameter` + non-`>` (e.g. `</parameters>`)
/// becomes unwritable — hence opt-in + BFCL A/B gate before default-on.
fn ebnf_until_close_ladder_opts(close: &str, force_close: bool) -> String {
    let chars: Vec<char> = close.chars().collect();
    debug_assert!(!chars.is_empty(), "close delimiter must be non-empty");
    let mut alts: Vec<String> = Vec::with_capacity(chars.len().max(1));
    let depth = if force_close && chars.len() > 1 {
        chars.len() - 1
    } else {
        chars.len()
    };
    for k in 0..depth {
        let neg = ebnf_class_escape(chars[k]);
        if k == 0 {
            alts.push(format!("[^{neg}]"));
        } else {
            let prefix: String = chars[..k]
                .iter()
                .copied()
                .map(ebnf_literal_escape)
                .collect();
            alts.push(format!("\"{prefix}\" [^{neg}]"));
        }
    }
    if alts.is_empty() {
        // Degenerate empty-close guard: accept any single byte.
        return "[^\\x00]".to_string();
    }
    alts.join(" | ")
}

fn ebnf_until_close_ladder(close: &str) -> String {
    ebnf_until_close_ladder_opts(close, false)
}

/// P1-1/P1-2 env opt-ins (read per call; set in the serving environment).
fn grammar_allow_empty_value() -> bool {
    std::env::var("ATLAS_GRAMMAR_ALLOW_EMPTY_VALUE").as_deref() == Ok("1")
}
fn grammar_force_close() -> bool {
    std::env::var("ATLAS_GRAMMAR_FORCE_CLOSE").as_deref() == Ok("1")
}

/// F2-2a (2026-06-02): structural ceiling on a parameter VALUE's `rest`
/// repetition, applied ONLY when the `ATLAS_GRAMMAR_VALUE_HARDEN` kill-switch
/// is on. A garbled/merged BPE close token (e.g. `</parameter_002e>`) can leave
/// the literal-close match unfired, so `rest ::= rest_part*` accepts forever and
/// the value runs to `max_tokens`. A bounded `rest_part{0,N}` makes an unclosed
/// value structurally impossible to grow past `N` bytes. ~6000 is far above any
/// legitimate single tool-arg value (a `write` `content` field) while still
/// finite. F1's per-generation cap is the primary runaway bound; this is a
/// grammar-level backstop kept behind the switch because grammar edits have
/// regressed before (Iter 48) and demand an isolated N=10 A/B.
const VALUE_REST_MAX_REPEAT: u32 = 6000;

/// Whether the F2 value-hardening kill-switch is on. Read once per call from
/// `ATLAS_GRAMMAR_VALUE_HARDEN`; OFF unless exactly `"1"`. OFF ⇒ the emitted
/// grammar is byte-identical to the historical `rest ::= rest_part*`.
fn value_harden_enabled() -> bool {
    std::env::var("ATLAS_GRAMMAR_VALUE_HARDEN").as_deref() == Ok("1")
}

/// Whether the SHORT shared `<tool_call>` trigger is forced under
/// `tool_choice="auto"`. Read once per call from `ATLAS_TOOL_SHORT_TRIGGER`;
/// OFF unless exactly `"1"`. OFF ⇒ the auto-mode triggers are byte-identical to
/// the historical per-tool LATE `<tool_call>\n<function=NAME` set.
fn short_tool_trigger_enabled() -> bool {
    std::env::var("ATLAS_TOOL_SHORT_TRIGGER").as_deref() == Ok("1")
}

/// Body EBNF for an XML-style `<parameter=NAME>VALUE{value_close}` parameter
/// block (a `<parameter=…>…{close}` sequence). The VALUE region accepts
/// arbitrary bytes up to the literal `value_close` via the generic
/// [`ebnf_until_close_ladder`]. SSOT — used by the primary + json_schema
/// fallback paths.
///
/// `value_close` is NOT hard-coded: each format supplies it through its
/// [`crate::tool_parser::ToolCallParser::param_value_close_delim`] impl, so the
/// value-content fix is dynamically dispatched per grammar — any format with a
/// `<…>VALUE<close>` region gets it, not just qwen3_coder.
///
/// F2-2a: when `ATLAS_GRAMMAR_VALUE_HARDEN=1` the `rest` rule is bounded
/// `rest_part{0,N}` instead of `rest_part*`; OFF (the default) emits the
/// byte-identical historical Kleene-star form.
///
/// TODO(F2-2b, 2026-06-02): also accept a merged-prefix close — the close
/// delimiter appearing as the leading bytes of a longer (garbled) BPE token —
/// so a drifted close still terminates the value. Routed through the same
/// trait-supplied `value_close` (no hard-coded per-model tokens). Deferred:
/// 2a (this) is the structural backstop; 2b is the next kill-switched step.
pub(crate) fn xml_param_value_body_ebnf(
    value_close: &str,
    param_names: Option<&[String]>,
) -> String {
    xml_param_value_body_ebnf_opts(
        value_close,
        param_names,
        grammar_allow_empty_value(),
        grammar_force_close(),
    )
}

/// Explicit-parameter core (unit-testable without env races).
pub(crate) fn xml_param_value_body_ebnf_opts(
    value_close: &str,
    param_names: Option<&[String]>,
    allow_empty_value: bool,
    force_close: bool,
) -> String {
    let ladder = ebnf_until_close_ladder_opts(value_close, force_close);
    let rest_rule = if value_harden_enabled() {
        format!("rest ::= rest_part{{0,{VALUE_REST_MAX_REPEAT}}}")
    } else {
        "rest ::= rest_part*".to_string()
    };
    // Echo-attractor fix (2026-07-09): constrain the parameter NAME to the
    // tool schema's property names when they are known. The historical
    // identifier rule accepted ANY identifier, so at long context a model
    // drifting into the wire-format echo attractor could emit
    // `<parameter=parameter>filePath>…` / `<parameter=write>content>…` —
    // grammar-LEGAL under the old rule, and the key-slot masking of `=`
    // actively converted a recoverable one-token echo into a permanently
    // mis-keyed call (live opencode collapse at 42.5k tokens, 2026-07-09).
    // With the alternation below, xgrammar's mask forces the key slot onto a
    // real schema property, making the mis-keyed shape unrepresentable.
    // `None` (schema-less / open schema) keeps the historical identifier rule.
    let paramname_rule = match param_names {
        Some(names) if !names.is_empty() => {
            let alts: Vec<String> = names
                .iter()
                .map(|n| serde_json::to_string(n).unwrap_or_else(|_| "\"\"".into()))
                .collect();
            format!("paramname ::= {}", alts.join(" | "))
        }
        _ => "paramname ::= [a-zA-Z_] [a-zA-Z_0-9]*".to_string(),
    };
    // Content-start rule: allow a leading whitespace run (INCLUDING `\n`),
    // then REQUIRE at least one non-whitespace char that is NOT `=` or `>`
    // before the rest. Boundary bugs closed here:
    //  - `leading_ws*` split: the old `[^ \t\r\n<]` masked the model's
    //    genuine top-1 at content-start (a leading `\n`) and — under FP8
    //    long-ctx drift — forced the argmax onto a wrong identifier
    //    runner-up (`lean`/`cargo`). The split unmasks `\n` while keeping
    //    the non-empty guard.
    //  - `=`/`>` exclusion (2026-06-03, diag agent acb6cb1): the param key
    //    closes with `>` and the tokenizer has ~198 `>X` MERGE tokens
    //    (`>=`=9628, `>>`, …). At the `<parameter=KEY>`→value boundary the
    //    model can emit the merged token `>=` (id 9628) — the `>` satisfies
    //    the `">"` literal and the glued `=` lands as the value's first char,
    //    producing `=axum::serve(...)` (the phantom-`=` that broke `edit`
    //    oldString matches and stalled the agent). Excluding `=`/`>` from
    //    `first_content` makes xgrammar's mask REJECT token 9628 at the
    //    boundary, forcing a standalone `>` (id 29) + a real content token.
    //    Legit `>a`/`>{`/`>"` merges stay legal (2nd byte passes); only a
    //    value that genuinely starts with `=`/`>` is disallowed (rare for
    //    code/TOML edit args). Parser is innocent; this is NOT numerics.
    //  - `<`-start UNBANNED (2026-07-09): the historical blanket `<`
    //    exclusion (Epoch-3 "close-tag-as-first-body-token" guard) made any
    //    value whose first non-WS char is `<` UNREPRESENTABLE — the mask
    //    deflected `<script>` / `<!DOCTYPE` at position 0, so the model
    //    could not write Svelte/HTML/XML files AT ALL. Live opencode
    //    session: every `.svelte` write came out script-logic-only (the
    //    masked model fell into JS-module mode), the model noticed and
    //    retried the identical write forever — the repeat-loop the
    //    repetition guards then cut mid-sentence. The BUG#2 negative-prefix
    //    ladder already prevents the original failure (the literal
    //    `{value_close}` as body) at EVERY position, so first_content now
    //    reuses the SAME ladder for its `<` arm: `<` is legal at content
    //    start unless it begins the exact close delimiter.
    // P1-1 (2026-07-09, env `ATLAS_GRAMMAR_ALLOW_EMPTY_VALUE=1`): the
    // non-empty guard makes an EMPTY parameter value unrepresentable, so a
    // model intending `<parameter=content></parameter>` is mask-coerced onto
    // garbage first bytes — at 43k depth this manufactured the byte-identical
    // `</parameter<parameter=` attractor. Opt-in allows the empty value
    // (immediate close). Trade-off: re-opens the Epoch-3 empty-garbage-call
    // shape — hence opt-in + BFCL A/B gate before default-on.
    let value_rule = if allow_empty_value {
        "value ::= leading_ws (first_content rest)?"
    } else {
        "value ::= leading_ws first_content rest"
    };
    format!(
        r#"root ::= param ("\n" param)*
param ::= "<parameter=" paramname ">" value "{value_close}"
{paramname_rule}
{value_rule}
leading_ws ::= [ \t\r\n]*
first_content ::= [^ \t\r\n<=>] | {first_lt_arms}
{rest_rule}
rest_part ::= {ladder}
"#,
        first_lt_arms = ladder_lt_arms(&ladder),
    )
}

/// The `<`-starting alternates of an [`ebnf_until_close_ladder`] alternation.
///
/// For close `</parameter>` the full ladder is
/// `[^<] | "<" [^/] | "</" [^p] | …`; this keeps only the arms that BEGIN
/// with the close delimiter's first char (`"<" [^/] | "</" [^p] | …`) — i.e.
/// "a `<` is fine unless it starts the exact close sequence". Used by
/// `first_content` so `<`-initial values (Svelte `<script>`, HTML
/// `<!DOCTYPE`) are representable while the literal close tag still cannot
/// open a value. Derived from the ladder (SSOT), not hard-coded per format.
fn ladder_lt_arms(ladder: &str) -> String {
    let arms: Vec<&str> = ladder
        .split(" | ")
        .filter(|arm| arm.starts_with('"'))
        .collect();
    if arms.is_empty() {
        // Single-char close delimiter: its ladder is just `[^c]`, which the
        // base `[^ \t\r\n<=>]` class already covers conservatively.
        return "[^ \\t\\r\\n<=>]".to_string();
    }
    arms.join(" | ")
}

/// Extract the property names a qwen3_coder `<parameter=NAME>` key slot may
/// take for this (sanitized) tool schema, or `None` to keep the permissive
/// identifier rule. `None` when: no/empty `properties`, or the schema
/// explicitly opts into arbitrary extra keys (`additionalProperties` set to
/// anything but `false`) — constraining those would mask legitimate calls.
pub(crate) fn schema_param_names(schema: &serde_json::Value) -> Option<Vec<String>> {
    let props = schema.get("properties")?.as_object()?;
    if props.is_empty() {
        return None;
    }
    match schema.get("additionalProperties") {
        None | Some(serde_json::Value::Bool(false)) => {}
        Some(_) => return None,
    }
    Some(props.keys().cloned().collect())
}

impl GrammarEngine {
    // ── Tool call grammars ──

    /// Compile a grammar for Hermes-format tool calls.
    ///
    /// Hermes format: `<tool_call>{"name":"fn","arguments":{...}}</tool_call>`
    ///
    /// Builds raw structural tag JSON with `at_least_one` / `stop_after_first`
    /// (bypasses xgrammar-rs wrapper which doesn't expose these fields).
    ///
    /// - `use_triggers=true` (tool_choice="auto"): triggers active, model chooses freely
    /// - `use_triggers=false` (tool_choice="required"): at_least_one + stop_after_first
    pub fn compile_hermes_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());
        let mut triggers = Vec::new();
        let mut seen_triggers = HashMap::<String, bool>::new();

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

            // Tag begin/end MUST match the model's template-demonstrated
            // format (`<tool_call>\n{"name": "X", "arguments": {…}}\n</tool_call>`
            // — newline after the tag, space after colons). The historical
            // compact form was OOD for the model: conditioned on a forced
            // compact call, its natural continuation was ANOTHER
            // `<tool_call>` (per the template's multi-call examples), which
            // the trigger engaged and the FSM made it fill with argmax junk
            // (`{"loan_amount":-1,"interest_rate":-0.0,…}` trailing calls,
            // 2026-07-03, reproduced with and without MTP).
            let begin = format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": ");
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "}\n</tool_call>",
            }));

            // 2026-07-03: SHORT shared trigger. The historical LATE trigger
            // `<tool_call>{"name":"` could never fire — the Qwen template's
            // own tool instructions demonstrate `<tool_call>\n{"name": …`
            // (newline + spaced JSON), so auto-mode constrained decoding
            // NEVER engaged and the model's habit of closing `arguments`
            // before trailing required params went unchecked (ST-995
            // non_live 85.42→39.72 collapse; `tool_choice="required"` —
            // grammar active from token 0 — produced the complete, correct
            // call for the same prompt). Trigger on the tag itself; the tag
            // `begin` then forces the compact wire format.
            let trigger = "<tool_call>".to_string();
            if !seen_triggers.contains_key(&trigger) {
                seen_triggers.insert(trigger.clone(), true);
                triggers.push(trigger);
            }
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        // auto: at_least_one=false (model freely chooses text or tool)
        // required: at_least_one=true + stop_after_first=true (EOS suppressed until one tool call)
        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }

    /// Compile a grammar for bare-JSON tool calls (no `<tool_call>` wrapper).
    ///
    /// Format: `{"name":"<one_of_tools>","arguments":<schema>}` — top-level
    /// JSON object, nothing else. Used by Nemotron-Super-120B which falls
    /// into degenerate token loops when the qwen3_coder `<tool_call>` wrapper
    /// is forced (its training distribution does not cover that prefix).
    ///
    /// - `use_triggers=true` (tool_choice="auto"): triggers active, model chooses freely
    /// - `use_triggers=false` (tool_choice="required"): at_least_one + stop_after_first
    pub fn compile_bare_json_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());
        let mut triggers = Vec::new();
        let mut seen_triggers = HashMap::<String, bool>::new();

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

            let begin = format!(r#"{{"name":"{name}","arguments":"#);
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "}",
            }));
        }

        let trigger = r#"{"name":""#.to_string();
        if !seen_triggers.contains_key(&trigger) {
            seen_triggers.insert(trigger.clone(), true);
            triggers.push(trigger);
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }

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
            // Echo-attractor fix (2026-07-09): the sanitized schema's property
            // names now constrain the `<parameter=NAME>` key slot (see
            // `xml_param_value_body_ebnf`). Previously the schema was
            // discarded here and any identifier was key-legal.
            let param_names = schema_param_names(&st.schema);
            let body_ebnf = xml_param_value_body_ebnf(value_close, param_names.as_deref());
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
        //   `<tool_call>\n<function=NAME>`. The model is free to emit a
        //   `<tool_call>` token and then *not* commit (e.g. by emitting
        //   plain prose afterwards), which is the ergonomic behaviour
        //   most pass-not-fail scenarios depend on (TC-11 mental math,
        //   TC-39 restraint, TC-43 ask-for-missing-arg, TC-48 multi-turn
        //   email composition). Late triggers preserve that freedom.
        //
        //   The closing `>` is part of the trigger so the trigger set is
        //   prefix-free. Issue #88: when one tool name is a textual prefix
        //   of another (`mcp_scrapling_get` vs `mcp_scrapling_get_prompt`
        //   — the natural `mcp_{server}_{tool}` MCP pattern), a trigger
        //   without the `>` (`…=mcp_scrapling_get`) is a prefix of the
        //   longer tool's tag begin (`…=mcp_scrapling_get_prompt>\n`), so
        //   that tag matches TWO triggers and xgrammar's triggered-tags
        //   converter rejects the entire grammar ("One tag matches
        //   multiple triggers"), silently disabling constrained decoding
        //   and letting the model concatenate/hallucinate tool names. The
        //   `>` terminator cannot appear inside a tool name, so `NAME>` is
        //   never a prefix of another tool's `NAME2>` — each tag matches
        //   exactly its own trigger.
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
            // #88: dedup defensively — two tools sharing a name would
            // otherwise produce duplicate triggers, which the converter
            // rejects as "multiple triggers" (the OpenAI/MCP contract
            // requires unique names, but a malformed tool list must not
            // hard-disable the grammar). Prefix-free: the closing `>` is
            // included so `NAME>` is never a prefix of another `NAME2>`.
            let mut seen = HashMap::<String, bool>::new();
            sanitized_tools
                .iter()
                .map(|st| format!("<tool_call>\n<function={}>", st.name))
                .filter(|trigger| seen.insert(trigger.clone(), true).is_none())
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
                let tag_entries_fallback: Vec<serde_json::Value> = sanitized_tools
                    .iter()
                    .map(|st| {
                        // Echo-attractor fix (2026-07-09): per-tool key-slot
                        // constraint, mirroring the primary path above.
                        let param_names = schema_param_names(&st.schema);
                        let body_ebnf =
                            xml_param_value_body_ebnf(value_close, param_names.as_deref());
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

    /// Compile a grammar for Gemma-4 native tool calls.
    ///
    /// Gemma-4's native format uses special sentinel tokens:
    ///   `<|tool_call>call:NAME{"key":"val",...}<tool_call|>`
    ///
    /// We use standard JSON for the argument block (unlike Gemma's original
    /// unquoted-key / `<|"|>` delimiter format). The existing parser
    /// (`parse_gemma4_native_call` in tool_parser.rs) accepts standard JSON
    /// transparently: the `gemma4_to_json` converter is a no-op when the body
    /// is already valid JSON, and unquoted-key quoting only fires when needed.
    ///
    /// Grammar-constraining the output ensures the model ALWAYS emits the
    /// exact `<|tool_call>call:NAME{...}<tool_call|>` framing instead of
    /// narrating tool calls in plain text (the WARN root cause on 26B Search
    /// and 31B Weather+Search).
    pub fn compile_gemma4_tool_grammar(
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

            // Gemma-4 sentinel tokens frame the call; JSON body in between.
            let begin = format!("<|tool_call>call:{name}");
            let end = "<tool_call|>";
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": end,
            }));
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        // Trigger selection mirrors compile_qwen3_coder_tool_grammar's use_triggers branch:
        // AUTO (use_triggers=true) triggers on the shared `<|tool_call>call:` prefix so the
        // model can freely choose prose vs a tool call. REQUIRED (use_triggers=false) triggers
        // on the SHORT opening sentinel `<|tool_call>` — a single BPE token. The long
        // multi-byte `<|tool_call>call:` prefix spans BPE merges that misalign with the tag
        // boundary; under at_least_one that dead-ended xgrammar's token FSM (HTTP 5xx /
        // truncated "read_fil" at tool_choice=required). The short single-token trigger is a
        // clean tag prefix and avoids the dead-end. (Triggers must be non-empty — xgrammar's
        // triggered_tags rejects an empty trigger set even with at_least_one.)
        let triggers = if use_triggers {
            vec!["<|tool_call>call:".to_string()]
        } else {
            vec!["<|tool_call>".to_string()]
        };

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
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
