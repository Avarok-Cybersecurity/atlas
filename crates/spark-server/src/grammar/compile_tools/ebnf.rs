// SPDX-License-Identifier: AGPL-3.0-only

//! EBNF helper primitives shared by the tool-call grammar compilers.

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
fn ebnf_until_close_ladder(close: &str) -> String {
    let chars: Vec<char> = close.chars().collect();
    debug_assert!(!chars.is_empty(), "close delimiter must be non-empty");
    let mut alts: Vec<String> = Vec::with_capacity(chars.len().max(1));
    for k in 0..chars.len() {
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
pub(super) fn short_tool_trigger_enabled() -> bool {
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
pub(super) fn xml_param_value_body_ebnf(value_close: &str) -> String {
    let ladder = ebnf_until_close_ladder(value_close);
    let rest_rule = if value_harden_enabled() {
        format!("rest ::= rest_part{{0,{VALUE_REST_MAX_REPEAT}}}")
    } else {
        "rest ::= rest_part*".to_string()
    };
    // Content-start rule: allow a leading whitespace run (INCLUDING `\n`),
    // then REQUIRE at least one non-whitespace char that is NOT `<`, `=`, or
    // `>` before the rest. Two distinct boundary bugs are closed here:
    //  - `<` exclusion + the `leading_ws*` split: the old `[^ \t\r\n<]`
    //    masked the model's genuine top-1 at content-start (a leading `\n`)
    //    and — under FP8 long-ctx drift — forced the argmax onto a wrong
    //    identifier runner-up (`lean`/`cargo`). The split unmasks `\n` while
    //    keeping the non-empty guard.
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
    format!(
        r#"root ::= param ("\n" param)*
param ::= "<parameter=" paramname ">" value "{value_close}"
paramname ::= [a-zA-Z_] [a-zA-Z_0-9]*
value ::= leading_ws first_content rest
leading_ws ::= [ \t\r\n]*
first_content ::= [^ \t\r\n<=>]
{rest_rule}
rest_part ::= {ladder}
"#
    )
}
