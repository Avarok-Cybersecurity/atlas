# XGrammar / MLC-LLM Feature Deep-Dive — What Atlas Isn't Using

**Date**: 2026-05-25
**Scope**: Atlas vendored xgrammar (`crates/xgrammar/`) vs. upstream `mlc-ai/xgrammar` main branch
**Trigger**: Empty `<parameter=key></parameter>` bodies slipping past Atlas's tier-0 regex `(<parameter=…>[ \t\r\n]*[^ \t\r\n<][^<]*</parameter>\s*)+` even with the `+` Kleene quantifier and the `[^ \t\r\n<]` first-byte anchor

---

## Executive Summary

Atlas's `compile_qwen3_coder_tool_grammar` routes the parameter body through `{"type":"regex","pattern": …}` and tries to enforce min-length-1 by hand inside that pattern. **Upstream xgrammar offers four progressively-stronger structural enforcement mechanisms that Atlas is not using**: (1) `qwen_xml`-styled `json_schema` content type with `minLength:1` on required strings — the schema-converter compiles this to a literal `[^]{1,}` EBNF repetition node (`schema/gen_scalar.rs:88`), bypassing the regex-FSM determinization edge case entirely; (2) the new `plus` / `repeat` / `star` / `optional` Format types that wrap any inner content in EBNF `+`/`{min,max}` at the AST layer; (3) the new `dispatch` / `token_dispatch` / `token_triggered_tags` types that constrain at token-id granularity rather than byte granularity (which avoids byte-level non-determinism around `\S` matchers altogether); and (4) per-vocab token-level structural rules. The Rust port in `crates/xgrammar/src/structural_tag/` is frozen at the pre-2025 upstream snapshot and has *none* of these — only `const_string`, `json_schema`, `any_text`, `sequence`, `or`, `tag`, `triggered_tags`, `tags_with_separator`, `grammar`, `regex`, and the now-deprecated `qwen_xml_parameter`. Top recommendation: switch the Qwen3-Coder parameter body from `regex` to `json_schema` style `qwen_xml` with the existing `enforce_min_length_on_required_strings`-augmented schema, which is the canonical upstream pattern and gives true FSM-level min-length-1 enforcement on every required string property without an Atlas-side port.

---

## 1. Why the `[\s\S]*\S[\s\S]*` Sandwich Doesn't Enforce

xgrammar's regex-to-EBNF converter (`regex/converter.rs:105-118`) treats `*` / `+` / `?` / `{m,n}` as **non-deterministic** quantifiers — the converter comment explicitly says "all repetition is handled non-deterministically." The Earley/FSM matcher then computes the union over all match paths. For `[\s\S]*\S[\s\S]*`:

- `\S` upstream-C++ expands to `[^[\f\n\r\t\v  ]` (and Atlas vendors the same — `regex/escape_handlers.rs:129`)
- The two `[\s\S]*` Kleene closures are both nullable (`*` allows zero repetitions)
- During FSM construction the ε-closure of "before `\S`" merges with the ε-closure of "after `\S`" through the trailing `[\s\S]*` accept-edge
- Net result: there exists an accepting path that consumes zero bytes of `\S`

This is **not a bug** in xgrammar — it's standard NFA-to-DFA semantics. The Kleene-sandwich pattern simply cannot enforce a non-empty middle when the FSM is constructed without backreferences.

Atlas's current tier-0 `[ \t\r\n]*[^ \t\r\n<][^<]*` (commit 2026-05-25) **does** enforce non-empty because `[^ \t\r\n<]` is consumed unconditionally (no preceding `*`). Empty-body failures must therefore come from a different path — likely a `</tool_call>` early-exit from the OUTER `+` quantifier closing before any `<parameter=…>` block emits a value (i.e. zero parameter blocks). Inspect `tag_entries_fallback` and trigger ordering before assuming the regex itself is wrong.

## 2. The Schema-Converter Path Atlas Isn't Using

`crates/xgrammar/src/schema/gen_scalar.rs:82-88` shows the XML-string path inside the `qwen_xml` schema converter:

```rust
if spec.min_length != 0 || spec.max_length != -1 {
    let repetition = if spec.max_length == -1 {
        format!("{{{},}}", spec.min_length)
    } else { format!("{{{},{}}}", spec.min_length, spec.max_length) };
    return Ok(format!("[^]{repetition}"));   // ← literal EBNF {1,}
}
```

When `qwen_xml` style is selected AND the property has `minLength`, the converter emits `[^]{1,}` (one or more arbitrary bytes) directly as the EBNF body inside `<parameter=KEY>…</parameter>`. This is a **deterministic structural enforcement** — no NFA ε-edges, no regex-determinization sandwich problem. The matcher's bitmask physically cannot accept `</parameter>` after `<parameter=key>` until at least one byte is consumed.

Atlas already has `enforce_min_length_on_required_strings` (`schema.rs:60`) injecting `minLength:1`. It is plumbed for every `compile_*_tool_grammar` except — critically — `compile_qwen3_coder_tool_grammar`, which discards the augmented schema (`let _ = &st.schema;` on line 246) and uses regex instead.

**One-line fix**: change the Qwen3-Coder content type from `regex` to:
```rust
"content": {"type": "json_schema", "style": "qwen_xml", "json_schema": st.schema}
```
Drops the bespoke regex, uses the deterministic schema FSM, and reuses the schema sanitization + minLength augmentation already in place.

Trade-offs the existing comment flags from "pre-2026-05-25 json_schema" attempts (filePath cascades, path truncation) were against `style: "json"` (escaped JSON-string body), NOT `style: "qwen_xml"` — the XML style uses `<parameter=KEY>VALUE</parameter>` framing where `VALUE` is plain text bounded by `</parameter>`, not a JSON-escaped string.

## 3. New Format Types Atlas's Vendored Port Lacks

Upstream `python/xgrammar/structural_tag.py` now defines these Format types (verified via WebFetch, 2026-05-25). Atlas's `crates/xgrammar/src/structural_tag/format.rs` has none:

| Upstream type            | Purpose                                                        | Atlas use case                                         |
|--------------------------|----------------------------------------------------------------|--------------------------------------------------------|
| `plus`                   | EBNF `+` over any nested Format                                | Wrap parameter block: 1+ `<parameter=…>…</parameter>`  |
| `repeat` (min, max)      | EBNF `{m,n}` over any nested Format                            | Bounded tool-call counts; ≤N parameters                |
| `star`                   | EBNF `*` over any nested Format                                | Optional preamble between tool calls                   |
| `optional`               | EBNF `?` over any nested Format                                | Optional `_think` field                                |
| `token` / `exclude_token`| Single token by ID or string                                   | Hard-block EOS in `required` mode (no string trigger)  |
| `any_tokens`             | Zero or more tokens, excluding set                             | Replace `any_text` for token-ID-precise free body      |
| `dispatch`               | Like `triggered_tags` but `loop:false` runs nested format once | "exactly one tool call" with byte-trigger              |
| `token_dispatch`         | `dispatch` keyed on token IDs                                  | Token-precise trigger that bypasses BPE re-tokenization|
| `token_triggered_tags`   | `triggered_tags` keyed on token IDs                            | Eliminates "trigger string spans BPE boundary" misses  |

The upstream docs page (xgrammar.mlc.ai/docs/api/python/structural_tag) **explicitly recommends** the pattern Atlas needs:

> "For non-empty parameter value enforcement, wrap parameter content with `PlusFormat(content=RegexFormat(pattern="[^<>]"))` to require at least one non-tag character."

This is structurally identical to what Atlas wants but expressed at the Format AST layer instead of inside a regex pattern. The `PlusFormat` wrapper compiles to a top-level EBNF `+` rule reference — the matcher cannot accept the outer end-marker before consuming at least one match of the inner regex.

## 4. `qwen_xml_parameter` Is Deprecated Upstream

Upstream `structural_tag.py:332-361` (verified 2026-05-25):

> "Deprecated. Use `JSONSchemaFormat` with `style='qwen_xml'` instead. This format remains available so existing serialized structural tags with `{'type':'qwen_xml_parameter'}` can still be loaded."

Atlas's parser (`structural_tag/parser.rs:90`) still maps `qwen_xml_parameter` to `JsonSchema { style: QwenXml }` — accidentally already the upstream-recommended path. The misleading comments in `compile_tools.rs:163-167` say Atlas "uses XGrammar's `qwen_xml_parameter` content type" but the code uses `regex` (line 262). Either align comment to code or — better — adopt the json_schema/qwen_xml path (§2).

## 5. MLC-LLM's Sampler Side

MLC-LLM's high-level engine layer (`python/mlc_llm/serve/engine.py`) is purely an OpenAI proxy. The actual enforcement is in C++ `serve/threaded_engine.cc` and `cpp/serve/grammar/*`. From PR/issue mining: MLC-LLM does **not** have a sampler-side "second-chance" rejection — when the grammar mask accepts a token, that token is emitted unconditionally. There is no MLC-LLM equivalent of Atlas's post-parse `validate_single_tool_call` retry path. The grammar must therefore be tight by construction. This argues *against* Atlas's current "permissive grammar + tight host-side validator" split for the Qwen3-Coder path.

## 6. Notable 2025–2026 Upstream Fixes That Matter

- **PR #579 (2026-03-31, merged)** — "accept `{n,-1}` as unbounded repeat in EBNF parser". Confirms `min_length:1,max_length:-1` => `{1,}` is the canonical idiom; Atlas can rely on it.
- **PR #585 (2026-04-13, merged)** — "remove restriction of unlimited". Loosens analyzer rules for unlimited content; relevant if Atlas attempts nested `any_text` inside `plus`.
- **PR #595 (2026-04-26, open)** — `additionalProperties:true` + non-required mask bug. Atlas's `sanitize_schema_for_grammar` injects `additionalProperties:true` for empty objects (line 256-260); audit whether this is hit.
- **PR #630 (2026-05-05, merged)** — RepetitionRangeExpander segfault. Affects `Repeat` expr lookup — Atlas's vendored `grammar/parser_repeat.rs:15` is implicated.
- **PR #641 (2026-05-14, open)** — reject empty enum arrays. Atlas already handles this (`sanitize.rs`).
- **Issue #443 (2025-09-24, open)** — "Debugging triggered_tags activation" — no maintainer response after 8 months. Atlas's symptom (model emits `<parameter=key></parameter>` despite a `+`-quantified regex) is structurally adjacent; do not expect upstream help.

## 7. Concrete Atlas-Side Recommendations (Ranked)

1. **Switch Qwen3-Coder parameter body to `json_schema` style `qwen_xml` with `minLength:1`** — schema converter emits literal `[^]{1,}` EBNF; deterministic FSM enforcement; reuses existing schema augmenter; no port work; ~10 lines changed in `compile_tools.rs:258-266` and `:340-346`. See §2.
2. **Port `PlusFormat` / `RepeatFormat` to the vendored structural-tag AST** — `format.rs`, `parser.rs`, `converter.rs`, ~150 LoC; gives Atlas a generic "1+ of X" wrapper for cases where `json_schema` is too rigid (e.g., bare regex content). The upstream-recommended idiom `Plus(Regex("[^<>]"))` becomes available. See §3.
3. **Replace `triggered_tags` byte-triggers with `token_triggered_tags`** — Qwen3-Coder's `<tool_call>` is *literally* a special token (id 151657 on the Qwen3 tokenizer). Triggering on the token ID instead of the byte string eliminates the entire class of "trigger detection lags by one BPE chunk" issues that Atlas chases via `compile_minimax_xml_tool_grammar`'s SHORT-trigger comment block. See §3.
4. **Audit the OUTER `+` quantifier in tier-0 regex for early `</tool_call>` exit** — current regex matches `(…)+ ` at the *content* layer but the structural-tag converter wraps the content in a `tag` whose `end` (`\n</function>\n</tool_call>`) competes with the inner `+`. If the model samples `</function>` immediately after `<function=name>\n`, the OUTER end fires before any `<parameter=…>` block — yielding the observed empty call. Add `at_least_one:true` on a synthetic inner `triggered_tags` for `<parameter=` itself, or insert a `ConstString("<parameter=")` requirement.
5. **Vendor sync** — Atlas's xgrammar fork is ~6 months behind upstream; pick up PR #585 (unlimited fix), #630 (Repeat segfault), and the new Format types as a single rebase. Net adds ~800 LoC vs. ~80 LoC removed from `sanitize.rs` workarounds that upstream now handles natively.

---

**File locations referenced**:
- `/workspace/atlas-mtp/crates/spark-server/src/grammar/compile_tools.rs` (Atlas tool-grammar compiler)
- `/workspace/atlas-mtp/crates/spark-server/src/grammar/schema.rs` (minLength augmentation, schema sanitization)
- `/workspace/atlas-mtp/crates/xgrammar/src/structural_tag/format.rs` (Format AST — incomplete vs upstream)
- `/workspace/atlas-mtp/crates/xgrammar/src/structural_tag/parser.rs` (Format parser — handles qwen_xml_parameter deprecated alias)
- `/workspace/atlas-mtp/crates/xgrammar/src/schema/gen_scalar.rs` (XML-string EBNF generator with `[^]{1,}` path)
- `/workspace/atlas-mtp/crates/xgrammar/src/schema/converter.rs` (XML_STRING TagDispatch rule definition)
- `/workspace/atlas-mtp/crates/xgrammar/src/regex/escape_handlers.rs` (\S → `[^[\f\n\r\t\v...]`)
- `/workspace/atlas-mtp/crates/xgrammar/src/regex/converter.rs` (regex quantifier handling — non-deterministic)
