# Research: llama.cpp GBNF & tool-call pipeline vs Atlas/xgrammar

**Date**: 2026-05-25
**Context**: Qwen3.6-35B-A3B-FP8 in opencode emits empty `<parameter=>` bodies under multi-turn agentic load. XGrammar's regex compiler does not enforce `\S` inside Kleene closures. Atlas needs to decide between fixing xgrammar, replacing with GBNF, or layering sampler-level enforcement.

---

## 1. GBNF: how llama.cpp enforces `\S` (and friends)

### 1.1 No `\S` shortcut — char ranges only

`src/llama-grammar.h` defines only nine element types: `LLAMA_GRETYPE_CHAR`, `LLAMA_GRETYPE_CHAR_NOT`, `LLAMA_GRETYPE_CHAR_RNG_UPPER`, `LLAMA_GRETYPE_CHAR_ALT`, `LLAMA_GRETYPE_CHAR_ANY`, plus `RULE_REF`, `ALT`, `END`, `TOKEN(_NOT)`. There is **no regex escape layer** — GBNF authors compose negated whitespace as `[^ \t\r\n]` (an `LLAMA_GRETYPE_CHAR_NOT` followed by char alternates). The character-class parser at `common/grammar-parser.cpp` `parse_sequence()` (~L614) lights up `CHAR_NOT` when it sees `[^` and emits members one by one.

### 1.2 `+`/`*` are structural, not Kleene closures over an NFA

llama.cpp does **not** compile to an NFA with ε-transitions. `handle_repetitions(min, max)` (~L568) expands quantifiers into *fresh synthesised rules at compile time* — for `+` (i.e. `min_times=1`):

```cpp
for (uint64_t i = 1; i < min_times; i++) {
    rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
}
```

The first occurrence is **inlined verbatim into the parent rule**, then a self-recursive rule reference handles the tail. Result: the at-least-one obligation lives in the rule shape itself — no ε-transition can skip it, no "optional Kleene closure sandwich" trick will accidentally produce zero matches. This is the structural property XGrammar's pushdown automaton does not give us for free.

### 1.3 The sampler: hard mask, no soft preference

`src/llama-grammar.cpp` `llama_grammar_apply_impl()` walks every candidate token's decoded UTF-8 against every active stack via `llama_grammar_match_char()` (~L884). Matches use a tidy XOR with the polarity bit:

```cpp
bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;
...
return std::make_pair(found == is_positive_char, pos);
```

Non-matching tokens are set to `-INFINITY` *before* sampling. So `[^ \t\r\n]+` literally cannot accept a whitespace-leading BPE token at position 1 — the logit is zero-probability, not low-probability. This is the same model that XGrammar implements, so the difference cannot be sampler semantics; it must be **the compiled grammar shape**.

### 1.4 JSON-schema → GBNF: `minLength` becomes shape, not constraint

`common/json-schema-to-grammar.cpp` `build_repetition(char_rule, min_items=1, max_items=∞)` emits `char char*` (i.e. the first `char` is mandatory, the rest are a tail). For tool-call arguments declared `{"type":"string", "minLength":1}`, the grammar prefix forces one non-`"`, non-`\\`, non-control byte before `*` becomes legal. **However**, `char` here is *any* non-quote/backslash/control byte, including spaces and tabs — llama.cpp does **not** enforce non-whitespace for tool string values, only non-empty. That's still strictly better than Atlas's current XGrammar behaviour, which can produce *zero-byte* values.

---

## 2. The XGrammar bug, located

`crates/xgrammar/src/regex/escape_handlers.rs:129` (Atlas) and the upstream `mlc-ai/xgrammar/cpp/regex_converter.cc` both contain:

```rust
b'S' => Some("[^[\\f\\n\\r\\t\\v\\u0020\\u00a0]"),
```

That's a malformed EBNF char class. The opening `[^` starts a negation; then `[` is consumed *as a literal class member*; the close `]` finishes the class. Effective semantics: "any char except `[`, `\f`, `\n`, `\r`, `\t`, `\v`, space, NBSP". **Spaces are excluded**, but so is `[`, and the bracket count is asymmetric. Worse: it's inside the rule body, so a downstream pushdown-automaton compiler may either reject the grammar, accept it as the above (over-broad) class, or — and this is what we see at runtime — wrap it inside a closure where the `\S` literal becomes ε-transitive.

The tests at `crates/xgrammar/src/regex/tests.rs:79-82` **codify the broken output** (`[^[\\f\\n\\r\\t\\v\\u0020\\u00a0]`) as the expected result. So the bug round-trips green.

Top-level `\S` is **structurally broken** in Atlas; inside-class `\S` (handled at `escape_handlers.rs:162`) emits the correct `\x00-\x08\x0E-\x1F\x21-\x9F\xA1-\U0010FFFF` range, which is why `[\s\S]+` looks like it works (it's a tautology that resolves to "any char") but `\S` by itself does not.

### Why the Kleene closure sandwich `[\s\S]*\S[\s\S]*` still fails

Even with the literal char-class bug aside, that pattern is a closure over a closure — the `\S` constraint sits between two ε-friendly Kleene stars. XGrammar's regex-to-EBNF lowering preserves quantifier symbols (see `regex/converter.rs:107`, `self.result.push(quant_char(c));`), and the downstream EBNF→pushdown compiler treats `X*` as "zero or more, ε-OK", so any path that takes ε on the first star and ε on the middle `\S` (which can be non-empty in theory but is not *structurally forced* in the rule shape) terminates at the close tag without ever consuming a non-whitespace byte. GBNF dodges this by inlining the minimum count into the rule shape (§1.2); XGrammar's NFA-style lowering does not.

---

## 3. llama.cpp's tool-call pipeline (PR #18675 autoparser)

1. **Chat template builder** (`common/chat.cpp`, `common/chat-peg-parser.cpp`) walks the model's tool schema and emits per-tool GBNF using `standard_constructed_tools()` for tagged formats. Each tool gets a `lazy` grammar that only activates after a trigger token (e.g. `<tool_call>`).
2. **Grammar compiler** lowers GBNF to the `llama_grammar_element` IR (§1.2).
3. **Sampler** hard-masks logits per active grammar stack (§1.3).
4. **Detokenizer** emits the raw text.
5. **PEG post-parser** re-parses the produced text against a hand-written PEG grammar to extract function name and arguments — this is where issues #20198 (arguments-as-object) and #20260 (text-before-`<tool_call>`) live. The post-parser is **independent** of the GBNF sampler-side grammar; it's the final structuring step.

For Qwen3 specifically there is no model-specific code path in `chat.cpp`; it falls through to the generic tagged-tools builder. Thinking/`<think>` handling is per-template (`THINK_START`/`THINK_END` for Ministral, Kimi K2, LFM2 — Qwen3 inherits the default). There is **no `preserve_thinking` flag for Qwen3** in upstream llama.cpp; the community-noted improvement is the PEG post-parser tolerating preamble text (#20260, still open as of search date).

Issue **#20164** (Qwen3.5-35B / Qwen3-Coder-Next) is Atlas's exact failure mode: under long context with multi-optional-arg tools, the model loops with one arg missing each time. The reporter's workaround — mark all params required — succeeds *because GBNF then forces `char+`-style structure for every required string*. Same fix surface as Atlas's `enforce_min_length_on_required_strings`.

---

## 4. GBNF vs XGrammar: enforcement strictness summary

| Property | GBNF (llama.cpp) | XGrammar (Atlas) |
|---|---|---|
| Compile model | Inline rule duplication + recursion | Regex → EBNF → pushdown-NFA |
| `+` semantics | Structural (`X X*`) | Quantifier-on-NFA-edge |
| `\S` shortcut | Not supported; must use `[^ \t\r\n]` | **Buggy literal** (`[^[\f\n\r\t\v ]`) |
| `[^…]` Kleene closure with embedded `\S` | N/A (no `\S`); negated class in `+` is structurally enforced | ε-transitive, **drops constraint** |
| Tool-call arg `minLength: 1` | `char char*` (non-empty, may be WS) | Was JSON-schema-enforced (broken under FP8 drift); now Atlas regex `[^ \t\r\n<][^<]*` |
| Char-class negation correctness | `CHAR_NOT` XOR — exact | Inside-class \S correct; **top-level \S malformed** |
| Sampler integration | Per-token hard-mask via stack walk | Per-token bitmask via FSM |

The sampler integration is functionally equivalent. The compile model is not. GBNF's rule-shape minimums survive any closure context; XGrammar's quantifiers do not.

---

## 5. Where Atlas already worked around it

`crates/spark-server/src/grammar/compile_tools.rs:258-266` rewrites the parameter-value pattern to:

```
(<parameter=[a-zA-Z_][a-zA-Z_0-9]*>[ \t\r\n]*[^ \t\r\n<][^<]*</parameter>\s*)+
```

This dodges `\S` entirely: `[^ \t\r\n<]` is an inside-class negated set (which the converter handles correctly via `escape_handlers.rs:158-163` ranges) and the `+` is on the outer block (structurally forced via xgrammar's quantifier expansion on the alternation, not on a Kleene-sandwiched `\S`). The trade-off documented in the file is that values starting with literal `<` won't match — acceptable for opencode workloads.

The MiniMax XML grammar at `compile_minimax_xml_tool_grammar` already applies F67 (short shared trigger) for the related "tool-call lockup" symptom. Same class of bug, different surface.

---

## 6. Top sources consulted

- ggml-org/llama.cpp `src/llama-grammar.{cpp,h}`, `common/grammar-parser.cpp`, `common/json-schema-to-grammar.cpp`, `common/chat.cpp`, `common/chat-peg-parser.cpp`
- llama.cpp issues #20164 (Qwen3 optional-arg loop), #20260 (PEG preamble), #20198 (arguments-as-object), #20345 (thinking + grammar)
- llama.cpp PR #18675 (autoparser refactor)
- mlc-ai/xgrammar `cpp/regex_converter.cc` (confirmed same `\S` bug upstream)
- Atlas `crates/xgrammar/src/regex/escape_handlers.rs:129`, `crates/xgrammar/src/regex/tests.rs:79-82`, `crates/spark-server/src/grammar/compile_tools.rs:200-266`
