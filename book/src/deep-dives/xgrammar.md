# Constrained Decoding (XGrammar)

Constrained decoding lets Atlas force the model to produce output that conforms to a grammar — the subset of tokens that could continue the current partial output while keeping the output valid is computed at every step; invalid tokens get their logits set to `-inf` before sampling.

Atlas uses **XGrammar** for this. It is the machinery that makes tool calls reliable (no invented field names, no broken JSON/XML), and it's the substrate for any "structured output" feature (`response_format`, JSON schema conforming, etc.).

## The problem

Without constrained decoding, an LLM producing a tool call can:

- Invent field names that don't exist in the schema.
- Forget commas.
- Close JSON objects with the wrong bracket type.
- Break inside a string literal because the tokenizer merged characters across a boundary.
- Switch formats mid-call (XML close tag when the template expects JSON).

Every one of these has been seen in the wild on large models, even Qwen3.5-class models. A clean tool-call parser can't fix them — the broken token sequence comes out of the sampler before the parser sees it.

Constrained decoding intervenes upstream: at the sampler, we know which tokens are syntactically legal *next* given the current state of the output. Everything else gets masked.

## XGrammar's trick

The naive version of constrained decoding is expensive: at every sampling step, compile the grammar against the current output prefix and enumerate the allowed next tokens. That's a parser pass over the vocabulary (tens of thousands of tokens) per step — prohibitive.

XGrammar ([paper](https://arxiv.org/abs/2411.15100), Xiamen University / CMU 2024) does two things that make it tractable:

1. **Pre-compiles a token-bitmap automaton.** For each grammar state, precompute a bitmask over the whole vocabulary indicating which tokens are legal. At runtime, transition the automaton by the sampled token, look up the new bitmap — O(1) per step.
2. **Handles tokeniser boundary cases.** Real-world tokenizers merge characters across grammar-legal boundaries (e.g. the byte-pair-encoded token `",\n"` crosses a JSON key/value boundary). XGrammar's compiler handles these *at compile time* by enumerating all byte-level prefixes each token can legally complete.

The cost is a grammar compilation step (~ms for typical JSON schemas), amortised across all requests using that schema.

## How Atlas uses it

Atlas ships XGrammar as a vendored dependency (`vendor/xgrammar-rs/`) with Rust bindings. The call sites:

- **Tool calls** — when the request includes `tools: [...]`, Atlas derives an XGrammar grammar from the function schemas + the model's tool-call format (Hermes JSON, Qwen3-coder XML, Mistral JSON). The grammar enforces: opening delimiter → valid function name → opening args bracket → schema-conforming JSON/XML → closing delimiter. `--tool-max-tokens` caps the total argument-generation length.
- **Response-format structured output** — OpenAI-compatible `response_format: {type: json_schema, json_schema: {...}}`. Atlas compiles the schema into an XGrammar grammar and constrains the entire response.
- **Reasoning boundaries** — the reasoning parser uses a lightweight grammar to enforce that `<think>...</think>` blocks close cleanly when `--max-thinking-budget` kicks in, preventing the unclosed-think bug that blocked Claude Code compatibility on Qwen3.6.

At the sampling step:

```rust
let logits = /* model logits [vocab_size] */;
if let Some(grammar) = active_grammar_for_request {
    let mask = grammar.current_token_mask();    // &[u32] bitmap
    apply_mask_in_place(&mut logits, mask);     // -inf for disallowed tokens
}
let token = sampler.sample(&logits);
grammar.advance(token);                         // transition automaton
```

The mask-apply and advance calls are both O(1) — a single bitmap test per token, a single state transition per step.

## The integration history

XGrammar integration shipped across two substantial work items:

- **`xgrammar-integration-plan`** (initial) — wiring the FFI, compiling grammars on request, applying masks in the sampler, streaming boundary handling.
- **`xgrammar2-upgrade-plan`** (current) — moved to the 2.0 API with better handling of long schemas, on-the-fly grammar recompilation, and support for Anthropic-style nested-XML tool-call formats.

The payoff compounds with MTP: constrained decoding inside an MTP draft mask blocks draft tokens that would break the grammar, raising the draft acceptance rate from ~70% to ~95% during tool calls. The +37% tool-call throughput win (referenced in the [MTP chapter](./mtp.md)) is a direct result.

## Opencode & markdown fences

A specific bug worth noting: when a model emits a tool call inside a markdown code fence, Atlas's tool-call parser originally ate the surrounding fence characters — the closing backticks came through as "extra content" and broke downstream code that expected clean JSON. Fixed in wave-1 of the bug sweeps by making the parser markdown-fence aware; XGrammar then enforces the fence is balanced.

A related hallucination class: the Qwen3-coder XML format allows the model to emit the literal string `</tool_call>` inside a JSON string value. The parser now disambiguates, and XGrammar's grammar masks it at the source.

## When to turn it off

XGrammar is lightweight but not free. For vanilla free-form generation (no tools, no response_format), the sampler skips the mask path entirely — there's no `active_grammar`. For workloads that explicitly want the model to deviate from a schema (creative tool exploration), setting `tool_choice: "none"` disables the grammar.

The one place where constrained decoding can interact badly with sampling: very low-entropy grammars combined with `temperature=0` greedy sampling can produce repetitive output if the grammar masks the "natural" next token. `--default-top-n-sigma` and `--default-min-p` help; dropping `temperature` below 0.1 is rarely worth it on constrained paths.

## Files to read

- `vendor/xgrammar-rs/` — the XGrammar Rust bindings.
- `crates/spark-server/src/grammar.rs` — per-request grammar compilation and mask application.
- `crates/spark-server/src/tool_parser.rs` — tool-call format → grammar translation.
- `docs/design/xgrammar-integration-plan.md`, `xgrammar2-upgrade-plan.md`.
- `docs/design/tool-calling-gap-analysis.md` — the broader tool-call quality story.
