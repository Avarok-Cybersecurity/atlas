# Jinja Template Overrides

By default Atlas renders chat from the **model's OWN** `chat_template.jinja` /
`tokenizer_config.json`. Atlas's cross-cutting behaviors — auto-closing a
dangling `<think>` before a `<tool_call>` in history, stripping inline
`<|think_on|>`/`<|think_off|>` control tokens, and mapping `reasoning_effort`
→ thinking — are applied in **Rust message-preprocessing** (see
`crates/spark-server/src/tokenizer/message_preprocess.rs`), so they work for
every model without a bespoke template copy.

Drop a `.jinja` file here named by `model_type` only when a model needs a
template fix that the Rust preprocessing can't express (e.g. MiniMax's
`_args.items()` iteration, Gemma-4's `strip_thinking` macro). The file's
presence is the **opt-in**: it takes precedence over the model's own template.
Serve with `--disable-template-overrides` to ignore this directory entirely
and force every model onto its own template + the Rust behaviors.

> `holo3_1_moe.jinja` is now REDUNDANT — it is a byte-copy of Holo-3.1's own
> template plus the three behaviors now handled in Rust (the fixture tests in
> `tokenizer/message_preprocess/tests.rs` prove Holo renders correctly off its
> own template). It is kept for the moment only because
> `tokenizer/tests.rs::render_holo_template_*` still reads it directly; it
> will be deleted together with those tests.

## Naming Convention

The filename must match the model's `model_type` from `config.json`:

| Model | model_type | Override file |
|-------|-----------|---------------|
| Qwen3.5-35B/122B MoE | `qwen3_5_moe` | `qwen3_5_moe.jinja` |
| Qwen3-Next-80B | `qwen3_next` | `qwen3_next.jinja` |
| Nemotron-H | `nemotron_h` | `nemotron_h.jinja` |

> `qwen3_5.jinja` (dense 27B family) was RETIRED 2026-08-14. It was a stale
> April-2026 strict subset of the checkpoints' own templates (no
> `preserve_thinking`, no Qwen3.8 `reasoning_effort` block), and because
> Qwen3.5-27B, Qwen3.6-27B AND Qwen3.8-27B all report `model_type =
> "qwen3_5"`, the one file silently forced an old template onto every new
> checkpoint generation. The dense family now renders model-first off each
> checkpoint's own `chat_template.jinja`; byte-parity of the Qwen3.6 render
> with the retired override is locked by golden tests in
> `crates/spark-server/src/tokenizer/tests/qwen_dense.rs`. Per-target
> template knobs (`preserve_thinking`) come from MODEL.toml `[behavior]`
> and per-request `chat_template_kwargs`, not from a template fork.

## Priority

1. Override template from this directory — **opt-in by file presence**, unless
   serving with `--disable-template-overrides`
2. Template from `tokenizer_config.json` / `chat_template.jinja` (the model's
   own — the default for models without an override file)
3. Default ChatML fallback (lowest priority)

## Usage

```bash
# Example: apply community fix for Qwen3.5 tool calling
curl -o jinja-templates/qwen3_5_moe.jinja \
  https://raw.githubusercontent.com/eugr/spark-vllm-docker/.../chat_template.jinja
```

The server logs which source was used:
```
Using override Jinja template from jinja-templates/qwen3_5_moe.jinja (7800 chars)
```
