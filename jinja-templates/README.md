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
Set `ATLAS_DISABLE_TEMPLATE_OVERRIDES=1` to ignore this directory entirely and
force every model onto its own template + the Rust behaviors.

> The former `holo3_1_moe.jinja` override was removed: it was a byte-copy of
> Holo-3.1's own template plus the three behaviors now handled in Rust.

## Naming Convention

The filename must match the model's `model_type` from `config.json`:

| Model | model_type | Override file |
|-------|-----------|---------------|
| Qwen3.5-35B/122B MoE | `qwen3_5_moe` | `qwen3_5_moe.jinja` |
| Qwen3.5-27B Dense | `qwen3_5` | `qwen3_5.jinja` |
| Qwen3-Next-80B | `qwen3_next` | `qwen3_next.jinja` |
| Nemotron-H | `nemotron_h` | `nemotron_h.jinja` |

## Priority

1. Override template from this directory — **opt-in by file presence**, unless
   `ATLAS_DISABLE_TEMPLATE_OVERRIDES=1`
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
