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

0. `--chat-template sharp|sharp-no-terse|PATH` (or `AVAROK_CHAT_TEMPLATE`, the
   flag wins), **opt-in and off by default**. It replaces whatever the steps below
   chose, drops the `openai/` variant so nothing shadows it, and applies even
   with `--disable-template-overrides`. `model` (the default) leaves steps 1-3
   untouched. See "Chat template choice" below.
1. Override template from this directory — **opt-in by file presence**, unless
   serving with `--disable-template-overrides`
2. Template from `tokenizer_config.json` / `chat_template.jinja` (the model's
   own — the default for models without an override file)
3. Default ChatML fallback (lowest priority)

## Chat template choice (`--chat-template`)

| Value | Template |
|---|---|
| `model` (default) | unchanged: steps 1-3 above |
| `sharp` | the embedded Sharp template for Qwen3.5/3.6/3.8 (terseness system prompt on, history reasoning kept, stricter tool-call format); attribution in `SHARP-CHAT-TEMPLATE-ATTRIBUTION.md` |
| `sharp-no-terse` | the same template with the terseness system prompt pinned off, to separate its effect from the retention and tool-format fixes |
| any other value | a Jinja file read from that path; an unreadable or empty file fails startup, and so does an unknown word such as `shrap` |

The value comes from `--chat-template`, else `AVAROK_CHAT_TEMPLATE`; an empty
value counts as unset. The boot log prints `chat template: sharp (<version>)`
and `/props` reports the selected template text. A choice also marks the
template as owning the Qwen XML tool prompt (when it declares `tools` and
renders the `<function=...>` format), so the parser's own tool prompt is not
injected a second time. Models with a checkpoint-native chat encoding
(DeepSeek V4, Kimi K3) refuse a template instead of ignoring the flag.

Per request, `chat_template_kwargs.terse` (true or false) reaches the template
as the variable `terse`; `--default-chat-template-kwargs '{"terse":false}'`
sets the server-wide default. Only the Sharp template reads it. Sharp is for
answer-style work with thinking on; keep the model's own template for every
kernel, KL, prefill and decode benchmark, because Sharp adds about 150 prompt
tokens to each request.

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
