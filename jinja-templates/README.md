# Jinja Template Overrides

Drop `.jinja` files here named by `model_type` to override the chat template
from `tokenizer_config.json`. This lets you apply community fixes or custom
templates without re-downloading model weights.

## Naming Convention

The filename must match the model's `model_type` from `config.json`:

| Model | model_type | Override file |
|-------|-----------|---------------|
| Qwen3.5-35B/122B MoE | `qwen3_5_moe` | `qwen3_5_moe.jinja` |
| Qwen3.5-27B Dense | `qwen3_5` | `qwen3_5.jinja` |
| Qwen3-Next-80B | `qwen3_next` | `qwen3_next.jinja` |
| Nemotron-H | `nemotron_h` | `nemotron_h.jinja` |

## Priority

1. Override template from this directory (highest priority)
2. Template from `tokenizer_config.json` (ships with model weights)
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
