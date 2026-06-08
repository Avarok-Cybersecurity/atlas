#!/usr/bin/env python3
"""Reproduce Atlas's exact tokenization of the probe JSON.

Uses Atlas's own OpenAI-variant Jinja template (jinja-templates/openai/qwen3_5_moe.jinja)
to render the same string Atlas sees, then encodes via the model's HF tokenizer.

Writes /tmp/atlas_tokens_dgx2.json in the same format as
/tmp/atlas_tokens.json so hf_dual_forward.py can consume it.
"""
from __future__ import annotations

import json
import pathlib

import jinja2
from transformers import AutoTokenizer

TEMPLATE_PATH = pathlib.Path("/workspace/atlas-mtp/jinja-templates/openai/qwen3_5_moe.jinja")
TOKENIZER_SNAP = "/workspace/.cache/huggingface/Qwen3.6-35B-A3B-FP8-dequanted-BF16"
PROBE_PATH = pathlib.Path("/workspace/atlas-dumps/numdrift/atlas_turn11_probe.json")
OUT_PATH = pathlib.Path("/tmp/atlas_tokens_dgx2.json")
TARGET_TOKEN_COUNT = 9780  # what Atlas reports today


def normalize_tool_call_arguments(messages):
    """Atlas's chat_impl.rs F76: pre-parse tool_call argument strings into dicts."""
    out = []
    for m in messages:
        m2 = dict(m)
        if m2.get("tool_calls"):
            new_calls = []
            for tc in m2["tool_calls"]:
                tc2 = dict(tc)
                if tc2.get("function") and isinstance(tc2["function"].get("arguments"), str):
                    fn2 = dict(tc2["function"])
                    try:
                        fn2["arguments"] = json.loads(fn2["arguments"])
                    except Exception:
                        pass
                    tc2["function"] = fn2
                new_calls.append(tc2)
            m2["tool_calls"] = new_calls
        out.append(m2)
    return out


def main() -> None:
    probe = json.loads(PROBE_PATH.read_text())
    template_src = TEMPLATE_PATH.read_text()

    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )
    tmpl = env.from_string(template_src)

    messages = normalize_tool_call_arguments(probe["messages"])
    tools = probe.get("tools")

    rendered = tmpl.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="high",
        disable_tool_steering=False,
        add_vision_id=False,
    )
    print(f"rendered len chars: {len(rendered)}")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_SNAP)
    ids = tok(rendered, add_special_tokens=False, return_tensors=None)["input_ids"]
    print(f"token count: {len(ids)}")
    print(f"first 10: {ids[:10]}")
    print(f"last 10:  {ids[-10:]}")

    out = {
        "prompt_len": len(ids),
        "all_tokens": ids,
        "generated_tokens": [],
    }
    OUT_PATH.write_text(json.dumps(out))
    print(f"wrote {OUT_PATH}")

    if len(ids) != TARGET_TOKEN_COUNT:
        print(
            f"\nWARN: count {len(ids)} != target {TARGET_TOKEN_COUNT} "
            f"(Atlas-reported); template/encoder difference may bias HF dump"
        )
    else:
        print(f"\nMATCH: count == {TARGET_TOKEN_COUNT}")


if __name__ == "__main__":
    main()
