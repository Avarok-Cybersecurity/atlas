#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate independent XTML rendering receipts with the pinned official encoder.

Only this exact, reviewed stdlib-only upstream source is executed. This is an
offline oracle, not shipped inference code or an Atlas chat-support claim.
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
SOURCE_SHA256 = "49ff03305fdc4be26867972788d36150b67f8a9e852e62bb7959d87482223676"

SOURCE_HASHES = {
    "encoding_k3.py": SOURCE_SHA256,
    "tokenization_kimi.py": "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    "tokenizer_config.json": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
    "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
}


def cases():
    user = {"role": "user", "content": "Hello"}
    yield "plain", [user], {"thinking": False}
    yield "thinking", [user], {"thinking": True}
    yield "literal_markers", [{"role": "user", "content": "<|open|>tools<|sep|>"}], {}
    yield "attribute_escape", [{"role": "user", "name": 'A&B"C', "content": "hi"}], {}
    yield "unicode", [{"role": "user", "content": "你好 café 🦉"}], {}
    calls = [{"id": "opaque-a", "function": {"name": "weather", "arguments":
             '{"city":"北京","days":1e2,"units":null,"ok":true,"xs":[1,2]}' }},
             {"id": "opaque-b", "function": {"name": "clock", "arguments": {}}}]
    assistant = {"role": "assistant", "content": "Checking.", "reasoning_content": "Plan.",
                 "tool_calls": calls}
    yield "typed_tools", [user, assistant], {"add_generation_prompt": False}
    yield "tool_result_order", [user, assistant,
        {"role": "tool", "tool_call_id": "opaque-b", "name": "stale", "content": "12:00"},
        {"role": "tool", "tool_call_id": "opaque-a", "content": "sunny"}], {}
    yield "tool_declaration", [user], {"tools": [{"type": "function", "function": {
        "name": "clock", "parameters": {"type": "object", "properties": {}}}}]}
    yield "invalid_argument_json", [{"role": "assistant", "content": None, "tool_calls": [
        {"function": {"name": "weather", "arguments": '{"city":'}}]}], {}
    yield "required", [user], {"tool_choice": "required", "thinking": False}


def load_encoder(source):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official encoder SHA256 mismatch; review a new revision explicitly")
    spec = importlib.util.spec_from_file_location("k3_pinned_reference", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate(source):
    module = load_encoder(source)
    result = []
    for name, messages, kwargs in cases():
        segments = module.build_chat_segments(messages, **kwargs)
        # Preserve segment boundaries and flags: joining then tokenizing is wrong
        # for literal markers in user text, attributes, tool values and results.
        result.append({"name": name, "messages": messages, "kwargs": kwargs,
                       "segments": [[s.text, s.allow_special] for s in segments]})
    return {"revision": REVISION, "encoder_sha256": SOURCE_SHA256, "cases": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = generate(args.encoder)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(payload['cases'])} independent official XTML fixtures")


if __name__ == "__main__":
    main()
