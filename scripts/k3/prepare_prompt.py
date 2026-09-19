#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare official K3 token-array completion requests; this is not a chat server."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizer import read_pattern, read_ranks
from xtml_reference import load_encoder, REVISION, SOURCE_HASHES


def encode_segments(encoding, segments):
    ids = []
    for segment in segments:
        if segment.allow_special:
            ids.extend(encoding.encode(segment.text, allowed_special="all"))
        else:
            ids.extend(encoding.encode(segment.text, disallowed_special=()))
    return ids


def prepare(source, messages, model, max_tokens, thinking, tools=None):
    import tiktoken
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned K3 source hash mismatch: {name}")
    encoder = load_encoder(source / "encoding_k3.py")
    config = json.loads((source / "tokenizer_config.json").read_text())
    ranks = read_ranks(source / "tiktoken.model")
    pattern = read_pattern((source / "tokenization_kimi.py").read_text())
    specials = {config["added_tokens_decoder"].get(str(i), {}).get(
        "content", f"<|reserved_token_{i}|>"): i for i in range(len(ranks), len(ranks) + 256)}
    if specials.get("<|end_of_msg|>") != 163586:
        raise ValueError("Unexpected official K3 end-of-message token")
    encoding = tiktoken.Encoding(name="k3-offline-prompt", pat_str=pattern,
                                mergeable_ranks=ranks, special_tokens=specials)
    segments = encoder.build_chat_segments(messages, tools=tools, thinking=thinking)
    ids = encode_segments(encoding, segments)
    if not ids or max_tokens < 1:
        raise ValueError("A nonempty prompt and positive max_tokens are required")
    payload = {"model": model, "prompt": ids, "max_tokens": max_tokens,
               "temperature": 0, "stream": False, "stop": ["<|end_of_msg|>", "[EOS]"]}
    receipt = {"schema": 1, "revision": REVISION, "prompt_tokens": len(ids),
               "thinking": thinking, "eos_token_id": 163586,
               "prompt_token_ids_sha256": hashlib.sha256(
                   json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
               "sources": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                   for name in ("encoding_k3.py", "tokenization_kimi.py",
                                "tokenizer_config.json", "tiktoken.model")},
               "output_contract": "Raw XTML; no reasoning/tool demultiplexing or automatic tool execution"}
    return payload, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--messages", type=Path, required=True, help="JSON array of official messages")
    parser.add_argument("--tools", type=Path, help="Optional JSON tool declarations; output remains raw XTML")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--thinking", choices=("on", "off"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload, receipt = prepare(args.source, json.loads(args.messages.read_text()), args.model,
                               args.max_tokens, args.thinking == "on",
                               json.loads(args.tools.read_text()) if args.tools else None)
    with args.output.open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
    receipt_path = args.output.with_suffix(args.output.suffix + ".receipt.json")
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(f"Prepared {receipt['prompt_tokens']} tokens; POST payload to /v1/completions")


if __name__ == "__main__":
    main()
