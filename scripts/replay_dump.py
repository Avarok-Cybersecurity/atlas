#!/usr/bin/env python3
"""Replay captured requests from a `--dump` JSONL file.

The server's `--dump` path writes each incoming chat-completions body
verbatim as JSONL. This script reads that file and re-POSTs selected
requests so we can:

  1. Reproduce a failing turn against a fresh server (wiped caches)
     to disambiguate whether the failure is prompt-induced or
     state-induced.
  2. Replay a full multi-turn sequence under different server flags
     (Phase 2 of the degeneration-investigation plan).

Usage:
    replay_dump.py [--host HOST:PORT] [--dump PATH] [--variant TAG]
                   (--seq N | --full | --list)
                   [--stream / --no-stream]
                   [--override KEY=VAL ...]
                   [--out PATH]

Examples:
    # Phase 1: replay the failing turn on a fresh server.
    replay_dump.py --seq 3 --variant P1 --out /workspace/replay-seq3-P1.jsonl

    # Phase 2 V4: replay all 3 turns with rep_penalty=1.05 injected.
    replay_dump.py --full --variant V4 --override repetition_penalty=1.05 \\
                   --out /workspace/replay-full-V4.jsonl

    # List what's in the dump.
    replay_dump.py --list
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_HOST = "localhost:8888"
DEFAULT_DUMP = "/workspace/atlas-opencode-dump-original.jsonl"


def load_requests(path: str) -> list[dict]:
    """Return the dump's request entries in file order."""
    reqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warn: bad line: {e}", file=sys.stderr)
                continue
            if o.get("kind") == "request":
                reqs.append(o)
    return reqs


def apply_overrides(body: dict, overrides: list[str]) -> dict:
    """Apply KEY=VAL overrides to a request body (shallow, numeric-aware)."""
    for ov in overrides:
        if "=" not in ov:
            print(f"warn: override missing '=': {ov!r}", file=sys.stderr)
            continue
        k, v = ov.split("=", 1)
        # Try numeric, then bool, then leave as string.
        try:
            parsed: object = int(v)
        except ValueError:
            try:
                parsed = float(v)
            except ValueError:
                lower = v.lower()
                if lower in ("true", "false"):
                    parsed = lower == "true"
                else:
                    parsed = v
        body[k] = parsed
    return body


def parse_sse_response(raw: str) -> dict:
    """Condense an SSE stream into a summary for decision-gating.

    Returns dict with: content (concatenated), tool_calls (parsed),
    finish_reason, usage, first_chunk_at_ms (approximate), num_chunks.
    """
    content = ""
    tool_calls: list[dict] = []
    finish_reason = None
    usage = None
    num_chunks = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        num_chunks += 1
        choices = obj.get("choices") or []
        if not choices:
            if obj.get("usage"):
                usage = obj["usage"]
            continue
        c0 = choices[0]
        delta = c0.get("delta") or {}
        if isinstance(delta.get("content"), str):
            content += delta["content"]
        if isinstance(delta.get("tool_calls"), list):
            for tc in delta["tool_calls"]:
                idx = tc.get("index", 0)
                while len(tool_calls) <= idx:
                    tool_calls.append({"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if "name" in fn and fn["name"]:
                    tool_calls[idx]["name"] = fn["name"]
                if "arguments" in fn and fn["arguments"]:
                    tool_calls[idx]["arguments"] += fn["arguments"]
        if c0.get("finish_reason"):
            finish_reason = c0["finish_reason"]
        if obj.get("usage"):
            usage = obj["usage"]
    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": usage,
        "num_chunks": num_chunks,
    }


def parse_anthropic_sse_response(raw: str) -> dict:
    """Condense an Anthropic /v1/messages SSE stream into a summary.

    Anthropic's stream emits typed events (message_start,
    content_block_start, content_block_delta, content_block_stop,
    message_delta, message_stop) instead of OpenAI's flat
    `choices[0].delta.{content,tool_calls}` shape. We aggregate to the
    same summary fields so decision-gating stays uniform.
    """
    content = ""
    tool_calls: list[dict] = []
    finish_reason = None
    usage = None
    num_chunks = 0
    # Tracks per-content-block state so input_json deltas accumulate
    # into the right tool_call entry.
    block_index_to_tc: dict[int, int] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        num_chunks += 1
        ev = obj.get("type") or ""
        if ev == "message_start":
            msg = obj.get("message") or {}
            if msg.get("usage"):
                usage = msg["usage"]
        elif ev == "content_block_start":
            idx = obj.get("index", 0)
            block = obj.get("content_block") or {}
            if block.get("type") == "tool_use":
                tc_idx = len(tool_calls)
                tool_calls.append(
                    {"name": block.get("name") or "", "arguments": ""}
                )
                block_index_to_tc[idx] = tc_idx
        elif ev == "content_block_delta":
            idx = obj.get("index", 0)
            delta = obj.get("delta") or {}
            dt = delta.get("type")
            if dt == "text_delta":
                content += delta.get("text") or ""
            elif dt == "input_json_delta":
                tc_idx = block_index_to_tc.get(idx)
                if tc_idx is not None:
                    tool_calls[tc_idx]["arguments"] += delta.get(
                        "partial_json"
                    ) or ""
        elif ev == "message_delta":
            d = obj.get("delta") or {}
            if d.get("stop_reason"):
                finish_reason = d["stop_reason"]
            if obj.get("usage"):
                # Anthropic spec: usage on message_delta is the
                # cumulative output_tokens count.
                usage = (usage or {}) | obj["usage"]
        # message_stop / content_block_stop / ping → no aggregation
    # Map Anthropic stop reasons back to OpenAI vocabulary so the
    # downstream degeneration_fingerprint logic interprets them
    # consistently.
    if finish_reason == "tool_use":
        finish_reason = "tool_calls"
    elif finish_reason == "end_turn":
        finish_reason = "stop"
    elif finish_reason == "max_tokens":
        finish_reason = "length"
    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": usage,
        "num_chunks": num_chunks,
    }


def parse_anthropic_blocking_response(obj: dict) -> dict:
    """Condense an Anthropic non-streaming /v1/messages response."""
    content_text = ""
    tool_calls: list[dict] = []
    for block in obj.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            content_text += block.get("text") or ""
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "name": block.get("name") or "",
                    "arguments": json.dumps(block.get("input") or {}),
                }
            )
    stop = obj.get("stop_reason")
    if stop == "tool_use":
        finish_reason = "tool_calls"
    elif stop == "max_tokens":
        finish_reason = "length"
    elif stop == "end_turn":
        finish_reason = "stop"
    else:
        finish_reason = stop
    return {
        "content": content_text,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": obj.get("usage"),
        "num_chunks": 1,
    }


def degeneration_fingerprint(summary: dict) -> dict:
    """Classify a response against the Phase 1 decision-gate definition.

    Degeneration = >=100 consecutive whitespace tokens OR EOS with no
    tool_call OR completion_tokens>=200 with <5% non-whitespace
    content. Returns a dict with booleans + diagnostic counts.
    """
    content = summary.get("content") or ""
    tool_calls = summary.get("tool_calls") or []
    finish = summary.get("finish_reason")
    usage = summary.get("usage") or {}
    ctoks = int(usage.get("completion_tokens") or 0)

    total = len(content)
    whitespace = sum(1 for ch in content if ch in " \t\n\r")
    non_ws_ratio = (total - whitespace) / total if total else 0.0

    # Longest consecutive whitespace run (proxy for "100 consecutive
    # whitespace tokens" — we have characters, not tokens, but this is
    # the best available signal without a tokenizer in the replay).
    max_run, run = 0, 0
    for ch in content:
        if ch in " \t\n\r":
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    has_real_tc = any((tc.get("name") or "") for tc in tool_calls)
    # Character-count proxy for the "100 consecutive whitespace tokens"
    # clause — tokens are 1-4 chars; 100 tokens ~ 100-400 chars. Use 100
    # as the conservative floor.
    collapsed_ws = max_run >= 100
    long_non_ws_starved = ctoks >= 200 and non_ws_ratio < 0.05
    stopped_no_tc = finish == "stop" and not has_real_tc and ctoks > 50

    degenerate = collapsed_ws or long_non_ws_starved or stopped_no_tc
    return {
        "degenerate": degenerate,
        "reason_collapsed_ws": collapsed_ws,
        "reason_long_non_ws_starved": long_non_ws_starved,
        "reason_stopped_no_tc": stopped_no_tc,
        "completion_tokens": ctoks,
        "max_whitespace_run_chars": max_run,
        "non_whitespace_ratio": round(non_ws_ratio, 3),
        "has_real_tool_call": has_real_tc,
        "finish_reason": finish,
    }


def post_request(host: str, endpoint: str, body: dict) -> tuple[str, dict]:
    """POST `body` to http://{host}{endpoint}. Returns (raw_response, summary)."""
    body_path = "/tmp/replay_body.json"
    with open(body_path, "w") as f:
        json.dump(body, f)
    url = f"http://{host}{endpoint}"
    curl_cmd = [
        "curl",
        "-sN",
        "--max-time",
        "300",
        url,
        "-H",
        "Content-Type: application/json",
    ]
    # Anthropic clients always send anthropic-version; stay close to
    # what Claude Code actually transmits so the server's parsing
    # path matches production.
    if endpoint == "/v1/messages":
        curl_cmd.extend(["-H", "anthropic-version: 2023-06-01"])
    curl_cmd.extend(["-d", f"@{body_path}"])
    # --no-buffer for streaming; identical behavior for non-stream.
    r = subprocess.run(
        curl_cmd,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(
            f"error: curl exit={r.returncode} stderr={r.stderr[:400]}",
            file=sys.stderr,
        )
    raw = r.stdout
    is_anthropic = endpoint == "/v1/messages"
    if body.get("stream"):
        summary = (
            parse_anthropic_sse_response(raw)
            if is_anthropic
            else parse_sse_response(raw)
        )
    else:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return raw, {"parse_error": raw[:400]}
        if is_anthropic:
            summary = parse_anthropic_blocking_response(obj)
        else:
            msg = (obj.get("choices") or [{}])[0].get("message") or {}
            summary = {
                "content": msg.get("content") or "",
                "tool_calls": msg.get("tool_calls") or [],
                "finish_reason": (obj.get("choices") or [{}])[0].get("finish_reason"),
                "usage": obj.get("usage"),
                "num_chunks": 1,
            }
    return raw, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--dump", default=DEFAULT_DUMP)
    ap.add_argument("--variant", default="default", help="Tag written into output records")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--seq", type=int, help="Replay only the request with this seq")
    sel.add_argument("--full", action="store_true", help="Replay all requests in dump order")
    sel.add_argument("--list", action="store_true", help="Print one-line summary per request and exit")
    ap.add_argument("--stream", dest="stream", action="store_true", default=None)
    ap.add_argument("--no-stream", dest="stream", action="store_false")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a body field, e.g. --override repetition_penalty=1.05",
    )
    ap.add_argument("--out", default=None, help="Append JSONL records with (request, response, summary, fingerprint)")
    args = ap.parse_args()

    reqs = load_requests(args.dump)
    if args.list:
        for r in reqs:
            b = r.get("body") or {}
            msgs = b.get("messages") or []
            tools = b.get("tools") or []
            stream = b.get("stream")
            max_tok = b.get("max_tokens")
            last_user = next(
                (m for m in reversed(msgs) if m.get("role") == "user"),
                None,
            )
            preview = ""
            if last_user:
                c = last_user.get("content")
                preview = (c[:80] if isinstance(c, str) else repr(c)[:80])
            print(
                f"seq={r['seq']:>3} msgs={len(msgs):>2} tools={len(tools):>2} "
                f"stream={stream} max_tok={max_tok} last_user={preview!r}"
            )
        return 0

    if args.seq is not None:
        selected = [r for r in reqs if r["seq"] == args.seq]
        if not selected:
            print(f"error: no request with seq={args.seq} in {args.dump}", file=sys.stderr)
            return 2
    else:
        selected = reqs

    outf = open(args.out, "a") if args.out else None
    any_degenerate = False
    for r in selected:
        body = dict(r["body"])
        if args.stream is not None:
            body["stream"] = args.stream
        body = apply_overrides(body, args.override)
        endpoint = r["endpoint"]
        raw, summary = post_request(args.host, endpoint, body)
        fp = degeneration_fingerprint(summary)
        tc_names = [tc.get("name") or "" for tc in (summary.get("tool_calls") or [])]
        tag = "DEGENERATE" if fp["degenerate"] else "OK"
        print(
            f"[{args.variant}] seq={r['seq']:>3} {tag:<10} "
            f"ctoks={fp['completion_tokens']:>4} "
            f"ws_run={fp['max_whitespace_run_chars']:>3} "
            f"non_ws_ratio={fp['non_whitespace_ratio']:<5} "
            f"finish={fp['finish_reason']} "
            f"tool_calls={tc_names}"
        )
        if fp["degenerate"]:
            any_degenerate = True
        if outf:
            rec = {
                "variant": args.variant,
                "seq": r["seq"],
                "endpoint": endpoint,
                "request_body": body,
                "response_summary": summary,
                "fingerprint": fp,
                "response_raw": raw if len(raw) < 32_768 else raw[:32_768] + "...[truncated]",
            }
            outf.write(json.dumps(rec) + "\n")
            outf.flush()
    if outf:
        outf.close()
    return 3 if any_degenerate else 0


if __name__ == "__main__":
    sys.exit(main())
