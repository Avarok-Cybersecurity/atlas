#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Capture one diagnostic chat stream without changing the frozen ladder."""

import argparse
import base64
import codecs
import hashlib
import http.client
import ipaddress
import json
import pathlib
import socket
import threading
import time
import urllib.parse


class StreamAudit:
    """Observe SSE framing, event times and explicit completion evidence."""

    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.data = []
        self.done = False
        self.errors = []
        self.content = ""
        self.reasoning = ""
        self.tools = {}
        self.usage = None
        self.reason = None
        self.content_events = 0
        self.times = {f"first_{name}_s": None for name in ("role", "reasoning", "content", "tool")}
        self.times.update(finish_s=None, usage_s=None, done_s=None)

    def mark(self, name, stamp):
        if self.times[name] is None:
            self.times[name] = stamp

    def feed(self, raw, stamp):
        try:
            self.buffer += self.decoder.decode(raw)
        except UnicodeDecodeError:
            self.errors.append("invalid UTF-8")
            return
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if self.data:
                    self.consume("\n".join(self.data), stamp)
                    self.data = []
            elif line.startswith(":"):
                continue
            else:
                field, _, value = line.partition(":")
                if field == "data":
                    self.data.append(value.removeprefix(" "))

    def consume(self, text, stamp):
        if self.done:
            self.errors.append("data after [DONE]")
            return
        if text == "[DONE]":
            self.done = True
            self.times["done_s"] = stamp
            return
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            self.errors.append("malformed SSE JSON")
            return
        if not isinstance(doc, dict) or "error" in doc:
            self.errors.append("invalid or error stream event")
            return
        choices = doc.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            self.errors.append("probe requires one choice")
            return
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                self.errors.append("invalid choice")
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                self.errors.append("invalid delta")
                continue
            if delta.get("role"):
                self.mark("first_role_s", stamp)
            for field, name in (("content", "content"), ("reasoning_content", "reasoning"),
                                ("reasoning", "reasoning")):
                value = delta.get(field)
                if value is not None and not isinstance(value, str):
                    self.errors.append(f"invalid {field}")
                elif value:
                    if self.reason is not None:
                        self.errors.append("generated data after finish reason")
                    self.mark(f"first_{name}_s", stamp)
                    setattr(self, name, getattr(self, name) + value)
                    if name == "content":
                        self.content_events += 1
            calls = delta.get("tool_calls") or []
            if not isinstance(calls, list):
                self.errors.append("invalid tool_calls")
            else:
                for call in calls:
                    self.tool(call, stamp)
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in ("stop", "length", "tool_calls"):
                    self.errors.append(f"unsupported finish reason: {reason!r}")
                if self.reason is not None:
                    self.errors.append("duplicate finish reason")
                self.reason = reason
                self.times["finish_s"] = stamp
        usage = doc.get("usage")
        if usage is not None:
            if not isinstance(usage, dict) or any(
                    type(usage.get(key)) is not int or usage[key] <= 0
                    for key in ("prompt_tokens", "completion_tokens")):
                self.errors.append("usage needs positive integer prompt/completion tokens")
            self.usage = usage
            self.times["usage_s"] = stamp

    def tool(self, call, stamp):
        if not isinstance(call, dict) or type(call.get("index")) is not int or call["index"] < 0:
            self.errors.append("invalid tool index")
            return
        if self.reason is not None:
            self.errors.append("tool data after finish reason")
        function = call.get("function") or {}
        if not isinstance(function, dict):
            self.errors.append("invalid tool function")
            return
        target = self.tools.setdefault(call["index"], {"id": "", "name": "", "arguments": ""})
        for key, value in (("id", call.get("id")), ("name", function.get("name")),
                           ("arguments", function.get("arguments"))):
            if value is not None and not isinstance(value, str):
                self.errors.append(f"invalid tool {key}")
            elif value:
                target[key] += value
        self.mark("first_tool_s", stamp)

    def finish(self):
        try:
            self.buffer += self.decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            self.errors.append("incomplete UTF-8")
        if self.buffer or self.data:
            self.errors.append("unfinished SSE event")
        if not self.done:
            self.errors.append("missing [DONE]")
        if self.reason is None:
            self.errors.append("missing finish reason")
        if self.usage is None:
            self.errors.append("missing usage")
        if not (self.content or self.reasoning or self.tools):
            self.errors.append("empty generation")
        if self.reason == "tool_calls" and not self.tools:
            self.errors.append("tool_calls finish without calls")
        if self.tools and self.reason != "tool_calls":
            self.errors.append("tool calls without tool_calls finish")
        for tool in self.tools.values():
            try:
                args = json.loads(tool["arguments"])
                if not isinstance(args, dict) or not tool["name"] or not tool["id"]:
                    raise ValueError("missing function identity or object arguments")
            except (ValueError, TypeError):
                self.errors.append("incomplete tool call")
        return {"passed": not self.errors, "errors": list(dict.fromkeys(self.errors)),
                **self.times, "content": self.content, "reasoning": self.reasoning,
                "tool_calls": self.tools, "content_events": self.content_events,
                "usage": self.usage, "finish_reason": self.reason,
                "scope": "single diagnostic stream; structural completion, not semantic correctness or campaign scoring"}


def capture(args):
    url = urllib.parse.urlsplit(args.url)
    if (url.scheme != "http" or url.username or url.password or url.query or url.fragment
            or url.path != "/v1/chat/completions" or not url.hostname
            or not ipaddress.ip_address(url.hostname).is_loopback):
        raise ValueError("use numeric loopback HTTP /v1/chat/completions on the owned server")
    if not (0 < args.timeout_s <= 300) or not (0 < args.max_bytes <= 16 * 1024 * 1024):
        raise ValueError("timeout must be (0,300] seconds and max-bytes (0,16777216]")
    raw_request = args.request_json.read_bytes()
    if len(raw_request) > args.max_bytes:
        raise ValueError("request exceeds byte limit")
    payload = json.loads(raw_request)
    if (not isinstance(payload, dict) or payload.get("stream") is not True
            or (payload.get("stream_options") or {}).get("include_usage") is not True
            or payload.get("n", 1) != 1):
        raise ValueError("request must set stream=true, include_usage=true and one choice")
    args.out.mkdir(mode=0o700, parents=False, exist_ok=False)
    (args.out / "request.json").write_bytes(raw_request)
    audit = StreamAudit()
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=args.timeout_s)
    start = time.monotonic()
    status = None
    count = 0
    expired = threading.Event()
    deadline = None
    with (args.out / "chunks.jsonl").open("x") as chunks:
        try:
            conn.connect()
            wire_socket = conn.sock

            def expire():
                expired.set()
                try:
                    wire_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            deadline = threading.Timer(max(0, args.timeout_s - (time.monotonic() - start)), expire)
            deadline.daemon = True
            deadline.start()
            conn.request("POST", url.path, body=raw_request, headers={
                "Content-Type": "application/json", "Accept": "text/event-stream"})
            response = conn.getresponse()
            status = response.status
            if status != 200:
                audit.errors.append(f"HTTP {status}")
            if response.getheader("Content-Type", "").split(";", 1)[0].strip() != "text/event-stream":
                audit.errors.append("response is not text/event-stream")
            while True:
                remaining = args.timeout_s - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("total probe deadline")
                if conn.sock is not None:
                    conn.sock.settimeout(remaining)
                chunk = response.read1(min(65536, args.max_bytes - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                stamp = time.monotonic() - start
                chunks.write(json.dumps({"elapsed_s": stamp, "data_base64": base64.b64encode(chunk).decode()}) + "\n")
                chunks.flush()
                if count > args.max_bytes:
                    audit.errors.append("response exceeded byte limit")
                    break
                audit.feed(chunk, stamp)
        except (OSError, http.client.HTTPException, socket.timeout) as exc:
            audit.errors.append(f"transport failure: {type(exc).__name__}")
        finally:
            if deadline is not None:
                deadline.cancel()
                deadline.join()
            conn.close()
    if expired.is_set() or time.monotonic() - start >= args.timeout_s:
        audit.errors.append("total probe deadline exceeded")
    report = audit.finish()
    report.update(url=args.url, http_status=status, received_bytes=count,
                  elapsed_s=time.monotonic() - start,
                  request_sha256=hashlib.sha256(raw_request).hexdigest())
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "errors": report["errors"], "out": str(args.out)}))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--request-json", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--max-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    try:
        return capture(args)
    except (OSError, ValueError, AttributeError) as exc:
        parser.exit(2, f"stream probe refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
