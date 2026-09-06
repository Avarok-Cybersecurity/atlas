#!/usr/bin/env python3
"""Concurrent streaming decode against an Atlas /v1/chat/completions endpoint.

For each concurrency C: launch C requests AT ONCE (distinct salted prompts of ~prompt_tokens,
natural/code answer class), stream them, and record per stream
  TTFT, decode wall (first->last content chunk), server-attested completion_tokens,
  decode tok/s = (completion_tokens - 1) / decode wall        (measure_decode.py's definition;
  under MTP the inter-chunk gap median is meaningless - drafted tokens arrive in bursts - so
  tokens/wall is the only honest per-stream rate),
and per cell
  aggregate_decode_tok_s = sum(completion_tokens) / (max last-chunk - min first-chunk),
  aggregate_wall_tok_s   = sum(completion_tokens) / cell wall (includes every TTFT),
plus `free -g`-class host memory and nvidia-smi util/power samples every --sample-interval
seconds. A watchdog aborts the run (and TERMs --server-pid) when MemAvailable or the swap
delta cross the thresholds: the earlier C=4 attempt at util 0.85 swapped the box.

Prints a FINGERPRINT line, one line per cell, a markdown table, and writes --json-out.
Exit codes: 0 ok, 2 some streams errored, 3 aborted by the memory watchdog.
"""
import argparse, json, os, random, signal, statistics, subprocess, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8899)
ap.add_argument("--model", default="auto", help="'auto' = first id from /v1/models")
ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--prompt-tokens", type=int, default=2000)
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--effort", default="low")
ap.add_argument("--label", default="short")
ap.add_argument("--salt", type=int, default=int(time.time()) % 1_000_000)
ap.add_argument("--timeout-s", type=int, default=5400)
ap.add_argument("--sample-interval", type=float, default=5.0)
ap.add_argument("--abort-avail-gb", type=float, default=8.0)
ap.add_argument("--abort-swap-delta-gb", type=float, default=4.0)
ap.add_argument("--server-pid", type=int, default=0, help="process group to TERM on abort (0 = none)")
ap.add_argument("--json-out", default="")
args = ap.parse_args()

WORDS = ("kernel stream tensor cache block schedule verify draft router expert layer norm gate highway "
         "carry snapshot commit rewind window hash gather slot arena page token prefix budget latency "
         "bandwidth roofline launch cooperative barrier reduction").split()
TASK = ("Write a complete Rust implementation of an LRU cache with generic key and value types, using a "
        "HashMap and a doubly linked list of indices into a Vec arena. Include get, put, capacity handling, "
        "and unit tests. Explain each design decision briefly in comments.")


def make_prompt(n_tokens, salt):
    # ~1.037 tokens/word on this vocabulary (measured; the server's usage.prompt_tokens is what we report).
    # The salt leads, so two prompts diverge at their first user token: no prefix-cache hit between cells.
    rng = random.Random(salt)
    body = " ".join(rng.choice(WORDS) for _ in range(max(0, int((n_tokens - 80) / 1.037))))
    return f"Session salt {salt}. Skim this log, then do the task below.\n\n{body}\n\nTask: {TASK}"


def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0])
    return d["MemAvailable"] / 1048576, (d["SwapTotal"] - d["SwapFree"]) / 1048576


def gpu_sample():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().replace(" ", "")
        return out or "na,na"
    except Exception:
        return "na,na"


def resolve_model():
    if args.model != "auto":
        return args.model
    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


class Watchdog:
    """Samples host memory + GPU every interval; sets .aborted and TERMs the server on pressure."""

    def __init__(self):
        self.samples, self.aborted, self.stop = [], False, threading.Event()
        self.swap0 = meminfo()[1]
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            try:
                avail, swap = meminfo()
                self.samples.append((time.time(), avail, swap, gpu_sample()))
                if avail < args.abort_avail_gb or swap - self.swap0 > args.abort_swap_delta_gb:
                    self.aborted = True
                    print(f"ABORT memory pressure: MemAvailable={avail:.1f}GB swap_used={swap:.1f}GB "
                          f"(start {self.swap0:.1f}GB) -> TERM server pgid {args.server_pid}", flush=True)
                    if args.server_pid > 0:
                        try:
                            os.killpg(args.server_pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    return
            except Exception as e:  # never let the sampler kill the measurement
                print(f"watchdog sample error: {e}", file=sys.stderr, flush=True)
            self.stop.wait(args.sample_interval)

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1] or self.samples[-1:]
        return dict(min_avail_gb=round(min((x[1] for x in s), default=-1), 1),
                    max_swap_used_gb=round(max((x[2] for x in s), default=-1), 1),
                    swap_delta_gb=round(max((x[2] for x in s), default=self.swap0) - self.swap0, 1),
                    gpu_samples=[x[3] for x in s][-6:])


def stream_one(model, prompt, out):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": args.effort}}
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); times = []; usage = None; finish = None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout_s) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch:
                    delta = ch[0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                        times.append(time.perf_counter())
                    if ch[0].get("finish_reason"):
                        finish = ch[0]["finish_reason"]
    except Exception as e:
        out.append({"error": f"{type(e).__name__}: {e}"[:300], "chunks": len(times)}); return
    if len(times) < 5:
        out.append({"error": f"only {len(times)} content chunks", "finish": finish}); return
    ct = usage["completion_tokens"] if usage else len(times)
    dw = times[-1] - times[0]
    out.append({"ttft_s": times[0] - t0, "decode_wall_s": dw, "completion_tokens": ct, "chunks": len(times),
                "prompt_tokens": usage.get("prompt_tokens") if usage else None, "finish": finish,
                "tok_s": (ct - 1) / dw if dw > 0 else 0.0, "t_first": times[0], "t_last": times[-1]})


def run_cell(model, c, rep, wd):
    out = []
    salt_base = args.salt * 1000 + c * 100 + rep * 10
    prompts = [make_prompt(args.prompt_tokens, salt_base + i) for i in range(c)]
    t_start = time.time(); p0 = time.perf_counter()
    ths = [threading.Thread(target=stream_one, args=(model, p, out)) for p in prompts]
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - p0
    ok = [o for o in out if "error" not in o]; errs = [o for o in out if "error" in o]
    tot = sum(o["completion_tokens"] for o in ok)
    span = (max(o["t_last"] for o in ok) - min(o["t_first"] for o in ok)) if ok else 0.0
    cell = dict(label=args.label, C=c, rep=rep, streams_ok=len(ok), errors=[e["error"] for e in errs],
                prompt_tokens=[o["prompt_tokens"] for o in ok], completion_tokens=[o["completion_tokens"] for o in ok],
                finish=[o["finish"] for o in ok], per_stream_tok_s=[round(o["tok_s"], 2) for o in ok],
                median_per_stream_tok_s=round(statistics.median(o["tok_s"] for o in ok), 2) if ok else 0.0,
                aggregate_decode_tok_s=round(tot / span, 2) if span > 0 else 0.0,
                aggregate_wall_tok_s=round(tot / wall, 2) if wall > 0 else 0.0,
                ttft_s=[round(o["ttft_s"], 1) for o in ok], cell_wall_s=round(wall, 1), aborted=wd.aborted)
    cell.update(wd.window(t_start, time.time()))
    return cell


def table(cells):
    lines = ["| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |",
             "|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|"]
    for x in cells:
        lines.append(f"| {x['label']} | {x['C']} | {x['rep']} | {x['streams_ok']}/{x['C']} | {x['per_stream_tok_s']} | "
                     f"{x['median_per_stream_tok_s']} | {x['aggregate_decode_tok_s']} | {x['aggregate_wall_tok_s']} | "
                     f"{x['ttft_s']} | {x['prompt_tokens'][:1]} | {x['min_avail_gb']} | {x['swap_delta_gb']} |")
    return "\n".join(lines)


def main():
    model = resolve_model()
    print(f"FINGERPRINT port={args.port} model={model} label={args.label} prompt_tokens~{args.prompt_tokens} "
          f"max_tokens={args.max_tokens} temp=0 effort={args.effort} stream=1 prompt_class=salted-log+code-task "
          f"concurrency={args.concurrency} repeats={args.repeats} salt={args.salt} "
          f"abort_avail_gb={args.abort_avail_gb} abort_swap_delta_gb={args.abort_swap_delta_gb} "
          f"date={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    wd = Watchdog(); cells = []; rc = 0
    try:
        for c in args.concurrency:
            for rep in range(args.repeats):
                if wd.aborted:
                    break
                cell = run_cell(model, c, rep, wd)
                cells.append(cell)
                print(f"CELL {json.dumps(cell)}", flush=True)
                if cell["errors"]:
                    rc = max(rc, 2)
                    for e in cell["errors"]:
                        print(f"  error: {e}", flush=True)
    finally:
        wd.stop.set(); wd.thread.join(timeout=args.sample_interval + 2)
        print(); print(table(cells), flush=True)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"fingerprint": vars(args) | {"model": model}, "cells": cells,
                           "mem_samples": wd.samples, "aborted": wd.aborted}, f, indent=1, default=str)
    if wd.aborted:
        print("RUN ABORTED by the memory watchdog", flush=True); rc = 3
    sys.exit(rc)


if __name__ == "__main__":
    main()
