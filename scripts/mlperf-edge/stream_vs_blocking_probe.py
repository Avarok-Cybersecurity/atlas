#!/usr/bin/env python3
"""Is the ~200 ms chat-vs-completions gap in the STREAMING path?

Phase timing already excluded request preparation: the whole chat pre-dispatch
chain (msg_entry 21us, thinking 0us, template render+tokenize 1.3-13ms,
loop_detect 17us, session_hash 2us, sampling+grammar 8us) totals 2-13 ms, not
200. And it is not extra tokens: on the long cell the chat render is only +11
tokens over the raw completions prompt yet the gap is still 240 ms.

That leaves everything AFTER pre-dispatch. This probe re-runs the same
comparison with stream=False, measuring total request latency instead of TTFT.
If the gap collapses, it lives in the SSE/streaming encoder; if it survives, it
is in the shared dispatch/generation path and the endpoint difference is
something else again.

Usage: stream_vs_blocking_probe.py <port> <model> [--reps 7]
"""
import argparse, json, statistics, time, urllib.request

CHUNK = ("def resolve(self, name):\n    for k in type(self).__mro__:\n"
         "        if name in vars(k): return vars(k)[name]\n    raise KeyError(name)\n\n")


def post_blocking(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return (time.time() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port"); ap.add_argument("model"); ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    M = a.model
    SHORT = CHUNK * 3
    for label, text in (("SHORT", SHORT), ("LONG", CHUNK * 340)):
        comp = {"model": M, "prompt": text, "max_tokens": 8, "temperature": 0,
                "seed": 42, "stream": False}
        chat = {"model": M, "messages": [{"role": "user", "content": text}],
                "max_tokens": 8, "temperature": 0, "seed": 42, "stream": False}
        post_blocking(a.port, "/v1/completions", comp)
        post_blocking(a.port, "/v1/chat/completions", chat)
        c = statistics.median([post_blocking(a.port, "/v1/completions", comp) for _ in range(a.reps)])
        h = statistics.median([post_blocking(a.port, "/v1/chat/completions", chat) for _ in range(a.reps)])
        print(f"{label:6s} BLOCKING  completions={c:7.1f}  chat={h:7.1f}  gap={h-c:+7.1f} ms", flush=True)
    print("\ngap collapses vs the streaming run (+208 / +240) -> the cost is the SSE/streaming path")
    print("gap survives                                        -> it is in shared dispatch/generation")


if __name__ == "__main__":
    main()
