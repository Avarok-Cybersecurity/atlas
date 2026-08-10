#!/usr/bin/env python3
"""Post-stress coherence check.

Aliased KV blocks corrupt output rather than crashing, so a clean refcount
log is necessary but not sufficient. Ask questions with checkable answers,
twice each (cold, then warm through the prefix cache), and verify both.
"""
import os, json, os, sys, time, urllib.request

URL = os.environ.get("ATLAS_URL", "http://localhost:8888/v1/chat/completions")
MODEL = os.environ.get("COHERENCE_MODEL", "laguna-s-2.1")

# Explicit system prompt. Sending none leaves whatever the chat template bakes in
# as the default, which is not what we want to be measuring here — this probe is
# checking that KV/prefix-cache paths return coherent text, so the instruction
# framing should be ours, minimal, and identical across models and runs.
SYSTEM = os.environ.get(
    "COHERENCE_SYSTEM",
    "You are a careful, precise assistant. Answer the question directly and "
    "correctly. Work step by step when a question needs it, and state the final "
    "answer explicitly. Do not refuse or hedge on simple factual or arithmetic "
    "questions.",
)
CASES = [
    ("What is 17 * 23? Reply with the number, then explain your working.", "391"),
    ("Name the capital city of Japan, then describe it in a few sentences.", "Tokyo"),
    ("Spell the word 'refrigerator' backwards, then explain how you did it.", "rotaregirfer"),
]


def ask(prompt):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "max_tokens": 300, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d["choices"][0]["message"]["content"]


if __name__ == "__main__":
    fails = 0
    for pas in ("cold", "warm"):
        for prompt, expect in CASES:
            try:
                out = ask(prompt)
                # Judge the STATED answer, not any substring. `expect in out` passed
                # a reply that opened with "271" for 17*23 and only reached 391
                # inside its working — a wrong headline scored as correct, which
                # inflates every number this probe reports. Accept only if the
                # expected value appears in the first or last non-empty line
                # (direct answer, or explicit final answer), and report the
                # "working only" case separately instead of silently passing it.
                lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
                head = lines[0].lower() if lines else ""
                tail = lines[-1].lower() if lines else ""
                e = expect.lower()
                ok = e in head or e in tail
                buried = (not ok) and e in out.lower()
                # Garbled output is the aliasing signature even when the
                # expected token happens to appear.
                printable = sum(c.isprintable() or c.isspace() for c in out)
                clean = len(out) > 0 and printable / len(out) > 0.98
                status = "OK " if (ok and clean) else ("WORKING-ONLY" if buried else "FAIL")
                if not (ok and clean):
                    fails += 1
                print(f"  [{pas}] {status} expect={expect!r} -> {out[:90]!r}", flush=True)
            except Exception as e:
                fails += 1
                print(f"  [{pas}] FAIL {expect!r}: {e}", flush=True)
            time.sleep(1)
    print(f"\n  coherence: {6-fails}/6 passed")
    sys.exit(1 if fails else 0)
