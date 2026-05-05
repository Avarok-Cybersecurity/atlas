#!/usr/bin/env python3
"""
test_session_isolation.py — Verify no SSM state mixing under concurrent load.

Sends concurrent requests with unique system prompts and factual questions,
then checks that each response matches its own prompt — not another session's.

Usage:
  python tests/test_session_isolation.py
  python tests/test_session_isolation.py --url http://host:port
"""

import argparse, json, sys, threading
from urllib.request import Request, urlopen


def send_request(url, session_id, system_prompt, user_prompt, results):
    payload = json.dumps({
        "model": "test",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "enable_thinking": False,
    }).encode()
    try:
        req = Request(f"{url}/v1/chat/completions", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            results[session_id] = text
    except Exception as e:
        results[session_id] = f"ERROR: {e}"


# 16 factual questions with unambiguous short answers
QUESTIONS = [
    ("What is the capital of France? One word only.", "paris"),
    ("What is 2+2? Just the number.", "4"),
    ("What color is the sky on a clear day? One word only.", "blue"),
    ("What planet are we on? One word only.", "earth"),
    ("How many legs does a spider have? Just the number.", "8"),
    ("What is the chemical symbol for water? Just the formula.", "h"),
    ("What language is spoken in Brazil? One word only.", "portuguese"),
    ("How many continents are there? Just the number.", "7"),
    ("What is the largest ocean? One word only.", "pacific"),
    ("What is the opposite of hot? One word only.", "cold"),
    ("How many days in a week? Just the number.", "7"),
    ("What gas do plants absorb? Chemical formula only.", "co"),
    ("What is the capital of Japan? One word only.", "tokyo"),
    ("How many sides does a triangle have? Just the number.", "3"),
    ("What is the boiling point of water in Celsius? Just the number.", "100"),
    ("What is the smallest prime number? Just the number.", "2"),
]


def run_test(url, label, session_ids, concurrent):
    n = len(session_ids)
    results = {}
    threads = []

    for i in session_ids:
        q, _ = QUESTIONS[i]
        sys_prompt = f"You are session {i}. Answer concisely."
        t = threading.Thread(target=send_request,
                             args=(url, i, sys_prompt, q, results))
        threads.append(t)

    if concurrent:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for t in threads:
            t.start()
            t.join()

    passed = 0
    failed = 0
    for i in session_ids:
        _, expected = QUESTIONS[i]
        text = str(results.get(i, "MISSING")).lower()
        ok = expected in text
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"    Session {i:2d}: {status} — expected \"{expected}\" in: {text[:60]}")

    return passed, failed


def main():
    ap = argparse.ArgumentParser(description="Test session isolation")
    ap.add_argument("--url", default="http://localhost:8888")
    args = ap.parse_args()

    total_pass = 0
    total_fail = 0

    # Test 1: Sequential — 16 sessions one at a time
    print("=== Test 1: Sequential (16 sessions) ===")
    p, f = run_test(args.url, "sequential", list(range(16)), concurrent=False)
    total_pass += p
    total_fail += f

    # Test 2: Concurrent — 16 sessions simultaneously
    print()
    print("=== Test 2: Concurrent (16 sessions) ===")
    p, f = run_test(args.url, "concurrent", list(range(16)), concurrent=True)
    total_pass += p
    total_fail += f

    # Test 3: Concurrent batches — 4 batches of 4
    print()
    print("=== Test 3: Concurrent batches (4x4) ===")
    for batch in range(4):
        ids = list(range(batch * 4, batch * 4 + 4))
        print(f"  Batch {batch + 1}: sessions {ids}")
        p, f = run_test(args.url, f"batch-{batch}", ids, concurrent=True)
        total_pass += p
        total_fail += f

    # Summary
    print()
    total = total_pass + total_fail
    print(f"=== Summary: {total_pass}/{total} passed, {total_fail} failed ===")
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
