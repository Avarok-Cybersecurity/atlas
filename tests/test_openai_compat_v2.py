"""
E2E smoke test for Atlas OpenAI API compatibility PR 4 (remaining gaps).

Covers items shipped after alpha-2.44:
  1. previous_response_id (stateful /v1/responses resume)
  2. Streaming /v1/responses (typed SSE events)
  3. store: true + GET /v1/chat/completions/{id} round-trip
  4. Refusal classifier populating message.refusal
  5. Real rate-limit enforcement (429 with retry-after)
  6. URL annotation extractor improvements
  7. 501 stubs on unsupported endpoints

Assumes an Atlas server is running on localhost:8888 (set ATLAS_URL to
override). The rate-limit test requires the server was started with
ATLAS_RATE_LIMIT_RPM=3 so a small burst exhausts the bucket; if not set,
the rate-limit assertion is skipped with a note.
"""
import json
import os
import sys
import time
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "requests>=2.31", "openai>=1.50"]
    )
    import requests

try:
    from openai import OpenAI
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "openai>=1.50"])
    from openai import OpenAI


BASE = os.environ.get("ATLAS_URL", "http://localhost:8888/v1")
MODEL = os.environ.get("ATLAS_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
c = OpenAI(base_url=BASE, api_key="sk-dummy")
HOST = urlparse(BASE).netloc

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"
PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
SKIP = f"{YELLOW}SKIP{RESET}"

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def section(label: str) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")


# ── 1. previous_response_id round-trip ─────────────────────────────────────
section("[1] Stateful /v1/responses (previous_response_id)")
try:
    r1 = requests.post(
        f"{BASE}/responses",
        json={
            "model": MODEL,
            "input": "Remember the number 42 for me.",
            "max_output_tokens": 40,
        },
        timeout=120,
    )
    r1.raise_for_status()
    b1 = r1.json()
    resp_id = b1["id"]
    assert resp_id.startswith("resp_"), f"expected resp_ prefix, got {resp_id}"
    record("first turn stored", PASS, f"id={resp_id}")

    r2 = requests.post(
        f"{BASE}/responses",
        json={
            "model": MODEL,
            "input": "What number did I ask you to remember?",
            "previous_response_id": resp_id,
            "max_output_tokens": 40,
        },
        timeout=120,
    )
    r2.raise_for_status()
    b2 = r2.json()
    # Best-effort content check: look for "42" in any text output.
    texts = []
    for item in b2.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    joined = " ".join(texts).lower()
    if "42" in joined:
        record("second turn recalled prior context", PASS)
    else:
        record("second turn recalled prior context", FAIL, f"text={joined[:120]!r}")

    # Missing previous_response_id → 400 with code=response_not_found.
    r3 = requests.post(
        f"{BASE}/responses",
        json={
            "model": MODEL,
            "input": "hello",
            "previous_response_id": "resp_does_not_exist_xyz",
            "max_output_tokens": 10,
        },
        timeout=30,
    )
    if r3.status_code == 400 and r3.json().get("error", {}).get("code") == "response_not_found":
        record("unknown previous_response_id → 400 response_not_found", PASS)
    else:
        record(
            "unknown previous_response_id → 400 response_not_found",
            FAIL,
            f"status={r3.status_code} body={r3.text[:160]}",
        )
except Exception as e:
    record("stateful responses", FAIL, f"{type(e).__name__}: {e}")


# ── 2. Streaming /v1/responses ─────────────────────────────────────────────
section("[2] Streaming /v1/responses (typed SSE events)")
try:
    with requests.post(
        f"{BASE}/responses",
        json={
            "model": MODEL,
            "input": "Say hello in three words.",
            "stream": True,
            "max_output_tokens": 30,
        },
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        seen = {
            "response.created": 0,
            "response.output_text.delta": 0,
            "response.completed": 0,
        }
        current_event = None
        final_text = ""
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw.startswith("event: "):
                current_event = raw[len("event: ") :].strip()
            elif raw.startswith("data: "):
                data = raw[len("data: ") :]
                if current_event in seen:
                    seen[current_event] += 1
                if current_event == "response.output_text.delta":
                    try:
                        obj = json.loads(data)
                        final_text += obj.get("delta", "")
                    except Exception:
                        pass
        ok = seen["response.created"] >= 1 and seen["response.completed"] >= 1
        if ok and seen["response.output_text.delta"] >= 1:
            record(
                "emitted created + delta + completed",
                PASS,
                f"events={seen}, text={final_text[:60]!r}",
            )
        else:
            record("emitted created + delta + completed", FAIL, f"events={seen}")
except Exception as e:
    record("streaming responses", FAIL, f"{type(e).__name__}: {e}")


# ── 3. store: true round-trip ─────────────────────────────────────────────
section("[3] store:true + GET /v1/chat/completions/{id}")
try:
    r = c.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Give me a 3-word greeting."}],
        max_completion_tokens=20,
        extra_body={"store": True},
    )
    cid = r.id
    assert cid.startswith("chatcmpl-"), f"unexpected id {cid}"
    record("created with store:true", PASS, f"id={cid}")

    g = requests.get(f"{BASE}/chat/completions/{cid}", timeout=30)
    if g.status_code == 200:
        body = g.json()
        if body.get("id") == cid:
            record("GET /v1/chat/completions/{id} returned stored body", PASS)
        else:
            record("GET body id matches", FAIL, f"got id={body.get('id')}")
    else:
        record("GET /v1/chat/completions/{id}", FAIL, f"status={g.status_code}")

    # Unknown id → 404.
    g2 = requests.get(f"{BASE}/chat/completions/chatcmpl-nope", timeout=10)
    if g2.status_code == 404:
        record("unknown id → 404", PASS)
    else:
        record("unknown id → 404", FAIL, f"status={g2.status_code}")
except Exception as e:
    record("completion storage", FAIL, f"{type(e).__name__}: {e}")


# ── 4. Refusal classifier ──────────────────────────────────────────────────
section("[4] Refusal classifier (message.refusal)")
try:
    # Force the pattern by asking the model to start with "I cannot".
    r = c.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Respond with exactly: I cannot help with that request.",
            }
        ],
        max_completion_tokens=30,
    )
    msg = r.choices[0].message
    if msg.refusal is not None and (msg.content in (None, "")):
        record("refusal populated + content nulled", PASS, f"refusal={msg.refusal!r}")
    else:
        record(
            "refusal populated + content nulled",
            FAIL,
            f"content={msg.content!r} refusal={msg.refusal!r}",
        )
except Exception as e:
    record("refusal", FAIL, f"{type(e).__name__}: {e}")


# ── 5. Rate-limit 429 ─────────────────────────────────────────────────────
section("[5] Rate-limit enforcement (429 + retry-after)")
rpm = os.environ.get("ATLAS_RATE_LIMIT_RPM_OBSERVED", "")
if not rpm:
    record(
        "rate-limit 429",
        SKIP,
        "server must start with ATLAS_RATE_LIMIT_RPM=3 (small) to trigger",
    )
else:
    try:
        statuses = []
        retry_after = None
        for _ in range(int(rpm) + 2):
            rq = requests.post(
                f"{BASE}/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                },
                timeout=20,
            )
            statuses.append(rq.status_code)
            if rq.status_code == 429:
                retry_after = rq.headers.get("retry-after")
                break
        if 429 in statuses and retry_after is not None:
            record(
                "burst triggered 429 with retry-after",
                PASS,
                f"statuses={statuses} retry-after={retry_after}",
            )
        else:
            record("burst triggered 429", FAIL, f"statuses={statuses}")
    except Exception as e:
        record("rate-limit", FAIL, f"{type(e).__name__}: {e}")


# ── 6. URL annotations ────────────────────────────────────────────────────
section("[6] URL annotation extractor quality")
# This test runs offline — we check the regression cases via a tiny payload
# echoed through the model. Practically, the extractor runs on assistant
# content; we can't force a specific URL in the response. So we instead
# assert that a prompt that induces a known URL yields at least one
# annotation with the expected shape.
try:
    r = c.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: See https://example.com/api for details.",
            }
        ],
        max_completion_tokens=30,
    )
    msg = r.choices[0].message
    content = msg.content or ""
    ann = getattr(msg, "annotations", None)
    if ann is not None and any("url_citation" in str(a) for a in ann):
        record("bare URL extracted as annotation", PASS, f"n={len(ann)}")
    elif "https://example.com" in content:
        record(
            "bare URL extracted as annotation",
            FAIL,
            f"URL in content but no annotation emitted: content={content[:120]!r}",
        )
    else:
        # Model didn't comply — can't test. Mark as skip.
        record("bare URL extracted as annotation", SKIP, "model didn't echo URL")
except Exception as e:
    record("URL annotations", FAIL, f"{type(e).__name__}: {e}")


# ── 7. 501 stubs on unsupported endpoints ─────────────────────────────────
section("[7] 501 stubs (batches / files / audio / images / moderations)")
for path in [
    "/batches",
    "/files",
    "/audio/speech",
    "/images/generations",
    "/moderations",
]:
    try:
        rq = requests.post(f"{BASE}{path}", json={}, timeout=10)
        if rq.status_code == 501:
            body = rq.json().get("error", {})
            if body.get("type") == "server_error" and body.get("message"):
                record(f"POST {path} → 501 + OpenAI error", PASS)
            else:
                record(f"POST {path} → 501", FAIL, f"body={body}")
        else:
            record(f"POST {path} → 501", FAIL, f"status={rq.status_code}")
    except Exception as e:
        record(f"POST {path}", FAIL, f"{type(e).__name__}: {e}")


# ── Summary ────────────────────────────────────────────────────────────────
section("Summary")
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
skipped = sum(1 for _, s, _ in results if s == SKIP)
for name, status, detail in results:
    tag = "PASS" if status == PASS else ("FAIL" if status == FAIL else "SKIP")
    print(f"  {tag:4}  {name}" + (f" — {detail}" if detail else ""))
print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")
sys.exit(0 if failed == 0 else 1)
