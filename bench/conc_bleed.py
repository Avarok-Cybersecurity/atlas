#!/usr/bin/env python3
"""
Cross-request BLEED + corruption detector for a running Atlas server.

WHY THIS EXISTS
---------------
Single-request probes cannot see the failure mode that matters most in
production: one request's content surfacing in ANOTHER request's response.
That was observed live (an agentic client received text from an unrelated
concurrent request), and no existing harness could reproduce it on demand.

HOW IT WORKS
------------
Each concurrent worker owns a UNIQUE canary token and a UNIQUE topic, and its
prompt asks for its own canary back. Verdicts per response:

  OK       answer contains ITS OWN canary and number
  TYPO     canary MISSPELLED but number correct — a model spelling artefact on an
           invented token, explicitly NOT corruption (see `_near`)
  PARTIAL  own canary, wrong number
  WRONG    no canary / wrong value        -> model weakness OR corruption
  BLEED    contains ANOTHER worker's canary or topic
           -> cross-request contamination. This is the smoking gun.

A SOLO reference pass runs every prompt sequentially first. That separates
"the model is weak on this prompt" from "concurrency corrupts it": a prompt
that is OK alone and WRONG/BLEED under load is an engine bug, not the model.

USAGE
-----
  python3 bench/conc_bleed.py <model> [concurrency] [rounds]

  PORT=8888          server port
  THINK=1            enable thinking (default OFF — see below)
  THINK_BUDGET=2048  per-request thinking budget, Anthropic-style
                     {"thinking":{"budget_tokens":N}}; implies THINK=1
  MAXTOK=200         max_tokens per request
  TOOLS=1            attach a tool schema and scan tool-call arguments too
  EXACT=1            deterministic-answer mode; the SENSITIVE bleed detector
  STREAM=1           use the SSE streaming endpoint — a DIFFERENT code path from
                     blocking, and the one agentic clients actually use
  PREFIX_WORDS=6000  shared-prefix length; MUST exceed the Marconi checkpoint
                     interval (4096 tokens) or the SSM snapshot path is never hit

Run it BOTH ways. Thinking ON and OFF exercise different code paths, and
thinking materially changes the result: with thinking ON and a small
max_tokens, several models spend the entire budget reasoning and return EMPTY
content, which scores WRONG and would MASK real bleed. Measured on Holo 3.1:
0/8 solo with thinking on, 8/8 solo with it off, same build, same prompts.
That is why thinking defaults to OFF here — a harness that cannot establish a
clean solo baseline cannot detect anything.

THE TRIGGER IS A SHARED PREFIX
------------------------------
Back-to-back on ONE Laguna-XS container, same detector, only prompt shape differs:

    DISTINCT prompts (PREFIX_WORDS=0)  ->  32/32 CLEAN
    SHARED 6000-word prefix            ->  4 BLEED, e.g. VORPAL -> "VORGYR 19"
                                           (VORPAL's "VOR" + GYRE's "GYR")

This RECONCILES this harness with bench/agentic/conc_harness.py rather than either
being wrong. That harness passes 7/8 at C=8 with clean KV health — and its own
source says it uses DISTINCT prompts deliberately, "otherwise the prefix cache
would dedupe". It therefore cannot trigger this. It is not a weak test; it tests
a different shape.

The shared-prefix shape is the PRODUCTION one: every agentic client sends a long
common system prompt and differs only in the tail. So "the agentic harness passes"
is a statement about distinct-prompt traffic ONLY, and says nothing about
shared-prefix concurrency — which is what real clients do.

Sequential is clean at BOTH shapes, so it needs concurrency AND sharing together.

COUNT BLEED, NOT WRONG
----------------------
EXACT mode scores a correct-but-prose answer ("The project codename is MIMSY...
MIMSY 31") as WRONG. Use BLEED events for the corruption rate: 4/32 = 12.5%.

INTERPRETING RESULTS
--------------------
  solo OK == N and concurrent all OK      -> clean
  solo OK == N but concurrent WRONG/BLEED -> CONCURRENCY BUG (the point of this)
  solo already failing                    -> fix the probe or the model config
                                             FIRST; the run tells you nothing
"""
import json, os, sys, urllib.request, concurrent.futures, collections

MODEL = sys.argv[1] if len(sys.argv) > 1 else "puzzle"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

PORT = os.environ.get("PORT", "8888")
URL = f"http://localhost:{PORT}/v1/chat/completions"
MAXTOK = int(os.environ.get("MAXTOK", "200"))
# Words of SHARED prefix prepended to every worker. This is the whole point:
# agentic clients send a long COMMON system prompt and differ only in the tail,
# so every request collides in the radix at the shared depth. A harness with
# short, mutually-distinct prompts never populates the cache and cannot see
# prefix-cache contamination at all. ~2000 words is ~2600 tokens, ~160 blocks
# at block_size 16. PREFIX_WORDS=0 reverts to the (weaker) no-sharing shape.
#
# MUST EXCEED THE MARCONI CHECKPOINT INTERVAL. Atlas writes an intermediate SSM
# checkpoint every 256 blocks (4096 tokens at block_size 16). A shared prefix
# SHORTER than that produces "Prefix cache hit: N tokens but no SSM snapshot —
# recomputing all KV": the KV radix hits but NO snapshot exists at or below the
# matched depth, so the SSM snapshot path is never exercised. Measured: a 2000-word
# prefix (3376 tok / 211 blocks) hit KV only. 6000 words is ~7800 tokens / ~490
# blocks, which straddles the interval and produces a reusable snapshot.
PREFIX_WORDS = int(os.environ.get("PREFIX_WORDS", "6000"))
BUDGET = os.environ.get("THINK_BUDGET")
THINK = os.environ.get("THINK") == "1" or BUDGET is not None

# Unique canary + unrelated topic per worker, so another worker's canary OR
# topic word appearing in a response is unambiguous contamination.
WORKERS = [
    ("ZANTHOR", "quartz mining", "7"),
    ("BRILLIG", "harbour dredging", "12"),
    ("VORPAL", "orchard grafting", "19"),
    ("SLITHY", "kiln firing", "23"),
    ("MIMSY", "rope splicing", "31"),
    ("GYRE", "lamp trimming", "44"),
    ("TULGEY", "salt panning", "58"),
    ("JUBJUB", "clock regulating", "63"),
]
ALL_CANARIES = [w[0] for w in WORKERS]
ALL_TOPICS = [w[1].split()[0] for w in WORKERS]


# TOOLS=1 attaches a tool schema and asks the model to call it. Agentic clients
# always send tools, and the tool path has its own parsing/grammar/steering code
# that plain chat never touches — a bleed that only manifests there would be
# invisible without this.
TOOLS = os.environ.get("TOOLS") == "1"
# STREAM=1 uses the SSE streaming endpoint. This is NOT cosmetic: Atlas's
# streaming and blocking paths are separate code with documented divergences
# (a known case cut generation at 141 tokens streaming vs 248 blocking), and
# agentic clients like opencode ALWAYS stream. A bleed that only manifests on
# the streaming path is invisible to a blocking-only harness.
STREAM = os.environ.get("STREAM") == "1"
# EXACT=1 is the SENSITIVE detector. It asks for a deterministic one-token-ish
# answer ("<CANARY><NUM>", nothing else) and compares EXACTLY. The prose mode
# cannot see fragment-level bleed because every worker's sentence opens with the
# same words; exact mode makes ANY deviation visible.
#
# MEASURED on Holo 3.1 (streaming, 8 workers, identical prompts, temp 0):
#   sequential : 0/24 deviations
#   concurrent : deviations EVERY trial, e.g. ZANTHOR -> "ZANTHORGY" where "GY"
#                is the opening of GYRE's canary; VORPAL -> "VORZPAL"
#                (a "Z" from ZANTHOR injected MID-TOKEN)
# Sequential-clean + concurrent-dirty on identical inputs is cross-request
# contamination, not model variance.
#
# IT IS NOT STREAMING-SPECIFIC. Measured on Holo 3.1, 32 concurrent requests per
# path, EXACT mode, identical totals:
#     streaming  28 OK / 3 WRONG / 1 BLEED
#     blocking   28 OK / 3 WRONG / 1 BLEED   <- same ZANTHORGY bleed
# An earlier reading called this a streaming bug because the PROSE mode showed
# nothing on blocking — prose hides fragment splices behind the identical sentence
# opening every worker produces. Always run EXACT on BOTH paths before attributing.
#
# Some corruptions are not canary fragments at all but tokens from elsewhere in the
# vocabulary — "Viktorches", "Trenutley", "Vývojpal" (non-ASCII) — i.e. a token
# sampled from the WRONG distribution, which is what reading another sequence's
# logits row would produce.
EXACT = os.environ.get("EXACT") == "1"
TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "record_site_entry",
        "description": "Record the project codename and entry count for a site log.",
        "parameters": {
            "type": "object",
            "properties": {
                "codename": {"type": "string", "description": "The project codename"},
                "entries": {"type": "integer", "description": "Number of log entries"},
            },
            "required": ["codename", "entries"],
        },
    },
}]


def _ask_stream(body, req_headers):
    """Accumulate SSE deltas, including tool-call argument fragments."""
    body["stream"] = True
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers=req_headers
    )
    text = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                text += d.get("content") or ""
                for tc in (d.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    text += " " + str(fn.get("name", "")) + " " + str(fn.get("arguments", ""))
    return text


def ask(prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": THINK},
    }
    if TOOLS:
        body["tools"] = TOOL_SCHEMA
        body["tool_choice"] = "auto"
    if BUDGET:
        body["thinking"] = {"budget_tokens": int(BUDGET)}
    headers = {"Content-Type": "application/json"}
    try:
        if STREAM:
            return _ask_stream(body, headers)
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        text = msg.get("content") or ""
        # Tool-call arguments carry the answer when TOOLS=1 — scan them too, or
        # every tool-calling response scores WRONG and real bleed inside the
        # arguments would be missed entirely.
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            text += " " + str(fn.get("name", "")) + " " + str(fn.get("arguments", ""))
        return text
    except Exception as e:
        return f"__ERR__ {type(e).__name__}"


# Built ONCE and reused verbatim by every worker and every round, so round 2+
# are genuine warm cache hits at the shared depth — the condition under which
# cross-request bleed was actually observed in production.
_rng = __import__("random").Random(20260728)
_VOCAB = ("ledger harbour quarry lantern trellis cistern meadow gantry pallet "
          "furrow beacon cobble thicket parapet spindle mortar wicker bramble "
          "conduit rafter").split()
SHARED_PREFIX = " ".join(_rng.choice(_VOCAB) for _ in range(PREFIX_WORDS))


def make_prompt(canary, topic, num, _pad=None):
    # SHARED prefix first (identical for all workers), then the per-worker
    # unique section. Requests therefore match each other deep into the radix
    # and differ only in the tail that carries the canary.
    head = f"Reference log:\n{SHARED_PREFIX}\n\n" if PREFIX_WORDS else ""
    body = (f"{head}"
            f"Project codename is {canary}. The project concerns {topic}. "
            f"The site log records exactly {num} entries.\n\n")
    if EXACT:
        # Deterministic target so ANY deviation is signal. Critically the reply
        # STARTS with the canary, so a foreign fragment cannot hide behind the
        # identical prose opening every worker would otherwise produce.
        return body + ("Question: Reply with EXACTLY the codename followed by the "
                       "number, nothing else. Start your reply with the codename.")
    return body + ("Question: What is the project codename, and how many entries "
                   "does the site log record? Answer in one short sentence.")


def _near(a, b, tol=2):
    """True if some token in `b` is within `tol` edits of `a` (same length only).

    Models misspell INVENTED tokens: Holo returned "VORRAL" for "VORPAL" with the
    number correct. Scoring that WRONG produced a false concurrency signal that
    cost real debugging time — the substance was right every time. A near-miss on
    the canary with the correct number is a spelling artefact, NOT corruption.
    """
    for w in b.replace(",", " ").replace(".", " ").split():
        if len(w) == len(a) and sum(x != y for x, y in zip(w, a)) <= tol:
            return True
    return False


def classify(idx, out):
    mine, num = WORKERS[idx][0], WORKERS[idx][2]
    up = out.upper()
    if EXACT:
        clean = "".join(ch for ch in up if ch.isalnum())
        if clean == f"{mine}{num}":
            return "OK", ""
        # Any deviation is corruption. Name the foreign canary when a fragment of
        # one is present — fragments as short as 2 chars have been observed.
        for j, c in enumerate(ALL_CANARIES):
            if j == idx:
                continue
            # >=3 chars. A 2-char fragment is NOT evidence: models garble invented
            # tokens, and a common digram (e.g. "GY") landing inside a mangled
            # canary looks identical to a real splice. Measured: at a 2-char
            # threshold nearly every "bleed" was "GY", which is exactly what a
            # coincidence distribution looks like; at 3 chars only genuine
            # multi-token splices survive ("VORGYR", "ZANGYR").
            for k in range(len(c), 2, -1):
                if c[:k] in clean.replace(mine, "").replace(num, ""):
                    return "BLEED", f"fragment {c[:k]!r} of {c}"
        return "WRONG", f"expected {mine}{num}"
    # Foreign canary/topic first — that is the finding this harness exists for,
    # and it outranks everything else.
    others = [c for j, c in enumerate(ALL_CANARIES) if j != idx and c in up]
    otop = [t for j, t in enumerate(ALL_TOPICS) if j != idx and t.upper() in up]
    if others or otop:
        return "BLEED", f"saw {others + otop}"
    if mine in up and num in out:
        return "OK", ""
    if mine in up:
        return "PARTIAL", "canary ok, number wrong"
    if num in out and _near(mine, up):
        return "TYPO", "canary misspelled, number correct — not corruption"
    return "WRONG", ""


def run_round(rnd):
    # IDENTICAL prompts every round on purpose: round 1 populates the cache,
    # rounds 2+ are warm. Varying them per round (an earlier version did) means
    # zero cache hits and the harness proves nothing about the cache.
    prompts = [make_prompt(c, t, n) for (c, t, n) in WORKERS[:CONC]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
        outs = list(ex.map(ask, prompts))
    return [(classify(i, o), o) for i, o in enumerate(outs)]


mode = (f"thinking={'ON' if THINK else 'OFF'} tools={'ON' if TOOLS else 'OFF'} "
        f"api={'STREAM' if STREAM else 'blocking'} exact={'ON' if EXACT else 'OFF'}")
if BUDGET:
    mode += f" budget={BUDGET}"
print(f"model={MODEL} port={PORT} concurrency={CONC} rounds={ROUNDS} {mode} "
      f"max_tokens={MAXTOK} shared_prefix_words={PREFIX_WORDS}")

print("=== SOLO reference (sequential — establishes the baseline) ===")
solo_ok = 0
for i, (c, t, n) in enumerate(WORKERS[:CONC]):
    v, _ = classify(i, ask(make_prompt(c, t, n)))
    solo_ok += v == "OK"
    print(f"  {c:<8} {v}")
print(f"  solo OK: {solo_ok}/{CONC}")
if solo_ok < CONC:
    print("  !! solo baseline is not clean — fix the probe/model config first;")
    print("     concurrent results below cannot distinguish model weakness from bleed.")

tally = collections.Counter()
bleeds = []
# Keep the actual text of non-OK responses. A verdict alone cannot distinguish
# "model produced no answer" from "model produced ANOTHER request's answer
# rephrased" — the latter is bleed the canary match may miss.
wrongs = []
print("\n=== CONCURRENT passes ===")
for r in range(1, ROUNDS + 1):
    line = []
    for i, ((v, why), out) in enumerate(run_round(r)):
        tally[v] += 1
        line.append(f"{WORKERS[i][0][:4]}:{v[0]}")
        if v == "BLEED":
            bleeds.append((WORKERS[i][0], why, out[:150]))
        elif v not in ("OK", "TYPO"):
            wrongs.append((r, WORKERS[i][0], v, out[:220]))
    print(f"  round {r}: " + " ".join(line))

print(f"\n=== TOTALS over {ROUNDS * CONC} concurrent requests ===")
for k, v in tally.most_common():
    print(f"  {k:<8} {v}")
if wrongs:
    print(f"\n--- {len(wrongs)} non-OK responses (inspect for disguised bleed) ---")
    for r, name, v, txt in wrongs[:8]:
        print(f"  round {r} [{name}] {v}: {txt!r}")
if bleeds:
    print(f"\n*** {len(bleeds)} CROSS-REQUEST BLEED EVENTS ***")
    for name, why, txt in bleeds[:6]:
        print(f"  [{name}] {why}\n      {txt!r}")
    sys.exit(1)
print("\n  no cross-request bleed detected")
