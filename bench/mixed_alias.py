#!/usr/bin/env python3
"""
Direct detector for mixed-step logits-row aliasing (decode lane <- prefill row).

THE BUG. In a mixed scheduler tick (>=2 prefills co-scheduled with >=1 active
decode) the model runs `decode_batch` first, writing lane i's logits to row i of
the shared logits arena, then runs the batched prefill, whose finishing streams
write row `stream_idx` of the SAME arena. The caller samples the decode rows
AFTER the prefill sub-pass, so decode lane i can sample the first-token
distribution of prefill stream i -- another request's tokens.

WHY THIS HARNESS AND NOT conc_bleed.py. The user-visible symptom (a stray
`<tool_call>` opener, or a reply drifting onto a foreign topic) shows up on only
~2/32 requests, because it needs a collision AND a visible consequence. That
rate is too low to A/B a fix against with any confidence. Here the consequence
is made deterministic instead:

  * ONE "counter" request generates a strictly predictable sequence (1, 2, 3...).
    Every token it emits is known in advance, so ANY substituted token is
    visible -- no judgement call, no model-quality confound.
  * While it decodes, N long-prompt requests are fired so they prefill
    CONCURRENTLY, which is what makes the scheduler take the mixed path at all.

A break in the counter's sequence is the injected foreign token. On a clean
engine the sequence is unbroken.

Usage: mixed_alias.py [rounds]
Env: PORT, MODEL, FILLERS (concurrent prefills), COUNT_TO, PREFIX_WORDS
"""
import json, os, re, sys, threading, time, urllib.request

PORT = os.environ.get("PORT", "8888")
MODEL = os.environ.get("MODEL", "laguna-s-2.1")
URL = f"http://localhost:{PORT}/v1/chat/completions"
FILLERS = int(os.environ.get("FILLERS", "7"))
COUNT_TO = int(os.environ.get("COUNT_TO", "220"))
PREFIX_WORDS = int(os.environ.get("PREFIX_WORDS", "6000"))
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def post(body, timeout=900):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def counter_request():
    """Strictly predictable decode stream. Greedy so the sequence is forced."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      f"Count from 1 to {COUNT_TO}. Output ONLY the numbers "
                      f"separated by single spaces, nothing else. Do not stop early."}],
        # Well above the 250 floor: the counter must still be decoding while the
        # filler prefills land, otherwise there is no mixed step to detect.
        "max_tokens": 1200,
        "temperature": 0.0,
        "logprobs": True, "top_logprobs": 5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return post(body)


def filler_request(i, rnd):
    """Long unique prompt -> multi-chunk prefill that overlaps the counter."""
    pad = " ".join(f"r{rnd}w{i}n{k}" for k in range(PREFIX_WORDS))
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      f"{pad}\n\nSummarize the above in one sentence."}],
        "max_tokens": 250,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        return post(body)
    except Exception as e:
        return {"__err__": f"{type(e).__name__}"}


def injection_report(text):
    """Describe the aliasing signature, if present.

    MEASURED shape of a real event (Laguna-S, 7 fillers, 4/4 rounds):

        "1 2 3 ... 23 24The user wants me to count from 1 to 220 ... \\n\\n1 2 3 ..."

    The counter is mid-sequence when a concurrent prefill stream finalizes into
    its logits row, so it samples THAT stream's first token -- `785` == "The",
    the single most common prefill first token in the INFO log, with NO leading
    space because it opens a fresh generation. Having emitted "The", the model's
    own context makes it write a preamble about its task and then start counting
    over. So the tell is not one odd number: it is prose appearing mid-sequence
    followed by a RESTART.

    A clean engine returns a purely numeric reply."""
    stripped = text.strip()
    if re.fullmatch(r"[\d ]+", stripped):
        return None
    m = re.search(r"(\d+)([A-Za-z][^\d]{0,80})", text)
    where = f"after {m.group(1)!r}: {m.group(2)[:60]!r}" if m else "prose in reply"
    restarted = bool(re.search(r"\d+\s*\n*\s*\b1 2 3 4 5\b", text))
    return f"{where}{' + SEQUENCE RESTART' if restarted else ''}"


def _seq_start(text):
    """Index in `text` where the real 1,2,3... run begins.

    The model often emits a preamble ("The user wants me to count from 1 to
    220...") even with thinking disabled. Naively scanning every number in the
    reply parses THAT as the sequence and reports phantom breaks -- measuring
    model verbosity instead of engine corruption. Anchor on the first literal
    "1 2 3" run and analyse only from there."""
    m = re.search(r"\b1\s+2\s+3\b", text)
    return m.start() if m else None


def check_sequence(text, upto):
    """Return (n_seen, breaks). A break is a deviation from 1,2,3,...

    Only SUBSTITUTIONS and reorderings count; a truncated-but-correct reply
    scores clean, since decode simply stopping early is not this bug."""
    start = _seq_start(text)
    if start is None:
        return 0, [("no 1 2 3 run found", text[:80])]
    nums = re.findall(r"\d+", text[start:])
    breaks = []
    expect = 1
    for tok in nums:
        if int(tok) != expect:
            breaks.append((expect, tok))
            # resync so one bad token doesn't cascade into hundreds of breaks
            expect = int(tok) + 1
        else:
            expect += 1
        if expect > upto:
            break
    return len(nums), breaks


def non_numeric_junk(text):
    """Foreign tokens inside the sequence -- the clearest injection signature.

    Scoped to the text AFTER the run starts, so the model's own preamble (which
    is legitimate output, not corruption) does not register."""
    start = _seq_start(text)
    if start is None:
        return ""
    stripped = re.sub(r"[\d\s,.]", "", text[start:])
    return stripped[:200]


print(f"model={MODEL} port={PORT} fillers={FILLERS} rounds={ROUNDS} count_to={COUNT_TO}")
print("counter decodes while N long prefills run -> mixed steps -> aliasing\n")

total_breaks = 0
total_junk = 0
for rnd in range(1, ROUNDS + 1):
    result = {}

    def run_counter():
        try:
            result["c"] = counter_request()
        except Exception as e:
            result["c"] = {"__err__": f"{type(e).__name__}"}

    t = threading.Thread(target=run_counter)
    t.start()
    # Let the counter get INTO decode before the prefills land: the mixed path
    # requires an already-active sequence. Too short and the counter is still
    # prefilling (prefill-only batch, different code path, no aliasing).
    time.sleep(3.0)

    fth = [threading.Thread(target=filler_request, args=(i, rnd)) for i in range(FILLERS)]
    for f in fth:
        f.start()
    for f in fth:
        f.join()
    t.join()

    c = result.get("c", {})
    if "__err__" in c:
        print(f"round {rnd}: counter ERROR {c['__err__']}")
        continue
    txt = c["choices"][0]["message"].get("content") or ""
    n, breaks = check_sequence(txt, COUNT_TO)
    inj = injection_report(txt)
    total_breaks += len(breaks)
    if inj:
        total_junk += 1
    status = "CLEAN" if inj is None else "CORRUPT"
    print(f"round {rnd}: {status}  numbers={n} breaks={len(breaks)}")
    if inj:
        print(f"    injection: {inj}")

print(f"\n=== TOTALS over {ROUNDS} rounds ===")
print(f"  rounds showing injection: {total_junk}/{ROUNDS}")
print(f"  sequence breaks: {total_breaks}")
print("  (clean engine => 0 injections; counter reply is purely numeric)")
