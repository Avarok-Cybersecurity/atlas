#!/usr/bin/env python3
"""Force an SSM tier FAULT-IN — the one path the agentic harnesses never hit.

WHY THIS EXISTS. Across four separate agentic runs the spill tier recorded
hundreds of spills and 0-1 fault-ins, so the read side (and the reaping fix that
hangs off its miss arm) has never actually executed. That is not a tuning
problem, it is structural: `try_fault_in_ssm_snapshot` bails immediately when

    if prefix_match.ssm_snapshot.is_some() { return None; }   // resident wins
    let key = prefix_match.ssm_snapshot_tier_key?;            // must be SPILLED

so a fault-in needs a prefix whose snapshot has been evicted from the resident
pool but is still in the tier. Agentic turns arrive back-to-back while the
snapshot is still resident, so the resident path always wins and the tier is
never consulted.

THE SHAPE THAT DOES TRIGGER IT — three phases:

  1. WARM   : N deep conversations. Each must be deeper than
              ATLAS_SSM_SPILL_MIN_TOKENS (default 1024) or eviction DROPS it
              instead of spilling ("SSM spill SKIPPED (cost gate)").
  2. CHURN  : M distinct deep conversations, M > --ssm-cache-slots, to push the
              warm snapshots out of the resident pool. That eviction is what
              writes them to the tier.
  3. RETURN : re-send each warm prefix VERBATIM plus a short new suffix. The
              radix match now finds an entry with no resident snapshot but a
              live tier key -> fault-in.

Run with ATLAS_SSM_TIER_TIMING=1 and count in the server log:
    "SSM tier fault-in: restored"      <- the read side firing (the goal)
    "SSM spill:"                       <- phase 2 doing its job
    "spill SKIPPED (cost gate)"        <- prefixes too shallow; raise PREFIX_TOKENS
    "SSM tier reap:"                   <- only if the disk cap dropped a blob first

Usage: ssm_faultin.py [warm] [churn] [--] ; env: PORT, MODEL, PREFIX_TOKENS
"""
import json, os, sys, urllib.request

PORT = os.environ.get("PORT", "8888")
MODEL = os.environ.get("MODEL", "qwen3.6-27b")
URL = f"http://localhost:{PORT}/v1/chat/completions"
# Words, not tokens; ~1.3 tok/word. 4000 words ~= 5.2K tokens, comfortably past
# the 1024-token spill gate at every intermediate checkpoint that matters.
PREFIX_TOKENS = int(os.environ.get("PREFIX_TOKENS", "4000"))
WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CHURN = int(sys.argv[2]) if len(sys.argv) > 2 else 40


def prefix(tag: int) -> str:
    """A long, UNIQUE-per-tag body. Unique so each gets its own radix entry;
    reproduced verbatim on the return leg so the prefix hash matches."""
    return f"Document {tag}. " + " ".join(f"d{tag}w{k}" for k in range(PREFIX_TOKENS))


def ask(body_text: str, question: str, max_tokens: int = 32) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{body_text}\n\n{question}"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        return f"ok/{d['usage']['prompt_tokens']}"
    except Exception as e:  # noqa: BLE001 - a failed probe request is data, not fatal
        return f"ERR {type(e).__name__}"


print(f"model={MODEL} warm={WARM} churn={CHURN} prefix~{PREFIX_TOKENS} words")

print("\n[1/3] WARM — deep prefixes, each gets a snapshot")
for i in range(WARM):
    print(f"  warm {i}: {ask(prefix(i), 'Summarize in one sentence.')}")

print(f"\n[2/3] CHURN — {CHURN} distinct deep prefixes to evict the warm set")
for i in range(CHURN):
    r = ask(prefix(1000 + i), "Reply with the single word OK.", max_tokens=8)
    if i % 10 == 0 or i == CHURN - 1:
        print(f"  churn {i}: {r}")

print("\n[3/3] RETURN — same prefixes again; these should FAULT IN")
for i in range(WARM):
    print(f"  return {i}: {ask(prefix(i), 'Now answer: what document number is this?')}")

print(
    "\nNow count in the server log:\n"
    "  docker logs <container> | grep -c 'tier fault-in: restored'   # the goal\n"
    "  docker logs <container> | grep -c 'SSM spill:'\n"
    "  docker logs <container> | grep -c 'spill SKIPPED'             # raise PREFIX_TOKENS\n"
    "  docker logs <container> | grep -c 'SSM tier reap:'"
)
