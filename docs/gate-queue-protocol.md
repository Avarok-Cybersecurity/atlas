# Benchmark records and the merge queue: the serialized-landing protocol

## The problem, precisely

The PR Benchmark Certifications check requires committed records proving the ten mandatory
benchmarks ran against the tree under test. Records are sha-anchored and are
invalidated by any performance-path change they did not cover — including
changes that arrive from *underneath*, when the merge queue composes the PR
with other PRs or with a `main` that moved after the records were measured.

This means a record-bearing performance PR can be **fully green at its head
and still bounce in the queue, repeatedly, with zero code defects**. Observed
2026-08-23: #732 bounced behind #731, then behind post-#733 `main`; #731 then
bounced behind post-#733 `main`. Every head was green the whole time.

This is not a gate bug. Two independently-measured campaigns do not compose:
the combined tree's interactions are unmeasured, and "measure-then-declare"
is the repo's core benchmark doctrine. The gate refusing to stitch records
together is the doctrine working. What *was* missing is (a) a name for the
failure — ten bare `NONE`s read as a broken PR, not a queue-composition
condition — and (b) a documented landing protocol. CI now emits a
`Queue-composition failure` annotation on `merge_group` gate failures, and the
protocol is below.

## The protocol: freeze → campaign → queue alone

1. **Freeze**: update the branch to *current* `main` (merge, don't rebase —
   rebasing orphans the record shas' ancestry) and push. Any later `main`
   movement restarts the protocol.
2. **Campaign**: run all ten gates against exactly the frozen sha, on a box
   with an exclusive GPU. Commit the records, push, wait for green.
3. **Queue alone**: enter the merge queue with **no other record-bearing PR
   ahead of you**, and do not let one enter until you have landed.

`scripts/queue-perf-pr.sh <branch>` performs the freeze and prints the
campaign commands.

## For queue administrators

Two record-bearing performance PRs can never successfully share a merge-queue
group; the second is guaranteed to bounce after burning a full CI run.
Consider restricting the queue's max group size to 1, or socially serializing
perf-PR landings. Non-perf PRs (docs, site, workflows, `scripts/`) compose
freely — none of their paths invalidate records.

## Known costs, accepted

Serialization means N perf PRs cost N campaigns even when each is individually
certified. The alternative — letting records compose — would declare unmeasured
interactions measured, which is precisely the class of silent regression the
gate exists to prevent (see the SPLIT=4, tree-scan, and P3 incidents: every
one was invisible to isolated numeric checks and caught only by task gates).

## Which gates want `mtp_gate=force`, and which must not have it

`atlas-recipes#16` pinned `mtp_gate: force` on the recipes backing the gates,
because in `auto` the MTP gate is a bandit arbiter that switches MTP↔serial at
runtime on wall-clock tok/s, and speculation is not output-neutral at
temperature 0. A campaign on 2026-08-28 confirmed the effect end to end:
`agentic-webserver` scored `followed_directions` **9/10 under `auto`** and
**10/10 under `force`**, one binary, one box, minutes apart. The 9/10 iteration
was not a broken build — the agent burned its turns on a `tower` 0.4-vs-0.5
dependency fight and stopped mid-repair.

That is not a reason to pin everything, and the same campaign showed why. The
rule is about the **shape of the bar**, not the engine:

| bar | example | mode |
|---|---|---|
| absolute / exact-match | `agentic-webserver` `followed_directions min = 10.0`, "takes no noise" | **pin `force`** — nondeterminism against a zero-headroom bar is a coin flip, and re-running until it passes is retry-until-green |
| empirical, measured | `bfcl-subset` `overall_accuracy min = 83.82` — "measured 84.22, less the documented ±0.4 MTP-nondeterminism noise floor" | **run in the mode the bar was measured in** |
| wall-clock / throughput | `decode-floor`, `concurrency-sweep` | **leave `auto`** — the arbiter optimises the metric under test; it is part of the product being benchmarked |

The middle row is the one that bites. `bfcl-subset`'s bar already *prices in*
the nondeterminism, quantified under `auto`, with 0.40 of headroom. Switching
that gate to `force` trades a bounded, documented band for an unquantified
systematic offset — `force` removes the exact serial episodes `auto` mixes in,
and MTP rollback/SSM-conv restore is inexact. A run judged that way is
uninterpretable against its own bar.

**Ordering.** To pin an empirically-barred gate, re-measure the bar under
`force` first, then pin. Never pin against a bar measured the other way — that
is `measure-then-declare` read backwards.

**Scope goes stale.** `#16` named the recipes backing the gates *on
2026-08-09*. The required `bfcl-subset` subject flipped from Qwen3.6-27B to
Qwen3.8-27B on 2026-08-15, so its pin followed the old subject and the gate it
was written for ran unpinned. When a `BENCH.toml` `default = true` moves to a
new recipe, the pin does not follow it. Check the gate→recipe map, not the
recipe list.
