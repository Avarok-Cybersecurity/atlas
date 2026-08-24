# Benchmark records and the merge queue: the serialized-landing protocol

## The problem, precisely

The PR benchmark gate requires committed records proving the ten mandatory
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
