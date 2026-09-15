# Certification: `spark bench certify`

A certification campaign is every required gate the current commit still
owes, run to completion, with records committed for each. `spark bench
certify` is that campaign in the binary — the plan comes from the same
single source of truth as `--pull-request-gate-check`, and its last word is
the same gate table.

```
spark bench certify                     # everything still open, on this box
spark bench certify --dry-run           # the plan and the preflight, nothing run
spark bench certify --gates bfcl-subset # one group (the shards it still owes)
spark bench certify --shards 8          # cut each group's draw eight ways
spark bench certify --pr 1027 --yes     # a real campaign, agentic gate confirmed
```

`bench` is the short spelling of `benchmark`; both work everywhere.

```
spark bench certify --with-nodes 10.10.10.2,dgx3.local   # this box + two nodes, in parallel
spark bench certify --with-nodes 10.10.10.2 --remote-only # from a laptop: nodes only
```

## `--with-nodes`

The same campaign, on several machines at once. Every address is asked
through `atlasctl bench nodes` (`ip[:port]`, `[v6]:port`, `host.local`,
`dns.name`; port omitted → atlasctl's peer port), and a node is **admitted**
only when it can sign records this repository will accept: bench enabled, the
box class being certified, a signer committed in `.github/record-signers/`,
not busy, nothing queued, memory and disk above the floors, a GPU it can
name. Every refusal is printed with its reason; the campaign runs on what was
admitted. This box is a node too, unless `--remote-only`.

**Planning.** Whenever a node is free it takes the longest remaining unit it
may run (longest-first list scheduling: within 4/3 of optimal, and the same
rule at plan time and at run time, so `--dry-run`'s makespan is what
happens). Among equals, a shard whose group already has a shard on that node
yields to one that does not, so losing a node costs one shard of a group,
not the whole of it.

**One server per recipe, not per unit.** A unit on this box runs as
`spark benchmark run … --serve-reuse`: instead of loading the checkpoint in
its own process it takes the server the previous unit left running — if,
and only if, that server is the one it would have started itself. The
server answers `GET /serve-config` with two digests, of its binary and of
the arguments it was started with; the unit renders its own recipe with its
own overrides (a hermetic `kat-equality-gate` and an open `bfcl-subset` are
different renderings) and compares. A match is reused; anything else is
stopped and replaced; a unit never takes a server this mode did not start
(`<ATLAS_HOME>/serve-lease.json` names the one it may). The campaign stops
the last one when it ends, and a lease whose campaign died is stopped by
the next campaign before its preflight. `--no-serve-reuse` restores a fresh
server per unit; `spark benchmark serve-release` stops a leased server by
hand. Every record's command line carries `--serve-reuse` when it applied,
so a number measured on a warm server says so.

**Shards.** Each benchmark group's draw is cut into `--shards N` slices, run
as `--param shard=i/N` of the group's own benchmark. The default is two per
box that will run (so the scheduler has slices to balance around
`kat-equality-gate`, the longest single unit), and one box alone runs the
whole draw as `0/1` — a shard costs a server start and a warm-up, and there
is nothing to balance against. A partition already begun at the anchor is
finished at its own count whatever `--shards` says, because the verdict never
assembles a partition across counts or commits.

**Speed-class gates spread only across boxes that are one box.** Before
anything starts, every pair of admitted nodes is checked by
`hardware::equivalence` (same GPU and driver line, clock ceiling within 1 %,
memory within 5 %, no thermal throttle, chassis within 15 °C — the fields
that told two "identical" GB10s apart by 0.66 tok/s). If every pair agrees,
Speed units go anywhere; otherwise they are **bundled** on the node with the
most headroom and the plan prints `WARNING speed-class gates BUNDLED on …`
with the concrete mismatch. CI re-checks the same rule from the records'
own captures (`docs/provable-benchmark-work.md` §5c).

**Cool-down.** Before a node takes another unit its hottest chassis zone
and the driver's thermal-throttle flag are read. At **80 °C** or above, or
with the throttle asserted, the node is **parked**: it takes nothing until
it is back at or below **70 °C** with the throttle clear (re-read every
60 s, at most 30 min, then it resumes with a warning), and every transition
is printed and logged as a `thermal` event. The rest of the fleet keeps
working — the scheduler is work-conserving, so pending units go wherever a
node is free; a parked box that hosts the bundled Speed class only delays
that class. A node that cannot report a temperature is never parked (said
once); the records' own captures still decide equivalence. The line is
absolute, not relative to rest: a GB10 rises 26–33 °C over rest under any
gate (healthy loaded boxes read 55–76 °C on 2026-09-15) and the box behind
the 0.66 tok/s incident read 89 °C with a driver-reported slowdown.
`--dangerous-ignore-thermals` turns every park into a warning and lets the
box keep taking units — the operator's hardware to risk; the records are
still judged by the equivalence policy at the end, so the flag ignores the
security action, never the evidence.

**Each remote unit** is submitted with an idempotent key
(`certify-<run>-<node>-<gate>`), followed over a re-attachable stream (a
lost link resumes from the last event; ten losses is a harness failure),
fetched, and placed only after this side has checked the record is for that
unit at the anchor on this class, completed, clean, and signed by a committed
key. A record that fails any of those is removed again and never retried on
that node. A node that fails twice in a row is retired for the campaign.
Ctrl-C or a drift on the guarded branch cancels every node's job.

**What atlasctl must have.** Each node runs an `atlasctl agent` with a
`bench.yaml` (atlas-recipes `docs/BENCH.md`) and has granted this machine
`bench` (`atlasctl peer grant-bench <fingerprint>` there); this machine
needs `atlasctl` on `PATH` or `--atlasctl PATH`, paired with each node.
`atlasctl` is run with this process's environment, so a submitter identity
kept outside the default directory is selected with
`ATLASCTL_CONFIG_DIR=/path spark bench certify …`.

## What it does, in order

1. **Plan.** `gate::check_gates` at the anchor (HEAD) says which required gates
   are not `Pass`; a benchmark group expands to the shards it still owes —
   `gate::shards_owed`, the verdict's own answer, so a shard the gate already
   accepts at the anchor is not re-measured; every unit carries
   the descriptor's `expected_secs`, refined by the newest completed run in
   `~/.atlas/runs` when there is one. The local order is the long shard sets
   first (a failure there must not wait five hours to be seen), then the
   Speed class shortest-first, then the rest.
2. **Preflight.** Refuses to spend a GPU minute unless: HEAD is the anchor, no
   `PERF_PATHS` file is uncommitted, the signing identity in `ATLAS_HOME` is
   committed in `.github/record-signers/`, `ATLAS_HOME` is writable, no other
   `spark` is running, host memory is at least as free as a self-start
   requires, there is a branch to guard (or `--no-guard`), and any gate that
   needs confirmation has `--yes`. Every refusal names its remedy.
3. **Lock.** `.oracle_should_begin_cert` at the repo root, the v1 schema the
   O.R.A.C.L.E skill and the old shell driver wrote. A live lock (its driver
   running, or a heartbeat under 30 minutes old) is refused by name; a dead
   one is archived beside itself and reclaimed — at once when its status says
   the campaign is over (`campaign_done`, `aborted`), however fresh its last
   heartbeat.
4. **Run.** Each unit is a child `spark benchmark run <id> --pull-request-gate
   --hardware <class> [--yes] [--param shard=i/n]` — the operator's own
   command line — with its stderr streamed and logged under
   `.certify/<anchor>/<id>[-s<i>of<n>].log`. The evidence is the record the
   child leaves in `.benchmarks/<id>/`: it must name the anchor and, for a
   shard, the very slice the unit stands for; its frame must have completed,
   and its verdict must be `PASS` (a shard's verdict is `Info` by design and
   counts as a completed shard only when it carries its shard identity and
   tallies). Exit codes and printed lines are not evidence.
5. **Guard.** Before every unit and every 60 s during one, the guarded branch
   (`REMOTE/BRANCH`, default HEAD's upstream) is fetched and diffed against
   the anchor over `PERF_PATHS`. A docs-only push is harmless; a perf-path
   push aborts the campaign, names the files, and kills the running child —
   twenty minutes lost instead of nine hours. A guard that cannot answer is
   retried — up to five checks in a row (~five minutes of a mute network),
   each one logged — and then aborts: "could not check" is never "safe", but
   one DNS blip is not a verdict either.
6. **Policy.** A verdict `FAIL` stops the campaign unless `--keep-going`; a
   retryable harness failure (the child died without writing a record) is
   retried once; a timeout is not; Ctrl-C cancels. A unit's deadline is a
   ten-minute serve allowance (the child starts a server and loads a
   checkpoint before its first sample, none of which is in the measured
   estimate) plus `expected × --timeout-factor`, plus a build allowance on a
   node that has not built the anchor.
7. **Last word.** The gate table for the anchor, then the record-agreement
   rule over every record a commit would add (one commit; one signer across
   the Speed class). `CERTIFIED` requires both.

## Exit codes

| code | meaning |
|---|---|
| 0 | certified: every required gate passes at the anchor and the added records agree |
| 1 | usage, preflight or harness error — nothing to bank |
| 2 | the campaign completed and at least one verdict was `FAIL`, or the final check still refuses a record |
| 3 | aborted: a perf path moved on the guarded branch, or Ctrl-C |

## `--json`

One object per line on stdout, `event` ∈ `plan`, `preflight`, `fleet` (with
`--with-nodes`: nodes, rejections, `speed_mode`, per-node queues, makespan),
`guard`, `start`, `line`, `done` (each with `node` under `--with-nodes`),
`summary`, `final`, each with an `at` timestamp. The human report is
suppressed.

## After a campaign

Commit the records (`git add .benchmarks`), then `/stamp` and `/seal` the PR.
The campaign already ran the agreement rule CI will run, so the commit will
not be refused for a reason the campaign could have seen.

## Superseded

`scripts/campaign-guard.sh` is the shell form of the guard and is kept for CI
and ad-hoc use; `campaign_pr.sh` / `post_and_bank.sh` are replaced.
