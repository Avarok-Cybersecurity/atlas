# Certification: `spark bench certify`

A certification campaign is every required gate the current commit still
owes, run to completion, with records committed for each. `spark bench
certify` is that campaign in the binary — the plan comes from the same
single source of truth as `--pull-request-gate-check`, and its last word is
the same gate table.

```
spark bench certify                     # everything still open, on this box
spark bench certify --dry-run           # the plan and the preflight, nothing run
spark bench certify --gates bfcl-subset # one group (its four shards)
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
yields to one that does not, so losing a node costs a quarter of a group.

**Speed-class gates spread only across boxes that are one box.** Before
anything starts, every pair of admitted nodes is checked by
`hardware::equivalence` (same GPU and driver line, clock ceiling within 1 %,
memory within 5 %, no thermal throttle, chassis within 10 °C — the fields
that told two "identical" GB10s apart by 0.66 tok/s). If every pair agrees,
Speed units go anywhere; otherwise they are **bundled** on the node with the
most headroom and the plan prints `WARNING speed-class gates BUNDLED on …`
with the concrete mismatch. CI re-checks the same rule from the records'
own captures (`docs/provable-benchmark-work.md` §5c).

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
   are not `Pass`; a benchmark group expands to its shards; every unit carries
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
   running, or a heartbeat under 30 minutes old) is refused by name; a dead one
   is archived beside itself and reclaimed.
4. **Run.** Each unit is a child `spark benchmark run <id> --pull-request-gate
   --hardware <class> [--yes]` — the operator's own command line — with its
   stderr streamed and logged under `.certify/<anchor>/<id>.log`. The evidence
   is the record the child leaves in `.benchmarks/<id>/`: it must name the
   anchor, its frame must have completed, and its verdict must be `PASS` (a
   shard's verdict is `Info` by design and counts as a completed member only
   when it carries its shard identity and tallies). Exit codes and printed
   lines are not evidence.
5. **Guard.** Before every unit and every 60 s during one, the guarded branch
   (`REMOTE/BRANCH`, default HEAD's upstream) is fetched and diffed against
   the anchor over `PERF_PATHS`. A docs-only push is harmless; a perf-path
   push aborts the campaign, names the files, and kills the running child —
   twenty minutes lost instead of nine hours. A guard that cannot answer
   aborts too: "could not check" is never "safe".
6. **Policy.** A verdict `FAIL` stops the campaign unless `--keep-going`; a
   retryable harness failure (the child died without writing a record) is
   retried once; a timeout (`expected × --timeout-factor` plus a build
   allowance) is not; Ctrl-C cancels.
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
