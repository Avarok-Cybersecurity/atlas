---
name: oracle_certification_state_check
description: "O.R.A.C.L.E::certification_state_check — BLOCKING pre-flight, in-flight and post-flight oracle for a benchmark certification campaign (~4.5-5 GPU-hours). Invoke BEFORE starting a certification campaign, running the gates, `spark benchmark run … --pull-request-gate`, or `queue-perf-pr.sh`; between and DURING gates; and AFTER the campaign before committing `.benchmarks/` records, commenting `/stamp` or `/seal`, or believing `stamp status` / `seal status` / `PR Benchmark Certifications`. Also invoke when a stamp or seal check looks red, when asking whether the box or GPU is free, whether the tree is frozen, whether a signer is committed, or whether records agree. Fourteen litmus tests — clean PERF_PATHS tree, frozen sha, binary built from it, ATLAS_HOME and signer, stray shard dirs, subjects and hermetic pins, BFCL scorer imports, box free with a self-filtered pgrep, campaign-guard, other PRs mid-certification, top of stack, stamp/seal job sequencing, one git_sha across added records, one Speed-class signer — each tied to the code that enforces it. Hard block with advisory overrides recorded in the gitignored lockfile `.oracle_should_begin_cert`, read on start to detect a concurrent campaign and removed on release. Born from the 2026-08-28 nine-hour campaign invalidated twenty minutes before its end, and from two same-day false reds where a /stamp or /seal status was evaluated before the bot processed the command."
argument-hint: "<pre | begin | during | post | release | status> [--pr <N>] [--branch <name>] [--session <id>] [--override T<n>[,T<n>] --reason \"…\"] [--abandon]"
allowed-tools: Bash, Read, Grep, Glob, Agent
---

# /oracle_certification_state_check — may this campaign start, continue, or be banked?

A certification campaign is ~4.5-5 GPU-hours and the box can do nothing else while it runs.
This skill puts a blocking oracle in front of it, beside it, and after it. The oracle is an
`opus` subagent (`.claude/agents/oracle_cert.md`). **Give it verbatim inputs, never a
summary**, and treat its one-line verdict as the decision.

Two incidents own this file. On 2026-08-28 a nine-hour, ten-gate campaign was
content-invalidated by a 23-line push twenty minutes into its final gate, and the loss was
discovered at the end. And twice on one day in 2026-09 a `/stamp` or `/seal` status was read
**before** the bot had processed the command; the stale red looked like a failure and was
reported as one. Everything below either prevents the first or dates the second.

## Reuse, not invention

| question | answered by | not by |
|---|---|---|
| did a perf path move? | `scripts/campaign-guard.sh <anchor> <branch>` (0/1/2) | a sha compare |
| can the box write, sign, find recipes? | `./target/release/spark doctor` | `ls -ld ~/.atlas` |
| what got recorded? | `spark benchmark --pull-request-gate-check --pr <N>` — read the words | the gate's `rc` |
| what counts as a perf path? | `sed -n '/pub const PERF_PATHS/,/];/p' crates/atlas-plugin/src/gate/coverage.rs` | a list copied here |
| is another PR mid-certification? | the bot's `<!-- atlas-certification-state: -->` marker, `isInMergeQueue`, added records | a guess from titles |
| were the stamp/seal jobs stale? | `gh api …/actions/runs/<id>/jobs` vs the `Stamp`/`Seal` check-run times | the colour of the check |

The one thing this skill adds is the lockfile, because none of the above remembers that a
campaign is in progress across sessions.

## Phases

```
pre ──▶ begin ──▶ during (× many) ──▶ post ──▶ release
 │        │            │                │
 │        │            │                └─ CERT-SEAL: commit records, /seal
 │        │            └─ CERT-ABORT: kill the gate, release --abandon
 │        └─ lockfile: evaluated → running_certification
 └─ lockfile: (none) → evaluating → evaluated
```

| verb | when | tests | writes |
|---|---|---|---|
| `pre` | before the first gate | T0-T12 (+T13/T14 if records already added) | creates lock `evaluating` → `evaluated` |
| `begin` | the moment before the driver launches | T0, T1, T2, T8, T9 | `evaluated` → `running_certification`, records `driver_pid` |
| `during` | between gates, and on a 5-min timer inside long ones | T0, T1, T2, T8, T9 | heartbeat, `current_gate`, `guard_last_rc` |
| `post` | after the last gate, BEFORE `git add .benchmarks`, BEFORE `/seal` | T0, T1, T2, T9, T12, T13, T14 + gate-check text | the `post` verdict |
| `release` | after `post` returned `CERT-SEAL` and records are pushed; or `--abandon --reason` | T0 | renames the lock to `.released.<ts>` |
| `status` | any time | none | nothing — prints the lock and its liveness |

`pre` is the default when no verb is given.

## Step 0 — read the lockfile before anything else

Every verb starts here. The file is `<repo-root>/.oracle_should_begin_cert` (gitignored).

```bash
root=$(git rev-parse --show-toplevel); lock="$root/.oracle_should_begin_cert"
[ -f "$lock" ] && jq . "$lock"
```

Then, in this order:

1. **No file, verb `pre`** → create it (below).
2. **No file, any other verb** → refuse: "no campaign is locked here; run `pre` first".
   Never create a lockfile from `begin`/`during`/`post`.
3. **File exists, LIVE, and not yours** → refuse, printing the owner block verbatim. **No
   override bypasses a live lock**; wait, or kill the named driver.
4. **File exists, yours, status matches the verb** → proceed.
5. **File exists and is STALE** → **do not delete it.** Rename to
   `.oracle_should_begin_cert.stale.<utc-ts>`, print the reason, and (for `pre` only) create a
   fresh lock whose `superseded` field records the old path and reason. For any other verb,
   refuse — a stale lock under `begin`/`during`/`post` means the campaign it described is gone.

### How stale is stale — liveness beats age

A lock whose campaign is demonstrably running is never stale, however old.

| status | stale when |
|---|---|
| `evaluating` | `updated_at` older than **20 min** (an evaluation takes 2-5) |
| `evaluated` | `expires_at` passed (**90 min**), OR `anchor_sha` != `git rev-parse HEAD`, OR `campaign-guard.sh` exits non-zero — a verdict describes one sha on one branch |
| `running_certification` | **ALL** of: `kill -0 <driver_pid>` fails; `pgrep -x spark` empty; `nvidia-smi --query-compute-apps` empty; heartbeat older than **30 min**. Any one liveness signal positive → LIVE. If LIVE and started >12 h ago, still LIVE — print "runaway?" with the pid and let a human decide. |

## `pre` — create, evaluate, record

**Create atomically.** `ln(2)` fails `EEXIST` for exactly one of two racing sessions:

```bash
sid=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid)
now=$(date -u +%Y-%m-%dT%H:%M:%SZ); tmp="$lock.$$.tmp"
jq -n --arg sid "$sid" --arg now "$now" --arg host "$(hostname)" --arg user "$(id -un)" \
      --arg cwd "$root" --arg br "$BRANCH" --arg sha "$(git rev-parse HEAD)" \
      --arg home "${ATLAS_HOME:-$HOME/.atlas}" '{
  schema:"oracle_should_begin_cert/v1", status:"evaluating",
  created_at:$now, updated_at:$now, expires_at:null,
  owner:{session_id:$sid, hostname:$host, user:$user, cwd:$cwd},
  campaign:{pr:null, branch:$br, anchor_sha:$sha, atlas_home:$home, driver_pid:null,
            driver_cmdline:null, started_at:null, current_gate:null, gates_done:[],
            heartbeat_at:null, guard_last_rc:null},
  oracle:null, overrides:[], history:[], superseded:null }' > "$tmp"
if ln "$tmp" "$lock" 2>/dev/null; then rm -f "$tmp"; echo "locked as $sid"
else rm -f "$tmp"; echo "LOST THE RACE"; jq . "$lock"; exit 1; fi
```

Keep `$sid` — every later verb passes `--session $sid`.

**Spawn the oracle** (`Agent`, `subagent_type: oracle_cert`) with, verbatim: the phase; the
lockfile contents; `$sid`; PR, branch, anchor sha, `$ATLAS_HOME`; your own `$$` and the
launcher line you will use; the exact override claim if any.

**Record**: re-read the lock, confirm `owner.session_id` is still `$sid` (if not, another
session superseded you — stop and print both), then write via temp + `mv -f`: `status=evaluated`,
`expires_at` = now + 90 min, `oracle` = the JSON, append `history`.

On `CERT-NO-GO` the lock **stays** as `evaluated` with the negative verdict, so a second
session sees that a campaign was attempted and why.

## `begin` — the read-on-start race check

Lock must be yours, `evaluated`, unexpired, verdict GO. Spawn the oracle for phase `begin`
(seconds). **This is the window between verdict and launch** in which a co-tenant can appear
or the branch can move — the second half of "read on start". On `CERT-GO`, launch the driver,
capture its PID, and write `status=running_certification` with `driver_pid`, `driver_cmdline`,
`started_at`, `heartbeat_at`, `expires_at=null`.

The driver runs this itself between Claude ticks, because a 1.6 h BFCL leg must not wait for
an agent turn:

```bash
scripts/campaign-guard.sh "$ANCHOR" "$BRANCH"; rc=$?
jq --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg g "$g" --argjson rc $rc \
   '.campaign.heartbeat_at=$now|.updated_at=$now|.campaign.current_gate=$g|.campaign.guard_last_rc=$rc' \
   "$lock" > "$lock.$$.tmp" && mv -f "$lock.$$.tmp" "$lock"
[ $rc -eq 0 ] || { pkill -x spark; echo "ABORT: guard rc=$rc"; exit 1; }
```

`pkill -x`, never `pkill -f` — the latter has killed its own shell here.

## `during` — continue or abort

Lock yours, `running_certification`, LIVE. The oracle re-runs the guard, the clean-tree check,
the anchor check (someone checking out another sha **in this worktree** is the same loss as a
push), and the box check. On `CERT-ABORT`: **kill the running gate rather than let it finish**
— a completed record for a superseded tree is the mistake repeated — then
`release --abandon --reason "<verdict line>"`.

## `post` — may the records be banked?

Run `spark benchmark --pull-request-gate-check --pr <N>` and hand the oracle the full text.
Its rule: refuse on `NONE`, `FAIL`, or `still need` — **`rc` is not the evidence**. The oracle
adds T13 (one `git_sha` across added records — the gate check does not ask this) and T14 (one
Speed-class signer). On `CERT-SEAL`: commit and push the records, wait for the `CI` run on the
new head to reach the point T12 allows, **then** comment `/seal`, then `release`.

## Overrides — advisory, recorded, never silent

```
/oracle_certification_state_check pre --pr 951 --override T10 --reason "owner serialising #946 by hand"
```

- The oracle still **runs** the overridden test and records its true `observed`.
- Both verdicts are printed and stored: `verdict` and `verdict_without_override`.
- Written twice: inside `oracle.override`, and appended to top-level `overrides[]`, never trimmed.
- `by` is `git config user.email` plus hostname; an empty `--reason` is refused.
- **Never overridable**: T0, T1, T3, T9, T13, the Speed-signer half of T14, and T12 in `post`.
  Each is a case where CI rejects the outcome anyway, so the override would only buy wasted hours.
- Quote any override verbatim in the wave report and the PR evidence comment. **An override
  nobody can see afterwards is not advisory, it is a bypass.**

## The JSON

Returned to the caller and stored unchanged under `oracle` in the lockfile. ISO-8601 UTC `Z`.

```json
{
  "schema": "oracle_certification_state_check/v1",
  "phase": "pre|begin|during|post",
  "verdict": "GO|NO-GO|CONTINUE|ABORT|SEAL|HOLD",
  "verdict_line": "CERT-GO (override: T10 — owner serialising #946 by hand)",
  "verdict_without_override": "NO-GO",
  "evaluated_at": "2026-09-11T14:02:11Z",
  "evaluation_ms": 41830,
  "subject": {
    "repo_root": "…", "pr": 951, "branch": "…", "head_sha": "…",
    "origin_branch_sha": "…", "origin_main_sha": "…", "merge_base": "…",
    "atlas_home": "…", "signer_fingerprint": "…", "signer_committed": true,
    "hostname": "dgx2", "caller_pid": 412233, "caller_launcher": "…"
  },
  "tests": [{
    "id": "T1", "name": "clean-perf-tree",
    "group": "lock|sha|tree|agreement|timing|environment",
    "result": "pass|fail|unknown|skipped|overridden",
    "observed": "pass|fail|unknown|skipped",
    "overridable": false, "overridden": false,
    "command": "git status --porcelain --untracked-files=all -- …",
    "evidence": "", "mechanism": "gate/mod.rs::dirty_perf_paths → check.rs fails a dirty-tree record",
    "remedy": "commit or stash, rebuild, re-run pre",
    "checked_at": "2026-09-11T14:01:40Z"
  }],
  "blocking": ["T10"],
  "override": { "tests": ["T10"], "reason": "…", "by": "…@dgx2", "at": "…Z", "invocation": "…" },
  "notes": ["TTFT baselines present for both gates; single runs suffice"],
  "next": "run: /oracle_certification_state_check begin --session 6f1c… --pr 951"
}
```

`result` is `overridden` iff `overridden`, and then `observed` holds the real outcome.
`blocking` is every test whose `result` is `fail` or `unknown`. `override` is `null` when none
was claimed.

## What the lockfile can and cannot promise

It is a file, not a mutex. Exactly:

- **Creation is atomic** on a local filesystem (`ln` → `EEXIST`), so two `pre` invocations in
  one checkout cannot both own a lock.
- **Every rewrite re-checks ownership**, and `begin` re-runs the cheap tests, so the window
  between verdict and launch is covered on both ends. What remains is the interval between
  `begin` writing `running_certification` and the driver's first `spark serve` appearing on the
  GPU (model load, minutes) — another checkout's `pre` in that window sees a free GPU.
- **The lock is per checkout, not per box.** A second worktree has its own root and its own
  lockfile. The box-level guard is T8 (GPU, `pgrep -x spark`, memory), which is real but polls.
- **It does not stop a human** running `spark benchmark run` by hand, nor a push from another
  machine — that is `campaign-guard.sh`'s job, and why `during` is on a timer.
- **GitHub state lags.** T10 reads the marker and queue flag as of now.
- **Session identity is a uuid you carry**, not a pid: each Bash call is a new shell. The
  driver's pid is the only long-lived one, and it is the liveness signal.

What the skill does about the residual TOCTOU: both sides re-read (`begin`, and the oracle's
T0), evidence is kept rather than deleted (`.stale.*`, `.released.*`), "could not tell" is
treated as "not safe", and the expensive guard runs on a timer inside the driver where no agent
turn is needed.

## Where this sits

- `oracle` rules on stack ORDER before registration; this rules on STATE before the GPU is
  spent. Different questions, both blocking.
- `/measurement-discipline` governs the numbers a record carries; this governs whether the
  record may exist.
