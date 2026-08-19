# governance/ — the PR journey ledger

One file per pull request, `pr-<n>.jsonl`, each line one
`atlas_governance::Event`. `.benchmarks/` answers *"did this commit pass?"*;
this directory answers *"how did this pull request get here?"* — what the
advisory classifier thought, with what status, at what time.

## Only the harvester writes here

No human and no PR-branch job commits these files. The pipeline is:

1. `ci.yml` / `pr-categorize` (advisory, `contents: read` **on purpose** — it
   consumes model output derived from an attacker-authored PR title) appends a
   `Category` line locally and uploads it as the `governance-event-pr<n>`
   workflow artifact. It cannot push, even in principle.
2. `.github/workflows/governance-harvest.yml` — running **default-branch
   code** — resolves the PR number from the run's own `head_sha` (never from
   the artifact's name, which is only a search hint), re-validates every line
   through `ledger_harvest`, and commits the result here.

`ledger_harvest` rejects: events claiming a different PR than the run record
proves, anything that is not a `Category` event (Gate and Measurement events
are written where those things happen, beside the `.benchmarks/` record), and
malformed lines (wholesale — a truncated upload cannot contribute a prefix).
Categories that no longer resolve in the taxonomy are dropped with a warning.

## Grow-only, union-merged

Each file is a CRDT G-Set, not a log: lines are only ever appended, identity
is `(head_sha, run_id, attempt, kind)` with the timestamp deliberately
excluded, so a replayed run converges instead of accumulating. `.gitattributes`
already declares `governance/*.jsonl merge=union`, so two branches that each
carry harvested lines merge textually without conflict, and
`Journey::deduplicated` collapses any duplicate on read.

Consequence: never edit or delete lines. A wrong opinion is still a true
record of what the classifier said; correcting history is exactly what a
ledger must not do.

## The bounded window, and `archive.csv`

One file per PR, forever, is not a plan — it was 75 files before anything
capped it. The directory is now a bounded FIFO queue: the newest **100** PRs
keep their own `pr-<n>.jsonl`, and older ones are removed from the tree and
recorded in `archive.csv`:

```csv
pr,hash,merged_at
519,d3b8ec461c3d5257cb99479a95695ec8f512073b,2026-08-15T20:41:16Z
```

`hash` is the commit that last carried that file, so the record is not gone —
it is one command away, byte-identical:

```bash
git show d3b8ec461c3d5257cb99479a95695ec8f512073b:governance/pr-519.jsonl
```

That is the whole design: **the git tree is still the store, and the CSV is
the index into it.** Nothing is copied, so nothing can drift from the original.
`merged_at` is the PR's own merge date (empty for a PR closed without merging),
because the commit date is already recoverable from `hash` and would say
nothing new.

This does not weaken the grow-only property above. Eviction removes a whole
FILE from the working tree; it never removes a LINE from a file, so the G-Set
semantics within each ledger are untouched. `archive.csv` is itself append-only
and carries `merge=union` for the same reason the ledgers do.

**An open PR is never evicted.** `gate::required::intent_source()` resolves a
ledger by bare path with no fallback, so an evicted file does not raise — the
PR silently reports `NotRecorded` and drops to path-only gating, which is a
wrong answer with no error attached. The window yields to this rule: when too
many old PRs are open, the directory is allowed to run slightly over 100 rather
than break gating, and it self-corrects as they close.

The decision lives in `.github/scripts/governance-evict.sh` with a test table
in `governance-evict-test.sh`, for the same reason the harvest triage does —
a decision expressed in a YAML block is a decision nobody can run.

⚠️ Archive rows point into git history. A history rewrite (the repo had one in
August 2026, to purge `.buncache`) invalidates every `hash` recorded before it.

## The `intent:<path>` override label

A maintainer can preempt the model entirely by applying a label of the form
`intent:<taxonomy-path>` (e.g. `intent:performance/decode`) to the PR. The
categorize job validates the path against `.github/pr-taxonomy.json`, skips
the model call, and records the ledger line with `--status ok`. This works on
fork PRs (labels are maintainer-controlled, not author-controlled) and is the
rerun/escape hatch when the free-tier endpoint is down or wrong.

## Advisory — enforcement is deferred, deliberately

Nothing here is read by `spark benchmark --pull-request-gate-check`'s verdict.
The gate renders the intent-derived benches (`required_for = by_path ∪
by_intent`) as an **advisory** block only. Escalating intent into the blocking
verdict is designed but deferred until the abstain-rate evidence this ledger
accumulates shows the classifier answers often enough to be load-bearing —
the harvest workflow publishes that histogram in its step summary every run
and warns above 25% non-answers. An abstaining classifier that could block
merges would convert endpoint downtime into repo downtime; an advisory one
converts it into a recorded abstention, which is the correct failure mode.

## How a harvest lands

The harvester opens a pull request from the rolling branch `bot/governance-harvest`
and requests auto-merge, so ledger commits arrive through the merge queue with
every required check intact. `main` needs no bypass actors and no direct-push
token.

The author of that PR decides whether it can drive itself. A PR opened with
`GITHUB_TOKEN` never triggers workflow runs (GitHub blocks that to stop
recursion), so its required checks sit unreported and someone has to push the
branch by hand to wake CI. A PR opened by a **GitHub App** triggers them
normally. Configure the App once and the loop is unattended.

### Configuring the harvester App

1. Create the App under the organisation, at Settings, Developer settings,
   GitHub Apps, New GitHub App.
   - Uncheck **Active** under Webhook.
   - Repository permissions: **Contents** read and write, **Pull requests**
     read and write. Nothing else.
   - Where can this GitHub App be installed: **Only on this account**.
2. Note the **Client ID**, then **Generate a private key** and keep the
   downloaded `.pem`.
3. **Install App**, scoped to the `atlas` repository only.
4. Register it with the repository:
   ```bash
   gh variable set GOVERNANCE_APP_CLIENT_ID --repo Avarok-Cybersecurity/atlas --body '<client id>'
   gh secret set GOVERNANCE_APP_PRIVATE_KEY --repo Avarok-Cybersecurity/atlas < path/to/key.pem
   ```

`governance-harvest.yml` mints an installation token when
`GOVERNANCE_APP_CLIENT_ID` is set and falls back to `GITHUB_TOKEN` when it is not, warning on the PR that its
checks need a nudge.

## When the model cannot answer

`unknown` implies no benchmarks, so an abstain is safe — the path rules in
`gate::coverage` still apply. It is useless as evidence though, and two causes
produced it systematically.

**Dependabot never sees repository secrets.** GitHub keeps a separate store for
them, so `OPENROUTER_API_KEY_FREE` is empty on every Dependabot PR and the
descent abstains on all of them, permanently. **A spent free-tier daily quota**
does the same until the quota resets.

`ci.yml` now answers from the changed paths when they answer the question by
themselves: everything under `.github/` is `infrastructure/ci`, a pure
manifest change is `infrastructure/build-system`, and so on. It fires only when
EVERY changed path falls in one bucket, and it refuses to guess about engine
internals — a diff touching `crates/` or `kernels/` can mean anything, and
inventing an intent there would be fabrication.

The model's own verdict is recorded first and unedited, abstain included. The
path-derived answer is an additional line, because `required::report` unions
every `ok`/`partial` value. That way an intent is available without hiding the
abstain rate, which is the number the enforcement decision is waiting on.

## Backfilled lines

Lines carrying `"run_id": "backfill-<date>"` and `"attempt": 0` were written by
a maintainer, not by a harvest, for PRs that abstained before the fallback
existed. Everything else in this directory comes from the harvester. A backfill
states a category a human is willing to defend; where the changed paths did not
settle it, the judgement is recorded here rather than guessed by a rule:

| PR | category | why |
|---|---|---|
| 541 | `infrastructure/ci` | site E2E spec fix |
| 542 | `infrastructure/ci` | CodeRAG workflow change |
| 549 | `performance/speculation` | MTP accept-bucket accounting across `spark-model` and `spark-server` |
| 554, 556, 558 | `infrastructure/build-system` | Dependabot manifest and lockfile bumps |
