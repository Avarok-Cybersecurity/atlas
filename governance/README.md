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
