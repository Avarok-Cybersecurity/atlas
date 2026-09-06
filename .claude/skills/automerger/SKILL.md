---
name: automerger
description: "Find, group and land open PRs as certified STACKS — one certification campaign per stack instead of one per PR. Invoke when a PR backlog needs triage, when related PRs should be stacked on one base chain, or when asking 'which of these are already merged?'. Wraps /iterate for the wave loop and adds merge economics, a supervised agent ladder, and an O.R.A.C.L.E order gate that must clear a stack's SEQUENCE before it is registered (order cannot be changed afterwards), and a rulebook of 33 traps mined from real incidents (each with the check that proves compliance). Born from a 98-PR backlog where 12 of the first 19 PRs triaged were ALREADY on main — closed only because nobody deleted them after batching."
argument-hint: "<repo or scope> [every <N>m]"
---

# /automerger — land the backlog as certified stacks

## The economics, first

Certification is the scarce resource. A PR touching `PERF_PATHS` (`crates`,
`kernels`, `Cargo.*`, `vendor`, `jinja-templates`, `rust-toolchain.toml`,
`3rdparty_patches`) re-opens all 11 gates and owes a **~5 GPU-hour campaign**.

Three numbers set every priority in this skill:

| | |
|---|---|
| **~5 GPU-hours** | one certification campaign |
| **~3 workflow runs** | the cost of a single PR comment |
| **12 of 19** | PRs triaged that were **already on `main`** — closed only because nobody deleted them after batching |

That last number is the point. **The cheapest question is "is this already
in?", and it resolves most of a backlog in seconds.** Ask it before planning
anything. A campaign spent re-certifying merged code is the most expensive
mistake available.

## Non-negotiables

**Assert the branch before every commit.** In a session juggling worktrees,
rebases and detached-HEAD compositions, `git commit` lands wherever HEAD
points — this run lost seven trace commits to a silently detached HEAD, and
"committed locally, push held" pushed a REF that no longer contained them.
Before any wave commit: `[ "$(git branch --show-current)" = "<expected>" ]` or
stop. Corollary: never trust `git show HEAD:` for a comparison — name the
branch explicitly; a wrong-tree read here manufactured a phantom "lost hunk".

**The `NO AUTOMERGE` label is a human veto and absolute.** A PR carrying it —
or marked draft — is untouchable: never stacked, never sieved-and-commented,
never stamped, sealed, closed, or merged, no matter what any scan or sieve
says about it. Humans add it when a PR is too complex, deliberately parked, or
simply theirs to drive. The inventory still LISTS such PRs (with the label
noted) so the trace shows they were seen, not skipped by accident — but every
action stops there. Check the label at ACTION time, not just scan time: a
label added mid-run must take effect immediately. The same veto applies to
the pre-existing `DO NOT MERGE — YET` label and to any PR whose title carries
"DO NOT MERGE" — those are the same human signal in older dialects (e.g.
#658), and a veto is read by intent, not by exact spelling.

- **Closing is autonomous, but only with a named artifact** on `main` — a file,
  constant or symbol at a specific `path:line` — posted as a comment **before**
  the close.
- **Campaigns are never autonomous.** Compose, gate offline, stamp, then **stop
  and report** "ready to certify, N PRs, one campaign". Campaign-free stacks
  land without asking.
- **Every wave ends CITED or AMENDED** (see *Living rulebook*).

## Protocol

Run this inside `/iterate` — invoke it with the goal
*"land the open PR backlog as certified stacks, one campaign per stack"* so the
wave loop, negative-control discipline and tabular reporting come for free.
The phases below replace `/iterate`'s generic "find" step.

### A — Inventory (never skipped)

1. `git fetch --unshallow || git fetch --deepen=2000`, then assert
   `git rev-list --count origin/main` is sane. **A truncated clone makes
   `merge-base` return empty, which reads as "orphan branch" and is false.**
   Re-check in every new worktree — truncation is not always flagged shallow.
2. Classify every open PR with `.github/scripts/pr-containment-scan.py`:
   *contained / applies / conflicts / no-base*, judged **commit by commit**.
3. Classify campaign cost with `.github/scripts/stack-plan.py`: does the diff
   touch `PERF_PATHS`?

The four-category table tells you what the real blocker is before you plan
anything. On the run this skill was built from it was *staleness needing author
rebases* (56 of 89), not certification cost.

### B — Retire what needs no merge

Close contained and superseded PRs. **Comment first, close second**, and end
every close with the recourse offer:

> If any hunk here did not make it across, point at it and I will lift it onto
> a fresh branch.

### B2 — Sieve (mandatory before any PR is included in a stack)

Not every open PR deserves to merge: some are sloppy, some are useless, some
may be malicious. Every candidate passes **three sieves of static analysis**
before inclusion, run by a **sonnet** agent per cluster reading the actual
diff (per-PR static review is diligent reading, not deep design judgement —
the Adversary tier stays on fable):

| sieve | looks for | label on failure |
|---|---|---|
| **security** | CI/workflow tampering, gate/threshold edits riding along, new network calls, secrets, dependency confusion (every lockfile change named and checksum-verified against the registry), command injection, unjustified `unsafe` | `sieve:security` |
| **integrity** | tests that cannot fail, production paths bypassed for tests, dead code, a diff contradicting its own description, silent error swallowing, debug leftovers | `sieve:integrity` |
| **goal** | does it advance the repo's actual goals, or is it useless, stale, or duplicative | `sieve:goal` |

Verdicts are per-sieve PASS/FAIL with a named artifact (file:line, symbol,
package/version) — never a vibe. A PR failing ANY sieve is **excluded from the
stack**, labelled, and given a comment that states the specific defect and the
specific cure ("what clears it"), plus an offer to write the cure ourselves.
PRs passing all three get `sieve:cleared`. Labels go through the REST API
(`gh api -X POST repos/$R/issues/$N/labels`) — `gh pr edit --add-label` fails
SILENTLY on gh 2.45 (GraphQL deprecation), as does `gh pr edit --base`; verify
every label/retarget by reading it back.

### C — Group

Stacks are **one subsystem, one author where possible**. Confirm chain ancestry
with `git merge-base --is-ancestor` per pair; run file-overlap analysis
(`comm -12` of sorted `git diff --name-only` sets) and classify each hit
before touching anything.

### C1 — Build a NATIVE stack. Do not compose commits into a new PR.

★ **THIS IS THE STEP THAT HAS BEEN DONE WRONG.** Corrected 2026-09-06 by the
repo owner, after four stacks in a row were built the wrong way.

**WRONG — what not to do:** cherry-pick or merge the constituent PRs' commits
onto a fresh `stack/<group>` branch, open ONE aggregation PR for the lot, and
close the originals once it lands. It gets a campaign down to one, and it
throws away everything else: per-PR review threads, per-PR authorship in the
UI, the ability to drop one constituent without rebuilding the branch, and the
stack icon — GitHub cannot see a stack that exists only as a merge commit.
Reviewers get a 100-commit diff with no seams.

**RIGHT:** leave every PR on its own branch, with its own author and its own
review. Make the stack out of the BASE CHAIN and then register it.

1. **Base-link, bottom to top.** The bottom PR's base is `main`; every other
   PR's base is the HEAD BRANCH of the PR below it. The API enforces exactly
   this: *"Each pull request's base ref must match the previous pull request's
   head ref."*

   ```bash
   gh api -X PATCH repos/$REPO/pulls/<n> -f base=<head-branch-of-the-PR-below>
   ```

   (`gh pr edit --base` fails silently on gh 2.45 — see the gh-version trap.)

2. **Register the stack**, numbers **bottom first**:

   ```bash
   gh api -X POST repos/$REPO/stacks \
     -F "pull_requests[]=<bottom>" -F "pull_requests[]=<next>" ... -F "pull_requests[]=<top>"
   ```

3. **Read it back.** `gh api repos/$REPO/stacks/<n>` and confirm `open: true`
   and the ORDER. A silent-failure gh path is never trusted on its exit code.

4. **Extend later** with `POST /repos/$REPO/stacks/<n>/add`, body
   `{"pull_requests":[...]}` — the first new PR's base must match the current
   TOP PR's head ref. **Remove** with `POST /repos/$REPO/stacks/<n>/unstack`.

Reference: `https://docs.github.com/en/rest/pulls/stacks`.

★ **ORDER IS THE ONE DECISION YOU CANNOT TAKE BACK.** Everything else in
this protocol is revisable — a constituent can be dropped, a base retargeted,
a branch force-pushed. The SEQUENCE cannot. A registered native stack has no
reorder endpoint, and every workaround costs something:

| you try | what actually happens |
|---|---|
| `PATCH /pulls/{n} -f base=...` on a stacked PR | **422** — *"Cannot change the base branch because the pull request is part of a stack."* |
| `POST /stacks/{n}/unstack` with every member | removes all but one; the survivor sits in a stack reporting `open: false` and then refuses `unstack` with *"cannot be removed from this stack"* |
| force-push the reordered chain first | if a PR's commits land UNDER its own base branch, GitHub sees its head as an ancestor of its base and marks it **MERGED into that branch** — the PR is gone as a review unit |

So the order is decided BEFORE step 2, it is cleared by O.R.A.C.L.E (C2), and
its justification is restated in every wave report (below) for as long as the
stack is open.

**Two placement rules that follow from bottom-up merging:**

- **A fix more than one layer needs goes at the BOTTOM.** Every layer *below* a
  fix is unprotected by it, and lower layers merge first.
- **A repair belongs in the layer that broke it, not above it.** A stack merges
  bottom-up, so a repair riding one layer above the damage leaves the damaged
  layer red and unmergeable while the tip looks green.

**What this changes downstream, and it is not cosmetic:**

- The **top** PR is the one whose merge lands the whole stack, so it is the one
  that pays for certification. That is exactly what `is_stack_layer` decides —
  a PR with another open PR stacked ABOVE it is a lower layer and skips the
  nine release-matrix builds. A composed aggregation PR has nothing above it,
  so every constituent it replaced paid nothing and the economics looked the
  same by accident.
- **Never merge a base out from under its children.** `--delete-branch` on a
  lower layer CLOSES every PR stacked on it, irrecoverably. Retarget the child
  first, then merge the base.
- Dropping a bad constituent is a `PATCH base` on the PR above it plus an
  `unstack`, not a branch rebuild.

### C2 — O.R.A.C.L.E clears the ORDER before the stack is registered

**O**rdering **R**eview **A**nd **C**hain-**L**egitimacy **E**xaminer. An
`opus` subagent (`.claude/agents/oracle.md`), spawned once per proposed stack,
**before** C1 step 2. Its verdict is blocking: no `POST /stacks` without an
`ORDER-OK` on the trace.

It is deliberately NOT the Adversary. The Adversary asks whether this GROUP
should be certified together; ORACLE asks whether this SEQUENCE is the right
one — and that is the question whose answer cannot be revised afterwards.

**Give it verbatim inputs, never a summary:** every PR number, title, head
branch and `gh pr diff --name-only`; the proposed bottom-to-top order; per
adjacent pair the `git merge-base --is-ancestor` result and the `comm -12`
file overlap; and which PRs are red on which required checks.

**It answers six questions in writing, each with evidence:**

1. **Dependency direction** — does any layer need something a layer ABOVE it
   introduces? Bottom-up merging means that layer would land broken.
2. **Shared-fix placement** — is any fix needed by more than one layer? It
   belongs at the bottom, not wherever its author happened to put it.
3. **Independence** — is each layer reviewable and mergeable on its own? A
   layer that only makes sense with the one above it should be one PR.
4. **Blast radius of the top** — the top pays for certification. If the top is
   also the layer most likely to fail a gate, a failure cannot be attributed
   without bisecting; reorder or split.
5. **Merge-order safety** — does any layer delete, rename or move a file a
   lower layer depends on?
6. **Irreversibility** — state the cost of being wrong here, and whether the
   proposer is confident enough to spend it.

**Verdict is `ORDER-OK` or `ORDER-WRONG` plus the corrected order.** Anything
hedged is `ORDER-WRONG`. Record the verdict, its timestamp and its reasoning
on the trace; the reasoning becomes the wave report's justification table.

### D — Gate the top of the stack offline

Every one of these caught a real defect on the run this skill came from. Run
them all before spending a second of GPU:

```
cargo check --workspace --all-targets      # PATH=/usr/local/cuda/bin:$PATH
cargo test --workspace
LoC cap: no .rs over 500 that was not already over on main
bash .github/scripts/certification-selftest.sh
python3 scripts/check_spdx.py
```

### E — Adversarial review (blocking)

The stack plan goes to the **Adversary** before anything is published. It must
clear all four questions or the wave stops. See *Supervision*.

### F — Publish

Comment protocol and stack map, below.

### G — Certify once, then merge

Stamp → **stop for approval** → campaign → seal **last**. A seal is voided by
the next commit; a stamp is not.

## One certification per stack

This is the change that makes stacking pay. Native GitHub stacked PRs
(`gh extension install github/gh-stack`, public preview 2026-07-30) run the full
pipeline on **every layer** — "existing reviews, checks, and merge requirements
all work out of the box" — so on their own they make the economics *worse*:
N layers, N campaigns.

Split the lanes instead:

| lane | runs on | why |
|---|---|---|
| cheap correctness — fmt, clippy, tests, LoC, SPDX | **every layer** | each layer is independently reviewable, and these cost seconds |
| expensive — benchmark gate + release matrix | **the aggregation PR only** | it is the only tree that reaches `main` |

**The aggregation PR is the one with nothing stacked above it** — no open PR
targets its head ref, and that is the *whole* test. The base ref is
deliberately no part of it: under gh-stack's merge-the-top model the top of a
stack has a non-main base yet is exactly the PR whose merge lands everything
on `main`, so a "base != main ⇒ layer" clause waves the landing tree through
uncertified. (Caught live on the #655 → #651 → #650 chain: with the base
clause, all three layers classified as "lower" and nothing would ever have
certified.) Fork PRs never classify as layers — `--base` matches branch names,
and a fork whose branch name coincides with a stack's base branch must
certify, not skip. Both holes have selftest controls proven able to fail.

> **R-43. A stack is a BASE CHAIN plus a registration, not a branch with
> everyone's commits on it.** Composing constituents into one aggregation PR
> gets the campaign count right and loses per-PR review, per-PR authorship, the
> ability to drop one constituent, and the stack icon — GitHub cannot see a
> stack that exists only as a merge commit. EVIDENCE: 2026-09-06, four stacks
> built by cherry-pick before the repo owner pointed at
> `docs.github.com/en/rest/pulls/stacks`. CHECK: every PR keeps its own branch;
> each base = the head branch of the PR below; `POST /repos/{owner}/{repo}/stacks`
> bottom-first; then READ THE STACK BACK and confirm `open: true` and the order.

**Native stacks (public preview) — the mechanics that matter:**

- **Registration — the exact call that works** (proven on stack #885,
  2026-09-04; this is what makes the stack icon appear). The chain must
  already be base-linked (each PR's base = the head branch of the PR below),
  and the numbers go **bottom first**:

  ```bash
  gh api -X POST repos/$REPO/stacks \
    -F "pull_requests[]=<bottom>" -F "pull_requests[]=<next>" ... -F "pull_requests[]=<top>"
  # verify:  gh api repos/$REPO/stacks        (lists stacks; open:true = live)
  # inspect: gh api repos/$REPO/stacks/<stack_number>
  ```

  The response is the stack object (`number`, `open`, ordered
  `pull_requests[]`) — read it back and confirm `open: true` and the order,
  the same read-back discipline every silent-failure gh path demands. The UI
  banner ("This pull request can be stacked…") offers the same conversion by
  hand; the gh-stack CLI needs gh >= 2.90 and is NOT required — the plain
  REST call above works on gh 2.45.
- **Workflow signal**: a registered stack adds
  `github.event.pull_request.stack.{number,size,position,base}` to the PR
  payload — position is 1-based from the bottom, the TOP is
  `position == size`. The classify gate prefers this (race-free, fork-proof)
  and falls back to the open-PR query for unregistered chains; garbage falls
  through, never into a skip.
- **No automatic CI dedup**: GitHub runs every layer's workflows
  independently — the repo-side gate is what buys one certification.
- **Landing**: native stacks merge BOTTOM-UP, each layer rebasing onto main —
  through a merge queue that is one merge_group certification PER LAYER.
  So land a certified stack by **collapsing**: retarget the certified top PR
  to `main` and merge it as ONE queue entry; the layers close as contained.

Three constraints, each learned by breaking them:

1. **Job-level `if:`, never a trigger-level `paths:` filter.** A required check
   that is never *created* blocks the merge forever.
2. **The required context must still report.** Skip the *job*, not the context
   (the `if: always()` summary-job shape).
3. **Ambiguity fails safe toward certifying.** An extra campaign is
   recoverable; an uncertified merge is not.

Certification then measures the **composed** tree, so a regression is
attributable to the stack, not a constituent. That is the price of one campaign
for N PRs — which is why the Adversary must ask whether a failure here could be
attributed, and why every source branch is kept intact so a suspect component
can be dropped and the rest re-certified.

## Wave reports (required)

Every periodic wave report — the tabular update to the user, not just the
trace — **must carry the ordering justification for every open stack**: the
bottom-to-top order, one line per layer saying why it sits where it does, and
ORACLE's verdict with its timestamp.

```
Stack #947 — ORACLE 21:14Z: ORDER-OK

| # | layer | why it sits here |
|---|---|---|
| 1 | #946 ffmpeg ETXTBSY | BOTTOM: three layers above hit this flake, and a stack merges bottom-up |
| 2 | #937 /stamp in-flight | depends on nothing above; its selftest stub conflicts with main and is resolved here |
| 3 | #938 concurrency group | independent of #937, but shares certification-selftest.sh so it follows it |
| 4 | #939 gate classification | TOP: PERF_PATHS, so this is the layer that pays for the campaign |
```

This is not decoration. Order is the only decision in this protocol that
cannot be revised after registration, so it is restated every wave while there
is still time to act on it — **and a justification nobody can write is an
order nobody checked.**

## Comment protocol (required)

For a stack `D → C → B → A` (arrow = *targets*), **A** is the bottom
(base `main`, merges first):

- **A — the aggregation PR — gets the full PLAN comment**: what the stack
  contains, why these group, merge order and rationale, campaign accounting
  (one campaign for N PRs), and what was excluded and why.
- **B, C, D each get a TABLE comment** listing every PR in the stack, **marking
  itself** (`← this PR`), in merge order, with per-row notes.
- **Every PR gets the stack map** (`.github/scripts/render-stack-map.py`).

**The command parser reads only the first word of the first line.** Explanation
comment first, **bare command second** — a `/stamp` appended to prose is
silently ignored.

## Supervision — five agents around the loop

Cheapest tier first, so monitoring never burns an expensive model and
escalation is triggered by **measurement**, not by feel.

| tier | model | role |
|---|---|---|
| **Monitor** | `haiku` | timestamp each wave, record deltas, detect stalls. **Reports only** — never kills, pauses or retries |
| **Triage** | `sonnet` | read the **ledger, not the world**; return `continue` / `slump` / `blocker` / `external` |
| **ORACLE** | `opus` | **stack ORDER only** (C2). Blocking, once per proposed stack, before registration |
| **Adversary** | `fable` (fallback `opus`) | gate every stack before publish |
| **Breaker** | `opus` | root-cause a confirmed blocker. **One at a time**, never twice on the same signature without new evidence |

**Stall is measured, not felt.** The Monitor flags:

| stall | test |
|---|---|
| no-progress | no new wave entry for > 2 tick intervals |
| repeat | same blocking item named in ≥ 3 consecutive ticks |
| queue | hosted jobs **executed** over the window == 0 while runs are queued — count executions, *not* queue depth |
| loop | same command or error signature ≥ 3 times |

**Triage escalates on evidence already in the ledger — it never waits for more
when both halves are present.** `external` requires (a) a **window**
measurement, not a snapshot, and (b) a **discriminating intervention** (e.g.
"freed 24 concurrency slots → 0 starts" rules out caps). When one is missing,
`continue` and name the missing half. When the ledger already contains both,
**escalating is mandatory** — returning `continue` past a confirmed stall is
the polling-a-dead-queue failure this ladder exists to prevent. (This rule was
added because the first negative control caught Triage doing exactly that: the
discriminating measurement was in the ledger and it asked for more evidence.)

**The Adversary must answer, in writing, on the trace:**

1. Is every "already landed" verdict backed by a **named artifact**, not a
   cherry-pick result?
2. Could a campaign failure be **attributed** to one constituent, or would it
   need bisecting across unrelated subsystems?
3. Does anything here spend a campaign a cheap check could have avoided?
4. What is the most likely way this stack fails, and is that failure mode
   covered by an offline gate?

A stack that cannot clear all four does not publish.

## When to stop

Stopping is the decision that fails most often, so it is specified. Every
costly failure on the run this skill came from was a non-stopping failure: a bot
that never stopped recreating its PR, polling that continued against a dead
queue, escalation that would spawn Opus every tick.

Stop when:

- a stack is published and a campaign is requested → **stop, await approval**;
- the Adversary blocks the same stack twice → **stop, surface to the user**;
- a queue stall is confirmed over a window → **stop spawning work**, and report
  the measurement that discriminates between hypotheses;
- no PR remains that can be resolved without a human → **stop the loop**.

Say which of the four it was.

## The trace — an orchestration graph

Append every wave to `docs/AUTOMERGER.md`. Not prose: a **temporal interaction
graph** whose events are typed
`spawn / delegate / communicate / tool / return / aggregate / stop`, with
wall-clock and elapsed. That is the substrate credit assignment needs — and
credit assignment is the open problem when one campaign covers nine PRs and
fails.

Each wave records: typed events; every verdict **with the artifact proving it**;
every grouping and why; **cost** as a first-class term (campaigns and
runner-minutes, measured not estimated); and every mistake the run made and how
it was caught.

## Living rulebook — cite or amend

Every wave ends with one of:

- **CITED** — the rule IDs applied and where
  (*"R-05 named-artifact → closed #712 citing `verify_e2.rs:44`"*).
- **AMENDED** — a new rule learned this wave, **written back into this file**
  with its evidence, its cost, and the check that proves compliance.

A wave that cites nothing and amends nothing either did no judgement or is not
recording it. Treat that as a defect and say so.

---

# RULEBOOK

33 rules, distilled by three adversarial passes from 59 lessons mined from real
incidents. Every rule carries **EVIDENCE** (what it cost) and **CHECK** (the
command or comparison that proves compliance). A rule you cannot check is
decoration; a rule without evidence is an opinion.

AUTOMERGER RULEBOOK — 33 rules (30 distilled from 59 candidates × 3 adversarial reviews, + 3 on stack order)

═══ DO NOT (highest cost first) ═══

1. DO NOT start a multi-GPU-hour certification campaign until the tree is final and `gh pr checks` shows 0 failing / 0 pending non-held checks — record the head SHA and certify exactly it; if HEAD moves mid-campaign, every prior gate record is invalid.
   EVIDENCE: one mid-campaign fix commit invalidated 10 gates (BFCL legs run ~1.6h each); a separate 1.5 GPU-hour leg wrote zero records.
   CHECK: `git rev-parse HEAD` equals the recorded campaign SHA before every gate; `git status --porcelain` empty in the campaign worktree.

2. DO NOT let a bot close-and-recreate its own PR, and never leave a `workflow_run:` trigger unguarded against the bot's own refs — force-push one long-lived PR.
   EVIDENCE: the harvest bot opened 12 PRs in one day (#854…#871) → ~144 workflow runs, roughly half the runner queue starved for hours.
   CHECK: `gh pr list --author <bot> --state all --json number,createdAt` shows ≤1 bot PR created/day; the workflow_run YAML carries a head_branch/actor guard.

3. DO NOT comment on any PR that stays OPEN while an `issue_comment`-triggered workflow shares a cancel-in-progress concurrency group with the check-writing `pull_request_target` run — one comment can permanently cancel the required CLA check. Batch triage decisions; unblock cancelled checks by re-running WITHOUT commenting. Atomic close comments and bare bot commands are the explicit exceptions. If you ever author workflows: never mix a check-writing and a non-check-writing event in one cancel-in-progress group.
   EVIDENCE: 13 of 60 open PRs (22%) blocked by a permanently-cancelled CLAAssistant; one comment on #844 queued 11 runs.
   CHECK: before `gh pr comment`, `grep -l issue_comment .github/workflows/*.yml` cross-checked for shared cancel-in-progress groups; after any comment, `gh pr checks` shows no cancelled required context.

4. DO NOT read `git cherry-pick A..B` printing "is now empty" as containment — the range command stops at the FIRST empty commit; a PR is contained only if EVERY commit picks empty.
   EVIDENCE: #702 had four empty commits followed by one that CONFLICTS; commit-by-commit judgment dropped "contained" from 5 to 2 (false positives #673, #702, #777 — live PRs nearly closed).
   CHECK: iterate `git rev-list A..B` one commit at a time; every pick must resolve empty.

5. DO NOT close a PR as already-landed without a NAMED artifact on main — a file/constant/symbol at a specific path:line, cited in the close comment — tested against MAIN (never a sibling branch), with the absence probe iterating the PR's own `gh pr diff --name-only` list. In this squash-merge repo, recorded SHAs are non-ancestors BY DESIGN: compare trees/content, never commit ancestry; verify any claimed landing commit with `git merge-base --is-ancestor`, and if content landed rewritten, name where.
   EVIDENCE: this discipline alone kept a twice-buggy scanner from corrupting any of 17 closes; #814 was fully landed but the probe grepped gates.js while the PR adds gate-lineage.js; #405 landed rewritten as ssm_batched_copy.rs (179 lines, 8 tests).
   CHECK: `git grep <symbol> origin/main -- <path>` hits before the close is posted.

6. DO NOT resolve conflicts on an external contributor's PR by guessing at intent — request an author rebase, naming the conflicting files. Only mechanical fixes with exactly one defensible reading (compile fix after clean rebase, authorship metadata) are permitted, committed as a SEPARATE commit under your own identity so they stay reviewable as yours.
   EVIDENCE: 56 of 89 backlog PRs conflict; "guessing at intent there is how a rebase quietly changes behaviour" (#582, ten files) — followed by a ~5 GPU-hour campaign certifying wrong code under the author's name.
   CHECK: `git log origin/<their-branch> --author=tbraun96` shows no new content commits; any fix commit's author is your own, with the contributor's patch-ids unchanged.

7. DO NOT declare a PR "can't affect X" from reading its diff's guards — refutation requires a same-box A/B (origin/main vs PR head, identical config/recipe) plus a grep of every flipped flag's CONSUMERS. Until both exist the claim is "unexamined", never "refuted".
   EVIDENCE: #831 was declared inert twice ("every changed line is behind if args.dflash"); the control showed main 10/10·10/10 vs PR 9/10·7/10 — spec_think was consumed on the plain lane despite the dflash_ prefix.
   CHECK: both artifacts recorded before writing "refuted": consumer-grep output and side-by-side A/B results.

8. DO NOT treat a red required check as the author's failure or as broken until you classify it — red = failed | cancelled | held. Conclusion "cancelled" → re-run WITHOUT commenting, never blame the author; a HELD lane (waits on /stamp//seal) reproduces identically on a fresh PR → leave it, never "repair" by close-and-rebuild.
   EVIDENCE: cancelled runs are indistinguishable from failures in `gh pr checks` (#840 could never merge; the wrong move was blaming the CLA signer); held-counted-as-broken caused 4 production incidents and drove the 144-run churn day.
   CHECK: `gh run view <id> --json conclusion` returns "failure" on a job that does not wait for a dispatch comment, before any repair or blame.

9. DO NOT excuse a failing gate with a cached "known flaky" belief — read the last committed records for that gate first; committed history outranks memory, including the user's own memory files.
   EVIDENCE: memory said the gate "drops ~1 sample in 4"; the last seven records on main were all ws_ok=10 followed=10 PASS (08-22 c481c309b0 … 08-30 f0f6e48845) — the failure was a real regression, nearly normalized.
   CHECK: any flakiness claim quotes record SHAs from `git log -p -- <records path>`.

10. DO NOT act on a raw sweep/checker's findings — validate every new checker against the full corpus AND a known-good control, then reproduce each finding against the actual defect before fixing or building guards. When a sweep contradicts a hand-proven fact, suspect the sweep. Sweeps set priorities; only verified evidence justifies a close or a fix.
    EVIDENCE: 17 of 19 wave-24 findings were false; six checker bugs in one session, each caught only by running the checker against ground truth.
    CHECK: every acted-on finding has a reproduction artifact (command + failing output) recorded before the fix.

11. DO NOT grep a test's NAME — or a tail/head-truncated log — for failure evidence; match the result token (FAILED/ok) on the result line of the FULL saved log, and assert the tests executed.
    EVIDENCE: `grep -q 'a_hanging_decoder'` matched the passing "test … ok" line, fabricating a 3/3 "deterministic reproduction" (truth: 0 failures in 30); a `tail -25` pipe fabricated a false "my tests didn't run".
    CHECK: detection matches result tokens or the runner's exit code, and `grep 'test result:' full.log` proves execution.

12. DO NOT push new PRs or diagnose from a single snapshot during runner-queue starvation (queued>20, in_progress=0) — measure completed work over two samples ≥30 min apart, and add load only for the fix itself.
    EVIDENCE: two new PRs added 24 queued runs during a 0-executing window ("I'm contributing to the congestion I'm trying to fix"); the decisive measurement — freeing 24 slots produced ZERO starts, queue head 3h17m old — came only after two contradictory escalations to the user.
    CHECK: during starvation, `gh pr list --author @me --json createdAt` shows nothing new but the fix; block claims cite windowed completion counts.

13. DO NOT tell anyone a PR "isn't blocked" after checking only conflicts/mergeability — merge-queue position blocks just as hard.
    EVIDENCE: "I said 807/808/809 weren't blocking #800. By conflict that's true, but they sit ahead of it in the merge queue" — a user-facing wrong answer.
    CHECK: enumerate all three before the claim: `gh pr view --json mergeable`, `gh pr checks`, merge-queue position.

═══ DO ═══

14. DO close superseded PRs and cancel their queued CI the moment a consolidated stack replaces them — but cancel runs only on branches you own; cancelling a third party's check-writing run can permanently redden their PR.
    EVIDENCE: cancelling freed 24 queue slots (50 → 29 queued) while 0 hosted jobs executed and #856/#868 starved.
    CHECK: `gh run list --branch <old> --status queued` returns empty for every superseded branch.

15. DO stamp/exempt a PR that owes no certification immediately — decided by the mechanical test (diff touches zero PERF_PATHS entries), never by "looks governance-only" — and before bulk-stamping docs-only PRs, check the release workflow for a paths filter; without one each stamp burns the full 9-leg release matrix.
    EVIDENCE: #867 (governance/ only) re-failed the held gate every ~25 min → ~57 red runs/day; stamping docs-only #856 launched nine cross-platform builds (~60-90 runner-minutes).
    CHECK: `gh pr diff <n> --name-only | grep -f PERF_PATHS` empty before stamping; the gate context turns green in `gh pr checks` after.

16. DO open a large backlog with a throwaway-worktree per-commit cherry-pick sweep — capturing each pick's own exit code (`out=$(git cherry-pick ...); rc=$?`, never a pipeline's `$?`) and backing the loop with the decisive end-check: does the composed result differ from main at all?
    EVIDENCE: the 91-PR sweep gave contained 2 / applies 33 / conflicts 56 / orphans 0 — the real blocker was staleness, not certification cost; 17 PRs resolved at 0 campaigns. The `$?`-of-a-pipeline bug silently stopped a scan at 4 of 17 PRs (second occurrence in one session).
    CHECK: the four-category table covers all open PRs; the sweep directs effort only — each individual close still requires rule 5's artifact.

17. DO deepen the clone (`git fetch --unshallow || git fetch --deepen=2000`) before trusting any empty merge-base or ancestry claim — and re-check in EVERY new worktree; each can be truncated independently, and truncation is not always flagged as shallow.
    EVIDENCE: "main has only 6 commits locally" produced a public wrong "#621 is an orphan touching 3573 files" claim; properly cloned (375 commits) it has merge base c1ae36af and touches 2 files. The trap recurred in the very next worktree, after a written warning.
    CHECK: `git rev-list --count origin/main` is sane before any merge-base verdict.

18. DO compose stacks deliberately: confirm chain ancestry with `git merge-base --is-ancestor` per PR pair first, run file-overlap (`comm -12` of the two sorted `git diff --name-only` sets, classifying each hit — module-declaration lists are usually additive) before any stack-of-stacks. The cherry-pick arithmetic here is for the CONTAINMENT sweep (R-16) and for rebasing a layer onto a new base — **never for building the stack itself**, which is a base chain plus a registration (C1), never composed commits.
    EVIDENCE: the naive per-PR loop made #838 come out "empty" and #844/#845 fake-conflict; the tip+unique redo produced "a clean 8-commit stack with no conflicts"; skipping is-ancestor risks re-certifying contained work at ~5 GPU-hours/campaign.
    CHECK: no "empty" picks mid-chain; the overlap listing exists in the plan before the first pick runs.

19. DO fold dependency-bump PRs by taking each Cargo.toml change and regenerating Cargo.lock ONCE at the end — never cherry-pick them.
    EVIDENCE: five dependabot bumps all touch Cargo.lock; the fold gave one certification campaign instead of five, with zero pairwise lockfile conflicts.
    CHECK: each pick's file list excludes Cargo.lock; the single terminal `cargo generate-lockfile` succeeds.

20. DO pre-run CI's exact check commands locally while runs sit queued.
    EVIDENCE: a /stamp dispatch (a ~10-second job) sat queued 210 minutes; each failure found by CI instead of locally cost a 30-80+ minute queue round-trip, worst case hours at 0 executing.
    CHECK: a local run log with rc=0 exists before each push.

21. DO confirm a check is a required context before trusting it to gate anything — a reporting check that is not required gates nothing.
    EVIDENCE: "The site's 492 unit tests gate nothing: 'Site unit tests' is not a required check" (#810) — a failing test could not block a deploy, for an unknown period.
    CHECK: the exact context name appears in `gh api repos/{o}/{r}/branches/main/protection --jq '.required_status_checks.contexts[]'`.

22. DO fix a contributor's CLA/attribution problem by amending ONLY authorship metadata, proving the tree hash unchanged, and choosing the email their commit history supports.
    EVIDENCE: #828 — tree hash 71c27e7b identical before and after the amend (credit moved, content did not); rrstesiak@hotmail.com chosen on 138-vs-2 commit dominance.
    CHECK: `git rev-parse HEAD^{tree}` identical pre/post amend.

23. DO copy edit anchors from the file's actual bytes (`grep -n`, `od`) — never from sed/display output — and assert each anchor matches exactly once before writing anything.
    EVIDENCE: the indented-output trap hit three times in one session (≥9 anchor misses total); a 0-match sabotage anchor made a negative control pass vacuously — a fake green on the control itself.
    CHECK: `grep -cF "$anchor" file` == 1 before every write.

24. DO prove every new guard/check by sabotaging the exact BEHAVIOUR it protects — leaving the file syntactically valid; syntax breakage reds the positive test and proves nothing — watching it go red, then green on restore; verify result rows append BEFORE the summary/exit code is computed; and run it against the pre-fix tree to prove it catches the original defect, not a shadow of it.
    EVIDENCE: "all triage cases pass" printed before the new rows — a check that could not fail; a stray `set -e` silently truncated the suite at 97 of 101 checks across every prior wave; assert-preflight run at pre-fix HEAD refused on both write classes, proving it would have caught #843 and #847 on the day each was written.
    CHECK: sabotage → suite rc=1; restore → rc=0; pre-fix checkout → nonzero.

25. DO make every negative control assert WHICH guard fired (its diagnostic string), never a bare or platform-dependent exit code (>128 signal rc's differ between shells: SIGPIPE is 141 locally, 2 in CI) — and mutation-test assertion greps (disable the action; a grep matching the QUERY about the action stays green) plus assert stubs actually produce their declared outputs.
    EVIDENCE: a control asserting only rc=1 stayed green at 32/32 with its guard deleted (a sibling guard exited 1 — unsigned records could have entered .benchmarks/ unnoticed); the pinned-141 control redded a full CI cycle on a healthy branch; `certed()` matched its own `--jq` lookup and reported a certificate that was never posted.
    CHECK: controls pin exit code AND diagnostic message; disabling the action flips the assertion red; `test -f <stub-output>` after invocation.

26. DO give every guard an anti-vacuity control and an honest scope: a guard that stops finding work must FAIL, never pass vacuously; pair each refusal/error-path check with proof the side effect did NOT happen and that error codes surface rather than swallow; and write the guard's blind spots into the guard itself, with a control input for every capability it claims.
    EVIDENCE: the DFlash2 gate had been passing on partially-vacuous cells; pr-review.sh exited 0 on a model-endpoint 500 with no comment posted — indistinguishable from success (5 real gaps found in one selftest pass).
    CHECK: guard on empty/absent input → nonzero; each refusal test pairs with `test ! -f <side-effect>` / zero request-log entries; the guard file contains its limits block.

27. DO split a file over the LoC cap by exact piecewise copy PROVEN function-by-function, choosing the moved half so verdict/gate logic stays inside the audited boundary; use the allow-list only when no clean seam exists (e.g. one 500-line function), always with a rationale comment AND a tracking issue.
    EVIDENCE: the body comparison caught two would-be-shipped defects in one session (a truncated `use` block, an unbalanced cut); silent allow-listing previously grew 1121- and 1484-line monoliths that took a dedicated wave to lift.
    CHECK: per-function body diff shows zero differences; every allow-list entry has an adjacent rationale and `gh issue view <n>` exits 0.

27b. DO treat a fix that REMOVES a bound (a clamp, a truncate, a saturating cast) as a security change, not a validation change: prove (a) every path that can still reach the now-unbounded value passes the replacement check — trace each adapter to the function that actually calls it, not to the module — and (b) the downstream consumer is safe WITHOUT the bound, so a missed path degrades rather than crashes.
    EVIDENCE: the top_logprobs fix deleted `n.min(20)` at the wire edge and moved the check to `validate_input`. Both halves held — `responses.rs:154` calls `chat_completions_inner`, which calls `validate_input` at its own line 227, and the SSOT top-k math clamps with `k.min(indexed.len().saturating_sub(1))`, so even the u8 ceiling of 255 cannot read out of bounds — but neither was proven by the lane that made the change, and a reviewer died mid-sentence raising exactly this.
    CHECK: for each adapter, name the enclosing function of the validation call and show the adapter reaches it; and quote the downstream clamp expression.

═══ ETIQUETTE ═══

28. DO close someone's PR only via the atomic `gh pr close <n> --comment '<full reasoning>'` — so the explanation can never lag the close.
    EVIDENCE: #699/#703/#704 were closed silently and needed a retroactive apology pass ("Apologies for closing before commenting; the reasoning belongs here").
    CHECK: no bare `gh pr close` without `--comment` in the session's command history.

29. DO end every already-landed close with the recourse sentence: "If any hunk here did not make it across, point at it and I will lift it onto a fresh branch."
    EVIDENCE: posted verbatim on #699 and #703 — cheap insurance against the empty-cherry-pick false-positive class measured the same session (three near-miss wrong closes).
    CHECK: the close comment body contains the recourse sentence before posting.

30. DO put a bot command as the first word of the first line of its own bare comment — explanation comment first, command comment second.
    EVIDENCE: "Parser confirmed: first word of the first line — that's why my explanation ate the /stamp; the standalone one registered (Stamp: success)."
    CHECK: the command comment matches `^/<cmd>` and the dispatch appears in `gh run list` (or the bot replies).

═══ STACK ORDER (the irreversible decision) ═══

31. DO settle the ORDER of a stack before registering it — a native stack cannot be reordered. `PATCH /pulls/{n} -f base=` on a stacked PR returns 422 "Cannot change the base branch because the pull request is part of a stack"; `POST /stacks/{n}/unstack` can strand one member in a stack reporting `open: false` that then refuses further unstacking; and if a reorder push puts a PR's commits underneath its own base branch, GitHub marks that PR MERGED into the branch and it is gone as a review unit. O.R.A.C.L.E (C2) must return `ORDER-OK` first.
    EVIDENCE: reordering stack #940 on 2026-09-06 to move the ffmpeg fix to the bottom cost #944 — auto-merged into `fix/gate-sources-must-be-classified`, reopened as #946 under a rebuilt stack #947.
    CHECK: an `ORDER-OK` verdict on the trace before any `POST /stacks`, and `gh api repos/$REPO/stacks/<n>` read back showing the intended order.

32. DO put a fix that more than one layer needs at the BOTTOM, and a repair in the layer that broke it — a stack merges bottom-up, so everything below a fix is unprotected by it and a repair above the damage leaves the damaged layer red.
    EVIDENCE: the ETXTBSY flake fixed at the TOP of #940 still failed the required `cargo test --workspace` on #880 (19:32Z) and #938 (20:10Z) 40 minutes apart; moving it to the bottom turned #938 green with no other change. Separately #881 sat red on clippy+fmt while the tip was green, because the repair rode one layer above the code it repaired.
    CHECK: run each layer's local checks AT THAT LAYER, not only at the tip — every layer independently green.

33. DO print the stack's ordering justification in every wave report while the stack is open — one line per layer, plus ORACLE's verdict and timestamp.
    EVIDENCE: on 2026-09-06 a wrong order sat visible in the base chain for six hours and nobody, the author included, articulated it until the same flake failed the same test twice.
    CHECK: the wave report contains an ordering table for every open stack; a stack with no justification is treated as unchecked, not as fine.

