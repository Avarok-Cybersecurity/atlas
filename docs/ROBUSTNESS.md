# Certification pipeline robustness

An append-only record. One entry per wave: what was found, the evidence, what
changed, and — the part that matters — what the *negative control* proved.

A guard that has only ever been seen to pass is indistinguishable from a guard
that cannot fail, and the second kind is worse than no guard at all, because it
reports safety. So every entry here names the control and what it caught.

**The gate:** `bash .github/scripts/certification-selftest.sh` — offline, no
network, no GPU. It runs in the security job on every PR.

---

## Baseline — 2026-09-03

Before any of this work:

| Surface | Tested in CI? |
|---|---|
| `gate/signing.rs` | ✅ 12 Rust tests |
| `gate/card.rs` | ✅ 9 Rust tests |
| `seal-coverage.py` | ❌ none |
| `assert-cmd-runner-safe.py` | ❌ none |
| `render-certificate.py` | ❌ none |
| `certification-state.sh` | ❌ none |
| `pr-review.sh` | ❌ none |
| `ci.yml` stamp / seal / expedite logic | ❌ none |
| one-commit-one-signer step | ❌ none |

Open at baseline: PR #843 red on `PR benchmark gate` and `seal status` — both
expected (unstamped, unsealed), not defects.

---

## Wave 1 — the shell half had no tests at all

**Found.** The Rust half of the certification pipeline has 21 tests that run on
every PR. The shell and Python half — which decides who may stamp, who may seal,
whether the self-hosted runner can reach untrusted code, and whether a record
was signed — had **zero**. Every one of those guards had been verified once, by
hand, in a terminal, and then trusted permanently.

That is the highest-severity finding available here, because it is not a bug in
any one guard: it is the absence of anything that would notice a bug in any of
them.

**Changed.** Added `.github/scripts/certification-selftest.sh` — 19 checks, of
which **11 are negative controls** — and wired it into the security job, which
already runs on every PR, so this costs no new queue slot.

**The control proved it.** The suite was sabotaged four ways and watched go red:

| Sabotage | What went red |
|---|---|
| `seal-coverage.py` stops refusing unsupported CODEOWNERS patterns (fail-open regression) | all 3 fail-closed controls |
| `assert-cmd-runner-safe.py` blinded to head-ref checkouts | `control: checks out the PR head on the command runner` |
| `render-certificate.py` stops un-hiding co-author slots | `three authors -> 1 visible slots` |
| — | positives stayed green in every case |

The third reproduces a real bug shipped earlier in the day: co-authors were
silently dropped from a certificate that rendered and looked correct. It is now
impossible to reintroduce without CI saying so.

**A mistake worth recording.** The first sabotage attempt broke Python syntax
rather than behaviour, so the *positive* test failed and the fail-closed
controls stayed green. That looked like a caught regression and was not — it
proved only that the suite notices a file that will not parse. Sabotage has to
target the behaviour under test, or the control is theatre.

**Still open.** `certification-state.sh` and the `ci.yml` stamp/seal/expedite
shell have no coverage in the suite yet. The one-commit-one-signer step is
tested only in a scratch repo, by hand.

---

## Wave 2 — the gate itself was not portable

**Found.** CI went red on `cargo deny` — the job wave 1 had just added the
self-test to. Not a flake: the suite's three certificate-rendering checks failed
on `ubuntu-latest` because `render-certificate.py` imports `segno`, which had
been installed by hand on this box and on avarok but exists nowhere in CI.

The gate built to catch regressions was itself broken in a way that only CI
could see. Worth stating plainly: wave 1 reported "19 passed" from a machine
where the dependency happened to be present, and that number was true and
useless.

**Changed.** The suite now checks its prerequisites up front — `python3`, `jq`,
PyYAML, segno — and exits 2 with a named list if any are absent. The security
job installs them before running it.

**The control proved it.** With a shimmed `python3` that reports segno missing,
the suite exits **2** and names `python3-segno`, rather than running a reduced
set of checks and reporting success. With the prerequisites present it is 19/19
again.

The tempting fix was to skip the QR-dependent checks when segno is unavailable.
That would have turned a suite that *cannot run* into a suite that *reports
success* — the precise failure mode this file exists to prevent, reintroduced
inside the file itself.

**Still open.** Unchanged from wave 1: `certification-state.sh` and the
`ci.yml` stamp/seal/expedite shell have no coverage; the one-commit-one-signer
step is exercised only by hand.

---

## Wave 3 — a control that asserted an implementation detail

**Found.** With the dependency fixed, CI still failed — on the SIGPIPE control
itself, not on the code it guards. The pre-fix form exits **2 in CI** and **141
locally**: `jq` traps `EPIPE` and exits 2 with a diagnostic, while other builds
take the signal and die silently with 141. The control asserted `rc=141`, so it
passed on this box and failed on `ubuntu-latest`.

The guarded behaviour was correct on both machines the entire time. The control
was wrong.

**Two wrong fixes, both tried.** First `rc=141` → "non-zero **and** the output
mentions a broken pipe". That went red locally, where the process dies silently
and prints nothing. Pinning the *message* is the same mistake as pinning the
*code*: both are platform detail dressed up as an invariant.

**Changed.** The control now asserts the property that is actually invariant:
the **piped** form fails on the very input where the **unpiped** form, asserted
green one line above, succeeded. That isolates the pipe as the cause without
depending on how the platform reports it.

**The control proved it.** Substituting the fixed (unpiped) command in for the
"pre-fix" one turns that control red — so it is still measuring something. 19/19
locally.

**The lesson, since it generalises.** A negative control that pins an exit code
or an error string is testing the platform, not the property. Ask what would
still be true on a machine you have never used.

**Still open.** `certification-state.sh` and the `ci.yml` stamp/seal/expedite
shell remain uncovered. CI has not yet confirmed this wave.

---

## Wave 4 — the ci.yml decision logic, and a control that measured nothing

**Found.** Four shell blocks inside `ci.yml` decide whether anything merges
uncertified — the stamp verdict, the seal verdict, the alias that mirrors
certification into a required check, and the one-commit-one-signer step. They
live inside a workflow file, where nothing could execute them. None had ever
run outside a real CI job.

**Changed.** The suite now extracts each block from `ci.yml` with PyYAML and
runs it against a stubbed `gh`. 13 new checks, 7 of them controls. Total 32.

**The control proved it — and then failed to.** Three sabotages of `ci.yml`:

| Sabotage | Caught? |
|---|---|
| alias flips `WEB_ONLY = "true"` to `!= "false"` (the fail-open doctrine violation) | ✅ `control: a broken classifier (empty web_only) stays red` |
| stamp stops short-circuiting non-PR events (would wedge the merge queue) | ✅ `merge_group was held` |
| **one-commit step stops requiring a signature on added records** | ❌ **32/32, green** |

The third is the finding. The control asserted only `rc=1`, and several guards
in that step overlap: a record with no sidecar trips the signature check *and*
the one-signer check, because "no sidecar" reads as a distinct signer. Deleting
the guard under test left the suite green, because a different guard caught the
same fixture.

**A control that passes when the thing it tests has been deleted is measuring
nothing.** It is the exact failure this record exists to catch, and it was in a
check written one wave earlier specifically to catch such things.

**Fixed.** `want_rc_msg` pins the exit code *and* which guard fired. The three
one-commit controls now assert their own diagnostic — `Unsigned record added`,
`span more than one commit`, `more than one signer`. Re-running sabotage 3 now
fails correctly, naming the missing message.

**Still open.** `certification-state.sh` has no coverage. The seal job's
`merge_group` branch — which derives the PR number from a queue branch name — is
exercised only by hand.

---

## Wave 5 — the state machine, and two harness bugs that looked like results

**Found.** `certification-state.sh` picks one of eleven states from the PR's
merged flag, its mergeable_state, its queue entry and three check-run
conclusions. It is what the bot shows an author. Nothing tested it.

**Changed.** 11 checks driven by a stubbed `gh`, covering every stage the ladder
can reach plus the two precedence rules that outrank it. Total 43.

**Two of my own bugs, both of which first read as findings.**

*One.* Four state checks failed with empty output. The obvious reading was a
defect in the script. Running it directly showed it emitting `stage-1`
perfectly — the fault was my stub: `[ -n "$v" ] && echo "$v"` returns non-zero
on an empty value, so the stub exited 1 and every *absent check run* looked like
an *API failure*. Fixed by printing unconditionally. Had I trusted the first
reading, I would have "fixed" a script that was already correct.

*Two.* Sabotage B — forcing `has_seal=true` — showed the suite green at 43/43,
which reads as a hole. It was not. The anchor matched **zero** times: the real
line has two spaces after the semicolon and quotes around `success`. The
sabotage never applied. Re-run with an asserted anchor, it turns **four** checks
red, including both "a failed Seal is not a seal" controls.

A silently no-op'd sabotage is indistinguishable from an uncaught regression,
and both look like green. Assert the anchor before drawing the conclusion.

**The controls proved it.** Removing the merged-outranks-everything rule turns
`a merged PR reads merged` and `merged outranks a full stage-3 board` red.
Treating any Seal conclusion as a seal turns four red.

**Still open.** The seal job's `merge_group` branch — deriving a PR number from
a `gh-readonly-queue/<base>/pr-<N>-<sha>` branch name — is exercised only by
hand. `/expedite` has no end-to-end coverage; its `admin`-only refusal and its
required-reason refusal are both untested.

---

## Wave 6 — who may bypass the gate

**Found.** The command handlers decide who may release the expensive lane, who
may vouch for a diff, and — with `/expedite` — who may merge without proving
anything at all. None was tested. `/expedite` is the highest-consequence path in
the pipeline and had zero coverage from the moment it was written.

**Changed.** 11 checks, 8 of them controls, driven through a `gh` stub that
records every call. Total 54.

**The assertion that matters.** A refusal is not merely "prints a message" — a
refused command must create **no check run**. A refusal that still mints the
mark is cosmetic, and would be invisible in a log. Every control here asserts
both halves: a comment was posted *and* no `Stamp`/`Seal`/`Expedite` was minted.

**The controls proved it.** Three sabotages, each with an asserted anchor so a
silent no-op could not masquerade as a pass:

| Sabotage | What went red |
|---|---|
| `/expedite` accepts `write`/`maintain`, not just `admin` | `control: write access cannot expedite` |
| `/expedite` stops requiring a reason | both reason controls |
| `/seal` accepts authorship instead of write access | `control: authorship was accepted as a seal` |

The last is worth stating plainly: authorship and ownership are different
claims. A stamp says "this is ready to cost an hour of runners", which its
author is well placed to judge. A seal says "a codeowner has read this diff",
which its author is not. Conflating them would let anyone self-certify their own
work, and now that cannot regress silently.

**Still open.** The seal job's `merge_group` branch — deriving a PR number from
`gh-readonly-queue/<base>/pr-<N>-<sha>` — is exercised only by hand. `/help` and
`/review` have no coverage; both are low-consequence (they post text and cannot
change a verdict), which is why they are last.

---

## Wave 7 — the merge-queue seal path, and the hollow control again

**Found.** The seal job also runs inside the merge queue, where there is no
`pull_request` payload and the PR number must be recovered from a
GitHub-generated branch name, `gh-readonly-queue/<base>/pr-<N>-<sha>`. Wrong
here means either the queue deadlocks or an unsealed entry lands. Hand-tested
only.

**Changed.** 8 checks, 5 controls. Total 62.

**A wrong assertion of mine.** The first version asserted the recovered PR
number appears in the success output. It does not — the success path prints only
the sha. The number is named on the *failure* path, so the assertion moved
there: `PR #840` in the refusal proves both that the number was derived from the
branch and that the right PR was consulted. A wrong number would have looked up
someone else's seal and passed.

**The hollow control, a second time.** Sabotage A deleted the fail-closed branch
and the suite stayed green at 62/62. Not a safe outcome — an accident. With the
branch gone the script still exits 1, via
`[: REFUSE: integer expression expected` falling through to the generic "Not
sealed" message. Safe today, fragile tomorrow, and it tells the operator the
wrong thing: *"no seal"* when the truth is *"we could not look"*.

The control asserted `rc=1` and nothing else, so it could not tell a deliberate
refusal from an arithmetic crash. It now pins the message — `Could not read the
seal` — and re-running the sabotage fails correctly.

This is the same defect class as wave 4, found in a different file one wave
apart. The generalisation is worth stating: **when several paths can produce the
same exit code, asserting the exit code tests none of them.** Pin the
diagnostic.

**Still open.** `/help` and `/review` have no coverage. Both only post text and
cannot change a verdict, which is why they are last — but "cannot change a
verdict" is itself an untested claim.

---

## Wave 8 — "it cannot change a verdict" was an inspection, not a test

**Found.** `/help` and `/review` post text and mint nothing. That was true by
reading the file, which is exactly the kind of claim that stops being true
quietly. It matters more than it looks: `/review`'s permission check is
deliberately weak — anyone who can comment may ask it a question — because it
only posts prose. Give it the ability to create a check run and an outside
contributor could mint their own `Seal` by asking about the diff.

**Changed.** `.github/scripts/assert-command-authority.py`: only
`/stamp and /seal` and `/expedite` may POST a check run, re-run CI, or PATCH,
PUT or DELETE anything. It also pins the workflow to `permissions: {contents:
read}`, since every step inherits the top-level grant and none of them needs
more. 5 checks, 4 controls. Total 67.

**The controls proved it**, on synthetic workflows and then on the real one:

| Sabotage | Result |
|---|---|
| `/review` gains `POST .../check-runs -f name=Seal` (real file, anchor asserted) | ❌ refused |
| top-level permissions widened to `contents: write, checks: write` (real file) | ❌ refused |
| `/review` gains `/rerun` (synthetic) | ❌ refused |
| a text-only `/review` | ✅ passes |

**What this closes.** The three preceding waves tested whether a guard reaches
the right verdict. This one tests whether a command *has the authority to reach
a verdict at all* — a different question, and the one an attacker would ask
first. The permission checks in wave 6 only matter while the weakly-checked
commands stay powerless.

**Still open.** `pr-review.sh`'s own behaviour — its refusals on a missing key
or a non-200 — is untested; the harness would need to fake an OpenRouter
endpoint. The bot's state-comment editing (marker lookup, in-place PATCH) has no
coverage.

---

## Wave 9 — /review's refusals, and a stub that lost the message

**Found.** `pr-review.sh` refuses in three ways — no model id, no API key, a
non-200 from the endpoint — and every one of them exits **0** on purpose: a
`/review` that cannot reach a model must not fail anyone's CI. That makes the
exit code useless as an assertion, so all three had to be pinned by message.
None was tested.

**Changed.** 8 checks, 5 controls. Total 75.

**A harness bug that read as a script bug.** Five checks failed with "it never
explained itself". The script had explained itself perfectly: it posts with
`gh api ... -F body=@-`, so the message arrives on **stdin**, and my stub only
recorded argv. Third time this record notes the same shape — the first reading
of a red test was "the code is broken" and the truth was "the harness is".

**The controls proved it**, each with an asserted anchor:

| Sabotage | What went red |
|---|---|
| the missing-key guard removed — would call the endpoint with no credential | both the message check and `no key -> no request attempted` |
| a non-200 reported as success | `a 500 is reported, not swallowed` |
| the `UNTRUSTED` fence renamed | `PR prose is not fenced` |

The last is the one worth keeping. A PR's title and body are attacker-controlled
text that gets fed to a model. They are fenced between `UNTRUSTED` markers with
the system prompt told to treat anything inside as data. Remove the fence and a
PR title becomes an instruction — and the guard now notices.

**Still open.** The bot's state-comment editing — finding its own comment by the
`<!-- atlas-certification-state:… -->` marker and PATCHing it in place — has no
coverage. That is the last uncovered path.

---

## Wave 10 — the bot's comment lifecycle, and two hollow assertions of mine

**Found.** The bot keeps ONE comment and edits it in place, except on merge,
where it posts a second one — because GitHub does not notify on an `@mention`
added by *editing* a comment, so tagging the authors in the edited comment would
alert nobody. It also fires on `check_run` completions *after* a merge, so
without an idempotency guard a contributor is tagged once per event. None of it
was tested. This was the last uncovered path.

**Changed.** 6 checks, 4 controls. Total 81.

**Two hollow assertions, both mine, both found by the checks disagreeing.**

*One.* `certed()` grepped the recorded calls for `atlas-certificate`. The
idempotency **query** contains that literal string inside its `--jq` filter, so
the helper matched the *lookup* and reported a certificate that was never
posted. It could not distinguish "asked whether one exists" from "posted one".
Now it requires a `POST` to the comments endpoint carrying the marker.

*Two.* With that fixed, the positive went red: "merged -> no certificate". The
bot was right again. My `rsvg-convert` stub was a no-op that produced no file,
so the `sha256sum` naming it failed and `set -euo pipefail` aborted the step
before the POST. A stub that does not produce its output is not a stub, it is a
different failure.

That is the fourth time in this record that a red check meant the harness was
wrong, not the code. The pattern is consistent enough to state as a rule: **when
a new test fails against code that has been working, suspect the test first.**

**The controls proved it**, each anchor-asserted:

| Sabotage | What went red |
|---|---|
| the certificate loses its once-only guard | `a certificate already posted is not posted again` |
| the bot posts instead of editing | `an existing comment is edited, not duplicated` |
| the certificate is posted regardless of state | `a certificate was posted for a PR that has not merged` |

**Coverage is now complete.** Every gate, command and bot path in the
certification pipeline has at least one check and at least one negative control
proven able to fail: seal coverage, runner safety, certificate rendering,
pr-review truncation and refusals, the stamp/seal/alias/one-commit decision
logic, the eleven-state machine, command permissions, command authority, the
merge-queue seal path, and the bot's comment lifecycle.

**Still open.** Nothing structural. The suite runs offline, so it cannot catch a
defect that only appears against the real GitHub API — the App's token
permissions, for instance, are asserted nowhere and would fail only in
production.

---

## Wave 11 — the offline gap: is the App still allowed to act?

**Found.** Wave 10 closed the last *logic* path and named what remained: the
suite runs offline, so it proves the pipeline decides correctly but not that it
is still permitted to act. Every write goes through an installation token, and
an App whose grant narrows — an org policy, an edit to the App, a permission
added to the manifest and never accepted — keeps working right up until someone
types `/stamp`.

Worse, several of those failures are **silent by design**. The bot swallows API
errors so a broken lookup cannot fail a contributor's CI; the certificate step
treats an unreadable comments API as "already posted". Both are correct
behaviours, and both are exactly why a permission regression would go unnoticed
until a release was blocked.

**Changed.** `certification-preflight.yml` probes each permission with the
smallest real call that needs it — weekly and on demand. It names *why* each
one matters, so a failure says what will break rather than which endpoint
returned 403:

    pull_requests:read   /stamp resolves the head sha and the author
    checks:read          every gate reads Stamp, Seal and Expedite
    issues:read          the bot finds its own comment by marker
    members:read         /stamp and /seal check who is asking
    actions:read         /stamp re-runs the held CI run
    contents:read        the bot-cards branch hosts generated certificates
    checks:write         /stamp, /seal and /expedite mint their marks

`checks:write` is probed for real, by creating a check run named "Certification
preflight". Without it the three commands that mint marks are all dead, which is
the whole pipeline. A check run is additive and self-describing, so the probe
costs one line in the checks list and leaves nothing to clean up.

**The control proved it.** All six read probes pass against the live API with a
valid token, and **all six fail** with an invalid one — so the probes are
reaching GitHub rather than passing vacuously.

**Why this is not in the self-test.** It needs the network and a real App token,
which the offline suite deliberately does not have. Running it per-PR would also
add a check run to every PR for no benefit. Weekly catches a grant that was
narrowed; on-demand covers the case where someone has just edited the App.

**Still open.** Nothing known. The pipeline's logic has 81 offline checks with
50 controls, and its authority now has a live probe with a control. What is not
covered — and cannot be, short of a staging org — is GitHub changing the
semantics of an endpoint underneath us.

---

## Wave 12 — the guard on the guard

**Found.** Eighty-one checks and fifty controls hang on **one line in one
workflow**. Delete that line and every one of them stops running — silently,
with `cargo deny`, the job it lived in, still green. Ten waves of work,
removable in a one-line diff nobody would notice in review.

The good half: `cargo deny` *is* a required context, so a **failing** suite does
block a merge. The gap was never enforcement of the result; it was enforcement
of the **wiring**.

**Why it could not go in the suite.** If the step is removed the suite never
executes, so a check inside it can never fire. A guard cannot notice its own
absence. It has to live somewhere that still runs.

**Changed.** `assert-selftest-wired.py`, hosted in the **LoC-cap job** — a
different required context. Removing the suite from `security.yml` now breaks
`Enforce ≤500 LoC per source file` instead. It asserts two things: that some job
invokes the suite, and that the job reports under a name branch protection
requires — because a suite in a job nobody has to pass is a suite that can be
ignored.

**The controls proved it**, both anchor-asserted:

| Sabotage | Result |
|---|---|
| the self-test step deleted from `security.yml` | ❌ *"nothing runs certification-selftest.sh"* |
| the suite moved to a non-required job | ❌ *"runs, but only in non-required job(s)"* |

**On the literal in that file.** `REQUIRED_CONTEXTS = {"cargo deny"}` is
hardcoded, because the assertion must work without network access. If branch
protection changes, that line is what needs updating — and the failure message
says exactly that rather than leaving someone to guess.

**Still open.** Nothing known. This wave closed the last structural gap I can
name: the pipeline's logic is covered offline, its authority is probed live, and
its wiring is now guarded from a job that cannot be removed in the same edit.

---

## Wave 13 — the Rust tests were trusted the way the shell was before wave 1

**Found.** `gate/signing.rs` and `gate/card.rs` have 21 tests that run on every
PR, and this record has leaned on that fact since wave 1 to argue the Rust half
was covered. Not one of them had ever been **shown to fail**. That is exactly
the position the shell half was in before any of this started, one level up:
green tests, trusted because they are green.

**Changed.** Nothing in the code. Three sabotages, to find out whether the
existing tests measure anything.

| Sabotage | Caught by |
|---|---|
| signature verification always succeeds | `editing_the_record_breaks_the_signature`, `a_record_cannot_be_repointed_at_another_commit`, `a_signature_from_another_record_does_not_transfer` |
| `sig_path` uses `with_extension("sig")` | `the_sidecar_path_appends_and_does_not_eat_a_versioned_model_name`, `no_committed_record_escapes_the_cutover_unsigned` |
| card formats a metric to 3 decimals instead of 1 | `formats_round_the_way_a_reader_expects` |

All three caught, by the tests written for them. The Rust half is genuinely
covered — now demonstrated rather than assumed.

**A sabotage of mine that was inert, and why it matters.** The first attempt at
the `sig_path` bug used `with_extension("json.sig")`. On
`…-qwen3.8-27b.json` that yields `…-qwen3.8-27b.json.sig` — byte-identical to
appending. The suite stayed green and I very nearly recorded an uncovered path.
It was not one: the sabotage did not change behaviour. The real bug is
`with_extension("sig")`, which produces `…-qwen3.8-27b.sig` and is caught
immediately.

Twice now a green result has meant "my sabotage did nothing" rather than "the
guard is missing" — wave 5 with a non-matching anchor, and here with a matching
anchor that made no behavioural difference. **Asserting the anchor is necessary
and not sufficient. Check that the edit changes what the code does.**

**Still open.** Nothing known. Every layer — shell logic, Rust logic, live
authority, and the wiring that runs it all — has now been shown to fail when the
thing it guards is broken.

---

## Wave 14 — landing it, and the pipeline exercising itself

**Found.** Thirteen waves of guards protect nothing while they sit on a branch.
The suite gates `main` only once it is *on* `main`.

**Changed.** Nothing new. `/stamp` and `/seal` on #843, dispatched against the
branch's own workflow so they ran on the self-hosted runner the PR introduces.

**The measurement.** Both marks recorded in **47 seconds** for two commands.
Earlier the same day, on hosted runners, a single `/stamp` waited **220
minutes**. Same repo, same commands, same account.

That number is the point of the runner change, and it is now demonstrated by the
change certifying itself rather than by a benchmark written to flatter it.

**What this wave actually proves.** The pipeline ran its own three stages on the
PR that hardens it: `/seal` verified codeowner coverage across the diff,
`/stamp` released the held lane, both marks landed on the head, and the 81-check
suite plus the wiring guard ran as required contexts on the same commit.

**Still open.** Nothing known. The record's last entry is the merge itself.

---

## Wave 15 — #843 merged, and the merge exposed a real defect

**#843 merged** at 21:10:18Z as `dc6dd82a31`. The suite, the wiring guard, the
authority assertion and the preflight are all on `main`, and the guard confirms
the suite still runs in a required job.

**The bot answered the merge in 3 seconds** and finished in 25, on the
self-hosted runner. It posted the first certificate this pipeline has ever
produced — #836 and #840 could not certificate themselves, because a
comment-handler workflow runs from the default branch *as it was when the event
fired*, and the machinery was landing in those very merges.

**Then the verification found a defect.** The certificate comment shipped a
broken image: `bot-cards/pr-843-112aac4b.png` is a **404**. `rsvg-convert` and
`segno` are both present on avarok, the branch exists — but it still contains
only `README.md`. The upload never happened, almost certainly for want of
`contents: write`, and the PUT is deliberately non-fatal so that a missing grant
cannot swallow the certificate itself. Non-fatal made it **silent**.

**Two fixes, because there were two defects.**

*The bot* now checks the object exists before linking it, and falls back to the
committed generic certificate with a `::warning` when it does not. A generic
image is worse than a bespoke one; a 404 is worse than both.

*The preflight* — the guard I built in wave 11 specifically to catch permission
gaps — probed `contents:read` and **never `contents:write`**, which is the
permission that actually failed. It now performs a real write to `bot-cards` and
deletes the probe file afterwards.

**The control proved it.** A new suite check drives the bot with a stub whose
contents lookup fails and asserts the comment does not reference the missing
object. Reverting the bot to link unconditionally turns it red. 82 checks now.

**The lesson.** Wave 11 asserted every permission the pipeline *reads* and one
it writes, and I recorded it as closing the offline gap. It closed most of it.
The write that mattered was the one I did not think to probe — and the failure
mode I had designed in (non-fatal upload) is exactly what hid it. **A guard
built from your own model of the system inherits the blind spots of that model.
Only running the real thing found this.**

**Still open.** The App very likely lacks `contents: write` on this repo. The
preflight will now say so on its next run; granting it is a change to the App's
permissions that a human has to accept.

---

## Wave 16 — the probe confirmed the diagnosis

**Ran the new preflight against the live App.** It reports exactly what wave 15
predicted from the 404 alone:

    ok    pull_requests:read     /stamp resolves the head sha and the author
    ok    checks:read            every gate reads Stamp, Seal and Expedite
    ok    issues:read            the bot finds its own comment by marker
    ok    members:read           /stamp and /seal check who is asking
    ok    actions:read           /stamp re-runs the held CI run
    ok    contents:read          the bot-cards branch hosts generated certificates
    FAIL  contents:write         certificate images CANNOT be uploaded
    ok    checks:write           /stamp, /seal and /expedite mint their marks

So the certification App has every permission the pipeline needs **except**
`contents: write`, and that single gap is exactly what made #843's certificate
ship a broken image. Nothing else is affected: `/stamp`, `/seal` and `/expedite`
all mint their marks, because `checks:write` is present.

**What this closes.** The defect is understood, the code no longer posts a
broken link, and the guard that missed it now catches it — verified by running
it rather than by reasoning about it. Everything in this record that can be
fixed in code has been.

**What only a human can do.** Granting `contents: write` to the certification
App is a change to the App's repository permissions, accepted in the GitHub UI.
Until then the certificate falls back to the generic image and the preflight
stays red — deliberately, because a red check that names a real missing grant is
the correct state, not something to paper over.

---

## Wave 17 — the fix verified in production, and where this stops

**#846 merged** at 22:47:51Z as `1415fc5e14`, and the fix is confirmed against
the real bot rather than a stub:

    src="https://raw.githubusercontent.com/.../main/docs/diagrams/states/certificate-merged.png"
    HTTP/2 200

The certificate on #846 links the committed generic image and **resolves**,
where #843's linked a generated object that was never uploaded and returned 404.
The bot also emitted the warning it was given: *"Certificate image was not
uploaded ... The App likely lacks contents:write."* Correct behaviour, correctly
explained, in production.

## Where this stops, and why

**The goal is met for everything reachable in code.** Every gate, command and
bot path in the certification pipeline has at least one check and at least one
negative control that has been *watched to fail*:

| Layer | Proven by |
|---|---|
| shell/Python logic — 82 checks | 20+ sabotages |
| ci.yml decisions, state machine, permissions, authority | 15 sabotages |
| Rust logic — 21 tests | 3 sabotages |
| live App authority — 8 permissions | an invalid-token control, and one real gap it caught |
| suite wiring | 2 sabotages, guarded from a different required job |

**Six of the seventeen findings were defects in the verification, not the
pipeline** — hollow controls that passed with the guard deleted, sabotages that
were inert, harness bugs that read as code bugs. Those are the entries worth
re-reading, because they are the failure mode that survives a green suite.

**One real production defect was found**: #843's certificate shipped a broken
image, caused by a permission gap the preflight had not thought to probe. Fixed,
with a control, and the probe now covers it.

**What is still open, and cannot be closed from here.** The certification App
lacks `contents: write`. The preflight reports it, the bot degrades safely
around it, and granting it is a change to the App's repository permissions that
a person accepts in the GitHub UI. Until then certificates carry the generic
image.

**What can never be closed.** The suite runs offline and the probe runs against
one repo. Neither can catch GitHub changing an endpoint's semantics underneath
us. That is a limit, not an oversight, and it is written here so nobody later
mistakes this record for a claim of completeness.

---

## Wave 18 — a suite that reported for months and gated nothing (#810)

**Found.** Not by sweeping for a mechanism, but by reading the open issues:
#810 had already written the defect down. `Site unit tests` ran on every pull
request and on every merge-queue entry, reported green, and blocked neither the
merge nor the deploy. Two independent reasons, and fixing either alone would
have left the other:

| | before | after |
|---|---|---|
| required contexts on `main` | 19, none of them the site suite | 20 |
| `deploy` needs | `build` | `[build, unit]` |

The issue notes what this cost: a latching-state regression (#805) reached
`main` and was only caught later, in the window when the suite could not import
a `.svelte.js` rune module at all.

**Verified.** Three consecutive `site.yml` runs on three different PR branches
each show `Site unit tests: success` alongside `Deploy to avarok via rsync:
skipped` — the suite was demonstrably running and demonstrably not consulted.

**Fixed.** `deploy` now needs `unit` as well as `build`, and `Site unit tests`
was added to branch protection through the additive contexts endpoint rather
than a full `PATCH`, so the other nineteen contexts and every unrelated setting
could not be clobbered by a malformed payload. Diffed before and after: one
context added, none removed, the rest of the protection object identical.

**Proved, and proved the proof.** `assert-site-tests-gate.py` was written and
run *before* the fix, against the unfixed tree, where it refused. It is hosted
in `certification-selftest.sh` — a required context — rather than in the site's
own lane, because a suite cannot be relied on to notice it has been unwired
from itself. Four controls, each pinning the message and not just the exit code
(three distinct defects leave through the same `fail()`, so an exit code alone
distinguishes none of them): drop `unit` from `deploy`'s needs, give `unit` an
`if:`, give `pull_request` a `paths:` filter, delete `unit` outright.

**The third control failed on first run, and was right to.** YAML 1.1 parses a
bare `on:` key as the boolean `True`, so `d["on"]` raised `KeyError` and the
sabotage never applied — the guard then passed, and the control went red for
exactly the right reason. This is the wave 5 failure recurring in a new
disguise: a sabotage that does not land makes green mean nothing. It was caught
here only because the control asserted the guard *must* fail; had it been
written the usual way round, an inert edit would have read as a passing test.

**Why the second assertion exists.** Requiring a context is only safe if the
job reports on *every* PR — a job held behind an `if:` is never created for the
PRs it skips, and GitHub waits on it forever. Five workflows in this repo carry
that scar. So the guard also refuses if `unit` grows an `if:` or a `needs:`, or
if `site.yml`'s `pull_request` trigger grows a `paths:` filter. The fix and the
thing that makes the fix dangerous are pinned in the same file.

**Also this wave.** Closed #839 as a duplicate of #835 — my own issue, filed a
day after an existing one describing the same C=2 bimodality. The measurements
were moved onto #835 before closing, so consolidating cost no evidence.

**What is still open.** The branch-protection half is not expressible in the
tree, so no committed file can guard it; if someone removes the context, only
this record and the API say it was ever there.

---

## Wave 19 — the same defect, one workflow over: the guard's self-test was required, the guard was not

**Found by propagating, not by luck.** Wave 18 fixed one instance; step 6 of a
wave is to ask where else the *mechanism* lives. Sweeping every workflow for
jobs that run on pull requests but are neither a required context nor depended
upon by another job returned ten, of which one was the same defect with a
sharper edge:

| job | reports as | required? |
|---|---|---|
| `self-test` | Merge-ancestry guard self-test | **yes** |
| `guard` | PR shares history with its base | **no** |

The test that proves the guard *can* fail was mandatory. The guard's verdict on
your actual branch was advisory. A PR of the #452 class — an orphan branch whose
root commit is a whole-tree snapshot, which GitHub reports as `MERGEABLE`
because it computes mergeability over trees rather than ancestry, and whose
squash-merge silently reverts everything before it — would have gone red on
`guard` and merged regardless.

**Why it was advisory, and why that was not fixable by simply requiring it.**
`guard` carried `if: github.event_name == 'pull_request'`, because it read
`github.event.pull_request.*`. A merge-queue entry has no such payload, so the
job would never be created there, the required context would never appear, and
every queue entry would block forever. The `if:` was load-bearing. Requiring the
context without removing it would have converted a silent gap into an outage.

**Fixed** by resolving the pair per event: `pull_request` from the payload
(base re-resolved from `origin/<base.ref>`, since `base.sha` is the tip at PR
*creation* and goes stale), `merge_group` from `merge_group.base_sha`/`head_sha`,
and `workflow_dispatch` reporting success with a notice — it carries neither
pair, and a red there would assert "unrelated histories", which is a different
claim than "nothing to compare". In the queue the answer is trivially yes; that
foregone report is the price of the verdict being enforceable on the events
where it is not foregone, and the comment in the workflow says so.

**Proved twice, at two levels.** The workflow-level guard from wave 18 was
generalised from one hard-coded site check into a declarative table
(`assert-gates-are-wired.py`) rather than copied — the second instance was
found *by* generalising it, since the new entry refused on an unmodified tree.
Seven controls now, each pinning the message and not just the exit code:
dropping `unit` from `deploy`'s needs, an `if:` on the site suite, a `paths:`
filter, deleting the suite, restoring the ancestry guard's `if:`, dropping the
`merge_group` trigger, and **renaming a required job** — the quietest failure of
the set, since the job still runs and still passes while reporting under a name
branch protection is not waiting for.

Separately, the guard's own `run:` block was extracted and driven through all
three events against real commits, then against a purpose-built orphan: the
`merge_group` path exits 1 with `NO MERGE BASE`. Without that last step the
three green runs would only have shown the path executes, not that it can
still refuse.

**What is still open.** Requiring `PR shares history with its base` must wait
until this change is on `main` — the queue leg does not exist until then, and
requiring a context before the job that produces it can deadlock the queue.
Sequencing, not oversight; it is the last step of this wave and is recorded
here so that an unrequired context later is read as a regression rather than
the intended state.

---

## Wave 20 — the stamp that stamped nothing

**Found by using the pipeline, not by reading it.** `/stamp` on #847 minted a
green `Stamp` check in 39 seconds, and the benchmark gate stayed red saying
*"Comment /stamp to release certification"* — which had just been done. The
handler's own comment says it: **"★ A stamp that does not RE-RUN anything is a
stamp that does nothing."** It was doing nothing.

Three defects, stacked, each hiding the next:

| # | defect | why it was invisible |
|---|---|---|
| 1 | the App installation lacks `actions: write` | no probe covered it |
| 2 | the preflight probed `actions:**read**` | its justification read "/stamp re-runs the held CI run" — a *write* |
| 3 | the re-run's stderr went to `/dev/null` | the reason was destroyed at the moment it was produced |

The third is what cost the time. `gh api -X POST .../rerun >/dev/null 2>&1 ||
echo "could not re-run"` reports that something failed and discards what. The
diagnosis needed a token comparison — my own PAT re-runs the same run with
`rc=0` — to establish what the log had thrown away.

**This is #843 one endpoint family over.** There, `contents:write` was absent,
the certificate upload is non-fatal by design, and the preflight probed
`contents:read`. Same shape, same silence, and the lesson had not propagated
from `contents` to `actions`.

**Fixed.** The re-run's stderr is captured and surfaced three ways: an `::error`
annotation, a `> [!WARNING]` block appended to the PR comment carrying the
verbatim API response, and a non-zero exit so the command job is red. The mark
is still recorded first — a stamp survives new commits and is worth keeping even
when the re-run fails — but the command no longer reports success for work it
did not do. The preflight's three hand-rolled write probes became one
`probe_write` helper, and `actions:write` joined them, probed by enabling an
already-enabled workflow: idempotent, needs the permission, and this very
workflow is provably enabled because it is the one running.

**Proved against history, which is the only control that counts here.**
`assert-preflight-covers-writes.py` derives, from the workflows themselves,
every write call that *swallows its own failure*, and requires a `probe_write`
for each. Run against the tree as it stood at `HEAD` — before any of this
wave's edits — it refuses on **both** `actions:write` and `contents:write`.
It would have caught #843 and #847 alike, on the day each was written.

Its scope is deliberately narrow and the file says why: a write that fails
*loudly* under `set -e` announces itself the first time it breaks, and demanding
a probe for it would mean inventing noisy probes (posting and deleting comments,
creating labels) for failures that are already self-announcing. Only suppressed
writes are pinned. Widening it would be the busywork the guard exists to
displace.

**Five controls, and the joining one matters.** Losing either `probe_write` is
caught — and the `contents:write` case doubles as proof that the guard rejoins
backslash continuations, since that PUT spans five lines with its `||` on the
last. Matched line-by-line it would read as *unsuppressed*, and the guard would
have passed by construction. Three more drive the extracted `/stamp` step with a
stubbed `gh` whose re-run returns 403: the step must exit non-zero, the comment
must say the lane was not released, and it must carry the API's own words.
Reverting the fix turns all three red; that was checked, not assumed.

**What is still open, and needs a person.** The App must be granted
`actions: write` (and still `contents: write`) and the installation must accept
both. Until then `/stamp` records a mark, posts the warning, and goes red — the
honest behaviour, but the lane still needs a manual re-run or any new commit.

---

## Wave 21 — a stray `set -e` had been silently truncating this suite

**The wave's own tooling was the first defect it found.** The vhost controls
added below reported nothing at all: the suite stopped after 97 checks with exit
1 and no summary line. Cause: the `/stamp` control added in wave 20 ended with
`set -e` to "restore" state that never existed — this suite runs `set -uo
pipefail`, with errexit deliberately **off**, because every control in it runs a
command that is *supposed* to fail. The first such control after that line
killed the run.

For one wave, `certification-selftest.sh` was a suite that stopped a quarter of
the way through and looked like an ordinary failure. Same non-zero status, no
summary, thirty checks silently not run — including every control this wave was
adding. It is the exact defect this whole record is about, committed by the file
whose job is to catch it.

**Fixed** by removing the stray `set -e`, and then by making the failure
impossible to miss: the EXIT trap now reports `SUITE TRUNCATED: exited after N
checks, before the summary` unless the summary was reached. Reintroducing the
`set -e` prints it after 97 checks — verified, not assumed.

**Then the wave proper.** Three public vhosts, one nginx rule, three separate
incidents, each found by hand after the fact:

| vhost | what was lost | how |
|---|---|---|
| docs | every HTML doc served with no security headers at all | `Cache-Control` inside `location ~* \.html$` |
| site | `Alt-Svc` on every proxied response; the dotfile refusal went out bare | headers inside `location /` |
| site | `Referrer-Policy`, which the other two sent | drift nobody was watching for |

`add_header` does not accumulate across contexts: a location declaring *any*
`add_header` discards every one inherited from the server block. All three were
fixed reactively. Nothing stopped a fourth.

`assert-vhost-headers.py` pins four things: the core set is declared at *server*
level in every vhost; no `add_header` appears inside any location at all
(refusing the construct outright, rather than reasoning about which uses would
be safe — per-path values belong in a `map`, which is how blog and docs compute
Cache-Control today); the core set is identical across the three; and
`X-XSS-Protection`, if present, is exactly `"0"`.

**One live defect fixed.** `atlasinference.io` was serving
`x-xss-protection: 1; mode=block` — confirmed by request, and absent from the
other two. The header is deprecated, every current browser has removed the
auditor it controls, and OWASP's Secure Headers Project recommends `"0"`
because the legacy filter is itself exploitable. Set to `"0"` rather than
deleted: with no header at all a legacy browser falls back to its default, which
is the filter **on**.

**Controls.** Four, each pinning the message. The docs incident is
reconstructed rather than replayed, and the file says so — that vhost has a
single commit, so its pre-fix text is not in the tree and there was nothing
honest to replay against. The sabotage harness asserts the edit actually
changed the file before trusting the result, which caught one control whose
escaping produced a Python `SyntaxError` instead of an edit.

**Not pinned, deliberately: HSTS.** None of the three sends
`Strict-Transport-Security`. That is a real gap, but a browser caches HSTS for
`max-age`, and choosing that value — and whether to preload — is a policy
decision for a person, not a default a linter installs. Raised, not taken.

**Two areas dug and found clean**, recorded so they are not re-dug: 115
`#[ignore]`d Rust tests are all documented GPU tests and the `--ignored` lane in
`kernel-compile.yml` genuinely executes 7 of them (`running 6 tests` /
`running 1 test` in the log, not a hollow `ok`); and there are no skipped JS
tests and no tautological assertions anywhere in the tree.

---

## Wave 22 — one dig found nothing, the next found a required check that is structurally blind

**First dig: the Rust half of the gate. Nothing found, and that is the result.**
The hypothesis was that `scoring.rs` (which decides whether a measured value
passes its bound) and `record_path.rs` had no tests — a first grep for
`scoring::` in the test files returned zero. That grep was wrong: both modules
are re-exported through `record.rs`, so tests call `check_record` and `date_of`
unprefixed. `scoring::check_record` has three named tests, `date_of` has one,
and every gate module but `check_fmt.rs` (27 lines, no public functions) is
exercised. Recorded here so it is not re-dug, and because a hypothesis that
survives one grep and dies on the second is worth writing down as a warning
about the first grep.

**Second dig: the security alert every push in this session printed.**
`GHSA-rhfx-m35p-ff5j` — `lru < 0.16.3`, `IterMut` violating Stacked Borrows —
was open against `main`'s lockfile.

The interesting part is not the crate. It is that **`cargo deny` is a required
context, it runs `check advisories`, it was green, and the advisory was open on
the same lockfile.** Not a misconfiguration: `cargo deny` resolves against
RustSec, Dependabot resolves against the GitHub Advisory Database, and this
advisory carries a GHSA id and **no RUSTSEC id**. No configuration of
`cargo deny` can see it. A green `cargo deny` is therefore not the claim
"no known advisory affects this lockfile", which is how a required check named
`cargo deny` reads.

**Exposure, established rather than assumed.** `lru` is not a direct dependency
of anything in the workspace; it enters twice-removed through `ratatui 0.29`.
Its only use there is `type Cache = LruCache<(Rect, Layout), (Segments,
Spacers)>` in `layout/layout.rs`, and the methods called on it are `cap()`,
`get()` and `resize()`. That file contains **zero** occurrences of `iter_mut`,
and `tui-textarea` — the other path to `ratatui` — does not depend on `lru` at
all. The advisory is about `lru::IterMut` specifically, so the vulnerable
iterator is never constructed in this graph. The alert is dismissed as
`not_used` with that reasoning and the condition for re-opening it (a ratatui
bump that iterates the cache mutably, or `lru` becoming direct).

**No guard was added, deliberately.** The honest options were a fragile one —
re-checking a vendored dependency's source in CI, which breaks the moment
ratatui is bumped and would fail for the wrong reason — or a CI step reading
the Dependabot alerts API, which needs a token permission this repo's
`GITHUB_TOKEN` does not carry, and there are already two permission grants
waiting on a person. Adding a third that sits red would be worse than the gap.
What was added instead is the blind spot written into `deny.toml` itself, beside
the `db-urls` line that causes it, so the next reader of that file learns it
there rather than by discovering an open alert behind a green check.

**Open, and a decision for a person:** whether to gate on Dependabot alerts at
all. Today they report on the default branch and block nothing.

---

## Wave 23 — two more capabilities that fail silently, found by sweeping for suppressed failure

**The sweep.** Every workflow step line ending in `|| true`, and every
`continue-on-error`. Twelve `|| true` lines; ten are benign (optional file
copies, diagnostic `ls`, apt fallbacks that nothing depends on). Two were not.

**Defect 1 — a missing render tool swallowed the whole certificate.**
`certification-bot.yml` installs `librsvg2-bin` and `segno` with `|| true`, then
runs `rsvg-convert` under `set -euo pipefail` with **no check that the install
worked**. If apt failed, the step died *before* the comment POST: a merged PR
got no certificate **and no comment at all**. The design three lines below
already states the rule it was breaking — a missing `contents:write` "cannot
swallow the certificate itself", which is exactly what a missing renderer did.

Fixed with a `render_ok` guard: absent tools now emit a warning, skip the
render, and fall through to the generic image that the fallback already exists
to serve. Degrade the picture, never the certificate.

**The control had to manufacture the absence.** This host has both tools
installed, so "the tool happens to be missing" would be the wave-2 mistake — a
control that only holds on one machine. It runs the extracted step against a
PATH containing the stubs plus a symlink farm of the commands the step actually
uses, resolved with `command -v` at runtime so it adapts to wherever they live,
and it asserts the farm did not leak `rsvg-convert` before trusting the result.
Reverting the guard turns both new controls red.

**Defect 2 — `/seal` minted its mark and never refreshed the job that reads it.**
This is wave 20's defect in the other verb, and it was live on this very PR:
`Seal` went green in 33 s while `seal status` stayed red. `ci.yml` has no
`check_run` trigger, so minting a check run does not re-evaluate anything, and
the re-run block was `if [ "$VERB" = "/stamp" ]`. The only way to refresh
`seal status` was to push a commit — **which is the one thing that voids a
seal**. The handler's own maxim, written for stamps, was never applied to the
verb next to it.

Re-running is safe for a seal: it adds no commit, and a seal is voided by
commits, not by CI runs.

**One existing control went red, correctly.** Rewording the warning from "the
held lane was not released" to "CI was not re-run" — necessary, since it now
covers both verbs — broke a control that pinned the old wording. That is a
message-pinned assertion doing its job: it noticed the text it depends on
changed, which is the behaviour that distinguishes it from asserting an exit
code six defects share.

**Not a defect, checked and recorded:** `coverage.yml`'s and `coderag.yml`'s
`continue-on-error: true` are deliberate and documented at their call sites, and
`ci.yml:1062` records that a previous one was made enforcing.
