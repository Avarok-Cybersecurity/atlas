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
