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
