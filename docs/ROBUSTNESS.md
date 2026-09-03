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
