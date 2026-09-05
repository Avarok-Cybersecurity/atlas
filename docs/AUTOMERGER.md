# AutoMerger — orchestration trace

An append-only record, one entry per run. Unlike `ROBUSTNESS.md` this is not
prose: each entry is a **temporal interaction graph** — typed events with
wall-clock and elapsed time — because when one campaign certifies nine PRs and
fails, credit assignment is the problem, and a graph of who decided what, on
which evidence, at what cost, is the substrate that makes it tractable.

**Event types** (one per line, in time order):

| type | meaning |
|---|---|
| `spawn` | a supervision or worker agent was started (say which tier and why) |
| `delegate` | work handed to an agent (the task, verbatim) |
| `communicate` | a message between agents, or an escalation flag |
| `tool` | a consequential external action (close, comment, push, stamp) |
| `return` | an agent's result folded back in |
| `aggregate` | verdicts combined into a decision |
| `stop` | a stop condition fired — name which of the four |

**Every entry also records:** each verdict *with the artifact proving it*; each
grouping and why; **cost as a first-class term** — campaigns (GPU-hours) and
runner-minutes, measured, not estimated; every mistake the run made and how it
was caught; and a closing **CITED** or **AMENDED** line (see the skill's
"Living rulebook"). An entry missing that line is a defect in the run.

Format is one fenced block per wave so the events stay greppable:

```text
2026-09-04T00:00:00Z +0m00s  tool       <what> — <evidence>
```

---

## Run 0 — 2026-09-04 — building the machinery (this file, the skill, the gate)

The run that created AutoMerger, recorded in its own format. Costs are real:
the hosted-runner pool was down the whole time (0 executions across 40+ queued
runs), so every published run this day cost queue slots, not runner-minutes.

```text
2026-09-04 +0m  aggregate  backlog triage complete: 19 PRs verdicted, 12 already on main or superseded — each close cites a named artifact, none cites a cherry-pick result
2026-09-04 +0m  tool       composed stack/certified-2026-09-04: 9 PRs, 16 commits, 51 files; offline gates EXIT=0, 5503 tests pass
2026-09-04 +1h  spawn      fable ×3 (adversarial lens: vacuity / contradiction / unverifiability) over 59 mined lessons
2026-09-04 +2h  return     30 rules survived, each with EVIDENCE and CHECK; 29 dropped with reasons recorded in the skill
2026-09-04 +2h  aggregate  rulebook embedded in .claude/skills/automerger/SKILL.md — the skill edits this list as it learns (cite-or-amend)
2026-09-04 +3h  tool       stack-map SVG template + render-stack-map.py committed (90a6055a8e); footer-crop and dead-space defects found by LOOKING at the PNG, not by any assertion
2026-09-04 +4h  tool       ci.yml: is_stack_layer classify step + expensive-lane deferral committed (a431c8b2c0)
2026-09-04 +4h  aggregate  negative controls: fail-safe flipped → 4 rows red; alias branch removed → red; matrix-summary branch removed → red; selftest 124/124 green after restore
2026-09-04 +4h  stop       stop condition "campaign-owing work is ready" does not apply — this branch touches no PERF_PATHS; proceeding to publish
2026-09-04 +5h  spawn      supervision negative controls: Monitor (haiku), Triage (sonnet), Adversary (fable), each fed input that MUST produce the failing verdict
2026-09-04 +5h  return     Monitor: flagged no-progress (5 tick intervals since last wave entry) and repeat (same blocker 4 consecutive ticks) — PASS
2026-09-04 +5h  return     Triage: returned `continue` past a ledger that already held the window measurement AND the discriminating intervention — FAIL; the retracted-escalation cautionary tale biased it toward under-escalation, the polling-a-dead-queue failure itself
2026-09-04 +5h  aggregate  doctrine amended, not the agent: `external` is MANDATORY when both halves are already in the ledger; retest on the same ledger → `external`, citing both — PASS
2026-09-04 +5h  return     Adversary: BLOCKED the sabotaged stack — named the cherry-pick-emptiness trap on the planted #708 verdict, and surfaced two defects the control did not plant (two perf constituents = non-attributable campaign; no mergeable-tree assertion) — PASS
2026-09-04 +6h  tool       gate tested against the LIVE #655→#651→#650 chain — all three layers classified "lower": the base!=main clause misclassifies the top of a native stack, i.e. the landing tree would merge UNCERTIFIED
2026-09-04 +6h  aggregate  rule corrected to the plan's own words — a layer is a PR with something stacked above it, nothing else; fork PRs excluded (branch-name collision on --base); both holes given selftest controls proven able to fail; live retest: exactly one aggregation PR per chain
```

**Cost:** 0 campaigns, 0 hosted runner-minutes (pool down; all checks executed
locally). **Mistakes made and caught:** two edit anchors missed on first
attempt (assert-before-write caught both at zero cost); a selftest insertion
orphaned an `else` and truncated the suite at 32 checks — caught by the suite's
own `REACHED_SUMMARY` trap, which exists because of rule R-13 (checks appended
after a suite's exit can never fail).

**AMENDED:** the layer rule is "something stacked above", *only* — the
`base != main` clause certified nothing on a real 3-layer chain and would have
let the top of a native stack merge uncertified. Evidence: live classification
of #650/#651/#655. Check: the selftest rows "nothing stacked above it
certifies — whatever its base" and the fork-PR control.

**AMENDED:** Triage escalation criterion added to the skill's supervision
section — `external` requires a window measurement plus a discriminating
intervention, and is *mandatory* once both are in the ledger. Evidence: the
first negative control caught Triage returning `continue` past a confirmed
stall whose discriminating measurement it had already been handed. Check:
re-run the control ledger; the verdict must be `external` citing both halves.

**CITED:** R-13 (suite-truncation trap → the `REACHED_SUMMARY` guard did its
job); anchor rule (assert every anchor exactly-once before writing → 2 misses
cost nothing); fail-safe direction (every way the stack lookup can break lands
on "certify", proven by sabotage, not asserted in a comment).

---

## Run 1 — 2026-09-04 — first live run: pool recovery, super-stack published

```text
2026-09-04T21:27Z +0m   tool       runner-pool discriminator: job log stamped 21:26 EXECUTED on a hosted runner — outage over; old 3h45m queue drained
2026-09-04T21:30Z +3m   tool       #875 red = STAMPED:false (held-by-design, not broken); stamped; cancelled CLA re-run
2026-09-04T21:31Z +4m   aggregate  #845 red = same held signature — benign, no action; watcher event closed
2026-09-04T21:33Z +6m   tool       pushed held branches: stack/certified-2026-09-04, stack/ci-cost-controls, fix/harvest-feedback-loop (#873) — all behind main by 0
2026-09-04T21:35Z +8m   aggregate  constituent mapping DERIVED from branch content, not memory: 8 PRs (#869 #742 #745 #837 #838 #844 #845 #777); compaction-carried "#799/#842" do not exist in this repo — R-verify-before-quote
2026-09-04T21:36Z +9m   spawn      Adversary (fable) on the super-stack plan, before publication (§E blocking gate)
2026-09-04T21:38Z +11m  tool       ci-cost-controls found to CONFLICT with #876 (6 hunks, same mechanism) — folded into #876 instead of fanning out; found + fixed its defect in transit: builds_binaries skip had NO acceptance branch, dry-run summary went red on exactly the diffs the feature serves; selftest rows added, control proven able to fail
2026-09-04T21:40Z +13m  return     Adversary: PUBLISH conditional on 7 items (2 GPU smokes, SHA pin, merge-base recheck at approval, BFCL draw fingerprint, lockfile audit, attribution map)
2026-09-04T21:42Z +15m  tool       condition 6 executed: getrandom 0.4.3 present, toml 0.8.23/ratatui 0.29.0 identical to main — cited by version line (first read misjudged multi-version pins; corrected before publishing)
2026-09-04T21:44Z +17m  tool       #877 opened (aggregation PR, head pinned 44130974a0), stamped; plan + 7 conditions in body; attribution map written up front
2026-09-04T21:46Z +19m  communicate self-marking table comment posted on all 8 constituents with merge-order rationale and freeze request
2026-09-04T21:47Z +20m  stop       stop condition 1: stack published and campaign owed — HALTED at "ready to certify", awaiting explicit approval (conditions 1-2, 4-5 run at approval time)
```

**Cost:** 0 campaigns spent. 1 stamp on #875, 1 on #877; 9 comments (8 constituent tables + 1 stamp), each a deliberate protocol action, not triage churn.

**CITED:** R-named-artifact (constituent mapping derived from the branch, phantom #799/#842 rejected); R-stack-dont-fan-out (ci-cost-controls folded into #876 on a 6-hunk conflict); R-comment-then-close etiquette pre-staged in every constituent comment (recourse offer included); R-mid-campaign-push trap → head SHA pinned and freeze requested in writing.

---

## Run 2 — 2026-09-04 — sieves, the native stack, and two exclusions

```text
2026-09-04T21:49Z +0m   communicate user directive: three sieves (security/integrity/goal) before any PR is included; failures labelled + commented
2026-09-04T21:50Z +1m   tool       #875/#877 CLA reds = CANCELLED by comment-race (fix waits in #868, queued); both re-run
2026-09-04T21:51Z +2m   tool       'Certification commands' red on main root-caused: /stamp raced #877's first CI run, rerun API 403 'already running' treated as fatal; fixed with two-path handling (undecided stamp job → no re-run needed; decided → bounded wait then re-run); 2 selftest rows, pre-fix handler fails both
2026-09-04T21:52Z +3m   spawn      4 sieve agents (fable), one per cluster, all three sieves each, read-only
2026-09-04T21:55Z +6m   tool       labels created: sieve:security/integrity/goal/cleared (422s were >100-char descriptions)
2026-09-04T21:58Z +9m   communicate user: #877 is the collapse model, not a native stack — fix and research the Stacks API
2026-09-04T22:00Z +11m  tool       reorder rebase: full per-cluster grouping CONFLICTS (gdn/dflash interleave in ssm_gdn_b.rs); minimal reorder (one commit moved) yields TREE-IDENTICAL head b0a5664df3 — the 5503-test validation carries
2026-09-04T22:02Z +13m  tool       own rulebook violated then vindicated: `cherry-pick -q` usage error swallowed by a pipe (R-capture-rc); redone with captured rc
2026-09-04T22:04Z +15m  tool       8 layers cut and pushed; #877 retargeted to L7 via REST (gh pr edit silently failed AGAIN — R-gh-edit-lies verified twice in one hour)
2026-09-04T22:06Z +17m  return     sieve verdicts: INCLUDE #869 #777 #745 #845 #837 #838; EXCLUDE #742 (EOS suppression wholly untested), #844 (fp8 twin + slot-order fix: receipts, no in-tree tests) — every verdict artifact-named
2026-09-04T22:08Z +19m  tool       labels applied via REST after gh pr edit --add-label ALSO failed silently; cure-comments on #742/#844 with re-inclusion offers
2026-09-04T22:10Z +21m  communicate user found docs/optimizing-ci-for-stacked-pull-requests: native payload stack.{position,size}; gate upgraded to prefer it (4 selftest rows, flipped-comparison control red)
2026-09-04T22:12Z +23m  tool       STACK #885 REGISTERED via POST /repos/{repo}/stacks — 8 PRs bottom-up #884..#877, open:true; the icon the user asked for
2026-09-04T22:13Z +24m  aggregate  wrong-branch near-miss: gate edits attempted while HEAD was wip/stack-reorder; every anchor assert refused against the wrong tree — zero damage (R-assert-anchors pays again)
```

**Cost:** 0 campaigns. 7 layer PRs opened (cheap lanes only — the gate defers their expensive lanes), 2 cure-comments, 8 labels.

**AMENDED:** sieve doctrine (§B2) written into the skill: three sieves, artifact-named verdicts, label + cure comment, REST for labels. Native-stack mechanics written into "One certification per stack": REST registration, `stack.position/size` payload preferred by the gate, no automatic CI dedup, and the landing rule — native stacks merge bottom-up (one merge_group certification PER LAYER), so a certified stack lands by collapsing the top to one queue entry.

**CITED:** R-capture-rc (violated, caught in-wave by its own usage error); R-gh-edit-lies (twice: --base and --add-label both silently no-oped; REST + read-back both times); R-assert-anchors (refused edits against the wrong checked-out branch).

**Sieve consequence:** #742 and #844 are excluded from certification until their named tests exist. The stack layers containing them (#883, #879) stay in the chain for review, but the campaign will not be requested until either (a) the tests are written and the sieve re-run clears them, or (b) the stack is recomposed without them.

---

## Run 3 — 2026-09-04 — cures verified, our own leftovers eaten

```text
2026-09-04T22:20Z  return     cure agent: 3 commits, each with an OBSERVED negative control; found that `cargo test -p spark-server --lib` matches 0 scheduler tests (they live in the bin target) — the lib filter silently runs nothing
2026-09-04T22:22Z  tool       cures re-run first-hand: 5/5 pass in --bins with 2352 filtered (the suite really ran); pushed; new pin 82552fe34d; stack #885 tracked the force-push, chain intact
2026-09-04T22:24Z  spawn      re-sieve of #742/#844 cures on SONNET (tier moved by user directive)
2026-09-04T22:26Z  tool       NO AUTOMERGE veto label created + doctrine (absolute, action-time, includes drafts, DO NOT MERGE — YET, and title-carried DO NOT MERGE)
2026-09-04T22:30Z  aggregate  scanner surfaced OUR OWN forgot-to-delete pattern: #870/#874 are the pre-consolidation intermediate stacks, still open
2026-09-04T22:32Z  tool       #870 proven contained (merge-tree returns the stack's own tree); #874 proven contained in the PRE-CURE tree (divergence = the cure commits only); both closed with evidence + recourse offer
2026-09-04T22:33Z  tool       #621 proven byte-identical-contained in QUEUED #868 — close staged for after #868 actually merges, never before
```

**CITED:** R-named-artifact (merge-tree hashes quoted in both closes); R-eat-your-own-dogfood — the forgot-to-delete pattern the skill was born from was recreated by our own consolidation, caught by the scanner the skill mandates; R-test-really-ran (the --lib filter that matches nothing).

---

## Run 3b — 2026-09-04 — sieve gate closed

```text
2026-09-04T22:35Z  return     re-sieve (sonnet): BOTH CURES CLEARED — and it verified rather than accepted: mutation-tested #742's cure (deleting the disjunct fails exactly the in-think test), confirmed the slot-sort extraction verbatim against the call site, reproduced the oracle compile check, confirmed base-not-twin grading
2026-09-04T22:37Z  tool       labels swapped to sieve:cleared on #742/#844 (REST DELETE + POST, read back); cleared-comments carry the residuals honestly
2026-09-04T22:38Z  aggregate  residuals promoted to campaign-prep conditions on #877: the fp8 oracle's GPU verdict is UNOBSERVED (mandatory run at prep); grammar-armed thinking turns take a different disjunct and are uncovered
2026-09-04T22:39Z  stop       stop condition 1 re-armed: all 8 constituents sieve-cleared, stack pinned 82552fe34d, offline-green — the ONLY remaining gate is explicit human approval for the campaign
```

**CITED:** R-verify-dont-accept (the re-sieve mutation-tested the cure instead of reading it); R-residuals-are-conditions (unobserved GPU verdict became a named prep step, not a footnote).

---

## Run 4 — 2026-09-04 — stack pipeline goes two-wide

```text
2026-09-04T22:50Z  aggregate  #693 proven hunk-identical to stack commit d8778c441d — the scanner tests against main and cannot see stack carriage; commented, labelled, ninth constituent of #885's accounting
2026-09-04T22:52Z  spawn      sieve (sonnet) on #705/#781 for the next stack
2026-09-04T22:58Z  return     both INCLUDE with live-tree verification: #705's unload path drop-safe (registry.rs:143 sole caller), #781's skip reproduces documented zero-contribution EP semantics (forward_prefill_routed.rs:135-143 + loaders_moe.rs:62-65), not an error swallow
2026-09-04T23:00Z  tool       stack/perf-plumbing composed OFFLINE in .wt-perf (2 PRs, authorship preserved, clean cherry-picks); cargo check EXIT=0; atlas-core 128/128; spark-runtime 277/277; LoC cap ok — NOT pushed (CI conservation during runner starvation)
2026-09-04T23:01Z  aggregate  pipeline now two-wide: #885 fully gated awaiting approval; perf-plumbing composed awaiting adversary + runner recovery
```

**CITED:** R-scanner-blind-to-stacks (a carried PR reads "applies" vs main — check candidate PRs against pending stack trees, not only main); R-conserve-starved-CI (compose and gate offline, publish when the pool can absorb it).

---

## Run 4b — 2026-09-04 — perf-plumbing published

```text
2026-09-04T23:15Z  return     Adversary (fable): PUBLISH, 4 conditions — patch-id-verified the composition equals the sieved diffs; caught that the campaign's single-node config is exactly the config #781 has never executed (its evidence is two-node EP=2) → blocking single-node MoE smoke; EP=2 claim stays uncertified by the campaign
2026-09-04T23:17Z  tool       #891 opened (stack/perf-plumbing, 2 PRs, disjoint phase signatures as the attribution map); constituents #705/#781 given self-marking table comments with the freeze request
2026-09-04T23:18Z  stop       stop condition 1 re-armed for stack 2: published, campaign owed, awaiting approval — now TWO stacks at the approval door (#877: 9 constituents; #891: 2)
```

**CITED:** R-adversary-earns-its-seat (third consecutive review that found a condition no checklist held: the config-coverage gap between two-node evidence and a single-node campaign).

---

## Run 5 — 2026-09-04/05 — the docs pair rides the owed campaign

```text
2026-09-04T23:30Z  return     sieve (sonnet) on #667/#669: both INCLUDE — and it caught the premise error: #669 is bench-harness CODE in crates/, #667 is rustdoc in crates/; a "docs stack" would owe a full campaign by path rule
2026-09-04T23:32Z  aggregate  economics decision: fold both into #891, whose campaign is already owed — marginal certification cost zero; a standalone pair would have spent ~5 GPU-h on comments and one bench cell
2026-09-04T23:34Z  tool       cherry-picked (authorship preserved), offline gates re-green (cargo check EXIT=0, atlas-plugin 900/900), pushed; #891 pin now 85b190322f, 4 constituents
2026-09-04T23:35Z  communicate delta re-clearance requested from the SAME adversary agent (resumed with context) rather than a fresh review — the four answers only need the delta judged
2026-09-04T23:36Z  tool       #833 and #646 proven carried by #875 (merge-tree returns its own tree); commented, labelled, close-on-landing staged
```

**CITED:** R-path-rule-owns-the-economics (a "docs" PR in crates/ owes a campaign no matter what the lines contain — group by owed-campaign, not by content type); R-resume-dont-respawn (the adversary kept its context; a delta needs a delta review).

---

## Run 5b — 2026-09-05 — delta re-clearance

```text
2026-09-05T00:05Z  return     Adversary delta review: RE-CLEARED at 85b190322f — verified both folds by patch-id, not narrative; corrected MY count (#667 = 4 commits, not 5); flagged that #669's new cell may gate in its own PR only because it is a content-independent binary contract; surfaced the 13/14 pre-existing mixed-media red as a provenance note so a red video leg is never misattributed to this stack
2026-09-05T00:07Z  tool       record corrected on #667 and #891 — the wrong count was mine, said so plainly
```

**CITED:** R-report-your-own-errors (the correction names whose mistake it was); R-provenance-beats-blame (a pre-existing red recorded before it can be misread as a regression).

---

## Run 6 — 2026-09-05 — #866 rides, composition frozen

```text
2026-09-05T00:20Z  return     sieve (sonnet) #866: INCLUDE — both defects confirmed LIVE on main (debug_assert! forward.rs:249; unconditional rollback rollback.rs:411), fixes fail-fast, tests pin the real outage numbers, zero overlap with either stack
2026-09-05T00:22Z  aggregate  the phantom "#799/#842 PRs" from compaction resolved: they were ISSUE numbers, carried by PR #866's commit subjects
2026-09-05T00:24Z  tool       folded into #891 (pin 338a6babd7), offline gates re-green (rollback 25/25, vision 1/1); labelled + commented; SECOND delta re-clearance requested from the same adversary; composition DECLARED CLOSED — unbounded folding is scope creep with a pin attached
```

**CITED:** R-verify-before-quote (the phantom numbers finally traced to their artifact); R-close-the-composition (a stack that keeps growing never reaches the approval door).

---

## Run 6b — 2026-09-05 — #891 fully gated

```text
2026-09-05T00:35Z  return     Adversary second delta: RE-CLEARED at 338a6babd7 — disjointness proven by file-set intersection; found the vision tests STRONGER than my summary (4 tests incl. the exact 1024-boundary case); caught that my "1/1" run used the dev profile while the commit's claim requires RELEASE
2026-09-05T00:36Z  tool       discharged in-wave: both vision-bound tests re-run under --release, 2/2 pass — the ensure! is proven in the profile where a debug_assert! would have vanished
2026-09-05T00:37Z  stop       stop condition 1: BOTH stacks fully gated at the approval door — #877 (9 constituents) and #891 (6 constituents); 15 backlog PRs, two campaigns, zero started without approval
```

**CITED:** R-profile-matters (a release-only guarantee tested in dev proves nothing — the adversary's nit was the whole point of the original fix); R-composition-closure (clearance is pinned to a closed set).

---

## Run 7 — 2026-09-05 — the stall diagnosed to its class, and a retraction

```text
2026-09-05T00:50Z  aggregate  RETRACTION: the 23:00 "pool recovering" call was WRONG — queue length fell by supersession churn, not throughput; the class split proves it: every completion since 22:00 is a SELF-HOSTED lane (bot/CLA on avarok-cmd), every hosted run is queued
2026-09-05T00:52Z  tool       githubstatus.com: Actions operational, no incident — the cause is org-side
2026-09-05T00:53Z  aggregate  verdict: org Actions spending-limit exhaustion is the strongest remaining hypothesis (signature: hosted queues indefinitely, no errors, self-hosted unaffected, no incident; morning intervention already ruled out concurrency caps)
2026-09-05T00:54Z  communicate PushNotification sent — only the org owner can check Settings→Billing→Actions; every landing is blocked on it
```

**CITED:** R-queue-length-lies (measure per-class executions, not queue depth — the same rule the Monitor table encodes, violated by me at 23:00 and caught by the class split).

---

## Run 8 — 2026-09-05 — the detached-HEAD incident, and CI's selftest failure

```text
2026-09-05T02:10Z  aggregate  #876's "cargo deny" red = the certification SELFTEST failing 3 rows ON CI while 135/135 locally — my classify() harness greps stdout, but inside Actions GITHUB_OUTPUT points at the step-output file and emit's lines vanish from the pipe
2026-09-05T02:20Z  aggregate  INCIDENT while fixing it: stackA HEAD was silently DETACHED — runs 3-7 trace commits sat on the stack's tree, not the branch; every push since pushed a ref missing them; a `git show HEAD:` against the wrong tree also manufactured a phantom "lost hunk" (the branch's emit was 4-arg all along)
2026-09-05T02:25Z  tool       repair: orphan line held ONLY docs/AUTOMERGER.md (117 lines) — rescued and appended to the branch's trace; wrong-tree edit discarded; back on feat/automerger-skill
2026-09-05T02:30Z  tool       real fix: classify() pins GITHUB_OUTPUT=/dev/stdout; wildcard-branch coverage row added; suite 136/136 bare AND under simulated Actions env; un-pinning under that env reproduces CI's exact 3 failures
2026-09-05T02:32Z  aggregate  AMENDED: assert-branch-before-commit written into the skill's non-negotiables; trust no HEAD-relative read for comparisons
```

**CITED:** R-reproduce-the-runner (the failure was invisible locally until the Actions env was simulated — 2>/dev/null in a harness hides exactly the evidence CI dies on); R-assert-anchors (the wrong-tree edit was refused by anchors ONCE and slipped through a sed the second time — sed has no anchors, prefer asserted replaces).

---

## Run 9 — 2026-09-05T01:50Z — a clock misread, retracted

```text
2026-09-05T01:50Z  aggregate  RETRACTION: the "jobs start then die, newest completion 2h old" verdict was built on an imagined clock — `date -u` reads 01:49; the 01:25 completion was ~20 minutes old, normal for kernel-compile-length jobs. The pool is throttled-slow, nothing more dramatic. Several earlier trace timestamps in runs 4-8 are likewise approximate and should not be used for duration math.
2026-09-05T01:50Z  aggregate  rule: never state a duration without printing the clock beside the timestamp in the same command.
```

**CITED:** R-print-the-clock (a duration claim without a same-command timestamp is a guess wearing units).

---

## Run 10 — 2026-09-05T02:40Z — overnight mode engaged

```text
2026-09-05T02:40Z  communicate user directive: work overnight; hourly waves of 3-4 fable deep-inspection agents (opus 5 once fable limits hit) improving DX, robustness, and atlasctl real-life usage
2026-09-05T02:40Z  tool       atlasctl 0.2.0 found installed; `doctor` on this box already reports two REAL problems (config-dir uid mismatch 1000 vs 996; sparkrun registry redirect to Atlas-Inf/sparkrun-recipes, which Atlas does not control) — the ctl lane starts with live material
2026-09-05T02:40Z  spawn      wave 1: 4 fable agents in isolated worktrees (nightly/w1-{dx,rob,ctl,journey}), local commits only, negative controls mandatory, no pushes into the starved CI pool
```

---

## Run 11 — 2026-09-05T03:00Z — nightly wave 1: journey lane returns

```text
2026-09-05T03:00Z  return     journey lane (fable): 7 doc-fix commits on nightly/w1-journey, docs-only (4 files, +52/-6), tree clean, nothing pushed
2026-09-05T03:02Z  tool       coordinator VERIFIED the three load-bearing claims first-hand rather than accepting them:
                              - `cargo test -p spark-server --lib scheduler` -> "running 0 tests ... ok" (242 real tests live in the bin target) CONFIRMED
                              - PR template's checklist line fails TWICE on Linux: without CUDARC_CUDA_VERSION -> cudarc build.rs:124 nvcc panic; with it, --all-features pulls objc2 -> "only works on Apple platforms" BOTH REPRODUCED
                              - the template genuinely shipped that line (read from origin/main) CONFIRMED
2026-09-05T03:03Z  aggregate  worst finding is not a doc gap but an INVERSE one: CONTRIBUTING told contributors scripts/check.sh was broken when it had been fixed — docs lagging a fix, which erodes trust in every other instruction
```

**CITED:** R-verify-dont-accept (three agent claims re-run by the coordinator before entering the record); R-negative-control (the agent quoted failing output for each fix).

---

## Run 12 — 2026-09-05T03:10Z — nightly wave 1: atlasctl lane, three real defects

```text
2026-09-05T03:10Z  return     atlasctl lane (fable): full CLI sweep + a real run/stop inference cycle; 3 defects fixed in the UPSTREAM repo (Avarok-Cybersecurity/atlas-recipes, branch fix/doctor-and-registry-ux, 3 commits on cf7ec41), read-only clone in scratchpad, nothing pushed
2026-09-05T03:12Z  tool       coordinator verified the dangerous one first-hand: ~/.config/atlasctl is owned by uid 1000 = `nologik`, a LIVE login account whose agent answers on 127.0.0.1:34333 right now. doctor's printed remedy `sudo chown -R 996 ...` would take a working install's identity from that user. CONFIRMED destructive advice.
2026-09-05T03:13Z  tool       also confirmed: NO sparkrun binary anywhere on this box (only ~/.config/sparkrun remains), yet doctor says "a sparkrun install was found" and prescribes `pipx uninstall sparkrun` — which would answer "Nothing to uninstall"; the "redirect is compiled into sparkrun" claim is asserted about a binary that does not exist
2026-09-05T03:14Z  aggregate  worst NEW defect: `atlasctl registry add <name> ftp://...` HANGS FOREVER — git spawns git-remote-ftp with no deadline; still alive at T+3m30s, killed by hand. Fix refuses helper transports before any network I/O and adds http low-speed deadlines
2026-09-05T03:15Z  aggregate  real-life validation PASSED: run 7s -> ready 100s -> inference 52.9 tok/s TTFT 168ms -> stop 6s, GPU back to 0%, no leftover containers
```

**CITED:** R-verify-dont-accept (the destructive-advice claim re-proven by the coordinator against live process state before entering the record); R-exercise-it-for-real (the run/stop cycle is what turned a CLI review into a validated user journey).

---

## Run 13 — 2026-09-05T03:25Z — nightly wave 1: DX lane + the first nightly stack

```text
2026-09-05T03:25Z  return     DX lane (fable): 3 commits — cudarc nvcc probe honours CUDA_HOME, serve port-conflict error names port/holder/remedy, CONTRIBUTING check.sh correction
2026-09-05T03:27Z  tool       coordinator verified BOTH halves of the front-door control: on the unfixed tree, `CUDA_HOME=... cargo check -p cudarc` with nvcc off PATH panics at vendor/cudarc/build.rs:124; on the fixed tree the identical env compiles. Port-conflict test passes and its pre-fix failure was quoted by the lane.
2026-09-05T03:29Z  aggregate  CROSS-LANE FINDINGS the coordinator had to resolve (neither agent could see them alone):
                              (a) dx and journey INDEPENDENTLY found the same stale check.sh paragraph and both rewrote it -> merge conflict; convergent discovery is evidence the warning really was misleading, so both corrections were merged rather than one dropped
                              (b) journey's new comment asserted cudarc "does not consult CUDA_HOME" — TRUE today, FALSE the moment dx's fix lands. A docs-only stack landing first would have shipped a claim its sibling stack invalidates. Rewritten as advice, not as a claim about one build script.
2026-09-05T03:31Z  tool       stack/nightly-docs composed: 8 commits, docs-only (4 files) -> CAMPAIGN-FREE, can land the moment CI revives
2026-09-05T03:31Z  aggregate  economics split recorded: dx's cudarc + serve_router fixes touch vendor/ and crates/ -> PERF_PATHS -> they owe a campaign and CANNOT join the docs stack; they also cannot join #891 (its clearance is pinned to a closed composition). They become a third stack.
```

**CITED:** R-coordinator-owns-the-seams (parallel lanes cannot see each other; cross-lane staleness and conflicts are the coordinator's job, and both appeared in the very first wave); R-path-rule-owns-the-economics (docs-only lands free; vendor/ owes a campaign — split the stack on that line, not on subject).

---

## Run 14 — 2026-09-05T03:45Z — nightly wave 1 closed: two stacks composed

```text
2026-09-05T03:45Z  return     robustness lane (fable): 3 commits, all the PCND class (bad user input silently changing behaviour); spark-server 2344/2344; discipline note — it extracted each lenient parser VERBATIM first and wrote the strict test against it, so every control failed for the right reason before the fix existed
2026-09-05T03:47Z  tool       coordinator re-ran the worst one's control: reverting `Some(v) => v.as_str().map(Some).ok_or_else(...)` to the original lenient `Ok(v.as_str())` turns `a_non_string_mapped_value_is_an_error_not_a_silent_disable` RED. Honest nuance: the sibling typo test stayed green under that particular revert — its guard lives in the format-matching arm, a different code path, so my sabotage exercised one of the two halves, not both.
2026-09-05T03:50Z  tool       stack/nightly-hardening composed (5 lane commits + 1 fmt): cargo check EXIT=0, spark-server 2345/2345, selftest 110/110
2026-09-05T03:51Z  aggregate  COMPOSITION-ONLY DEFECT found: `cargo fmt --check` failed on the composed tree while both lanes were individually clean — an unformatted match arm in the list-query parser. Neither agent could have seen it; it exists only in the merge. Fixed as its own commit.
2026-09-05T03:52Z  aggregate  recorded-not-fixed (correctly deferred by the lane): 8 hot-path `let _ =` CUDA event record/wait sites (silent KV corruption class, unprovable without GPU); ATLAS_SPEC_ENTRY_PIN's inverted-intent fallback where `off` parses to ENABLED-at-8 and is CODIFIED BY TESTS — changing it reverses an asserted decision, so it is a daytime call, not a night edit
```

**CITED:** R-compose-then-gate (the fmt defect existed only in the composition — gate the merged tree, never trust per-lane green); R-defer-what-you-cannot-prove (the lane recorded three findings with file:line instead of half-fixing them at night).

---

## Run 15 — 2026-09-05T03:05Z — nightly-hardening cleared, and a correction to my own reporting

```text
2026-09-05T03:05Z  return     Adversary (fable) on stack/nightly-hardening: PUBLISH on 3 conditions. It re-executed rather than read: 11/11 new tests visibly run (2346 filtered, not a phantom pass), and it rebuilt one negative control from scratch — test-only hunk of 1e70e89d93 applied to its PARENT -> 1 failed, green on the stack
2026-09-05T03:06Z  aggregate  CORRECTION TO MY OWN EARLIER REPORT: I called the tool_defaults.toml defect operator-facing ("a typo silently disabled the parser"). It is NOT. The file is include_str!-baked (tool_defaults_lookup.rs:24) — VERIFIED by the coordinator — so no operator can typo it. The real exposure is a COMMITTED BUILD DEFECT, which is what the 2026-08-27 incident actually was. Still a real fix; smaller blast radius than I stated.
2026-09-05T03:08Z  aggregate  ECONOMICS CORRECTION, and it is the valuable part: this stack touches NO perf path (no kernels/ops/prefill/decode/scheduler), so gate C never triggers. A full ~5 GPU-h A/B/D campaign on it would be waste. Correct spend is C2 + one boot smoke (minutes) — or ride the next perf-bearing stack's campaign. My "three stacks owe three campaigns" framing was wrong.
2026-09-05T03:10Z  tool       condition 1 discharged: CHANGELOG entry naming all operator-visible changes, with limit>100 clamp->400 called out as the single case an external client could have relied on; tool_defaults deliberately EXCLUDED with the include_str! reason stated. Suite still 2345/2345.
2026-09-05T03:11Z  aggregate  conditions 2-3 are morning gates: no GPU leg until pushed and hosted CI (incl. the Windows `check` job) is green — CI is stalled, so waiting is the instruction, not substituting
```

**CITED:** R-verify-dont-accept (the include_str! check overturned my own published claim); R-path-rule-owns-the-economics — applied in the OTHER direction this time: the rule that says a docs PR in crates/ OWES a campaign also says a non-perf-path stack does NOT owe the full one; R-adversary-earns-its-seat (fourth consecutive review to find something no checklist held).

---

## Run 16 — 2026-09-05T03:45Z — wave 2 scheduler lane: a wave-1 claim downgraded

```text
2026-09-05T03:45Z  return     sched lane (fable) on wave 1's three DEFERRED items: A PARTIAL, B VERIFIED-as-intended, C PARTIAL — and it argued DOWN its predecessor rather than inheriting the alarm
2026-09-05T03:46Z  aggregate  Item A downgraded with evidence: the 8 swallowed fences CAN only fail on a poisoned CUDA context, where every later kernel launch fails loudly too — so the harm is losing the earliest, best-attributed canary, NOT silently-wrong tokens. Wave 1's "silent KV corruption" framing was too strong.
2026-09-05T03:47Z  tool       coordinator verified the sharpest correction: prefill_a_step.rs:395's `let _ = model.synchronize` sits INSIDE `if std::env::var("ATLAS_VISION_TIMING").is_ok()` — a debug-timing block. Wave 1 listed it as a corruption site; it can only skew a timing log. REFUTED, confirmed by reading origin/main.
2026-09-05T03:48Z  aggregate  Item C also narrowed: free_sequence_dispatch frees main KV blocks UNCONDITIONALLY before the fallible calls, so an Err leaks drafter-KV/chunk-meta, not main KV. Real, smaller.
2026-09-05T03:49Z  aggregate  Item B: the inverted-intent pin is an ASSERTED decision (#513, documented three times, codified by tests) — the lane implemented only the safe half (warn that the value was ignored and the pin stays ENABLED) and left a precise daytime proposal, including which two test lines would have to change. Exactly the deferral discipline asked for.
2026-09-05T03:50Z  tool       commit 345d782fe0: SSOT fence helper across all 8 sites + 2 free_sequence error logs + the pin warning. 239/239 scheduler tests, fmt/clippy clean, mutation control (DEFAULT_PIN_TOKENS 8->9 fails the guard test).
2026-09-05T03:51Z  aggregate  honest boundary stated by the lane and accepted: CPU mocks are infallible, so the error BRANCHES are unreachable in test — what is proven is compile-correctness plus success-path preservation. Any policy stronger than log-and-continue needs GPU fault injection.
```

**CITED:** R-verify-dont-accept, applied wave-on-wave — the value of a second pass is that it can argue the first one DOWN, and this one did, twice; R-defer-what-you-cannot-prove (the pin proposal names the exact test lines a daytime engineer must change).

---

## Run 17 — 2026-09-05T04:00Z — wave 2 API lane: the server asserting the opposite of the truth

```text
2026-09-05T04:00Z  return     api lane (fable): 3 commits on nightly/w2-api, 7/7 new tests, tree clean
2026-09-05T04:02Z  tool       coordinator verified the headline claim by grep on origin/main: `ResponsesStreamEvent::Failed` occurs EXACTLY ONCE in the whole tree — in the event-NAME MAP — and is constructed nowhere. The failure path existed on paper only. CONFIRMED.
2026-09-05T04:03Z  aggregate  the defect: a mid-stream engine failure (OOM/scheduler abort/watchdog) was log-only, finish_reason stayed "stop", so the SSE stream ended with a spec-perfect `response.completed` + `status:"completed"` + zeroed usage. An SDK client had NO programmatic signal the output was truncated — and the half-answer was then PERSISTED as a good turn that previous_response_id happily resumed from. This is the only defect found tonight where the server actively asserts the opposite of what happened.
2026-09-05T04:04Z  aggregate  also fixed: malformed/unknown /v1/responses input items were SILENTLY DROPPED (a dropped function_call desyncs its paired function_call_output); top_logprobs>20 silently clamped instead of 400.
2026-09-05T04:05Z  aggregate  lane discipline worth keeping: it reported that after restoring a mutated file with `mv`, cargo REUSED THE STALE BINARY (mtime unchanged) and its green run was therefore meaningless until a touch+rebuild. It said so rather than reporting the first green. That is the a-passing-test-may-not-have-run trap, caught by the agent on itself.
2026-09-05T04:06Z  aggregate  recorded-not-fixed: top_logprobs without logprobs:true still silently enables logprobs (needs a wire-level seam); api/chat/mod.rs is 511 lines — ALREADY over the 500 cap on origin/main, before any edit tonight
```

**CITED:** R-verify-dont-accept (the dead-enum claim settled by grep on origin/main, not by reading the report); R-a-passing-test-may-not-have-run (the lane caught cargo serving it a stale binary mid-control and disclosed it).

---

## Run 18 — 2026-09-05T04:15Z — obs lane verified; fable limit reached, ladder falls back to opus 5

```text
2026-09-05T04:15Z  return     obs lane (fable): 3 commits, 5/5 new tests, tree clean
2026-09-05T04:16Z  tool       coordinator verified the headline on origin/main: error_body — the SSOT every OpenAI error envelope passes through — contained ZERO tracing calls, and there is no TraceLayer on the router. A failing request produced NO server-side record at all; the 500 existed only in the client's response body. CONFIRMED by grep of both.
2026-09-05T04:17Z  aggregate  also fixed: cause collapse ({e} -> {e:#} so OOM / kernel-launch / swap-out stop looking alike), and a dropped response sender reported as "Inference cancelled" when it actually means the SCHEDULER DIED — three in-tree comments already called that message misleading; and per-key runtime-quantization fallbacks (uncalibrated-scale numerics) that logged at debug only, now warn-once-per-kind with counted debug
2026-09-05T04:18Z  aggregate  the lane stated the level rule it applied rather than bumping indiscriminately: warn iff operator-actionable AND bounded — 5xx per occurrence (rare when healthy), 4xx stays debug (one bad client could spam it), load-time per-key fallbacks warn once per kind (a mixed checkpoint hits 100+ keys). It also declined to bump two SSM/GDN debugs that already follow first-occurrence-info + counted-debug
2026-09-05T04:20Z  communicate FABLE SESSION LIMIT REACHED — the ctl2 lane died mid-write ("resets 2am America/New_York") while composing a dial_and_pair deadline fix. Per the standing overnight instruction the ladder falls back to OPUS 5 for nightly lanes from here.
2026-09-05T04:21Z  aggregate  wave 2 accounting: sched / api / obs COMPLETE and verified; ctl2 incomplete and being relaunched on opus 5 with its partial work named so the replacement does not redo it
```

**CITED:** R-verify-dont-accept (error_body's zero tracing calls and the absent TraceLayer both confirmed on origin/main before entering the record); R-state-the-rule (a level-bump wave is only trustworthy if the rule is written down and exceptions are named).

---

## Run 19 — 2026-09-05T06:20Z — wave 2 composed: stack/nightly-serve-truth

```text
2026-09-05T06:20Z  tool       composed sched + api + obs into stack/nightly-serve-truth: 8 commits, 30 files, NO cherry-pick conflicts between the three lanes
2026-09-05T06:22Z  tool       gates: cargo check EXIT=0, spark-server 2351/2351, selftest 110/110 — but `cargo fmt --check` DIRTY
2026-09-05T06:23Z  aggregate  DIAGNOSED PER-LANE rather than assumed: unlike wave 1 (where the fmt defect existed ONLY in the composition), this time the api lane ITSELF was dirty — sched and obs were individually clean. The lane had reported "tree clean" meaning `git status`, which is not `cargo fmt --check`. Different cause, same catch; my wave-1 generalisation would have mis-attributed it.
2026-09-05T06:24Z  aggregate  lane-instruction gap recorded for wave 3: "tree clean" must be defined as git status AND cargo fmt --check AND clippy, or lanes will keep reporting the first and meaning all three
2026-09-05T06:25Z  tool       fmt applied as its own commit, suite re-run 2351/2351, fmt now clean; stack at 8 commits
```

**CITED:** R-compose-then-gate (twice in two waves the composed tree caught what per-lane green did not); R-diagnose-dont-generalise (checking each lane separately showed this was a lane defect, not the composition defect I had seen before — the same symptom with a different cause).

---

## Run 20 — 2026-09-05T06:35Z — wave 2 ctl2 (opus 5): two of wave 1's own claims DEFEATED

```text
2026-09-05T06:35Z  return     ctl2 relaunched on OPUS 5 after the fable limit; 6 new commits, atlasctl-src suite 866/866, clippy+fmt clean
2026-09-05T06:36Z  aggregate  IT DEFEATED WAVE 1, TWICE, WITH MEASUREMENTS — the whole point of an adversarial second pass:
                              (1) the "low-speed stall deadline" claim is FALSE: timed https://192.0.2.1 (TEST-NET-1) at 132.95s WITH the settings vs 135.16s WITHOUT — identical, both bounded by the OS TCP SYN timeout, because curl's low-speed applies only to a transfer that STARTS. Worse: a hang is still reachable via `ssh://`, an ALLOWED transport with no client-side banner deadline.
                              (2) the uid-mismatch "do NOT chown" warning was printed BELOW the chown command it warns about — coordinator confirmed on the fb6cd8c blob: `sudo chown -R` at line 111, the warning at line 121. A reader following top-down destroys the install before reaching the caution. Fixed by inverting the remedy order (5d04b87).
2026-09-05T06:38Z  aggregate  the scheme-refusal half of d45275c HELD under 30 inputs incl. bypass attempts (`ftp:/host/x` falls through to ssh not the ftp helper; `FTP://` case-folded; `ext::sh -c` refused by git's own protocol.allow; `--upload-pack=` neutralised by the pre-existing `--`)
2026-09-05T06:39Z  aggregate  it also REWROTE the dead predecessor's partial work rather than finishing it: the hang it targeted is real (4 unbounded read_frame awaits, no caller wraps them) but its error text asserted a cause measured FALSE (a plain axum listener rejects the ClientHello in 2.7ms, so the operator never reaches that timeout)
2026-09-05T06:40Z  aggregate  NEW defects found: a timed-out dial read as "the machine answered and refused" (bare anyhow! has no error source, so reach::never_reached mis-walked and told users to mint a fresh code for a code never presented); `registry update` never carried the stall settings that `add` had; a failed `docker pull` aborted a launch whose image was ALREADY LOCAL
2026-09-05T06:41Z  aggregate  and it declined honestly: no wall-clock bound on git clone/fetch (needs a ProcessRunner trait change, 3 impls — out of lane scope, recorded in code docs); a credential-prompt hang it SUSPECTED but could not reproduce, so did not claim
```

**CITED:** R-verify-dont-accept, wave-on-wave and now with instruments — a claim about a timeout is worth exactly the stopwatch behind it, and 132.95 vs 135.16 killed a fix's headline; R-report-your-own-errors (the lane's predecessor and its sibling were both corrected in public).

---

## Run 21 — 2026-09-05T06:45Z — /automerger invoked; the dead sieve's question answered by hand

```text
2026-09-05T06:45Z  communicate user invoked /automerger; skill loaded and executed against live state
2026-09-05T06:46Z  aggregate  TWO agents died to API errors (sieve retry: "Connection lost mid-response"; TUI lane: "Request timed out"). The sieve died mid-sentence on a REAL question: "A removed clamp is a real risk — does every path reaching resolve_top_logprobs also pass validate_input, and is downstream bounds-safe?"
2026-09-05T06:47Z  tool       coordinator answered it BY HAND rather than spawning a third agent that might also die:
                              (a) path: responses.rs:154 calls chat_completions_inner; validate_input is at chat/mod.rs:227, INSIDE chat_completions_inner (fn starts 214) — the adapter DOES inherit the check
                              (b) type: top_logprobs is Option<u8>, ceiling 255, not unbounded
                              (c) downstream: the SSOT top-k math clamps `let nth = k.min(indexed.len().saturating_sub(1))` (spark-model traits/logprobs.rs:51) — 255 cannot panic or read OOB; worst case is an oversized alternatives list
                              VERDICT: the clamp removal is safe on both halves. The api lane asserted this without proving it; the reviewer was right to ask.
2026-09-05T06:49Z  tool       #876's original run (created 01:00) was found 100% queued — 9/9 jobs — while 19 newer runs completed around it: permanently starved, not slow. Pushed 12 held trace commits to supersede it; new run 33950388528 created 06:38 on head c5693e2730.
```

**AMENDED — rule 27b (removing a bound is a security change):** a fix that deletes a clamp must prove every remaining path to the value passes the replacement check (naming the *enclosing function* of that check, not the module) AND that the downstream consumer is safe without the bound, so a missed path degrades instead of crashing. Evidence and check are written into the rulebook.

**CITED:** R-verify-dont-accept (the lane's "adapters inherit it" claim was true but unproven — proving it took four greps); R-stop-spawning-into-a-failing-mode (two agents died to API errors on the same task; the third attempt was made by hand).

---

## Run 22 — 2026-09-05T13:00Z — CAMPAIGNS APPROVED; the first tripwire caught a vacuous control

```text
2026-09-05T12:43Z  communicate user approved the campaigns and opened dgx2 + dgx3 for parallel work
2026-09-05T12:44Z  tool       Step-0 safety on all three boxes: dgx1 0% idle no containers; dgx2 (spark-43fa) 0%; dgx3 (spark-28c2) 0%; no foreign serve anywhere
2026-09-05T12:44Z  tool       condition 4 discharged AT APPROVAL TIME: both stacks behind origin/main by 0; pins current
2026-09-05T12:45Z  tool       authoritative work-list from the branch's own binary: ELEVEN gates owed for #877 (agentic-webserver, bfcl-subset, bfcl-subset-echolp, vision/video-fidelity, ttft-warm/cold, ssm-state-poisoning, decode-floor, concurrency-sweep, concurrency-sweep-dflash2) — every record invalidated by Cargo.lock + device code across 26 targets
2026-09-05T12:47Z  aggregate  MY OWN RULE 16 VIOLATED AND CAUGHT: I read the oracle's exit code from `$?` after a pipe — that was tail's status (0). The oracle's real code was 2. Re-run without the pipe.
2026-09-05T12:48Z  tool       oracle exit 2 = "absent from this target set": ATLAS_TARGET_HW/MODEL/QUANT are BUILD-time selectors (the oracle's own doc line shows them on `cargo run`). Rebuilt target-scoped for gb10/qwen3.6-27b/nvfp4.
2026-09-05T12:57Z  aggregate  ★ THE TRIPWIRE PAID FOR ITSELF BEFORE ONE GPU-HOUR: the oracle graded every twin byte-identical and then FAILED ITSELF — "CONTROL 1-ULP perturbation detected=false / FAIL — this harness is VACUOUS". Its control flipped the LOW MANTISSA bit of one bf16 activation; summed over K=5120 and rounded back to bf16 that delta vanishes, so the harness could not distinguish identical from different and every byte-identical verdict it had ever printed was worthless.
2026-09-05T12:58Z  tool       fixed: perturb an EXPONENT bit (bf16 [lo,hi]; hi bit 0 = exponent LSB, a factor-of-two move). Re-run: CONTROL detected=true on every shape, PASS on every twin, EXIT=0. Condition 3 now GENUINELY met — #844's fp8 twin parity is verified for the first time.
2026-09-05T12:59Z  tool       tree finalised BEFORE the campaign per rule 1: oracle fix committed, new pin f3f77c82fc, pushed. The stamp bound to 82552fe34d must be re-issued against the new pin.
2026-09-05T13:00Z  tool       parallel fan-out: dgx2 (~/wt-877) and dgx3 (/workspace/wt-877) both at the stack pin, release builds running; dgx2 needed HOME not /workspace (permission denied)
```

**AMENDED — rule 24 extension (a control must survive the arithmetic it polices):** a negative control perturbing an INPUT must be large enough to survive the operation under test — accumulation length and output rounding both attenuate it. A 1-ULP mantissa flip vanished across K=5120 into bf16. CHECK: the control must print `detected=true`; a harness whose control cannot fire must exit nonzero and say it is vacuous (this one did, which is the only reason it was caught).

**CITED:** R-16 (pipeline `$?` — violated by me, caught in the same minute); R-24 (prove every guard can fail — the oracle proved itself unable and said so); R-1 (finalise the tree before the campaign — the oracle fix landed before any gate started).
