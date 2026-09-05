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
