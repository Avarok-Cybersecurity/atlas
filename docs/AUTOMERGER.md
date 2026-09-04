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
