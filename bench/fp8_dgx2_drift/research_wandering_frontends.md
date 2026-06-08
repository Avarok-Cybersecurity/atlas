# Anti-Wandering in Agentic Coding Frontends — Field Report

**Context**: opencode + Qwen3.6-35B-A3B-FP8 shows 30-40% of runs ending with
5-7 bash/read calls and zero file writes. We need to know how the wider
ecosystem solves this so we can either configure opencode out of the
problem, port a fix server-side, or upstream a patch.

---

## BIG ANSWER FIRST

**Yes — opencode has a per-agent `steps` config that caps tool-call rounds
and forces the model to summarize when hit.** Setting `steps: 8`-`12` in the
agent definition for Qwen3.6 is the single most direct lever we have. The
legacy key was `maxSteps` (deprecated). When the cap is reached, opencode
auto-injects an instruction telling the model to *"respond with a
summarization of its work and recommended remaining tasks"* — which is
exactly the nudge a wandering model needs to commit to an edit.

```json
{
  "agent": {
    "qwen36-coder": {
      "description": "Edit-focused agent for Qwen3.6",
      "prompt": "You are an editor. Prefer Write/Edit over Read after the second exploration call.",
      "steps": 10
    }
  }
}
```

Source: opencode docs, "Agents" page — confirmed via direct fetch.

The complementary lever is opencode's **doom-loop detector**
(`packages/opencode/src/session/processor.ts` lines 32, 425-449,
threshold = `DOOM_LOOP_THRESHOLD = 3`). It fires only when the *same
tool with byte-identical inputs* runs 3x in a row — so it catches pure
repetition but **does not catch Qwen3.6's pattern of "5 different
bash calls, no writes"**. That's the gap we're hitting.

---

## Per-frontend mechanism table

| Frontend | Anti-wander lever | Source location | Default | Effectiveness |
|----------|------------------|-----------------|---------|---------------|
| **opencode** | `steps` cap + summarize-on-hit | docs/agents/ | unset (∞) | High once configured; off by default |
| **opencode** | Doom-loop detector (3x identical) | `processor.ts:425-449` | always on | Low — only catches *identical* repetition |
| **opencode** | Permission ask on `"doom_loop"` | `processor.ts:441` | always on | Halts loop, but needs human |
| **opencode** | Context-overflow auto-summary | session/overflow.ts via processor | 90% ctx | Backstop only, doesn't prevent wandering |
| **Claude Code SDK** | `max_turns` / `maxTurns` | docs `agent-loop` | unset | Hard cap, returns `error_max_turns` |
| **Claude Code SDK** | `max_budget_usd` | docs `agent-loop` | unset | Dollar-budget kill |
| **Claude Code SDK** | Plan-mode prompt-only guard | system reminder | opt-in | Soft; no tool deny-list |
| **Roo Code** | `ToolRepetitionDetector` | `src/core/tools/ToolRepetitionDetector.ts` | always on | Catches identical-call loops |
| **Roo Code** | `ConsecutiveMistakeError` | `src/core/task/Task.ts:53-56` | 3 mistakes | Aborts task on 3 errors |
| **Cline (≥3.35)** | *Removed* max-requests | release notes v3.35 | — | Open issue #7585 — regressions reported |
| **Aider — Architect mode** | Two-model: planner→editor | `aider/coders/architect_coder.py` | opt-in | High — editor only writes diffs |
| **Aider — Code mode** | Single LLM emits diff fences directly | `aider/coders/editblock_coder.py` | default | Prompt forces SEARCH/REPLACE block syntax |
| **Continue.dev** | `streamText` step limit (AI SDK) | `core/llm/streamChat.ts` | SDK default | Per-step, not per-task |
| **Cursor Composer** | Closed-source; no public docs | — | — | Reddit reports of stuck composers |
| **Devin / Cognition** | "Session Insights" post-hoc rubric | blog | — | Post-mortem only — doesn't stop live loop |

---

## Concrete code & quotes

### opencode — exact doom-loop check (the lever that *doesn't* fire for us)

`packages/opencode/src/session/processor.ts`, lines 32 + 424-449:

```typescript
const DOOM_LOOP_THRESHOLD = 3
// ...
const parts  = MessageV2.parts(ctx.assistantMessage.id)
const recentParts = parts.slice(-DOOM_LOOP_THRESHOLD)
if (
  recentParts.length !== DOOM_LOOP_THRESHOLD ||
  !recentParts.every(
    (part) =>
      part.type === "tool" &&
      part.tool === value.name &&
      part.state.status !== "pending" &&
      JSON.stringify(part.state.input) === JSON.stringify(input),
  )
) {
  return
}
yield* permission.ask({
  permission: "doom_loop",
  patterns:   [value.name],
  sessionID:  ctx.assistantMessage.sessionID,
  metadata:   { tool: value.name, input },
  always:     [value.name],
  ruleset:    agent.permission,
})
```

This only matches if the **last three tool calls are the same tool with the
same JSON-serialised input**. Qwen3.6's failure mode is *five different bash
commands, no writes* — every input differs, threshold never trips.

### opencode — hard stop conditions (`session/index.ts` / `session.ts`)

Per the cefboud.com deep-dive, the loop terminates via:

```typescript
stopWhen: () => steps.length >= 1000 || processor.getShouldStop()
```

So opencode's only hard cap absent `steps` is **1000 tool rounds**, which is
not a guard — it's a budget ceiling.

### Roo Code — `ToolRepetitionDetector` (parallel design)

`src/core/tools/ToolRepetitionDetector.ts` — three methods
(`isRepetitive`, `recordInvocation`, `reset`) keyed on `(toolName, params)`,
default `DEFAULT_CONSECUTIVE_MISTAKE_LIMIT = 3`. Throws
`ConsecutiveMistakeError` from `Task.ts:53-56` and aborts the task with a
user-visible error. Same design as opencode's doom-loop — same blind spot
to *non-identical* exploration.

### Aider — Architect/Code split

The **architect coder** runs the reasoning model with a system prompt that
explicitly forbids emitting diffs:

> "Don't show me the entire updated file. Just tell me your plan in a
>  concise way."

Then the **editor coder** runs a cheaper/faster model with a prompt that
*requires* SEARCH/REPLACE blocks. Wandering is structurally prevented
because:

1. The planner can't write (architecturally disabled).
2. The editor's system prompt demands code fences — bare exploration text
   triggers a re-prompt.

This is the strongest anti-wandering design in the table. The cost is
2x model calls per task.

### Claude Code — agent loop with `max_turns`

From `code.claude.com/docs/en/agent-sdk/agent-loop`:

> "You can cap the loop with `max_turns` / `maxTurns`, which counts
>  tool-use turns only. … When either limit is hit, the SDK returns a
>  `ResultMessage` with `error_max_turns`."

Claude Code itself does **not** ship a Qwen-style doom-loop guard. It
relies on (a) Claude's strong instruction-following + (b) the
`PreToolUse` hook so embedders can write their own anti-wander policy.
This is why teams using Claude Code rarely see wandering — the model
respects "Use Write/Edit liberally" without external enforcement.

### Cline — removed the safety net

> "In Cline v3.35 … the max requests limit was removed, which previously
>  served as a safety mechanism."
> — *Cline docs / changelog*

Issue **#7585** (Nov 2025): users report Cline now loops indefinitely.
**This is exactly the failure mode we should not replicate.** Cline's
trajectory tells us removing the cap is worse than having one.

---

## Five concrete patterns we can adopt (ordered by effort vs payoff)

### 1. Configure opencode `steps: 10` for the Qwen3.6 agent profile **(do this today)**

Single-line config change, zero server work. The summarize-on-hit
behaviour acts as a built-in *"you've done N exploratory calls, please
write the files now"* injection — opencode already does this, we just
need to enable it. Cost: occasional truncated session on legitimately
deep work — mitigated by setting `steps: 12` to start and tuning.

### 2. Server-side write-pressure system message (port of Aider architect pattern)

After `N` tool calls in the same Atlas request where none are
file-modifying (Write/Edit), inject a system-role message:

> "You have made N exploration calls without writing any files. State
>  your plan in one sentence, then call Write or Edit to commit it. If
>  more information is genuinely required, say so explicitly."

This is **server-side** (Atlas tool-call middleware) so it works for
every frontend, not just opencode. Threshold ~5 matches what we observe.

### 3. Tool-class repetition detector (broader Roo/opencode pattern)

Track the last K tool *classes* (read-class vs write-class vs exec-class),
not just identical inputs. If the last `K=5` are all read-class, inject
a summarize-and-commit prompt. This fixes the exact gap in opencode's
doom-loop detector that's hurting us — same algorithm, broader key.

### 4. Two-stage planner/editor agent for Qwen3.6 specifically (Aider port)

For users complaining about wandering, ship an opencode agent definition:

```json
"qwen36-architect": {
  "model": "qwen3.6-35b-fp8",
  "prompt": "Output a numbered plan, max 3 steps. Do not call tools.",
  "tools": { "*": false }
},
"qwen36-editor": {
  "model": "qwen3.6-35b-fp8",
  "prompt": "Apply the plan. Use Write/Edit only. Never Read more than 2 files.",
  "steps": 8
}
```

Composes opencode's existing primitives — no Atlas changes.

### 5. Upstream patch to opencode: extend doom-loop to "no-write window"

PR to `sst/opencode` adding a sibling check at `processor.ts:425`:

```typescript
const recentTools = parts.slice(-NO_WRITE_WINDOW)
  .filter(p => p.type === "tool")
const allReadClass = recentTools.length >= NO_WRITE_WINDOW &&
  recentTools.every(p => READ_CLASS_TOOLS.includes(p.tool))
if (allReadClass) {
  yield* injectSystemReminder(
    "You have made several exploration calls without modifying files. " +
    "Commit your next step with Write or Edit, or explicitly state what " +
    "information is still missing.")
}
```

Same data structures, same permission infrastructure, broader semantics.
This is a tractable upstream PR (~50 LOC) and fixes the issue for everyone.

---

## Prompt-engineering patterns that empirically reduce wandering

- **Aider tips page**: *"For complex changes, discuss a plan first" via
  `/ask`; once approved, drop the prefix and execute.* — forces the
  plan/execute split into the human workflow.
- **Claude Code system prompt** (per Piebald-AI dump): communication-style
  prompt tells the model to *"write concise end-of-turn summaries"* and
  *"avoid comments and planning documents in code"* — discourages
  meta-discussion that often precedes wandering.
- **Roo Code prompt-engineering docs**: *"Divide complex tasks into
  smaller, well-defined steps. When rejecting an action, explain why."* —
  steerable feedback loop; the user becomes a wander-recovery vector.
- **opencode community config** (`joelhooks/opencode-config`, gotar et al.):
  ship custom agents with very specific tool allow-lists, e.g. an
  "editor" agent with only `Write`, `Edit`, `Bash` — making Read literally
  impossible for that agent forces it to write.

---

## Effectiveness evidence

- opencode PR **#3445** ("doom loop detection") was merged after GLM-4.6 and
  Grok-Code were observed looping on identical tool calls. Maintainers
  report it eliminates *identical* repetition; the recurring follow-up
  issue **#12716** ("Doom loop is not caught during reasoning or output")
  + PR **#12623** confirm the original detector misses non-identical
  patterns — same blind-spot we hit.
- Roo Code's `consecutiveMistakeLimit = 3` is preserved across releases —
  community signal it works; removing it (Cline path) regressed
  immediately.
- Aider's architect mode is recommended specifically for o1-class
  reasoning models that "excel at reasoning but struggle with precise
  file editing" — the canonical wander profile. Reports on the
  Aider Discord and HN consistently rate it the most reliable mode for
  weaker editors. (Same logic applies to Qwen3.6-35B FP8.)
- Claude Code's `max_turns` is documented as production-default; missing
  it is called out as a footgun: *"Without limits, the loop runs until
  Claude finishes on its own … can run long on open-ended prompts."*

---

## Recommendation for Atlas (one paragraph)

Ship two things together:
**(a)** an `atlas-recipes` opencode profile for Qwen3.6 with
`steps: 10` plus a system-prompt nudge "Prefer Write/Edit over Read after
the second exploration call" — this is a 5-minute change that uses
opencode's existing summarize-on-cap path; and
**(b)** a server-side write-pressure injector (pattern #2 above) that
fires after 5 consecutive non-edit tool calls, so the fix applies
regardless of frontend. Together these mirror opencode's existing
doom-loop detector but cover the *exploration without commit* class,
which is the actual Qwen3.6 failure mode.

---

## Sources

- opencode docs — Agents: https://opencode.ai/docs/agents/
- opencode docs — Config: https://opencode.ai/docs/config/
- opencode source — processor.ts (dev branch):
  https://github.com/sst/opencode/blob/dev/packages/opencode/src/session/processor.ts
- opencode PR #3445 (doom loop detection):
  https://github.com/sst/opencode/pull/3445
- opencode issue #12716 + PR #12623 (doom-loop blind spot):
  https://github.com/sst/opencode/issues/12716
- opencode deep-dive (cefboud): https://cefboud.com/posts/coding-agents-internals-opencode-deepdive/
- Claude Agent SDK loop:
  https://code.claude.com/docs/en/agent-sdk/agent-loop
- Claude Code system-prompt dump (Piebald-AI):
  https://github.com/Piebald-AI/claude-code-system-prompts
- Roo Code error handling (DeepWiki):
  https://deepwiki.com/RooCodeInc/Roo-Code/5.5-error-handling-and-retries
- Roo Code prompt engineering:
  https://docs.roocode.com/advanced-usage/prompt-engineering
- Cline v3.35 release notes: https://cline.bot/blog/cline-v3-35
- Cline issue #7585 (regressed looping):
  https://github.com/cline/cline/issues/7585
- Aider modes: https://aider.chat/docs/usage/modes.html
- Aider tips: https://aider.chat/docs/usage/tips.html
- Continue.dev agent mode: https://docs.continue.dev/agent/how-it-works
- Cognition / Devin: https://cognition.ai/blog/how-cognition-uses-devin-to-build-devin
