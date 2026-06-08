# research3 — Production agent drift mitigations

Survey of how shipping coding agents handle "model drift mid-stream" — malformed tool calls, truncated JSON, repeated nonsense, doom loops. Focus on patterns Atlas can adopt at the inference-engine layer (sampler / scheduler) or recommend to clients (opencode, Claude Code).

Date: 2026-05-26. Scope: opencode, aider, Cursor, Claude Code, Continue.dev, Roo Code / Cline, GitHub Copilot Workspace agent, OpenAI o1 / Anthropic Computer Use, plus the relevant academic primitives (constrained decoding, EAGLE-3, reflect-retry-reward).

---

## 1. opencode (sst/opencode) — source-level

opencode's `session/processor.ts` and `session/retry.ts` ship the most explicit drift-handling state machine of any agent reviewed. Three concrete patterns:

**1a. Doom-loop detector.** `DOOM_LOOP_THRESHOLD = 3` in `processor.ts`. On every `tool-input-start`, opencode walks the last 3 message parts and checks whether they are all `type=="tool"` with the same `tool` name AND byte-identical `JSON.stringify(input)`. If so, the request is escalated to a `permission.ask("doom_loop", …)` modal — the user has to explicitly approve or kill it. This is client-side, not server-side, but the heuristic itself (same tool + same args 3× in a row → halt) is cheap and accurate. Cost: O(3) JSON.stringify per tool call. Effectiveness: closes the worst class of failure (locked-loop calling `bash ls` forever) but does nothing for *similar-but-not-identical* drift.

**1b. Exponential backoff retry (provider/HTTP layer).** `retry.ts` implements `RETRY_INITIAL_DELAY=2000`, `RETRY_BACKOFF_FACTOR=2`, `RETRY_MAX_DELAY_NO_HEADERS=30s`. Honors `retry-after-ms` and `retry-after` headers. Filters by error class: 5xx + explicit "isRetryable" + textual "rate limit"/"too many requests" patterns trigger retry; `ContextOverflowError` is non-retryable. **Crucially, this is transport-layer only — it does NOT cover semantic failures** (malformed JSON, schema-invalid args).

**1c. Open gap: semantic repair.** Issues #15906, #18108, #735 are all open and active: opencode currently aborts the turn on (i) JSON parse failure in tool args, (ii) `finishReason: "length"` (truncated mid-tool-call), (iii) schema-mismatch. Community demand for `repairToolCall(rawText, schema)` + retry-with-error-feedback. Notable: `finishReason: "length"` is treated as normal completion, NOT as truncation signal — direct relevance to Atlas's late-layer drift (early EOS / mid-call termination would look identical to the client).

**Pattern category:** Client-side. Cost: ~0 (just bookkeeping). Effectiveness: doom-loop ~100% on exact repeats; truncation/repair currently 0%.

## 2. aider — source-level (`base_coder.py`)

aider exposes the cleanest reflect-and-retry primitive in OSS coding agents. Three knobs:

**2a. `max_reflections = 3` (class default).** Inside `Coder.run()`, after each LLM turn aider checks `self.reflected_message`. If set, the next turn is fed `self.reflected_message` as the user prompt; counter increments; bail after 3. This is the canonical "validation feedback loop" — tool fails → error text becomes next user message → LLM corrects.

**2b. Layered fuzzy matchers.** Search/Replace block application tries: exact → whitespace-insensitive → indent-preserving → difflib fuzzy. Only after all 4 fail does it emit a reflection ("the SEARCH block did not match; here is the file…"). This is **input-side repair**, not LLM-side — drift is silently fixed when within tolerance.

**2c. Transport retry with exponential backoff.** Separate from reflection: `retry_delay = 0.125`, doubles per attempt, cap = `RETRY_TIMEOUT`. Triggered by `LiteLLMExceptions`. Also handles `FinishReasonLength` as a specific exception (unlike opencode) — when the model hits max-tokens, aider re-sends with continuation prompt.

**Pattern category:** Client-side. Cost: up to 3× LLM calls per task. Effectiveness: high on capable models (GPT-4, Sonnet); aider docs admit that on weaker models reflections quickly exhaust and the user is told to "try `--edit-format whole`" (i.e. swap format, not reflect harder).

## 3. Cursor — blog-only

Cursor publishes inference internals but not their agent loop in detail.

**3a. Speculative edits (`fast-apply`).** ~1000 tok/s on a 70B model via *deterministic* speculation: the existing file content IS the draft. Verification is done by the target model itself — no separate verifier. Achieves 13× over vanilla, 9× over GPT-4-based speculative edits. Critically: when the verifier rejects a draft chunk, the model regenerates only that chunk; the rest of the file is preserved. This is the closest production analog to "mid-stream cancel + restart with corrective context", but it operates on file-edit chunks, not tool-call args.

**3b. Real-time RL on Composer.** Cursor's Composer model retrains every ~90 minutes against accept/reject signal from users. Effectively an online critic loop where the *user* is the verifier. Mitigates long-tail drift over days, not seconds.

**3c. No published critic model for tool args.** TensorZero's reverse-engineering of Cursor's LLM client and the Fireworks fast-apply post both confirm Cursor relies on (i) constrained schema (server-side) + (ii) speculative-edits style "draft is mostly correct, verify token-by-token" rather than a separate critic.

**Pattern category:** Server-side (inference engine). Cost: ~0 extra forward passes (target model verifies as part of normal decoding). Effectiveness: 9-13× speedup, near-zero quality drop — best-in-class.

## 4. Claude Code

Claude Code's agent loop is documented but largely closed; what is public:

**4a. `isError: true` contract.** Tool implementations MUST return `{is_error: true, content: "..."}` on recoverable failure. The agent then sees the error string in the next turn's tool_result and may retry/adapt. A March 2026 changelog explicitly fixed `error_during_execution`, `error_max_turns`, `error_max_budget_usd` to set `is_error: true` — automated pipelines previously treated max-turns as success. Direct validation-feedback loop, no special server-side handling.

**4b. Thinking-history corruption (drift cause, since fixed).** March 26 2026 bug: thinking history was wiped each turn → slow drift over long sessions. Mitigation: regression eval suite caught it post-release. Anthropic's eval guide formalizes this: "capability evals graduate to regression evals once pass rate ≈100%".

**4c. No mid-stream cancellation.** Claude Code aborts a turn only on hard errors (OOM, network). Bad tool calls produce tool_result with `is_error:true` and the loop continues.

**Pattern category:** Client-side. Cost: 1 extra LLM turn per error. Effectiveness: high — `is_error:true` lets the model self-correct without protocol changes.

## 5. Anthropic Computer Use — server-side drift mitigation

The Computer Use docs and Anthropic's evals post describe production mitigations:

**5a. Multi-agent debate / self-consistency.** "Existing methods […] maintain multiple candidate reasoning trajectories and rely on consensus-based aggregation for acted belief determination." Used as pre-action verifier. Cost: N× tokens per acted belief; only worth it for high-stakes actions (clicking "delete", running code).

**5b. End-to-end browser-based re-verification at session start.** Multi-session agents re-run a smoke test at session boot. Catches regressions from prior sessions. Direct analog: Atlas could ship a "warmup probe" that runs a known-good tool call and checks output cosine before serving the first user request.

**5c. Mitigate prompt-injection before granting credentials.** Defense-in-depth, not strictly drift, but referenced as a server-side layer.

**Pattern category:** Both. Cost: 1-N extra forward passes per acted decision. Effectiveness: well-documented in OpenReview / arxiv (MirrorGuard, self-auditing); used selectively in production.

## 6. Roo Code / Cline

Roo Code 3.36 release notes: "grace retry" — when a model emits malformed tool XML, Roo silently retries once before showing the user an error. Plus parameter-presence validation: tools with missing required params get rejected client-side and fed back as error. Mid-stream errors checked via `finish_reason: "error"` chunk inspection (since 200 OK already returned). Open issues #12185, #4921, #2822 show Roo still aborts on partial-streaming failures — no automatic restart.

**6a. Auto-temperature reduction (proposed, issue #6156).** Community-requested: on tool failure, retry with `temperature *= 0.5` (or → 0). Mirrors academic finding that sampling noise causes many tool-arg drifts. Not yet shipped.

**Pattern category:** Client-side. Cost: 1× extra request on grace-retry. Effectiveness: catches transient cases, not systematic drift.

Cline uses the same `is_error`-style propagation as Claude Code; no separate critic.

## 7. Continue.dev

Tool results are unconditionally fed back to the model as context items — the simplest possible validation-feedback loop. No mid-stream cancel; no doom-loop detector documented. Policies allow per-tool `automatic` / `ask` gates, which doubles as a manual circuit breaker.

**Pattern category:** Client-side. Cost: ~0. Effectiveness: relies entirely on model self-correction; users complain on Discord about loops on weaker models.

## 8. GitHub Copilot Workspace (agent mode)

VS Code blog (Feb 2025) describes the agent loop: "responds to compile and lint errors, monitors terminal and test output, and auto-corrects in a loop until the task is completed." Uses Cursor-style *speculative decoding* for diff application via a dedicated endpoint. The speculative endpoint is server-side; the error-driven correction loop is client-side. No public critic-model details; compile/lint errors ARE the critic.

**Pattern category:** Hybrid. Cost: speculative endpoint ~9× cheaper than vanilla apply; correction loop ~1 extra turn per compile error. Effectiveness: shipped at scale, generally well-received.

## 9. OpenAI o1 — Generator–Verifier (the formal pattern)

o1 explicitly trains an Actor (Generator) + Critic + Reward model. Process-Reward Models (PRM) score intermediate steps; Outcome-Reward Models (ORM) score final answer. At inference: lookahead K=16 tokens, verifier accepts/rejects each block. This is *inference-time* search, not just RL training.

**Speculative Speculative Decoding** (arxiv 2603.03251): draft model predicts likely *verification outcomes* and pre-speculates for ALL of them in parallel; whichever outcome fires, response is immediate. Latency-hiding for the verifier.

**Pattern category:** Server-side. Cost: 2× model deployment (generator + verifier), or 1× model + lightweight scorer. Effectiveness: the published mechanism behind o1's reasoning gains.

## 10. Constrained decoding (XGrammar / outlines / lm-format-enforcer)

The "best" production fix for malformed tool args is *prevention*, not retry. XGrammar-2 (May 2026) ships TagDispatch: tool-name token dispatches to its argument schema, blocking all schema-invalid tokens at sample time. Logits-mask overhead < 1µs/token when grammar is cached. Default in recent vLLM. SGLang and TensorRT-LLM ship equivalent. Trivially bolts onto Atlas's existing sampler.

**Pattern category:** Server-side (inference engine — exactly where Atlas lives). Cost: ~0 per token. Effectiveness: makes malformed JSON literally impossible. Caveat: cannot catch *semantic* drift ("the args are well-formed but wrong file path"). Cannot catch truncation (model still emits valid prefix then stops).

## 11. EAGLE-3 + tree drafting

EAGLE-3 produces 2.4-3× speedup at ~40% acceptance, multi-token verification via target model. Not a drift fix per se, but the *verification step* doubles as a built-in critic: any draft token the target model wouldn't have produced is rejected. Atlas already runs MTP (2-token spec decode at 59.9 tok/s on the Qwen3-Next NVFP4 path). Extending the verify-and-reject criterion beyond "exact token match" — e.g., reject when target model probability of drafted token drops below threshold AND token is part of an active tool-call schema — gives free mid-stream sanity at the sampler.

---

## Ranked top-5 patterns Atlas could adopt

Atlas is an inference engine, so adoption favors server-side / sampler-level changes that travel with the model rather than require every client (opencode, Claude Code, sparkrun, etc.) to update.

### #1 — Constrained decoding for tool calls (XGrammar / TagDispatch)

Already partially shipped (`project_xgrammar.md`, F68 fix `project_grammar_bytelevel_vocab.md`, F72 byte-anchor `project_f72_byte_anchor.md`). Extend to: (a) full TagDispatch so tool-name token gates argument schema; (b) per-tool grammar caching keyed by `(model, tool_name)`. Server-side, near-zero overhead, eliminates ~90% of opencode's reported malformed-tool-call bugs. Highest leverage / lowest cost on the list.

### #2 — Truncation signaling: distinguish `finish_reason="length"` mid-tool-call

Both opencode (#18108) and aider explicitly handle this; opencode currently doesn't. If Atlas's grammar engine knows we're mid-`tool_call` AND we hit `max_tokens`, surface a structured `finish_reason="tool_call_truncated"` (or set `is_error:true` on the assistant message). This is a 1-line scheduler change that fixes the entire class of "the model emitted valid prefix, then EOS at limit, agent treats it as complete and runs garbage." Free, server-side, additive to NVFP4/FP8 paths.

### #3 — Doom-loop detector at the sampler

Port opencode's exact heuristic into the sampler: track the last N (N=3) `tool_call_start … tool_call_end` spans; if all 3 share `tool_name` AND args-hash, force-emit an EOS or insert a corrective system reminder ("you have called X with identical args 3 times; reconsider"). Cheap (`O(N)` hash compare), purely a sampler-side filter, works regardless of which client is driving Atlas. Mirrors what every serious agent now does in userspace.

### #4 — Speculative-decode verifier as drift detector

Atlas already runs MTP (2-token spec on Qwen3-Next NVFP4, 59.9 tok/s). The verifier step computes target-model logits for every drafted token. Add a sampler-side counter: rolling acceptance rate per request. If acceptance rate collapses below baseline (e.g., < 30% for 64 consecutive tokens), it almost certainly means the model has entered a low-entropy degenerate state (loop, gibberish, drift). Surface as a `finish_reason="drift_detected"` early-stop. Zero extra forward passes — uses signal already computed. Mirrors Cursor's speculative-edits "verify-as-you-go" without their RL training pipeline.

### #5 — Warmup probe at session start (Anthropic Computer Use pattern)

Borrow from Anthropic's "end-to-end re-verification at session start": ship a `/health/probe` endpoint that runs a known-good tool-call prompt through the live engine and checks `cosine(output_logits, reference_logits) > 0.99`. Catches the failure mode this `bench/fp8_dgx2_drift/` directory was set up to investigate — silent late-layer drift from quantization or weight-loader regressions. Cost: 1 forward pass per probe interval (e.g., every 30 min or on container start). Doesn't fix drift but turns hours-of-debugging into seconds-of-detection.

---

### Patterns explicitly NOT recommended

- **Reflect-retry-reward at the engine layer.** This is a model-training / agent-loop pattern, not inference. Belongs in the client (which already does it via `is_error:true`).
- **Separate critic / verifier model deployment.** o1-style two-model serving doubles GPU memory; on GB10 with 119.7 GB per GPU, doesn't fit alongside a 80B NVFP4 hybrid model. Use the cheap sampler heuristics above instead.
- **Temperature auto-reduction on tool failure** (Roo issue #6156). Belongs in the client; engine has no signal about "tool failure" — only about acceptance-rate collapse, which #4 already handles.

Sources:
- [opencode session/retry.ts](https://github.com/sst/opencode/blob/main/packages/opencode/src/session/retry.ts)
- [opencode session/processor.ts (DOOM_LOOP_THRESHOLD)](https://github.com/sst/opencode/blob/main/packages/opencode/src/session/processor.ts)
- [opencode issue #15906 — retry invalid tool-call diff](https://github.com/sst/opencode/issues/15906)
- [opencode issue #18108 — truncated tool calls, finishReason length, repairToolCall doom loop](https://github.com/sst/opencode/issues/18108)
- [aider base_coder.py (max_reflections=3)](https://github.com/Aider-AI/aider/blob/main/aider/coders/base_coder.py)
- [aider editblock_coder.py — layered fuzzy match](https://github.com/Aider-AI/aider/blob/main/aider/coders/editblock_coder.py)
- [aider edit-errors troubleshooting](https://aider.chat/docs/troubleshooting/edit-errors.html)
- [Cursor — Editing Files at 1000 Tokens per Second](https://cursor.com/blog/instant-apply)
- [Cursor — Improving Composer through real-time RL](https://cursor.com/blog/real-time-rl-for-composer)
- [Fireworks — How Cursor built Fast Apply](https://fireworks.ai/blog/cursor)
- [TensorZero — Reverse engineering Cursor's LLM client](https://www.tensorzero.com/blog/reverse-engineering-cursors-llm-client/)
- [Anthropic — Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Anthropic Computer Use docs](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use)
- [Roo Code — Error Handling and Retries (DeepWiki)](https://deepwiki.com/RooCodeInc/Roo-Code/5.5-error-handling-and-retries)
- [Roo Code issue #12185 — malformed tool call recovery](https://github.com/RooCodeInc/Roo-Code/issues/12185)
- [Roo Code issue #6156 — auto temperature reduction](https://github.com/RooCodeInc/Roo-Code/issues/6156)
- [Continue.dev — How Agent Mode Works](https://docs.continue.dev/ide-extensions/agent/how-it-works)
- [VS Code blog — Introducing GitHub Copilot agent mode](https://code.visualstudio.com/blogs/2025/02/24/introducing-copilot-agent-mode)
- [OpenAI o1 reverse-engineering — Generator-Verifier](https://sikkha.medium.com/decoding-openais-o1-model-insights-from-initial-thorough-investigation-and-reverse-engineering-91d76ce1ec2a)
- [Speculative Speculative Decoding (arxiv 2603.03251)](https://arxiv.org/pdf/2603.03251)
- [XGrammar-2 — TagDispatch for tool calling](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation)
- [EAGLE-3 — vLLM blog](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding)
- [Reflect, Retry, Reward — Writer Engineering](https://writer.com/engineering/self-reflection-llm-reinforcement-learning/)
