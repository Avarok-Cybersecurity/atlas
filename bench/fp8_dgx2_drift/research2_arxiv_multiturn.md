# Multi-Turn Agentic Coherence & Tool-Call Faithfulness: NEW Literature (Jun 2025 - May 2026)

**Scope**: Updates the April-2025 synthesis (`research_multi_turn_arxiv.md`) with work appearing on arXiv between June 2025 and May 2026. The angle is Atlas serving Qwen3.6-35B-A3B-FP8 (hybrid 30 SSM + 10 attention) to opencode, where structurally-valid-but-semantically-wrong tool args appear by turn 5+ (wrong file paths, repeated failed actions, drift on filename strings). Grammar enforcement, sampler forced-token bypass, and tier-2 strict path validators are already in. The open failure mode is the model picking a syntactically legal but semantically wrong argument value. Prior coverage of Lost-in-the-Middle, NoLiMa, FlowKV (2505.15347), ZeroTuning (2505.11739), DuoAttention, MInference, Q-Filters, "Drift No More?" (2510.07777), Multi-IF, PrefEval, BFCL-v3, LongFuncEval (2505.10570), LZ Penalty (2504.20131), Attractor Cycles (2502.15208), Self-Correction Bench (2507.02778), EAQuant (2506.13329), Qwen3-Coder-Next (2603.00729), Kimi K2, DeepSeek-V3.2, SWE-EVO is NOT repeated here.

---

## 1. New diagnostic / measurement work

### 1.1 AgentHallu (arXiv:2601.06818, Jan 2026)
First end-to-end hallucination-attribution benchmark for agents. 693 trajectories, 7 frameworks, 14 sub-categories. Critical finding for Atlas: **the best frontier model resolves only 11.6% of tool-use hallucinations correctly, vs 41.1% step-localization overall**. "Incorrect argument", "missing tool", and "unnecessary tool" are the top failure sub-types - exactly Atlas's symptom cluster. This is the closest external benchmark for what opencode is doing wrong.

### 1.2 Information Fidelity in MCP Agents (arXiv:2602.13320, Feb 2026)
Theoretical framework: cumulative tool-call distortion is a **martingale with linear-in-T growth and O(sqrt T) deviation bounds**. Empirically tracks on Qwen2-7B, Llama-3-8B, Mistral-7B. Two implications for Atlas: (a) opencode's per-turn drift is mathematically expected, not a bug, so the right intervention shrinks the *slope*, not the mechanism; (b) the bound is over hybrid discrete-fact + continuous-semantic distortion, which matches Atlas's symptom (semantically-wrong filename strings rather than parse failures).

### 1.3 Internal Representations as Hallucination Indicators (arXiv:2601.05214, Jan 2026)
Train a **lightweight last-layer classifier** on (tool-call, correct/hallucinated) pairs; at inference, score each proposed call in real time and gate execution. 86.4% detection accuracy, real-time, "particularly excelling at detecting parameter-level hallucinations and inappropriate tool selections". This is the most directly Atlas-relevant finding in the entire 2026 cohort.

### 1.4 Beyond Resolution Rates (arXiv:2604.02547) + Inside the Scaffold (arXiv:2604.03515)
Failure trajectories are 12-82% longer than successful ones on OpenHands / SWE-agent / Prometheus. Repository navigation dominates agent activity over patch writing. Confirms the SWE-EVO "stuck-in-loop" finding from a different angle: Atlas's "repeated attempts at failed actions" is the dominant cost driver, not novel errors.

---

## 2. New inference-time interventions targeting the Atlas failure mode

### 2.1 TACT - activation steering for overthinking/overacting (arXiv:2605.05980, May 2026)
**Most relevant new paper for Atlas.** TACT (Think-Act Calibration via activation Steering) labels coding-agent trajectory steps as overthinking, overacting, or calibrated, finds the hidden states separate linearly along two drift axes (AUC ~= 0.9), and **projects drifted activations back toward calibrated at inference time**. Results: +5.8 pp on SWE-bench Verified for Qwen3.5-27B, +4.8 pp on Gemma-4-26B-A4B-it, with 26% fewer steps-to-resolve. Pure inference-time, no training. Works on Qwen3-family architectures with similar hidden-state geometry, so plausibly transfers to Qwen3.6-35B-A3B.

### 2.2 ASA - Activation Steering Adapter for Tool Calling (arXiv:2602.04935, Feb 2026)
Lightweight inference-time router that reads intermediate activations, infers domain, and produces adaptive steering strength. Training-free. On MTU-Bench with Qwen2.5-1.5B, **strict tool-use F1 jumps 0.18 -> 0.50; false-positive rate drops 0.15 -> 0.05** using only 20 KB of portable assets and no weight updates. Directly aimed at Atlas's failure mode: tool-arg correctness, not tool-selection. Cross-model transferability claimed.

### 2.3 Reinforced Agent (arXiv:2604.27233, Apr 2026, ACL 2026)
Moves evaluation into the execution loop. A **specialized reviewer agent inspects each provisional tool call before execution**. +5.5% irrelevance detection, +7.1% on multi-turn (BFCL + tau2-Bench). o3-mini achieves a 3:1 benefit-to-risk ratio. Heavier than a classifier (it is another model), but Atlas already supports multi-model routing via the EP=2 path, and the reviewer can be a smaller fast model.

### 2.4 SideQuest - model-driven KV-cache management (arXiv:2602.22603, Feb 2026)
Uses the **reasoning model itself** to reason about token utility and emit eviction decisions as an auxiliary parallel task, so the management tokens don't pollute main memory. 65% peak-token reduction on agentic tasks with minimal accuracy loss, trained on 215 samples. Plausibly compatible with hybrid SSM-attention models because the auxiliary task is at the prompt level, not the KV-architecture level.

### 2.5 CodeComp - structural-aware KV compression for coding agents (arXiv:2604.10235, Apr 2026)
Attention-only KV compression baselines **systematically discard structurally critical tokens** (call sites, branch conditions, assignments). CodeComp uses a Code Property Graph (Joern) to mark span-level structural protection. Recovers most of full-context accuracy under aggressive compression and matches uncompressed patch-generation quality. Integrates into SGLang; the same hook would work in Atlas's KV manager. Directly relevant because opencode's wrong-filename failures correlate with eviction of earlier "this is the canonical name" tokens.

### 2.6 TriAxialKV - mixed-precision KV for agentic inference (arXiv:2605.17170, May 2026)
Three-axis tagging - temporal recency, modality, semantic role - feeds INT2/INT4 allocation under a fixed budget. On Qwen3-VL-32B-Thinking running an OSWorld computer-use agent it matches **BF16 KV accuracy at 4.5x larger cache and +30% throughput**. The semantic-role axis is the lever Atlas needs: tool schema and prior tool-call observations get protected from over-quantization, while filler/scratchpad turns get INT2.

### 2.7 KVSink - attention-sink preservation under KV quantization (arXiv:2508.04257, COLM 2025)
Mechanistic analysis of why FP8 KV quantization hurts long context: **attention sinks emerge beyond the initial token positions** and standard "Preserve-First-N" misses them. KVSink is a plug-and-play sink predictor with negligible overhead. Reduces reliance on 16-bit numerical outliers. Critical for Atlas because we have FP8 KV cache active, and the Atlas memory `project_qwen36_phase2b_softmax_expf.md` already shows FP8 KV regresses at deep layers L31-L39 from K/V magnitude outliers - exactly the population KVSink targets.

### 2.8 Continuum - KV-cache TTL for multi-turn agents (arXiv:2511.02230, Nov 2025)
Pinned KV cache with TTL chosen by reload-cost vs queue-delay tradeoff, plus program-level FCFS. **>8x improvement in average job completion time** on real agents (SWE-Bench, BFCL, OpenHand). Less about coherence directly, more about preserving the right prefix across opencode turns so the system prompt and tool catalog don't get re-tokenized noisily. GitHub: `Hanchenli/vllm-continuum`.

### 2.9 XGrammar-2 (arXiv:2601.04426, Jan 2026)
Successor to the XGrammar Atlas already runs. Two new primitives: **TagDispatch** (first-class tag-triggered structure switching) and **Cross-Grammar Cache** (substructure cache reuse across grammars). >6x faster compilation, near-zero end-to-end overhead. Lets Atlas swap from "tool-call opener" grammar to "per-tool argument schema" grammar without recompiling, which is what is currently missing in the tier-2 path validators.

### 2.10 Natural Language Tools (NLT, arXiv:2510.14453, Oct 2025)
Replace JSON tool-call output with natural-language tool selection followed by a deterministic JSON wrapper. **+18.4 pp accuracy, -70% output variance** across 10 models, 6,400 trials. Open-weight models gain the most. Bypasses the "model has to generate a JSON object literally as a token stream" problem. Heavy paradigm shift but the largest single-intervention number in the recent literature.

### 2.11 AdaDec - uncertainty-guided adaptive decoding (arXiv:2506.08980, Jun 2025)
**Pause-then-rerank at high-entropy positions only**. Detects high Shannon entropy at the current decode step, pauses, reconsiders the top-k logits, then continues. **+20.9 pp Pass@1 on HumanEval+/MBPP+/DevEval over greedy**. Training-free; thresholds are calibrated per-model on a held-out set. The "correct token is present but not top-ranked" finding is the exact mechanism behind Atlas's wrong-but-valid filenames: the right path was in top-5, model picked top-1.

### 2.12 ToolSpec - schema-aware speculative decoding (arXiv:2604.13519, Apr 2026)
FSM that **alternates deterministic schema-token filling with speculative generation only for variable fields**. Atlas already runs MTP speculative decoding, so this is a drop-in upgrade of the draft head to consume the tool schema as a side channel - keeps the throughput win Atlas has from MTP while killing draft tokens that violate the schema. Better fit for Atlas than EAGLE-3 because Atlas already has the MTP infra; ToolSpec changes the draft policy, not the verify path.

### 2.13 Component-Aware Self-Speculative Decoding for Hybrid Models (arXiv:2605.01106, May 2026)
First speculative-decode paper that explicitly targets **hybrid SSM+attention architectures** including Falcon-H1 and Qwen3.5 (interleaved Gated DeltaNet + softmax attention). The key insight: SSM pathways have O(1) sequence-length cost while attention has O(n), so the draft head should be SSM-biased. Qwen3.6-35B-A3B has the same hybrid topology (30 GDN + 10 attention) - this is the closest architectural match in the literature to Atlas's primary served model.

### 2.14 Steer Like the LLM + Spherical Steering (arXiv:2605.03907, arXiv:2602.08169)
Activation-steering follow-ons: "Steer Like the LLM" makes steering follow prompt position-distribution; "Spherical Steering" rotates along a geodesic toward truthful direction. Both training-free, stackable with TACT/ASA.

---

## 3. Why FP8 still matters and why the new work helps

The prior synthesis (research_multi_turn_arxiv.md section 2) established that FP8 + MoE + multi-turn jointly produce instability invisible to single-pass cosine. The 2025-2026 work adds three new mechanisms:

1. **KVSink** explains *which* tokens FP8 quantization breaks (post-initial sink tokens), giving Atlas a concrete protect-list.
2. **TriAxialKV** gives a budget-aware policy for keeping schema and prior-tool-call tokens at higher precision than scratchpad.
3. **CodeComp** gives a structural-importance signal for coding workloads specifically, complementing what plain attention scores would prune.

These three together address the FP8 KV-cache deep-layer regression (L31-L39) Atlas already sees in `project_qwen36_phase2b_softmax_expf.md`, without changing weight quantization.

---

## 4. Ranked top-5 interventions Atlas should consider

Ranked by (evidence x architectural fit x serving-side implementability). All are inference-time only.

| Rank | Intervention | Paper | "What we'd do" sketch |
|---|---|---|---|
| 1 | **Last-layer hallucination-gate classifier on each proposed tool call** | Internal Reps (2601.05214) | Train ~5 MB MLP on (last-layer-hidden, was-arg-correct?) pairs harvested from opencode logs; gate `<tool_call>` emission - if score < threshold, force re-sample at the args span. |
| 2 | **TACT activation steering along overthinking/overacting axes** | TACT (2605.05980) | Label N=500 opencode turns as overthinking/overacting/calibrated, fit the two drift axes on Qwen3.6 hidden states (per-decoder-layer linear probes), project drifted activations back at inference. Pure Atlas residual-stream hook. |
| 3 | **AdaDec pause-then-rerank at high-entropy tool-arg positions** | AdaDec (2506.08980) | At every decode step, compute Shannon entropy of next-token logits; if above per-model threshold AND we are inside a tool-call argument span (which Atlas's grammar already knows), pause one step, re-rank top-k with a tighter lookahead, then emit. |
| 4 | **CodeComp + KVSink hybrid KV protection** | CodeComp (2604.10235) + KVSink (2508.04257) | Mark filename / tool-id / system-prompt tokens with a "do-not-quantize-below-FP8-fine" flag at insertion time; predict post-initial attention sinks via KVSink at the same hook. Targets Atlas's measured L31-L39 deep-layer FP8 KV regression. |
| 5 | **XGrammar-2 argument-level schema enforcement** | XGrammar-2 (2601.04426) | Upgrade current EBNF (tool-call opener only) to per-tool argument schemas via TagDispatch; each tool name selects its argument grammar with the Cross-Grammar Cache amortizing compile cost. Keeps the existing grammar engine. |

**Honorable mentions** (cost or weaker direct evidence): ASA (2602.04935) = TACT for tool-calling specifically, natural follow-up if TACT lands; Reinforced Agent (2604.27233) needs a separate reviewer model; ToolSpec (2604.13519) natural next step once MTP draft policy is tunable; Component-Aware Self-Spec (2605.01106) best architectural fit but is a draft-head rewrite, not a serving hook.

**Explicitly NOT recommended**: AEPO (2510.14545), GTPO (2511.14846), Tool-R0 (2602.21320) - all require RL training. NLT (2510.14453) - paradigm shift incompatible with opencode MCP. EAGLE-3 - lower fit than ToolSpec given MTP already runs.

---

## 5. Bibliography (new since Apr 2025)

Hallucination measurement: 2601.06818 (AgentHallu), 2602.13320 (Information Fidelity / MCP martingale), 2601.05214 (Internal Reps gate), 2604.02547 (Behavioral Drivers), 2604.03515 (Inside the Scaffold).

Inference-time interventions: 2605.05980 (TACT), 2602.04935 (ASA), 2604.27233 (Reinforced Agent), 2602.22603 (SideQuest), 2604.10235 (CodeComp), 2605.17170 (TriAxialKV), 2508.04257 (KVSink, COLM 2025), 2511.02230 (Continuum), 2601.04426 (XGrammar-2), 2510.14453 (NLT), 2506.08980 (AdaDec), 2604.13519 (ToolSpec), 2605.01106 (Component-Aware Self-Spec Hybrid), 2605.03907 (Steer Like the LLM), 2602.08169 (Spherical Steering).

Out-of-scope (training required): 2510.14545 (AEPO), 2511.14846 (GTPO), 2602.21320 (Tool-R0).

Context engineering reference: 2510.04618 (ACE) - structured-bullet contexts, +10.6% agents, complementary to but not directly Atlas-applicable since opencode controls its own context format.

---

**Last updated**: 2026-05-26. Word count ~1850.
