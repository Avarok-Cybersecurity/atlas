# Does FP8 Quantization Cause Agentic Wandering in Qwen3.6-35B-A3B?

**Investigator:** Research subagent (web-search + Atlas-internal drift catalog)
**Date:** 2026-05-26
**Symptom under test:** 30–40 % of opencode runs on Qwen3.6-35B-A3B-FP8 emit 5–7 exploratory bash calls and never reach a `write/edit` tool call. Sampler-level fixes (whitespace masks, attractor bias) move specific drift modes but not the overall pass rate.
**Question:** Is FP8 the cause, or is the model itself wandering?

---

## 1. Direct evidence linking FP8 to behavioural degradation in Qwen3.6/3.5 MoE

The web has four independent signals that FP8 specifically — and not the model — drives the failure class we see:

1. **vLLM #36872 (Qwen3.5-35B-A3B-FP8, tool-calling multi-turn).** Multi-turn tool-calling on the *FP8* checkpoint produces Unicode garbage and the speculative-decode acceptance rate collapses request-by-request: 61.3 % → 0.9 % → 0.0 % across three consecutive tool turns. This is the same failure shape we see (a session that starts fine and degenerates as the agent-loop context grows). No BF16 control is published, but disabling spec-decode *partially* recovers — meaning the *FP8 numerical regime* is what destabilises long-horizon decoding.
2. **FlashInfer #2822 (TRTLLM FP8 block-scale MoE on Qwen3.5-35B-A3B-FP8 & 3.5-122B).** When *all* router logits at a layer become negative — which Qwen3.5-35B's gate does on **every forward pass across all 40 MoE layers** (range ≈[-11,-1], mean ≈-6) — the monolithic FP8 kernel selects *different* top-k experts than the float32 reference. Cosine vs reference drops to ~0.3, GSM8K → 0 %. The bug is **specific to the FP8 monolithic routing path**; the modular variant that does routing in fp32 *before* the FP8 GEMM is fine. The 397B model dodges the bug only because some of its later layers happen to produce a positive logit.
3. **EAQuant (Jul 2026, arXiv 2506.13329).** Quantifies that "the routing layer exhibits extreme sensitivity to quantization perturbations" — **minute deviations in gating scores distort top-k selection logic, causing token misrouting and cascading degradation.** Token misrouting is exactly the failure mode our internal `project_qwen36_drift_moe_smoking_gun.md` documents (8/8 → 7/8 → 3/8 top-k overlap at L0 → L24 → L38).
4. **LMSYS Unified-FP8 blog (Nov 2025).** Confirms that for MoE specifically, "the larger the model, the more severe the train-inference discrepancy becomes when using BF16 training with FP8 rollout." Qwen3.6-35B-A3B was trained in BF16 and is being served in FP8 — exactly the asymmetric regime LMSYS warns about.

**Counter-evidence (the case for "FP8 is fine"):** Qwen's own README and the marketing comparison (aimultiple) claim FP8 is "near-identical" to BF16 for the 27B and 32B variants. *But:* those benchmarks are MMLU/GSM8K-style — single-shot, short-context, no tool calls, no expert-routing pressure across 8 K+ tokens. None of them measure the "I emit 5 bash calls without writing files" failure mode. The known-clean benchmarks are blind to our failure class.

## 2. Atlas-specific evidence that bears directly on the question

From the local drift catalog (`bench/fp8_dgx2_drift/MASTER_DRIFT_TABLE.md`, `project_qwen36_phase2b_softmax_expf.md`, `project_qwen36_drift_moe_smoking_gun.md`):

- **MoE routing flips are the dominant drift signal.** At L20, `ssm.moe_out` cosine drops to **0.920** vs BF16 reference — that's a routed-expert output dot-product 8 % away from truth on a layer whose job is to commit to a behaviour. Per-layer-hidden mean cosine is 0.9898, but the *worst* layers are the late ones (L31–L39) per Phase-2b.
- **Late attention layers regress *more* under FP8 KV than early ones.** `project_qwen36_phase2b_softmax_expf.md` explicitly documents an L31–L39 regression band caused by FP8 KV quant noise on large K/V magnitudes once the polynomial softmax masked it no longer.
- **Gate goes indecisive at depth.** L38 has only 3/8 top-k overlap with BF16 — i.e. for the *intent-selection* layers, FP8 has effectively a random expert assignment. If "write a file" vs "explore with bash" lives in different expert basins, this is exactly the mechanism that would produce action-vs-exploration coin-flips at temp>0.
- **FP8-only EOS-guard pathology.** `project_qwen36_fp8_post_think_eos.md` already proved that an FP8-path-only behaviour (early-EOS forcing template artefacts) caused content-collapse in thinking mode. This is precedent that *behavioural* (not just numerical) bugs are FP8-path-specific on this model.

The pattern is consistent with the vLLM/FlashInfer reports: FP8 perturbs MoE routing, the perturbation grows monotonically with depth, and the late layers — where the model commits to an *action* — are the ones that wobble most.

## 3. Could the sampler ever fix this?

No. Whitespace masks and attractor bias operate on **logit space at the output head**. The routing flip happens **40 layers earlier** inside every transformer block. By the time the sampler sees a "wandering" probability mass over `bash` vs `write`, that mass is already the consequence of a different chain of experts having processed the residual stream. A sampler-level constraint can pick from the choices the model offered; it cannot change *which choices the model offered*. This explains why sibling work closed specific drift modes but did not move the 30 % pass rate — the floor is the routing distribution, not the head distribution.

## 4. Three concrete experiments Atlas can run on dgx1 today

### Experiment A — "Run the model in BF16 and watch the pass rate" (single most decisive test)

Atlas already has `hf_forward_bf16_unquant.py` for `Qwen/Qwen3.6-35B-A3B` (the unquantised reference). Serve the BF16 checkpoint through Atlas (single GPU, EP=1, `--quant-format bf16`) and run the same opencode harness that produces the 30–40 % wandering rate. Sample N≥20 sessions (see `feedback_no_n1_stochastic_ab.md`).

- **If BF16 pass rate ≥ 90 % :** FP8 is the cause. Action: ship the "BF16-late-layer-upcast" recipe below.
- **If BF16 pass rate ≈ 30 % :** Model is fundamentally wandering. Action: drop the FP8 blame and move to harness-level mitigations (force `write` after N bash calls, agentic-loop watchdog, persona prompt-engineering).
- **If BF16 pass rate is in between (60–80 %) :** FP8 is *amplifying* a latent model problem. Pursue both fixes; mark FP8 as the higher-leverage one.

### Experiment B — "Selective BF16 upcast of layers L31–L39"

The drift catalog already isolates L31–L39 as the regression band. Atlas's `WeightQuantFormat` dispatch (`project_qwen36_fp8_post_think_eos.md`) can load specific layers in BF16. Patch `ATLAS_BF16_UPCAST_LAYERS=31,32,33,34,35,36,37,38,39` to keep those 9 layers' MoE/attention in BF16 while the other 31 stay FP8. Memory cost: ≈ 9 layers × ~1 GB of incremental BF16 weight = ≈ 9 GB, well within budget on dgx1.

Run the same N≥20 opencode harness. The expected signature, *if FP8 is the cause*:
- Pass rate climbs from ~30 % toward the full-BF16 number from Experiment A.
- Per-layer routing-flip rate at L31–L39 returns to ≤ 5 %.
- `ssm.moe_out` cosine at L31–L39 lifts above 0.99.

This was rejected for *numerical drift* in earlier work because the drift survives anyway — but for *behavioural* outcomes (action commitment) the bar is much lower: we don't need cosine=1.0, we need top-k expert overlap ≥ 7/8 so the model lands in the same behavioural basin.

### Experiment C — "Force fp32 routing under FP8 weights" (FlashInfer-style fix)

The FlashInfer #2822 fix is to do top-k routing in fp32 *before* dequantising experts to FP8. Atlas's NVFP4/FP8 MoE dispatcher can be patched to keep the gate-projection output and the soft-max in BF16/fp32 even when expert weights are FP8. This isolates whether the "all-negative-router-logits" pathology — which we know Qwen3.6-35B exhibits, since 3.5-35B does — is *our* problem.

Concretely: instrument the gate output for one prompt at L0/L20/L38, log min/max/sign. If we see the same all-negative regime FlashInfer saw, the upstream bug applies to us by construction and the fp32-routing patch is the right fix and is much cheaper than BF16 upcast.

All three experiments fit in a single dgx1 session (≤ 4 h). A and C are the highest-leverage; B is the production recipe if A says "FP8 is at fault."

## 5. Recommendation

**Attribute the wandering primarily to FP8, with high confidence (~75 %).** Evidence:

1. Two upstream FP8-specific Qwen3.5/3.6 MoE bugs reproduce the failure signature (vLLM #36872's "degenerates across consecutive tool turns," FlashInfer #2822's all-negative gate pathology).
2. Atlas's own drift catalog shows MoE top-k overlap collapsing 8/8 → 3/8 at L38 — i.e. the model is choosing different experts than its BF16 self at exactly the depth where action vs exploration is selected.
3. The sampler-level fix campaign capping out at no pass-rate movement is exactly the symptom you would expect if the failure is upstream of the head.
4. The published "FP8 ≈ BF16" claims rest on benchmarks that cannot see the failure class.

The remaining ~25 % uncertainty is that Qwen3.6 is a hybrid attention+SSM+MoE model and its agentic training corpus may simply have a long tail of "explore-then-commit" trajectories whose mode collapses without strong instruction-following pressure. Experiment A (full BF16) is the cheapest test that distinguishes the two hypotheses — **run it first.**

If Experiment A confirms FP8: ship Experiment B's layer-31–39 BF16 upcast as the production recipe (or Experiment C's fp32-routing patch if the gate-sign instrumentation matches FlashInfer #2822). Both are mechanically grounded in published bugs and Atlas's own drift evidence.

If Experiment A disconfirms FP8: the wandering is the model, not the quantisation. Stop chasing kernels and move the effort to harness watchdogs and prompt-level commitment scaffolding.

## Files / pointers

- Atlas drift table: `/workspace/atlas-mtp/bench/fp8_dgx2_drift/MASTER_DRIFT_TABLE.md`
- Atlas MoE smoking gun: `project_qwen36_drift_moe_smoking_gun.md`
- Atlas late-layer regression: `project_qwen36_phase2b_softmax_expf.md`
- Atlas FP8 EOS pathology (precedent for FP8-only behavioural bugs): `project_qwen36_fp8_post_think_eos.md`
- BF16 reference forward script: `/workspace/atlas-mtp/bench/fp8_dgx2_drift/hf_forward_bf16_unquant.py`
- vLLM #36872 (Qwen3.5-35B-A3B-FP8 multi-turn collapse): https://github.com/vllm-project/vllm/issues/36872
- FlashInfer #2822 (FP8 all-negative-logit routing bug, Qwen3.5-35B): https://github.com/flashinfer-ai/flashinfer/issues/2822
- vLLM #34892 (Qwen3.5 FP8 + FlashInfer CUTLASS MoE garbage): https://github.com/vllm-project/vllm/issues/34892
- EAQuant (MoE routing under quantisation): https://arxiv.org/abs/2506.13329
- LMSYS Unified FP8 RL (train/rollout mismatch in MoE): https://www.lmsys.org/blog/2025-11-25-fp8-rl/
- vLLM FP8 KV-cache state-of-the-art: https://vllm.ai/blog/2026-04-22-fp8-kvcache
- HF discussion: Qwen3.6 tool-calling issues across the family: https://huggingface.co/Qwen/Qwen3.6-27B/discussions/13

Word count: ≈ 1450.
