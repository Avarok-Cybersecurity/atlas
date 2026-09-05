# Wider native EXL3 MTP review — 2026-09-05

**Current status:** rebased onto `avarok/main` at `8682329cc` (rebased tip `0d099342e`). Runtime/kernel/dependency trees are unchanged and the rebased release build succeeds. All three widths match 15/15 complete greedy responses. **Agentic profile correction:** the completed two/three-draft runs omitted `ATLAS_AGENTIC_PRESERVE_THINKING=1`; they are compatibility evidence only. Requested preserve-thinking agentic reruns and direct thinking-speculation on/off parity control are pending.

This follow-up to [MTP_REVIEW.md](MTP_REVIEW.md) validates one, two and three drafts on the single-sequence, batched-highway Qwen3.8-Flash-Next EXL3 path. Performance observations are preliminary, single-harness results; the model-card agentic runs are single-shot sanity checks.

## Changes

- K3/K4 target sampling uses the serial sampler and real token emission between rows. Rejected or finished suffixes do not change penalty history, RNG position, grammar or thinking state.
- Qwen drafts start from the accepted target highway row. Chained drafts retain the preceding draft's output in the private arena. Full acceptance consumes the pending auxiliary-state span.
- The eligible single-sequence Qwen tool path retains up to three drafts. Draft zero uses the current grammar mask; suffix candidates are checked by the target against live grammar state after each emission. DFlash and concurrent grammar requests retain their previous restrictions.
- Four-row GDN FFN verification now uses native decode routing. It previously fell into grouped prefill and bypassed the stable expert grid; two greedy reasoning traces diverged. A per-row FFN bisect restored both, and the corrected batched route then restored both too.
- The completion event now follows auxiliary PLE restoration, and context-limit checks see committed rows rather than the entire speculative width.
- Startup rejects active Qwen verification without `ATLAS_QWEN4EXP_MTP_HC_BATCHED=1`: the old fallback overwrites the earlier logits/highway rows needed by acceptance-aware sampling.

Shared code touched: model factory activation validation and single-sequence MTP scheduler dispatch. No new cross-model trait or kernel interface was introduced.

## Validation

Current-source server tests: 2,361 passed, 12 ignored. The real XGrammar tests cover malformed suffix correction, transitions out of thinking, EOS and early length termination. Model tests cover row addressing, masked draft sampling and the startup configuration guard. CUDA-feature workspace clippy passes. The GB10 HC byte-parity regression passes both attention and MLP sites at K2/K3/K4.

Workspace-wide formatting, license and typo checks still report pre-existing branch issues; changed code is checked separately. No Docker image was rebuilt or published in this follow-up. Numeric and behavioral validation uses the native release binary on GB10.

## Measurement scope

The complete fingerprint, raw request usage, acceptance logs and benchmark run records accompany the results. All compared widths use the same binary, checkpoint and serve settings, varying `--num-drafts`. Three-draft validation had two additional diagnostic warmup prompts before the recorded repeats; throughput summaries are observations rather than a strict speedup claim. Prefix caching is disabled, BF16 KV is used, context is 32,768 and concurrency is one. The existing ASR GPU process remains resident. Short prompts use temperature zero and three repeats; agentic requests omit sampling overrides so MODEL.toml controls model-card sampling, with low reasoning effort.

An earlier configured-two-draft agentic run actually used one draft because the old grammar clamp was active. It passed compatibility, but is excluded from wider-draft evidence. Final acceptance telemetry must show K3/K4 execution during tool requests.

## Vendor review

See [EXL3_VENDOR_REVIEW.md](EXL3_VENDOR_REVIEW.md). Core reference math sources already match upstream v1.4.6. A standalone cooperative graph capture/replay test passes on GB10; full EXL3 capture is still unverified. Fusing routing and activation staging is another bounded candidate. No kernel speedup is claimed from this review.

## Completed wider-draft results

Both widths match all five complete greedy responses, including reasoning, in three repeats (15/15 each). The fixed weather-tool request also returns the expected Paris call. Thinking speculation is enabled via `ATLAS_DFLASH_SPEC_THINK=1`; this is the current flag name. These output checks are scoped to the tested prompts, not a universal logit-byte identity claim.

| Drafts | Model-card agentic | Turns | Run record |
| --- | --- | ---: | --- |
| 2 | Pass: webserver and directions | 14 | `run-1788622909880428198` |
| 3 | Pass: webserver and directions | 19 | `run-1788622181031575941` |

**Profile limitation:** these recorded agentic runs had model-card sampling enabled but preserve-thinking disabled. Their turn counts must not be used to characterize the requested preserve-thinking workload.

Both agents wrote the project/tests, ran them, launched and curled the server, and tore it down. The three-draft run crossed 8K context. These are single-shot stochastic sanity checks, not comparative speed gates. Raw records, output choices, per-request usage and complete configuration fingerprint: [mtp-wide-results.json](mtp-wide-results.json).

Local TUI launcher: `/home/ms/run-atlas-pr834-tui.sh`; use `MTP_DRAFTS=2` or `MTP_DRAFTS=3`. Default remains one draft. The launcher enables thinking speculation and the CPU post-thinking sampling policy used in the parity checks.

## Measurement lesson

Pin `ATLAS_AGENTIC_SAMPLING=model-card` and `ATLAS_AGENTIC_PRESERVE_THINKING=1` independently in every intended agentic sanity command and record both in its fingerprint. `ATLAS_DFLASH_SPEC_THINK=1` controls speculation during the current reasoning span; it does not preserve prior turns' reasoning. Verify the actual harness process environment instead of inferring this setting from serve flags.

Rebase details: [MTP_REBASE.md](MTP_REBASE.md).
