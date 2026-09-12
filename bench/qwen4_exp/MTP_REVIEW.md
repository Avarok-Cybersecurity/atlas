# Native EXL3 MTP review — 2026-09-05

Follow-up: [wider drafts, live grammar, and thinking validation](MTP_WIDE_REVIEW.md).

Branch: `wip/exl3-mtp-batched`, starting at `283d780d1`.
Hardware: GB10, native EXL3 checkpoint
`/tank/exl3-ckpt/qwen38-flash-next-4.05bpw`, BF16 KV,
32,768 context, one sequence, `reasoning_effort=low`, one draft (K=2).

## Findings and changes

The last commit's claim that exact verification and batching are mutually
exclusive was too strong. Two changes retain decode arithmetic while sharing
launches:

* HC staging, activation and mixing are row-independent. Batch those stages,
  keeping each cuBLAS projection at M=1. This saves three launches per extra
  row at each HC site without changing output bytes.
* EXL3 MoE batching changed both routing and expert reduction geometry. Use
  decode GEMV/top-k per row, then retain the single-token expert kernel shape
  and split-K grid while scheduling additional expert slots in waves inside
  one cooperative launch. The resident grid remains bounded by the SM count.

The remaining greedy discrepancy was not exclusively a layer numerical bug.
The K=2 verifier sampled both rows before emitting either, using stale penalty
history and thinking/tool state for the bonus. It also processed rejected
suffixes and used a different equal-logit tie policy from serial CPU sampling.
K=2 now copies both logit rows once and reuses the serial processor: sample
row 0, broadcast acceptance, emit row 0, and only then sample an accepted bonus.
Rejected or finished suffixes are never processed. This also preserves processed
logprobs, request sampling parameters and per-output-position seeds.

The shared Verify penalty builder now retains request `logit_bias`, including
in fast-path eligibility checks. DFlash's existing sampling order and tie policy
are unchanged. Adaptive sampling and wider K=3/K=4 scheduler state sequencing
are outside this change; the end-to-end configuration validated here is K=2.

Shared implementation changes to review: `exl3_mgemm` gained an explicit
optional replay-group argument; ordinary callers pass `None`. The serial logits
processor lost an unused model argument. No dependency or kernel source changes
are required for these fixes.

## Measurements

Machine-readable measurements: [mtp-review-results.json](mtp-review-results.json).

Five fixed greedy prompts, 250-token cap, probes disabled:

| Arm | Median server tokens/s | Output comparison |
| --- | ---: | --- |
| Original serial | 11.89 | Reference; post-think GPU argmax enabled |
| Fixed K=2, per-row experts | 10.65 | All five reasoning traces match serial |
| Fixed K=2, batched experts with stable geometry | 13.48 | 5/5 complete messages byte-identical to per-row experts |

The fast arm is 26.6% faster than the fixed per-row reference and 13.4% faster
than the original serial median. Serial's post-thinking GPU argmax has a
different tie policy from its CPU sampler, so raw output equality against that
shortcut is a separate issue. Use `ATLAS_NO_THINKENDED_GPU_ARGMAX=1` for a
controlled serial numerical comparison. The GPU tie kernels were not changed.
With that shortcut disabled, the fast K=2 arm matches **5/5 complete serial
messages byte-for-byte**, including both reasoning and answer text.

Model-card `agentic-webserver`, one smoke iteration per arm:

| Arm | Webserver / directions | Turns | Total wall | Effective tokens/s |
| --- | --- | ---: | ---: | ---: |
| Serial | 1/1, 1/1 | 11 | 347.94 s | 7.44 |
| Fast K=2 | 1/1, 1/1 | 8 | 244.12 s | 8.37 |

Both agents wrote the project and tests, ran tests and the server, obtained
`pong`, and tore the server down. These are stochastic smoke tests, not a
statistically established speed gate. Fewer turns explain part of the wall-time
improvement. The harness's `decode_tps` includes request/prefill overhead;
server per-response decode rates are the metric used in the greedy table.
Existing unrelated GPU services remained resident for both arms; the hardware
precheck was explicitly bypassed. No other model's performance is inferred.

## Reproduction

Build the native target using the repository's CUDA/CUTLASS/FlashInfer setup:

```bash
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next \
ATLAS_TARGET_QUANT=nvfp4 cargo build --release -p spark-server --bin spark
```

Alongside the existing native EXL3 serve environment, enable:

```bash
export ATLAS_QWEN4EXP_MTP=1 ATLAS_QWEN4EXP_MTP_VERIFY=1
export ATLAS_DFLASH_SPEC_THINK=1 ATLAS_QWEN4EXP_MTP_HC_BATCHED=1
export ATLAS_VERIFY_EXL3_ROW_ROUTER=1 ATLAS_VERIFY_EXL3_STABLE_GRID=1
export ATLAS_NO_VERIFY_ROW_FFN=1
# Serve arguments: --speculative --num-drafts 1 --mtp-gate force
```

The last three flags are a coupled experimental configuration: the FFN override
allows batching, and the router/grid flags preserve its decode arithmetic.
Keep the other row-exact legs enabled. These flags are opt-in; serial decode
and other models keep their existing dispatch. For the per-row reference,
omit all three flags.

Run the agentic workload with the model-card override; otherwise the harness
pins greedy sampling:

```bash
ATLAS_AGENTIC_SAMPLING=model-card target/release/spark benchmark run \
  agentic-webserver --yes --url http://127.0.0.1:8892 \
  --model qwen4exp-exl3 --param iterations=1 --format json
```

## Regression coverage

* Real-weight HC GPU test: exact hidden and injection bytes for attention and
  MLP sites at K=2/3/4. The original M=K projections fail this test.
* Routed-expert GPU test: 18 fixtures, 138,240 BF16 elements, zero differences
  from serial. Previous batching differs in every fixture: 30,149 elements.
  Covers 4/5/6-bit weights, top-k 3/10, and K=2/3/4, with finite/nonzero checks.
* Pure grid tests check serial split geometry, cooperative residency, slot
  coverage and invalid dimensions across 600 configurations.
* K=2 behavioral tests exercise the actual sampler and emission code for
  presence penalties, logprobs, thinking-to-tool transition, equal-logit ties,
  and rejected/finished suffixes. Request-bias tests cover greedy and sampled
  temperatures plus tool-opener bias rules.

```bash
ATLAS_HC_TEST_DATA=/tank/atlas-testdata/qwen4exp_hc \
  cargo test --release -p spark-model hc_verify_batched_stages_match_serial_bytes \
  -- --ignored
cargo build --release -p spark-model --features gpu-examples \
  --example exl3_native_parity
EXL3_VERIFY_GRID_ONLY=1 target/release/examples/exl3_native_parity
```

Validation logs and complete benchmark responses for this session are under
`/tmp/atlas-mtp-review/`. The final release build, CUDA workspace clippy and
server suite passed (2,346 passed, 12 ignored; unset `NO_COLOR` for the TUI
color assertions). The prescribed
`--all-features` command fails on Apple-only `objc2` on Linux. Existing branch
formatting, vendor license-header and EXL3-symbol typo-check failures remain;
the change does not claim those repository-wide gates passed.
