# Device-side speculative acceptance: attribution

Row `mtp_device_accept` (off on every target; lane L12). The MTP verify path picks each
verify position on the device for rows the scheduler's planner accepts (no grammar, no `logit_bias`, no
live LZ or DRY penalty, no stateful thinking stage), instead of copying `[K, vocab]` BF16 logits to the
host and running the pre-sample pipeline there.

## Upstream

- Project: Cinference, https://github.com/satellitedown/cinference, commit `383e5db`, itself derived
  from NInfer by Neroued and contributors, https://github.com/Neroued/ninfer.
- License: Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). Cinference's NOTICE paragraph is
  reproduced in `NOTICE`.
- Files and lines translated (not copied):
  - `src/ops/kernel/speculative_round.cuh:323-455`, `speculative_accept_greedy_drafts_kernel`: the
    sampled branch's acceptance of a one-hot draft `d` with probability `p(d)` (`u < p(d)`), the
    residual pick with `d` excluded on the first rejection, and the bonus pick from the column after the
    last draft.
  - `src/ops/kernel/sampling_device.cuh:79-114`: `sampling_splitmix64` and `sampling_uniform` (a
    24-bit uniform that is a pure function of seed, position and purpose, so it is the same on every
    replay), and the candidate order `sampling_better` (higher value first, lower index on a tie).
  - `src/ops/kernel/sampling_device.cuh:401-438`: `sampling_pick_from_support`, the inverse-CDF pick
    over a normalized support with one excluded token.

## Atlas files

- `kernels/ada/common/spec_accept_sample_ada.cu`: the kernel (header names the source lines).
- `crates/spark-model/src/port/spec_accept/reference.rs`: the CPU model of the kernel (same header).
- Atlas-native, no upstream code: `crates/spark-model/src/port/spec_accept/{mod.rs, blob.rs, launch.rs,
  metrics.rs, reference_tests.rs}`, `crates/spark-model/examples/spec_accept_oracle.rs`,
  `crates/spark-model/src/layers/ops/port_rows/mtp_device_accept.rs`,
  `crates/spark-server/src/scheduler/verify_pipeline_helper/{device_accept.rs, device_accept/plan.rs,
  device_accept_tests.rs}` and the hook lines in `verify_pipeline_helper.rs`.

## Changes for Atlas

- Sampling semantics re-derived for Atlas `SamplingParams` as `spark_runtime::sampler` applies them:
  finite logits only, `s = v / T`, top-k, then `w >= min_p * w_max`, then the inclusive cumulative
  `w / sum >= top_p`, in the host's sequential f32 order. Upstream caps every support at 20 candidates;
  Atlas has no cap, so a top-k up to 64 is exact and an uncapped top-p or min-p is served only when its
  support closes within 64 candidates (otherwise the row reports `support` and keeps the host
  pipeline).
- Exact top-k by radix select over monotone order keys with a lowest-index tie rule, instead of
  upstream's per-thread top-20 lists and serial merge.
- The host pre-sample pipeline (the `min_tokens` EOS mask, the think and tool masks, the tool-opener
  suppression, repetition, presence and frequency penalties, the A4 floor bias) is applied from a sparse
  edit list with explicit round-to-nearest intrinsics, so greedy picks equal the host's
  `argmax_first_wins_f32` bit for bit; upstream applies presence and frequency penalties from a count
  table only.
- Two regimes per span (the step-start state and the state after a `</think>` pick inside thinking),
  mirroring Atlas's verify pick loop.
- One CTA per (position, regime) job over Atlas's `[K, vocab]` verify rows; 64-bit absolute positions;
  the RNG key is the request seed or a fresh host key per call.
- Greedy picks are bit-identical to the host; sampled picks are distribution-exact, not bit-identical
  because the RNG stream differs from the host sampler's `StdRng`.

## Verification

- CPU: `crates/spark-model/src/port/spec_accept/reference_tests.rs` (edit arithmetic against
  `apply_penalties_and_bias` bit for bit; chi-square over 10^5 draws per configuration against the exact
  support and against the host sampler; support closure and fallbacks; blob round trip) and
  `crates/spark-server/src/scheduler/verify_pipeline_helper/device_accept_tests.rs` (greedy spans equal
  `pick_positions_from_host` in every accepted pipeline regime; sampled spans match its distribution).
- GPU (deferred): `examples/spec_accept_oracle.rs`, kernel against the CPU model on fixed logits at the
  Qwen3.5 vocabulary.
