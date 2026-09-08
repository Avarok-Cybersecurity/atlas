# NVIDIA Qwen3.8 NVFP4 / MTP handoff — 2026-09-05

NVIDIA's checkpoint now loads and runs MTP. **MTP is still experimental:**
three plain-text greedy prompts match serial completely, including reasoning;
the tool request emits the same `get_weather({"city":"Paris"})` call but
has different reasoning. Do not claim complete tool/thinking parity or an
agentic pass. The user requested a wrap-up before the longer agentic run.

## Runtime and checkpoint

- Runtime source: `98595757e` (fixture/launcher commits change no runtime code).
- Native binary: `/home/ms/atlas/target/release/spark`, SHA-256
  `4dd28c0ea88e13b507ecee060b431100524f98bc579df237af06318c427bab9a`.
- Checkpoint: `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
  `fab0aecb760cec45227f6656abcaafa11abca87a`, downloaded under
  `/tank/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/`.
- Host: GB10 `dgx-00`, CUDA 13.0, driver 580.173.02; ASR co-tenant remained.
- Full flags, prompts, responses, tests and memory samples:
  [nvidia-mtp-results.json](nvidia-mtp-results.json).

## Changes

1. `33bb7bd29`: wire the existing FP8 gather and cached slot scales into PLE.
   The NVIDIA table is FP8 E4M3 with one BF16 scale; the prior loader refused
   it. BF16 and EXL3 gather branches retain their arithmetic.
2. `7b226ec73`: fix **shared quantization detection**. Loading FP8 MTP tensors
   previously changed the main model's detected format from NVFP4 to FP8.
   Positive main-expert NVFP4 triplets now precede fallback FP8 sniffing;
   explicit configuration retains precedence. Draft experts still take the
   existing FP8 → BF16 → runtime NVFP4 conversion path.
3. `98595757e`: extend `ATLAS_VERIFY_ROW_PROJ=1` to BF16 GDN output projections.
   Verification previously used GEMM while serial used GEMV. Per-row replay
   now uses the serial reduction. Other quantization branches are unchanged.
4. Fixture extraction and the exact NVIDIA launch profile are committed in
   [nvidia_mtp_fixtures.py](nvidia_mtp_fixtures.py) and
   [serve_nvidia_mtp.sh](serve_nvidia_mtp.sh).

## Verification

- Model unit suite: **778 passed, 18 ignored**. Model clippy and formatting of
  changed Rust files passed. Native CUDA server rebuilt; executable runs
  without `LD_LIBRARY_PATH`.
- PLE GPU tests: all finite E4M3 encodings and 16 real checkpoint rows match
  independent BF16 oracles exactly, with segmented offsets, reordered/repeated
  IDs and output guards. PyTorch separately confirmed the checkpoint oracle.
- GDN GPU tests: real NVIDIA and synthetic weights match serial BF16 bytes at
  K=2/3/4 after the fix. Old GEMM is a failing negative control.
- Final server comparison: 3/3 arithmetic/prose/code choices match exactly.
  Tool function, arguments and finish reason match; full choice does not.
  Assigned tool IDs are excluded from comparison.
- Actual accept logs confirm speculation engages during thinking. Only two
  drafts and one active sequence were exercised in final serving checks.
- No final agentic run, long MTP throughput comparison, wider-draft serving
  sweep, NVFP4 concurrency test, Docker rebuild or full model matrix was run.
  Earlier EXL3 agentic/concurrency results do not validate this checkpoint.

The final serial coding baseline is **preliminary, single-harness**: three
512-token LRU-code repeats at temperature 0 and low thinking effort observed
median server generation rate **20.782 tok/s**, median request wall **25.388 s**.
There is no matching final MTP throughput measurement and no speedup claim.

## Launch and memory

The local `/home/ms/run-atlas-pr834-tui.sh` now points to NVIDIA with two drafts;
`/home/ms/run-atlas-nvidia-qwen38-tui.sh` is the explicit NVIDIA alias. The prior
EXL3 launcher is preserved at `/home/ms/run-atlas-pr834-exl3-tui.sh`.
Test servers have been stopped so the TUI can be launched manually.

```bash
ATLAS_SPARK_BIN=/home/ms/atlas/target/release/spark \
QWEN4EXP_PATH=/tank/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fab0aecb760cec45227f6656abcaafa11abca87a \
MTP_DRAFTS=2 bash bench/qwen4_exp/serve_nvidia_mtp.sh
```

The profile is C=1, context 32768, BF16 KV, utilization 0.71, prefill chunks
2048, PLE max tokens 3072, four SSM snapshot slots and 1024² vision pixels.
It keeps `ATLAS_VERIFY_ROW_PROJ=1`, default row-exact FFN and speculation during
thinking. Do not copy `ATLAS_NO_VERIFY_ROW_FFN=1` from the EXL3 profile.
Utilization 0.68 was insufficient for MTP's inference reserve plus KV and
correctly refused. The earlier large-scratch profile left little host headroom.
Memory figures in the raw record cover short requests, not two full contexts.

## Next work

1. Investigate the remaining forced-tool reasoning difference. Serial emits
   a tool call inside thinking, then simulates a tool response before F7 hoists
   the call; MTP returns the same call without that reasoning. Determine whether
   the first divergence is logits, parser/finish timing, or verify acceptance.
   `ATLAS_LOGIT_PROBE=1` provides existing hidden/logit probes. Do not assume
   the checkpoint is at fault.
2. Run `agentic-webserver` with **both**
   `ATLAS_AGENTIC_SAMPLING=model-card` and `ATLAS_AGENTIC_PRESERVE_THINKING=1`.
   Keep served reasoning effort low and `ATLAS_DFLASH_SPEC_THINK=1` deliberate.
3. After correctness, obtain ≥3 same-profile serial/MTP coding repeats before
   claiming a speedup. The conservative verifier repeats BF16 GEMVs per row;
   byte-identical batched GEMV is a possible future optimization to validate.
4. Revisit concurrency only with a fresh memory budget; EXL3 C=2 does not carry
   over. Existing broader PR hygiene failures are recorded in MTP_PERFORMANCE.md.

GPU fixtures can be reproduced with `python3 bench/qwen4_exp/nvidia_mtp_fixtures.py
<SNAPSHOT> <OUTPUT>`. Set `ATLAS_PLE_FP8_FIXTURE=<OUTPUT>/ple-fp8-checkpoint.json`
and `ATLAS_GDN_OUT_WEIGHT=<OUTPUT>/gdn-out-weight.bin`, then run CUDA-built model
unit tests filtered to `ple_fp8` and `bf16_verify_out` with `--ignored
--test-threads=1`. Local detailed artifacts remain under `/tmp/atlas-nvidia-mtp`.
