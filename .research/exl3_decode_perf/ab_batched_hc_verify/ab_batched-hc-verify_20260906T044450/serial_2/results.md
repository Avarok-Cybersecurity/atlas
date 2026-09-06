# Concurrency sweep — 2026-09-06T05:18:23-04:00 arm=serial_2

Fingerprint: `/home/ms/atlas/.claude/worktrees/wf_a8fa8242-651-7/.research/exl3_decode_perf/ab_batched-hc-verify_20260906T044450/20260906_051115_concurrency_sweep_serial_2/fingerprint.txt`. Per-stream decode tok/s = (completion_tokens-1)/decode wall;
aggregate decode tok/s = sum tokens / (max last chunk - min first chunk); the incl.-TTFT column divides by the cell wall.
Host memory from /proc/meminfo; nvidia-smi memory is [N/A] on GB10 (unified memory) — see mem_samples.csv.

## arm: short

FINGERPRINT port=8899 model=qwen3.8-flash-next-exl3 label=short prompt_tokens~2000 max_tokens=300 temp=0 effort=low stream=1 prompt_class=salted-log+code-task concurrency=[1, 2, 4] repeats=3 salt=4242 abort_avail_gb=8.0 abort_swap_delta_gb=4.0 date=2026-09-06T05:12:08

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| short | 1 | 0 | 1/1 | [27.15] | 27.15 | 27.24 | 20.41 | [3.6] | [2025] | 23.2 | 0.0 |
| short | 1 | 1 | 1/1 | [27.5] | 27.5 | 27.59 | 20.83 | [3.5] | [2034] | 23.2 | 0.0 |
| short | 1 | 2 | 1/1 | [26.72] | 26.72 | 26.81 | 20.46 | [3.4] | [2018] | 23.1 | -0.0 |
| short | 2 | 0 | 2/2 | [11.81, 13.97] | 12.89 | 23.7 | 20.83 | [3.5, 6.9] | [2023] | 22.5 | -0.0 |
| short | 2 | 1 | 2/2 | [12.3, 13.84] | 13.07 | 23.96 | 21.03 | [3.4, 6.9] | [2020] | 22.4 | -0.0 |
| short | 2 | 2 | 2/2 | [11.9, 14.26] | 13.08 | 23.88 | 20.98 | [3.4, 6.9] | [2048] | 22.4 | -0.0 |
| short | 4 | 0 | 4/4 | [5.48, 6.7, 6.24, 6.03] | 6.14 | 21.86 | 20.56 | [3.4, 13.7, 10.3, 6.9] | [2031] | 21.3 | -0.0 |
| short | 4 | 1 | 4/4 | [6.43, 5.57, 7.03, 6.02] | 6.23 | 22.34 | 20.99 | [10.4, 3.4, 13.7, 6.8] | [2007] | 21.2 | -0.0 |
| short | 4 | 2 | 4/4 | [6.38, 7.36, 5.59, 6.01] | 6.2 | 22.38 | 21.03 | [10.2, 13.5, 3.4, 6.9] | [2020] | 21.1 | -0.0 |

## mHC batched verify liveness / declines (serve.log)

    2026-09-06T09:12:03.424740Z  INFO spark_model::model::trait_impl::verify_hc_multi_gate: mHC batched verify: DISARMED by ATLAS_NO_MTP_HC_BATCH_VERIFY
    2026-09-06T09:12:58.975598Z  INFO spark::scheduler::verify_hc_multi_step: mHC batched verify gate: arm=KILLED(ATLAS_NO_MTP_HC_BATCH_VERIFY) n_verify=2 row_cap=16 shared_dense=true shared_head=true shared_hc=false

## warm cell pass 1 (512-token prompts, 128 tokens; pass 2 = prefix-cached)

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| warm1 | 1 | 0 | 1/1 | [28.52] | 28.52 | 28.74 | 22.57 | [1.1] | [548] | 22.6 | 0.0 |
| warm1 | 2 | 0 | 2/2 | [15.0, 12.85] | 13.93 | 25.89 | 23.36 | [2.2, 1.1] | [552] | 22.1 | -0.0 |
| warm1 | 4 | 0 | 4/4 | [6.65, 6.45, 6.96, 7.45] | 6.81 | 24.95 | 23.55 | [2.3, 1.1, 3.4, 4.5] | [553] | 21.1 | -0.0 |

## warm cell pass 2 (512-token prompts, 128 tokens; pass 2 = prefix-cached)

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| warm2 | 1 | 0 | 1/1 | [28.61] | 28.61 | 28.83 | 26.9 | [0.2] | [548] | 23.1 | 0.0 |
| warm2 | 2 | 0 | 2/2 | [14.84, 14.51] | 14.68 | 28.51 | 27.79 | [0.2, 0.5] | [552] | 22.1 | 0.0 |
| warm2 | 4 | 0 | 4/4 | [7.3, 7.46, 7.4, 7.35] | 7.38 | 28.59 | 28.22 | [0.2, 0.7, 1.0, 0.4] | [553] | 21.1 | 0.0 |

## Host memory over the whole run (boot included)

samples=85 min_MemAvailable_GB=21 max_swap_used_GB=1
