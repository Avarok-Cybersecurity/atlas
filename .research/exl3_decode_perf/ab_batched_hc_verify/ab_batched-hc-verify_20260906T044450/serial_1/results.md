# Concurrency sweep — 2026-09-06T04:51:59-04:00 arm=serial_1

Fingerprint: `/home/ms/atlas/.claude/worktrees/wf_a8fa8242-651-7/.research/exl3_decode_perf/ab_batched-hc-verify_20260906T044450/20260906_044451_concurrency_sweep_serial_1/fingerprint.txt`. Per-stream decode tok/s = (completion_tokens-1)/decode wall;
aggregate decode tok/s = sum tokens / (max last chunk - min first chunk); the incl.-TTFT column divides by the cell wall.
Host memory from /proc/meminfo; nvidia-smi memory is [N/A] on GB10 (unified memory) — see mem_samples.csv.

## arm: short

FINGERPRINT port=8899 model=qwen3.8-flash-next-exl3 label=short prompt_tokens~2000 max_tokens=300 temp=0 effort=low stream=1 prompt_class=salted-log+code-task concurrency=[1, 2, 4] repeats=3 salt=4242 abort_avail_gb=8.0 abort_swap_delta_gb=4.0 date=2026-09-06T04:45:44

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| short | 1 | 0 | 1/1 | [27.13] | 27.13 | 27.22 | 20.39 | [3.7] | [2025] | 22.2 | 0.0 |
| short | 1 | 1 | 1/1 | [27.37] | 27.37 | 27.46 | 20.74 | [3.5] | [2034] | 22.2 | -0.0 |
| short | 1 | 2 | 1/1 | [26.74] | 26.74 | 26.83 | 20.46 | [3.5] | [2018] | 22.1 | -0.0 |
| short | 2 | 0 | 2/2 | [13.64, 11.91] | 12.78 | 23.65 | 20.79 | [6.9, 3.5] | [2023] | 21.6 | -0.0 |
| short | 2 | 1 | 2/2 | [14.18, 11.99] | 13.09 | 24.07 | 21.14 | [6.9, 3.4] | [2020] | 21.5 | -0.0 |
| short | 2 | 2 | 2/2 | [14.18, 12.12] | 13.15 | 24.32 | 21.31 | [6.9, 3.4] | [2048] | 21.4 | -0.0 |
| short | 4 | 0 | 4/4 | [5.46, 6.71, 5.82, 6.47] | 6.14 | 21.82 | 20.5 | [3.5, 13.9, 7.0, 10.5] | [2031] | 20.1 | -0.0 |
| short | 4 | 1 | 4/4 | [5.61, 6.83, 6.35, 5.93] | 6.14 | 22.15 | 20.81 | [3.5, 13.8, 10.4, 6.9] | [2007] | 20.1 | -0.0 |
| short | 4 | 2 | 4/4 | [5.61, 6.14, 6.46, 6.93] | 6.3 | 22.43 | 21.08 | [3.4, 6.8, 10.2, 13.8] | [2020] | 20.5 | -0.0 |

## mHC batched verify liveness / declines (serve.log)

    2026-09-06T08:45:38.973509Z  INFO spark_model::model::trait_impl::verify_hc_multi_gate: mHC batched verify: DISARMED by ATLAS_NO_MTP_HC_BATCH_VERIFY
    2026-09-06T08:46:35.138816Z  INFO spark::scheduler::verify_hc_multi_step: mHC batched verify gate: arm=KILLED(ATLAS_NO_MTP_HC_BATCH_VERIFY) n_verify=2 row_cap=16 shared_dense=true shared_head=true shared_hc=false

## warm cell pass 1 (512-token prompts, 128 tokens; pass 2 = prefix-cached)

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| warm1 | 1 | 0 | 1/1 | [28.6] | 28.6 | 28.83 | 22.64 | [1.1] | [548] | 22.6 | 0.0 |
| warm1 | 2 | 0 | 2/2 | [13.64, 14.61] | 14.12 | 26.09 | 23.4 | [1.1, 2.2] | [552] | 22.0 | -0.0 |
| warm1 | 4 | 0 | 4/4 | [6.35, 6.72, 6.97, 7.45] | 6.84 | 24.96 | 23.56 | [1.1, 2.3, 3.4, 4.5] | [553] | 20.1 | -0.0 |

## warm cell pass 2 (512-token prompts, 128 tokens; pass 2 = prefix-cached)

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| warm2 | 1 | 0 | 1/1 | [28.71] | 28.71 | 28.93 | 26.99 | [0.2] | [548] | 22.3 | 0.0 |
| warm2 | 2 | 0 | 2/2 | [15.01, 14.66] | 14.83 | 28.59 | 27.87 | [0.2, 0.5] | [552] | 21.2 | 0.0 |
| warm2 | 4 | 0 | 4/4 | [7.34, 7.43, 7.33, 7.46] | 7.38 | 28.76 | 28.25 | [0.2, 0.5, 0.7, 0.9] | [553] | 20.1 | 0.0 |

## Host memory over the whole run (boot included)

samples=85 min_MemAvailable_GB=20 max_swap_used_GB=1
