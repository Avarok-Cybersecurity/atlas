# Concurrency sweep — 2026-09-06T05:01:38-04:00 arm=hcbatch_1

Fingerprint: `/home/ms/atlas/.claude/worktrees/wf_a8fa8242-651-7/.research/exl3_decode_perf/ab_batched-hc-verify_20260906T044450/20260906_045206_concurrency_sweep_hcbatch_1/fingerprint.txt`. Per-stream decode tok/s = (completion_tokens-1)/decode wall;
aggregate decode tok/s = sum tokens / (max last chunk - min first chunk); the incl.-TTFT column divides by the cell wall.
Host memory from /proc/meminfo; nvidia-smi memory is [N/A] on GB10 (unified memory) — see mem_samples.csv.

## arm: short

FINGERPRINT port=8899 model=qwen3.8-flash-next-exl3 label=short prompt_tokens~2000 max_tokens=300 temp=0 effort=low stream=1 prompt_class=salted-log+code-task concurrency=[1, 2, 4] repeats=3 salt=4242 abort_avail_gb=8.0 abort_swap_delta_gb=4.0 date=2026-09-06T04:52:59

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| short | 1 | 0 | 1/1 | [27.15] | 27.15 | 27.24 | 20.4 | [3.7] | [2025] | 22.3 | 0.0 |
| short | 1 | 1 | 1/1 | [27.48] | 27.48 | 27.57 | 20.83 | [3.5] | [2034] | 22.2 | -0.0 |
| short | 1 | 2 | 1/1 | [26.73] | 26.73 | 26.82 | 20.47 | [3.4] | [2018] | 22.2 | -0.0 |
| short | 2 | 0 | 2/2 | [9.59, 9.34] | 9.46 | 16.9 | 15.38 | [3.5, 7.0] | [2023] | 21.6 | -0.0 |
| short | 2 | 1 | 2/2 | [8.94, 9.05] | 9.0 | 16.25 | 14.87 | [6.9, 3.4] | [2020] | 21.5 | -0.0 |
| short | 2 | 2 | 2/2 | [9.58, 9.74] | 9.66 | 17.29 | 15.72 | [6.9, 3.5] | [2048] | 21.4 | -0.0 |
| short | 4 | 0 | 4/4 | [3.8, 4.83, 4.0, 4.49] | 4.24 | 14.61 | 14.01 | [7.0, 3.5, 10.4, 13.8] | [2031] | 20.3 | -0.0 |
| short | 4 | 1 | 4/4 | [3.82, 4.0, 4.92, 4.53] | 4.27 | 14.67 | 14.07 | [6.9, 10.3, 3.4, 13.8] | [2007] | 20.2 | -0.0 |
| short | 4 | 2 | 4/4 | [4.67, 5.19, 4.17, 3.92] | 4.42 | 15.05 | 14.43 | [13.7, 3.4, 10.3, 6.8] | [2020] | 20.1 | -0.0 |

## mHC batched verify liveness / declines (serve.log)

    2026-09-06T08:52:54.683036Z  INFO spark_model::model::trait_impl::verify_hc_multi_gate: mHC batched verify: ARMED row_cap=16 shared_dense=on shared_head=on shared_hc=off (kill: ATLAS_NO_MTP_HC_BATCH_VERIFY)
    2026-09-06T08:53:50.324579Z  INFO spark::scheduler::verify_hc_multi_step: mHC batched verify gate: arm=ARMED n_verify=2 row_cap=16 shared_dense=true shared_head=true shared_hc=false
    2026-09-06T08:53:50.324647Z  INFO spark_model::model::trait_impl::verify_hc_multi: mHC verify: MULTI-SEQUENCE pass n=2 ks=[3, 3] R=6 shared_dense=true shared_head=true shared_hc=false

## warm cell pass 1 (512-token prompts, 128 tokens; pass 2 = prefix-cached)

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| warm1 | 1 | 0 | 1/1 | [28.54] | 28.54 | 28.76 | 22.57 | [1.1] | [548] | 21.7 | 0.0 |
| warm1 | 2 | 0 | 2/2 | [11.11, 11.96] | 11.54 | 20.31 | 18.71 | [2.2, 1.1] | [552] | 21.2 | 0.0 |
| warm1 | 4 | 0 | 4/4 | [5.41, 3.94, 4.08, 4.35] | 4.22 | 15.32 | 14.82 | [1.1, 2.3, 3.4, 4.5] | [553] | 20.2 | 0.0 |

## warm cell pass 2 (512-token prompts, 128 tokens; pass 2 = prefix-cached)

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| warm2 | 1 | 0 | 1/1 | [28.63] | 28.63 | 28.86 | 26.89 | [0.2] | [548] | 22.2 | 0.0 |
| warm2 | 2 | 0 | 2/2 | [11.8, 10.62] | 11.21 | 20.9 | 20.52 | [0.2, 0.5] | [552] | 21.2 | -0.0 |
| warm2 | 4 | 0 | 4/4 | [6.15, 4.19, 4.22, 4.37] | 4.3 | 16.75 | 16.61 | [0.2, 0.5, 0.7, 0.9] | [553] | 20.1 | -0.0 |

## Host memory over the whole run (boot included)

samples=114 min_MemAvailable_GB=20 max_swap_used_GB=1
