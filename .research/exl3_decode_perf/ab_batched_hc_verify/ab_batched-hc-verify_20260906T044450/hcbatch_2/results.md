# Concurrency sweep — 2026-09-06T05:11:07-04:00 arm=hcbatch_2

Fingerprint: `/home/ms/atlas/.claude/worktrees/wf_a8fa8242-651-7/.research/exl3_decode_perf/ab_batched-hc-verify_20260906T044450/20260906_050146_concurrency_sweep_hcbatch_2/fingerprint.txt`. Per-stream decode tok/s = (completion_tokens-1)/decode wall;
aggregate decode tok/s = sum tokens / (max last chunk - min first chunk); the incl.-TTFT column divides by the cell wall.
Host memory from /proc/meminfo; nvidia-smi memory is [N/A] on GB10 (unified memory) — see mem_samples.csv.

## arm: short

FINGERPRINT port=8899 model=qwen3.8-flash-next-exl3 label=short prompt_tokens~2000 max_tokens=300 temp=0 effort=low stream=1 prompt_class=salted-log+code-task concurrency=[1, 2, 4] repeats=3 salt=4242 abort_avail_gb=8.0 abort_swap_delta_gb=4.0 date=2026-09-06T05:02:39

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| short | 1 | 0 | 1/1 | [27.16] | 27.16 | 27.26 | 20.42 | [3.7] | [2025] | 22.3 | 0.0 |
| short | 1 | 1 | 1/1 | [27.44] | 27.44 | 27.53 | 20.77 | [3.5] | [2034] | 22.2 | 0.0 |
| short | 1 | 2 | 1/1 | [26.75] | 26.75 | 26.84 | 20.45 | [3.5] | [2018] | 22.2 | 0.0 |
| short | 2 | 0 | 2/2 | [9.59, 9.34] | 9.47 | 16.89 | 15.36 | [3.5, 7.0] | [2023] | 21.6 | -0.0 |
| short | 2 | 1 | 2/2 | [10.86, 10.14] | 10.5 | 18.2 | 16.44 | [3.5, 7.0] | [2020] | 21.6 | -0.0 |
| short | 2 | 2 | 2/2 | [10.56, 10.65] | 10.61 | 18.88 | 17.01 | [6.9, 3.5] | [2048] | 21.5 | -0.0 |
| short | 4 | 0 | 4/4 | [4.07, 4.52, 5.0, 3.85] | 4.3 | 14.79 | 14.19 | [10.4, 13.9, 3.4, 6.9] | [2031] | 20.3 | -0.0 |
| short | 4 | 1 | 4/4 | [4.7, 3.98, 3.8, 4.41] | 4.2 | 14.62 | 14.02 | [3.5, 10.4, 7.0, 13.9] | [2007] | 20.2 | -0.0 |
| short | 4 | 2 | 4/4 | [4.78, 3.89, 4.49, 4.14] | 4.32 | 14.92 | 14.31 | [3.4, 6.9, 13.7, 10.3] | [2020] | 20.2 | -0.0 |

## mHC batched verify liveness / declines (serve.log)

    2026-09-06T09:02:34.690521Z  INFO spark_model::model::trait_impl::verify_hc_multi_gate: mHC batched verify: ARMED row_cap=16 shared_dense=on shared_head=on shared_hc=off (kill: ATLAS_NO_MTP_HC_BATCH_VERIFY)
    2026-09-06T09:03:30.254943Z  INFO spark::scheduler::verify_hc_multi_step: mHC batched verify gate: arm=ARMED n_verify=2 row_cap=16 shared_dense=true shared_head=true shared_hc=false
    2026-09-06T09:03:30.255019Z  INFO spark_model::model::trait_impl::verify_hc_multi: mHC verify: MULTI-SEQUENCE pass n=2 ks=[3, 3] R=6 shared_dense=true shared_head=true shared_hc=false

## warm cell pass 1 (512-token prompts, 128 tokens; pass 2 = prefix-cached)

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| warm1 | 1 | 0 | 1/1 | [28.07] | 28.07 | 28.29 | 22.31 | [1.1] | [548] | 22.5 | 0.0 |
| warm1 | 2 | 0 | 2/2 | [10.97, 11.77] | 11.37 | 20.01 | 18.43 | [2.3, 1.1] | [552] | 22.0 | 0.0 |
| warm1 | 4 | 0 | 4/4 | [4.13, 5.39, 4.33, 4.77] | 4.55 | 16.04 | 15.48 | [2.3, 1.1, 3.6, 4.8] | [553] | 21.0 | -0.0 |

## warm cell pass 2 (512-token prompts, 128 tokens; pass 2 = prefix-cached)

| arm | C | rep | ok | per-stream decode tok/s | median | aggregate decode tok/s | aggregate incl. TTFT | TTFT s | prompt tok | min avail GB | swap delta GB |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| warm2 | 1 | 0 | 1/1 | [28.71] | 28.71 | 28.94 | 27.0 | [0.2] | [548] | 23.1 | 0.0 |
| warm2 | 2 | 0 | 2/2 | [11.81, 10.62] | 11.22 | 20.91 | 20.53 | [0.2, 0.5] | [552] | 22.1 | 0.0 |
| warm2 | 4 | 0 | 4/4 | [4.62, 4.79, 6.14, 5.41] | 5.1 | 18.49 | 18.33 | [0.4, 0.7, 0.2, 1.0] | [553] | 21.0 | 0.0 |

## Host memory over the whole run (boot included)

samples=112 min_MemAvailable_GB=20 max_swap_used_GB=1
