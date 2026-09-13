# Rental soak — fill after C0–C7 (now)

C0–C7 lab boxes are checked on the 0.40B twin (PR #1053). Dummy NCCL is not C7.
This file is the soak runbook. Lab twin cannot hold official MXFP4 (~1.56 TB). **Booking 8×B300 is for CR1–CR3**, not for finishing KDA/MLA. 5090 cannot launch SM121.

## What is green (lab)

| Gate | Evidence |
| --- | --- |
| C0 | official parse + 96-shard map + known-bads |
| C1 | 8/8 first-token + 8/8 until-EOS vs HF on spark2 |
| C2–C6 | 0.40B twin tests on spark2 (`K3_TWIN`, `83c0d34`) with known-bads that moved |
| C7 | 0.40B TP=2 spark1+spark2 aviation 16 tokens == TP=1; kill rank 1 timed out |

## What is green on lab serve (2026-09-13)

- CUDA KDA + CUDA MLA default (`K3_CUDA_KDA=0` / `K3_CUDA_MLA=0` CPU escape): spark1 `9d6c273` serve with **no** `K3_CUDA_MLA` env — aviation greedy bee-fly, first id 1459. mix=0 → `to to to…` (first id 308). Logs: `kda_decode` and `mla_decode`. Live left on CUDA default.
- Packed LatentMoE **launches** `moe_w4a16_grouped_gemm_ptrtable_e8m0` (`089834512`). 0.40B has no packed experts → CPU MoE. Lookup-fail does not silent-CPU.
- Dummy TP=2 NCCL hidden=7168 spark1+spark2.

## Still host / untested on official shards

- Projections, AttnRes, router/SiTU/down/up/shared experts
- GEMM never run against 96-shard official MXFP4 (correct: not downloaded)
- `K3_ALLOW_MXFP4=1` required on rental load

## Box to book

Prefer **8×B300** (official MXFP4). Alternate 16×H200 / 16×B200. Need IB. Pin the **then-current** vLLM K3 image the week of booking. Serve flags: `K3_ALLOW_MXFP4=1`. CUDA MLA/KDA default on (`K3_CUDA_MLA=0` / `K3_CUDA_KDA=0` CPU). Lab 0.40B serve is **not** the soak.

## Soak protocol (CR1–CR3)

Same node, official `moonshotai/Kimi-K3` MXFP4, Atlas vs that vLLM image.

| ID | Check |
| --- | --- |
| CR1 Load | Completes without OOM |
| CR2 Quality | First 32 greedy tokens agree with same-box vLLM (or documented sampling delta) |
| CR3 Speed | Publish Atlas/vLLM TTFT and decode tok/s at C=1 and C=8, ISL/OSL 2k/256 and 8k/256. No required win. |

NCCL pin on Sparks (do not copy to rental blindly): `NCCL_SOCKET_IFNAME=enp1s0f1np1` `NCCL_IB_HCA=rocep1s0f1`. Re-discover HCAs on the rented box.

## Decision after soak

More kernel work vs declare architecture complete. Write the pack: load logs, mem, C=1/C=8 tables, 10-prompt greedy dump.
