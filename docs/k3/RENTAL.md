# Rental soak — fill after C0–C7 (now)

C0–C7 lab boxes are checked on the 0.40B twin / dummy path (PR #1053, 2026-09-12).
This file is the soak runbook. Lab twin cannot hold official MXFP4 (~1.56 TB). **Booking 8×B300 is for CR1–CR3**, not for finishing KDA/MLA. 5090 cannot launch SM121.

## What is green (lab)

| Gate | Evidence |
| --- | --- |
| C0 | spark2 `kimi_k3` loader tests + atlas-core parse |
| C1 | 8/8 first-token vs HF goldens after FLA-unbounded KDA gate (`f5a3b99`). Teacher-force-to-EOS 8/8. Until-EOS greedy re-run in flight. |
| C2–C6 | twin tests on spark2 |
| C7 | dummy NCCL TP=2 hidden=7168 spark1+spark2 `MATCH True`; drop-rank-1 `DROP True` |

## What is green on lab serve (2026-09-13)

- CUDA KDA + CUDA MLA default (`K3_CUDA_KDA=0` / `K3_CUDA_MLA=0` CPU escape): aviation greedy bee-fly at `0898345` with `K3_CUDA_MLA=1`; mix=0 → `to to to…`. Logs: `kda_decode` and `mla_decode`. Recopy after default-on commit.
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
