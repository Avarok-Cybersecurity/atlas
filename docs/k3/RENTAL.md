# Rental soak — fill after C0–C7 (now)

C0–C7 lab boxes are checked on the 0.40B twin / dummy path (PR #1053, 2026-09-12).
This file is the soak runbook. **Do not book until S1 GPU + S6 `spark serve` dummy also exist.**
CPU C1 is not a Hopper substitute. 5090 cannot launch SM121.

## What is green (lab)

| Gate | Evidence |
| --- | --- |
| C0 | spark2 `kimi_k3` loader tests + atlas-core parse |
| C1 | 8/8 first-token vs HF goldens after FLA-unbounded KDA gate (`f5a3b99`). Teacher-force-to-EOS 8/8. Until-EOS greedy re-run in flight. |
| C2–C6 | twin tests on spark2 |
| C7 | dummy NCCL TP=2 hidden=7168 spark1+spark2 `MATCH True`; drop-rank-1 `DROP True` |

## What is **not** green (blocks a useful soak)

- `K3BoundLayer::decode` LinearAttention default is CUDA `kda_decode` (`K3_CUDA_KDA=0` is the CPU escape). FullAttention MLA is CPU unless `K3_CUDA_MLA=1` (opt-in; C1 unproven). Host still does projections / AttnRes / MLP. **Do not book.** spark1 recopy `7238f64bf` CUDA KDA default **matches C1** aviation greedy (nvfp4 ships `kda_decode`); mix=0 still moves tokens.
- Serve-path token match is not a Hopper soak. Projections / AttnRes / MLP still host.
- MXFP4 GPU grouped GEMM: KERNEL.toml extra_cu + packed lander when `K3_ALLOW_MXFP4=1`. spark2 nvcc still required to compile the extra_cu PTX. K3BoundLayer LatentMoE still host.
- Official 1.56 TB not downloaded (correct)

## Box to book (when S6 exits)

Prefer **8×B300** (fits official MXFP4). Alternate 16×H200 / 16×B200. Need IB. Pin the **then-current** vLLM K3 image the week of booking — do not assume a tag from this file.

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
