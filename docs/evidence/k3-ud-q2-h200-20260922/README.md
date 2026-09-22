# Kimi-K3 UD-Q2_K_XL Atlas TP8 proof (2026-09-22)

Honest status: Atlas spark served Unsloth UD-Q2_K_XL on 8x H200. /v1/completions
returned tokens. The tokens are not 408. This is load+forward bring-up, not a quality pass.

## Cluster
- shadeform / metrale-h200, 8x NVIDIA H200 (143771 MiB each)
- After bind: ~130271 MiB used/rank (~127.2 GiB), ~12885 MiB free
- Health :18889 ready, model kimi-k3, 8 spark ranks
- Process start 2026-09-22 01:33:28 UTC
- Binary /data/workspace/atlas/target/k3-hopper/release/spark
- sha256 bf74955574d401e3979acbea978778f33deedb264dc0f6cb627eaf48f30505ce
- Git HEAD 2aa987224c5f68deb6f9dec94dcec65174770e9f (feat/kimi-k3-ud-q2)
- PR https://github.com/Avarok-Cybersecurity/atlas/pull/1228
- Flags: --tp-size 8 --ep-size 1 --world-size 8 --max-seq-len 512 --kv-cache-dtype bf16
- Weights: /data/workspace/models/kimi-k3-q2/UD-Q2_K_XL (19 shards, ~802 GiB disk)

## Completions (all /v1/completions except the 400)

| file | prompt | text | finish | TTFT ms | n |
| --- | --- | --- | --- | --- | --- |
| completion-17x24-1.json | What is 17 * 24? Reply with the number only. | ",." | timeout | 571628 | 2 |
| completion-17x24-2.json | same | ",." | timeout | 553927 | 2 |
| completion-408-completions.json | Reply with exactly 408 and nothing else. | ",," | timeout | 352765 | 2 |
| completion-aviation-vrvr.json | According to all known laws of aviation, | "vrvr" | timeout | 319135 | 2 |
| completion-408-chat-400.json | chat, same 408 ask | 400 XTML | n/a | 1 | 0 |

Expected for the math prompts: 408. Observed punctuation / "vrvr". First token id 11 on the 17x24 and 408 prompts.

## Why the text is wrong and slow
q/k/v/o, routed down/up, shexp, and IQ2 mix run on the host (D2H packed bytes). CUDA only replaces KDA recurrent + MLA SDPA. moe_w4a16 is MXFP4/E8M0, so use_cuda_moe stays false. GPU util during forward ~0-8%. /v1/completions ignores JSON timeout:1800 and uses the 300 s server default. Prefill overruns it; decode stops after 2 tokens.

## Shapes that fit (do not reverse)
TP=8 EP=1, hidden=7168. 896 is cursed: hidden/tp=7168/8=896, and n_experts=896.
Routed down on-rank [3584, 896] (hidden/tp), not 3584x7168.
Routed up [896, 3584]. shexp [7168, 768]/[768, 7168].
attn_k_b / attn_v_b stay split. Do not fuse into kv_b.
Packed IQ2/IQ3 stays packed. Replicating routed down/up as BF16 OOM at 148.30 GB on shard 17/19.

## Preflight lie
Log said ~112.25 GiB/rank (disk/tp + 12 GiB). nvidia-smi after bind is ~130.3 GB/rank.
