# Strix Halo (gfx1151) — Qwen3.6-27B NVFP4: MLPerf-grade results

Native-HIP Atlas (`spark`) serving `unsloth/Qwen3.6-27B-NVFP4` on a single AMD Strix
Halo (Radeon 8060S, gfx1151, 60 GB GTT), vs llama.cpp on the same box. All Atlas
numbers gated on coherence + KL-faithfulness (see Validation below).

## Head-to-head — matched 4-bit quality

| Front | **Atlas NVFP4 + MTP** | llama.cpp Q4_K_M | llama.cpp rocmfp4-lean | llama.cpp Q2-MTP (2-bit) |
|---|---|---|---|---|
| Decode (tok/s) | **16.2 – 16.7** | 10.9 | 12.5 | 20.7 |
| Prefill, cold (tok/s) | **~178** | 58.8 | — | 157 |
| Prefill, warm partial-hit | **1.46 s** | — | — | — |
| BFCL-v4 single-turn accuracy | **89.82** | 88.60 | — | lower (2-bit) |
| MTP speculative decode | eff **1.72×** | — | — | 2-bit only |

At matched 4-bit quality, Atlas NVFP4 **leads every front** — decode +30–53 %,
prefill ~3×, accuracy +1.2. llama.cpp's only faster figure is its **2-bit** decode
(20.7 tok/s), a lower quality tier that does not meet the MLPerf accuracy target.

## Recommended serve config

```bash
spark serve unsloth/Qwen3.6-27B-NVFP4 \
  --host 0.0.0.0 --port 8081 --max-seq-len 32768 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype bf16 --max-batch-size 1 \
  --speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000 \
  --disable-tool-grammar true --enable-prefix-caching \
  --ssm-cache-slots 48 --ssm-checkpoint-interval 16
```
Env: `ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1`.

**Warm-prefill (`--ssm-checkpoint-interval 16 --ssm-cache-slots 48`):** the default
interval 256 checkpoints SSM state only every 4096 tokens, so a sub-4096 partial-prefix
cache hit recomputes the whole SSM tail. Interval 16 restores a checkpoint near the match
point → a 4709-token partial-hit prefill drops **4.20 s → 1.46 s (2.9×)**, output-identical.
Use `--ssm-cache-slots 48` (~7.3 GB), **not** 128 (~19 GB) which OOMs the 60 GB GTT pool.
Neutral for single-turn; a win for multi-turn agentic.

## Validation (deterministic, temp = 0)

- **Coherence:** 14/14 (`.claude/skills/strix-fp8-verify/prompts.json`).
- **KL faithfulness:** `tok_agree = 1.0000`, `kl_mean = 1e-5` vs the proven fingerprint
  (`coherence_kl.py --compare`). The warm-prefill config is bit-identical to baseline.
- **MTP enabled:** K=2 self-speculation, ~72 % draft acceptance, eff 1.72 tokens/step.

## Decode ceiling (why 4-bit is the wall)

Decode is LPDDR5X-bandwidth-bound: ~14 GB of 4-bit weights ÷ ~182 GB/s achieved ≈ 13
tok/s base × MTP ≈ 16.7. lm_head, FFN, and attention are already NVFP4; the 4-bit
byte-well is dry. The one large MTP-overhead lever — skipping the verify-time host-logits
pipeline — **fails the KL gate** (returns raw GPU argmax, but the slow-path quality nudges
are load-bearing: 31 % of greedy tokens change, `tok_agree = 0.69`). Only output-faithful
levers are viable. Beating llama's 2-bit decode (20.7) at NVFP4 quality would require a
genuine sub-4-bit rocmfp4 GEMV (fewer bytes/token) — a separate, KL-gated effort.
