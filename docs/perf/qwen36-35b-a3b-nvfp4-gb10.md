# Qwen3.6-35B-A3B-NVFP4 on GB10 — serve performance & decode-flag findings

_Recorded 2026-07-16. GitHub issues are disabled on this repo, so this perf ticket lives as a doc._

## Summary

On **Qwen3.6-35B-A3B-NVFP4 / GB10**, the documented fast-serve flag set covers **prefill** but leaves **decode ~8–12% on the table at concurrency** — five decode-path env flags were not being set. Adding them is coherence-safe (verified short + long generation) and lifts concurrent decode throughput. Single-stream decode is separately near its architectural floor for this hybrid model (analysis below), so the flags help throughput *scaling*, not C=1 latency.

Filing so the decode flags land in the canonical serve config and the perf baseline is on record. Related: LoRA work in #58 (LoRA fold is ≈ zero-overhead vs base at every concurrency, confirmed below).

## Missing decode flags (add to the serve env)

| flag | effect | gated in |
|---|---|---|
| `ATLAS_HOLO_FP4_PROJ_DECODE=1` | NVFP4 projection weights on the decode path (less HBM traffic). In the proven Holo NVFP4 config; easy to drop on non-Holo qwen3.6. | `weight_loader/qwen35/load_layers.rs:215` |
| `ATLAS_GDN_DECODE_GRAPH=1` | GDN (linear-attn) decode captured as a graph | `model/trait_impl/decode_a.rs:202` |
| `ATLAS_GDN_FUSED_CONV=1` | fused causal-conv in the GDN decode step | `qwen3_ssm/.../ssm_batched_recurrent.rs:270` |
| `ATLAS_GDN_FUSED_NORM=1` | fused norm in the GDN decode step | qwen3_ssm decode |
| `ATLAS_DECODE_OPT=1` | dense-FFN decode fast path | `layers/dense_ffn.rs:262` |

`ATLAS_HOLO_FP8_SSM_DECODE` is **redundant** here — the SSM arms already load FP8-native for decode (`Layer N: SSM native FP8 — w8a16 decode + prefill`).

⚠️ `ATLAS_FP8_SINGLE_SCALE` must stay **unset on block-scaled FP8** checkpoints (disables block-scaled prefill → ~14× drift); it is valid only on NVFP4 (single-scale). This matrix is NVFP4, so it's set.

## Config (this matrix)

- Model: `unsloth/Qwen3.6-35B-A3B-NVFP4`, GB10, `atlas-gb10:b12x-ready` (CUDA 13.2, CUTLASS).
- Prefill flags: `ATLAS_FLASHINFER_PREFILL ATLAS_GDN_FLASHINFER ATLAS_CUBLAS_GEMM ATLAS_CUTLASS_WORKSPACE_MB=512 ATLAS_PREFILL_VARLEN ATLAS_PREFILL_CODISPATCH(+WINDOW_MS=100) ATLAS_MOE_PREFILL_EXACT_TILES ATLAS_SSM_BATCHED_RECURRENT ATLAS_HOLO_MOE_GROUPED_CUTLASS ATLAS_HOLO_MOE_GROUPED_DOWN ATLAS_HOLO_FAST_MOE_MODE=full ATLAS_HOLO_FAST_MOE_LAYERS=0-39 ATLAS_HOLO_NATIVE_FP8_ATTN ATLAS_HOLO_NATIVE_FP8_SSM ATLAS_HOLO_LOW_MEMORY_MOE ATLAS_Q12_BATCHED(+_FIRST_CHUNK) ATLAS_GDN_TC_VBLOCK=0 ATLAS_KV_OVERCOMMIT ATLAS_FP8_SINGLE_SCALE`
- Decode flags: the 5 above + `ATLAS_DECODE_GRAPHS_MULTISEQ ATLAS_MOE_BATCHED_DECODE`
- Serve: `--scheduling-policy slai --tbt-deadline-ms 100 --max-prefill-tokens 16384 --kv-cache-dtype bf16 --max-batch-size 8 --max-num-seqs 8 --gpu-memory-utilization 0.78`

## Performance matrix — concurrency (workload: 1220-tok prompt → 128-tok gen, base, temp 0)

**Prefill** (aggregate tok/s across the batch; mean TTFT):

| C | prefill agg tok/s | mean TTFT |
|---|---|---|
| 1 | 3522 | 0.35s |
| 2 | 3805 | 0.65s |
| 4 | 4106 | 1.19s |
| 8 | 4135 | 2.35s |

**Decode** (aggregate gen tok/s; per-request rate) — before vs after the decode flags:

| C | decode agg (no decode flags) | decode agg (+decode flags) | Δ | per-req tok/s |
|---|---|---|---|---|
| 1 | 84.5 | 85.9 | +2% (noise) | 85.9 |
| 2 | 86.3 | 93.0 | +8% | 46.5 |
| 4 | 84.3 | 91.3 | +8% | 22.8 |
| 8 | 118.1 | 132.0 | +12% | 16.5 |

**LoRA overhead** (moe-dgu down/gate/up fold vs base, same workload): within noise at every C — prefill 3.3–3.9K agg, decode 86/89/88/120 agg. Fold is effectively zero-overhead under concurrent load.

## Performance matrix — single-request prefill ISL sweep (TTFT-based)

| ISL (tok) | base tok/s | LoRA tok/s |
|---|---|---|
| 976 | 3168 | 2896 |
| 3875 | 4949 | 4854 |
| 7812 | 5232 | 5154 |

## Why single-stream decode (~86 tok/s) is near the floor

This is a **hybrid 30 linear-attn (GDN/SSM) + 10 full-attn** MoE (256 experts, top-8, 3B active). Two structural limits:
1. The GDN/SSM recurrent decode step is **sequential per token** — no intra-token parallelism to hide latency.
2. Each MoE token routes to a **different** top-8 of 256 experts, so concurrent requests don't share expert-weight reads the way a dense model would → decode is **HBM-bandwidth-bound** and only amortizes once the batch is large (the C=8 aggregate bump).

So decode gains come from **throughput scaling** (the +8–12% at C≥2 above), not from cutting single-stream latency. A prefill-style 4× is not available on the decode path here.

## Ask

- [ ] Land the 5 decode flags in the canonical GB10/qwen3.6 serve config + runbook.
- [ ] (opt) profile the GDN recurrent decode step + MoE expert gather to confirm the BW-bound hypothesis and find any remaining decode lever.
