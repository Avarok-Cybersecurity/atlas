# Where the 16-row decode `attn` phase actually goes (#927, H100)

Phase-1 attribution for the "multi-row decode attention" lever. **Headline: the
lever as briefed is already in the tree.** Every per-row loop in the multi-seq
attention path was batched on or before the measured tip `c1b67f022`; at n=16
there are no per-row launches left to remove. The phase is not launch-bound —
it is the `w8a16_gemv` ALU wall, and 2/3 of it is the dense FFN that lives
inside the attention layer, not attention at all.

## Anchors (all measured, 1xH100, Qwen/Qwen3.8-27B-FP8, graphs OFF)

`ATLAS_MS_PROFILE` carries only `total / ssm / attn / head` — there are no
attention sub-phase fields in the logs (`decode_a2.rs:594`). Everything below
the anchors is DERIVED; the arithmetic is shown so it can be falsified (Rule 8).

| run | config | n | attn ms (16L) | attn µs/layer | ssm µs/layer |
|---|---|---|---|---|---|
| r5 H | `ATLAS_FFN_NO_BATCH16=1` (the brief's config) | 16 | 18.82 | **1177** | 1248 |
| r5 F | #927 batch16 tier ON | 16 | 19.90 | 1244 | 1319 |
| r5 H | tier OFF | 8 | 16.49 | 1031 | 1035 |
| r5 F | tier ON | 8 | 12.28 | 768 | 773 |
| r4 C | tier ON (irrelevant at m<=4) | 4 | 7.36 | 460 | 451 |
| r4 C | " | 2 | 5.77 | 361 | 349 |

Model shapes (`kernels/hopper/qwen3.8-27b/MODEL.toml`): h=5120, nq=24, nkv=4,
hd=256, gated Q (q_proj_dim=12288), kv_dim=1024, per_seq_qkv=14336 BF16,
inter=17408, 16 attention layers each with its own dense FFN.

## Splitting 1177 µs/layer at n=16

1. `ATLAS_FFN_NO_BATCH16` moves **only** the dense FFN. An SSM layer's sole
   consumer of that switch is its FFN, and it moved 1319 → 1248 = **−71 µs**.
   The attention layer moved 1244 → 1177 = **−67 µs** — the same number, which
   confirms the attention layer's own FP8 tiers (`ATLAS_NO_FP8_QKV_BATCH`,
   the o_proj `wide` arm) did not move, as their gates say.
2. #927's H100 receipt for the FFN at M=16 (`dense_ffn_m16_tc.rs` SSOT):
   gate 0.260 + up 0.260 + down 0.330 = **850 µs/layer** on the batch16 GEMV.
   With (1): the tile-GEMM FFN this config actually runs ≈ **779 µs/layer**.
3. Remainder = 1177 − 779 = **398 µs/layer** of attention proper.
4. Splitting (3) by weight bytes ÷ the same receipt's measured GB/s
   (342 GB/s at N=17408/K=5120; 270 GB/s at the deeper-K shape):

| sub-phase | launches/layer at n=16 | kernel | n=16 µs/L | n=4 µs/L |
|---|---|---|---|---|
| dense FFN (in-layer) | 6 | `w8a16_gemm_n128_m128` (tile) / `w8a16_gemv_batch16` | **779** (66%) | ~300 |
| QKV projections | 3 + 1 deinterleave | `w8a16_gemv_batch16_strided` ×3 (73.4 MB) | **215** (18%) | ~63 |
| o_proj | 1 | `w8a16_gemv_batch16` (31.5 MB) | **105** (9%) | ~31 |
| norms / RoPE / KV write / paged decode / gate / residual | 7 | see inventory | **~78** (7%) | ~40 |

Cross-check at n=4: 300+63+31+40 = 434 vs 460 measured. At n=8 (tier ON):
FFN_gemv(8) ≈ 520 (from the same A/B: 1031 − 768 = 263 = tile − gemv at m=8)
+ QKV/o_proj ~195 + ~53 = 768 — exactly the measured 768. The model closes on
three independent points, so the split is load-bearing, not decorative.

## Launch inventory per attention layer (n>=2, this model, no LoRA/MLA/HC)

| phase | launches | batched? | where |
|---|---|---|---|
| `rms_norm_residual` | 1 | yes, n rows in grid | `multi_seq/mod.rs` |
| `ms_phase_qkv` (FP8 block-scaled) | 3 GEMV + 1 `deinterleave_qg` | yes, band 2..=16 | `qkv_fp8_batch.rs` |
| q/k RMS norm | 2 | yes, `rms_norm_strided` | `qkv.rs::ms_qkv_norms` |
| RoPE | 1 | yes, `rope_strided` | `attn.rs::ms_phase_rope` |
| KV write | 1 | yes, `reshape_and_cache` num_tokens=n | `attn.rs::ms_phase_cache_write` |
| paged decode attention | 1 | yes, `num_seqs` in the grid; Q read in place | `run_paged_decode.rs` |
| sigmoid gate + o_proj | 2 | yes, `sigmoid_gate_mul_batched`, `w8a16_gemv_batch16` | `attn/o_proj.rs` |
| FFN (norm + gate/up/silu/down + residual) | 6 | yes, `forward_prefill` M=n | `multi_seq/ffn.rs` |
| **total** | **~18** | **0 per-row loops** | |

18 launches × 16 layers ≈ 288/step; at ~4 µs of host issue that is ~1.2 ms of
the eager 18.82 ms, and ~0 in production (graphs on). The per-row loops the
brief targets — per-seq RoPE (258 launches/step), per-seq q/k norm (516/step),
per-seq KV write, per-seq Q staging copies, per-row QKV/o_proj GEMV — were all
removed before `c1b67f022`; their kill switches (`ATLAS_NO_ROPE_STRIDED`,
`ATLAS_NO_QK_NORM_STRIDED`, `ATLAS_NO_ATTN_BATCH_CACHE_WRITE`,
`ATLAS_NO_ATTN_Q_INPLACE`, `ATLAS_NO_FP8_QKV_BATCH`) are the receipts.

## What is left, and why it is one kernel

97% of the phase (779 + 215 + 105 = 1099 of 1177 µs) is the **same W8A16
block-scaled FP8 matmul family**. `w8a16_gemv_batch4.cu`'s inner loop spends,
per 16 weight bytes: 1 B load, 16 LUT loads + 16 scale multiplies, then per row
2 `uint4` A loads + 16 BF16→FP32 converts + 16 FFMA. At MAX_M=16 that is
**~36 ALU ops and 32 activation loads per weight byte**, which is why the
#927 receipt reads 342 GB/s against ~3000 GB/s of HBM3 — and why the phase grows
2.5× for 4× the rows while the KV traffic (~40 KB/token) stays trivial.

Each thread owns ONE output column, so the 16 activation converts and the 32
`uint4` A loads it does per weight byte are repeated by every other column's
thread. Giving a thread `N_COLS` adjacent columns amortises both over `N_COLS`
weight bytes **without touching any accumulator's operand order**:
36 → 27 ops/byte at N_COLS=2, → 23 at N_COLS=4 (floor 18: the 16 FFMA are the
work). That is the next bit-exact lever, and it is what Phase 2 implements.

The reassociating alternative already exists — `w8a16_gemm_m16` (#927,
`ATLAS_FFN_M16_TC`, ≤2 BF16 ULP, off by default). The N_COLS tier is its
bit-exact sibling: less upside, no numerics seam.

## Not the lever

* **The paged decode kernel.** One launch, `num_seqs` in the grid, and at
  nq=24/hd=256/GQA 6 the KV traffic is ~25 µs/layer with L2 reuse. It is inside
  the 78 µs "remainder" row — there is nothing to win there.
* **Launch count / graph size.** Already 18/layer, and production captures.
* **Turning #927's batch16 tier back on at n=16.** The A/B says it costs
  +5.4%: at m=16 the M-padded tile GEMM is cheaper than the 36-op/byte GEMV.
  The crossover sits between m=8 (GEMV wins by 263 µs/layer) and m=16.
