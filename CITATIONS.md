# CITATIONS

This branch (`feature/tq-plus-integration`) integrates TurboQuant+ work from
prior art that upstream Atlas does not credit. Local-only development; not for
publication or upstream PR.

## Prior art (in order it should be cited)

### 1. Google — TurboQuant (canonical)
- **TurboQuant: Online Vector Quantization with Near-optimal Distortion
  Rate** — Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni.
  arXiv:[2504.19874](https://arxiv.org/abs/2504.19874) (April 2025, 25
  pages).
- Establishes that random rotation of input vectors induces a
  concentrated Beta distribution on coordinates, after which the same
  scalar Lloyd-Max quantizer can be applied per coordinate. For inner
  product estimation specifically, the paper proposes a two-stage
  approach: an MSE-optimal quantizer followed by a 1-bit Quantized JL
  transform on the residual to remove bias. Information-theoretic
  near-optimal distortion at all bit widths; the paper reports absolute
  quality neutrality at 3.5 bits per channel and marginal degradation
  at 2.5 bpc on KV cache quantization.
- Atlas's `wht_bf16.cu` implemented plain WHT (no random sign mask) —
  strictly weaker than the canonical Randomized Hadamard form.

### 2. `TheTom/turboquant_plus` — TurboQuant+ umbrella research repo
- Primary research dumping ground for the TQ+ work that spans multiple
  downstream inference engines (Atlas, llama.cpp, vLLM, etc.).
  [https://github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
- ~15 papers under `docs/papers/` plus reference quant/dequant
  implementations and bench harnesses. Where the per-feature designs
  cited below (matched-norm L2, sparse V, asymmetric K/V, InnerQ,
  layer-aware V) are documented and benchmarked.

### 3. `TheTom/llama-cpp-turboquant` — TurboQuant+ engine reference
- First public llama.cpp fork implementing a complete TurboQuant KV
  cache (`--kv-cache-dtype turbo3/turbo4/turbo8`).
  [https://github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
- Source: `ggml/src/ggml-cuda/turbo-wht.cu`, `turbo-quant.cuh`,
  `turbo-innerq.cu`.
- The sign arrays vendored into `kernels/gb10/common/tq_plus_signs.cuh`
  are byte-identical to `TURBO_WHT_SIGNS1`/`TURBO_WHT_SIGNS2` in
  `turbo-quant.cuh` (seed=42 Rademacher draws).
- The CLI surface `turbo3 / turbo4 / turbo8` that Atlas adopted matches
  this fork's prior public CLI.

### 4. TurboQuant+ paper set — per-feature designs
Pieces relevant to this branch's port sequence, all from
`TheTom/turboquant_plus/docs/papers/`:

- `asymmetric-kv-compression.md` — K is bandwidth-critical, V can tolerate
  harder quant. Motivates `KvCacheDtype::TurboKV { k: Turbo4, v: Turbo3 }`.
- `layer-aware-v-compression.md` (LA-V7) — boundary V layers (first/last N)
  must stay higher precision; intermediate layers compress aggressively.
- `sparse-v-dequant.md` — at decode, only dequant V rows where attention
  weight > threshold (skip < threshold → 30-50% V-dequant cost savings).
- `weight-compression-tq4.md` — TQ4_1S 4-bit weight quantization with N(0,1)
  Lloyd-Max + WHT rotation per weight tile. Distinct codebook from KV cache
  (which uses N(0, 1/d) since post-WHT the per-coordinate variance is 1/d).
- `triattention-v3.md` — long-context attention eviction policy.
- `moe-v-compression-frontier.md` — V compression for MoE models specifically.

## What this branch changes vs upstream Atlas (`87b7bb3`)

### Two-sided Rademacher signs in WHT (commit `3822f8a`)

| File | Change |
|---|---|
| `kernels/gb10/common/tq_plus_signs.cuh` | new — `TQP_SIGNS1_128 / TQP_SIGNS2_128` + apply helpers |
| `kernels/gb10/common/wht_bf16.cu` | `TQ_PLUS_SIGNS`-gated: signs1 before / signs2 after butterfly for hd=128; new `wht_bf16_inplace_inv` kernel for post-attention un-rotation |
| `qwen3_attention/types.rs` | adds `wht_bf16_k_inv` handle |
| `qwen3_attention/init.rs` | loads inverse kernel |
| `qwen3_attention/decode/attention_forward.rs:478` | routes iWHT to `wht_bf16_k_inv` |

With `TQ_PLUS_SIGNS` undefined: kernels are byte-equivalent to upstream Atlas
(A/B baseline). With it defined: hd=128 forward + inverse rotations carry the
canonical two-sided sign masks; attention dot product preserved since
`(S2·H·S1)·(S1·H·S2)^T = I`.

### N(0, 1/d) codebook replacement + amax-loop removal

| File | Change |
|---|---|
| `kernels/gb10/common/reshape_and_cache_turbo.cu` | TURBO4 + TURBO3 codebooks + bounds + MAX defines swapped to N(0,1/128); turbo4/turbo3/turbo8 write paths drop the per-group amax compute and write a unit FP8/BF16 scale for byte-layout compat |
| `kernels/gb10/common/paged_decode_attn_turbo4{,_128,_512}.cu` | 16-level codebook → N(0,1/128) |
| `kernels/gb10/common/paged_decode_attn_turbo3{,_128}.cu` | 8-level codebook → N(0,1/128) |
| `kernels/gb10/common/paged_decode_attn_turbo8{,_128,_512}.cu` | 16-level codebook (used for outlier handling) → N(0,1/128) |
| `crates/spark-runtime/src/kv_dequant.rs` | `TURBO4_LUT`, `TURBO3_LUT` → N(0,1/128) |

Behavior: post-WHT values feed the matching N(0,1/d) Lloyd-Max codebook
directly — no per-group amax loop on the write path, decoder still multiplies
by per-group scale (now always 1.0) for byte-layout backward compat. A
follow-up commit will drop the scale section entirely to reclaim ~11% KV
storage (4.5 → 4.0 bits/elem for turbo4).

### Sparse V dequant (attention-gated row skip, both paths)

All 9 `paged_decode_attn_turbo{2,3,4,8}*.cu` kernels: wraps the per-position V
load + dequant in `if (exp_new > TQ_PLUS_SPARSE_V_THRESHOLD)` on the remainder
loop AND `if (exp_factors[b] > TQ_PLUS_SPARSE_V_THRESHOLD)` per row on the
batched (BC=4) path. Below threshold (default 1e-3) the V vector is left at
zero, saving the V-data + V-scale loads + dequant. Defined as a `#define` so
workloads can override; ref policy in
`turboquant_plus/docs/papers/sparse-v-dequant.md`.

The batched-path version requires moving the `v_vals[BC][VEC_BF16]` load to
AFTER the softmax `exp_factors[]` computation so the per-row exp magnitude
gates the load. Port of Tom's `de44bfe60` (+22% decode at 32K M5 in
`llama-cpp-turboquant`).

### LA-V7 boundary-V protection

Verified: Atlas's `--kv-high-precision-layers` already implements first-N +
last-N → BF16 in `build_layer_kv_dtypes`. Default `"auto"` (=2) covers Tom's
LA-V7 primitive. Per-layer dtype vector flows through scheduler/dispatch. No
port work needed.

### InnerQ — per-channel Q/K equalization machinery

| File | Change |
|---|---|
| `kernels/gb10/common/tq_plus_innerq.cuh` | new — namespace + state declarations + `apply_innerq_scale[_inv]_128` + `accumulate_innerq_calibration_128` helpers |
| `kernels/gb10/common/tq_plus_innerq.cu` | new — `d_innerq_scale[128]`, `d_innerq_scale_inv[128]`, sq-accum, active/calibrating flags. Host controllers: `turbo_innerq_start_calibration(target, strength)` and `turbo_innerq_finalize(group_size, strength)` |

Default state: identity scales (1.0), active=0. Currently a stand-alone
infrastructure drop — integration with `wht_bf16_inplace` requires that
kernel to take a per-tensor `scale_inv` pointer arg (Q gets `scale_inv`, K
gets `scale` after WHT, V gets nothing). Tom's reference design in
`ggml-cuda/turbo-wht.cu` templates on `direction` and passes `scale_inv`
nullable through op_params. Atlas's kernel signature can grow a 4th `kind`
parameter (Q=0, K=1, V=2, output=3) once Rust callers are updated to thread
the tensor kind in. Calibration controller exported as `extern "C"` so a
Rust shim can drive it from a CLI flag like `--innerq-tokens=N`.

W_k pre-multiplication by per-output-channel `scale` at calibration finalize
(per Tom's `turbo_innerq_publish` hook) is part of the weight-pre-rotation
work tracked separately.

### Weight pre-rotation helper (Q/K/V/O at model load)

| File | Change |
|---|---|
| `crates/spark-model/src/weight_loader/qwen35/load_layers/tq_plus_weight_rotation.rs` | new — `apply_canonical_rotation_inplace(gpu, weight, outer, n_heads, head_dim, stream)` reuses `wht_bf16_inplace` kernel with grid=outer×n_heads so every contiguous head_dim chunk of a `[outer, n_heads*head_dim]` weight matrix is rotated independently; `weight_rotation_enabled()` reads `TQ_PLUS_WEIGHT_ROTATION=1` env var. |
| `crates/spark-model/src/weight_loader/qwen35/load_layers.rs` | adds `mod tq_plus_weight_rotation;` |

Integration site (not yet wired): in `attention_arms::load_bf16_then_nvfp4`,
between `shard_dense_bf16` and `quantize_to_nvfp4`, call
`apply_canonical_rotation_inplace` on the bf16 sharded weight for each of
Q/K/V/O. After all 4 are rotated and quantized, the runtime
`wht_bf16_inplace` calls in `write_kv_cache.rs` and
`attention_forward.rs:398/478` become no-ops (or should be skipped via the
same env var). Per-token savings: 4 kernel launches × 40 layers = 160
launches per token.

FP8-native weight path: a future pass needs to dequant→rotate→requant; not
covered by this helper which operates on bf16.

### Asymmetric K/V dtype scaffold

| File | Change |
|---|---|
| `crates/spark-runtime/src/kv_cache.rs` | adds 3 asym variants: `Turbo4KTurbo3V`, `Turbo4KTurbo8V`, `Turbo3KTurbo8V` to `KvCacheDtype`. New `kv_pair()` returns the (K, V) symmetric pair; `is_asymmetric()` predicate. Display + FromStr cover all three (parse names `turbo4k_turbo3v`, `turbo4k_turbo8v`, `turbo3k_turbo8v` plus aliases `turbo4k3v` etc.) |
| 9 files in `crates/spark-model/...` + `crates/spark-server/...` | exhaustive `match` arms extended to route asym variants to their K-side existing kernels (compiles green). 16 match arms covered programmatically; remaining 1 site (`paged_attn_batched.rs`) absorbs via existing `(dtype, _) =>` catch-all. |

Stub-routing behavior: each asym variant currently dispatches the SAME write
and decode kernels as its K-side symmetric counterpart, so storage-byte and
runtime behavior match the K-side dtype only — V side asym savings are NOT
realized until proper asym kernels exist. Tom's
`turboquant_plus/docs/papers/asymmetric-kv-compression.md` predicts ~14%
bandwidth saving at decode for `Turbo4KTurbo3V` once the V-side gets its
own 3-bit decode path.

Remaining work for true asymmetric:
- New write kernels per combo (`reshape_and_cache_flash_turbo4k_turbo3v` etc) that pack K with one layout and V with another in a single launch
- New decode kernels (`paged_decode_attn_turbo4k_turbo3v`) that mix K-side and V-side dequant logic
- Storage byte accounting for asym layouts in `kv_cache.rs` block-stride helpers

## Outstanding (not yet ported)

- 256/512 sign arrays (Rademacher seed=42, derived from same random draws)
- Block-layout refactor to drop the per-group scale section (~11% KV bandwidth saving)
- Pre-rotation of Q/K/V/O projection weights at model load (folds runtime WHT into projection gemm — saves 4 kernel launches × 40 layers per token)
- Asymmetric K/V dtype (`Turbo4K + Turbo3V` etc) — Rust enum + CLI + new asym decode kernels
- InnerQ per-channel scale_inv (Q/K equalization with calibration phase)

## License note

Upstream Atlas is AGPLv3 + a CLA assigning commercial relicense rights to a
proprietary Enterprise Edition. This branch is **local-only**; do not push to
any public fork or upstream PR. If TQ+ work needs to be made public, file a
new fork under Tom Turney's account with full prior-art chain (1) → (2) → (3)
above prominently cited and the original `TheTom/llama-cpp-turboquant`
attribution at the top of every modified kernel.
