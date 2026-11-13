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

## Port sprint sources (`port/consumer-opt`)

Projects, papers and reports the port sprint's code is copied from, translated
from or built on as an idea, one entry per source, with its license and the
Atlas files that carry it. License texts are in `LICENSES/`; notices are in
`NOTICE`.

### Bench quality and long-context scripts (lane L02)

These are the sources of the measurement scripts kept in `~/atlas-wt/bench/` (outside this repository). None of
their code is copied; the entries record the task definitions, protocols and datasets they follow.

- **RULER** (NVIDIA, Apache-2.0, https://github.com/NVIDIA/RULER, `benchmark/ruler/synthetic.yaml`, `data/synthetic/*.py`, `eval/synthetic/constants.py`): the task list (single, multikey, multivalue and multiquery needle-in-a-haystack, variable tracking, common and frequent word extraction, SQuAD and HotpotQA question answering), the needle and question templates, and the `string_match_all` / `string_match_part` scoring rules. Independent generators with deterministic seeding; the NVIDIA Apache-2.0 notice stays in `ruler.py`. No RULER data or fixture is copied.
- **InfiniteBench Retr.KV** (idea only): key-to-value lookup over random UUIDs, the task where approximate attention lost 4 to 11 points in the RetrievalAttention paper. Own generator, no data copied. Atlas: `retr_kv` in `ruler.py`.
- **TensorFold `tensorfold kl --dump/--against`** (Apache-2.0, ashhart/TensorFold, commit 3885dce, `zig/src/cli/cuda_kl.zig`): teacher-forced full-vocabulary KL, NLL and top-1 at every position from the reference's own greedy reply. Idea and protocol only. Atlas: `tf_kl.py`.
- **Cinference `docs/perplexity.md`** (Apache-2.0, commit 383e5db): fixed 4,096-token windows with a 2,048 stride, every token scored once, per domain, unrounded report. Idea and protocol only. Atlas: `ppl.py`.
- **KVMem** (arXiv:2609.04852, no license on the reference repository): the identity canary (an approximating feature with a budget at least the context must reproduce the feature-off greedy tokens), the flat-VRAM canary, and judging a needle test by whether the needle's block was in the selected window. Independent implementation from the published description; nothing read from `kvmem` or `kvmem-llama.cpp`. Atlas: `canaries.py`, `niah_grid.py`.
- **Datasets read at run time, never committed**: PG-19 (DeepMind, Apache-2.0), SQuAD v2 and HotpotQA (CC BY-SA 4.0, local evaluation only), MMLU (MIT), GSM8K (MIT). Commands and the license of each are in `bench/data/README.md`.

### Request evidence and confidence summary

- **TensorFold** (Apache-2.0), `docs/cache-benchmarks.md` and `tools/bench_evidence.py`, commit 3885dce: the definition of the per-request fingerprints `token_sha` (token-id hash) and `output_sha256` (hash of the content and reasoning digests, independent of chunk boundaries), the cold/reused `cached_tokens` rule, and the practice of comparing a drafted run, a concurrent run and a resumed run by fingerprint. Idea and definition only; no code copied. Atlas: `crates/spark-server/src/api/evidence.rs`, `crates/spark-server/src/api/evidence/`.
- **Cascade routing on a cheap confidence signal** (idea only, as in RouteLLM-style threshold calibration): escalate a request to a larger model when the small model's own confidence is low. The summary is a measurement; whether it is a useful trigger is decided by an offline AUROC study, not assumed. Atlas: `crates/spark-server/src/scheduler/conf_summary.rs`.

- **vLLM adaptive verification** (Apache-2.0, commit 8a36f08), `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`
  and `acceptance_estimator.py`: the cost-optimal draft budget and the online logistic acceptance calibration
  (damped Newton-IRLS). Translated for `scheduler/mtp_dcut_cost.rs` and `scheduler/mtp_accept_estimator.rs`
  (row `mtp_dcut_cost_model`). Attribution: `DCUT-V2-ATTRIBUTION.md`.
- **D-Cut** (arXiv 2607.14647): adaptive verification-depth pruning by survival score, the v1 policy that D-Cut v2
  extends (`scheduler/mtp_dcut.rs`).

### Microbench examples (L07)

Methods only; every file is Atlas-native (SPDX line only) and no source was
copied.

- Cinference (https://github.com/satellitedown/cinference, commit `383e5db`,
  Apache-2.0, derived from NInfer by Neroued and contributors): the per-T linear
  sweep and its `T1_lin_x` metric (`bench/README.md`, "Linear Op benchmark";
  `docs/maintainer/linear-tuning.md` sections 1 to 4), the measured bandwidth
  ceiling probe (`tools/hbm_bandwidth_probe.cu`), the oracle discipline
  (`docs/maintainer/op-development.md` sections 6.1 to 6.3) and the profiler
  boundary (`--profile-measured`). Atlas files:
  `crates/spark-model/examples/{linear_t_sweep,hbm_bandwidth_probe}.rs` and
  `examples/common/{sweep_report,hbm_ptx,profiler}.rs`.
- TensorFold (https://github.com/ashhart/TensorFold, commit `3885dce`,
  Apache-2.0): distinct weight copies in GEMV microbenchmarks
  (`docs/recipes/qwen3.8-27b.md:84-86`), rotated and interleaved arm order
  (`zig/src/cuda/qmmf.md`, "Device checks and timing") and planted controls
  (`docs/kernel-contract.md:1-10`). Atlas files:
  `examples/common/{cold_l2,planted}.rs`.
- vLLM (https://github.com/vllm-project/vllm, commit `8a36f08`, Apache-2.0):
  the kernel-microbenchmark skill (`.agents/skills/kernel-microbenchmark/SKILL.md:26-31,121-127`,
  rotating buffers larger than the L2) and the pinned-memory regression gate
  (`benchmarks/benchmark_pin_memory.py:1-40`, one fresh process per arm).
  Atlas files: `examples/common/{cold_l2,pinned_arms}.rs`.
- LMCache (https://github.com/LMCache/LMCache, commit `3aefae1`, Apache-2.0):
  the pinned pool brief's measurement (pin 1, 4 and 16 GiB one call against 64 MB
  chunks, whether WSL2 refuses; `lmcache/v1/memory_allocators/lazy_memory_allocator.py:80-200`).
  Atlas file: `examples/pinned_probe.rs`.

### GPU backend extensions (L08)

- Cinference (https://github.com/satellitedown/cinference, commit `383e5db`,
  Apache-2.0, derived from NInfer by Neroued and contributors):
  `src/core/decode_graph.cpp:62-80`, the CUDA Graph topology signature,
  translated to Rust in `crates/spark-runtime/src/gpu/port_ext.rs` and
  `crates/spark-runtime/src/cuda_backend/gpu_impl_graph_update.rs`.
- LMCache (https://github.com/LMCache/LMCache, commit `3aefae1`, Apache-2.0):
  `csrc/cuda/mp_mem_kernels.cu:358-849`, design of the batched copy path
  (`cuMemcpyBatchAsync`, the 12.8 versus 13.x signature, the support probe);
  no code copied.
- vLLM (https://github.com/vllm-project/vllm, commit `8a36f08`, Apache-2.0):
  `csrc/cache_kernels.cu:110-190`, design of the `cuGetProcAddress`
  resolution and the per-copy fallback; no code copied. Atlas file:
  `crates/spark-runtime/src/cuda_backend/gpu_impl_batch_copy.rs`.
- See `GPU-BACKEND-EXT-ATTRIBUTION.md`.

- **ReplaySSM record and fold** (Cinference, satellitedown/cinference commit 383e5db, derived from Neroued/ninfer,
  Apache-2.0), `docs/maintainer/replayssm-gdn.md` sections 3 to 7: verify keeps the GDN state out of DRAM, records
  the per-row update terms, and folds the accepted prefix once after the verdict. Idea only, no code copied: Atlas's
  own K=4 pair (`kernels/gb10/common/gated_delta_rule_wy4_woa.cu`) and the Ada K-generic pair
  (`kernels/ada/common/gated_delta_rule_wyn_woa.cu`, row `gdn_woa_mtp`) reproduce Atlas's wy2/wy3/wyN expressions.

### MTP head v2: frequency-ranked draft vocabulary and device-resident propose chain (L11)

- **Cinference / NInfer** (Apache-2.0), commit 383e5db. Data: the token-count file `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64` (token counts by NInfer authors; source corpus not published), reduced to the top 131,072 ids. Translated: `tools/convert/proposal.py` `select_shortlist` (ranking with forced special tokens) and `add_proposal` (gather of the shortlisted head rows); `src/ops/kernel/speculative_round.cuh` `proposal_remap_token_ids_kernel` (row to token id after the draft argmax). Design only: the single-graph MTP round of `src/models/qwen3_5/program/speculative/mtp.cpp`. Atlas: `crates/spark-model/data/mtp_vocab_rank_qwen35.{u32,json}`, `crates/spark-model/data/gen_mtp_vocab_rank.py`, `crates/spark-model/src/layers/mtp_head/{vocab_rank.rs, ranked_head.rs, propose_chain.rs}`, `kernels/ada/common/draft_vocab_remap.cu`. Notice in `NOTICE`; details in `MTP-DRAFT-VOCAB-ATTRIBUTION.md`.

- **Cinference speculative acceptance** (Apache-2.0, satellitedown/cinference commit 383e5db, derived from
  Neroued/ninfer), `src/ops/kernel/speculative_round.cuh` (`speculative_accept_greedy_drafts_kernel`) and
  `src/ops/kernel/sampling_device.cuh`: one-hot draft acceptance with residual sampling, the splitmix64 uniform keyed by
  seed, position and purpose, and the inverse-CDF pick with an excluded token. Translated for
  `kernels/ada/common/spec_accept_sample_ada.cu` and its CPU model `crates/spark-model/src/port/spec_accept/reference.rs`
  (row `mtp_device_accept`). Attribution: `SPEC-ACCEPT-ATTRIBUTION.md`.
- **Speculative sampling**: Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative
  Decoding", ICML 2023 (arXiv:2211.17192); C. Chen et al., "Accelerating Large Language Model Decoding with Speculative
  Sampling" (arXiv:2302.01318). The acceptance rule (accept a draft with probability min(1, p/q), else sample the
  normalized residual) whose one-hot-q case the device path implements.

- **Cinference prompt lookup** (Apache-2.0, satellitedown/cinference commit 383e5db, derived from Neroued/ninfer),
  `src/models/qwen3_5/program/speculative/prompt_lookup.{h,cpp}`: hashed 8/4/2-token suffix windows, period
  continuation, the decayed per-window acceptance estimate with its long-match floor, and lookup rounds (a near-certain
  copy verified alone). Translated for `scheduler/copy_draft/{index.rs, estimator.rs, policy.rs}` (row `copy_draft`).
  Attribution: `COPY-DRAFT-ATTRIBUTION.md`.
- **TensorFold copy index** (Apache-2.0; pre-0.6.0 code also MIT, ashhart/TensorFold commit 3885dce),
  `zig/src/core/copy_index.zig`: the latest-earlier-occurrence copy contract and its plain-scan oracle, and filling the
  MTP verify chain from a copy. Translated for `scheduler/copy_draft/{index.rs, legacy.rs}`.
- **KVMem** (arXiv:2609.04852; repository carries no license, idea only, no source read): 16-token copy windows,
  backward match extension, the source cursor kept across accepted rounds, and the per-request yield gate (EWMA 0.25,
  advantage 1.3, two cold rounds, a probe after eight refusals). Independent implementation in
  `scheduler/copy_draft/{estimator.rs, policy.rs}`.
- **Prompt lookup decoding** (A. Saxena, 2023, https://github.com/apoorvumang/prompt-lookup-decoding): the public
  technique behind Atlas's `--ngram-speculative` proposer (`ngram.rs`), which the copy drafter's legacy index serves.

### DFlash and the Qwen3.5-4B drafter (L14)

- Chen, Liang and Liu, "DFlash: Block Diffusion for Flash Speculative
  Decoding", arXiv:2602.06036 (2026). The method behind `--dflash`; Atlas's
  `BlockDiffusionDraftHead` is its own implementation. Port sprint use: the
  Ada 4B pairing, `kernels/ada/holo-3.1-4b/MODEL.toml` `[dflash]`.
- `z-lab/Qwen3.5-4B-DFlash` (https://huggingface.co/z-lab/Qwen3.5-4B-DFlash),
  Z Lab and Modal, Apache-2.0, about 1.27 GB BF16, not bundled. Its
  `config.json` is a test fixture
  (`crates/spark-model/src/weight_loader/dflash_loader/fixtures/`). The model
  card reports speedups on B200 with SGLang (up to 4.60x at concurrency 1
  with block 16, 2.61x at concurrency 32 with block 8); those numbers are the
  upstream's, on another card and engine, and are not Atlas measurements.

### Cinference: L2 cache-policy hints and the fused RMSNorm FP8 producer (L15)

- Source: Cinference, https://github.com/satellitedown/cinference, commit
  `383e5db`, derived from NInfer (https://github.com/Neroued/ninfer).
  Apache-2.0.
- Used: the `createpolicy.fractional.L2::evict_first` weight-load policy and
  its CUDA 13.1 ptxas caveat (`src/ops/common/memory.cuh:72-99`), small
  snippet adapted, in the `_l2ef` entry points of
  `kernels/ada/common/fp8_gemv_ada.cu` and `fp8_gemv_ada_batchm.cu`; the idea
  of an RMSNorm that hands the FP8 GEMM its quantized input
  (`src/ops/linear/fp8/fp8_rmsnorm_quantize.cu:1-98`), implemented
  independently for Atlas's 1x128 groups in
  `kernels/ada/common/rmsnorm_quant_fp8_ada.cu`.
- Rows: `l2_evict_first_weights`, `rmsnorm_fp8_quant_fused` (both off). The
  batch-M verify GEMV (`ffn_fp8_verify_gemv`) is Atlas-native.
- Notice: `NOTICE`; notes: `CINFERENCE-L2-HINT-ATTRIBUTION.md`.

### Affine 4-bit route: TensorFold and the MLX format (L17a, L17b, L17c)

- TensorFold (https://github.com/ashhart/TensorFold, commit 3885dce), Copyright
  2026 TensorFold contributors, Apache-2.0 (code written before 0.6.0 also MIT).
  Source of the Ada affine kernels (`kernels/ada/common/affine_frag_ada.cuh`,
  `affine4_lane_ada.cu`, `affine_pack_ada.cu`, `affine_prefill_ada.cu`,
  `affine_swiglu_sums_ada.cu`, `affine_n_ada.cu`, `affine_codes_ada.cuh`) and of
  the host code translated into `crates/spark-model/src/port/affine/`. The
  microtest gates (float64 bound, five-run byte repeat, planted controls,
  distinct weight copies for DRAM-bound timing) follow its
  `docs/kernel-contract.md` and recipes. Files, line ranges and changes:
  `TENSORFOLD-ATTRIBUTION.md`. TensorFold reports no lane-GEMV number on sm_89;
  Atlas's numbers come from `examples/affine_ada_microtest.rs` on the card.
- MLX affine quantization (https://github.com/ml-explore/mlx, Apple, MIT): the
  weight format the route stores (groups of 32, 64 or 128 inputs, a bf16 scale
  and bias per group, value = scale * code + bias, 4-bit codes packed eight to a
  u32, low nibble first) and the quantizer's rounding rules, as TensorFold
  implements them (`zig/src/core/affine4_host.zig`). Credited for the format;
  no MLX source is copied.

### Marlin W4A16 kernels for Ada (L18a)

- vLLM (https://github.com/vllm-project/vllm, commit `8a36f08`, Apache-2.0):
  `csrc/quantization/marlin/{marlin_template.h, dequant.h, marlin_mma.h,
  marlin_dtypes.cuh, marlin.cuh, kernel.h, generate_kernels.py}`,
  `marlin.cu:96-147` and `csrc/core/scalar_type.hpp`, copied into
  `kernels/ada/common/marlin_ada.cu` and the six `marlin_*_ada.cuh` headers.
  Marlin files: Copyright (C) Marlin.2024 Elias Frantar, adapted from
  https://github.com/IST-DASLab/marlin (Apache-2.0).
- Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, Dan Alistarh,
  "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language
  Models", arXiv:2408.11743.
- See `VLLM-MARLIN-ATTRIBUTION.md`.

### llama.cpp / ggml quant engine and the keep-packed GGUF route (L19)

- llama.cpp (https://github.com/ggml-org/llama.cpp, MIT, Copyright (c)
  2023-2026 The ggml authors; text in `LICENSES/llama.cpp-MIT.txt`): the CUDA
  quant engine vendored in `kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/`
  (`ggml-common.h` block layouts and `kvalues_iq4nl`, `vecdotq.cuh`,
  `mmq.cuh`, `quantize_impl.cuh`), and `ggml/src/ggml-quants.c`
  (`dequantize_row_iq4_xs`, `dequantize_row_iq4_nl`, `dequantize_row_q5_K`)
  and `ggml/src/ggml-cuda/{mmvq.cu, quantize.cu, convert.cu}`, translated
  for the Ada keep-packed IQ4_XS / Q4_K / Q6_K route:
  `kernels/ada/common/{kquant_q8_1_ada.cu, kquant_mmvq_q4_ada.cu,
  iq4_xs_mmq_ada.cu, dequant_iq4xs_q5k_bf16_ada.cu}`,
  `crates/spark-runtime/src/weights/gguf/dequant_cpu/iq4.rs` and
  `crates/spark-model/src/port/gguf_q4/`.
- The GGUF format and the IQ4_XS, Q4_K, Q5_K and Q6_K quantization types
  are llama.cpp's (`docs/` and `gguf-py/` of the same repository).
- See `GGUF-IQ4-ATTRIBUTION.md`.

### EXL3 trellis weights: ExLlamaV3, TensorFold and QTIP (L20a, L20b)

- ExLlamaV3 (https://github.com/turboderp-org/exllamav3), Copyright (c) 2025
  Turboderp, MIT (`LICENSES/ExLlamaV3-MIT.txt`). Source of the EXL3 format
  Atlas reads: trellis int16 tiles with a circular bitstream, the `3inst`,
  `mcg` and `mul1` codebooks, 1 to 8 bits in half-bit steps, and
  `W = diag(suh) H W_q H diag(svh)` with 128-point Hadamard rotations. No
  ExLlamaV3 source is copied.
- TensorFold (https://github.com/ashhart/TensorFold, commit 3885dce),
  Copyright 2026 TensorFold contributors, Apache-2.0 (code written before 0.6.0
  also MIT). Its independent Zig reader and reference decoder
  (`zig/src/core/exl3_format.zig`, `exl3_load.zig`, `exl3_layout.zig`,
  `exl3_prepare.zig`) are translated into
  `crates/spark-runtime/src/weights/exl3/`; its format description
  (`docs/recipes/exl3.md`) is the spec the reader's fixtures are written from;
  its numpy oracle (`zig/tests/fixtures/exl3_oracle.bin`, read from a local
  clone, not vendored) is the reader's independent check. Its EXL3 CUDA kernels
  are lane L20b's source. Files, line ranges and changes: `EXL3-ATTRIBUTION.md`.
- QTIP: A. Tseng, Q. Sun, D. Hou, C. De Sa, "QTIP: Quantization with Trellises
  and Incoherence Processing" (NeurIPS 2024; arXiv:2406.11235, cited from
  memory, check before publishing). The trellis-coded, Hadamard-incoherent
  quantization method EXL3 follows. Cited for the method; no code.

- **Qwen3.8-27B on a 16 GB card: the arczhi RTX 5060 Ti recipe and the MiaAI one-click kit** (sprint report
  77, sections 2, 7.1 and 12). Cited as evidence only, nothing copied: the recipe's measured memory budget and
  its `-ot token_embd.weight=CPU` placement (the embedding table read off the device) motivate the
  `embed_host_offload` row (`crates/spark-model/src/port/embed_host/`, an independent mapped-memory design);
  the MiaAI planner's rule that a measured ceiling overrides the formula only downward (`tools/profiles.py`,
  MIT, Copyright (c) 2026 Mia'a AI Lab; idea only) is the rule of
  `crates/spark-server/src/main_modules/serve_phases/preflight/measured_ceiling.rs`. The bench rows R0 to R12
  of `docs/porting/qwen3.8-27b-ada.md` follow report 77 section 12.5. The weights the target serves are
  Unsloth's GGUF of Qwen3.8-27B (Apache-2.0, Qwen team), downloaded by the owner, never redistributed.

### Host-RAM KV tier core (L26a)

- LMCache (https://github.com/LMCache/LMCache, commit `3aefae1`, Apache-2.0):
  object states and read locks (`l1_manager/base.py:132-188`), watermark
  eviction (`eviction_controller.py:160-195`), in-flight admission
  (`prefetch_controller.py:1114-1160`), tail-first LRU (`lru.py:72-95`),
  eviction-aware lazy store (`eviction_aware.py:60-140`), chained block keys
  (`token_hasher.py:233-262`), load-versus-recompute gate (`kv_load_policy.py`).
  Design only, no code copied.
- vLLM (https://github.com/vllm-project/vllm, commit `8a36f08`, Apache-2.0):
  CPU KV offload worker (`vllm/v1/kv_offload/cpu/gpu_worker.py:543-731`), ARC
  policy (`vllm/v1/kv_offload/cpu/policies/arc.py:28-60`), simple offload
  manager (`vllm/v1/simple_kv_offload/manager.py`). Design only, no code copied.
- Cinference (https://github.com/satellitedown/cinference, commit `383e5db`,
  Apache-2.0, derived from NInfer): the context-cost calibration acceptance rule
  (`bench/README.md:103-175`), applied to the tier's restore cost gate.
- KVMem, arXiv:2609.04852: read leases during DMA, publish after the full
  write, and the heat (selection frequency with periodic halving) policy.
  Independent implementation from the paper's description; the kvmem
  repositories carry no license and were not read.
- N. Megiddo and D. S. Modha, "ARC: A Self-Tuning, Low Overhead Replacement
  Cache", USENIX FAST 2003 (the ARC policy and the naive reference in the tests).
- Atlas files: `crates/spark-runtime/src/kv_host_tier/`. See
  `KV-HOST-TIER-ATTRIBUTION.md`.

- **Quest** (J. Tang, Y. Zhao, K. Zhu, G. Xiao, B. Kasikci, S. Han, "Quest: Query-Aware Sparsity for Efficient
  Long-Context LLM Inference", ICML 2024, arXiv:2406.10774): per-page key min/max summaries and the query-aware upper
  bound `sum_c max(q_c * kmin_c, q_c * kmax_c)` that ranks pages for decode attention. Independent implementation in
  `kernels/ada/common/attn_page_sparse_ada.cu` and `crates/spark-model/src/port/sparse_quest/` (row
  `attn_page_sparse`, approximate, off).
- **arczhi/llama-tierkv** (MIT, the ggml license, `LICENSES/llama.cpp-MIT.txt`),
  `scripts-5060ti/proto/proto_page_sparse.cu`: the RTX 5060 Ti page-sparse prototype whose correct bound (not the
  pager's `q * max(kmin, kmax)`) the port follows, as described in the port sprint's report 77 (section 5.2, BR-4);
  no source read. Attribution: `QUEST-PAGE-SPARSE-ATTRIBUTION.md`.

- **KVMem query-time bounded KV view** (arXiv:2609.04852; the kvmem/kvmem-llama.cpp repository carries no license,
  so it is idea only and no source was read): a per-page mean of the pre-RoPE keys captured at write time, the
  post-norm pre-RoPE query rows of the last 512 tokens of the final user message, page-softmax block scoring with a
  GQA mean or max-then-renormalize, and a picker that keeps sink, recent and query-span blocks and fills the budget by
  score in chronological order. Independent implementation in `crates/spark-runtime/src/kv_cache/sketch.rs`,
  `crates/spark-runtime/src/kv_select/`, `kernels/ada/common/{kv_page_sketch_ada.cu, kv_page_score_ada.cu}` and
  `crates/spark-model/src/port/kv_view/query_capture.rs` (lane L31; wiring and row `kv_view_selection` are L32's).

### RetroInfer retrieval attention (L33 host side; L34a, L34b kernels)

- RetroInfer: "RetroInfer: A Vector Storage Engine for Scalable Long-Context
  LLM Inference", arXiv 2505.02922 (VLDB 2026), and its code in
  microsoft/RetrievalAttention (https://github.com/microsoft/RetrievalAttention,
  commit 03f912c, MIT, Copyright (c) Microsoft Corporation; text in
  `LICENSES/RetroInfer-MIT.txt`). `cache_hub/kmeans.py`,
  `cache_hub/retroinfer_cache.py` and `config/config.py` are translated into
  the Rust host side and CPU oracle in
  `crates/spark-model/src/port/retro/{config.rs, oracle.rs, oracle_eval.rs,
  update.rs}`; the selector, lifecycle, shadow mode and verify union are
  Atlas-native designs on that algorithm. Notes: `RETROINFER-ATTRIBUTION.md`,
  `docs/porting/retro-attn.md`.
- RetrievalAttention: "RetrievalAttention: Accelerating Long-Context LLM
  Inference via Vector Retrieval", arXiv 2409.10516 (NeurIPS 2025): the
  out-of-distribution query argument and the partial-softmax merge the
  estimation zone generalizes. No code from it (the repository has none for it).
- **XAttention** (R. Xu, G. Xiao, H. Huang, J. Guo, S. Han, "XAttention: Block Sparse Attention with Antidiagonal
  Scoring", arXiv:2503.16428, 2025): antidiagonal scoring of attention blocks with a stride, and block selection by a
  cumulative softmax-mass threshold, for block-sparse prefill. The reference repository has no detected license, so it
  was not read and no code was copied: an independent implementation from the paper's description in
  `kernels/ada/common/prefill_block_sparse_ada.cu` and `crates/spark-model/src/port/sparse_prefill/` (row
  `prefill_sparse_attn`, APPROXIMATE, off on every target). Atlas's choices: 128-token blocks on absolute positions,
  the strided logit as the mean of its S products, forced sink, diagonal and previous blocks, and a cut found by a
  deterministic binary search on the FP32 mass.

### Host-resident window desk and windowed MTP KV (L36)

- **TierKV pager** of `arczhi/llama-tierkv` (MIT, Copyright (c) 2023-2026 The ggml authors), `kv-paging-design.md`, as described in the port plan's report 77 section 2.5. Design only, nothing fetched or copied: a VRAM desk of head sinks (2,048 positions), a recent window and recalled pages over a host-RAM store; staging at most once per 1,024 positions and only after evictions; MTP draft contexts excluded from paging. Atlas reads the desk through a compacted paged block table instead of staging pages into ordinary KV cells. Atlas: `crates/spark-model/src/port/kv_window/`, `crates/spark-runtime/src/kv_host_tier/pages.rs`, `kernels/ada/common/kv_page_fetch_ada.cu`. Notice in `NOTICE`.
- **LMCache** (Apache-2.0), commit 3aefae1, `lmcache/v1/distributed/l1_manager/base.py`: host copies are published only after their copy completed and their memory is kept while a copy touches it. Design only. Atlas: `crates/spark-runtime/src/kv_host_tier/pages.rs`, `crates/spark-model/src/port/kv_window/fetch.rs`.
- **Quest** (Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", ICML 2024): the page upper-bound scorer the desk's recall seam (`selector::PageScorer`) is built for; the scorer itself is lane L30's.

### LFM3 support (L37, L38)

- Liquid AI, LFM3-1.7B (https://huggingface.co/LiquidAI/LFM3-1.7B, released 2026-10-27): 24
  layers of gated short convolutions (width 7), sliding-window and global GQA attention with a
  per-head sigmoid output gate, sandwich RMSNorms and a dense SwiGLU FFN. The weights are under
  the LFM Open License v1.0 (commercial use not licensed at USD 10M annual revenue or more) and
  are not part of Atlas. Facts used: config, tensor names, sampling defaults, the XML tool-call
  format (model card and `tokenizer_config.json`).
- HuggingFace transformers `models/lfm3` (Apache-2.0, Copyright 2026 The HuggingFace Team) and
  vLLM `vllm/model_executor/models/lfm3.py` (Apache-2.0, Copyright contributors to the vLLM
  project, commit 8a36f08): the reference implementations the loader and the config parser are
  translated from. Files, upstream lines and changes: `LFM3-ATTRIBUTION.md`.
- llama.cpp `src/models/lfm3.cpp` (MIT, Copyright (c) 2023-2026 The ggml authors): cross-check
  of the tensor map and of the zero-frequency partial RoPE. No source copied.

### Sharp chat template

- **peculiar-ragdoll/Qwen-Sharp-Chat-Templates** (Apache-2.0), `chat_template.jinja`, `template_version qwen3.8-froggeric-v22.5.0`, obtained 2026-11-12: embedded verbatim (plus one header comment) as the opt-in chat template `--chat-template sharp`. Based on **froggeric/Qwen-Fixed-Chat-Templates** v22.5 (Apache-2.0), which is based on the Qwen3.5 chat template (Apache-2.0, Alibaba Cloud / Qwen team). Atlas: `crates/spark-server/assets/chat_templates/qwen-sharp-v22.5.0.jinja`, `SHARP-CHAT-TEMPLATE-ATTRIBUTION.md`.
- The upstream project's reported gain (fewer thinking tokens per correct answer on 27B-class models) is its own claim, measured by it and not by Atlas. Atlas's own measurement, on the 4B and 0.8B models with thinking on, is `bench/sharp_eval.py`; until that has run, the template is an opt-in answer-style feature and never part of a kernel or decode benchmark.
- Evaluation sets named by `bench/sharp_eval.py` (question ids only are committed, no dataset text): **MMLU-Pro** (TIGER-Lab/MMLU-Pro, MIT; Wang et al. 2024, arXiv:2406.01574) and **GSM8K** (openai/gsm8k, MIT; Cobbe et al. 2021, arXiv:2110.14168). Statistics: McNemar's exact test and a paired bootstrap, computed by the script itself.

- **ARC technical report** ("ARC: Learned Model Routing for Long-Horizon Agentic Workflows", Rayline Team, 2026,
  https://static.rayline.ai/rayline/reports/arc/latest), section 2.2 and Appendix A: the expected-utility rule
  `V * p - sum_k n_k * rho_k`, the prefer-local threshold, the switching cost at user-turn boundaries
  `L_t * [max(rho_in, rho_write) - rho_read]+`, and decision timing (task start, subagent start, compaction, user
  turn); the Sage (V = 4.5) and Swift (V = 0.1) presets of the Apache-2.0 ARC-1.0 release. Formulas only, no code:
  `crates/atlas-route-policy/src/{utility.rs, prefer_local.rs, timing.rs, types.rs}`. Attribution:
  `RAYLINE-ROUTING-ATTRIBUTION.md`.
- **Cascade routing** (J. Dekoninck, M. Baader, M. Vechev, "A Unified Approach to Routing and Cascading for LLMs",
  arXiv 2024, ICML 2025; eth-sri/cascade-routing, Apache-2.0, commit 9da6cf7): the expected-utility formalism for
  routing under a cost trade-off. Idea only, no code read into Atlas: `crates/atlas-route-policy/src/utility.rs`.
- **rayline-core** (atlasfutures/semantic-router, branch vsr-next, commit f6d0393, Apache-2.0):
  `rayline-core/execution/{events.go, deployment_wire.go, capacity.go}`, event inference from consecutive requests
  and context-capacity facts with measured, bounded and estimated provenance. Translated from Go for
  `crates/atlas-route-policy/src/{events.rs, capacity.rs}`.
- **Raydock** (rayline-ai/raydock, commit 20f02d5, Apache-2.0), `crates/rayline-cli/src/hybrid_profile.rs`: the
  `min_p_pass` name and the utility versus prefer-local policy choice. Naming only, no code copied.

<!-- @port:citations -->
