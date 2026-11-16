# Consumer GPU lane: status and measured results

This branch adds an Ada Lovelace target (`sm_89`, brought up on an RTX 4060 Ti 16 GB) and a set of opt-in optimizations for decode speed, 4-bit weights, KV caching and long context. Everything new is default-off except the rows `kernels/ada/HARDWARE.toml` declares on for Ada: `ffn_bf16_cublas_prefill`, `attn_gated_bf16_batchm`, `memory_fit_consumer` and `decode_split_silu` (on for every target). Other hardware targets are unchanged. This page says, lane by lane, what has a GPU result, what does not, and what failed. For the Ada bring-up itself, read `consumer-nvidia-ada.md`.

## Status labels

| Label | Meaning |
|---|---|
| MEASURED | A GPU measurement or a passing GPU gate exists. |
| EXPERIMENTAL | Code and tests exist, but there is no GPU result. |
| REJECTED | Measured, no gain, stays off. |
| KNOWN LIMITATION | Works partly. The evidence column says what fails. |

A label says what was observed, not how good the result is: MEASURED includes lossy rows. Rates: `C=1` is one user (tokens per second for that user). `C=8` is eight users at once, and the figure is the total across all eight. Details are under "Measurement method".

## Hardware and model

All GPU numbers come from one RTX 4060 Ti 16 GB (AD106, 34 SMs, 288 GB/s spec, about 260 GB/s sustained in a read probe), WSL2 Ubuntu, CUDA 13.3. The model is Qwen3.5-4B unless a row says otherwise. Nothing on this page is a projection.

## Stacked configuration, final result

The stack is int4 on all three projection groups (FFN, GDN, attention) plus MTP speculative decoding with 1 draft, `min_tokens_gpu_argmax` and `mtp_width_policy`. Enable it with `AVAROK_DENSE_AFFINE4_FROM_BF16=1 AVAROK_GDN_AFFINE4_FROM_BF16=1 AVAROK_ATTN_AFFINE4_FROM_BF16=1 AVAROK_MIN_TOKENS_GPU_ARGMAX=1 AVAROK_MTP_WIDTH_POLICY=1 --speculative --num-drafts 1`. All four arms below ran with `AVAROK_MIN_TOKENS_GPU_ARGMAX=1`, so the comparison is like for like. Total tokens per second across all users (at C=1 that is one user):

| Concurrent users | BF16 | FP8 FFN | FP8 FFN + MTP (3 drafts) | Stack | Stack vs FP8 FFN |
|---|---|---|---|---|---|
| 1 | 29 | 40 | 40 | 98 | 2.45x |
| 2 | 56 | 77 | 75 | 185 | 2.4x |
| 4 | 107 | 143 | 117 | 294 | 2.1x |
| 8 | 195 | 222 | 134 | 381 | 1.7x |

- Verdict: the stack gained at every concurrency. All four arms show only the 4 known coherence failures (see Known limitations), none new.
- The stack is lossy. Mean KL against BF16 is 0.0337, top-1 agreement 14.4%, and 1 of 24 prompts never diverges. FP8 FFN on its own: KL 0.0029, 6 of 24 never diverge. The stack stays opt-in.
- `min_tokens_gpu_argmax` removes a per-step logits copy that only fixed-length requests (`min_tokens` equal to `max_tokens`, as the benchmark sends) pay. Ordinary requests do not pay it, so do not expect its share of these gains in normal use.

## Lane tables

Each row gives the lane id and what it is, its label, the evidence with its setup, and the switch that enables it. `AVAROK_*` names are environment variables. Percentages compare the row against its own base arm in the same test, so compare a row's two numbers, not rows with each other. KL is mean KL divergence of top-20 token probabilities against that base arm, unless it says BF16.

### Foundation: Ada target, BF16 default, FP8 FFN, memory fit

| Lane | Label | Evidence | Enable |
|---|---|---|---|
| Ada target and BF16 prefill default (cuBLASLt for the BF16 FFN prompt path) | MEASURED | 4B BF16 prefill 794 to 3,684 tok/s (single user, 3,921-token prompt, 5-run median, new default vs previous dispatch); decode 28.4 and 28.6 tok/s single user | On for Ada. Off: `AVAROK_FFN_BF16_CUBLAS_PREFILL=0` |
| FP8 dense FFN (`dense_fp8_from_bf16`) | MEASURED | 4B prefill 4,135 tok/s, decode 38.4 tok/s (single user, same prompt, 5-run median); peak VRAM 11,492 MiB vs 13,596; KL 0.0016 against BF16 | `AVAROK_DENSE_FP8_FROM_BF16=1` |
| 16 GB memory fit, base (MTP refusal fix, NVFP4 fallback skip, smaller recurrent ring) | MEASURED | 4B FP8 + MTP boots at 8K context, peak 11,753 MiB; 4B at batch 8: ring 1.59 GB (was 3.19), peak 15,775 MiB | On for Ada |
| 16 GB memory fit, later sizing rows (sized MTP headroom, skip unused MTP block, draft head rows, `--no-vision`) | EXPERIMENTAL | On in every GPU run, but no standalone before and after number | `AVAROK_MEMORY_FIT_CONSUMER=0` turns it off |
| Qwen3.5 GGUF config mapping | MEASURED | Real 27B file: 866 tensors, 0 unmapped; the live server builds its model config from the GGUF metadata (qwen3_5, 64 layers, hidden 5120) | Automatic for GGUF |
| llama.cpp-compatible API (`/completion`, `/tokenize`, `/props` and others) | EXPERIMENTAL | 2,768 server tests pass; live, the benchmark uses `/tokenize` and reads `/props` on every serve; the conformance script and `/completion` grammar, stream and `/embedding` have not run on a GPU | Added routes |
| Workaround for a suspected CUDA 13.3 `ptxas` code-generation issue in `fp8_gemv_ada_batchm_7` | MEASURED (the fix); cause SUSPECTED, no independent reproducer | Illegal-address faults fixed (`cb4dea889`); memcheck 0 errors; scan of 216 PTX files found 0 hazards | Always on |

### Instrumentation and routing study (L04 to L07)

| Lane | Label | Evidence | Enable |
|---|---|---|---|
| L04: routing gate study (Rust `route_gate` tool) | EXPERIMENTAL | 73 tests pass; goldens 3 of 26 within 5e-3 (reference tolerance not met); encoder GPU parity 5 of 5, cosine 0.99994; the AUROC study was not run | Standalone binary, no serve switch |
| L05a: per-request evidence fields, speculation off, terse template switch | MEASURED | Evidence fields appear in GPU run records; the speculation-off switch is used by the identity checks; the terse switch is exercised by the `sharp-no-terse` arm | Request fields `atlas_evidence`, `atlas_confidence`, `atlas_speculative: false`, `chat_template_kwargs.terse` |
| L05b: confidence summary | EXPERIMENTAL | Declared off on every target; no GPU run with it on | `AVAROK_CONF_SUMMARY=1` |
| L06a: scheduler instrumentation (MTP round timing, gauges) | MEASURED | 32 timing lines captured; a verify round costs 1.29x a plain step (about 1.2x predicted) | `AVAROK_MTP_TIMING=1` |
| L06b: D-Cut v2 cost model for MTP draft length | EXPERIMENTAL | Row exists, default off, no GPU run | `AVAROK_MTP_DCUT_COST=1` |
| L07: microbenchmark examples | MEASURED | Sustained read 260 GB/s (spec 288); pinned host to device 6.5 GB/s, device to host 6.6 GB/s; NVFP4 W4A16 GEMV at 85% of the bandwidth ceiling at M=1; int4 W4A16 14 gates pass | `cargo run --example` with `gpu-examples` |

### Decode, speculation and FP8 kernels (L08 to L16)

| Lane | Label | Evidence | Enable |
|---|---|---|---|
| L08: GPU backend extensions (graph update, batched memcpy, events, host register) | MEASURED | Found and fixed an illegal-address fault when a batched copy targets registered host memory (`5d13b8768`); probe arms and round-trip gates pass after | Used by L09 and L25 to L27 |
| L09: CUDA graph topology classes and exec update | REJECTED | C=8 total: 122.0 and 121.3 tok/s vs base 123.2 and 119.9 (+0.1%) | `AVAROK_GRAPH_EXEC_UPDATE=1` |
| L10: GDN write-on-accept state updates | REJECTED | Oracles bit-equal on GPU for K=2 to 16; C=8 total: 120.9 and 122.2 tok/s vs base (0%) | `AVAROK_GDN_WOA`, `AVAROK_GDN_WOA_MTP` |
| L11a: MTP draft vocabulary ranked by frequency | REJECTED | Draft acceptance 53.3% (12,508 of 23,478) vs 55.8% (11,448 of 20,502) for id order; greedy output equal on 7 of 8 prompts; no speed number isolated | `--mtp-vocab-rank freq` |
| L11b: graphed MTP propose chain | REJECTED | C=8 total: 122.6 and 122.7 tok/s vs base 123.2 and 119.9 (+0.9%); bit-identical to off | `AVAROK_MTP_PROPOSE_GRAPH=1` |
| L12: device-side speculative acceptance | MEASURED | Oracle: 40,000 picks all equal; C=1 36 to 39 tok/s (+6.8%); C=8 total 74 to 127 (+72.8%, against a slow host-side base, so the gain is inflated); KL 0.00003 | `AVAROK_MTP_DEVICE_ACCEPT=1` with `--speculative` |
| L13: round plan and copy drafter (drafts copied from earlier context, always verified) | MEASURED | Identity 24 of 24 prompts; 6 of 6 copy cells; +17.0% on the copy-heavy cell. A later re-run showed 5 to 7 coherence failures in 3 cells (usual count is 4), not triaged | `--copy-draft off,chain,rounds,auto` or `AVAROK_COPY_DRAFT` |
| L14: DFlash drafter (Qwen3.5-4B-DFlash) | EXPERIMENTAL | No speed result. The plain run is refused for memory (13.32 GB needed vs 8.08 GB free); the fit option boots with a 1,024-token context cap | `--dflash`, `AVAROK_DFLASH_FIT=1` |
| L15a: FP8 batch-M verify and decode GEMV, L2 cache hints | MEASURED | Microtests pass (95 and 28 rows), memcheck 0 errors; decode GEMV C=1 36 to 38 (+5.4%), C=8 total 190 to 199 (+4.5%), KL 0.00000; verify GEMV 184 us at M=8; L2 hints have no serve result | `AVAROK_FFN_FP8_DECODE_GEMV=1`, `AVAROK_FFN_FP8_VERIFY_GEMV=1`, `AVAROK_L2_EVICT_FIRST=1` |
| L15b: fused RMSNorm and FP8 quantization in prompt processing | REJECTED | Microtest 25 of 25 pass. Time to first token: 3,878 tokens 882 to 885 ms, 5,349 tokens 1,202 to 1,202 ms, 8,045 tokens 1,844 to 1,838 ms; outputs identical on 24 of 24 prompts | `AVAROK_RMSNORM_FP8_QUANT=1` |
| L16: tensor-core chunked GDN prefill | MEASURED | Microtest at T=16384: 10,119 us vs 30,564 us for the FLA kernel (3.02x) and 22,919 us scalar. Serve (FP8 FFN, FP8 KV, 1 slot, one run per arm, about 5,349-token prompt): time to first token 1,262 to 1,097 ms (-13.0%), about 4,238 to 4,876 prompt tok/s (+15%); KL 0.00089 vs off, top-1 54.3%, 10 of 24 prompts never diverge | `AVAROK_GDN_PREFILL_TC=1` |

### 4-bit weights and the 27B (L17 to L24)

| Lane | Label | Evidence | Enable |
|---|---|---|---|
| L17a: affine int4 core (GEMV, pack, quantizer, W4Linear) | MEASURED | All-projection int4, no speculation: C=1 39 to 64 tok/s (+64.7%), C=8 total 209 to 314 (+50.4%), KL 0.0388. Microtest: 254 gates; 15 control oracles misfired from a test bug (fixed, clean rerun not recorded) | `AVAROK_DENSE_AFFINE4_FROM_BF16=1`, `AVAROK_GDN_AFFINE4_FROM_BF16=1`, `AVAROK_ATTN_AFFINE4_FROM_BF16=1` |
| L17b: affine prompt GEMMs (BF16, W4A8 int8) | EXPERIMENTAL | No GPU result, no int4 prefill number | `AVAROK_AFFINE4_PREFILL_INT8=1` |
| L17c: affine code widths 2, 3, 5, 6, 8 | EXPERIMENTAL | No GPU result | `--w4-bits N` with `--w4-group 64` |
| L18a: Marlin int4 kernels | MEASURED | Microtest exit 0: 76 row-invariant passes, 30 of 30 entries, 0 failures; gate-up u4b8 at M=1 366 us | Through L18b |
| L18b: Marlin host side (config, repack, RTN, W4Linear) | MEASURED | Sealed-panel KL 0.0203 against BF16; C=1 44.4 (affine) vs 46.0 (Marlin) tok/s; the paired comparison did not pass its gate; the RedHatAI W4A16 checkpoint boots and serves with only the known coherence failures, speed and KL unmeasured | `AVAROK_MARLIN_W4A16=1` |
| L19: GGUF IQ4_XS and Q4_K kept packed | MEASURED | 147 gates and controls, 0 failed; llama.cpp MIT license text and notice added; end to end on the 27B blocked (L24) | `AVAROK_GGUF_NATIVE_IQ4=1` |
| L20a: EXL3 reader | EXPERIMENTAL | No GPU or serve run | `AVAROK_WEIGHTS_EXL3=1` |
| L20b: EXL3 kernels and W4Linear | EXPERIMENTAL | `exl3_microtest` has no GPU result | `AVAROK_WEIGHTS_EXL3=1` |
| L21: int4 dense-FFN dispatch | MEASURED | vs FP8 + MTP (3 drafts) base: C=1 +14.0%, C=8 total +3.4%; KL 0.0113, top-1 97.9% | `AVAROK_DENSE_AFFINE4_FROM_BF16=1` |
| L22a: int4 GDN projection dispatch | MEASURED | Only inside the all-projection rows (L17a), no GDN-only test | `AVAROK_GDN_AFFINE4_FROM_BF16=1` |
| L22b: int4 attention projection dispatch | MEASURED | Same as L22a | `AVAROK_ATTN_AFFINE4_FROM_BF16=1` |
| L23a: int4 loader routes and decision tables | MEASURED | The int4 rows and the stack load through it; stack C=8 total 381 tok/s | `--w4-bits`, `--w4-group`, `--w4-clip-search` |
| L23b: int4 residency, preflight, NVFP4 wrapper, install, self-check | KNOWN LIMITATION | NVFP4 GEMV alone is at 85% of ceiling (L07); the three serve cells for the affine and NVFP4 wrapper failed and are not diagnosed | `AVAROK_DENSE_NVFP4_FROM_BF16=1` |
| L24: Ada 27B target, embedding offload, packed LM head | KNOWN LIMITATION | The 27B never boots: pre-flight needs 29.48 GiB vs 14.84 GiB free (the mixed-quant GGUF keeps only IQ4_XS, Q4_K and Q6_K packed, the rest expands to BF16). The refusal is correct | `--lm-head-packed`, `AVAROK_EMBED_HOST_OFFLOAD=1` |

### KV tiers, radix cache, sparse and retrieval attention (L25 to L36)

| Lane | Label | Evidence | Enable |
|---|---|---|---|
| L25: KV block IO, batched swap, pinned pool, SSM state LRU | KNOWN LIMITATION | Multi-turn reuse test: cache hit 0 on turns of 7,059 to 7,571 prompt tokens (floor 50%). Server log: prefix hit of 992 tokens (62 blocks) but no SSM snapshot, so everything is recomputed | `AVAROK_KV_BATCHED_SWAP=1`, `AVAROK_SSM_STORE_LRU=1`, `AVAROK_SSM_TIER=1`, `--pinned-pool-chunk-mb` |
| L26a: host-RAM KV tier core | MEASURED | Round trip at 576, 8,000 and 32,768 tokens: exit 0, 2,048 blocks byte-exact (debug build); tier on does not hurt decode (5 rows pass) | `AVAROK_KV_HOST_TIER=1`, `--kv-host-tier-gb N` |
| L26b: host KV tier glue, hooks, prefetch | KNOWN LIMITATION | Stores blocks (960 written, 9 lookups found them) but restores 0 on this hybrid model: every hit logs "no SSM snapshot". Restore probe, identity, time-to-first-token and decode checks failed | Same, plus `AVAROK_SSM_TIER=1`, `--kv-tier-writeback-mb-per-step` |
| L27: async restore, disk tier, warm restart | KNOWN LIMITATION | Crash fixed (L08); server arm 0 errors. Async test failed: nothing was parked in the async arm; restore-window p99 108 to 122 ms | `AVAROK_KV_RESTORE_ASYNC=1`, `AVAROK_KV_TIER_DISK=1`, `--kv-tier-dir`, `--kv-tier-disk-gb` |
| L28: radix LRU index and copy-on-write partial tail | MEASURED | Victim select 361 ns vs 11,729 ns (32.5x); copy-on-write fired 30 copies, 0 fallbacks (11 rows pass) | `AVAROK_RADIX_LRU_INDEX=1`, `AVAROK_PREFIX_COW_TAIL=1` |
| L29: sparse-attention seam | EXPERIMENTAL | No GPU result | Used by L30 to L34 |
| L30: Quest page-sparse decode (approximate) | EXPERIMENTAL | `page_sparse_microtest` and serve cells have no result | `AVAROK_ATTN_PAGE_SPARSE=1`, `--attn-sparse-topk-pages` |
| L31 and L32: bounded KV view core and wiring (approximate) | EXPERIMENTAL | No GPU result | `AVAROK_KV_VIEW=1`, `--kv-view-budget-tokens` |
| L33: RetroInfer host side | EXPERIMENTAL | Dense 64K needle baseline recalls 4 of 4; the shadow-mode run did not complete | `AVAROK_RETRO_ATTN=1` |
| L34a and L34b: RetroInfer k-means, zone selection, gather and estimation kernels | EXPERIMENTAL | `retro_build_microtest` and `retro_attn_microtest` have no GPU result | `AVAROK_RETRO_ATTN=1` |
| L35: block-sparse prefill (approximate) | REJECTED | 32K-token kernel microtest: 1.19x vs BF16 (100,910 vs 85,108 us) and 0.96x vs FP8 (59,658 vs 61,981 us); serve time to first token at 32K 18,155 to 17,077 ms (-5.9%, one run per arm) | `AVAROK_PREFILL_SPARSE_ATTN=1` |
| L36: window desk and windowed MTP KV | MEASURED | 4B at 32K, FP8 KV: 7 rows pass, only the 4 known coherence failures. The 27B at 262K was not run | `AVAROK_KV_WINDOW_DESK=1`, `--kv-window-tokens`, `--kv-window-head-tokens` |

### Other models, templates and routing code (L37 to L41)

| Lane | Label | Evidence | Enable |
|---|---|---|---|
| L37: LFM3 config, loader, FP8 route, chat template, tool parser | EXPERIMENTAL | The serve used to stop at model build (the checkpoint's embedding has 128,000 rows, the shape check expected the tokenizer's 125,033). The shape check now compares against the checkpoint's own vocabulary size and unit tests pass; the serve has not been re-run since the fix, so there is no speed or KL | `AVAROK_LFM3_FP8_ALL=1` for the FP8 route |
| L38: LFM3 short-conv kernels and layer | MEASURED | Short-conv microtest exact; the layer inside a serve has not run | Used by L37 |
| L39: Sharp chat template, jinja `startswith` and `endswith`, evaluation | REJECTED | MMLU-Pro pilot, 60 questions, 1 seed, 4B, thinking on: stock 53.3%, sharp 38.3% (-15.0 points, McNemar p 0.035), sharp-no-terse 51.7% (-1.7 points, interval [-11.7, +8.3]). Small sample, no benefit shown | `--chat-template sharp` or `sharp-no-terse` (`AVAROK_CHAT_TEMPLATE`) |
| L40: routing policy crate `atlas-route-policy` | EXPERIMENTAL | 4 test files including property tests; whole-workspace test run 7,125 passed. CPU only | Library crate, no serve switch |
| L41: model swap timing and residency metrics | EXPERIMENTAL | CPU-side metrics, no GPU run | `AVAROK_MODEL_SWAP_TIMING=1` |

### Additional decode rows

Each row is a separate switch, tested against its own base arm. C=1 is tokens per second for one user, C=8 the total across eight.

| Row | Label | Evidence | Enable |
|---|---|---|---|
| Tensor-core FFN kernel, 16 rows | MEASURED | Microtest 190 pass, memcheck 0 errors, flat 91 to 103 us for M=1 to 16 vs 158 to 166 us for the batch GEMV at M=8. Serve C=8 total: +14.6% then +3.4% on two runs of the plain-decode variant; the rerun of the 3-draft variant collapsed (-52% at C=8, -30% at C=1) and is treated as disturbed. Spread unresolved | `AVAROK_FFN_FP8_M16_TC=1` |
| Tensor-core FFN, 32-row version | MEASURED | Microtest 445 pass, memcheck 0 errors, 102 us at M=32. Serve C=8 total: +3.4% (`tc32mtp3`, 3 drafts) and +16.6% (`tc32mtp3all`; its C=1 is -3.7% with 6.3% noise) | Same switch |
| mtp_width_policy (MTP by live batch width) | MEASURED | C=1 -2.0% (inside noise), C=8 total 162 to 186 (+15.3%); KL 0.00004 | `AVAROK_MTP_WIDTH_POLICY=1` with `--speculative` |
| gdn_verify_gemv | MEASURED | 1 draft: C=8 total 164 to 175 (+6.8%). 3 drafts: C=1 30 to 51 (+68.5%, noise 9%), C=8 total 56 to 76 (+35.3%); KL 0.0003 | `AVAROK_GDN_VERIFY_GEMV=1` |
| lm_head_verify_gemv | MEASURED | C=1 58 to 57 (-0.5%), C=8 total 177 to 187 (+5.4%); KL 0.00000 | `AVAROK_LM_HEAD_VERIFY_GEMV=1` |
| min_tokens_gpu_argmax | MEASURED | C=1 36 to 38 (+6.9%), C=8 total 190 to 213 (+12.0%); output identical. Applies only to requests with `min_tokens` set | `AVAROK_MIN_TOKENS_GPU_ARGMAX=1` |
| FP8 GDN projections | MEASURED | C=1 39 to 45 (+16.4%), C=8 total 208 to 220 (+5.7%); KL 0.00147 | `AVAROK_GDN_FP8_FROM_BF16=1` |
| FP8 attention projections | MEASURED | C=1 39 to 40 (+3.7%), C=8 total 208 to 212 (+2.1%); KL 0.00107 | `AVAROK_ATTN_FP8_FROM_BF16=1` |
| FP8 coverage combined | MEASURED | Both FP8 conversions together (GDN and attention): C=1 +28.0% (39 to 50), C=8 total +10.9% (208 to 230), KL 0.00179. With MTP 1 draft: C=1 51 to 74, C=8 total 165 to 240, KL 0.00168. With tensor-core FFN and `min_tokens_gpu_argmax`, no speculation: C=1 39 to 49, C=8 total 208 to 290 (+39.5%), KL 0.00179; its base lacked `min_tokens_gpu_argmax`, so a like-for-like figure is not measured | The three FP8 rows above, plus E01 |
| Int4 with MTP | MEASURED | All-projection int4 + MTP 1 draft: C=1 53 to 100 (+90.4%), C=8 total 169 to 361 (+113.5%); KL 0.0401. Lossy; task accuracy not measured | The L17a switches with `--speculative --num-drafts 1` |
| FP8 LM head | MEASURED | FP8 FFN route, 8 slots: C=1 38.9 to 42.8 (+10.0%); C=8 total 208.5 to 152.1 (-27.1%); KL 0.00067; peak VRAM about +0.6 GB. Helps one user, hurts eight; not a default | `--lm-head-dtype fp8` |
| Fused SwiGLU quantization kernel for the FP8 FFN prompt path | EXPERIMENTAL | `fp8_ada_swiglu_microtest` and an A/B have no GPU result | Part of the FP8 FFN route |

### Profile of the final stacks

An `nsys` capture of 35-second windows at 8 streams (CUDA graphs traced per node, so profiled speed is not benchmark speed): the GPU is busy 95.6% of the time on the int4 + MTP stack and 94.3% on the FP8 coverage + tensor-core stack; idle time between kernels is under 5%. On the int4 stack, `dense_gemv_bf16_batchm` takes 21%, the three 4-bit GEMV kernels (`affine4_lane_*`) about 44% together, `gated_delta_rule_decode_f32` 13% and `paged_decode_attn` 9%. Remaining gains must come from the kernels, not from host or graph overhead. The 21% BF16 kernel is consistent with the BF16 LM head, but that attribution is an inference, not checked.

## Measurement method

- **Noise bar.** An A/B test runs a row against a base arm in the same run, 2 repetitions, with a 3 percent noise bar. A difference inside 3 percent is noise. One row reports 6.3 percent and another 9 percent for its own C=1 noise.
- **C=1 and C=8.** C=1 is one user alone: tokens per second for that user. C=8 is eight users at once: tokens per second summed over all eight (divide by 8 for per user). A concurrency of 2 or 4 is read the same way.
- **Identity gate.** "Identical" means greedy output matched the base arm token for token on 8 test prompts at C=1. Rows that change numerics are not held to it and report KL instead.
- **KL against BF16.** Mean KL divergence over the top-20 token probabilities. For the stacked run it is against BF16. For row tests it is against the base arm of that test and says so. 0 means identical.
- **Sealed panel.** A fixed panel of 50 streams of 512 positions (25,600 positions) that the arms are scored on, with bootstrap confidence intervals. Against BF16: affine FFN KL 0.01385 [0.01168, 0.01630], Marlin FFN KL 0.02031 [0.01755, 0.02335].
- **Coherence suite.** About 89 checks. Four fail identically in every build (see Known limitations). A row that adds failures beyond those four is not accepted.
- **Fixed-length requests.** The benchmark sends fixed-length requests. That makes every decode step copy logits to the host, which ordinary requests do not do. The stacked run and E03i report that effect explicitly.

## Known limitations

- **Host KV tier restores nothing on this hybrid model.** The tier stores blocks (960 written, 9 lookups found them) but restores 0. The 4B is a hybrid model: every hit logged "no SSM snapshot", and with the SSM tier switched on (`AVAROK_SSM_TIER=1`) a prefix hit still often had no recurrent-state snapshot at the matching boundary, so the whole prefix was recomputed. Restore results with that switch set are not yet available. See L25, L26b, L27.
- **Long context fails at large sizes.** Needle recall with BF16 KV: served at 31,650 prompt tokens (12.6 s), failed at 95,000 with "Inference cancelled" (HTTP 500). With FP8 KV: recalled at 95,684 tokens (51.9 s) and 159,807 tokens (112.4 s, 15.0 GB of 16 used), failed at 261,632 with the same HTTP 500. The needle grid recalled 100% (4 of 4 per cell) up to 162,816 tokens with FP8 KV. The failures are not diagnosed.
- **Four coherence-test failures appear on every build.** Name recall, empty system message, 5,000-character system message and temperature 0 fail identically in BF16, FP8 and every new row. The suite sends 10 to 500 `max_tokens` and the model's thinking budget is longer than that, so the reasoning uses the whole budget before any answer text. These are test-limit failures, not new regressions, but they mean the suite cannot judge those four behaviors.
- **The 27B has not been run.** The GGUF expands to 29.48 GiB against 14.84 GiB free and the pre-flight refusal is correct. No 27B speed, KL or accuracy number exists (L24).
- **int4 is lossy.** The stacked run has KL 0.0337 against BF16 and top-1 agreement of 14.4%. The int4 rows have KL 0.0388 to 0.0401 against their base arms. Task accuracy on real work (an evaluation set) has not been measured. Do not make it a default without that.
- **Some gains are not settled.** The tensor-core FFN gain is positive but its run-to-run spread is unresolved (between about 3 and 15 percent at 8 users, E01). The FP8 coverage 8-user figure includes a benchmark-method effect its base lacked (E07).
- **Not measured.** Quantized llama.cpp or any other engine at 4-bit; draft-depth sweep on recorded coding-assistant traces; a standalone memory A/B for E11b; DFlash speed (L14); EXL3, int8 prefill and affine width kernels on a GPU (L17b, L17c, L20a, L20b); sparse and retrieval attention (L29 to L34); LFM3 serving (L37).
- **Prefill numbers are not like for like.** The tensor-core GDN prefill rate (4,238 to 4,876 tok/s) used a 5,349-token prompt, one run per arm; the FP8 FFN figure of 4,135 tok/s used a 3,921-token prompt. Do not compare them directly.

## What is not in this pull request

- Nothing described here is uncommitted.
- Future work, not written or not committed: the EXL3 install arm for the 27B (the 3.0 bits per weight route would be about 9.5 GiB resident) and the mixed-type GGUF route that would let the 27B's UD-IQ4_XS file fit.
- Planned lanes that were never written: draft tree on host and device (W01, W02), sliding-window ring KV (W03), GDN slot state for verify (W04), two-stream GDN overlap (W05), an LFM3 endpoint (W06), and the routing front end pieces (X01 to X03, X05, X06).
- The benchmark harness that produced these numbers is a separate tool and is not part of this branch.
