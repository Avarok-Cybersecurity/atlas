# PR carve-up plan for Avarok PR #834 (EXL3 / QTIP trellis checkpoint support)

Source branch: `wip/exl3-upstream` @ `da09c7bcf` (== `wip/exl3-research` tip, == PR #834 head).
Base: `avarok/main` @ `ef8ca11d9` (2026-09-01, "#831 dflash: serve the record shape by default").
Branch shape: 11 non-merge commits + 2 merge commits. Code-only delta (crates/ + kernels/):
144 files, +19,474 / -381. Research artifacts: `.research/` 71 files, +17,381 lines, 6.07 MB
(none of it exists on main). PR #834 as filed: 215 files, +36,855 / -381.

Everything below was derived from `git log --stat`, `git diff`, and the CI workflow
definitions on the branch; the "measured" numbers are the ones in the commit messages
and PR body, not re-measured (no GPU use in this task).

---

## 0. Facts that constrain the carve-up (found while reading, worth knowing up front)

| # | Finding | Consequence |
|---|---------|-------------|
| F1 | `.github/workflows/file-size-cap.yml` is GREEN on main and goes RED on this branch: **8 non-allow-listed `.rs` files > 500 LoC** (table in section 2.9). | Every carved PR must split, not allow-list (repo rule). The splits are listed per PR below. |
| F2 | `.licenserc.yaml` requires `SPDX-License-Identifier: AGPL-3.0-only` on every `kernels/**/*.{cu,cuh}`. The 11 vendored files in `kernels/gb10/common/exl3_vendor/` carry `SPDX-License-Identifier: MIT` (correctly — they are turboderp's code). | The "SPDX license headers" CI job will fail PR-5 unless `kernels/gb10/common/exl3_vendor/**` is added to `paths-ignore`, exactly as `kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/**` already is. Add that line in PR-5. |
| F3 | `kernels/gb10/common/*.cu` compiles for EVERY gb10 model target (22 targets; 2 distinct flag sets after the common/model KERNEL.toml merge). Measured nvcc `--ptx` cost of the new files (sm_121f, -O3, strict `--Werror all-warnings`): `exl3_matmul.cu` **40.4 s** (211 entries, 27 MB PTX), `exl3_moe.cu` **18.6 s** (16 entries), `exl3_reconstruct.cu` 2.0 s (25 entries), `embed_from_argmax.cu` 0.3 s. | The "nvcc -> PTX (all gb10 targets)" CI job (took 9m19s on #821, 30-min timeout, 4 vCPU) grows by roughly 2 × (40+19+2) s ÷ parallelism — fine, but PR-5 and PR-6 are the ones that move it. |
| F4 | `ATLAS_TARGET_MODEL=* cargo check -p atlas-kernels` on the tip: **PASS, 74 s** on 20 workers, 507/4617 unique nvcc invocations (9.1× dedup; the pre-branch figure in the workflow comment is 449/3583). EXL3 PTX present for every target (`t*__exl3_matmul.ptx`, `t*__exl3_moe.ptx`, `t*__exl3_reconstruct.ptx`). | The "build all targets" rule holds for the whole branch; nothing model-specific shadows the new common kernels. |
| F5 | `kernels/strix/common/` is a curated symlink set (99 files) into `gb10/common/`; the EXL3 kernels are NOT symlinked. | SCALE/HIP release legs never see the vendored PTX inline asm. `launch_cooperative` is `bail!` under `cfg(atlas_scale)`. The **Windows CUDA release leg does** compile gb10 `common/` with `-std=c++17` + strict warnings — first place a Windows-nvcc warning in vendored code would surface; only the release-matrix run tells. |
| F6 | The `avarok/main` merge (`0c34c15ec`) is a clean auto-merge: `git merge-tree 40249f111 ef8ca11d9` produces the identical tree (`22dc24f7…`). It carries no manual content. | Drop it; rebasing onto main subsumes it. |
| F7 | The #821 merge (`e0d0efbcd`) is NOT clean: conflicts in `cuda_backend.rs` and `cuda_backend/gpu_impl.rs`, hand-resolved (main's `AllocRecord` ledger kept; #821's `forget_alloc -> Option<usize>` + `live_count_bytes()` grafted on; the trail moved into `alloc_ledger.rs`). #821 itself is still OPEN, `mergeable: CONFLICTING`, and its own CI fails the 500-LoC cap. No EXL3 code references `release_state`/`live_count_bytes`. | The resolution `git diff 0c34c15ec e0d0efbcd` (9 files, +195/-9) is exactly "#821 rebased on main". Push it to #821's branch and land #821 first, independently; the EXL3 series has no compile-time dependency on it (section 4). |
| F8 | `sm_count`, `create_event`, `record_event`, `stream_wait_event`, `event_synchronize` already exist on main's `GpuBackend`. The branch adds only `launch_cooperative`, `launch_cooperative_typed`, `set_kernel_max_dynamic_smem`, `KernelLaunch::cooperative()`, the per-backend kernel-handle cache, and the mock recording for them. | The runtime-plumbing PR is smaller than the commit messages suggest (~350 lines of runtime src). |
| F9 | Three mechanical extractions of PRE-EXISTING code are buried in the feature commits: `moe/forward_prefill.rs` -> `forward_prefill_topk.rs` (routing block, 40249f111), `qwen3_attention/prefill/paged_qkv.rs` -> `paged_qkv_lora.rs` (e61387092), `qwen3_ssm/…/ssm_batched.rs` -> `ssm_batched_log.rs` (e61387092). | Carve each out as the FIRST commit of its PR (or one tiny prep PR) so the reviewer sees a pure move followed by a small feature diff. |
| F10 | Both parity examples skip the real-tensor legs when `EXL3_REAL_DIR`/`.research/real_tensor` is absent (`legs.rs:167`, `exl3_reconstruct_parity.rs:159`). | The 820 KB real-tensor fixture need not be committed; the fetch script + env var is enough for the gate runner. |
| F11 | CPU gates on the tip (this task, scratch target dir, exact build env): `cargo build --release -p spark-server --bin spark` **PASS 161 s**; kernels(*) check **PASS 74 s**; spark-model / spark-runtime lib tests and `clippy --all-targets` — see section 6 for results. | Baseline is green before carving. |

---

## 1. Dependency graph of the changes

```
                 ┌──────────────────────────────────────────────────────────────┐
                 │  #821 QSA/PLE per-sequence device-state leak fix (d68696b1b)  │  independent
                 │  (needed OPERATIONALLY for long single-node EXL3 sessions;    │  (land first,
                 │   no compile-time coupling to anything EXL3)                  │   own PR)
                 └──────────────────────────────────────────────────────────────┘

 [V] fix(vision): ViT MLP width from fc1 tensor (da09c7bcf)          independent, 2 files

 [R] spark-runtime backend plumbing (from f1aa57395 + e61387092)     independent
     cuLaunchCooperativeKernel + cuFuncSetAttribute FFI; GpuBackend::launch_cooperative{,_typed};
     set_kernel_max_dynamic_smem; KernelLaunch::cooperative(); per-backend kernel-handle cache;
     mock recording (cooperative_launch_count, max_dynamic_smem_calls, kernel_lookups_snapshot)
                                  │
 [K1] reconstruct kernel + CPU reference + Exl3Weight + store dtypes (49887504e)   independent
      kernels/gb10/common/exl3_reconstruct.cu (24 K×cb instances + f16->bf16 [out,in] converter)
      weights/exl3.rs: is_exl3_f16_aux/.suh/.svh exemption, WeightDtype F16(aux)/UInt16/Int32,
      Exl3Weight::from_store/to_bf16, Exl3Codebook (.mul1 flag), cpu_ref (tile decode, both-side
      Hadamard), store_tests; examples/exl3_reconstruct_parity
                │
 [M] load-time materialization (043fe9e60 + the eb5862361 warning tweak)          needs K1
     weight_map/exl3_materialize.rs core (experts -> runtime NVFP4 triplet, dense -> BF16),
     detect_quant_format "exl3" -> ModeloptFormat, WeightStore insert/remove, serve_load hook,
     factory::build_model idempotent hook
                │
 [N] PLE n-gram row decoder + sharded trellis walk (059baeeb3 + eb5862361)        needs M, K1(cpu_ref)
     embed_from_argmax.cu::batched_embed_exl3, ops::sampling wrapper, PleLayer NgramRowFormat,
     qwen4_exp/ple.rs (monolithic + 128-shard layouts), register_exl3_ngram_sidecar,
     cpu_ref::decode_ngram_row + ring cross-test, parity ngram legs   ==> FIRST FULL BOOT (EP=2)
                │
 [X] native trellis matmul + lm_head served packed (f1aa57395)                      needs R, K1, M
     kernels exl3_matmul.cu + exl3_vendor/ (10 .cuh, MIT), ops/exl3_matmul{,/mgemm}.rs,
     model/lm_head_exl3.rs, factory install, exl3_graph_veto at decode/verify sites,
     native keep-packed policy in exl3_materialize (ATLAS_EXL3_NATIVE), K∈{2,4} gate,
     examples/exl3_native_parity (42 legs)
                │
 [E] native routed MoE experts (40249f111)                                          needs X (+N for a bootable gate)
     kernels exl3_moe.cu + exl3_moe_kernel/common.cuh, mgemm entries, hadamard_inner additions;
     ops/exl3_matmul/{moe_decode,moe_prefill,moe_prefill_overflow}; moe/{forward_exl3,
     forward_prefill_exl3,tables,ptr_table_build,exl3_tables_tests}; Exl3MoeState (slabs+locks+
     event fence, single dispatcher); veto in every verify site; nvfp4_detect resolve_key_variant;
     qwen4_exp probe/ffn; exl3_materialize_moe / moe_exl3 (ATLAS_EXL3_NATIVE_MOE)
                │
 [D] native GDN + attention dense linears + ONE shared Exl3LaunchState (e61387092)  needs E (as written)
     layers/exl3_dense{,/attn_dispatch}, ops/exl3_dense{,/launch_state,/stage}, qwen3_ssm/init_exl3,
     qwen3_attention/init_exl3 + qkv_exl3, qwen35 loader exl3_dense_arms, qwen4_exp/exl3_dense,
     exl3_materialize_dense(+tests), lm_head moved onto the shared state, MoE tables moved onto it
     (ATLAS_EXL3_NATIVE_DENSE, sub-gates _GDN/_ATTN)
```

Coupling notes the graph hides:

* **X depends on M, not just K1**: the "keep packed under `ATLAS_EXL3_NATIVE`" decision is a
  branch inside `materialize_exl3_impl` (exl3_materialize.rs +183 in f1aa57395). Native serving is
  a policy of the materializer, so M must land before X.
* **D depends on E only through the launch state migration**: e61387092 introduces
  `Exl3LaunchState` and then REWRITES E's `Exl3MoeState` tables onto it (`moe/tables.rs` -116,
  `ptr_table_build.rs` 55 lines churn) and moves the LM head's private locks buffer onto it. If the
  carve-up is allowed to restructure (not this task), introduce `Exl3LaunchState` in X (the LM head
  is its first user) so E builds on it directly and D's tables churn disappears. Otherwise keep the
  order X -> E -> D and accept the churn.
* **R is independent of K1**: the reconstruct kernel is an ordinary `launch`; nothing in K1 needs
  cooperative launch. R and K1 can be reviewed in parallel.
* **N is the first PR that can boot the real checkpoint**, and only at EP=2 as measured
  (materialized experts are 67.95 GB of NVFP4 triplets; the single-node fit arrives with E).
  Pre-KV for the compat path on one GB10 would be ≈ 43.7 − 30.67 + 67.95 ≈ 81 GB — plausibly fits
  at util ≥ 0.75 but was never measured; do not promise it.

---

## 2. Proposed PR sequence (8 PRs + #821 on its own)

Ordering: #821, PR-0, PR-1 and PR-2 are independent and can go up together. PR-3 .. PR-7 are a
strict chain. Every PR: builds with the exact env, `cargo test --release -p spark-model --lib`
(+ `-p spark-runtime --lib` where runtime is touched), `clippy --release --all-targets` on the three
crates, `rustfmt --edition 2024` on touched files, file-size cap green, plus the reviewer's 5 GPU
gates for any `crates/` change. "GPU gate" below = what the reviewer must additionally run on a GB10.

### PR-0 — fix(vision): size the ViT MLP from the fc1 tensor, not vision_config
* **Commits**: da09c7bcf (code half only; drop the two `.research/boot/*.sh` edits).
* **Files** (2, +29/−4): `crates/spark-model/src/weight_loader/qwen35.rs`,
  `crates/spark-model/src/layers/vision_encoder/enc_impl/vit_block.rs` (allow-listed, 525 -> 525).
* **Gates**: CPU: existing lib tests (suggest adding a unit test on the width-resolution rule:
  tensor width wins over config, warn-logged, equal is silent). GPU: the existing vision fidelity
  gate on a non-EXL3 vision model (tensor == config ⇒ must be byte-identical) — this is a
  no-op there by construction. The EXL3 measurement (red circle: "A black triangle on a white
  background" -> "I see a red circle.") needs PR-4+ to reproduce; cite it, don't gate on it.
* **Risk**: low. Behavior changes only when the fc1 tensor disagrees with `intermediate_size`.
* **Default-off**: not flag-gated; inert for every checkpoint whose tensor matches its config.
* **Why first**: 2 files, real silent-corruption fix, zero coupling.

### PR-1 — spark-runtime: cooperative launch, dynamic-smem opt-in, kernel-handle cache
* **Commits**: runtime slices of f1aa57395 (`cuda_backend.rs` +27, `gpu_impl.rs` +72, `gpu.rs`
  +68, `gpu/mock.rs` +81, `kernel_args.rs` +104) and of e61387092 (`cuda_backend.rs` +12,
  `gpu_impl.rs` +11: the `kernel_cache: Mutex<HashMap<String,u64>>` + hit path in `kernel()`).
* **Files** (5): `crates/spark-runtime/src/{gpu.rs, gpu/mock.rs, kernel_args.rs,
  cuda_backend.rs, cuda_backend/gpu_impl.rs}`. ~350 lines net.
* **Splits required (F1)**: `gpu.rs` 490 -> 558 and `gpu/mock.rs` 463 -> 544 both cross the cap
  in this PR; `gpu_impl.rs` 489 -> ~560. `GpuBackend` is one trait and `impl GpuBackend for
  AtlasCudaBackend` is one impl block, so split the NON-trait items: move `KernelLaunch` (the
  typed-args builder, incl. `cooperative()`) out of `gpu.rs` into `gpu/launch.rs`; move the mock's
  recording accessors (`launches_snapshot`, `kernel_lookups_snapshot`, `max_dynamic_smem_calls`,
  `cooperative_launch_count`, …) into `gpu/mock/recording.rs`; move the kernel-cache lookup + audit
  body out of the trait fn into an inherent helper in `cuda_backend/kernel_lookup.rs`.
* **Gates**: CPU: spark-runtime lib tests — the three new mock-backed tests in
  `kernel_args.rs` (`launches_stay_eager_unless_cooperative_is_requested`,
  `cooperative_routes_to_the_cooperative_path_with_args_intact`,
  `max_dynamic_smem_opt_in_is_recorded_per_kernel`). GPU: a production boot of the target model
  with gate-OFF flags + the 5 standard gates — this is the ONLY PR in the series that changes a
  hot path for every model (`GpuBackend::kernel` now caches by `module::func`; cache hits skip the
  `kernel_audit` row and the `cuModuleGetFunction`). The startup audit still sees the first
  (miss) lookup of every kernel, so `kernel_audit::classify_failures` semantics are unchanged —
  reviewer should confirm that reading.
* **Risk**: low-medium (the cache is cross-cutting; the cooperative path is unreachable until PR-5).
* **Default-off**: `launch_cooperative` has no caller; `set_kernel_max_dynamic_smem` has no caller;
  the cache is behavior-preserving for callers that store handles at init (all of them today).
* **Alternative** if reviewers prefer plumbing with its consumer: fold this PR into PR-5 and land
  the kernel-handle cache alone as a 40-line PR. The cache is the one piece that deserves isolation.

### PR-2 — EXL3 trellis reconstruct kernel + independent CPU reference + store dtypes
* **Commits**: 49887504e (code half).
* **Files** (8, +1,677/−11): `kernels/gb10/common/exl3_reconstruct.cu` (600),
  `crates/spark-runtime/src/weights/exl3.rs` (736 at this commit), `weights.rs` (+27),
  `weights/loader/load_fns.rs` (+15: `.suh/.svh` F16 exemption in all three ingest paths),
  `weights/name_utils.rs` (+12), `fast_weights/header.rs` (+16: `I16`->UInt16, `I32`->Int32,
  F16 store-legal for aux only; the old `assert!(from_safetensors_str("F16").is_err())` goes),
  `crates/spark-model/Cargo.toml` (example registration, `required-features = ["cuda","gpu-examples"]`),
  `crates/spark-model/examples/exl3_reconstruct_parity.rs` (278). Plus the fetch script
  (`scripts/dev/exl3_fetch_tensor.py`, moved from `.research/fetch_exl3_tensor.py`).
* **Splits required**: `weights/exl3.rs` is 736 here (915 on the tip). Land it already split:
  `weights/exl3.rs` (predicates, `Exl3Weight`, `Exl3Codebook`, `k_bits_from_trellis_dim`, launch
  wrappers `reconstruct_had_{f16_device,bf16}` ≈ 340) + `weights/exl3/cpu_ref.rs` (≈ 280) +
  `weights/exl3/store_tests.rs` (≈ 115), using `#[path]` submodules like `exl3_matmul/`.
* **Gates**: CPU: runtime lib tests (285 -> 285 at the time; the store_tests cover codebook flag,
  MSB-first bit order, K from trellis dim). GPU: `cargo run --release --features cuda,gpu-examples
  --example exl3_reconstruct_parity` — 3 shapes × 8 (K, codebook) legs × both stages
  BYTE-IDENTICAL GPU vs CPU, 1-bit negative controls, plus the real tensor leg with
  `EXL3_REAL_DIR` pointing at the fetched `turboderp/Qwen3.8-Flash-Next-exl3` tensor. Kernel CI
  (`*`) picks up the new common file (+2 s/flag set).
* **Risk**: low. Ingest change: safetensors `I16`/`I32` tensors were previously a load ERROR
  (`from_safetensors_str` rejected them) and now land as raw containers; F16 tensors whose name
  matches `.suh`/`.svh` are no longer rounded to BF16. No existing supported checkpoint ships
  either, so no serving path changes. The wire/peer manifest now admits F16 — RDMA reviewers.
* **Default-off**: nothing dispatches the kernel; `Exl3Weight::from_store` has no caller yet.
* **Licensing**: `exl3_reconstruct.cu` is an AGPL-headered PORT of MIT code with in-file
  attribution ((c) 2025 turboderp). MIT permits this provided the notice is retained; reviewer
  should read the header once. `.licenserc.yaml` passes as-is for this PR.

### PR-3 — EXL3 loader: materialize trellis linears into loader-native tensors (compat path)
* **Commits**: 043fe9e60 (code half) + the 10-line materializer warning downgrade from eb5862361.
* **Files** (7, +352/−3): `crates/spark-model/src/weight_map/exl3_materialize.rs` (272 here),
  `weight_map.rs` (+3), `quant_format/mod.rs` (+23: `quant_method: "exl3"` -> `ModeloptFormat`,
  loud warning if raw trellis tensors survive = call-order bug), `factory/build.rs` (+15,
  allow-listed 900 -> 915), `crates/spark-runtime/src/weights.rs` (+15: `WeightStore::{insert,
  remove}`), `weights/exl3.rs` (+16), `crates/spark-server/src/main_modules/serve_load.rs` (+11,
  allow-listed: hook before preflight + quant detection).
* **Gates**: CPU: model lib tests — `no_exl3_is_noop`, `routes_experts_to_triplet_and_attention_to_bf16`
  (mock GPU), the EP expert name-parse lock (suffix-agnostic filtering); runtime 286/286.
  GPU: (a) a production NVFP4 boot of the target model + 5 gates — the pass must be a no-op
  (`store_has_exl3 == false`); reviewer checks the serve_load scan cost in the boot log.
  (b) The EXL3 checkpoint does NOT boot yet (PLE table undecoded — the pass warns) — say so in
  the PR body; the boot gate belongs to PR-4.
* **Risk**: low. The one new always-on step is the store scan in `serve_load`; everything after
  it is behind `store_has_exl3`.
* **Default-off**: a store with no trellis tensors is untouched (unit-tested).
* **Note for the body**: state the double-quantization limitation (EXL3 K bpw -> BF16 -> NVFP4)
  and that experts occupy NVFP4 footprint regardless of source bitrate — this PR is the
  compatibility path, PR-5..7 are the fidelity/memory path.

### PR-4 — EXL3 PLE n-gram row decoder (decode-on-gather) + sharded trellis walk: first full boot
* **Commits**: 059baeeb3 + eb5862361 (code halves).
* **Files** (11, ≈ +690/−32): `kernels/gb10/common/embed_from_argmax.cu` (+76:
  `batched_embed_exl3` — ring-state extraction, mul1 decode, row scale, per-head bias, BF16 out),
  `crates/spark-model/src/layers/ops/sampling.rs` (+33 wrapper), `layers/ple.rs` (+2),
  `layers/ple/layer.rs` (+76: `NgramRowFormat::{Bf16, Exl3}` + gather branch),
  `weight_loader/qwen4_exp/ple.rs` (+104 then +51/−13: EXL3 route when the deferred trellis is
  present, K from words/row, monolithic `[320M,61]` AND 128 × `shard_{i}.trellis [2500012,41]`
  layouts into the same segmented row cache), `weight_map/exl3_materialize.rs` (+143:
  `register_exl3_ngram_sidecar` — defers the trellis for the row cache, uploads `head_bias` with
  exact f16 bits, renames sidecar id tables to the PLE loader's names),
  `crates/spark-runtime/src/weights/exl3.rs` (+100: `cpu_ref::{ngram_words_per_row,
  decode_ngram_row}` + closed-form ring cross-test), `serve_load.rs` (+7),
  `examples/exl3_reconstruct_parity.rs` (+76 ngram legs). Plus `scripts/dev/exl3_check_real_ngram.py`
  + `exl3_ngram_codec.py` (the independent third implementation, from `.research/`).
* **Splits required**: `layers/ple/layer.rs` 498 -> 573 (of which +21 is #821; +55 is this PR) —
  move the EXL3 gather branch to `ple/layer_exl3.rs`. `exl3_materialize.rs` — put the sidecar
  registration in `weight_map/exl3_ngram_sidecar.rs` from the start. `weights/exl3.rs` — the ngram
  CPU reference goes in `weights/exl3/ngram_ref.rs`.
* **Gates**: CPU: runtime 288/288 (2 new ngram CPU-ref tests: LSB-first-per-u16 bit order — a
  DIFFERENT order from the 16×16 tile format — and the two-derivation ring agreement); model
  642/642. GPU: parity ngram legs (GPU vs CPU BIT-IDENTICAL at K=4 and K=6);
  `exl3_check_real_ngram.py` against the real `ngram_embedding.safetensors` (data offset 5400,
  not 0); **the full boot**: `turboderp/Qwen3.8-Flash-Next-exl3 @ 2.05bpw_h4_ng4`, EP=2
  (gx10-9959 + dgx-00), CTX 8192, util 0.6 — measured 72.9/73.0 GB pledged and honored,
  37,341 linears materialized in ~30 s/rank, 320M-row K=4 table 26.2 GB packed, cold gather 256
  misses = 14 ms, warm 512/512 hits = 96 µs, TTFT 718 ms warm, ~20 tok/s, temp=0 coherent
  ("The capital of France is Paris."), `finish_reason: stop`. Two boxes required.
* **Risk**: medium. The BF16 `NgramRowCache` is stated untouched (byte-agnostic `row_stride`) but
  `PleLayer` gains a format enum on the hot gather path — the 5 gates on a BF16-PLE model are the
  regression check. `ATLAS_PLE_CACHE_SLOTS` must be ≥ tokens × 16 for a 32K prefill (PR body).
* **Default-off**: `NgramRowFormat::Exl3` only when the deferred trellis sidecar is registered;
  no EXL3 store ⇒ nothing registered ⇒ `Bf16` path exactly as before.

### PR-5 — EXL3 native trellis matmul: vendored fused GEMM/GEMV/mgemm, lm_head served packed
* **Commits**: f1aa57395 minus its runtime slice (PR-1) and minus `.research/`.
* **Files** (≈ 40, ≈ +5,500): kernels `exl3_matmul.cu` (211 at this commit: 80 gemm × shapes
  1–4, 72 mgemm pointer-table MoE form, 48 gemv, 3 converters; Blackwell-reachable
  instantiations only, cb0 dropped) + `exl3_vendor/{codebook, exl3_compat, exl3_devctx, exl3_dq,
  exl3_gemm_inner, exl3_gemm_kernel, exl3_gemv_kernel, exl3_kernel_map, hadamard_inner,
  ptx}.cuh` (MIT, verbatim, adaptations listed in each header); model
  `layers/ops/exl3_matmul.rs` (384) + `exl3_matmul/mgemm.rs` (188), `model/lm_head_exl3.rs`
  (372), `factory/build.rs` (+109: probe modules, install `set_lm_head_exl3`; allow-listed,
  915 -> ~1002 — consider `factory/exl3.rs`), `model/types.rs` (+8), `impl_a1.rs`/`impl_a3.rs`,
  `trait_impl/{decode_a, decode_a2, decode_b2, lm_head_batched, verify_b, verify_c, verify_c2}`
  (graph-capture veto + lm_head arm), `quant_dispatch.rs` (+16), `weight_map/{exl3_materialize
  (+183 native keep-packed policy, K∈{2,4} gate), expert, nvfp4_detect (+15), quant_helpers,
  quantized}`, `weight_loader/qwen4_exp/probe.rs`, `quant_format/mod.rs`; runtime
  `weights/exl3.rs` (+5); `examples/exl3_native_parity/{main, legs, truth, util, bench}.rs`
  (1,070). `.licenserc.yaml`: add `kernels/gb10/common/exl3_vendor/**` to `paths-ignore` (F2).
  `kernels/gb10/common/exl3_vendor/README.md`: upstream repo + commit SHA + file list +
  adaptation log (replaces the `.research/exllamav3_ref` snapshot references in the headers).
* **Splits required**: `exl3_materialize.rs` crosses 500 here (272 + 143 + 183) — the native
  policy (`exl3_native_enabled`, `exl3_native_serves{,_with}`, `exl3_native_supported`) goes to
  `weight_map/exl3_native_policy.rs`, tests to `exl3_materialize_tests.rs`.
* **Gates**: CPU: 291 runtime lib tests (mock cooperative), model lib tests. GPU:
  `exl3_native_parity` **42 legs PASS** — rotation stage BIT-EXACT vs CPU FWHT (pin:
  `RS = f32::from_bits(0x3db504f3)`; the decimal literal is 1 ULP high), gemv/gemm rel-RMS
  ~3e-4 across every K/cb/shape vs derived gates, negative controls blow up, real checkpoint
  tensor, mgemm 4-expert routing smoke; native weight fidelity vs materialized-NVFP4: 2.9e-4 vs
  1.9e-3 (~6.5×). Boot A: `ATLAS_EXL3_NATIVE=1` (lm_head only, ~0.86 GB saved) — coherent temp=0,
  byte-identical across runs, every decode token through the cooperative GEMV; 36.4 µs/launch at
  [2560->10240] K=4 (32.8 at occ=2); the clean `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` refusal
  at occ=3. Boot B: gate-OFF, must equal the PR-4 reference. Kernel CI `*` (+40 s/flag set).
  SPDX job green only with the F2 ignore line. Windows CUDA release leg (F5).
* **Risk**: medium-high. Cooperative kernels can never be graph-captured, so the veto touches
  every capturing site; the split-K locks GEMM is kept off co-dispatched side streams by the
  K∈{2,4} GEMV-servable gate. Reviewer focus: the veto sites, the `Exl3LmHead` private locks
  buffer (moved onto the shared state in PR-7 — or, if restructuring is allowed, introduce
  `Exl3LaunchState` here), `kernel_args.rs` typed cooperative packing.
* **Default-off**: `ATLAS_EXL3_NATIVE` unset ⇒ `lm_head_exl3 == None` ⇒ every veto term is
  `false` ⇒ non-native path byte-identical (asserted by Boot B).
* **Optional shrink**: move `exl3_matmul/mgemm.rs` + the mgemm parity leg to PR-6 (they have no
  caller until then); keep the kernel file whole so the vendored set is reviewed once.

### PR-6 — EXL3 native routed MoE experts (single-node fit)
* **Commits**: 40249f111 minus `.research/`. First commit = the mechanical
  `forward_prefill.rs -> forward_prefill_topk.rs` routing extraction (F9), no behavior change.
* **Files** (≈ 50, ≈ +5,900/−240): kernels `exl3_moe.cu` (139, 16 instances, separate module
  so a missing module fails at load) + `exl3_vendor/{exl3_moe_kernel, exl3_moe_common}.cuh`,
  `exl3_matmul.cu` (+212), `hadamard_inner.cuh` (+208), `exl3_compat.cuh` (+11); model
  `layers/ops/exl3_matmul/{moe_decode (335), moe_prefill (424), moe_prefill_overflow (199)}.rs`,
  `layers/moe/{forward_exl3 (288), forward_prefill_exl3 (280), forward_prefill_topk (114),
  tables (+332), ptr_table_build (+300), exl3_tables_tests (133), mod, init, forward*.rs (+5..17
  each)}`, `layer/transformer_layer.rs` (+10 `exl3_graph_veto`; allow-listed), `layers/mod.rs`
  (+12), `qwen3_attention/trait_impl.rs` (+9), `qwen3_ssm/trait_layer.rs` (+10),
  `trait_impl/{verify_b, verify_c, verify_c2, verify_d, verify_e, verify_e2 (+10: the OR over
  layers), verify_fused}`, `weight_loader/qwen4_exp/{ffn (+101), probe, ple, mod}`,
  `weight_map/{exl3_materialize (+364), exl3_materialize_moe (338), moe_exl3 (120), nvfp4_detect
  (+172), weight_map.rs}`; runtime `weights/exl3.rs` (+79/−21); examples `legs_moe (398),
  legs_moe_prefill (438), legs_moe_prefill_debug (319)`, main/util/bench/legs edits.
* **Splits required**: `nvfp4_detect.rs` 466 -> 601 — `resolve_key_variant` + its positive/
  negative-control tests to `weight_map/nvfp4_detect/exl3_keys.rs`. `exl3_materialize.rs` grows
  +364 here; with the PR-4/PR-5 splits done, the MoE policy is already in `exl3_materialize_moe.rs`
  — keep the core under 500 by moving the `Exl3MaterializeStats` reporting out.
* **Gates**: CPU: 664 lib tests incl. `exl3_tables_tests` (dense EP-local pointer tables with −1
  remote indices; per-layer K/cb uniformity with atomic keep-or-materialize), `resolve_key_variant`
  ± controls, `moe_gate_off_experts_materialize_exactly_as_before`. GPU: **45 parity legs PASS**
  (gemv/gemm/mgemm/3×-mgemm decode/fused prefill/EP sentinel/overflow skew — the skew leg gates at
  the fused-tier tolerance rel_rms 1.9e-3 vs 5.0e-3 on the BF16 path it replaced); single-node boot
  `ATLAS_EXL3_NATIVE=1 ATLAS_EXL3_NATIVE_MOE=1`: 73,728 projections packed = **30.67 GB vs
  67.95 GB** NVFP4 triplets (37.3 GB saved), 43.7 GB pre-KV, 1.12M KV tokens at util 0.6,
  48/48 layers + lm_head native, temp=0 coherent, **11.0 tok/s** C=1 over 300 tokens, 3.3K
  prompt TTFT 6.1 s cold / **214 ms** prefix-cache hit (3,280/3,286 reused; SSM/PLE/QSA aux
  restore coherent); gate-OFF boot identical; `compute-sanitizer` run on the shared-expert
  triplet path (the 4×-OOB CUDA-700 it fixes); a C≥2 mixed prefill+decode soak (two streams,
  the event fence). Kernel CI `*` (+19 s/flag set). Serve note: `--ssm-cache-slots 16` too small
  for agentic traffic, use 32–64.
* **Risk**: HIGH — the most stateful PR: `Exl3MoeState` (~240 MB slabs + locks inside the util
  pledge) admits ONE dispatcher via host in-flight claim + device event fence across the prefill
  and decode streams (two partially co-resident spin-barrier kernels would deadlock); graph veto
  ORed into six capturing verify sites + the decode sites, deliberately separate from
  `decode_graph_unsupported` so QSA/PLE verify-graph behavior with gates off stays byte-identical;
  load-path fixes to the qwen4_exp namespace probe and `nvfp4_detect` (a Bf16Raw misdeclaration
  read a triplet as BF16 `[n,k]`).
* **Default-off**: `ATLAS_EXL3_NATIVE_MOE` unset (validated at load together with
  `ATLAS_EXL3_NATIVE`) ⇒ `exl3_native_active() == false` ⇒ the early-returns in every MoE
  forward are not taken, `nvfp4_detect` per-key resolution is not engaged, experts materialize
  exactly as in PR-3 (unit-tested).
* **Also**: the EP contract (dense EP-local tables, −1 remote) is enforced, but the EP=2 native
  boot is NOT in the commit's measurements — say "single-node validated; EP=2 native unmeasured".

### PR-7 — EXL3 native GDN + attention dense linears + one shared launch state
* **Commits**: e61387092 minus its runtime slice (PR-1) and `.research/`. First commits = the two
  mechanical extractions (F9): `paged_qkv.rs -> paged_qkv_lora.rs`, `ssm_batched.rs ->
  ssm_batched_log.rs`.
* **Files** (≈ 55, ≈ +5,100/−390): model `layers/exl3_dense.rs` (469) + `exl3_dense/attn_dispatch.rs`
  (287), `layers/ops/exl3_dense.rs` (514) + `exl3_dense/{launch_state (238), stage (190)}.rs`,
  `ops/exl3_matmul/mgemm.rs` (+88), `qwen3_attention/{init_exl3 (87), init, types, decode/
  attention_forward (+39; allow-listed), decode/attention_forward_{kv,oproj}, prefill/{paged_qkv,
  paged_qkv_lora, cache_skip_qkv, paged_oproj, mod}, trait_impl{.rs, /multi_seq/{qkv, qkv_exl3 (62),
  o_proj, mod}}}`, `qwen3_ssm/{init_exl3 (119), init, mod, ssm_forward, trait_decode_batched
  (allow-listed 1281 -> 1296), trait_decode_multi_seq/{ssm_batched, ssm_batched_log}, trait_layer,
  trait_prefill_helper, trait_prefill_proj}`, `model/lm_head_exl3.rs` (onto the shared state),
  `layers/moe/{ptr_table_build, tables}` (onto the shared state), `weight_loader/qwen35/
  load_layers{.rs, /attention_arms, /exl3_dense_arms (136), /linear_attn_arms (+124/−?)}`,
  `weight_loader/qwen4_exp/{exl3_dense (140), ffn, probe, mod}`, `weight_map/{exl3_materialize
  (+157), exl3_materialize_dense (389), exl3_materialize_dense_tests (458), ssm_qwen35}`;
  kernels `exl3_matmul.cu` (+41); examples `legs_dense (340), legs_dense_attn (371),
  legs_dense_gdn (376)`, main.
* **Splits required**: `layers/ops/exl3_dense.rs` 514 — move `Exl3DenseWeight` (~70 lines) to
  `ops/exl3_dense/weight.rs`. `linear_attn_arms.rs` 548 -> 604 is allow-listed but +56 in an
  allow-listed file will be noticed; the new arm is `exl3_dense_arms.rs`, so the growth should be
  the dispatch lines only.
* **Gates**: CPU: 683 lib tests incl. `exl3_materialize_dense_tests` (458 lines). GPU: **139 parity
  legs PASS** (dense shapes at m ∈ {1,4,8,64,700} incl. row batching at the slab capacity, the
  strided qkv+z pair writing the fused `[Q|K|V|Z]` row (ld 16384, Z at column 10240) with
  poison-checked pad columns, q gated-interleave through `deinterleave_qg`, negative codebook
  control); boot `ATLAS_EXL3_NATIVE=1 _MOE=1 _DENSE=1`: BF16 linears 332 -> 176, **1.34 GB vs
  5.35 GB**, pre-KV 43.7 -> 39.0 GB (+21% KV tokens), answers TOKEN-IDENTICAL to the gate-OFF
  binary, 3.6K prompts read their embedded codes, C=4 distinct 3.6K prompts co-dispatched through
  the shared state (no bleed, no deadlock), decode **12.6–13.0 tok/s** (+15–18%), cold TTFT
  @3.6K +10% (two cooperative GEMMs replace one cuBLASLt GEMM for the in_proj pair — known
  lever: a large-M reconstruct-to-BF16 prefill tier); gate-OFF boot identical to the PR-6
  reference (332 linears, 43.7 GB, same answers). Sub-gates `ATLAS_EXL3_NATIVE_GDN=0` /
  `_ATTN=0` each opt a family back out — run both.
* **Risk**: HIGH — this PR touches the attention and SSM decode/prefill paths that EVERY
  Qwen3.5-family model runs (early-returns in `attention_forward`, `paged_qkv`, `ssm_forward`,
  `trait_decode_batched`, multi_seq qkv/o_proj), and the qwen35 loader arms. The 5 standard gates
  on a non-EXL3 Qwen3.5/3.6/3.8 model are the real regression check. `Exl3LaunchState` is anchored
  WEAKLY per process so loader (layers) and factory (LM head) land on one instance and a hot-swap
  builds a fresh one; a second host thread BLOCKS on the section (co-dispatched prefill at C≥2)
  rather than being refused; a stream change waits on the previous section's fence.
* **Default-off**: `ATLAS_EXL3_NATIVE_DENSE` unset ⇒ no `Exl3GdnWeights`/`Exl3AttnWeights`
  installed ⇒ every new branch is an `if let Some(..)` miss; the transposed twins and runtime
  BF16->NVFP4 requant stay exactly as on main.
* **Optional split** (9 PRs total): PR-7a = dense linear + launch state + GDN (the measured decode
  win, sub-gate `_GDN`), PR-7b = attention (`_ATTN`, decode-neutral, the loader-arm changes).
  `attn_dispatch.rs`, `init_exl3.rs` (attention), `qkv_exl3.rs`, `exl3_dense_arms.rs` and
  `legs_dense_attn.rs` are cleanly 7b.

### 2.9 File-size cap — the 8 violations and which PR must fix each

| File | main | tip | Attribution | Fix in |
|------|-----:|----:|-------------|--------|
| `spark-model/src/weight_map/exl3_materialize.rs` | — | 931 | 272 (PR-3) +143 (PR-4) +183 (PR-5) +364 (PR-6) +157 (PR-7) | split from PR-3 onward: core / `exl3_ngram_sidecar.rs` / `exl3_native_policy.rs` / tests |
| `spark-runtime/src/weights/exl3.rs` | — | 915 | 736 (PR-2) +16 +100 (PR-4) +5 +79 (PR-6) | PR-2: `exl3.rs` + `exl3/cpu_ref.rs` + `exl3/store_tests.rs`; PR-4: `exl3/ngram_ref.rs` |
| `spark-model/src/weight_map/nvfp4_detect.rs` | 466 | 601 | +15 (PR-5), ≈+120 net (PR-6: `resolve_key_variant` + tests) | PR-6: `nvfp4_detect/exl3_keys.rs` (+tests) |
| `spark-runtime/src/cuda_backend/gpu_impl.rs` | 489 | 590 | +28 (#821 resolution) +72 (PR-1) +11 (PR-1 cache) | PR-1: `cuda_backend/kernel_lookup.rs`; #821 must also get under 500 on its own (489+28) |
| `spark-model/src/layers/ple/layer.rs` | 498 | 573 | +21 (#821) +55 (PR-4) | PR-4: `ple/layer_exl3.rs`; #821 alone is 519 — #821 needs its own split |
| `spark-runtime/src/gpu.rs` | 490 | 558 | +68 (PR-1) | PR-1: `gpu/launch.rs` (`KernelLaunch`) |
| `spark-runtime/src/gpu/mock.rs` | 463 | 544 | +81 (PR-1) | PR-1: `gpu/mock/recording.rs` |
| `spark-model/src/layers/ops/exl3_dense.rs` | — | 514 | PR-7 | PR-7: `ops/exl3_dense/weight.rs` |

Allow-listed files that grow on the branch (legal, but reviewers will see the warning):
`factory/build.rs` 900 -> 1002 (PR-3 +15, PR-5 +109 — extract `factory/exl3.rs`),
`serve_load.rs` 1273 -> 1289, `linear_attn_arms.rs` 548 -> 604, `transformer_layer.rs` 646 -> 680
(+24 is #821), `attention_forward.rs` 877 -> 912, `trait_decode_batched.rs` 1281 -> 1296,
`moe/forward.rs` 755 -> 772, `verify_e.rs` 865 -> 869, `types.rs` 644 -> 652, `decode_a2.rs`
632 -> 641, `quantized.rs` 680 -> 689, `impl_a1.rs` 867 -> 870, `init.rs` 675 -> 676,
`multi_seq/qkv.rs` 1110 -> 1115, `load_layers.rs` 1025 -> 1028.

---

## 3. `.research/` artifacts — keep in-tree vs docs PR vs drop

Nothing under `.research/` exists on main. Code references to it: `examples/exl3_native_parity/
legs.rs:160` (default `EXL3_REAL_DIR = .research/real_tensor`, skips if absent) and a doc comment
in `weights/exl3.rs:736`. The vendored `exl3_vendor/*.cuh` headers cite
`Snapshot original: .research/exllamav3_ref/<file>`.

| Artifact | Size | Disposition |
|----------|-----:|-------------|
| `fetch_exl3_tensor.py` | 2 KB | **Keep, relocate** to `scripts/dev/exl3_fetch_tensor.py` in PR-2 — it produces the `EXL3_REAL_DIR` fixture the parity gates use. |
| `check_real_ngram.py`, `exllamav3_ref/ngram_codec.py` | 6 KB | **Keep, relocate** to `scripts/dev/` in PR-4 — the "independent third implementation" the ngram real-row validation rests on. |
| `real_tensor/{trellis.bin 819 KB, suh.bin, svh.bin, mul1.bin, meta.txt}` | 830 KB | **Do not commit.** Regenerable from HF via the fetch script; both examples skip gracefully (F10). Change the `legs.rs` default to env-only with the same skip message `exl3_reconstruct_parity` prints. (The repo does carry pinned GDN binaries, so a committed fixture is not unprecedented — but 820 KB of a third-party model's weights in-tree is a licensing question the reviewer should not have to answer.) |
| `ngram_rows.bin`, `ngram_head_bias.bin` | 5.6 KB | Consumed only by `check_real_ngram.py`; move next to it or drop (regenerable). |
| `EXL3_DECODE_FINDINGS.md` | 12 KB | **Docs PR** -> `docs/porting/exl3.md`: format facts (MSB-first-per-u32 tile bitstream vs LSB-first-per-u16 ngram ring; procedural codebook mul1/mcg/3inst; `.mul1` holds the multiplier constant; data offset 5400 in the ngram file), the two layouts, the flag ladder, the measured numbers, known limitations. |
| `vision_exl3_map.md` | 34 KB | **Docs PR**, trimmed: the per-bpw vision tensor map, the fused BF16 `attn.qkv` observation, the `vision_k6.safetensors` sidecar gap, the 36/52/68/84/100 GB packed-fit estimates and "native gate must widen to K∈{3,5,6}; K=7 has no kernels". |
| `exllamav3_ref/` (40 files: .cu/.cuh/.cpp/.h snapshots + `py_*.py` torch-side sources) | ~3.4 MB | **Drop from upstream.** The device code is already vendored verbatim under `kernels/gb10/common/exl3_vendor/` with MIT headers; the Python/C++ host side is not used. Replace the header citations with the upstream git commit SHA + path and add `exl3_vendor/README.md` (PR-5). Keep the snapshot on the research branch for diffing. |
| `native_port_map{.json,_raw.txt}`, `moe_map_raw.txt`, `moe_native_map.json` | 574 KB | **Drop** (port bookkeeping). One paragraph of the docs PR can say what was and was not instantiated (203 entries, cb0 dropped, K∈{2,3,4} fused envelope). |
| `ckpt_meta/{shard1_header.json 4.1 MB, config.json, ngram_header.json, hdr8/ng8.bin}` | 4.1 MB | **Drop** — the 4 MB file is a safetensors header dump, regenerable; `config.json` is the HF config. |
| `boot/{boot_native_moe,boot_native_dense,smoke_native_moe,build_boot,memwatch}.sh`, `decode_arm_build.sh`, `prefill_arm_build.sh`, `pf_debug_run.sh`, `boot/*_dbg.sh`, `boot/*_san.sh` | 8 KB | **One canonical serve recipe** into the docs PR (or `scripts/serve-exl3-native.sh` beside the existing `scripts/start-*.sh`), carrying the QSA cap = CTX and PLE cap = prefill-chunk rule from da09c7bcf and the `--ssm-cache-slots 32–64` note. Drop the dbg/san/memwatch variants (box-specific paths). |

Net: three small scripts travel with PR-2/PR-4; two markdown files + one recipe become a docs PR
that can land any time after PR-4 (it describes the boot) — or be split so the format section lands
with PR-2. Everything else stays on `wip/exl3-research`.

---

## 4. The two merge commits

* **`0c34c15ec` Merge avarok/main** — clean auto-merge (F6), zero manual content, brought in #827,
  #829, #830 (blog/site) and #831 (dflash default). **Rebase away.** Every carved PR is created
  from `avarok/main` with `git cherry-pick`/`git rebase --onto`; the merge has nothing to carry.
  Note the carve-up must be done against a main that has NOT moved under the branch: the base is
  `ef8ca11d9`, which is also the current `avarok/main` (verified with `git ls-remote`), so today
  there is nothing to re-resolve.
* **`e0d0efbcd` Merge #821 (fix/qsa-ple-seq-state-leak)** — NOT clean (F7). Do not carry #821
  inside any EXL3 PR:
  1. `git diff 0c34c15ec e0d0efbcd` is precisely "#821 on top of current main" (9 files,
     +195/−9). Apply it to a fresh branch off main (or rebase `fix/qsa-ple-seq-state-leak` and
     resolve with that diff) and force-push #821's head so it turns green. The resolution keeps
     main's `AllocRecord` ledger and grafts `forget_alloc -> Option<usize>` +
     `live_count_bytes()` onto `alloc_ledger.rs`; the ≥32 MB alloc/free trail now prints a running
     live total. #821 additionally needs its own 500-LoC fixes: `ple/layer.rs` 519 and
     `gpu_impl.rs` 517 after the resolution (`transformer_layer.rs` is allow-listed).
  2. Land #821 first. The EXL3 PRs then rebase over it trivially — no EXL3 file overlaps with
     #821 except `ple/layer.rs` (both add to different regions) and `gpu_impl.rs`
     (PR-1 adds trait fns, #821 edits `alloc`/`free`).
  3. If #821 stalls, the EXL3 series still builds and passes its gates; only the long single-node
     session leak (~739 MB/request) remains, and that is #821's problem statement, not EXL3's.
* Rebasing rather than keeping is also what keeps each PR's `git log` to its own commits, which is
  what the "small, gate-able" ask means in practice. The PR #834 branch itself should stay as the
  integration/research branch (it is the only place the two merges and `.research/` live).

---

## 5. Reviewer-facing summaries (paste into each PR body)

**PR-0 — fix(vision): ViT MLP width from the fc1 tensor.** EXL3 exports pad the ViT
`linear_fc1/fc2` from the config's `intermediate_size` 4304 to 4352 (128-wide trellis tile); the
padded fc1 rows are zero-scale/zero-bias so the math is unchanged, but the fc2 GEMM indexed its
`[hidden, inter]` weight at row stride 4304 against a 4352-wide tensor — in bounds, no fault,
silently garbage image embeddings in all 27 blocks. The loader now takes the width from
`blocks.0.mlp.linear_fc1.weight` when it disagrees with the config (warn-logged). Measured on the
2.05bpw EXL3 serve: red circle "A black triangle on a white background." -> "I see a red circle.";
blue square "A yellow circle." -> "A blue square." Non-EXL3 checkpoints (tensor == config) are
unchanged.

**PR-1 — spark-runtime: cooperative launch + dynamic-smem opt-in + kernel-handle cache.** Adds
`GpuBackend::launch_cooperative{,_typed}` (`cuLaunchCooperativeKernel`; default: refuse, since a
kernel that `grid.sync()`s would deadlock under a plain launch), `set_kernel_max_dynamic_smem`
(`cuFuncSetAttribute`, sticky per `CUfunction`, called once at handle resolution — the EXL3 GEMM
needs 90 KB), `KernelLaunch::cooperative()`, and mock recording with three unit tests. Also a
per-backend `module::func -> handle` cache in `GpuBackend::kernel`: by-name lookups per launch were
pushing ~1,000 `kernel_audit` rows/token (~1.3 MB/s unbounded host growth) plus a
`cuModuleGetFunction`; hits now record nothing, misses behave exactly as before. No caller of the
cooperative path lands here; the cache affects every model and is gated by a production boot.

**PR-2 — EXL3 trellis reconstruct kernel + independent CPU reference.** Self-contained port of
ExLlamaV3's fused reconstruct + both-side Hadamard (MIT, (c) 2025 turboderp, attribution in-file):
24 (K=1..8 × 3 codebooks) instances plus an f16 -> `[out,in]` BF16 layout converter, and a CPU
reference written from the format spec rather than transcribed from the kernel. GPU vs CPU is
BYTE-IDENTICAL on 3 shapes × 8 legs × two stages, on 1-bit negative controls, and on a real tensor
from `turboderp/Qwen3.8-Flash-Next-exl3`. Format facts the tests pin: the tile bitstream is
MSB-first within each u32 (an LSB-first model mismatches ~100%); the codebook is a 2–3 instruction
procedural generator (mul1/mcg/3inst) with no stored table; the `.mul1` scalar stores the
codebook's own multiplier constant. Store plumbing: `I16`->UInt16 and `I32`->Int32 raw containers,
and the F16->BF16-at-load conversion exempts `.suh/.svh` (exact f16 bits are decode inputs). Nothing
dispatches the kernel yet.

**PR-3 — EXL3 loader: load-time materialization (compatibility path).** A one-shot in-place store
rewrite hooked at `serve_load` (before preflight + quant detection) and idempotently at
`factory::build_model`: routed/shared experts -> reconstruct -> BF16 (one-tensor transient) ->
runtime NVFP4 -> ModelOpt-style triplet (quantizing inside the pass is load-bearing: experts are
~90% of parameters and cannot all sit as BF16); everything else -> BF16 `.weight`. `quant_method:
"exl3"` maps to `ModeloptFormat`; every per-model loader consumes the result verbatim with zero
per-arm changes. 37,341 linears materialize in ~30 s/rank. A store with no trellis tensors is
untouched (unit-tested on the mock GPU). This is the double-quantization path (EXL3 K bpw -> BF16
-> NVFP4); the native path follows in PR-5..7. The qwen4_exp checkpoint does not boot until PR-4
decodes its PLE table.

**PR-4 — EXL3 PLE n-gram row decoder (decode-on-gather): first full boot.** The 320M-row PLE
n-gram table ships as a standalone `exl3_ngram_trellis` (26.2 GB packed at K=4 vs 102 GB BF16);
each row is one fp16 scale word + a 160·K-bit tail-biting mul1 ring, LSB-first per u16 — a
different bit order from the tile format, pinned by tests. The `NgramRowCache` faults RAW rows into
the existing pinned arena unchanged; `batched_embed_exl3` does ring-state extraction + mul1 decode +
row scale + per-head bias on gather. GPU vs CPU bit-identical at K=4/6; real rows validated through
an independent third implementation. Both published layouts (monolithic `[320M,61]` and 128 ×
`shard_{i}.trellis`) walk into the same segmented row cache. Booted `Qwen3.8-Flash-Next-exl3
2.05bpw` at EP=2 (gx10 + dgx-00), CTX 8192, util 0.6 honored at 72.9/73.0 GB: cold gather 256
misses = 14 ms, warm 512/512 hits = 96 µs, TTFT 718 ms warm, ~20 tok/s, temp=0 coherent and
correct. BF16-PLE models take the `Bf16` format path exactly as before.

**PR-5 — EXL3 native trellis matmul: vendored fused GEMM/GEMV/mgemm, lm_head served packed.**
ExLlamaV3's fused kernels vendored verbatim (MIT, (c) 2025 turboderp; 211 Blackwell-reachable
extern-C entries: 80 gemm, 72 mgemm pointer-table MoE form, 48 gemv, 3 converters) fuse the entire
pipeline — diag(suh)+H128 input rotation, in-register trellis decode, m16n8k16 MMA, H128+diag(svh)
epilogue — so callers pass raw activations and receive un-rotated outputs. Launch wrappers use
upstream's occupancy-capped grid math (verified deadlock-free live, including the clean
`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` refusal at occ=3). Parity: 42 GPU legs PASS — rotation
BIT-EXACT vs a CPU FWHT (`RS = f32::from_bits(0x3db504f3)`; the decimal literal is 1 ULP high),
gemv/gemm rel-RMS ~3e-4 across every K/cb/shape, real checkpoint tensor, mgemm 4-expert smoke;
native weight fidelity 2.9e-4 vs 1.9e-3 for the materialized-NVFP4 path (~6.5×). Under
`ATLAS_EXL3_NATIVE=1` (default OFF) the lm_head serves from its packed trellis (13.1 MB vs 14.7 MB
NVFP4 vs 52.4 MB BF16 for that tensor; 36.4 µs/launch at [2560->10240] K=4); live boot coherent
and byte-identical across runs. Cooperative launches are never graph-captured: a narrow
`exl3_graph_veto` is ORed into the capturing verify sites, kept separate from
`decode_graph_unsupported`. Native K is gated to the GEMV-servable {2,4} set so the split-K locks
GEMM stays off co-dispatched streams. Gate-OFF path byte-identical.

**PR-6 — EXL3 native routed MoE experts: the 63 GB checkpoint fits one GB10.**
`ATLAS_EXL3_NATIVE_MOE=1` (layered on `_NATIVE=1`, default OFF, combination validated at load)
keeps all 512 routed experts packed and serves them through upstream's own tiers: decode (T≤8) =
3× cooperative `exl3_mgemm` with routing weights folded into the fp32 down reduction; prefill (T>8)
= sort-by-expert + ONE fused persistent `exl3_moe` launch (16 instances, separate module) for
experts with ≤128 rows, plus a chunked `exl3_gemm` overflow tier on the same trellis at the same
fp16 precision (a 4096-token batch averages 80 rows/expert, so overflow is routine). Result: 73,728
projections packed = 30.67 GB resident vs 67.95 GB as NVFP4 triplets (37.3 GB saved); 43.7 GB
pre-KV; 1.12M KV tokens at util 0.6; 48/48 layers + lm_head native; EP=2 no longer required.
Single node C=1: temp=0 coherent, 11.0 tok/s decode, 3.3K-token TTFT 6.1 s cold / 214 ms on a
prefix-cache hit. Contracts enforced at runtime: dense EP-local pointer tables with −1 remote
indices; per-layer K/cb uniformity with atomic keep-or-materialize (K∈{2,3,4}); ONE dispatcher over
the shared slabs/locks (host in-flight claim + device event fence across the prefill/decode
streams); graph veto in every capturing verify site; load-time probes for both kernel modules. Two
load-path bugs the real boot found are fixed (namespace probe refusing packed experts; a Bf16Raw
misdeclaration reading the shared-expert NVFP4 triplet as BF16 `[n,k]`, 4× past the buffer — pinned
by compute-sanitizer, now resolved per key and unit-tested). 45 parity legs PASS; the overflow-skew
leg gates at rel_rms 1.9e-3 vs 5.0e-3 on the BF16 path it replaced. Gate-OFF experts materialize
exactly as before (unit-tested).

**PR-7 — EXL3 native GDN + attention dense linears, one shared launch state.**
`ATLAS_EXL3_NATIVE_DENSE=1` (default OFF; `_GDN=0`/`_ATTN=0` opt a family back out) keeps the 36
GDN families (in_proj_qkv [2560->10240], in_proj_z [2560->6144], out_proj [6144->2560]) and 12
attention families (q [2560->12288], k/v [2560->512], o [6144->2560]) packed and serves them
through one reusable native dense linear: bf16->f16 ingress into a model-shared staging slab,
`exl3_gemv` (m≤8, f32 C) or `exl3_gemm` (m>8, row-batched), egress contiguous or STRIDED — so the
GDN pair writes the existing fused `[Q|K|V|Z]` row (ld 16384, Z at column 10240) with zero consumer
changes; attention drops the runtime BF16->NVFP4 requant and the transposed twins. Every
cooperative launch in the model — MoE, dense, and the LM head — now runs under ONE
`Exl3LaunchState` (locks + CUDA event fence + host section mutex; a second thread blocks instead of
being refused; a stream change waits on the previous section's fence). Result: BF16 linears 332 ->
176, 1.34 GB resident vs 5.35 GB, pre-KV 43.7 -> 39.0 GB (+21% KV tokens); answers token-identical
to the gate-OFF binary; C=4 co-dispatched 3.6K prompts clean; decode 12.6–13.0 tok/s (+15–18%);
cold TTFT @3.6K +10% (two cooperative GEMMs replace one cuBLASLt GEMM for the in_proj pair — a
large-M reconstruct-to-BF16 prefill tier is the known lever). 139 parity legs PASS. Gate-OFF boot
identical to the previous reference (332 linears, 43.7 GB, same answers).

**Docs PR — docs/porting/exl3.md.** Format notes, the flag ladder
(`ATLAS_EXL3_NATIVE` -> `_MOE` -> `_DENSE`, sub-gates, `ATLAS_EXL3_GEMV_OCC`,
`ATLAS_EXL3_DENSE_STAGE_ROWS`, `ATLAS_EXL3_MOE_PREFILL_BATCH_TOKENS`), the measured lever ladder
(20 -> 11.0 -> 12.6–13.0 tok/s across compat/native-MoE/native-dense on one GB10; 63 GB checkpoint
at 30.67 GB experts), the serve recipe (QSA cap = CTX, PLE cap = prefill chunk,
`--ssm-cache-slots 32–64`, `ATLAS_PLE_CACHE_SLOTS ≥ tokens×16`), known limitations (vision
`vision_k6.safetensors` sidecar on 4.05bpw; K∈{3,5,6} gate widening for higher bpw; K=7 has no
kernels; no quality A/B vs the NVFP4 checkpoint yet), and the vendored-code provenance table.

---

## 6. Gate results on the tip (this task; CPU only, scratch `CARGO_TARGET_DIR`, exact env)

Env: `PATH=/usr/local/cuda/bin:$PATH CUTLASS_HOME=/home/ms/cutlass FLASHINFER_HOME=/home/ms/flashinfer
RUSTFLAGS="-L/home/ms/nccl/build/lib -L/usr/local/cuda/lib64" ATLAS_TARGET_HW=gb10
ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4`, `--locked`, nvcc 13.0.

| Gate | Result | Log |
|------|--------|-----|
| `ATLAS_TARGET_MODEL='*' cargo check -p atlas-kernels -j 20` ("build all targets") | **PASS**, 74 s; 507/4617 unique nvcc invocations, 9.1× dedup; EXL3 PTX emitted for every target | `gate0_kernels_star.log` |
| `cargo build --release -p spark-server --bin spark -j 8` | **PASS**, 161 s | `gate1_build.log` |
| `cargo test --release -p spark-model --lib -j 8` | **PASS**, 124 s; 683 passed, 0 failed, 11 ignored (matches the e61387092 message) | `gate2_model_tests.log` |
| `cargo test --release -p spark-runtime --lib -j 8` | **PASS**, 112 s; 291 passed, 0 failed, 12 ignored (matches the f1aa57395 message) | `gate2b_runtime_tests.log` |
| `cargo clippy --release -p spark-runtime -p spark-model -p spark-server --all-targets -j 8` | **PASS**, 153 s; zero non-kernel warnings (only the pre-existing `atlas-kernels` closure/include notes) | `gate3_clippy.log` |
| Direct `nvcc --ptx` timing of the 4 new/changed common kernels (strict flags) | 40.4 s / 18.6 s / 2.0 s / 0.3 s, all clean under `--Werror all-warnings` | `nvcc_time.out` |
| file-size-cap emulation over `crates/` | **8 violations** (section 2.9); main is clean | `loc_gate_full.sh` |

Not run (GPU or reviewer-side): the 5 GPU gates, the parity examples, any boot, the SPDX
license-eye container check (docker), the Windows CUDA release leg.

Logs live in `/home/ms/.claude/jobs/5a7bd33d/tmp/upstream/`.

---

## 7. Things to decide before carving

1. **PR-1 standalone or folded into PR-5** (section 2, PR-1 "Alternative"). Recommendation:
   standalone — it is the only change with a blast radius over every model.
2. **Introduce `Exl3LaunchState` in PR-5 instead of PR-7** to avoid PR-6's `Exl3MoeState` locks
   being rewritten one PR later (`tables.rs`/`ptr_table_build.rs` churn). This is a code
   restructuring of the branch, not a cherry-pick; worth it if the reviewers will read PR-6 and
   PR-7 back to back.
3. **Split PR-7 into GDN and attention halves** (9 PRs) if the reviewers want each family's
   loader-arm changes isolated.
4. **Real-tensor fixture**: fetch-script + env (recommended) vs committing 820 KB of model weights.
5. **#821 first**: someone must push the `git diff 0c34c15ec e0d0efbcd` resolution to
   `fix/qsa-ple-seq-state-leak` and split `ple/layer.rs` / `gpu_impl.rs` there.
