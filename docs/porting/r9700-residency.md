# R9700 weight residency: where the 32 GB goes

**Board:** AMD Radeon AI PRO R9700, gfx1201, 32624 MiB VRAM (31.9 GB), SCALE
1.7.1.
**Branch:** `amd/r9700-target` at `34176fc7f`.
**Checkpoint:** `unsloth/Qwen3.8-27B-NVFP4`, a compressed-tensors
`format = mixed-precision` release: FP8 E4M3 with a per-CHANNEL `[N,1]` scale
for `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`,
`lm_head` and the MLPs of layers 56..63; packed NVFP4 for every other MLP; BF16
for embeddings, norms and the vision tower. 1968 tensors, 21.81 GiB on disk.
**Serve:** `ATLAS_TARGET_HW=r9700 ATLAS_TARGET_MODEL=qwen3.8-27b
ATLAS_TARGET_QUANT=nvfp4`, `ATLAS_W4A16_VARIANT=v1
ATLAS_NO_GDN_FP8_PREFILL=1`, `--oom-guard-mb 1024 --gpu-memory-utilization 0.80
--max-seq-len 4096 --max-batch-size 4`.

## The failure this document explains

Every weight loads. The post-load audit reports `Weights: 21.81 GB, estimated
free 9.3 GB, actual free 9.2 GB`. The serve then dies inside
`ModelWeightLoader::load_layers` at layer 28 of 64 with

```
cuMemAlloc_v2 failed: status 2, requested 167772160 bytes
```

167,772,160 bytes is `ssm_qkvz_size() * hidden_size * 2` = `16384 * 5120 * 2`,
the BF16 `[Q|K|V|Z]` concatenation `gpu_concat_rows` builds at
`weight_loader/qwen35_dense.rs:1476`. Layer 28 is a linear-attention layer and
that is its first large allocation.

The allocation ledger at that moment: **2631 allocations, 33.73 GB live**, no
owner. sysfs VRAM peak 34.1 GB, i.e. the board was full.

## The ledger reproduces exactly from the code

This is the part worth trusting the rest of the document on. Every site in the
sweep is predicted to the tenth of a MiB by shape arithmetic over
`kernels/r9700/qwen3.8-27b/MODEL.toml` plus the GDN head geometry
(`linear_num_key_heads = 16`, `linear_num_value_heads = 48`, both head dims
128, from the checkpoint's own `config.json`).

Layers 0..27 have completed; layer 28 is a linear-attention layer whose dense
FFN has been built and whose three SSM projections have been dequantised.
Layer types alternate on a period of 4, so indices 3, 7, 11, 15, 19, 23, 27 are
the seven full-attention layers already built and 21 are SSM.

| ledger site | measured | predicted | what it is |
|---|---|---|---|
| `fast_weights/mod.rs:434` | 22,332.5 MiB x1968 | 21.81 GiB | the store, one `gpu.alloc(meta.len)` per checkpoint tensor |
| `weight_map/quantized.rs:261` | 5,202.5 MiB x157 | 84 FFN + 3 (layer 28) x 42.5 + 28 attn x (30/2.5/2.5/15) + 42 SSM x (40/15) = **5,202.5** | transposed packed NVFP4 |
| `weight_map/quant_helpers.rs:98` | 1,620.0 MiB x31 | 28 attention dequants x (120/10/10/60) = 1,400, plus layer 28's three live SSM dequants (100 + 60 + 60) = **1,620.0** | FP8 to BF16 dequant output |
| `weight_map/loaders_fp8.rs:229` | 1,505.0 MiB x70 | 28 attn x (30/2.5/2.5/15) + 42 SSM x (40/15) = **1,505.0** | `quantize_to_nvfp4` packed `[N,K/2]` |
| `weight_map/quantized.rs:262` | 650.3 MiB x157 | exactly 1/8 of `:261` (`K/16` vs `K/2`) = **650.3** | transposed NVFP4 group scales |

The x31 count at `quant_helpers.rs:98` is the important one. Twenty-eight of
those thirty-one dequant buffers belong to attention layers that finished
building seven, eleven and up to twenty-four layers ago. They are a **leak**,
not a transient: see "The attention BF16 dequant is never freed" below.

## Steady-state resident set

Extrapolated to all 64 layers from the same arithmetic. "Store" is
`WeightStore`; "layer-owned" is everything a loader allocated on top of it and
handed to a layer struct.

### What each family does

| family | on disk | what the loader materialises | store after build | store MiB | layer MiB |
|---|---|---|---|---|---|
| `embed_tokens` | BF16 `[248320,5120]` | nothing; `dense()` hands the store pointer through | **ALIVE** (zero-copy) | 2,425.0 | 0 |
| `lm_head` | FP8 + `[N,1]` BF16 scale | dequant to BF16 (`load_lm_head`), then a runtime NVFP4 copy (`lm_head_setup.rs`) | **DEAD**, and RELEASED since 2026-09-17 unless `--lm-head-dtype fp8` or `--dflash` binds it zero-copy | 1,213.0 | 3,107.0 |
| attn q/k/v/o, x16 | FP8 + `[N,1]` BF16 scale | dequant to BF16 (kept, see below), requant to NVFP4, transposed twin of each | **DEAD** | 1,600.6 | 5,000.0 |
| attn q/k norms, kv scales, x16 | BF16 | nothing | ALIVE | ~0.1 | 0 |
| SSM `in_proj_qkv`/`in_proj_z`/`out_proj`, x48 | FP8 + `[N,1]` BF16 scale | dequant to BF16 (freed), row-concat to `[Q\|K\|V\|Z]` (freed), requant to NVFP4, transposed twin of each, plus a `predequant_nvfp4_to_fp8` out_proj copy | **DEAD** | 5,282.0 | 7,380.0 |
| SSM `in_proj_a`/`in_proj_b`, x48 | BF16 `[48,5120]` | interleaved into a new `in_proj_ba` | **DEAD** (45 MiB total) | 45.0 | 45.0 |
| SSM `conv1d`/`A_log`/`dt_bias`/`norm`, x48 | BF16 / F32 | `conv1d` zero-copy; `A_log`/`dt_bias`/`norm` promoted to F32 when the checkpoint ships BF16 | ALIVE (conv1d) | 4.3 | 1.1 |
| dense FFN gate/up/down, layers 0..55 | packed NVFP4 (`weight_packed` + `[N,K/16]` FP8 scale) | `quantized_v2` binds the packed bytes ZERO-COPY, then `transpose_for_gemm` builds a second, transposed copy | **ALIVE** (decode reads the store's packed bytes) | 8,032.5 | 8,032.5 |
| dense FFN gate/up/down, layers 56..63 | FP8 + `[N,1]` BF16 scale | dequant to BF16 (freed by `quantized_from_fp8`), requant to NVFP4, transposed twin | **DEAD** | 2,040.8 | 2,295.0 |
| per-layer norms, x64, and `model.norm` | BF16 | nothing | ALIVE | 1.2 | 0 |
| `model.visual.*` | BF16 | bound by `Qwen35WeightLoader::load_vision_encoder` | ALIVE, not itemised | ~1,690 | not itemised |

### Totals

| | GiB | GB |
|---|---|---|
| store, ALIVE after build | 10.22 (+ ~1.65 visual) | 10.97 (+ ~1.77) |
| store, DEAD after build | 9.94 | 10.68 |
| store total (matches the measured 21.81 GiB) | 21.81 | 23.42 |
| layer-owned | 25.25 | 27.12 |
| **grand total** | **47.07** | **50.54** |

Of the layer-owned 25.25 GiB:

* **12.74 GiB is the second layout** (transposed NVFP4 twins): 8.96 GiB dense
  FFN, 2.90 GiB SSM, 0.88 GiB attention;
* **3.12 GiB is the attention BF16 dequant leak**;
* 1.41 GiB is the SSM `out_proj` FP8 predequant (30 MiB x48);
* 2.37 GiB is the BF16 `lm_head` and 0.67 GiB its NVFP4 copy;
* the rest is the NVFP4 base layouts that decode reads.

### Does the two-layout design alone exceed 32 GB minus a 4 GB floor?

Yes, and not marginally. Budget: 31.9 GB physical, 4 GB reserved for KV cache
plus the buffer arena plus the vision encoder's working set leaves **27.9 GB**
(25.99 GiB) for weights. Under `--gpu-memory-utilization 0.80` the pledge is
25.5 GB, so the real target is nearer 21 GB.

| configuration | GiB | GB | fits 27.9 GB? |
|---|---|---|---|
| today | 47.07 | 50.54 | no |
| release every dead store tensor except `lm_head` | 38.31 | 41.13 | no |
| ... and free the attention BF16 dequant | 35.18 | 37.78 | no |
| ... and release the FP8 `lm_head` too | 34.00 | 36.50 | no |
| ... and drop every transposed twin (`ATLAS_LOAD_TRANSPOSED_TWINS=0`) | 21.25 | 22.82 | **yes** |

**Release-on-consume is necessary and not sufficient.** It buys 8.76 GiB
(9.40 GB). The attention leak buys another 3.12 GiB. Together they take the
peak from 50.5 GB to 37.8 GB, which still will not load on this board. The
remaining 12.74 GiB is the second layout, and nothing short of dropping it (or
a single-layout prefill kernel) closes the gap.

### The lever that closes it, and what it costs

`ATLAS_LOAD_TRANSPOSED_TWINS` (`weight_loader/qwen35_dense/transposed_twins.rs`)
is `1` (build every twin; the pre-lever behaviour byte for byte, and the default
on every non-SCALE target), `0` (build none) or `auto` (build them only if
`gpu.free_memory()` after the checkpoint is resident exceeds their projected
bytes plus a 4 GiB reserve for the KV cache, the buffer arena and the vision
encoder's working set). Unset takes `cfg!(atlas_scale)`, which since 2026-09-17
is **`0`** on SCALE rather than `auto`. See "The cost" below. `serve-amd.sh`
exports `0` for r9700.

The projection is the SAME arithmetic that reproduces the measured ledger above,
and `transposed_twins_tests.rs` pins it to the whole MiB against these numbers,
so a drift between the projection and the ledger is a test failure rather than a
probe deciding about a model that is not the one being loaded:

| model | dense FFN | SSM | attention | total |
|---|---|---|---|---|
| `qwen3.8-27b` (64 layers, hidden 5120, inter 17408) | 9,180 MiB (8.96 GiB) | 2,970 MiB (2.90 GiB) | 900 MiB (0.88 GiB) | **13,050 MiB (12.74 GiB)** |
| `ornith-1.0-9b` (32 layers, hidden 4096, inter 12288) | 2,592 MiB (2.53 GiB) | 864 MiB (0.84 GiB) | 252 MiB (0.25 GiB) | **3,708 MiB (3.62 GiB)** |

The fused `[q|k|v]` attention twin is deliberately NOT priced: it is built only
when q/k/v share one `weight_scale_2`, which this checkpoint's per-projection
absmax makes false (see item 4 of the ranked list), and pricing a copy that is
usually absent would make `auto` refuse room it does not need.

**The cost, and the measurement that reversed its sign on this board.** With the
twins absent every fast arm of `w4_gemm!` is skipped and FFN prefill lands on
the plain `w4a16_gemm`. That was expected to be expensive: on **GB10**,
`w4a16_gemm` measured ~7.0 TFLOP/s against ~51 for `w4a16_gemm_t_m128` on
Gemma-4-31B, a 7x slower FFN prefill, and that is still why every non-SCALE
target builds the twins.

⚠️ **On gfx1201 it is the opposite.** The R9700 prefill measurement of
2026-09-17 (Ornith-1.0-9B with the twins against Qwen3.8-27B without) puts
`w4a16_gemm_t_m128` at **~1 TFLOP/s and the plain `w4a16_gemm` at ~4 TFLOP/s**.
The twin arm is the slower one here, so the 12.74 GiB buys nothing back and the
SCALE default is now `0`. The 7-vs-51 figures were never measured on SCALE and
this document should not have carried them forward as if they were. The SSM and
attention sides remain unmeasured on both targets.
Decode is untouched: it reads the packed original either way. Two further
consequences are deliberate: dropping the SSM twin also drops the 1.41 GiB
`out_proj` FP8 predequant (`qwen3_ssm/init_fp8.rs:110` keys off
`out_proj_nvfp4_t.is_some()`), and `finalize_nvfp4_mmq_load` is skipped with
them, because it is residency-neutral only while there are `_t` copies for it to
free.

**This is not the fix.** The fix is item 5's sibling: a prefill GEMM that reads
the packed `[N, K/2]` layout directly with a transposed tile walk, with the
128x128 `cp.async` tiling the `_t` kernels have. Then there is no second layout
to build or skip, on any target. On GB10 that removes the 7x; on gfx1201, where
the `_t` arm is the slower of the two, it removes a copy that was buying nothing
back. This lever buys a serve.

## DEAD versus ALIVE: the per-site argument

A store tensor is DEAD when its only consumer copied or re-encoded it into a
layer-owned allocation. It is ALIVE when a layer struct holds its device
pointer.

**DEAD, and provably so, because `dense_auto`'s FP8 arm always allocates.**
`weight_map/quant_helpers.rs:284-296` returns `w.ptr` unchanged for a BF16
tensor and routes FP8 E4M3 to `dequant_fp8_blockscaled_to_bf16`, which allocates
`total * 2` at `:98`. Every FP8 projection in this checkpoint therefore reaches
its layer through at least one fresh allocation, and the store's E4M3 bytes are
read exactly once, by the dequant kernel.

* attention q/k/v/o: `qwen35_dense.rs:713-722`, the `CompressedTensors` arm.
  `store.contains("{prefix}.weight_packed")` is false for these keys, so the
  loader dequants and calls `quantize_to_nvfp4`. `AttentionWeights` keeps only
  the NVFP4 result and the two `q_norm`/`k_norm` pointers.
* SSM `in_proj_qkv`/`in_proj_z`/`out_proj`: `qwen35_dense.rs:1470-1476` via
  `load_ssm_proj` to `dense_auto`. The concat copies, the requant copies, and
  the loader already frees both BF16 intermediates at `:1479-1480` and
  `:1692-1693`.
* dense FFN layers 56..63: `quantized_any`'s `has_fp8_dense` detection
  (`nvfp4_detect.rs:266-274`) routes them to `quantized_from_fp8`, which frees
  its own BF16 intermediate and returns NVFP4.
* `lm_head`: `loaders_b.rs:52-64` dequants FP8 to BF16.
* SSM `in_proj_a`/`in_proj_b`: BF16, so `dense_auto` returns the STORE pointer,
  and `interleave_ba` copies out of it. Dead, but aliased, and only 45 MiB.

**ALIVE.**

* `embed_tokens`, every norm, `conv1d`, the kv scales: bound zero-copy.
* **the packed NVFP4 dense FFN of layers 0..55.** `quantized_v2`
  (`quant_helpers.rs:355-383`) is pure pointer plumbing: `weight_packed` and
  `weight_scale` are the store's own allocations. The layer keeps that packed
  original for decode AND a transposed copy for prefill, so the store tensor is
  live for the life of the model. 8,032.5 MiB of the store is this, and it must
  be counted but must not be freed.
* the vision tower, because this loader binds a vision encoder, so
  `factory/build.rs:346`'s reclaim does not fire.

## Two findings that fall out of the accounting

### The attention BF16 dequant is never freed

`weight_loader/qwen35_dense.rs:713-722`:

```rust
let src = if store.contains(&format!("{prefix}.weight_packed")) {
    quantized_auto(store, &prefix, gpu, variant)?
} else {
    let dense_bf16 = dense_auto(store, &format!("{prefix}.weight"), gpu)?;
    quantize_to_nvfp4(&dense_bf16, full_n, full_k, gpu, absmax_k, quantize_k, stream)?
};
```

`dense_bf16` is dropped without `gpu.free`. Every sibling site frees its
equivalent: the `Standard | Fp8Dequanted` attention arm at `:870-873`, the SSM
path at `:1479` and `:1693`, `quantized_from_fp8` at `nvfp4_detect.rs:369`, and
the `Bf16Raw` arm of `quantized_any` at `:325-332`. This one does not, and it
is exactly the 28 stale allocations the ledger shows.

Cost on this checkpoint: 200 MiB per full-attention layer, **3.12 GiB (3.36 GB)
across the 16**. It fires on any CompressedTensors checkpoint whose attention
projections are not NVFP4-packed, which is the whole unsloth mixed-precision
family, on every target including NVIDIA.

**FIXED UNCONDITIONALLY**, one commit after it was first written down. It rode
`ATLAS_LOAD_RELEASE_SOURCES` only because that change was not allowed to move
NVIDIA behaviour, and on re-reading that constraint does not cover this: what
"byte-identical on NVIDIA" protects is which values the GEMMs read, and this
buffer has no reader: `quantize_to_nvfp4` has already consumed it into a fresh
NVFP4 allocation and `AttentionWeights` keeps only that result and the two norm
pointers. An allocation nothing reads is not behaviour. The FP8 dtype test stays
and is not the knob in disguise: it is the proof that `dense_bf16` is a fresh
allocation rather than the store's own pointer, which `dense_auto` returns
uncopied for a BF16 tensor.

### `spark-model/build.rs` documents a free that does not exist

The `atlas_scale` comment says the cfg exists "so the weight loader frees each
FP8 source tensor right after requant (see `quantized_from_fp8`)".
`quantized_from_fp8` (`nvfp4_detect.rs:366-370`) frees the **BF16 intermediate**
and leaves the FP8 source in the store. No `#[cfg(atlas_scale)]` appears
anywhere in `spark-model/src`. The cfg is real and load-bearing (it pins the
32-row prefill grid stride and the GDN prefill arm), but this particular
sentence describes an intent, not an implementation. `ATLAS_LOAD_RELEASE_SOURCES`
is that implementation.

## `ATLAS_LOAD_RELEASE_SOURCES`

`1`/`0`. Default **ON** under `cfg!(atlas_scale)`, **OFF** otherwise.

Why the asymmetry, stated plainly so it can be argued with:

* On GB10 the store's residency is close to free. 121 GB of unified memory
  against a 21.81 GiB checkpoint means the dead 9.94 GiB is not the thing
  standing between the serve and a KV cache, and the whole
  `prune_after_load` design (`weight_loader/mod.rs:240`) already exists for the
  cases where it is. Keeping the store intact costs nothing there and removes a
  class of use-after-free with no diagnostic: a store tensor freed while some
  arm nobody thought about still aliases it does not fault, it reads whatever
  the allocator handed out next.
* On a 32 GB discrete board the same 9.94 GiB is 31% of the card. It is the
  difference between loading and not, and the failure mode without it is the
  `cuMemAlloc_v2 status 2` above rather than a slow serve.

The knob is read once (`WeightStore::release_sources_enabled`) and the default
is a compile-time cfg rather than a hardware probe, so an NVIDIA build with the
variable unset executes exactly the instructions it executed before this
change.

### Release sites

Each releases `{prefix}.weight` and the scale keys that fed the dequant, after
the consuming kernel has completed on the load stream.

| site | file | what it releases | per model |
|---|---|---|---|
| attention q/k/v/o, CompressedTensors arm | `qwen35_dense.rs` | 4 FP8 tensors + scales x16 layers | 1.56 GiB |
| SSM `in_proj_qkv`/`in_proj_z`/`out_proj`, dequant path | `qwen35_dense.rs` | 3 FP8 tensors + scales x48 layers | 5.16 GiB |
| dense FFN gate/up/down, FP8 layers | `qwen35_dense.rs` | 3 FP8 tensors + scales x8 layers | 1.99 GiB |
| attention BF16 dequant (not a store tensor; a leaked derived buffer; NO LONGER on this knob, see above) | `qwen35_dense.rs` | the `dense_bf16` intermediate | 3.12 GiB |

NOT released, deliberately:

* **`lm_head` when a consumer binds it zero-copy, which the default serve does
  NOT.** `lm_head_setup.rs::native_fp8_lm_head_share` binds the store's FP8
  `lm_head.weight` ZERO-COPY, but it is only reached from two places:
  `setup_lm_heads` under `--lm-head-dtype fp8` (`config.lm_head_fp8`), and
  `build.rs` under `--dflash` for the drafter tail. With neither flag set,
  `load_lm_head` (`loaders_b.rs:52-64`) has already dequanted the FP8 bytes
  into a FRESH BF16 allocation and every head is built from that, so the store's
  copy has no reader at all. **RELEASED as of 2026-09-17** by
  `lm_head_setup::release_lm_head_source`, called from `build_model` right after
  `setup_lm_heads` and BEFORE the KV sizer, on the same
  `ATLAS_LOAD_RELEASE_SOURCES` knob as the sites above. 1.18 GiB. The guard is
  `lm_head_source_is_dead(source_is_fp8, lm_head_fp8, dflash, speculative)`, and
  `source_is_fp8` is the load-bearing term: on a BF16 or NVFP4-prepacked
  checkpoint `load_lm_head` / `weight_map::quantized` hand the STORE's pointer
  through uncopied, and releasing it there is a use-after-free on the first
  token. `lm_head_setup_tests.rs` pins all sixteen rows of that table.
* **SSM `out_proj` when `ATLAS_FP8_ROWWISE=1`.** `rowwise_fp8::load_fp8_per_row`
  returns `weight: w.ptr` (`rowwise_fp8.rs:179`), so under that flag the layer
  holds the store's bytes. The release predicate excludes it.
* **attention and SSM under `ATLAS_DENSE_FP8=1`.**
  `load_fp8_block_scaled_as_fp8weight` is also zero-copy on `.weight`. That arm
  is unreachable on a CompressedTensors checkpoint, but the predicate checks
  rather than assumes.
* **SSM `in_proj_a`/`in_proj_b`.** BF16, so `dense_auto` hands back the store
  pointer and the release would have to prove the loader had not already freed
  it through that alias. 45 MiB is not worth the proof obligation.
* **the packed NVFP4 dense FFN.** Alive. Freeing it is a use-after-free on every
  decode step.

### Ordering

`gpu.free` on a buffer a queued kernel still reads is undefined. Two of the
three producing operations already synchronise:
`quantize_to_nvfp4` ends with `gpu.synchronize(stream)`
(`loaders_fp8.rs:247`), and `transpose_for_gemm_gs` ends with
`gpu.synchronize(0)` (`quantized.rs:281`). The one that does NOT is
`dequant_fp8_blockscaled_to_bf16`, which deliberately skips its per-call
synchronize (`quant_helpers.rs:120-124`: it cost ~104 s of cold-load wall on a
30k-call MoE).

So the release helper synchronises the load stream itself before freeing
anything. It is one `cuStreamSynchronize` per release batch at load time, on a
path that already costs minutes, against a class of corruption that would
surface as wrong logits rather than a fault. Correctness over speed.

## Two latent double-frees this accounting surfaced

Neither is fixed here: both are on paths this checkpoint does not take, and
neither can be gated on a serve tonight. They belong in the record because they
are the same mistake `release_tensor` exists to make impossible, and because
switching them to it is a one-line fix with identical steady-state residency.

**`qwen35_dense.rs:1479-1480`** frees `qkv_dense.weight` and `z_dense.weight`
unconditionally after the concat. Those come from `load_ssm_proj` to
`dense_auto`, which returns the STORE's pointer for a BF16 tensor. On a
`Bf16Raw` GDN checkpoint the loader therefore frees store memory the store still
lists, and `WeightStore::release` frees it again at teardown. The same shape
applies to `out_proj_dense` at `:1693`.

**`nvfp4_detect.rs:332`**, the `Bf16Raw` arm of `quantized_any`, does
`gpu.free(w.ptr)` where `w` came straight from `store.get`. Its own comment
explains why the free is right (a 35B BF16 MoE would otherwise hold both the
~60 GB of BF16 experts and the ~22 GB of NVFP4 copies) and it is right. What is
wrong is the route: the store keeps the entry, so teardown frees it again. This
one fires on EVERY `Bf16Raw` checkpoint, which is every raw BF16 fine-tune Atlas
serves.

The new bookkeeping cannot see either, because the free goes through `gpu.free`
directly rather than through the store. `store.release_tensor(gpu, name)` frees
the same pointer and records it, so both become correct by substitution.

## Ranked list: what would have to change to fit

0. **Route the two direct store frees above through `release_tensor`.** No
   residency change at all, and it removes a double free that fires on every
   `Bf16Raw` checkpoint. Listed first because it is the cheapest and the only
   item that is a correctness fix rather than a memory one. **DONE**, plus a
   third of identical shape on the keep-packed Q2_0 GDN arm that this list did
   not walk. `free_maybe_store_owned` compares POINTERS rather than dtypes, so a
   TP shard, a dequant output and a concat, all of which reach those sites
   through the same variable, keep the plain `gpu.free` they had.
1. **Release the FP8 `lm_head`** behind a `!config.lm_head_fp8 &&
   !use_speculative` guard. 1.18 GiB, no kernel work. **DONE**, as
   `lm_head_setup::release_lm_head_source`. It did not need the loader trait to
   learn about the LM-head route after all: `build_model` already holds the
   store, the config and `dflash_args` at the point `setup_lm_heads` returns,
   and that point is also the last one BEFORE the KV sizer reads
   `free_memory()`, which matters, because a release the sizer cannot see is a
   release the KV cache does not get. `prune_after_load` would have been too
   late for exactly that reason. The guard grew a fourth term the list did not
   name: the checkpoint's `lm_head` must actually be FP8, because that is what
   proves `load_lm_head` made a copy rather than handing back the store's own
   pointer.
2. **Do not build the transposed dense FFN twins.** 8.96 GiB, the single
   largest item left. `DenseFfnWeights::{gate,up,down}_proj_t` are already
   `Option`, and `gemma4/loader_a.rs:30-52` is a working precedent
   (`ffn_transpose_fits`, `ATLAS_GEMMA4_FFN_TRANSPOSE=0`). The cost is written
   down in that same comment and it is large: with all three `None` every fast
   arm of `w4_gemm!` is skipped and prefill lands on plain `w4a16_gemm`, which
   the Gemma-4-31B measurement puts at ~7.0 TFLOP/s against ~51 TFLOP/s for
   `t_m128`. Call it a 7x slower FFN prefill. On a board that otherwise cannot
   load the model at all, that trade is available; it should be a knob with a
   measured A/B, not a silent default. **DONE**, as
   `ATLAS_LOAD_TRANSPOSED_TWINS` above, together with items 3 and 4: one lever
   for all three families rather than three, because a build that skips one and
   builds the others has no state anyone measured. **The A/B is now RUN, and it
   went the other way**: on gfx1201 `w4a16_gemm_t_m128` measures ~1 TFLOP/s and
   the plain `w4a16_gemm` ~4 (R9700 prefill, 2026-09-17, 9B with the twins
   against 27B without). The 7x above is GB10's and was never SCALE's, so the
   SCALE default moved from `auto` to `0`: dropping the twins is not a trade
   here, it is free.
3. **Do not build the transposed SSM twins.** 2.90 GiB. `qkvz_nvfp4_t` and
   `out_proj_nvfp4_t` are `Option` and every consumer is guarded
   (`trait_prefill_proj.rs:120`, `:327`, `trait_decode_batched.rs:424`, `:483`),
   but `predequant_for_prefill` keys off `out_proj_nvfp4_t.is_some()`
   (`init_fp8.rs:110`), so dropping it also drops the 1.41 GiB FP8 predequant,
   for 4.31 GiB total. Unmeasured cost. **DONE**, under the same lever as
   item 2.
4. **Do not build the transposed attention twins.** 0.88 GiB. Smallest of the
   three and the best-documented win (the fused `[q|k|v]` twin is not built at
   all on this checkpoint: `quantize_to_nvfp4` derives a per-tensor `scale2`
   from each projection's own absmax, so the `scales_equal` test at
   `qwen35_dense.rs:997-999` fails and the loader logs the warning). **DONE**,
   under the same lever as item 2. The fused twin is NOT in the projection for
   exactly the reason this item gives: pricing a copy that is usually absent
   would make `auto` refuse room it does not need.
5. **Keep the attention and SSM projections FP8 end to end.** This is the
   correct answer and it is blocked on a kernel: the checkpoint's scale is
   per-CHANNEL `[N,1]`, every `w8a16` kernel indexes `block_scale[n/128, k/128]`
   (`proj_is_fp8_any_scale` refuses it for exactly that reason), and the
   row-wise cuBLASLt FP8 GEMM is dead on this class
   (`rowwise_fp8.rs`, sm_121 heuristic status 15, and SCALE has no e4m3 MMA
   codegen on gfx1201 at all, which is why `ATLAS_W4A16_VARIANT=v1` is
   required). A per-row `w8a16_gemv`/`gemm` is the missing piece. It would save
   the 6.72 GiB of FP8 store plus the 9.6 GiB of NVFP4 the requant produces.
6. **Serve `Qwen/Qwen3.8-27B-FP8` instead.** Block-scaled FP8, which Atlas loads
   natively with no requant, is the layout this loader is fastest and smallest
   on. `kernels/r9700/qwen3.8-27b/MODEL.toml` already records that the
   mixed-precision checkpoint measured worse on the video benchmark's hardest
   leg.
7. **Serve a smaller model.** 27B at 4-bit plus a second 4-bit layout plus a KV
   cache is simply not a 32 GB workload on this loader today.

## What was NOT verified on macOS

`cargo check -p spark-model` does not build on macOS (pre-existing Linux-only
libc symbols in `spark-storage`), so every `spark-model` change in this work is
reviewed by reading, not by the compiler. The R9700 build is the real gate.
