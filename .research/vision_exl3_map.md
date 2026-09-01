# Vision (ViT) on the EXL3 Qwen3.8-Flash-Next export — map, defects, native plan, test plan

Synthesized 2026-09-01 from three read-only passes (local header inventory of
`/tank/exl3-ckpt/qwen38-flash-next-2.05bpw`, header-only inventory of all six
refs of `turboderp/Qwen3.8-Flash-Next-exl3`, and a code-reading pass over the
ViT dispatch + the in-progress native dense arm). Nothing here was built or
run on a GPU; every claim is header data or `file:line` in this worktree
(`wip/exl3-research`). Scratch data: `/home/ms/.claude/jobs/5a7bd33d/tmp/vision/`
(`local_inventory.md`, `remote_inventory.md`, `visual_tensors_full.json`,
`remote/<branch>/*.header.json`, `family_params.py`, `branch_fit.py`).

---

## 1. Executive summary

**Does image inference plausibly work TODAY on the 2.05bpw serve? — The tower
loads and runs, but the MLP output is silently wrong in all 27 blocks.**

What is consumed on the 2.05 serve (`/home/ms/run_exl3_ep2.sh`, binary
`/home/ms/spark-exl3`, port 8890):

- The ViT is present in the main index-listed shard `model-00005-of-00005.safetensors`
  (987 `model.visual.*` tensors, 449.5 MB). The stale claim in
  `.research/EXL3_DECODE_FINDINGS.md:204-206,225` and the module doc at
  `crates/spark-model/src/weight_map/exl3_materialize.rs:38-39` ("this branch
  ships NO vision shard") is wrong for this branch — the tower IS loaded, and the
  boot log `Qwen3.6 vision encoder loaded: depth=27, hidden=1152, heads=16,
  FP8-blocks>=4` (`crates/spark-model/src/weight_loader/qwen35.rs:266-271`)
  proves it. The `FP8-blocks>=4` suffix is a hard-coded literal, not a
  measurement (no visual tensor is FP8 here).
- `qwen4_exp.rs:446-458` delegates to `Qwen35WeightLoader::load_vision_encoder`
  (`qwen35.rs:160-273`), which reads:
  - `attn.qkv` from the **fused, unquantized BF16** `model.visual.blocks.{b}.attn.qkv.weight [3456,1152]`
    + `.bias [3456]` that the export keeps alongside the trellis q/k/v
    (`qwen35.rs:205-206`). Its bias equals `bf16(concat(q,k,v .bias))` to 1 ulp
    in all 27 blocks (`compare_bias.out`), and the NVFP4 baseline checkpoint's
    tower is BF16 with exactly this fused `[3456,1152]` tensor, so this is the
    original Qwen3.8 qkv passed through.
  - `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`, `merger.linear_fc1/fc2`
    from the **K=4 trellis reconstructed to BF16** by `materialize_exl3`
    (`exl3_materialize.rs:487-534`, BF16 arm; `wants_nvfp4_triplet` matches
    only `.mlp.experts.`/`.mlp.shared_expert.`, and no native keep predicate
    matches `model.visual.*` — `exl3_native_serves_with` at
    `exl3_materialize.rs:101-109`, `exl3_dense_prefix_family` at
    `exl3_materialize_dense.rs:248` matches only `.linear_attn.`/`.self_attn.`).
  - F16 biases/norms/patch_embed are converted to BF16 at ingest
    (`crates/spark-runtime/src/weights/loader/load_fns.rs:125-136`; only
    `.suh`/`.svh` keep F16 bits, `weights/exl3.rs:50-52`).
  - The separate trellis `attn.{q,k,v}_proj` are ALSO materialized (3 × [1152,1152]
    BF16 × 27 = **215 MB**) and never read by any loader — dead resident memory.

  So today's ViT is hybrid-fidelity: 16-bit attention qkv, 4-bit-reconstructed
  proj/fc1/fc2/merger.

- **The live defect.** `config.json vision_config.intermediate_size = 4304`
  (real — the NVFP4 baseline has `linear_fc2.weight BF16 [1152,4304]`), parsed
  at `crates/atlas-core/src/config/parsers/vision.rs:47` and never reconciled
  with tensor shapes. EXL3 pads to a 128-multiple: fc1 trellis `[72,272,64]`
  → out **4352**, fc2 trellis `[272,72,64]` → in **4352**, and materialization
  inserts `.weight` with `shape [n, k]` from the trellis dims
  (`exl3_materialize.rs:525-532`), i.e. fc2 = `[1152, 4352]`. The ViT launches
  fc2 with `k = inter = 4304` (`vit_block.rs:313-323` single-image,
  `:505-515` batched) and `dense_gemm_bf16_pipelined` reads
  `B[n*K + k]` (`kernels/gb10/common/dense_gemm_bf16.cu:10`), so every output
  row n ≥ 1 reads at a 48·n-element drift. Max index read
  1151·4304+4303 = 4,958,207 < 5,013,504 → **no fault, garbage fc2 output in
  every block**. fc1 (`[4352,1152]` read as `[4304,1152]`, stride 1152) and the
  merger (4608/2560 exact) are fine. Expected symptom: the model produces a
  fluent but image-unrelated (or hallucinated) description. Needs a GPU repro
  (section 5) to confirm.

**Higher-bpw branches.** Every bpw branch ships the same 987 `model.visual.*`
tensors AND the fused BF16/F16 `attn.qkv.weight` (no branch is trellis-only for
qkv). Vision K = 6/6/6/5/4 for 6.05/5.05/4.05/3.05/2.05. Reconstruct kernels
exist for K=1..8 (`kernels/gb10/common/exl3_reconstruct.cu:540-563`), so
materialize-to-BF16 works at K=5/6 too, with the SAME fc2 stride bug.

- **6.05 / 5.05 / 3.05**: vision in the last index-listed main shard → loads
  exactly like 2.05 (same defect, same dead q/k/v).
- **4.05bpw_h6_ng6 is the outlier**: vision lives ONLY in a separate
  `vision_k6.safetensors` (561.4 MB, F16 fused qkv / pos_embed) that is NOT in
  `model.safetensors.index.json` (0 visual entries; `quantization_config` also
  lacks `vision_bits`). Atlas's shard discovery reads only index-listed shards
  plus `extra_weights.safetensors` (`crates/spark-runtime/src/weights/loader.rs:23-66,164-171`)
  and the ngram sidecar (`register_exl3_ngram_sidecar`,
  `exl3_materialize.rs:149-160`). Nothing references `vision_k6`/`vision_*.safetensors`
  in `crates/`. On 4.05 the store holds zero `model.visual.*`, `qwen35.rs:180-185`
  logs "Vision encoder tensors absent … text-only mode" and
  `qwen4_exp/probe.rs:117` counts 0 ViT blocks. **Yes, a sidecar registration
  analogous to `register_exl3_ngram_sidecar` is needed** — a
  `register_exl3_vision_sidecar` that probes `model_dir/vision_*.safetensors`
  and loads it with no skip filter (or the untested workaround: symlink
  `vision_k6.safetensors -> extra_weights.safetensors`, which `loader.rs:164`
  loads with `no_skip`; materialize then sees its trellis tensors normally).
- Native EXL3 serving is gated to K ∈ {2,4} (`exl3_native_supported`,
  `exl3_materialize.rs:112-131`), so native vision (section 4) fires only on
  2.05; on 3.05+ the tower materializes.

---

## 2. Per-branch table

Sources: `remote_inventory.md` §1-4 (HF `refs`/`tree`, headers via HTTP Range),
`family_params.py` (per-family parameter counts from the local 2.05 shard
headers; trellis I16 `[in/16, out/16, 16K]` → params = in·out), `branch_fit.py`.

| bpw branch | files | vision shard | vision K | fused BF16 qkv | K: GDN / attn / experts / lm_head / shared / mtp-experts / mtp.fc | total on disk | ngram file |
|---|---|---|---|---|---|---|---|
| 6.05bpw_h6_ng6 | 13 shards + ngram + mtp patch | `model-00013-of-00013` (index-listed) | 6 | YES (BF16) | 8 / 8 / 6 / 6 / 8 / 6 / 7 | 139.04 GB | 39.04 GB, monolithic `[320001536,61]` |
| 5.05bpw_h6_ng6 | 11 shards + ngram + mtp patch | `model-00011-of-00011` (index-listed) | 6 | YES (BF16) | 7 / 7 / 5 / 6 / 7 / 5 / 6 | 123.25 GB | 39.04 GB, monolithic |
| 4.05bpw_h6_ng6 | 9 shards + ngram + **`vision_k6.safetensors`**, NO mtp patch | **`vision_k6.safetensors` 561.4 MB, NOT in index** | 6 | YES (**F16**) | 6 / 6 / 4 / 6 / 6 / 4 / 5 | 107.46 GB | 39.04 GB, monolithic |
| 3.05bpw_h5_ng5 | 7 shards + ngram + mtp patch | `model-00007-of-00007` (index-listed) | 5 | YES (BF16) | 5 / 5 / 3 / 5 / 5 / 3 / 4 | 85.14 GB | 32.64 GB, 128 × `shard_i.trellis [2500012,51]` |
| 2.05bpw_h4_ng4 (local) | 5 shards + ngram + mtp patch | `model-00005-of-00005` (index-listed) | 4 | YES (BF16) | 4 / 4 / 2 / 4 / 4 / 2 / 3 | 62.82 GB | 26.24 GB, 128 × `shard_i.trellis [2500012,41]` |

QSA indexer (`self_attn.indexer.index_qk_proj`) and mtp indexer follow the
expert K (6/5/4/3/2). `main` has README + qbench only. All bpw READMEs are the
upstream model card (no quant notes). The local checkpoint matches the remote
2.05 branch file-for-file by byte size.

### Single-node fit estimate at PACKED (native) size

Anchor: routed experts = 73,728 matrices `[2560,640]` = 120.796 B params;
packed bytes = params·K/8 + suh/svh F16 aux (471.9 MB) = **30.67 GB @ K=2**
(reproduced exactly by `family_params.py`). Applying params·K/8 + aux per
family with each branch's K ladder (`branch_fit.py`); non-trellis tensors
(embed_tokens BF16 1.27 GB, hyper-connection F16 1.27 GB, fused ViT qkv 0.215 GB,
PLE/other) are a constant 3.02 GB. The ngram file is NVMe-faulted through the
PLE row cache (not resident; ~344 MB at 4M slots per `run_exl3_ep2.sh`).

| branch | experts (+mtp experts) | dense (GDN+attn+shared+lm_head+mtp) | vision packed | packed linears | + non-trellis = resident weights | vs GB10 budget (121 GB × util) |
|---|---|---|---|---|---|---|
| 6.05 | 92.97 | 3.47 | 0.34 | 96.78 | **99.8 GB** | exceeds util 0.76 (92 GB); needs util ≥ 0.85 with ~0 KV — effectively EP=2 only |
| 5.05 | 77.55 | 3.10 | 0.34 | 80.99 | **84.0 GB** | fits util 0.76 with ~8 GB for KV/PLE/scratch — tight |
| 4.05 | 62.14 | 2.72 | 0.34 | 65.20 | **68.2 GB** | fits single-node (util 0.6 = 73 GB leaves ~5 GB; 0.76 leaves ~24 GB) |
| 3.05 | 46.72 | 2.27 | 0.28 | 49.27 | **52.3 GB** | fits comfortably |
| 2.05 | 31.31 | 1.81 | 0.22 | 33.35 | **36.4 GB** | fits comfortably |
| any (today's compat path) | 69.36 (NVFP4 re-quant, K-independent) | 8.16 (BF16) | 0.89+0.22 (BF16, incl. dead q/k/v) | — | **~80.6 GB** | why the run scripts use EP=2 |

Native serving only exists for K ∈ {2,4} today, so 3.05+ resident sizes are
hypothetical until the gate widens; K=7 (5.05 GDN/attn/shared, 6.05 mtp.fc)
has GEMM/GEMV kernels for K ∈ {2,3,4,5,6,8} / {2,3,4} only
(`layers/ops/exl3_matmul.rs:105-109,316`; `kernels/gb10/common/exl3_matmul.cu:34,61`).

---

## 3. Loader mapping table (2.05 local checkpoint; identical on 6.05/5.05/3.05, absent from the store on 4.05)

| tensor (per block b ∈ 0..26 unless noted) | on-disk dtype / shape | Atlas consumer | conversion on the way | status |
|---|---|---|---|---|
| `model.visual.patch_embed.proj.weight` | F16 `[1152,3,2,16,16]` | `vision_tensor_dense_auto` (`qwen35.rs:187`) → `patch_embed_w`, used as `[1152,1536]` BF16 GEMM B (`patch_embed.rs:156-166`) | F16→BF16 host RNE at ingest (`load_fns.rs:125-136`) | OK (3 mantissa bits lost vs F16; minor) |
| `model.visual.patch_embed.proj.bias` | F16 `[1152]` | `vision_tensor_dense_auto` (`qwen35.rs:189`) | F16→BF16 | OK |
| `model.visual.pos_embed.weight` | BF16 `[2304,1152]` | `vision_tensor_dense_auto` (`qwen35.rs:190`); host bilinear resample (`forward.rs:87-116`) | none | OK |
| `blocks.{b}.norm1.{weight,bias}`, `norm2.{weight,bias}` | F16 `[1152]` | `vision_tensor_dense_auto` (`qwen35.rs:202-204,212-213`) | F16→BF16 | OK |
| `blocks.{b}.attn.qkv.weight` | **BF16 `[3456,1152]` (unquantized copy)** | `vision_dense_auto` → `dense_auto_fp8_or_bf16` BF16 arm (`qwen35.rs:17-31`, `weight_map/model_a.rs:190-202`) → `qkv_w`; GEMM site `vit_block.rs:376-386` | none (raw ptr) | OK — 16-bit attention weights |
| `blocks.{b}.attn.qkv.bias` | BF16 `[3456]` | `vision_tensor_dense_auto` (`qwen35.rs:206`) | none | OK (== concat(q,k,v F16 bias) to 1 ulp) |
| `blocks.{b}.attn.{q,k,v}_proj.{trellis,suh,svh,mul1}` | I16 `[72,72,64]` K=4 + F16 `[1152]` ×2 + I32 | `materialize_exl3` BF16 arm → `.weight [1152,1152]` BF16 inserted; **no loader reads it** | trellis→BF16 reconstruct | **DEAD 215 MB resident** (store adopted into the model, `model/types.rs:594`) |
| `blocks.{b}.attn.{q,k,v}_proj.bias` | F16 `[1152]` | none | F16→BF16 | dead 0.19 MB |
| `blocks.{b}.attn.proj.{trellis,…}` | I16 `[72,72,64]` K=4 | materialize → `.weight [1152,1152]` BF16 → `vision_dense_auto` (`qwen35.rs:207`) → `proj_w`; site `vit_block.rs:437-447` | trellis→BF16 | OK (4-bit fidelity) |
| `blocks.{b}.attn.proj.bias` | F16 `[1152]` | `qwen35.rs:208` | F16→BF16 | OK |
| `blocks.{b}.mlp.linear_fc1.{trellis,…}` | I16 `[72,272,64]` → `[4352,1152]` | materialize → `.weight [4352,1152]` BF16 → `fc1_w`; site `vit_block.rs:486-496` launched with n = inter = 4304 | trellis→BF16 | OK by luck (reads rows 0..4303 at stride 1152; pad rows 4304.. have svh = bias = 0 exactly) |
| `blocks.{b}.mlp.linear_fc1.bias` | F16 `[4352]` (pad = 0) | `qwen35.rs:216` | F16→BF16 | OK (first 4304 used) |
| `blocks.{b}.mlp.linear_fc2.{trellis,…}` | I16 `[272,72,64]` → `[1152,4352]`; suh pad cols 4304..4351 NON-zero (±0.11) | materialize → `.weight [1152,4352]` BF16 → `fc2_w`; site `vit_block.rs:505-515` launched with **k = 4304** | trellis→BF16 | **DEFECT: row-stride mismatch, silent garbage in all 27 blocks** |
| `blocks.{b}.mlp.linear_fc2.bias` | F16 `[1152]` | `qwen35.rs:218` | F16→BF16 | OK |
| `merger.norm.{weight,bias}` | F16 `[1152]` | `qwen35.rs:238-239` | F16→BF16 | OK |
| `merger.linear_fc1.{trellis,…}` | I16 `[288,288,64]` → `[4608,4608]` | materialize → BF16 → `fc1_w`; site `merger.rs:68-78` | trellis→BF16 | OK (4608 % 128 == 0, no padding) |
| `merger.linear_fc2.{trellis,…}` | I16 `[288,160,64]` → `[2560,4608]` | materialize → BF16 → `fc2_w`; site `merger.rs:87-97` | trellis→BF16 | OK |
| `merger.linear_fc{1,2}.bias` | F16 `[4608]`/`[2560]` | `qwen35.rs:241-244` | F16→BF16 | OK |
| `deepstack_merger_list.*` | absent | loop over `deepstack_visual_indexes` = `[]` | — | n/a |
| `vision_k6.safetensors` (4.05 only) | F16/I16, 987 tensors | **nothing** — not index-listed, no glob | — | **NOT LOADED → text-only on 4.05** |

Memory today (decimal MB): fused qkv 215.2 + dead q/k/v 215.0 + proj 71.7 +
fc1 270.7 + fc2 270.7 (both at padded 4352) + merger 66.1 ≈ **1,109 MB** of
BF16 for a tower whose packed form is 224.7 MB.

---

## 4. Native EXL3 vision design

### 4.1 Call-site inventory
Every ViT weight GEMM funnels through `VisionEncoder::vit_gemm_bias`
(`crates/spark-model/src/layers/vision_encoder/enc_impl/vit_block.rs:19-63`):
`C[m,n] = A[m,k] @ B[n,k]^T` via `dense_gemm_bf16_pipelined` (grid
`[n/128, m/128]`), then a SEPARATE `vision_add_bias` launch
(`kernels/gb10/qwen3.8-flash-next/nvfp4/vision_encoder.cu:70-81`). C is always
a contiguous `[m,n]` block; the bias lands after the GEMM — exactly the shape
the EXL3 kernels (no bias epilogue) need. M = Σ patches of the batched images,
≤ `CEILING_MAX_PATCHES` = 16384 (`init.rs:34`); `buf_wide` is
`p_max × intermediate_size × 2` bytes (`init.rs:200`).

| # | site | k → n | EXL3 GEMM shape (exl3_matmul.rs:139-197) | C dest | anchor |
|---|---|---|---|---|---|
| 0 | patch_embed | 1536 → 1152 | n/a (F16 weight, not trellis) — stays BF16 | `buf_h1` | `patch_embed.rs:156-166` |
| 1 | attn.qkv | 1152 → 3456 | fused BF16 today; split native q/k/v = 3 × sh1 (k ≤ 2048, K=4) | `buf_wide` as `[M,3456]` arena, consumed as fused `[seq,3·H·D]` by `vit_rope_deinterleave` | `vit_block.rs:376-386` (batched), `:204-214` |
| 2 | attn.proj | 1152 → 1152 | sh1 | `buf_wide [M,1152]` → residual → `buf_h1` | `vit_block.rs:437-447` |
| 3 | mlp.fc1 | 1152 → 4352 | sh1 | `buf_wide [M,inter]` → GELU in place | `vit_block.rs:486-496` |
| 4 | mlp.fc2 | 4352 → 1152 | sh2 | `buf_h1 [M,1152]` → residual | `vit_block.rs:505-515` |
| 5 | merger fc1 | 4608 → 4608 | sh3 | `buf_merge_fc1` → GELU | `merger.rs:68-78` |
| 6 | merger fc2 | 4608 → 2560 | sh2 | `buf_out` row block, stride = n | `merger.rs:87-97` |

Stream: the encode runs on `gpu.default_stream()` (= the DECODE stream;
`model/trait_impl/prefill_a/vision.rs:25,93`, `scheduler/mod.rs:1154-1163`),
concurrently with chunked prefill on `prefill_stream` (fenced behind the
encode by `prefill_event`, `scheduler/prefill_a_step.rs:352-363`,
`phase_start_prefills.rs:130-146`). Vision never runs under CUDA-graph capture.

### 4.2 Reusable arm
`ops::exl3_dense_linear` / `exl3_dense_linear_shared_a`
(`layers/ops/exl3_dense.rs:162,177`; `Exl3DenseOut::{contiguous,strided}` at
`:143-152`) already do bf16→f16 ingress, GEMV (m ≤ 8) / GEMM tiering, row
batching at `stage.rows_cap` (2048 default, `layers/ops/exl3_dense/stage.rs:43-66`),
contiguous in-place f16→bf16 egress and the strided `_2d` egress via
`stage.c_f16`, all under ONE `Exl3LaunchState::section` per call
(`launch_state.rs:145-178`). A ViT arm is `vit_gemm_bias` → Exl3 branch =
`exl3_dense_linear(gpu, w, a, Exl3DenseOut::contiguous(c), m, stage, stream)`
then the existing `k_add_bias` launch. (These files are being edited by the
other workflow; signatures may still move.)

### 4.3 Options and memory (MB, decimal)
| option | qkv | proj/fc1/fc2/merger | resident ViT weights | saving vs today (1,109) |
|---|---|---|---|---|
| Zero-risk sub-step | fused BF16 (215.2) | materialized BF16 (679.2) | 894 | **215** (just stop materializing the unread `attn.{q,k,v}_proj`) |
| **B (recommended first)** | fused BF16 (215.2) | native packed 18.0 + 136.0 + 16.5 | **385.7** | **723** |
| A (all native) | split native q/k/v 54.3 (+ drop fused 215.2) | native | **224.8** (+8.4 to widen the stage `c_f16` to max_out 4608) | **884** |

Option A qkv: the checkpoint has separate trellis q/k/v with separate suh/svh
(cannot be concatenated into one packed tensor) →
`exl3_dense_linear_shared_a(&[(q, strided(buf_wide, 3456)), (k, strided(buf_wide + 1152·2, 3456)), (v, strided(buf_wide + 2304·2, 3456))], buf_h1, M, stage)`,
then ONE `vision_add_bias(buf_wide, attn.qkv.bias, M, 3456)` reusing the fused
BF16 bias (bit-equal to the concat).

### 4.4 Launch counts per image (27 blocks; B = ceil(M/2048) row batches)
Today: 8 launches/block (4 × gemm+bias) = 216/image. Option B: 9B+5/block →
378 at M=1024, 2,079 at M=16384. Option A: 16B+4/block → 540 / 3,564. Merger
adds 2 × (3B'+1), B' = ceil(M/4/2048). At ~5 µs/launch the worst case is
≈ +17 ms on a 16K-patch encode that takes seconds — SM time, not launch
overhead, is the lever.

### 4.5 Hazards
1. **fc2 stride bug (live, independent of native)** — section 1. Fix: derive
   `intermediate_size` from the fc1 `.weight` shape (`shape[0]` = 4352) or the
   trellis `shape[1]·16`, pass it to `VisionEncoder::new` (`qwen35.rs:261`)
   so `buf_wide` (`init.rs:200`), GELU (`vit_block.rs:498-503`) and add_bias
   cover 4352. fc1 pad rows have svh = bias = 0 → GELU(0) = 0 → fc2 pad columns
   contribute nothing. Alternative: slice the materialized fc2 to `[1152,4304]`.
2. **Padding contract for native fc2**: fc2's pad rows carry NONZERO suh/codes,
   so activation cols 4304..4351 MUST be exactly zero. That holds only if fc1
   writes all 4352 columns into a `[M,4352]` buffer (`buf_wide` is reused as the
   qkv arena and proj output, so stale bytes otherwise leak in). Keep fc1/fc2
   atomic (both native or both BF16) and size inter at 4352.
3. **Deadlock class**: the ViT runs on the default stream concurrently with
   prefill-stream persistent MoE / cooperative dense launches; two cooperative
   kernels partially co-resident deadlock and the split-K locks are shared
   (`launch_state.rs:9-38`). The ViT MUST dispatch through the model-shared
   `Exl3LaunchState` (sections + device fence), NOT an lm_head-style private
   locks buffer (`model/lm_head_exl3.rs:47-54`, justified by call-site
   exclusivity that a ViT does not have). Cost: decode/prefill latency jitter
   during image encodes (a qkv section at M=16384 = 8 row batches × 3 GEMMs).
4. **Stage geometry**: `NativeExl3::stage()` (`weight_loader/qwen4_exp/exl3_dense.rs:71-100`)
   sizes the shared stage from GDN/attention maxima and
   `Exl3DenseStage::get_or_create` refuses growth (`stage.rs:119-144`). With
   the GDN family on, max_out already includes `full_conv_dim`/`full_value_dim`
   (≥ 6144 ≥ 4608) and max_in ≥ 6144; with only the attention family on it may
   be smaller than the merger's 4608 — fold `max(.., 4608)` in whenever the
   vision gate is on, BEFORE the first layer creates the stage.
5. **Loader plumbing**: `NativeExl3` is a local inside
   `Qwen4ExpWeightLoader::load_layers` (`qwen4_exp.rs:262`); the trait's
   `load_vision_encoder(&self, store, config, gpu)` (`qwen4_exp.rs:446`) runs
   later from `factory/build.rs:330` (materialize `:108` → `load_layers` `:204`
   → vision `:330`) and cannot reach it — needs a
   `Mutex<Option<Arc<Exl3DenseStage>>>` on the (unit) loader struct
   (`qwen4_exp.rs:135`) or a factory-side handoff.
6. **Kept-packed prefixes have no `.weight`**: `vision_dense_auto`
   (`qwen35.rs:17-31`) does `store.get("{prefix}.weight")` first → must branch
   on `is_exl3_linear(store, prefix)` before the get, or the load fails.
7. **sh1 coverage**: k = 1152 ≤ 2048 with K=4 selects GEMM shape 1, which the
   LM never uses (hidden 2560) and which parity covers only at a single
   `[128→128]` m=17 case (`exl3_native_parity/main.rs:79`); the dense arm's
   `probe_kernels` probes sh2 only (`layers/exl3_dense.rs:96-100`). A ViT probe
   must add sh1 + sh3 (f16 and f32 variants) and parity should run at
   `[1152→1152]`, `[1152→4352]`, `[4352→1152]`, `[4608→4608]`.
8. **f16 ingress saturation**: every EXL3 input is post-LayerNorm, attention
   output, or GELU(fc1) — bounded; the raw residual never enters a trellis
   GEMM. fp16 C could saturate only if a pre-activation exceeds 65504
   (fc1 |bias| ≤ 9.4 — implausible); the f32-C arm exists as fallback.
9. **Memory bookkeeping**: Option A must `store.remove` + `gpu.free` the fused
   BF16 `attn.qkv.weight`; Option B must NOT materialize `attn.{q,k,v}_proj`.
   Kept-packed tensors must declare an alloc owner tag like the other native
   families or the alloc ledger will show them as anonymous.
10. `Exl3Weight::from_store` does a 4-byte D2H (mul1 readback) per tensor →
    164 stream syncs at load for the tower; acceptable, visible in cold-load.
11. Pre-existing 901-playbook hazard: the encode's pageable `copy_h2d_async`
    of pixels on the default stream (`patch_embed.rs:140-144`) — harmless today
    because decode graphs are vetoed whenever native EXL3 is on
    (`exl3_graph_veto`, `model/lm_head_exl3.rs:56-60`).

### 4.6 Ordered implementation plan (after `exl3_dense_linear{,_shared_a}` lands)
| step | what | anchors | LoC |
|---|---|---|---|
| 0 | **Bug fix, independent of native**: derive `intermediate_size` from the fc1 weight shape (4352) instead of `vcfg.intermediate_size`; keep the config value for the NVFP4 checkpoint (there fc1 is `[4304,1152]`, so "shape-derived" is correct on both). Validate with the section-5 image A/B. | `qwen35.rs:214-220,261`; `vision_encoder.rs:110`; `init.rs:200`; `vit_block.rs:498-515` | ~20 |
| 1 | Materialize keep-set: `ATLAS_EXL3_NATIVE_VISION=1` (requires `ATLAS_EXL3_NATIVE=1`; mirror `check_exl3_native_dense_gates`, `exl3_materialize_dense.rs:118-141`); `exl3_native_serves_vision(prefix)` = `starts_with("model.visual.")` with leaf ∈ {attn.proj, mlp.linear_fc1, mlp.linear_fc2, merger.linear_fc1, merger.linear_fc2} (+ `attn.{q,k,v}_proj` under `ATLAS_EXL3_NATIVE_VISION_QKV=1` = Option A); wire into `exl3_native_serves_with` and the keep loop as one atomic tower-wide set (all 164 pass `exl3_native_supported`: K=4, Mul1, %128). Same step, always: when the fused `attn.qkv.weight` exists and QKV is not native, remove+free the `attn.{q,k,v}_proj` trellis without writing a `.weight` (−215 MB regardless of native). Update the stale module doc (`exl3_materialize.rs:38-39`) and the test at `:743`. | `exl3_materialize.rs:101-109,421-486,536-540` | ~80 + mock tests |
| 2 | Loader plumbing: `Qwen4ExpWeightLoader` gets `Mutex<Option<Arc<Exl3DenseStage>>>` filled by `load_layers`; `NativeExl3::stage()` folds `max_in/max_out ≥ 4608` when the vision gate is on and `store_has_exl3` sees `model.visual.` trellis; `qwen4_exp::load_vision_encoder` passes the stage into a new `Qwen35WeightLoader::load_vision_encoder_with(store, config, gpu, Option<Arc<Exl3DenseStage>>)`. | `qwen4_exp.rs:135,262,446-458`; `qwen4_exp/exl3_dense.rs:71-100` | ~40 |
| 3 | Encoder types: `ViTBlock.{qkv_w,proj_w,fc1_w,fc2_w}` and `MergerLayer.{fc1_w,fc2_w}` become `enum VitLinear { Bf16(DevicePtr), Exl3(Exl3DenseWeight) }` (+ `VitQkv { Fused(DevicePtr), Split{q,k,v} }`); `VisionEncoder` gains `exl3_stage`; `vision_dense_auto` branches on `is_exl3_linear` → `Exl3DenseWeight::from_exl3(&Exl3Weight::from_store(..))` with a geometry check (4352 for fc1/fc2) and a `probe_kernels` variant covering sh1/sh2/sh3. | `vision_encoder.rs:57-79`; `qwen35.rs:17-31`; `layers/exl3_dense.rs:96` | ~120 |
| 4 | Dispatch: `vit_gemm_bias` takes `&VitLinear`; Exl3 arm = `exl3_dense_linear(.., Exl3DenseOut::contiguous(c), m, stage, stream)` then `k_add_bias`; `vit_qkv_gemm_bias` for Option A via `exl3_dense_linear_shared_a` with three strided outs into the `[M,3456]` arena + one add_bias with the fused bias. patch_embed stays BF16. | `vit_block.rs:19-63,376-386` | ~60 |
| 5 | GPU validation (other workflow's turn): (i) image parity A/B materialized-BF16 vs native via `ATLAS_DUMP_VIT=<dir>` (`enc_impl/utils.rs:33-59`) at the same inter=4352 — expect small K=4-vs-BF16 deltas at proj/fc/merger only, none at qkv under B; (ii) alloc-ledger check that the fused/dead bytes are gone; (iii) an image request concurrent with C≥2 decode + a chunked prefill (section/fence path across default vs prefill streams); (iv) one ≤ 8-patch image (GEMV tier) and one 16384-patch image (8 row batches; sh1/sh2/sh3 all exercised); (v) add `exl3_native_parity` cases at the ViT shapes. | — | — |

Estimate ~350-400 LoC, 1.5-2 days including GPU validation. Ship Option B
first (16-bit attention weights, −723 MB); Option A behind the sub-gate
(−161 MB more at K=4 attention quality). Step 0 should ship immediately and
separately — it is a correctness fix on the current default path.

For 4.05 (separate shard): a `register_exl3_vision_sidecar(model_dir, store, gpu)`
next to `register_exl3_ngram_sidecar` (`exl3_materialize.rs:149`), hooked at
the same place in `serve_load.rs`/`factory/build.rs` BEFORE `materialize_exl3`,
that globs `vision_*.safetensors` and loads it with a no-skip filter
(`loader.rs:163-171` pattern). Its F16 fused qkv / pos_embed / biases are
converted to BF16 by the normal ingest path.

---

## 5. Image test plan for the EXL3 serve

### 5.1 Servers
- **EXL3 under test**: `/home/ms/run_exl3_ep2.sh 0` on gx10-9959 (rank 0, port
  **8890**, model name `qwen4exp-exl3`, snapshot
  `/tank/exl3-ckpt/qwen38-flash-next-2.05bpw`, binary `/home/ms/spark-exl3`)
  and `/home/ms/run_exl3_ep2.sh 1` on dgx-00. CTX default 8192, util 0.6,
  `--kv-cache-dtype bf16`, `--enable-prefix-caching`,
  `--default-chat-template-kwargs '{"reasoning_effort":"low"}'`.
- **Baseline (A/B)**: `/home/ms/run_ep2.sh 0|1` — the NVFP4 qwen4_exp
  checkpoint `/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/129972269565f7f4f664fdf8dd42268d3bbda9fd`
  served by `/home/ms/spark-topk` on port **8889**, model name `qwen4exp`. Its
  ViT is plain BF16 (333 `model.visual.*` tensors, `linear_fc2.weight [1152,4304]`,
  fused `attn.qkv.weight [3456,1152]`), so it is the correct-tower reference.
  Both are EP=2 across the same two nodes; per the one-Atlas-instance rule run
  them one at a time, checking `free -g` first (util 0.76 + 0.6 will not fit
  together).
- Serve with `RUST_LOG=info` (both scripts default to it). For the parity dump
  add `ATLAS_DUMP_VIT=/home/ms/.claude/jobs/5a7bd33d/tmp/vision/dump_{exl3,nvfp4}`
  to the rank-0 environment (`enc_impl/utils.rs:33-59` snapshots BF16 buffers).

### 5.2 Images (no downloads)
- Existing on this box: `/home/ms/bonsai-vision-serve/test_red_circle.png` and
  `test_blue_square.png` (448×448 RGB8, 2.7 KB / 1.6 KB) → 28×28 = 784 patches
  → 196 visual tokens after 2×2 merge.
- Deterministic synthetic PNG with pure stdlib (no Pillow), 448×448, left
  half red, right half blue, black horizontal bar across the middle:

```python
# /home/ms/.claude/jobs/5a7bd33d/tmp/vision/mk_split.py
import zlib, struct
w = h = 448
raw = bytearray()
for y in range(h):
    raw.append(0)                                   # PNG filter byte
    for x in range(w):
        if 200 <= y < 248: raw.extend((0, 0, 0))    # black bar
        elif x < w // 2:   raw.extend((220, 30, 30))  # red left
        else:              raw.extend((30, 60, 220))  # blue right
def chunk(tag, data):
    c = tag + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) \
    + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
open("/home/ms/.claude/jobs/5a7bd33d/tmp/vision/split.png", "wb").write(png)
```

### 5.3 Request (OpenAI chat completions with a `data:` URI; same shape as `scripts/realistic_soak.py:83,167` and `scripts/test-qwen36-tool-image.sh`)

```python
# /home/ms/.claude/jobs/5a7bd33d/tmp/vision/img_req.py  <port> <model> <png>
import base64, json, sys, urllib.request
port, model, png = sys.argv[1], sys.argv[2], sys.argv[3]
url = "data:image/png;base64," + base64.b64encode(open(png, "rb").read()).decode()
body = {
  "model": model, "temperature": 0, "max_tokens": 250, "seed": 1,
  "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"},
  "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": url}},
      {"type": "text", "text": "Describe this image in one sentence: what shapes and colors do you see, and where are they?"}]}]}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
      data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
print(json.dumps(r["choices"][0], indent=1)); print("usage:", r.get("usage"))
```

curl equivalent (payload written by the script above or built inline):
`curl -sS http://127.0.0.1:8890/v1/chat/completions -H 'Content-Type: application/json' -d @req.json`.
Run each request twice (second hit exercises the prefix cache) and write full
output to a file under the scratch dir (never head/tail a long run).

### 5.4 Expected-answer style and pass criteria
- `split.png`: mentions **red on the left**, **blue on the right**, and a
  **black horizontal bar/stripe** across the middle. `test_red_circle.png`:
  "a red circle on a white background"; `test_blue_square.png`: "a blue
  square". `finish_reason: "stop"`, `usage.prompt_tokens` ≈ 196 + text tokens
  (identical between EXL3 and NVFP4 for the same image — a mismatch means the
  preprocessor path differs, not the ViT).
- Baseline (NVFP4, 8889) is expected to pass all three; that is the control
  that the request/preprocessing/token-splice path is right.
- EXL3 today (8890, unpatched) is predicted to FAIL: fluent text describing
  something unrelated, generic ("an abstract pattern"), or hallucinated — the
  fc2 stride bug scrambles every block's MLP. If it passes, the stride analysis
  is wrong and step 0 must be re-examined against `ATLAS_DUMP_VIT` dumps
  before changing anything.
- EXL3 after step 0 (inter derived from fc1 shape = 4352): should match the
  baseline's description semantically (not token-exactly — the tower is K=4
  reconstructed for proj/fc/merger). `ATLAS_DUMP_VIT` per-block dumps vs the
  NVFP4 dumps: cosine similarity of the final merger output should be high
  (> 0.99 expected at K=4); block-0 qkv output should be bit-close (same BF16
  weights, same input).
- Native (step 4) vs step-0 materialized: same criteria, plus the four
  concurrency/shape cases in plan step 5. Compare TTFT (`usage`/timing) at
  M=784 and at a 16384-patch image (a 2048×2048 PNG generated the same way).

### 5.5 What to watch in the log
- Boot: `Qwen3.6 vision encoder loaded: depth=27, hidden=1152, heads=16, FP8-blocks>=4`
  (ignore the FP8 suffix) and the namespace probe's ViT block count
  (`probe.rs:117`, expects 27); the EXL3 materialization line
  `... linears -> BF16 dense, ... kept packed for native serving`
  (`exl3_materialize.rs:541`) — under native vision the kept count rises by 137
  (B) or 164 (A) and the BF16 count falls accordingly.
- No CUDA 700 / no `EXL3 launch section` timeouts while an image request
  overlaps a text request at C=2 (`SEQS=2` in the run script).

---

## 6. Open questions

1. Has any image request been run against the 2.05 serve yet, and was the
   output coherent? The stride analysis predicts garbage; only the section-5
   A/B settles it.
2. Is the fused `attn.qkv.weight` bit-identical to the original Qwen3.8 BF16
   qkv? Strongly implied (bias equal to 1 ulp; the NVFP4 baseline has the same
   BF16 `[3456,1152]` tensor; same 215.2 MB on every branch), but a direct
   byte compare against the NVFP4 checkpoint's tensor (both on this box) would
   close it in minutes.
3. Does ExLlamaV3 itself read the fused copy or the trellis q/k/v at inference,
   and does it glob `vision_*.safetensors` (relevant to 4.05)? Check
   `.research/exllamav3_ref/` before choosing Atlas's discovery rule.
4. Why does 4.05 alone ship `vision_k6.safetensors` + F16 dtypes + no
   `vision_bits` + no mtp patch file (mixer inside shard 9)? Export-tool
   version drift vs a manual re-quant — HF commit history not fetched.
5. Option A vs B for attention fidelity: B keeps 16-bit qkv for 161 MB more;
   the fused copy exists precisely because turboderp kept qkv unquantized.
6. Private ViT `Exl3DenseStage` (~57 MB, decoupled geometry, still sharing the
   section mutex) vs widening the shared stage (+8.4 MB)?
7. Is sh1 (`exl3_gemm_k4_cb2_sh1_f16`) parity-validated at `[1152→*]`? Only a
   `[128→128]` m=17 case exists today.
8. How does the section/fence discipline affect image TTFT at C≥2 when a ViT
   projection alternates with prefill-stream MoE on the device — measure,
   don't model.
9. Should `exl3_native_supported`'s K ∈ {2,4} gate widen for vision only (K=5/6
   on 3.05+, GEMM has those K) — the concurrency argument for the gate (small-row
   GEMV envelope) does not apply to a tower that always has m ≥ 4 patches but
   the shared-locks argument does.
10. `mtp_hyper_connection_mixer_patch.safetensors` is read by nothing in
    `crates/`; on 4.05 the tensors are index-listed instead. Behaviour differs
    by branch if MTP ever needs the mixer.
