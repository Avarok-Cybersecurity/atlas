# cuMemAlloc_v2 granularity on GB10 — measured, and what it costs the EXL3 loader

Measured 2026-09-01 on gx10-9959 (GB10, driver 580.173.02 open kernel module, aarch64, 4 KiB host
pages, 121.63 GiB unified). Harness: `cu_granularity.py` (python3 + ctypes on libcuda.so.1; cuInit /
cuDevicePrimaryCtxRetain / cuMemGetInfo_v2 / cuMemAlloc_v2 / cuMemsetD8 / cuMemFree_v2 — no kernels,
no serve). Raw output: `cu_gran_2000.log` (+ `cu_gran_2000.json`); N=50 smoke run: same numbers.
Projection script: `project_waste.py` → `projection.md` (full per-family tables for 4.05/3.05/2.05).

## TL;DR

- **The waste is real and it is the ~14-18 GB gap.** On 4.05bpw the loader's 296,294 `cuMemAlloc`s
  (ledger site `fast_weights/mod.rs:413`, 63,057.7 MB requested) physically cost **1.28x** the
  requested bytes: the boot log's load-time growth was 79.73 GiB for 61.75 GiB of tensor bytes
  (+17.98 GiB); the model below predicts +17.86 GiB. Per shard: measured 9.75 GiB used per 7.51 GiB
  shard = 1.298x; predicted for a K=4 expert shard = 1.30x.
- **The hypothesis about WHICH tensors is wrong.** The tiny aux tensors (`.suh` 5 KB / `.svh` 1.3 KB /
  `.mul1` 4 B) are sub-allocated at **512 B granularity** and cost almost nothing (63 MiB model-wide,
  all checkpoints). The waste is the **2 MiB chunk tail behind each sub-2 MiB `.trellis` blob**: the
  driver packs allocations < 2 MiB into 2 MiB chunks and never splits an object across a chunk, so a
  K=4 expert trellis (800 KiB) fits 2 per chunk and burns 448 KiB of every chunk (224 KiB per
  tensor); K=3 (600 KiB) fits 3 and burns 248 KiB per chunk; K=2 (400 KiB) fits 5 and burns 48 KiB.
- **Projected saving of pooling, by checkpoint:** 4.05bpw **17.4 GiB**, 3.05bpw **6.8 GiB**, 2.05bpw
  **1.2 GiB**. Pooling only the aux tensors saves 63 MiB — not worth doing on its own.
- **Recommendation:** pool every allocation **< 2 MiB** (the chunk size) into large slabs (64–256 MiB,
  bump-allocated, 256-B aligned sub-offsets), grouped per (shard, load order) — details at the end.
  Slabs ≥ 1 GiB measured 1.006x; 2–19 MiB dedicated allocations measured 1.02–1.03x.

## 1. Measured footprint per request size (N = 2000 allocations per size, sequential)

`footprint` = cuMemGetInfo free-memory delta over the whole batch (touched with cuMemsetD8; the
alloc-time and post-touch numbers were identical in every row — memory is committed at cuMemAlloc,
not lazily). `VA stride` = sorted-pointer differences (the driver's sub-allocation layout).

| request | N | requested | footprint | per-alloc | **ratio** | VA stride (top) | us/alloc |
|---|---|---|---|---|---|---|---|
| 4 B | 2000 | 7.8 KiB | 1.75 MiB | 918 B | 229x (one 2 MiB chunk) | 512 B ×1999 | 7.6 |
| 1.25 KiB (svh) | 2000 | 2.44 MiB | 3.50 MiB | 1.8 KiB | 1.43x | 1.5 KiB ×1998 | 7.2 |
| 5 KiB (suh) | 2000 | 9.77 MiB | 9.62 MiB | 4.9 KiB | 0.99x | 5.0 KiB ×1995 | 7.0 |
| 32 KiB | 2000 | 62.5 MiB | 63.4 MiB | 32.5 KiB | 1.01x | 32 KiB ×1999 | 8.5 |
| 64 KiB | 2000 | 125.0 MiB | 127.1 MiB | 65.1 KiB | 1.02x | 64 KiB ×1999 | 10.2 |
| 213 KiB | 2000 | 416.0 MiB | 453.5 MiB | 232.2 KiB | 1.09x | 213 KiB ×1777, 344 KiB ×221 | 18.9 |
| 400 KiB (K=2 trellis) | 2000 | 781.3 MiB | 822.6 MiB | 421.2 KiB | **1.05x** | 400 KiB ×1600, 448 KiB ×397 | 28.8 |
| 600 KiB (K=3 trellis) | 2000 | 1.14 GiB | 1.34 GiB | 703.1 KiB | **1.17x** | 600 KiB ×1333, 848 KiB ×664 | 43.4 |
| 800 KiB (K=4 trellis) | 2000 | 1.53 GiB | 2.01 GiB | 1.03 MiB | **1.32x** | 800 KiB ×1000, 1.22 MiB ×997 | 61.9 |
| 1.17 MiB (K=6 shared_expert) | 2000 | 2.29 GiB | 4.03 GiB | 2.06 MiB | **1.76x** | 2.00 MiB ×1997 | 116.7 |
| 2 MiB | 2000 | 3.91 GiB | 4.02 GiB | 2.06 MiB | 1.03x | 2 MiB ×1997 | 114.6 |
| 2 MiB + 4 KiB | 2000 | 3.91 GiB | 4.17 GiB | 2.14 MiB | 1.07x | 4 MiB ×1997 (VA only) | 129.8 |
| 3 MiB | 2000 | 5.86 GiB | 6.01 GiB | 3.08 MiB | 1.03x | 4 MiB ×1997 (VA only) | 160.1 |
| 8 MiB | 1024 | 8.00 GiB | 8.18 GiB | 8.18 MiB | 1.02x | 8 MiB ×1021 | 346.3 |
| 18.75 MiB (K=6 in_proj_qkv) | 436 | 7.98 GiB | 8.16 GiB | 19.16 MiB | 1.02x | 20 MiB ×418 | 792.7 |

Cross-checks in the log: host `/proc/meminfo` MemFree and MemAvailable deltas equal the cuMemGetInfo
delta in every row (**on GB10 `cuMemGetInfo` free IS host MemFree** — 78.57 GiB both at start), and
the residual after freeing everything is ≤ ±10 MiB (noise; the "1.75 / 2.19 MiB" step sizes in the
raw log are the same 2 MiB chunk read through a jittery MemFree).

## 2. The allocator model this fits

1. **Requests < 2 MiB** are carved from **2 MiB chunks** with **512 B rounding** (4 B → 512 B stride;
   1280 B → 1536 B; 5120 B stays 5120). An object never straddles a chunk boundary, so the
   footprint of a size `s` is `2 MiB / floor(2 MiB / roundup(s, 512))` (+ ~2% chunk overhead, seen
   on the exact-fit 32/64 KiB rows). The stride pattern proves it: 800 KiB objects sit at 800 KiB
   then 1.22 MiB (= 2048 − 800 KiB) alternately — two per chunk, tail unused.
2. **Requests ≥ 2 MiB** get a dedicated allocation. VA is reserved in 2 MiB multiples (3 MiB → 4 MiB
   stride) but the *physical* footprint is `~1.02 × s` (2 MiB + 4 KiB → 2.14 MiB, i.e. page-granular,
   not 2 MiB-granular). One 1.5 GiB slab measured 1.006x, so the ~2% is a per-allocation cost that
   amortises away at slab sizes.
3. The **effective granularity is therefore two-tier**: 512 B for anything that packs many-per-chunk,
   and *"2 MiB divided by how many fit"* — 1.03–1.76x — for objects between ~200 KiB and 2 MiB. Every
   EXL3 expert trellis for this model family (409,600 / 614,400 / 819,200 B) lands in the bad tier.

Model vs measurement (from `projection.md`): 400 KiB 1.044 vs 1.05; 600 KiB 1.161 vs 1.17; 800 KiB
1.306 vs 1.32; 1.17 MiB 1.741 vs 1.76; 213 KiB 1.090 vs 1.09.

## 3. The real loader pattern (trellis, suh, svh, mul1 per projection), M = 2000 projections

| K | trellis | requested/proj | footprint/proj | **waste/proj** | ratio | us/proj (4 allocs) |
|---|---|---|---|---|---|---|
| 2 (2.05bpw experts) | 400 KiB | 406.3 KiB | 422.7 KiB | 16.5 KiB | 1.041x | 30.4 |
| 3 (3.05bpw experts) | 600 KiB | 606.3 KiB | 703.4 KiB | 97.2 KiB | 1.160x | 45.1 |
| 4 (4.05bpw experts) | 800 KiB | 806.3 KiB | 1054.7 KiB | **248.3 KiB** | **1.308x** | 64.1 |

Controls (same M, same tensors):

| variant | K=2 | K=3 | K=4 |
|---|---|---|---|
| all 4 tensors in ONE slab per 2000 projections, 256-B sub-offsets | 416.6 KiB/proj (1.025x) | 610.7 KiB/proj (1.007x) | 810.9 KiB/proj (1.006x) |
| trellis own alloc + suh/svh/mul1 pooled per projection ("aux-only") | 419.4 KiB (1.032x) | 702.6 KiB (1.159x) | 1054 KiB (1.307x) |

Aux-only pooling moves the ratio by 0.1%: the aux tensors were already 512-B packed. The trellis
blobs are the entire effect.

**Validation against the 4.05 boot** (`kladder/boot-405.log`): 296,294 loader allocs, 63,057.7 MB
requested (ledger); cuMemGetInfo "used" grew 79.73 GiB for 61.75 GiB of on-disk bytes → excess
17.98 GiB. Model: 73,728 routed + 1,536 mtp expert projections × 248.3 KiB = 18.2 GiB of chunk-tail
waste on the experts, less the model's non-expert share ≈ **17.86 GiB total** (§4). Per shard:
measured +9.75 GiB per 7.51 GiB shard (1.298x) vs 1.30x predicted for a shard that is ~95% K=4
experts. Time: ~64 us per projection × 75K ≈ **5 s of the 37 s load is spent inside cuMemAlloc**
(plus the matching cuMemFree storm at teardown).

## 4. Projected waste per checkpoint (full per-family tables in `projection.md`)

Populations from `.research/ckpt_meta/k_table_{2.05,3.05,4.05}bpw.md`: 73,728 routed + 1,536 mtp
expert projections at expert K, indexer at expert K, GDN/attention/shared_expert/lm_head/vision at
head K, `mtp.fc_*` at its own K, plus every non-trellis 16-bit tensor (embed, hc mixers, gates,
norms, biases). Model of §2 applied per tensor.

| checkpoint | requested (resident, no ngram) | modelled footprint | **waste** | of which experts (routed+mtp) | of which other <2 MiB | of which ≥2 MiB (~2% dedicated-alloc cost) |
|---|---|---|---|---|---|---|
| **4.05bpw** (experts K=4) | 63.53 GiB | 81.39 GiB (1.281x) | **17.86 GiB** | 17.61 GiB | 0.14 GiB (shared_expert K=6 blobs 128 MiB, vision attn 8 MiB, k/v_proj 2 MiB, small F16 ~3 MiB) | 0.11 GiB |
| **3.05bpw** (experts K=3) | 48.70 GiB | 55.82 GiB (1.146x) | **7.12 GiB** | 6.98 GiB | 0.04 GiB (vision attn K=5 25 MiB, k/v_proj 6 MiB, shared 6.5 MiB) | 0.10 GiB |
| **2.05bpw** (experts K=2) | 33.87 GiB | 35.34 GiB (1.044x) | **1.47 GiB** | 1.34 GiB | 0.04 GiB (shared_expert K=4 35 MiB, vision 5 MiB) | 0.09 GiB |

Note on the 2.05 "excluded" number: the previous stream saw ~15 GB excluded on 2.05 (≈6 GB after
the co-tenant + driver context). Granularity explains only ~1.5 GiB of that. Because `cuMemGetInfo`
free == MemFree here, **page cache counts as "used"** in `used_so_far` (`factory/build.rs:454`) —
the shard prefetch/readahead the fast loader requests (`NFS/shard prefetch requested`) is the likely
owner of the remainder. Pooling will not recover it; it is reclaimable cache, and it also depresses
the `.min(actual_free − reserve)` clamp. Worth a follow-up (drop-behind / `posix_fadvise(DONTNEED)`
after each shard, or reading MemAvailable instead of MemFree for the clamp), separate from pooling.

Also note: on the 4.05 native boot the 147 shared_expert projections (K=6, 1.17 MiB trellis, the
1.76x row) are *materialized* to NVFP4 in the first pass, so their trellis waste is transient and
replaced by 144 × 800 KiB NVFP4 weight blobs at 1.32x (~35 MiB). The 4.05 total above counts them
as kept-packed (128 MiB); either way they are noise next to the experts.

## 5. Projected saving of the three pooling scopes (slab footprint 1.006x, measured)

| scope | tensors moved | 4.05bpw | 3.05bpw | 2.05bpw |
|---|---|---|---|---|
| (a) only the aux tensors (suh/svh/mul1) of every trellis linear | 227,253 (465 MiB) | **63 MiB** | **63 MiB** | **63 MiB** |
| (b) all EXL3 tensors (trellis+aux) of kept-packed prefixes (experts, mtp experts, indexer, GDN, attn, shared_expert, lm_head, mtp fc) | 302,348 | **17.43 GiB** | **6.76 GiB** | **1.22 GiB** |
| (b′) same as (b) but only the tensors < 2 MiB | 302,211 | 17.39 GiB | 6.73 GiB | 1.20 GiB |
| (c) every tensor < 1 MiB model-wide | ~303,500 | 17.28 GiB | 6.76 GiB | 1.21 GiB |
| (c′) every tensor < 2 MiB model-wide (= the chunk size) | ~303,600 | **17.40 GiB** | **6.76 GiB** | **1.21 GiB** |

(c) < (b′) at 4.05 only because a 1 MiB threshold misses the 1.17 MiB shared_expert blobs (the worst
ratio in the model, 1.76x) while gaining ~4 MiB on small F16 tensors; a 2 MiB threshold gets both.
Pooling the ≥ 2 MiB tensors as well (b vs b′) buys 20–40 MiB — not worth the extra slab traffic or
the loss of per-tensor free for the big dense blobs.

## 6. Recommendation

**Threshold: pool every weight allocation with `len < 2 MiB`** (exactly the driver's chunk size —
anything smaller pays the chunk-tail tax, anything ≥ 2 MiB gets a dedicated allocation at ~1.02x
and is fine as it is). Do it in the loader (`fast_weights/mod.rs:413`) as a size test, not a
name/prefix test: that is what the driver keys on, it covers experts at every K plus the K=6 dense
stragglers, and it needs no knowledge of which prefixes the materialize pass will keep packed.

**Grouping / slab shape:**
- Bump allocator over **slabs of 64–256 MiB** (256 MiB → 0.5% of a shard; measured overhead at GiB
  scale is 0.6%, at 8–19 MiB it is 2%, so do not go below ~64 MiB). One live slab per copier thread
  (the copier is single-threaded today, so one), cut when the next tensor does not fit; the tail of a
  cut slab is < 2 MiB of waste per slab — with 256 MiB slabs that is < 1% worst case (~240 slabs for
  the 4.05 experts → ≤ 0.5 GiB worst case, typically far less since aux tensors backfill).
- **Sub-offset alignment 256 B** (what `cuMemAlloc` documents; the EXL3 kernels take 16-B vector
  loads on the trellis tiles and f16 vectors). Cost: ≤ 4 × 255 B per projection ≈ 36 MiB model-wide
  at 4.05. 512 B — what the driver actually hands out today — would double that; only choose it if
  a kernel is found to assume 512-B alignment (none does by contract).
- Group **in load order**, i.e. per shard: the safetensors index is sorted, so the four tensors of a
  projection and the 512 experts of a layer land adjacent, which is also good for the per-expert
  pointer tables (`layers/moe/ptr_table_build.rs` reads raw addresses, so views into a slab work
  unchanged) and for TLB locality on the grouped-GEMM reads.
- Alternative that keeps `WeightStore` semantics simplest: one slab per **(layer, expert family)**
  = 512 × (800 + 6.4) KiB ≈ 403 MiB at K=4 (≈ 302 MiB K=3, 201 MiB K=2). Naturally aligned to the
  unit the MoE layer owns, and a materialize-pass "convert this layer's experts" frees exactly one
  slab. Slightly more code in the loader (needs the expert-family key while copying); same saving.

**Ownership / free semantics that must change** (the invariant `weights.rs:467-474` states today —
"no loader inserts an `.offset()` view of a shared block into this map" — is deliberately broken):
- `WeightTensor` entries that live in a slab must not be `gpu.free`d individually.
  `WeightStore::release` (`weights.rs:480`) should free the slab list and skip slab-resident
  entries; give the store a `slabs: Vec<DevicePtr>` (or a `SlabId` per tensor / an address-range
  check) so `release` stays correct and idempotent.
- `exl3_materialize.rs:336-340` frees `.trellis/.suh/.svh/.mul1` after materializing a linear — for
  slab-resident tensors that free must become a no-op (dead bytes inside the slab) or a per-slab
  refcount decrement. Bounded cost: on the 4.05 native boot only 144 shared_expert projections +
  the vision/mtp dense linears are materialized, so ≤ ~0.5 GiB of dead slab bytes even with
  load-order slabs; with per-(layer, family) slabs it is exactly the freed layer.
  Grep confirms these are the only `gpu.free` of store tensors in the materialize pass
  (`exl3_materialize.rs:296` frees a temporary BF16, not a store entry).
- The alloc ledger (`cuda_backend/alloc_ledger.rs`) records requested bytes per `cuMemAlloc`, so
  after pooling `live_bytes()` and the physical footprint finally agree (today ledger 67.9 GB vs
  ~86 GB physical on 4.05); the "excluded" number in `factory/build.rs:509-515` will drop by the
  saving, and the `Fast-load pre-flight … 1.3x overhead` estimate (`weights/loader.rs:67-118`) —
  which happens to be right *because* of this waste at K=4 — can be tightened to ~1.03x for pooled
  loads.

**Side benefits:** ~5 s off the 4.05 load (75K × 4 → ~240 slab allocs), the same off teardown, and
the KV pool on a single GB10 gains the full 17 GiB at 4.05 (boot-405 had 31.0 GB actual free after
weights against a 7.3 GB reserve; pooled that becomes ~48 GB — roughly 1.7× the KV budget), 6.8 GiB at 3.05, 1.2 GiB at 2.05.

## Files

- `/home/ms/.claude/jobs/5a7bd33d/tmp/pool/cu_granularity.py` — the harness (also at
  `gx10-9959:/home/ms/cu_granularity.py`)
- `/home/ms/.claude/jobs/5a7bd33d/tmp/pool/cu_gran_2000.log`, `cu_gran_2000.json` — raw N=2000 run
- `/home/ms/.claude/jobs/5a7bd33d/tmp/pool/project_waste.py` → `projection.md` — per-family tables
