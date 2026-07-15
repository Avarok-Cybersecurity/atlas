# Atlas LoRA v0 — Implementation Status (M0 + M1-attention)

Implements the [MVP proposal](lora-mvp-proposal.md) / [codebase brief](lora-codebase-brief.md).
This is a **working POC**: a served, fine-tuned tiny model on GB10, verified end-to-end.

## What ships

**Serve a single PEFT LoRA adapter, loaded at startup, applied to every request as a
runtime BF16 delta** (`y += scale·(x@Aᵀ)@Bᵀ`) — never merged into the (NVFP4) base
weights. Zero new CUDA kernels; the deltas reuse `dense_gemv_bf16` / `dense_gemm_tc` /
`bf16_scaled_add` and are captured inside the existing CUDA decode graphs.

```
spark serve Hcompany/Holo-3.1-0.8B \
  --lora-adapter my-ft=/path/to/peft-adapter-dir \
  --max-lora-rank 64
```

### M0 — load, validate, account
- `--lora-adapter NAME=PATH_OR_HF_ID`, `--max-lora-rank` (64), `--max-loras` (8).
  Repeated flag → named reject (multi-adapter is M2).
- PEFT `adapter_config.json` parser (`atlas-core`), hard-fail with `REJECT(...)` reasons
  for every unsupported feature: non-LORA `peft_type`, DoRA, bias, rank/alpha patterns,
  `modules_to_save`, `all-linear`, absent `use_rslora`, `r=0`.
- Dedicated adapter safetensors loader (`spark-runtime`) — host F16→BF16 (the base
  `WeightDtype` whitelist rejects F16), header-only OOM preflight, named pickle reject.
- Key remap + per-`LayerType` allow-list (`spark-model`): only the **full-attention**
  layers × {k,v,o,gate,up,down} are accepted. Named hard rejections (never silent skips):
  `gated-q-proj` (holo's `attn_output_gate` interleaves Q+gate), `gdn-target`,
  `non-full-attention-layer`, plus a bidirectional tensor↔target audit and A/B shape audit.
- A/B packed **rank-padded to `max_lora_rank`** in one fixed-address pool with per-module
  `[max_loras]` device pointer tables (the frozen M2 layout contract; v0 fills slot 0).
- Pool VRAM is allocated before the KV-budget snapshot, so it is **budgeted against the KV
  cache** (GB10 unified-memory OOM = freeze).
- Scaling `alpha/r` (or `alpha/√r` under rsLoRA), read per adapter, never defaulted.

### M1 — runtime delta at the attention insertion points
- `apply_lora_delta` wired at **prefill k/v/o** and **decode k/v/o** (decode k/v applied
  before norm/RoPE/kv-cache-write, so the KV cache stores the adapted values, matching
  HF `k_norm(k_proj(x)+Δ)`).
- Deltas contract at the **padded** `max_rank` (B is packed with `max_rank` row stride);
  a real-rank contraction would misread every B row past the first when `r < max_rank`.
- `ATLAS_LORA_EAGER=1` disables decode-graph capture (debugging hatch); deltas are
  otherwise graph-safe (pool weights, arena scratch, and the f32 scale are load-time-fixed).
- `/v1/models` advertises the adapter name first; base-name requests get adapted output
  with a one-line warn (v0 always-on wart; per-request routing is M2).

## Verification (holo-3.1-0.8b on GB10, CUDA 13)

- **Offline parity oracle** (`scripts/reference_deltas.py`): the loaded A/B/scale reproduce
  the PEFT-exported reference deltas — 36/36 modules, `scaling=2.0`, **0.0 rel-err**.
- **`atlas-core` parser**: 11 unit tests (accept + every named reject) green.
- **Served, live**:
  - startup logs `LoRA adapter 'holo-tiny' installed on 6 layers (pool=117.0 MiB)`;
  - base output `"Paris."` → adapter output **differs** (delta is applied in the live
    decode path);
  - **graph == eager** (`ATLAS_LORA_EAGER=1`) — byte-identical, so deltas capture
    correctly inside CUDA graphs.

The test fixture (`test_data/lora-holo-tiny/`) is a **generated** PEFT adapter, deliberately
strong so its effect is unambiguous — no community adapter exists for Atlas's custom
NVFP4-packed bases, so a controllable fixture also exercises the reject/parity paths exactly.

## M2 — multi-adapter rotation over the RDMA weight tier (staged on this branch)

**Load MULTIPLE adapters and HOT-SWAP the active one at runtime**, plus RDMA
slot-staging so a rotation set larger than the resident slots can swap in from a
peer's RAM. Every new path is a **no-op when its flag/env is unset** — a single
startup adapter with no rotation env is byte-identical to M1.

- **Multi-adapter pack** (`lora/mod.rs`): `--lora-adapter` (already repeatable)
  now packs slots `0..N-1` via `load_lora_adapters_multi`. The per-adapter audit
  (classify / shape / target / `r<=max_rank`) and the frozen intra-slot layout
  (A contiguous into `[max_rank,in]`, B row-repacked `r→max_rank`) are UNCHANGED;
  slot `k` is the same walk at `off = k*pool_slot_bytes`. The `[max_loras]`
  pointer tables are now filled index-`k`-per-slot in a post-pass (still dormant).
  Per-adapter scale from each adapter's own `peft.scaling()`. Bails if
  `#adapters > --max-loras`. `load_lora_adapters_generic` stays as the
  single-adapter wrapper (slot 0, byte-identical).
- **Runtime rotation (eager-on-rotate)** (`impl_b3.rs`, `decode_a{,2}.rs`): a
  new `TransformerModel::lora_rotatable` (true when `slots>1` /
  `ATLAS_LORA_ROTATE=1` / `$ATLAS_LORA_PEER`) folds into the existing
  `lora_eager` gate, so an armed rotation runs decode **eager** — a
  `set_active_lora(name)` re-point is immediately live, never a stale graph
  replay. `Model::set_active_lora` (trait default: unsupported) re-installs the
  named slot's pairs and clears the decode-graph caches defensively. Rotation is
  applied by the scheduler at a **quiescent point** (no in-flight decode).
- **Control plane**: `POST /v1/lora/active {"adapter":"NAME"}` → a dedicated
  scheduler rotation channel (`scheduler::LoraRotation`, kept OUT of the sequence
  queue) → applied when `active`/`prefilling`/new-requests are all empty. All
  resident adapters are advertised by `/v1/models` (slot order, `data[0]` =
  default route).
- **RDMA slot-staging** (`spark-storage/weight_lora_rdma.rs` +
  `spark-model/lora/rdma_stage.rs`, gated `$ATLAS_LORA_PEER`): stage adapter dirs
  on `atlas-weight-peer` and RDMA-load a named adapter's A/B straight into a pool
  SLOT — landing byte-identical to the disk pack (same F16/F32→BF16 host convert
  as the disk adapter loader, same B row-repack). `TransformerModel::
  swap_lora_slot_from_peer` re-zeroes the slot, lands via
  `RdmaLoraLoader::stage_into_slot`, rebuilds the slot's pairs with the new
  r/scale, and re-installs if active. Peer fixes: `resolve_shards` now matches
  `adapter_model.safetensors`; `validate_dtype` now accepts `F16`.
- **Tests** (CPU, no GPU): slot-offset math (`slot k base == k*slot_bytes`;
  `module_slot_offsets` reproduces the pack walk and fills exactly one slot;
  non-full-attn → `None`), name→slot resolve, F32/F16/BF16→BF16 convert (matches
  `half::bf16::from_f32`), B repack pad-zeroing, land-target slot addressing, and
  slot-layer rebuild rank/pointers.

### M2 cut lines (honest)
- Batch-1 still holds: rotation is applied between batches (a rotation waits
  while decode is active), and multi-adapter does NOT imply batched multi-adapter
  BGMV (the dormant tables' direction). No per-request adapter auto-routing yet —
  `/v1/lora/active` sets ONE global active adapter; request `model` still routes
  by name for advertise, not per-request delta selection.
- The RDMA verbs data path is gated on `atlas_rdma_verbs` (rdma-core); without it
  `stage_into_slot` bails clearly. The pure convert/repack/offset logic is
  un-gated and unit-tested.

## Deferred (documented cut lines)
- **Dense-FFN delta** (gate/up/down): types + install are in place; the compute insertion in
  `dense_ffn.rs` is not wired. The fixture targets FFN too, so enabling it is additive.
- **Prefix-cache warm-hit path** (`cache_skip_qkv.rs`) and **multi-seq decode** — until
  wired, cache-hit prefills and concurrency ≥2 silently skip the deltas. **Run the POC at
  batch size 1.**
- **q_proj / GDN / MoE / MLA targets**, **per-request adapter routing + multi-adapter**
  (M2), **`lora_bgmv` kernel**, **TP>1** (startup-guarded to `world_size=1`).

## Base-branch build note
`research/lora` predates the switch to the `cu*` driver API, so `cuda_backend`'s
`cudaMemcpy2DAsync` needs `libcudart` linked; `spark-runtime/build.rs` now does so and adds
the CUDA-13 SBSA lib path. On `main` (driver-API) this is a harmless no-op.

## M2 per-request routing — status correction (2026-07-06, after commit d3ea611)

The WIP commit d3ea611 labelled the batched bgmv apply "BUGGY / DO NOT ENABLE".
Live re-test on holo-0.8b **corrects that**: the bgmv decode routing is **not**
buggy. The earlier "one-of-two-concurrent-garbled" was two known effects, not a
kernel defect:

1. **Scheduler.** Under the default `--scheduling-policy fifo`, two concurrent
   requests serialized and never decode-batched, so each fell to the single-seq
   path (= the global *active* adapter). Under `--scheduling-policy slai` they
   co-decode and the bgmv routes each sequence to its own adapter. The
   prefill+decode-consistent request (`starfall`) comes out **byte-clean**
   (`STARFALL-7725`, identical to the single-seq path) — the bgmv decode is correct.
2. **Prefill cut line.** A routed request still **prefills with the active adapter**
   (prefill doesn't route yet), so its prompt KV is active-flavored — the persona
   survives in decode but the exact codeword degrades. Fixed by the request-scoped
   selector applied to prefill (task: "Request-scoped AdapterSelector").

Net: bgmv decode routing works; the real remaining work is request-scoped routing
(esp. prefill) + adapter-correct KV + reliable batch>1 — NOT the kernel. Keep the
WIP disabled in prod until those land, but it is a correctness *sequencing* gap,
not a broken kernel.
