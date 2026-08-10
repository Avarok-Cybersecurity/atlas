# LoRA on the Atlas main decoder: embed/vocab overlay + MoE expert deltas — integrated implementation plan

**Date:** 2026-07-16
**Branch:** `feat/lora-moe-embed` (worktree `ternary-bonsai`)
**Scope:** two PEFT-adapter capabilities on the `TransformerModel` decoder path (`holo3_1_moe` / `qwen3_6_moe`, and the dense siblings where applicable):

- **Feature 2 — Token overlay:** `trainable_tokens` / `modules_to_save[embed_tokens|lm_head]` vocab-extension and embed/lm_head row replacement. **Ships first** (tractable, verified on-disk fixture, conceptually mirrors the NLLB token-overlay that lives on `feat/nllb-trainable-tokens`).
- **Feature 1 — MoE expert + router LoRA:** per-expert `mlp.experts.N.{gate,up,down}_proj` and router `mlp.gate` deltas. **Correctness-first**, with a clearly-marked fused/S-LoRA follow-up.

> The NLLB template files named across the findings (`token_adapter.rs`, `token_overlay.rs`, `nllb_*_overlay` kernels) are **not on this branch**. This is a from-scratch port; NLLB is a conceptual reference only.

---

## A. Reconciliation & ordering

Both features touch the same four choke files: `atlas-core/src/config/parsers/lora.rs`, `spark-model/src/lora/loading.rs` (`audit_adapter`), `spark-model/src/lora/types.rs`, and `spark-model/src/lora/key.rs`. Uncoordinated they will merge-conflict and double-edit the accept/reject ladder. The plan therefore front-loads a **single shared-foundation step** (§C) that lands with Feature 2's first PR (it is pure and ships first), after which each feature only edits its own new modules plus a small, disjoint set of already-reserved insertion points.

**Ordering:**

1. **Shared foundation (S)** — parser config-struct extension covering *both* features' fields + reject relaxations, and the `audit_adapter` interception hook point. Pure Rust, `cargo check`-able, zero runtime behaviour change.
2. **Feature 2 (F2-A/B/C)** — token overlay: parser accept already done in S; add overlay module, kernels, forward hooks. Turns the feature on behind presence-of-overlay-tensors, byte-identical for non-overlay adapters.
3. **Feature 1 (F1-P1)** — MoE expert/router LoRA, correctness-first, **no new CUDA**, single active adapter, reuses `apply_lora_delta`. Includes the `classify_key → LoraTarget` refactor (the one shared file F2 deliberately left untouched).
4. **Feature 1 follow-ups (F1-P2/P3)** — S-LoRA expert routing (grouped BGMV kernel) and fused on-disk import + fused-epilogue kernel. Deferred, interfaces pinned now for forward-compat.

**Key reconciliation decisions:**

- **F2 does not touch `key.rs`.** Overlay tensors (`…token_adapter.*`, bare `modules_to_save` `.weight`) are intercepted in `audit_adapter` *above* the `classify_key` call and `continue`d. This keeps F2 orthogonal to F1's `classify_key` return-type refactor.
- **F1 refactors `classify_key`'s return to `LoraTarget`.** The refactor is written to preserve F2's interception `continue` (which sits above the `classify_key` call), so the two never collide on that line.
- **The parser is edited once, in step S.** All new `PeftAdapterConfig` fields for both features, plus every reject relaxation, land together so the accept/reject ladder is touched exactly once.

---

## B. Model constants (authoritative, Holo-3.1-35B-A3B `index.json`)

`L = 40` layers (all MoE, all `LayerType::FullAttention` with MoE FFN), `E = 256` experts, `top_k = 8`, `hidden h = 2048`, `moe_intermediate_size inter = 512` (use `cfg.moe_intermediate_size_for(layer)`, **never** the dense `intermediate_size = 5120`), `vocab ≈ 256k`. Base experts are stored as **fused 3-D params** `mlp.experts.gate_up_proj [E,2·inter,h]` / `down_proj [E,h,inter]`; router `mlp.gate.weight [E,h]`; embed `model.…embed_tokens.weight [vocab,h]` BF16 (always BF16 at runtime via `dense_auto` dequant).

---

## C. Shared foundation (step S — lands in F2's first PR)

### S.1 `atlas-core/src/config/parsers/lora.rs` (~266 → ~340 LoC)

`PeftAdapterConfig` gains:
```rust
pub trainable_token_indices: Vec<u32>,  // flattened+deduped; empty = none  (F2)
pub modules_to_save: Vec<String>,       // accepted subset {embed_tokens, lm_head}  (F2)
pub lora_embedding: bool,               // low-rank embed LoRA present — Tier-2 gate  (F2)
```
`RawPeftAdapterConfig` gains `#[serde(default)] trainable_token_indices: Option<serde_json::Value>` (list **or** `{"embed_tokens":[…],"lm_head":[…]}` dict — parse both, union) and keeps existing `modules_to_save`. Add `#[serde(default)] target_parameters: Option<Vec<String>>` (F1).

Reject-ladder edits (order preserved):
- **`modules_to_save` (lines 150–155):** partition by leaf. `{embed_tokens, lm_head}` → store in `PeftAdapterConfig.modules_to_save`; anything else keeps `REJECT(modules_to_save)`.
- **`parse_trainable_tokens`** (new helper, mirrors `parse_layers_to_transform`): null/list/dict → `Vec<u32>` deduped. Absent ⇒ empty ⇒ no overlay (backward compatible).
- **empty `target_modules`:** allow when `!trainable_token_indices.is_empty() || !modules_to_save.is_empty()` (pure-overlay adapters target no LoRA module); otherwise keep the existing empty-list bail.
- **`target_parameters` (F1):** `Some(non-empty)` → `bail!("REJECT(target_parameters): fused expert LoRA deferred to F1-P3")`. Never silently ignore (no `deny_unknown_fields` today means it would otherwise be dropped).
- **`validate_target_module` (lines 242–262):** add `"gate"` to `PEFT_SUPPORTED_TARGET_MODULES` + explicit arm (router LoRA, leaf `gate`, distinct from `gate_proj`). Keep `embed_tokens|lm_head => REJECT(embedding)` for `target_modules` entries (legit only via `modules_to_save`); do not relax unless Tier-2 `lora_embedding` is implemented.
- `scaling()` unchanged — overlays and experts/router all inherit the per-adapter scale.

Keep the runtime mirror `lora/env.rs::validate_peft_config` in sync: add `"gate"` to the peft-name check; add `--max-lora-expert-rank` (default 16) and `--lora-experts=on|off` (default off) here for F1 (validated but unused until F1-P1).

### S.2 `spark-model/src/lora/loading.rs` — `audit_adapter` interception hook (~+10 LoC)

Establish the hook point (F2 fills the body, F1's refactor slots below it):
```rust
// F2: overlay tensors never reach classify_key.
if let Some((tensor, ekind)) = classify_overlay_key(name) {
    overlay_tensors.entry(ekind).or_default().set(tensor, name);
    continue;
}
let (layer, target, ab) = classify_key(name, cfg)?;   // F1 changes tuple → LoraTarget
```
Return `overlay_tensors` alongside `found`. Guard the `found.is_empty()` bail (line ~72): fatal only if `overlay_tensors` is *also* empty. Keep this edit ≤15 lines (loading.rs is 478 → budget-tight); delegate the collection struct to `lora/overlay.rs`.

### S.3 Test scaffolding

`lora_tests.rs`: update `dora_bias_rank_pattern_rejected_named` (currently asserts `modules_to_save:[lm_head]` rejects → now **accepts**); add `trainable_token_indices` list+dict parse, empty-`target_modules`-with-overlay accept, `modules_to_save:[q_proj]` still rejects, `target_parameters` present → named reject, `router_gate_accepted`.

---

## D. Feature 2 — token overlay (embed / lm_head / vocab-extension)

**Three on-disk mechanisms, two runtime paths:**

| Mechanism | On-disk (verified `/tank/kuku-v24.3`) | Runtime |
|---|---|---|
| `trainable_tokens` (primary) | `…token_adapter.base_layer.weight [vocab,h]` bf16 + `…token_adapter.trainable_tokens_delta [T,h]` f32; config `trainable_token_indices` | sparse `EmbedOverlay` (row-diff → compact rows) |
| `modules_to_save[embed_tokens\|lm_head]` | full `…embed_tokens.weight`/`…lm_head.weight [vocab,h]` bf16 | same builder, `delta=∅`, row-diff finds changed rows |
| `lora_embedding_A/B` | `[r,vocab]`+`[h,r]` | **Tier-2**: parse-accept, `REJECT[lora-embedding-unimplemented]` at load until kernel lands |

**Correctness constraint (departs from NLLB):** the decoder batches multiple sequences, each with its own adapter, row-routed by `seq_slot[i]` (`<0`=base). A global-active overlay would corrupt base rows in a mixed batch. **The overlay routes per-token by the existing `seq_slot` buffer**, exactly like the attention BGMV. No new `AtomicBool`.

### D.1 New files

| Path | ~LoC | Contents |
|---|---|---|
| `spark-model/src/lora/overlay.rs` | 300 | `EmbedOverlay`, `LmHeadOverlay`, `OverlayTensor`, `classify_overlay_key`, `build_overlay`, `clamp_trainable_to_vocab`, pure helpers (`build_override_set`, `override_source`) |
| `spark-model/src/lora/overlay_tests.rs` | 180 | host-only unit tests (port the 14 NLLB tests) |
| `spark-model/src/lora/overlay_tables.rs` | 150 | per-slot device pointer-table build/refresh (`[max_loras]` u64/u32), mirrors `refresh_slot_tables` |
| `spark-model/src/layers/ops/token_overlay.rs` | 140 | `ops::embed_rowdiff`, `ops::embed_overlay_routed`, `ops::lmhead_overlay_routed` launchers |
| `spark-model/src/model/token_overlay.rs` | 180 | `impl TransformerModel { apply_embed_overlay, apply_lmhead_overlay, overlay_active }` |
| `kernels/gb10/common/token_overlay.cu` | 180 | `embed_rowdiff_bf16`, `embed_overlay_routed_bf16`, `lmhead_overlay_routed_bf16`, `lmhead_overlay_routed_f32` |

### D.2 Changed files (all stay <500 LoC)

`lora/loading.rs` (build set after pack, ≤15 LoC via delegation) · `lora/types.rs` (+`LoraWeights.overlays: Option<TokenOverlaySet>` + accessor) · `lora/mod.rs` (`mod overlay; mod overlay_tables;`) · `model/types.rs` (+`overlay_kernels: OverlayKernels`) · `model/impl_a3.rs` (2 hook calls) · `model/impl_lora.rs` (resolve 4 kernel handles in `set_lora_weights`) · `model/trait_impl/prefill_b/embed_chunk.rs` (1 hook call after `batched_embed`) · `model/mod.rs` (`mod token_overlay;`).

### D.3 Interface contracts

```rust
struct EmbedOverlay { rows: DevicePtr /*[n,h] bf16*/, ids_dev: DevicePtr /*u32[n] asc*/,
                      slot_map: DevicePtr /*i32[vocab]*/, n_override: u32,
                      lmhead: Option<LmHeadOverlay> }  // Some => untied separate head
struct TokenOverlaySet { overlays: Vec<Option<EmbedOverlay>> /*len max_loras*/,
    embed_slot_map_table, embed_rows_table, lmhead_rows_table, lmhead_ids_table: DevicePtr /*u64[max_loras]*/,
    n_override_table: DevicePtr /*u32[max_loras]*/, max_n_override: u32 }
struct OverlayKernels { rowdiff, embed_overlay, lmhead_overlay_bf16, lmhead_overlay_f32: KernelHandle }
```
Tables are load-time-fixed addresses (graph-safe); the only per-step arg is `seq_slot`. On rotate/swap only cell `[k]` is rewritten (mirror `types.rs:refresh_slot_tables`). Kernel handles resolved with `layers::try_kernel(gpu,"token_overlay",…)` (null ⇒ feature silently unused).

**`build_overlay`** (port of the NLLB pipeline): early-out if no base tensor → `Ok(None)`; shape-validate `[R,h]`/`[T,h]` vs served `h`; `clamp_trainable_to_vocab` (drop `vocab≤idx<R` extension rows with warn + tail-order enforce, hard-error `idx≥R`); **row-diff kernel** over `r_eff=min(R,vocab)` (`max|base−served|>0.1` — the "correct on either served base" trick); `override_ids = sort(dedup(differing ∪ trainable))`, empty→`Ok(None)`; materialize compact `rows` (trainable→`delta[k]` delta-wins, else baked `base[id]`); host `slot_map[vocab]=-1` then `[id]=slot`, `ids_dev` ascending, H2D both; free the 4 raw buffers. `modules_to_save` case: same builder, `trainable=∅`, base = the full weight.

### D.4 Forward hooks

- **`apply_embed_overlay`** — grid `[num_tokens]`, block `[256]`; per row `s=seq_slot?seq_slot[r]:active_slot`, skip `s<0`, `slot=slot_map_table[s][token_ids[r]]`, if `≥0` copy `rows_table[s][slot]→hidden[r]` (full replace). **Placement: after gather, before `scale_embeddings`.** Single-token: between D2D copy (`impl_a3.rs:35`) and scale (`:37`). Batched: after `batched_embed` in `embed_chunk.rs`, before scale (disjoint from the vision-pad overwrite).
- **`apply_lmhead_overlay`** — grid `[num_tokens, max_n_override]`, block `[32]` (one warp per (row,j)); `s=…`, guard `j<n_override_table[s]`, `id=ids_table[s][j]`, warp-reduced `logits[row*vocab+id] = dot(hidden[row], rows_table[s][j])` (**write**, replace). **Placement: after base GEMV/GEMM, before softcap.** `lm_head` (`impl_a3.rs:264`, `is_fp32` picks kernel) and `lm_head_batched` (`:182`, always bf16).

### D.5 Tied/untied + NVFP4

Tie is **buffer aliasing** (`qwen35.rs:125`), not a config flag. If the adapter ships `lm_head.token_adapter.*` (or `modules_to_save` lists `lm_head`) → build a distinct `LmHeadOverlay`. Else, compute `tied = (lm_head_weight.0 == embed_tokens.0)` (or nvfp4/fp8 derived-from-embed): tied ⇒ **must** run lm_head overlay reusing embed `rows`/`ids` (`lmhead_rows_table[k]=embed_rows_table[k]`); untied + not shipped ⇒ NULL the cell so the kernel skips (embed-only correction). Embed table is always BF16 (dtype-agnostic replace); lm_head may be NVFP4/FP8 but the **logits buffer is BF16 (prefill/batched) or FP32 (single-token)** — the overlay writes recomputed columns into that buffer after the base projection, discarding the quantized column. No NVFP4 weight manipulation, no new quant kernel.

### D.6 Staged delivery

- **F2-A** (pure): step S parser edits + `classify_overlay_key` + pure helpers + host tests. Zero runtime change.
- **F2-B**: `token_overlay.cu` + ops launchers + `EmbedOverlay`/`build_overlay` + `TokenOverlaySet` + loader build. Overlay built, hooks not yet called.
- **F2-C**: the two forward hooks + model field/handle resolution + tied/untied + fp32 variant. The flip; gated on presence of overlay tensors → non-overlay adapters byte-identical to today.

---

## E. Feature 1 — MoE expert + router LoRA (correctness-first)

**Non-negotiable ordering:** land a *numerically correct* BF16 per-expert delta side-path that folds onto MoE output buffers via `apply_lora_delta` **verbatim**, base NVFP4/FP8 grouped GEMM **byte-identical**. Only after an oracle passes do we build the fused/S-LoRA kernels.

**Layout reconciliation (accept before coding):** real Holo/Qwen3.6 checkpoints store experts *fused*; a real PEFT export likely emits `target_parameters` fused keys, **not** `mlp.experts.N.*`. Therefore: **internal representation is always per-expert** (`BTreeMap<(u16,ExpertProj),LoraPair>`); **P1 accepts the per-expert on-disk spelling** (task requirement, trivially sliceable, tested with a synthetic per-expert fixture); **fused on-disk import is P3** and is a *named reject* until then. Flag to the human: a genuine Holo/Qwen3.6 PEFT export will not load until P3 — P1's test vehicle is a hand-authored/synthetic per-expert adapter.

### E.1 Phase map

| Phase | Content | New CUDA |
|---|---|---|
| **P1 (lands now)** | classifier/parser accept per-expert + router; `LoraTarget` threading; sparse per-expert storage + separate audited-size expert pool; **single-active-adapter** apply via host-synced per-expert `apply_lora_delta` loop (prefill sorted rows / decode `indices_dev`); router delta on gate logits | **No** |
| **P2** | multi-adapter S-LoRA: 2-D `(slot,expert)` route tables + expert-indexed grouped BGMV; remove host sync; graph-safe | Yes |
| **P3** | fused on-disk import (`target_parameters`/fused `gate_up_proj` decomposer) + fused grouped-delta kernel in the NVFP4/b12x grouped-GEMM epilogue | Yes |

### E.2 P1 classifier — `lora/key.rs` + new `lora/target.rs`

Change `classify_key` return `(usize, LoraModule, AdapterAb)` → `(usize, LoraTarget, AdapterAb)`.
```rust
// lora/target.rs (SPDX line 1, tests in own file — keeps types.rs under cap)
pub enum LoraTarget { Attn(LoraModule), Router, Expert { n: u16, proj: ExpertProj } }
pub enum ExpertProj { Gate, Up, Down }
pub struct ExpertLoraLayer { pairs: BTreeMap<(u16, ExpertProj), LoraPair> }  // sparse
```
Insert **before** the `other => REJECT[unsupported-module]` catch-all (`key.rs:100`): `"mlp.gate" => Router`; and `t if t.starts_with("mlp.experts.")` parsing `N.{proj}` with named rejects — `expert-lora-on-dense-model` (`num_experts==0`), `fused-expert-lora` (`experts.gate_up_proj`/bare `down_proj`), `malformed-expert-key/index`, `expert-out-of-range` (`n>=cfg.num_experts`), `unsupported-expert-proj`. Wrap the 7 dense arms as `Attn(LoraModule::…)`. The `key.rs:102` full-attention gate does **not** block MoE layers (they are `FullAttention`). **F2's overlay interception in `audit_adapter` stays above this call — do not disturb it.**

### E.3 P1 storage + threading — `lora/types.rs`, `loading.rs`, `impl_lora.rs`

Keep `LoraModule` (7 variants) intact for attention/dense. Add to `LoraLayerWeights`: `router: Option<LoraPair>`, `experts: Option<ExpertLoraLayer>`. `audit_adapter`/`pack_slot` map key → `(usize, LoraTarget)`; dense modules keep the `LoraModule::ALL` walk, experts get a **dynamic** walk over the audited `(layer,expert,proj)` set (no static 256-wide `ALL`). Compute expert/router `dims()` **at the `audit_adapter` call site** (layer known: expert = `(moe_intermediate_size_for(layer), h)` for gate/up, `(h, moe_size)` for down; router = `(num_experts, h)`) rather than threading a layer index into `dims()`. Extend `module_pair`/`refresh_slot_tables` for the two new fields (P1 experts touch only the single active/pinned slot cell). Install router + `ExpertLoraLayer` for the active adapter via a new MoE arm in `impl_lora.rs`.

### E.4 P1 pool sizing (quantified) — `slot_math.rs`, `env.rs`

Per-(layer,expert) BF16 bytes at padded rank R = `3·R·(h+inter)·2 = 15,360·R` (Holo). Dense-full (256×40) = **2.34 GiB/slot @ R16**, 9.4 GiB @ R64 — ~160× the 14.6 MiB attention slot, `×max_loras`. Strategy: **separate expert pool sized from the audited key set** (real adapters target subsets, e.g. 10 layers); hard VRAM preflight bail with exact figures (mirror `loading.rs:275`, allocate before KV snapshot); separate `--max-lora-expert-rank` (default 16); `--lora-experts=on` default off; P1 stores only the active/pinned adapter's expert pool (no CACHE tier). Router adds negligible `R·(h+E)·2` per layer.

### E.5 P1 apply — injection sites (NO new kernel)

- **Router (do first):** after base gate GEMM, before top-k — `forward_prefill.rs:200` / `forward.rs:126`. One `apply_lora_delta(x=layer_input[n,h], base_out=gate_logits[n,E], A_router[R,h], B_router[E,R], scale)` per layer. Reproduces PEFT `mlp.gate` (delta on routing logits pre-selection).
- **Expert prefill:** inside `run_routed_grouped_gemm`. **down_proj (primary):** after base down GEMM (`forward_prefill_routed.rs:478`, `expert_down_out[total_expanded,h]` sorted), **before** `moe_unpermute_reduce_indexed` (`forward_prefill.rs:351`) so the router weight multiplies (base+delta); `x`=post-SiLU `expert_gate_out[·,inter]`. **gate/up:** after gate_up GEMM (`:317`), before `silu_mul` (`:322`); `x`=`expert_input` gathered by `sorted_token_ids`.
- **Correctness-first apply (single adapter, host-synced offsets):** D2H `expert_offsets[E+1]` once per MoE layer (graph-breaking, legal in prefill); `for e in adapted_experts { rows=off[e+1]-off[e]; apply_lora_delta(x=sorted_in+off[e]*k_in*2, base_out=expert_out+off[e]*n_out*2, A,B,scale, m=rows, k_in, n_out) }`. Only adapted experts (sparse map) launch. Reuses `apply_lora_delta` byte-for-byte; base grouped GEMM untouched; **zero new CUDA**.
- **Expert decode:** `forward.rs` fused GEMV uses `indices_dev[top_k]` (unsorted). Loop the token's `top_k` selected experts; for each adapted pair `apply_lora_delta(m=1,…)` onto its expert-output slice (`forward.rs:462` down / `:394` gate-up). Batched-decode extends per token.
- **No `seq_slot` plumbing into MoE in P1** — single-active-adapter uses the installed-pair path.

### E.6 P1/P2/P3 boundaries

Deferred, interfaces pinned: **P2** `ExpertLoraRoute` `a/b_table [max_loras×E]` gathered `[slot·E+expert]` (outer product of `ExpertTables`' expert dim with the S-LoRA slot dim) + grouped BGMV kernel (`grid.z=expert`, `seq_slot[token]` for adapter) — design P1's `ExpertLoraLayer` to lay out into these without re-audit. **P3** fused-epilogue kernel folding the BF16 delta into the NVFP4/b12x grouped-GEMM output stage (`:437`/`:252`), and a decomposer slicing fused `[E,·,·]` → per-expert `LoraPair`s P1 already consumes. Also deferred: CUDA-graph capture of the expert path (P1 host sync blocks it; P2 restores), expert-pool CACHE/LRU (P2), `nemotron_moe` experts (mirror after P1), `mlp.shared_expert.*` (optional dense add, needs a `FfnComponent::Moe` install arm since `prefill_weights.rs:163` only reaches `Dense`).

---

## F. Kernel additions (consolidated)

| Kernel | Feature/phase | File | Signature sketch |
|---|---|---|---|
| `embed_rowdiff_bf16` | F2-B | `kernels/gb10/common/token_overlay.cu` | `(base, served: bf16[rows,h], flags: u8[rows], rows, h, thresh)` one thread/row |
| `embed_overlay_routed_bf16` | F2-C | ″ | `(ids:u32[n], seq_slot:i32[n]?, active:i32, slot_map_tab:u64[L], rows_tab:u64[L], out:bf16[n,h], h)` grid `[n]` |
| `lmhead_overlay_routed_bf16` | F2-C | ″ | `(hidden:bf16[m,h], seq_slot?, active, rows_tab, ids_tab, n_tab, logits:bf16[m,vocab], h, vocab)` grid `[m,max_n]` warp/pair |
| `lmhead_overlay_routed_f32` | F2-C | ″ | same, `float* logits` (single-token FP32 decode) |
| — (none) | **F1-P1** | — | reuses `apply_lora_delta` (GEMV + `scaled_add`) |
| grouped expert BGMV | F1-P2 | new `.cu` in `common/` | grouped analogue of `apply_lora_bgmv`, `grid.z=expert`, reads `expert_offsets` + `seq_slot` |
| fused grouped-delta epilogue | F1-P3 | fold into NVFP4/b12x grouped GEMM | folds BF16 delta into grouped-GEMM output stage |

Build notes (per `moe-kernel-build`): a new `common/*.cu` compiles into every target (dedups to ~1 nvcc call), module name = file stem `token_overlay`, no `KERNEL.toml` edit; all entry points `extern "C" __global__`, SPDX line 1, `(unsigned long long)id*h` indexing to avoid 32-bit overflow at large vocab×h. `ATLAS_KERNEL_SET_HASH` forces the `atlas-kernels` recrate (no stale PTX). Human builds in docker `atlas-gb10:b12x-ready`: `ATLAS_TARGET_MODEL='*' cargo build -p spark-server --release --bin spark --no-default-features --features cuda`. Pure-Rust type-check: `ATLAS_SKIP_BUILD=1 cargo check -p atlas-core -p spark-model`.

---

## G. Test plan (consolidated)

**Host-only unit (CI, CUDA-free — `atlas-core` + `spark-model`):**
- *Parser (step S):* `trainable_token_indices` list+dict; `modules_to_save:[lm_head]` now accepts (flip `dora_bias_rank_pattern_rejected_named`); empty-`target_modules`-with-overlay accepts; `modules_to_save:[q_proj]` rejects; `target_parameters` present → named reject; `router_gate_accepted`; `--max-lora-expert-rank` overflow reject.
- *F2 overlay:* `clamp_trainable_to_vocab` (idx≥R error, vocab≤idx<R skip+count, tail-order violation error, happy); `build_override_set` union/sort/dedup; `override_source` delta-wins; `classify_overlay_key` for all 4 tensor spellings + `.language_model.` multimodal segment + bare `modules_to_save` `.weight`, `None` for ordinary `.lora_A.weight`.
- *F1 classifier:* `…mlp.experts.7.gate_proj.lora_A.weight → Expert{7,Gate}` on FullAttention layer; `…mlp.gate.lora_A.weight → Router`; named rejects for `n>=num_experts`, fused `experts.gate_up_proj`, `num_experts==0`, `experts.7.foo_proj`.
- *F1 pool sizing:* golden `expert_pool_bytes(audited_set,R) == Σ 15,360·R` (like `slot_math_tests.rs:33`); expert/router `dims()` golden shapes using `moe_intermediate_size_for`.

**GPU (`#[ignore]` / gated harness, human runs build):**
- *F2:* `embed_overlay_routed` replaces exactly mapped rows, base rows byte-identical across a **mixed `seq_slot` batch** (the correctness invariant); `lmhead_overlay_routed` matches host dot reference for overridden ids, other columns untouched; fp32 parity.
- *F1 oracle (reference, not another Atlas path):* dequant base experts→BF16, apply `E_delta=scale·B·A` densely per expert in host/torch, compare full-MoE logits within BF16 tol. Cases: router-only (top-k selection changes as predicted); single-expert (only tokens routed to that expert change, rest bit-identical); all-experts small-rank on 1–2 layers (full match); **prefill vs decode** consistency (guards the two injection sites); **base isolation** (adapter loaded but request routed to base ⇒ bit-identical to no-LoRA).

**E2E (human, `/verify` on GPU):** F2 — serve holo/qwen with a kuku-style `trainable_tokens` adapter; assert the added token's logit + embedding row change and a co-batched base request is bit-identical to no-adapter; confirm ordering by a golden logit. F1 — synthetic per-expert adapter fixture (documented as the P1 vehicle; a real fused export needs P3).

Repo rules for every new file: SPDX line 1, ≤500 LoC/.rs, tests in own `*_tests.rs`, clippy deny-warnings, no `cargo fmt --all`, add unit tests alongside each edit, run `cargo test` on `atlas-core`+`spark-model` before commit, **push origin only**, do not build/deploy while a sibling subagent runs.

---

## H. Risk / what-can-land-now

**Lands now, zero runtime risk:** step S (parser accept-edits) and F2-A + F1's host-only classifier/parser tests are pure Rust, fully `cargo check`/`cargo test`-able, and change no runtime behaviour (nothing consumes the new config fields or `LoraTarget` variants yet). This is the safe first commit.

**Lands next, gated & byte-identical for existing adapters:** F2-B/C (token overlay) — the feature is a no-op unless overlay tensors are present and kernel handles resolve; non-overlay adapters take the exact current path. F1-P1 (expert/router) — behind `--lora-experts=on` (default off), single active adapter, side-path additive so base is untouched.

**Top risks:**
1. **Real MoE adapters won't load in F1-P1** (fused on-disk `target_parameters`). Mitigated by explicit named reject + loud human flag; P1 validated on synthetic per-expert fixtures. A genuine Holo/Qwen3.6 export is **P3**. *No file on `/tank` confirms the fused key spelling / A-B axis order — dump a real `LoraConfig(target_parameters=[…])` state_dict before writing the P3 decomposer.*
2. **Expert memory blow-up** if sized from config maxima or a user targets 256×40 @ R64 (9.4 GiB/slot). Mitigated: audited-set sizing, hard preflight bail with exact numbers, separate low expert-rank cap, default-off.
3. **Wrong fold order** (down_proj must precede `moe_unpermute_reduce_indexed`; gate/up precede `silu_mul`; embed overlay precedes `scale_embeddings`; lm_head overlay precedes softcap). Silent numeric corruption — the single-expert and golden-logit oracles are the guards.
4. **F1-P1 host-sync perf / graph break** (per-layer `expert_offsets` D2H + many small launches). Deliberate throwaway scaffold; **do not benchmark P1 for tok/s**; P2 removes it.
5. **Type-threading blast radius** (`classify_key → LoraTarget` touches `audit_adapter`/`pack_slot`/`module_pair`/`refresh_slot_tables`/install). Mitigated by keeping `LoraModule` intact and adding experts/router as *new* fields, and by F1 landing after F2 so the two never edit `classify_key`/`audit_adapter` in the same PR.
6. **`types.rs` LoC cap** — `LoraTarget`/`ExpertProj`/`ExpertLoraLayer` go in new `lora/target.rs`; F2 overlay types in `lora/overlay.rs`. `loading.rs` (478) is the tightest budget — keep both features' edits there ≤15 lines via delegation.
7. **Router instability** (low-rank routing-logit perturbation reshuffles expert selection). Support it (task asks) but keep independently toggleable; test base requests unaffected.
