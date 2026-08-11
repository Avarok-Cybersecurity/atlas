# SOLID LoRA — Atlas per-request adapters that are graph-safe, varlen-safe, and prefix-cache-safe

Date: 2026-07-16
Status: architecture + increment plan
Scope: attention S-LoRA (landed reference) · MoE router/expert fold (all base dtypes + decode) · embed/lm_head overlay apply
Supersedes the open items in `docs/design/lora-moe-embed.md`; that doc's §D.2/D.3 build-ordering is corrected here (§7.3).

---

## 0. Thesis

Atlas already has ONE correct LoRA route: **attention S-LoRA via `apply_lora_bgmv`** (`layers/ops/lora_delta.rs:269`). It is per-request, zero-overhead-when-off, and CUDA-graph-capturable. Every other adapter surface (MoE experts, MoE router, embed/lm_head overlay) must be rebuilt to the **same three-invariant contract** that route already satisfies:

1. **Identity flows ONLY through a per-step device buffer** whose *address is fixed at load time* and whose *contents are re-uploaded each step* (like `positions`/`block_table`). For attention that buffer is `seq_slot[n]` (`ctx.attn_metadata.seq_slot`). Adapter routing tables (`a_table`/`b_table`/`scale_table`) are `[max_loras]` device arrays at load-fixed addresses = stable kernel args across capture↔replay.
2. **No host D2H, no host-driven launch count, in the fold hot path.** All routing offsets/ids are consumed *device-side* as kernel args. The base grouped-GEMM already does this with `expert_offsets`; the fold must too.
3. **Per-row skip is a predicated kernel early-return**, not a host branch: `s < 0 ⇒ base row, no delta`; `table[s] == 0 ⇒ this module/slot doesn't adapt, no delta`.

Everything below is the mechanical application of that contract to the three remaining surfaces, plus the increment ordering that lands the cheapest correct slices first.

The current blocker is concrete and singular in shape: `apply_expert_lora_prefill_down` (`layers/moe/lora.rs:149`) does `ctx.gpu.copy_d2h(expert_offsets, ...)` then a **host `for w in work` loop** issuing one `fold_chunked` per adapted expert at host-computed offsets (`lora/expert_apply.rs:147-157`). That single D2H + data-dependent launch sequence is why the whole MoE feature is prefill-eager-only and `reject_decode_lora` (`moe/lora.rs:173`) hard-refuses decode/verify. Remove it and the feature becomes graph-safe.

---

## 1. End-state architecture

### 1.1 Two fold implementations, not five

The MoE base has five forward paths by dtype/shape (map in §4), but the fold reduces to **two output-buffer layouts**:

| Regime | Paths | Routing tensor (device-side) | Fold kernel |
|---|---|---|---|
| **SORTED / grouped** | NVFP4 grouped prefill (`forward_prefill.rs`), bf16-dequant (`forward_prefill_bf16.rs`), fp8 grouped (`forward_prefill_fp8.rs`) | `expert_offsets [ne+1]` + `sorted_expert_ids [te]` + `token_to_perm` (all in `gate_logits` scratch, produced by `moe_sort_by_expert`) | **grouped device fold** — one kernel, grid sized to `worst_case_m_tiles`, reads `expert_offsets` on device to derive each expert's `[row_off,rows)` span |
| **SLOT-MAJOR / unsorted** | batched short prefill (`forward_batched.rs`), decode (`forward.rs`) | `indices_dev [top_k]` per token + `weights_dev` | **gather-by-expert-id BGMV** — `apply_lora_bgmv` generalized: `seq_slot → indices_dev`, a/b tables keyed by expert id |

The fold math is dtype-independent: base weight NVFP4/bf16/fp8 only determines the *base* GEMM; the LoRA A/B are always BF16 and the fold always writes onto the BF16 output buffer (`expert_down_out` / `expert_gate_out`, sorted or slot-major). So the base dtype never forks the fold — only **layout** (sorted vs slot-major) and **capture-status** (prefill vs decode graph) do.

### 1.2 The per-request dimension: (slot, expert) grouping

Feature-1 today stores ONE `MoeLoraWeights` (`moe/lora.rs:36-48`), globally installed from the active slot (`impl_lora.rs:74,187`). SOLID makes it S-LoRA: the fold gains a **request-adapter dimension** on top of the expert dimension.

- **Sorted path:** the fold selects per-expert `(A_e, B_e, scale_e)` from a `[num_experts]` table *for the request-adapter that owns each sorted row*. The owning adapter is recovered by gathering a per-packed-row `adapter_slot` map through the same `token_to_perm` permutation the activations took. Combined key = `(slot, expert)`. Table shape becomes `[max_loras][num_experts]` of A/B pointers (a `u64` pointer table, exactly the `LoraRoute.a_table` pattern lifted to 2-D).
- **Slot-major path (decode/batched):** each expert slot already carries `indices_dev[flat_slot] = expert_id`; the request-adapter for that token is `seq_slot[token]` (the existing attention buffer, reused). Combined key = `(seq_slot[token], indices_dev[flat_slot])` → gather `(A,B,scale)` from the `[max_loras][num_experts]` table. This gives per-request routed experts "for free" from buffers that already exist.

Base rows (`seq_slot < 0`, or `adapter_slot` map cell `< 0`) skip; a `(slot,expert)` cell that is `0` (adapter doesn't adapt that expert) skips. Zero traffic, zero math — identical to the attention `a_table[s]==0` early-return.

### 1.3 Prefix-cache identity closure

`adapter_id: u64` (`traits.rs:115`) keys the KV-block radix roots (`radix_tree/inner.rs:67`, disjoint root per adapter) and the SSM snapshot hash (`radix_tree.rs:33-44`). Today `adapter_id_for_slot` (`lora/types.rs:304`) hashes only the **attention pool slot's `(name, generation)`** — it does NOT capture MoE-adapter or overlay identity. In the current single-bundle world MoE+attn+overlay ship under one name so generation bumps cover disk re-stage, but SOLID's per-request/decoupled routing breaks that coupling. **End state:** `adapter_id` must fold in every delta that touches a token's hidden state — MoE and embed overlay included — so a base prefix (`adapter_id==0`) is never reused by an adapted suffix and vice-versa. Concretely: keep one `adapter_id` per resolved *bundle* (name+generation) and forbid decoupled MoE-only rotation from reusing an unchanged `adapter_id` (§7.6).

### 1.4 Overlay (embed / lm_head) as a third routed table set

Embed and lm_head overlays follow the identical contract: `[max_loras]` u64 pointer tables (`embed_slot_map_table`, `embed_rows_table`, `lmhead_rows_table`, `lmhead_ids_table`, `n_override_table`) at load-fixed addresses; per-step identity via the same `seq_slot` device buffer (uniform `active` for single-request paths, real device `seq_slot` for the mixed-decode batch). The kernels (`kernels/gb10/common/token_overlay.cu`) are already compiled; the wiring (build split + forward hooks) is the work. Embed overlay writes the override embedding row *before* `scale_embeddings`; lm_head overlay *replaces* the recomputed logit column for overridden vocab ids *before* softcap.

---

## 2. Design principles (the "SOLID" in the name)

- **S — Single fold responsibility per layout.** Two fold ops (`grouped device fold`, `gather BGMV`), each owning exactly one output-buffer layout. No path-specific fold copies. Dtype never forks the fold.
- **O — Open to base dtypes, closed to fold rewrites.** Adding a base dtype (a future int4/mxfp) adds a base GEMM path; it reuses the existing fold unchanged because the fold binds to the BF16 output buffer, not the base weight.
- **L — Routing tables are substitutable.** Attention `LoraRoute`, MoE `(slot,expert)` table, and overlay tables are all "`[max_loras]` u64 pointer arrays at load-fixed addresses, identity via `seq_slot`." Any consumer kernel takes the same arg shape.
- **I — Identity channel is one interface.** `seq_slot` (device buffer, fixed address, per-step contents) is THE per-request identity channel for attention, MoE-slot-major, and overlay. Sorted MoE adds a derived `sorted_row_adapter` gathered through `token_to_perm` — still no new host channel.
- **D — No host in the fold.** The fold depends on device buffers (`expert_offsets`, `indices_dev`, `seq_slot`) as kernel args, never on a host readback. This is the invariant that makes it graph/codispatch/varlen/prefix-cache safe simultaneously.

---

## 3. Ordered increment plan

Legend: **[THIS PASS]** = lands in this change set · **[NEXT]** = immediate follow-up · **[LATER]** = deferred, design-locked but not scheduled.

### Increment 0 — Router-map plumbing + per-row adapter channel  **[THIS PASS]**
The prerequisite for everything else: a device-side per-packed-row adapter map and a `ForwardContext` channel to carry it.
- Add `ForwardContext.moe_row_adapter: Option<DevicePtr>` (`layer.rs:210`), an `[total_tokens]` i32 device buffer, `<0` = base, else `adapter_slot`. Mirrors `routed_lora_layers` (attention). Default `None` ⇒ byte-identical to today.
- Host builder `build_moe_row_adapter_host(cu_seqlens_host, adapter_slots, active) -> Vec<i32>` (pure, unit-tested, GPU-free), broadcasting each stream's `seq.adapter_slot` across its `[cu_seqlens_host[b], cu_seqlens_host[b+1])` span. **Keys off `cu_seqlens_host` / proc_count prefix-sums, NOT `b*chunk_len`** (varlen + partial-prefix-hit correctness, per codispatch-varlen finding §D.1).
- Uploaded to a fixed-address buffer (new arena slot), contents re-uploaded per step. Single-request/decode paths use a uniform fill (like `upload_seq_slot_uniform`, `impl_b1.rs:168`).

### Increment 1 — Device-side grouped down-fold (kills the D2H)  **[THIS PASS]**
Replace `apply_expert_lora_prefill_down`'s `copy_d2h` + host loop with a single grouped kernel.
- New kernel `moe_lora_grouped_down` reading `expert_offsets` + `sorted_expert_ids` (device) to derive each row's `(expert, row_off)`; grid sized to `worst_case_m_tiles = ceil(total_expanded / TILE_M)` (static). Math: `expert_down_out[row] += scale_e · (x[row] @ A_e^T) @ B_e^T` for `row ∈ [expert_offsets[e], expert_offsets[e+1])`.
- Consumes the existing fixed-address scratch (`MoeLoraWeights.xa`/`.delta`, `moe/lora.rs:46-48`) — no capture-breaking alloc.
- **Single-adapter for this increment** (one `(A_e,B_e)` table, no slot dim yet) — but reads `moe_row_adapter` to skip base rows (`<0` ⇒ no fold). This closes the "base request in a mixed batch is silently mis-computed" hazard (codispatch-varlen §C.2) even before full S-LoRA.
- Applies verbatim to NVFP4 / bf16-dequant / fp8 grouped prefill because all three write the same sorted BF16 `expert_down_out` (all-dtype §Summary). Remove the upstream `forward_prefill.rs:52-65` bf16/fp8 guard for the *down* fold once wired.
- Router fold (`apply_router_lora_prefill`, `moe/lora.rs:108`) is already device-clean (token-major, no offsets) — extend it to consult `moe_row_adapter` for the base-row skip.

### Increment 2 — Graph-safe prefill capture for grouped MoE fold  **[THIS PASS]**
With the D2H gone, drop the prefill-eager restriction for the grouped path: allow the grouped fold inside the prefill graph region. Keep `reject_decode_lora` for the slot-major decode/verify paths (Increment 4 lifts it). Verify capture cleanliness (no residual sync) against the `use_graphs` predicate (`decode_a.rs:208-215`).

### Increment 3 — S-LoRA (slot,expert) grouping for grouped prefill  **[NEXT]**
Lift the single-adapter table to `[max_loras][num_experts]`.
- Build `sorted_row_adapter[te]` by gathering `moe_row_adapter` through `token_to_perm` (`forward_prefill.rs:314`) — same permutation the activations took, so the down fold (which runs on sorted rows) reads the correct owning adapter per row.
- Fold kernel keys `(A,B,scale)` on `(sorted_row_adapter[row], sorted_expert_ids[row])`. Base/unmapped cells skip. This is the true per-request routed-expert path.
- Router fold reads `moe_row_adapter` directly (token-major, pre-permute).

### Increment 4 — Slot-major gather-BGMV decode/verify fold  **[NEXT]**
The genuinely new kernel for decode + short-batched (paths 4/5, all-dtype §2).
- `moe_lora_gather_bgmv`: per (token, expert-slot) row, `expert_id = indices_dev[flat_slot]`, `adapter = seq_slot[token]`, gather `(A,B,scale)` from the `[max_loras][num_experts]` table; two-kernel shrink/expand-fold like `apply_lora_bgmv`. Pure launches, pointer/value-stable args ⇒ **captures inside the decode graph**.
- Handles the **fused-SiLU friction**: down-proj `x` (post-SiLU `silu(gate)*up`) is computed inside `moe_expert_silu_down_shared*` and not materialized. Fold recomputes `silu(gate)*up` per slot into fixed scratch (preferred) rather than splitting the fused kernel. Gate/up folds have no such issue (`x = input_t`/`expert_input`).
- Once landed, replace `reject_decode_lora` (`moe/lora.rs:173`) with the gather-BGMV dispatch. This also fixes the **prefix-hit → decode-path bail** hazard (prefix-caching §5): a warm prefix that leaves a single suffix token (`proc_count==1`) currently routes MoE-adapted prefill into the rejecting decode fold and fails the request.

### Increment 5 — Gate/up-proj grouped fold  **[NEXT]**
Extend the grouped fold to gate/up (`set_lora_weights` currently bails on non-Down, `moe/lora.rs:68-78`). Requires materializing the sorted gate/up input (gather `expert_input` by `sorted_token_ids` into scratch) before folding onto sorted `expert_gate_out`/`expert_up_out`, injected between the grouped up-GEMM and `silu_mul` (`forward_prefill_routed.rs:317→322`). Same injection seam for bf16/fp8 (`:219→220`, `:483→492`).

### Increment 6 — Embed/lm_head overlay apply (Feature-2 F2-C)  **[THIS PASS for build+embed hook; NEXT for lm_head mixed-decode]**
Complete the overlay wiring (embed-overlay-flip finding). Two-stage build (§7.3), forward hooks (§8), routing via `seq_slot`. Embed hook + single-token/batched lm_head **[THIS PASS]**; the mixed-decode `seq_slot`-routed lm_head hook (`decode_b2.rs:97`) **[NEXT]** (needs the device `seq_slot` batch, lands with Increment 4's decode routing).

### Increment 7 — Prefix-cache identity closure  **[NEXT]**
Fold MoE + overlay identity into `adapter_id` (§1.3). Add an invariant test that a decoupled MoE rotation cannot reuse an unchanged `adapter_id`.

**This-pass boundary:** Increments 0–2 + 6(embed) land a *correct, graph-safe, base-dtype-complete grouped-prefill down-fold with per-row base skip* and the embed overlay. Full S-LoRA slot routing (3), decode fold (4), gate/up (5), and identity closure (7) are the ordered follow-ups. Nothing in this pass touches the loud bails except the down-path bf16/fp8 guard; every unimplemented surface keeps its loud `bail!`.

---

## 4. MoE forward path map (fold injection points)

All in `crates/spark-model/src/layers/moe/`.

| Path | fn (file:line) | base dtype | out-buf layout | routing (device) | down fold point | gate/up fold point | graph-safe |
|---|---|---|---|---|---|---|---|
| 1 NVFP4 grouped | `forward_prefill.rs:17` (routed `forward_prefill_routed.rs:34`) | NVFP4 | sorted `[te,·]` | `expert_offsets`+`sorted_expert_ids` (`:311-327`) | after down GEMM, before unpermute (`:379`, WIRED) | after up GEMM, before `silu_mul` (`forward_prefill_routed.rs:317→322`) | after Incr 1 |
| 2 bf16-dequant | `forward_prefill_bf16.rs:11` | bf16 | sorted `[te,·]` | idem (`:161-177`) | after down GEMM `:242`, before unpermute `:246` | `:219→220` | after Incr 1 |
| 3 fp8 grouped | `forward_prefill_fp8.rs:19` | fp8/W8A8 | sorted `[te,·]` | idem (`:301-313`) | after down block `:587`, before unpermute `:591` | `:483/:419→:492` | after Incr 1 |
| 4 batched short | `forward_batched.rs:12` | bf16/fp8/nvfp4 | slot-major `[top_k,·]` | `indices_dev`+`weights_dev` (`:123`) | after fused silu+down, before `moe_weighted_sum_blend` `:414` | between fused gate+up and silu+down | Incr 4 (gather-BGMV) |
| 5 decode | `forward.rs:28` | bf16/fp8/nvfp4 | slot-major `[top_k,·]` | `indices_dev`+`weights_dev` (`:92`) | after fused silu+down, before blend `:529` | between fused gate+up and silu+down | Incr 4 (MUST — capture) |

Router fold hook (all paths): after gate GEMM/GEMV, before top-k (`forward_prefill.rs:224` wired; add to bf16 `:121→128`, fp8 `:254→263`, batched `:97→127`, decode `:126→128`).

Bases bail today: bf16/fp8 grouped at `forward_prefill.rs:52-65`; decode/verify via `reject_decode_lora` at `forward.rs:35`, `forward_k2.rs:22`, `forward_k3.rs:19`, `forward_batched.rs:20`, `forward_atomic_c4.rs:22`, `forward_token_major.rs:27`. Keep each until its increment lands.

---

## 5. Interface contracts

### 5.1 Identity channel (shared)

```rust
// layer.rs — ForwardContext gains ONE MoE channel (mirrors routed_lora_layers)
pub moe_row_adapter: Option<DevicePtr>,   // i32[total_tokens], <0 = base; fixed address, per-step contents

// Attention reference, unchanged: ctx.attn_metadata.seq_slot : DevicePtr  (i32[n], <0 = base)
```

Host builders (pure, GPU-free, unit-tested — mirror `lora/slot_math.rs:50 build_seq_slot_host`):
```rust
fn build_moe_row_adapter_host(cu_seqlens_host: &[i32], adapter_slots: &[i32], active: i32) -> Vec<i32>;
// row r in stream b (cu_seqlens_host[b] <= r < cu_seqlens_host[b+1]) => resolve(adapter_slots[b], active)
// resolve(slot, active) = if slot >= 0 { slot } else { active }   (matches attention -1→active convention)
// but a GENUINE base request must carry a distinct <0 base sentinel (see §6 note) so it maps to base, not active.
```

### 5.2 Routing tables (shared shape — the "L/I" of SOLID)

```rust
// Attention (landed): LoraRoute { a_table,b_table,scale_table: DevicePtr /*[max_loras]*/, k_in,n_out,max_rank }
// MoE (new): 2-D lift — pointer tables indexed by (slot, expert)
pub struct MoeLoraRoute {
    pub a_table:     DevicePtr,  // u64[max_loras * num_experts] -> bf16* A_e  (0 cell => skip)
    pub b_table:     DevicePtr,  // u64[max_loras * num_experts] -> bf16* B_e
    pub scale_table: DevicePtr,  // f32[max_loras * num_experts]
    pub num_experts: u32, pub max_rank: u32, pub k_in: u32, pub n_out: u32,
}
```
All tables at **load-time-fixed addresses** (built in `set_lora_weights`), so they are stable kernel args across capture↔replay. Rotate/swap rewrites only the `[slot]` stripe (mirror `types.rs:refresh_slot_tables`).

### 5.3 Fold kernel signatures

```
// Grouped (sorted) — replaces apply_expert_lora_prefill_down's D2H+host loop
moe_lora_grouped_down(
    x:            bf16[te, moe_inter],   // post-SiLU expert_gate_out
    base_out:     bf16[te, hidden],      // expert_down_out (folded in place)
    expert_offsets: u32[ne+1],           // DEVICE (never D2H)
    sorted_expert_ids: u32[te],          // DEVICE
    sorted_row_adapter: i32[te]|NULL,    // DEVICE, <0 => base row skip (Incr 3; NULL => single-adapter Incr 1)
    a_table,b_table: u64[max_loras*ne],  // 0 cell => skip
    scale_table:  f32[max_loras*ne],
    xa_scratch:   bf16[te, max_rank],    // fixed-address arena
    ne,hidden,moe_inter,max_rank: u32)
    // grid = worst_case_m_tiles (static); each tile finds its expert span from expert_offsets on device.

// Slot-major (decode/batched) — gather-by-expert-id BGMV, captures in decode graph
moe_lora_gather_bgmv(
    x:            bf16[n_slots, k_in],   // per (token,slot); down uses recomputed silu(gate)*up scratch
    base_out:     bf16[n_slots, n_out],
    indices_dev:  u32[n_slots],          // expert id per slot (DEVICE)
    seq_slot:     i32[num_tokens]|NULL,  // request-adapter per token (DEVICE), <0 => skip
    top_k:        u32,                   // flat_slot = token*top_k + slot
    a_table,b_table: u64[max_loras*ne], scale_table: f32[max_loras*ne],
    xa_scratch:   bf16[n_slots, max_rank],
    ne,n_out,k_in,max_rank: u32)
    // grid.y = n_slots; per row: s=seq_slot[token]; if s<0 return; e=indices_dev[row];
    //   idx=s*ne+e; if a_table[idx]==0 return; fold.
```

### 5.4 Overlay tables + hooks (Feature-2)

Tables (`overlay_tables.rs::TokenOverlaySet`, all `[max_loras]`): `embed_slot_map_table` (→`i32[vocab]`, -1 default), `embed_rows_table` (→`bf16[n,h]`), `lmhead_rows_table`, `lmhead_ids_table` (→`u32[n]`), `n_override_table` (`u32`, 0 ⇒ lm_head skip), `max_n_override` (grid.y). Kernels: `embed_overlay_routed_bf16`, `lmhead_overlay_routed_{bf16,f32}`, `embed_rowdiff_bf16` (`kernels/gb10/common/token_overlay.cu`).

```rust
fn apply_embed_overlay(&self, ids_dev: DevicePtr, seq_slot: DevicePtr, active: i32,
                       out: DevicePtr, num_tokens: u32, stream: u64) -> Result<()>;
// no-op guard: overlays.is_none() || kernel null || (seq_slot null && active<0)
// insert AFTER gather, BEFORE scale_embeddings (impl_a3.rs:35→37 single; embed_chunk.rs:68→90 batched)

fn apply_lmhead_overlay(&self, hidden: DevicePtr, seq_slot: DevicePtr, active: i32,
                        logits: DevicePtr, m: u32, is_fp32: bool, stream: u64) -> Result<()>;
// insert AFTER base projection, BEFORE softcap (impl_a3.rs:264→266 single; :182→184 batched;
//   decode_b2.rs:97 mixed-decode with REAL device seq_slot — the one true per-request batch)
```

---

## 6. Injection points — precise seams

**MoE down (grouped):** replace body of `apply_expert_lora_prefill_down` (`moe/lora.rs:133`); call site unchanged (`forward_prefill.rs:379`). Add the same call to bf16 (`forward_prefill_bf16.rs:242→246`) and fp8 (`forward_prefill_fp8.rs:587→591`) after removing the down-path portion of the `:52-65` guard.

**MoE router:** `apply_router_lora_prefill` (`moe/lora.rs:108`, called `forward_prefill.rs:224`) — extend to read `moe_row_adapter`; add the call to bf16/fp8/batched/decode gate seams (§4).

**MoE decode/batched:** insert `moe_lora_gather_bgmv` between fused silu+down and `moe_weighted_sum_blend` (`forward.rs:529`, `forward_batched.rs:414`); gate/up between fused gate+up and silu+down. Replace `reject_decode_lora` dispatch only when Incr 4 lands.

**Embed overlay:** `embed()` (`impl_a3.rs:31`) and `prefill_b_embed_chunk_at` (`embed_chunk.rs:31`) gain `active_slot: i32`; threaded from `seq.adapter_slot` at every caller (decode_a/b/a2, verify_*, impl_b1:499). Single-token needs the token id on device (reuse `buffers.token_ids()` head + `copy_h2d_async`).

**lm_head overlay:** `lm_head()` (`impl_a3.rs:192`), `lm_head_batched()` (`:72`), `mixed_final_norm_lm_head()` (`decode_b2.rs:26`) gain `active_slot: i32` (+ `seq_slot: DevicePtr` for mixed decode). The mixed-decode loop (`decode_b2.rs:60-97`) inlines per-token GEMV into `logits[i*v]` — the overlay hook goes **once after the loop** over `[padded_n, v]` with the device `seq_slot` (`AttnMetadataDev.seq_slot`, `layer.rs:111`); this is the site a global-active overlay would corrupt base rows, so `seq_slot` routing (`<0`⇒skip) is mandatory here.

**Note — true base opt-out gap (call out for implementer):** the attention path's `-1 ⇒ active` convention means a real row's `-1` currently *defers to the installed active adapter*, not base. For a genuinely mixed base+adapter batch, base requests need a sentinel distinct from `-1→active`. Two options: (a) resolve base to a reserved slot whose table cells are all `0` (so the existing `table[s]==0` skip fires), or (b) extend the convention to a second negative value. Option (a) reuses the existing kernel skip with zero kernel change and is preferred. This is the one place SOLID must diverge from the landed attention convention to be correct in mixed batches (per per-request-zero-overhead §Level-2 note and codispatch-varlen §D).

---

## 7. Build ordering & lifecycle

### 7.1 MoE table build
Built in `set_lora_weights` (`impl_lora.rs:58`) after the attention install (`:77`), from the resident pool, into load-fixed `[max_loras*num_experts]` tables. Cell `(slot,expert)` = 0 unless that adapter adapts that expert. Rotate/swap rewrites one slot stripe.

### 7.2 Guards to keep (loud, correct)
- `expert_pack::validate` master gate `REJECT[expert-lora-disabled]` (`expert_pack.rs:52`), rank cap `:63`.
- `set_lora_weights` gate/up bail (`moe/lora.rs:68-78`) until Incr 5.
- `reject_decode_lora` (`moe/lora.rs:173`) until Incr 4.
- `classify_key` reject arms (fused-expert, on-dense, oob, non-full-attn, `key.rs:110-154`); `parsers/lora.rs` `REJECT(target_parameters)` (`:188`) — the dominant real-world fused-MoE arm.
- Overlay `reject_pending_overlay` keeps its `lora_embedding_seen` Tier-2 bail (`overlay.rs:190`); drop only the `token-overlay-pending-apply` blanket bail (`audit.rs:78`) when the apply lands.

### 7.3 Overlay two-stage build (corrects `lora-moe-embed.md` §D.2/D.3)
The served embed/lm_head tables don't exist at loader time — build ordering: `load_lora_adapters` (`build.rs:107`, Step 1, `WeightStore` alive) precedes weight load (`:149`, Step 2) precedes `set_lora_weights` (`:636`, Step 8). Row-diff needs *both* the raw adapter tensors (Step 1) and the served embed table (Step 8), which never coexist. So:
- **Stage 1 (loader `load_lora_adapters_multi`):** upload RAW overlay tensors to scratch device buffers while `WeightStore` is alive; stash `Option<OverlayRaw>` per slot on `LoraWeights.overlay_raw`.
- **Stage 2 (`set_lora_weights`):** served embed/lm_head now exist → `embed_rowdiff` kernel → `build_override_set` → compact rows + `slot_map` + `ids_dev` → `TokenOverlaySet::from_slots` → free raw buffers.

New files (SPDX line 1, tests in own `*_tests.rs`, ≤500 LoC each): `lora/overlay_build.rs`, `lora/overlay_tables.rs`, `layers/ops/token_overlay.rs`, `model/token_overlay.rs`. Model fields: `overlays: Option<TokenOverlaySet>` (None ⇒ byte-identical off), `overlay_kernels: OverlayKernels` (resolved once via `try_kernel`, null-on-miss ⇒ silently inert).

### 7.4 Tied/untied lm_head
`tied = lm_head_weight.weight.0 == embed_tokens.weight.0 || lm_head_nvfp4.is_some() || lm_head_fp8.is_some()`. Tied ⇒ `lmhead_rows_table[k] = embed_rows_table[k]`. Untied + shipped `lm_head.token_adapter.*` ⇒ distinct `LmHeadOverlay`. Untied + not shipped ⇒ `n_override_table[k]=0` (embed-only correction). Quantized head recomputes `dot(hidden, override_row_bf16)` — override rows ARE the embed rows, correct.

### 7.5 Prefill-as-decode hazard
`use_decode_path = proc_count==1 && effective_seq_len_start>0` (`forward_layers.rs:109`). A warm prefix hit leaving one suffix token routes MoE-adapted prefill into the decode fold. Until Incr 4, either force `proc_count>1` or disable the decode shortcut when a MoE adapter is active on the request. Incr 4 resolves it structurally.

### 7.6 adapter_id closure (Incr 7)
`adapter_id_for_slot` (`types.rs:304`) must incorporate MoE + overlay identity, not just attention `(name,generation)`. In the single-bundle world a `generation` bump on re-stage (`loading.rs:432`) covers disk swaps; add an invariant that a MoE-only decoupled rotate cannot leave `adapter_id` unchanged (else a cached prefix computed under the old MoE delta is reused with the new one).

---

## 8. Zero-overhead + graph-safety proof sketch

### 8.1 Zero overhead when no adapter installed (Level 1)
Every apply site is behind `if let Some(ref lw) = self.lora` / `if self.overlays.is_some()` / `ctx.moe_row_adapter.is_some()`. When `None`: exactly one enum discriminant test, no kernel, no buffer touch. Buffer builders short-circuit: `upload_seq_slot*` return `DevicePtr(0)` when `self.lora == None` (`impl_b1.rs:149,177`); `moe_row_adapter` is `None`; `overlays` is `None`. The whole pipeline returns null sentinels and every fold site is untouched — **byte-identical to a no-LoRA build**. Proven by the landed attention path (per-request-zero-overhead §Level-1); the new surfaces replicate the exact predicate.

### 8.2 Zero overhead when installed but request opts out (Level 2)
Per-row skip is a predicated kernel early-return, never a host branch:
- Attention: `seq_slot[row] < 0 ⇒ return` and `a_table[s]==0 ⇒ return` (`lora_bgmv.cu:63,65,153,155`).
- MoE grouped: `sorted_row_adapter[row] < 0 ⇒ return`; `a_table[slot*ne+e]==0 ⇒ return`.
- MoE slot-major: `seq_slot[token] < 0 ⇒ return`; table-cell 0 ⇒ return.
- Overlay: `active<0 || seq_slot[r]<0 ⇒ skip`; `slot_map[id]<0 ⇒ skip`; `j>=n_override[s] ⇒ skip`.
A base row costs one block-schedule + one i32 load + one predicated return per (layer×module×kernel): zero math, zero memory traffic, folds nothing. A base request in a mixed batch pays nothing and corrupts nothing (§6 base-sentinel note ensures base maps to a skip, not to `active`).

### 8.3 Graph-safety (capture ⇒ replay correctness)
Capture is `CU_STREAM_CAPTURE_MODE_RELAXED` on a non-default stream (`gpu_impl.rs:295-329`). What breaks it: host sync (`cuStreamSynchronize`), D2H copy (`copy_d2h`), and host-driven data-dependent launch count. The current MoE fold hits all three via `copy_d2h(expert_offsets)` + host `for w in work` loop (`moe/lora.rs:149`, `expert_apply.rs:147-157`).

SOLID eliminates all three:
1. **No D2H:** `expert_offsets`/`sorted_expert_ids`/`indices_dev`/`seq_slot` are consumed as *device kernel args*. Proof this is already legal: the base grouped GEMM reads `expert_offsets` as a kernel arg and never D2Hs it (`forward_prefill_routed.rs:37,150`, ~15 kernel calls); the only base-path D2H of offsets is the opt-in `ATLAS_MOE_PREFILL_EXACT_TILES` grid-sizing read, itself explicitly `!ctx.graph_capture`-gated (`:71-75`) — i.e. even the base path treats an offsets D2H as capture-incompatible.
2. **Static launch shape:** grid sized to `worst_case_m_tiles`/`num_experts` (compile/config constants), not to a device-read count. Each tile derives its expert span from `expert_offsets` *inside* the kernel. Replay issues an identical launch sequence.
3. **Pointer/value-stable args:** all routing tables are load-fixed addresses; scratch (`xa`/`delta`) is fixed-address arena; only `seq_slot`/`moe_row_adapter` *contents* change per step, at fixed addresses re-uploaded before `begin_capture` (same phasing as `positions`/`block_table`). This is exactly why `apply_lora_bgmv` captures inside the decode graph (`lora_delta.rs:260-264`); the MoE folds inherit it.

Therefore the grouped fold captures in the prefill region (Incr 2) and the gather-BGMV captures in the decode graph (Incr 4).

### 8.4 Codispatch / varlen / prefix-cache safety
- **Varlen/packed:** the per-row adapter map is built from `cu_seqlens_host` (`layer.rs:128`, the packing SSOT = Σ proc_count), NOT `b*chunk_len`, so unequal per-stream lengths and partial-prefix-cache hits align correctly (codispatch-varlen §D.1). The down fold gathers the adapter through `token_to_perm` — the same permutation the activations took — so sorted-row identity is exact.
- **Codispatch:** a mixed co-dispatch batch folds each row with its own adapter (or skips if base). No batch-level adapter taint: the base bails at `forward_prefill.rs:52-65` become per-row skips, so one adapter'd request no longer forces the fold on the shared batch (codispatch-varlen §D.5). (Attention LoRA is currently NOT threaded into the kernel-batched codispatch path at all — `batch_kernel.rs:383-397` sets `attn_metadata:None`; wiring `moe_row_adapter` + `seq_slot` here is part of Incr 0/3.)
- **Prefix-cache:** KV radix roots and SSM snapshots are already `adapter_id`-keyed and isolated (radix_tree `inner.rs:67`, `radix_tree.rs:33-44`; verified by `radix_tree/tests/adapter.rs`). The cached-prefix skip floor (`layer_kv_write_start`, `forward_layers.rs:118-124`) means the fold is neither re-applied nor lost on reused positions — correct for attention today, and correct for MoE once `adapter_id` closes over MoE identity (Incr 7). Base prefix (`adapter_id==0`) is never reused by an adapted suffix.

---

## 9. Test & oracle plan

### 9.1 Host-only unit tests (no GPU)
- `build_moe_row_adapter_host`: varlen `cu_seqlens_host`, base sentinel, active-defer — assert per-row map matches by-hand spans (mirrors `slot_math.rs` tests).
- `expert_delta_workitems` (`expert_apply.rs:51`): offsets `[0,3,3,10,...]`, adapted `{0,1,5,255}` — expert 1 (zero rows) and 256 (oob) skipped; byte ranges for 0/5/255 exact.
- `classify_key` matrix (`key.rs`): every accept/reject arm — `mlp.gate`→Router, `experts.5.down_proj`→Expert{5,Down}, `experts.5.gate_up_proj`→fused-reject, non-full-attn→reject, oob→reject.
- `PeftAdapterConfig::scaling` parity (`parsers/lora.rs:82`): `alpha/r` and `alpha/√r` (rslora) golden shared with `reference_deltas.py:73`.
- Overlay compaction: `OverlayRaw`→`EmbedOverlay` with host row-diff stub; delta-wins via `override_source`; full-save (`trainable=∅`); tied vs shipped-lmhead table wiring.

### 9.2 Delta-parity oracle (Tier A, no GPU)
`scripts/reference_deltas.py` already emits `scaling·(B@A)` per module and canonicalizes to `model.layers.L.mlp.experts.E.down_proj` / `mlp.gate` unchanged. Rust diffs loaded A/B/scale product vs `{path}.delta` at ≤1e-2 rel. This is the buffer-level correctness gate independent of forward wiring.

### 9.3 End-to-end prefill-logit oracle (Tier B, GPU serve)
Diff surface = legacy `/v1/completions` loglikelihood (`echo:true, logprobs:5, max_tokens:0`; per-request `adapter` field routes via `resolve_request_adapter_slot`, `api/completions.rs:190-206`). New `scripts/moe_lora_oracle.py`:
1. Reference: `PeftModel.from_pretrained` on Qwen3.6-35B-A3B; capture prefill `token_logprobs` + top-5 with adapter ON and (via `disable_adapter()`) OFF.
2. Atlas: same prompts with/without `adapter`.
3. Assert `max_abs(atlas_adapter − peft_adapter) ≤ tol` (start 5e-2), AND the **base-vs-adapter cross-check**: `sign(atlas_adapter − atlas_base) == sign(peft_adapter − peft_base)` per position — the single test that catches a *silently-inert* fold or wrong `expert_offsets` mapping (the two most likely bugs).
Caveats as asserts: quant-noise floor (calibrate NVFP4-vs-bf16 base run first); test **router-only and down-only separately** before combined (a router delta flips top-k routing → discontinuous logits, can mask a down bug); `max_tokens:1` with a MoE adapter must 500/bail loudly (decode not yet folded) — that is itself a test.

### 9.4 Synthetic adapter generator
Extend `scripts/gen_lora_adapter.py` with `--variant moe-experts`, geometry from `config.json` (not the NVFP4 header), full-attention layers via `text_config.layer_types` (NOT hardcoded `[3,7,...]`). Emit unfused per-expert `down_proj` + `mlp.gate` (rare in the wild — the generator is load-bearing because the fused 35B only admits `target_parameters`, which Atlas rejects). Coverage knobs: `--experts 0,1,5,255` (gap + last-valid), `--rank {8,32}` (cap test), `--proj {down,gate}` (gate drives the `set_lora_weights` bail), `--bad-expert 256` (oob reject). `adapter_config.json` via `LoraConfig.save_pretrained`.

### 9.5 Graph-capture regression
A decode/prefill capture smoke that fails if any fold path reintroduces a D2H/sync under capture: run the grouped fold inside a captured prefill region and the gather-BGMV inside a captured decode graph, assert `cuGraphInstantiate` succeeds and replay logits match eager.

### 9.6 Prefix-cache correctness
Adapter-isolation tests (`radix_tree/tests/adapter.rs`) extended to assert a MoE-adapted prefix and a base prefix over identical tokens get disjoint blocks (post Incr 7); and that a warm-hit single-suffix-token MoE request does NOT bail (post Incr 4).

---

## 10. Risk register

| Risk | Mitigation |
|---|---|
| Grouped fold grid over-provisioned (`worst_case_m_tiles`) wastes launches on empty experts | tiles early-return on empty span (`expert_offsets[e]==expert_offsets[e+1]`); same cost model as base grouped GEMM |
| Fused-SiLU recompute in decode fold doubles silu cost | recompute is `moe_inter`-wide per adapted slot only; base rows skip; acceptable vs splitting the fused kernel |
| Base-sentinel divergence from attention `-1→active` | §6 option (a): reserve an all-zero table slot for base; reuses existing `table[s]==0` skip, zero kernel change |
| `adapter_id` not yet closing over MoE/overlay → stale prefix reuse under decoupled rotate | Incr 7 + invariant test; until then MoE ships in the single bundle so `generation` covers it |
| Codispatch attention LoRA currently unwired (`attn_metadata:None`) | Incr 0/3 wires `seq_slot`+`moe_row_adapter` into `batch_kernel.rs:383-397`; until then codispatch is base-only (loud, not silent) |

---

## Appendix — anchor index
Capture: `gpu_impl.rs:295/306/329`; `decode_a.rs:208-215,239-356,402`. Attention ref: `lora_delta.rs:41-56,260-281`; `lora_bgmv.cu:50-65,139-155`; `impl_b1.rs:140,168,187-194`; `slot_math.rs:50,83,98`. MoE fold: `moe/lora.rs:36-48,108,133,149,173`; `expert_apply.rs:51,134,147-157`. Sorted producers: `forward_prefill.rs:224,311-327,379`; `forward_prefill_routed.rs:34,37,71-75,150,317-322`. bf16/fp8: `forward_prefill_bf16.rs:161-177,219,242,246`; `forward_prefill_fp8.rs:301-313,483,587,591`; guard `forward_prefill.rs:52-65`. Slot-major: `forward_batched.rs:12,123,414`; `forward.rs:28,35,92,529`. Packing/varlen: `batch_kernel.rs:149-310,383-397`; `batched_layer.rs:250-259`; `layer.rs:128,210-253`; `eligible.rs:128,178`. Prefix cache: `prefix_cache.rs:138-243`; `radix_tree/inner.rs:58-125`; `radix_tree.rs:33-44`; `types.rs:304-314`; `loading.rs:432`; `forward_layers.rs:109,118-124`. Overlay: `token_overlay.cu`; `overlay.rs:186-266`; `audit.rs:52-78,125`; `loading.rs:159,233-282,316,337`; `build.rs:107,149,544,636`; `impl_lora.rs:58-101`; `impl_a3.rs:31-38,192-271`; `embed_chunk.rs:59-90`; `decode_b2.rs:26,60-97,101-156`. Tests/oracle: `gen_lora_adapter.py`, `reference_deltas.py:73,92`, new `moe_lora_oracle.py`; `key.rs:108-154`; `parsers/lora.rs:159-212`; `completions.rs`.
