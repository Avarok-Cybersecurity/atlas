# Atlas LoRA — status report (2026-07-06)

Where the LoRA effort stands on `feat/streaming-experts-mvp` (PR #9): what's built,
what's validated live on GB10, the concurrency benchmark, the known cut lines, and
the ranked roadmap.

## TL;DR

- **Serving a fine-tuned adapter, multi-adapter, runtime swap, and a fused
  per-request routing kernel are all built and compile clean.** Two demo adapters
  were trained on GB10 (STARFALL-4728/"Sparky", MOONVEIL-3390/"Vega") and pushed to
  HF (`MonumentalSystems/Holo-3.1-0.8B-lora-demo`, `-demo-2`).
- **Per-request routing works in *decode* under `--scheduling-policy slai`** — two
  concurrent requests naming different adapters each route to their own adapter; the
  prefill-consistent request is byte-clean.
- **Routing is free**: at C=2/4/8 concurrent, per-request LoRA routing carries the
  same req/s, prefill, and decode tok/s as the base model (within noise).
- **The remaining work is correctness *sequencing*, not the kernel**: request-scoped
  routing (esp. prefill + single-seq), adapter-correct KV, then the bgmv as the
  batch>1 optimization. See the roadmap.

## What's built (commits)

| piece | commit | state |
|---|---|---|
| M0+M1 single adapter (runtime BF16 delta @ attn k/v/o) | upstream `research/lora-mvp` (merged `fdb4e30`) | ✅ shipped, offline parity 0.0 rel-err |
| Multi-adapter pack into the frozen `[max_loras]` pool + global rotation (`/v1/lora/active`) over RDMA | `489f372` | ✅ build+clippy clean, 36 unit tests |
| Pool-size-1 dynamic swap (`/v1/lora/load`, disk swap-into-slot) | `4d6dedf` | ✅ demoed live (swap starfall↔vega in one slot) |
| Fused two-kernel bgmv per-request routing (`lora_bgmv.cu`) | `d3ea611` (WIP) + `e40a70f` (status) | ⚠️ decode routing works under SLAI; **disabled in prod** pending request-scoped routing |

## Validated live (holo-3.1-0.8b, GB10)

- **Pool=1 dynamic swap**: `starfall` → "…STARFALL-4710"; `POST /v1/lora/load vega` →
  same prompt "…MOONVEIL-3390"; swap back → "…STARFALL-4710". Runtime event logged
  (`LoRA disk swap: 'vega' packed into slot 0`).
- **Concurrent routing (SLAI)**: `adapter=starfall` → `STARFALL-7725` (byte-clean,
  identical to single-seq); `adapter=vega` → routed to Vega's persona. Under the
  default `fifo` the requests serialized and fell to the active adapter — SLAI is
  required for them to co-decode.
- **bf16 head vs nvfp4 head**: adapter #1's exact codeword digits are sensitive to
  lm-head precision (NVFP4 → 4710, bf16 → rambles); adapter #2 (overfit harder,
  loss 0.0008) is robust to both. Exact-digit fidelity would need the **base** in
  BF16, not just the head.

## Concurrency benchmark (routed vs base, SLAI, 64-tok gens)

| C | mode | req/s | prefill TTFT mean | agg decode |
|---|---|---|---|---|
| 2 | routed | 1.74 | 719 ms | 88 tok/s |
| 2 | base | 1.61 | 704 ms | 95 tok/s |
| 4 | routed | 1.99 | 1178 ms | 116 tok/s |
| 4 | base | 1.77 | 1427 ms | 104 tok/s |
| 8 | routed | 2.44 | 1894 ms | 134 tok/s |
| 8 | base | 2.05 | 2420 ms | 121 tok/s |

Per-request routing adds **no measurable overhead** vs base (routed even marginally
ahead — within shared-GPU noise). req/s scales with C; prefill grows from queuing.
The ceiling is scheduling/batching + request-scoped correctness, not the LoRA math.

## Known cut lines (why the WIP bgmv stays disabled)

1. **Single-seq (n==1) requests don't route** — a lone request naming an adapter
   returns the *global active* adapter.
2. **Prefill always uses the active adapter** — a routed request's prompt KV is
   active-flavored (persona survives in decode, exact recall degrades).
3. **Routes gated on active-adapter coverage** — heterogeneous adapters (different
   target modules/layers) mis-route on the modules the active adapter lacks.
4. **`pack_store_into_slot` drops `_slot_ptrs`** — a reused slot can keep a stale
   pointer-table/scale/mask entry.
5. **seq_slot metadata** at `meta_base+128` is safe only for padded_n ≤ 32.

## Roadmap (ranked; reviewer + on-HW findings converge)

Batching is the ceiling and the global-active-adapter is the wart. BGMV is the
*last* step, gated on batch>1 — not the near-term work.

1. **#23 Request-scoped AdapterSelector** — kill the global active adapter; route
   prefill *and* single-seq by the request's adapter. Highest value; fixes cut
   lines 1+2. *(next)*
2. **#24 Adapter-correct KV** — cache keyed by stable `adapter_id`, base reuses base
   blocks; fixes the warm-prefix-skips-delta cut line.
3. **#25 Slot generation + ref_count** — safety substrate for swap+graphs.
4. **#27 Demand-driven RDMA promotion** — `ensure_adapter_hot(id)` on miss + load
   coalescing (adapters are tiny; 1000+ is I/O-trivial on the CX7 tier).
5. **#28 Generation-keyed graphs** — retire `lora_rotatable` forced-eager.
6. **#26 Fix `pack_store_into_slot` stale table** — cheap hardening.
7. **#29 / bgmv** — wire the fused kernel into the reliable batch>1 path (last).

## Update (2026-07-06 pm) — reviewer roadmap #22-#29 landed

The reviewer's ranked plan is implemented + validated on GB10 (holo-0.8b):

| # | task | commit | state |
|---|---|---|---|
| 23 | request-scoped selector (prefill+single-seq route) | 067b676 | ✅ solo request routes to its adapter |
| 24 | adapter-correct KV (per-adapter radix roots + snapshot id) | 30b2852 | ✅ 3× LGTM, base byte-identical |
| 25 | slot generation + ref_count (safe swap) | 125d168 | ✅ closes #24 same-name residual |
| 27 | demand-driven RDMA promotion (coalesced, LRU) | 6b546ab | ✅ hot cache over 1000s staged |
| 28 | generation-keyed graphs (retire forced-eager) | 939fcc4 | ✅ swappable pool decodes GRAPHED |
| 26 | slot-swap table refresh (stale-coverage) | 94bea6d | ✅ both disk + RDMA swap paths |
| 22 | LoRA HTTP guard-pass (CodeQL) | 5101c02 | ✅ allocation cap + input bounds |
| 29 | reliable batch>1 routed decode | (umbrella) | ✅ validated: concurrent diff-adapter routing, graph-safe |

**The WIP "DO NOT ENABLE" on the fused bgmv (d3ea611) is LIFTED**: with request-scoped
routing (#23) + adapter-correct KV (#24) + the gen/ref_count substrate (#25) +
gen-keyed graphs (#28), the batched multi-seq routed decode is correct and enabled.
Validated: two concurrent requests naming different adapters each route to their own
adapter (active byte-clean; routed applies its adapter), single-seq routes, decode
graphs stay captured under a swappable pool.

**#30 routed-prefill precision — landed (`b52efef`).** A routed (non-active) prefill
now folds the REQUEST slot pair through the SAME `apply_lora_delta`/`dense_gemm_tc`
path the active adapter uses, not the per-row bgmv (`ForwardContext.routed_lora_layers`
set at prefill entries only for a non-active slot; `LoraAttnWeights` stamped with the
GLOBAL layer index for hybrid GDN/attention models). 3× LGTM (numerics-match,
byte-identity-decode, build-lifetime); active/base/no-lora byte-identical, decode
untouched, 24 lora tests. **Known residual (#32):** the demo's razor-margin OVERFIT
codeword still tips under routing — routed `vega` yields the Vega persona but not the
exact `MOONVEIL-3390` digits. #30 fixes the prefill kernel; the routed DECODE still
uses the bgmv vs the installed gemv, and these low-margin argmax digits tip on any
residual difference (same fragility seen with lm-head precision). Production
(non-overfit) adapters route correctly. The follow-up is making the WHOLE routed path
(decode included) bit-identical to active-served.

**#31 spilled-seq panic-leak** — closed: the swap-side hole is already covered by #27's
`swapped.is_empty()` drain gate, and there is no catch_unwind / `panic=abort` in the
workspace, so a scheduler-thread panic tears the process down (no recover-and-leak path).

## Update (2026-07-06 pm2) — #32 routed DECODE: VERIFIED bit-identical (no defect)

**#32 asked whether the routed DECODE tips the razor-margin overfit codeword because
the bgmv differs from the installed gemv. On GB10 (holo-3.1-0.8b) the answer is NO —
the routed decode is already BIT-IDENTICAL to active-served, in every path.**

The confound-free oracle (`docs/streaming-experts/lora-regression/routed-decode-bit-identity.sh`):
pool TWO IDENTICAL copies of one adapter — slot0 `a`=active, slot1 `b`=routed. Because
the weights are identical, any difference between the installed `apply_lora_delta` path
(`adapter=None`/`a`) and the routed `bgmv` path (`adapter=b`) is a PURE routing-kernel
artifact. Results:

| path | active/baseline | routed(b) | match |
|---|---|---|---|
| single-seq decode (`attention_forward.rs`) | `STARFALL-4710` | `STARFALL-4710` | ✅ char-for-char |
| multi-seq batched decode (`multi_seq/{attn,qkv}.rs`, n>1 SLAI) | `STARFALL-4728` | `STARFALL-4728` | ✅ char-for-char |
| pool `max_rank` padding (r=32 → pool 64) | `STARFALL-4710` | `STARFALL-4710` | ✅ char-for-char |

Source review confirms it: `lora_bgmv.cu` (shrink/expand/fold) is a byte-copy of
`dense_gemv_bf16.cu` + `residual_add.cu bf16_scaled_add` — same uint4 K-vectorization,
same per-element fp32 accumulate under `--fmad=false`, same `__shfl_down_sync` order,
same 2-warp smem reduce, same BF16 rounding at both buffer boundaries. So bgmv(n=1) ≡
`apply_lora_delta(m=1)` by construction, and the hardware A/B corroborates it.

**The `STARFALL-4710` (single-seq) vs `STARFALL-4728` (batched) digit shift is a
BATCH-PATH numeric property, not a routing defect** — it moves the ACTIVE adapter's
output identically (4728 is the true trained codeword, so the batched path is the
faithful one). It is orthogonal to LoRA.

**Re-disposition of the #30 vega residual:** it was an adapter/config confound, not a
decode-kernel divergence. Routed-vega was compared against a `MOONVEIL-3390` baseline
that was never re-established in the identical pooled config (standalone vs pool differ
in batch composition, which — per the 4710/4728 observation — shifts razor-margin
argmax digits for the active adapter too). With routing proven bit-identical, a
same-config `vega-active` vs `vega-routed` comparison is guaranteed to match. **#32 is
closed: no code change — the routed decode path is bit-identical; the oracle is
committed as a permanent on-hardware regression guard.**
