# PRD: Atlas K3 Architecture Bring-Up

**Working title:** Atlas K3 Architecture Bring-Up (lab → rented Hopper+)
**Owner:** Tom
**Engines:** Atlas (primary), vLLM (reference / bake-off)
**Upstream:** `Avarok-Cybersecurity/atlas` (default branch `main`, AGPL-3.0 + CLA)
**Status:** Lab-first. Official weights and rental silicon are a soak, not a debugger.

---

## Binding constraint (read this first)

Architecture is completed at home. Rental silicon is used only after the graph, loaders, hybrid cache, and dual-node comms are correct.

| Lab host | What it is | What it is for |
| --- | --- | --- |
| **spark1** | NVIDIA DGX Spark, GB10, SM121, 128 GB LPDDR5x UMA (~120 GB usable), CX7 | Head / rank 0. Single-node twins, dummy, MXFP4 fixtures, bake-off API on `:8888`, NCCL master |
| **spark2** | Same class box on the same network | Worker / rank 1. Joins only for dual-node TP/EP (C7) and fabric health. No public client API (`:8889` if anything listens) |
| Workstation **5090** | RTX 5090, SM120, ~32 GB | Correctness GPU only. Twins, CPU+GPU numeric diffs, HF reference. Not a product SKU |

Two Sparks together are ~240 GB unified. Official `moonshotai/Kimi-K3` MXFP4 is **~1.561 TB across 96 shards**. spark1+spark2 cannot hold official weights. There is **no “full K3 on lab” milestone**.

**Hard rental gate:** do not book, reserve, or start an 8×B300 / 16×H200 / 16×B200 node until C0–C7 are green on lab hardware and `docs/k3/RENTAL.md` is filled. Burning Hopper time on a wrong KDA state layout is the failure mode this PRD exists to prevent.

Work is tracked as **one draft umbrella PR** on Atlas (`wip/k3-bringup`). That PR is a working log. It is never `/stamp`’d, `/seal`’d, or squash-merged as-is. Winning slices are extracted later into atomic certified PRs.

**RST (Rapid Software Testing) is required at every S-rung and C-gate.** See `docs/k3/RST.md`. A green checkbox without a session sheet (charter, named oracle, known-bad that the instrument actually failed) is a check, not a test. Independent charters run in parallel when they do not share a GPU.

---

## 1. Problem

Atlas does not serve Kimi K3. K3 is a 2.8T hybrid MoE (KDA + gated MLA + AttnRes + Stable LatentMoE + native MXFP4). Official serving needs ~1.56 TB of weights and an 8×B300 or 16×H200/B200-class box.

You cannot hold the official checkpoint on a 5090 or two Sparks. You can implement and verify the architecture, loaders, hybrid cache, and dual-node comms locally, then rent silicon only for a full-weight soak and a fair same-box vLLM comparison.

## 2. Goal

Ship an Atlas `(hardware, model, quant)` target for Kimi K3 that:

1. Correctly executes the K3 graph (not a “close enough” GDN/Mamba stand-in).
2. Matches a reference implementation token-for-token on a tiny twin.
3. Runs dual-Spark TP/EP on a mid-size dummy across **spark1 + spark2**.
4. On rented Hopper/Blackwell, loads official MXFP4 weights and reports TTFT / tok/s / ITL vs the same-box vLLM K3 image.

Non-goals for this PRD: beating published GB300-NVL72 marketing numbers; vision-first multimodal parity; 1M-context production SLA; training; landing the umbrella PR onto `main`.

## 3. Users and use cases

| User | Need |
| --- | --- |
| You (engine author) | Deterministic bring-up path that does not burn rental budget on graph bugs; one Atlas PR that is the lab notebook |
| Atlas operators (later) | `spark serve moonshotai/Kimi-K3` on GB10 clusters and enterprise nodes |
| Bake-off reader | Same-prompt, same-hardware Atlas vs vLLM table |

## 4. In scope / out of scope

### In scope

* Text backbone: KDA, gated MLA, AttnRes, Stable LatentMoE, SiTU-GLU, shared + routed experts, hybrid KV/state cache, prefix caching
* Config + safetensors (and MXFP4 packed) weight map
* OpenAI-compatible `/v1/chat/completions` + `/v1/completions` for bake-off
* Reasoning / tool parsers only if required for a valid chat template (can stub)
* Dual-Spark NCCL/RoCE path on **spark1 + spark2**
* Golden tests vs HF / vLLM eager on tiny models
* Lab benchmark harness reused on rental boxes
* One draft umbrella PR that holds the pile of code + comments until slices are extracted

### Out of scope (v1)

* MoonViT-V2 production path (optional stub: reject images)
* DSpark / DFlash until text decode is correct
* Prefill/decode disaggregation, Mooncake, 1M ctx
* 5090 as a product SKU (5090 is a correctness GPU only)
* GGUF as the production weight format
* AMD / Ascend
* Merging the umbrella branch to `main`

---

## 5. Lab inventory and hardware truth table

### 5.1 Named lab hosts

Treat these hostnames as first-class. Do not invent IPs, RoCE iface names, or `NCCL_IB_HCA` pins in this PRD — those change with which QSFP is cabled. Freeze them on P0 day 1 in `docs/k3/LAB.md` from `hostname`, `ip -br addr`, and `ibdev2netdev` on the live boxes.

| Hostname | Role | Ports | Notes |
| --- | --- | --- | --- |
| **spark1** | Head / rank 0 | Atlas serve `:8888`, NCCL master `:29500` | Primary workhorse. All single-node K3 work runs here first |
| **spark2** | Worker / rank 1 | `:8889` if a process listens; not the client endpoint | Joins only for C7 and fabric health |
| 5090 workstation | Correctness GPU | local | SM120. See §9 day-1 launch test |

Per Spark (GB10 / DGX Spark class):

* Blackwell GPU, compute capability **12.1 (`sm_121`)**, 6,144 CUDA cores
* 128 GB LPDDR5x coherent UMA, Atlas-measured usable ~119.7 GB (`free -h` / CUDA global mem, not a discrete-VRAM `nvidia-smi` number)
* ~273 GB/s memory bandwidth
* ConnectX-7, dual QSFP, 200 Gbps RoCE between spark1 ↔ spark2
* Combined usable ~240 GB

### 5.2 What fits where

| Machine | Memory | Role |
| --- | --- | --- |
| 5090 | ~32 GB | Unit tests, 49M shape twin, 0.40B twin if it fits, CPU+GPU numeric diffs |
| **spark1** | ~120 GB UMA | Twins, production-width dummy, MXFP4 fixtures, single-node bake-off |
| **spark1 + spark2** | ~240 GB UMA + CX7 | Dual-node TP=2 / EP, mid dummy, Atlas vs vLLM on non-K3 and dummy-K3 |
| Rented 8× B300 | ~2.30 TB | Official MXFP4 load, C=1 and C=8 bake-off (preferred single-node) |
| Rented 16× H200 | ~2.26 TB | Official MXFP4 load (fits, tighter headroom) |
| Rented 16× B200 | ~2.88 TB | Official MXFP4 load, two-node |
| 16× Spark | ~1.9 TB | Only if you later want a Spark-native full K3; **not** this PRD’s rental default |

Weights-only fit check against the ~1.561 TB official checkpoint (before KV / activations / graphs):

| Config | Aggregate HBM | Official K3 |
| --- | --- | --- |
| 8× H200 (8×141 GB) | 1,128 GB | Does not fit |
| 8× B200 (8×180–192 GB) | 1,440–1,536 GB | Does not fit (short on the order of 100+ GB) |
| 16× H200 | 2,256 GB | Fits |
| 8× B300 (8×288 GB) | 2,304 GB | Fits — preferred rental |
| 16× B200 | 2,880 GB | Fits |
| spark1+spark2 | ~240 GB | Does not fit |

Two Sparks cannot load official K3. Do not plan a “full model on lab” milestone.

### 5.3 Dual-Spark fabric (P0, before any K3 dummy TP)

Prove the cable, not the model.

1. Passwordless SSH `spark1 ↔ spark2`.
2. On both boxes: `ibdev2netdev`. Record which RoCE twins are **Up**. GB10 exposes four HCAs; two are typically Down. Atlas GB10 guides often pin `enp1s0f0np0`; other Spark labs pin `enp1s0f1np1`. Use whichever QSFP is actually cabled.
3. Pin `NCCL_SOCKET_IFNAME` to the Up twin and `NCCL_IB_HCA` to the matching HCA. Unpinned NCCL picks a dead HCA and dies with an unhandled system error.
4. NCCL all_gather spark1 ↔ spark2.
5. Launch an **already-shipping** Atlas MoE (not K3) with the dual-node helper (`scripts/start-ep2.sh` or current equivalent) using `HEAD_IP=<spark1>` `WORKER_IP=<spark2>`, and produce a comparison JSONL vs vLLM on the same two boxes.

GB10 NCCL defaults that have been required on this class of box: `NCCL_NVLS_ENABLE=0`, `NCCL_NET_GDR_LEVEL=0`, `NCCL_NET_GDR_C2C=0`, `NCCL_DMABUF_ENABLE=0`, `NCCL_PROTO=Simple`, `NCCL_ALGO=Ring`. Confirm against current Atlas GB10 docs on P0 day 1 and freeze the working set in `docs/k3/LAB.md`.

---

## 6. Architecture requirements (must implement)

Preserve these invariants from `moonshotai/Kimi-K3` (scale may shrink in twins; shape may not).

| Block | Production K3 | Twin may shrink | Must keep |
| --- | --- | --- | --- |
| Depth / mix | 93 layers (1 dense + 92 MoE); 69 KDA + 24 gated MLA as 3×KDA + 1×MLA, plus a trailing MLA so the last layer is always global | Layer count | Interleaved KDA and MLA; at least one KDA run and one MLA layer |
| Hidden / heads | 7168, 96 heads | Yes | Separate KDA vs MLA modules |
| MoE | 896 routed, top-16, 2 shared, latent 3584, expert FFN 3072 | Expert count, dims | Latent down → experts → up |
| AttnRes | Block residual mixing (block size 12) | Block size | Cross-layer residual attention, not a vanilla residual add |
| Act | SiTU-GLU | Betas may stay fixed (prod β=4, β_lin=25) | Not vanilla SwiGLU |
| Cache | KDA recurrent + conv state and paged MLA KV on one prefix | State sizes | Hybrid cache manager |
| Weights | MXFP4 routed experts (E8M0 block scales), MXFP8 activations, higher prec on attn / norms / routers / shared experts / head / embeddings | BF16/FP8 OK on twins | Official path is MXFP4 |
| Pos | NoPE / K3 positional scheme from official `config.json` | — | Match config, not generic RoPE-only |
| Chat | Programmatic template + reasoning/tool regions | Can stub | Needed before “product serve” |
| Dense | Layer 0 only (`first_k_dense_replace=1`) | — | First layer dense in prod and in the 0.40B twin |

Released-config details the twins must preserve even when dims shrink (from official `config.json`, not the announcement deck):

* MLA: `mla_use_nope=true`, `mla_use_output_gate=true`; prod ranks `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`
* KDA: `use_full_rank_gate=true`, `gate_lower_bound=-5.0`; decay gate stays low-rank `f_a` / `f_b`
* AttnRes tensors: `self_attention_res_proj`, `mlp_res_proj`, `*_res_norm`, model-level `output_attn_res_proj`, `attn_res_block_size=12`
* Latent MoE wrappers: `routed_expert_{down,up}_proj` + `routed_expert_norm`
* SiTU: `hidden_act=situ`, `activation_situ_beta=4.0`, `activation_situ_linear_beta=25.0`
* First layer dense (`first_k_dense_replace` pattern)

Do not reuse GDN / Mamba-2 / Qwen3-Next kernels as “KDA.” Treat KDA as a new attention backend with its own recurrent + conv state layout.

Reuse, do not rewrite:

* MiniMax / DeepSeek-V4 MoE dispatch and EP
* DeepSeek-V4 MXFP4 E8M0 loader / GEMM (Atlas already has this path)
* Existing MLA KV paging where shapes match
* Existing dual-Spark NCCL pin + bake-off harness style

Write new: KDA backend, AttnRes, SiTU-GLU, latent project, hybrid cache manager, `kimi_k3` weight loader, chat-template stub.

Pinned twins (lab only):

* **Token-exact reference:** `inference-optimization/Kimi-K3-0.40B` — keeps 3:1 KDA:MLA (layers 0–2 KDA, 3 MLA, 4–6 KDA, 7 MLA), AttnRes block 4, 8 routed / top-2 / 1 shared. Architecture gold, not quality gold. `custom_code`; generate via `model.language_model.generate`.
* **Shape/graph only:** `cneuralnetwork/smol-kimi-k3` (49M, 8k BPE, TinyStories). Do **not** use it for C1 token-exact vs the official K3 tokenizer.
* **Official weights:** `moonshotai/Kimi-K3` — rental only. Never download 1.56 TB in lab.

---

## 7. Success metrics

### 7.1 Correctness (this is the rental gate)

Do not book rental hardware if any of C0–C7 is red.

| ID | Criterion | Pass |
| --- | --- | --- |
| **C0** | Official `config.json` parsed. `MODEL.toml` + factory + weight-name map cover every production tensor class without downloading shards | Loader dry-run lists the 96-shard map and refuses missing keys |
| **C1** | `Kimi-K3-0.40B` greedy decode matches HF reference for 128 tokens × 8 prompts | Exact token match. 49M is C1-shape only (graph runs; no official-tokenizer match) |
| **C2** | Prefill-then-decode equals full-prefill logits on the twin | atol/rtol written in the test file (not “agreed later”) |
| **C3** | Prefix-cache hit (shared prompt) equals no-cache decode | Exact tokens |
| **C4** | MLA layer KV and KDA state advance on the same logical positions, including after a prefix hit | Asserted in unit tests |
| **C5** | AttnRes output matches reference block for a fixture residual stack | Max abs err bound in the test file |
| **C6** | LatentMoE: router top-k + expert mix matches reference for frozen gate scores | Exact expert IDs + close outputs |
| **C7** | Production-width dummy, TP=2 on spark1+spark2, matches single-GPU dummy tokens on spark1 | Exact |

### 7.2 Performance (lab, not headline)

On spark1+spark2, dummy or proxy models: report TTFT p50/p99, ITL p50, tok/s at C=1 and C=4. Label the dual-Spark dummy table as **comms-bound**. Do not tune K3 kernels against that number. No pass/fail vs vLLM until rental.

### 7.3 Performance (rental soak — only after C0–C7)

Same node, official MXFP4, identical prompt set (e.g. 2k in / 256 out and 8k in / 256 out), C=1 and C=8:

| Metric | Target |
| --- | --- |
| CR1 Load | Completes without OOM |
| CR2 Quality | First 32 greedy tokens agree with same-box vLLM (or documented sampling delta) |
| CR3 Speed | Publish Atlas/vLLM ratio for TTFT and decode tok/s — no required win for v1 |

---

## 8. Step-up ladder (binding order)

Rental is the last rung. Each rung has a lab host and an exit. Do not skip rungs.

### S0 — Harness, references, fabric (3–5 days) · hosts: spark1, spark2, workstation

Deliverables

* `docs/k3/LAB.md` — hostnames, IPs, Up RoCE iface, `NCCL_IB_HCA` pin, SSH proof, NCCL all_gather log
* `docs/k3/BAKEOFF.md` — endpoints, flags, JSONL result schema
* `docs/k3/UMBRELLA.md` — living index of the draft PR (see §16)
* `docs/k3/PRD.md` — this document, in-tree
* Script: `bench_openai.py` hitting Atlas and vLLM
* Pinned refs: `inference-optimization/Kimi-K3-0.40B` and/or `smol-kimi-k3`. Do **not** pin a vLLM K3 image tag until P6 booking week
* Dual-Spark health: NCCL test, QSFP up, passwordless SSH
* Factory / `MODEL.toml` / weight-name map dry-run (C0)

Exit: one **existing** Atlas model and one vLLM model on spark1+spark2 produce a comparison JSONL. C0 green. Umbrella PR open as draft.

### S1 — Model graph on twins (1–2 weeks) · hosts: 5090 and/or spark1

Deliverables

* `kernels/gb10/kimi-k3/MODEL.toml`
* spark-model K3 config + factory + weight loader (BF16/FP8 first)
* Layer loop: embed → `[KDA or MLA + AttnRes + dense/MoE]` → norm → lm_head
* Tests under `ATLAS_SKIP_BUILD` mock GPU for sequence; GPU tests on 5090 **only if** the day-1 SM120 launch test passes, otherwise on spark1

**P1 hour 1:** document whether the 5090 can launch Atlas GB10 SM121 binaries / PTX. If no, drop 5090 from all GPU kernel milestones; keep it for HF / PyTorch / shape debug.

Exit: C1–C6 on 0.40B (token-exact) and 49M (shape) on the host that can actually run the tests.

### S2 — Hybrid cache and long-ish context (1 week) · host: spark1

Deliverables

* Unified cache: paged MLA KV + KDA state pages
* Prefix match unit that does not require a KDA snapshot every token
* Chunked prefill for the twin (e.g. 4k)

Exit: C3 plus 8k context on twin without NaNs.

### S3 — Production-width dummy, single node (few days) · host: spark1

Deliverables

* Local config: hidden/heads like production (7168 / 96), 2–4 layers, few experts, random or sliced weights
* Serves on spark1 via `spark serve <local-k3-dummy>`

Exit: dummy greedy decode is deterministic and stable on spark1.

### S4 — Dual Spark dummy at real width (1–2 weeks) · hosts: spark1 + spark2

Deliverables

* TP=2 across spark1+spark2; optional EP if expert count ≥ 8
* vLLM serving the same local dir for an informational bake-off
* Lab table: Atlas vs vLLM TTFT/tok/s on dummy — labeled comms-bound

Exit: C7. spark1 single-GPU dummy tokens == spark1+spark2 TP=2 tokens.

### S5 — MXFP4 loader and kernels (1–2 weeks, still lab) · host: spark1

Deliverables

* atlas-quant MXFP4 (E8M0 block scales) matching official packed expert layout
* Reuse DeepSeek-V4 MXFP4 path; do not invent a second stack
* Dequant or native FP4 GEMM on GB10 for small expert tensors
* Synthetic packed tensors in tests. Fixture from **one real shard header**. Do not download 1.56 TB

Exit: loader round-trip on a single expert shard fixture. Twin may stay BF16.

### S6 — Product surface (few days) · host: spark1

Deliverables

* Chat template enough for completions bake-off
* `--language-model-only` behavior (ignore vision)
* Explicit error if checkpoint does not fit device mem
* Runbook: `docs/k3/RENTAL.md`

Exit: `spark serve <local-k3-dummy>` and vLLM dummy both answer the harness. C0–C7 green. RENTAL.md filled.

### S7 — Rental soak (1–3 days wall clock) · rented 8×B300 or 16×H200/B200

**Forbidden until S0–S6 and C0–C7 are green.**

Deliverables

* Cheapest box that fits + IB (prefer 8×B300)
* Official `moonshotai/Kimi-K3` on Atlas and the then-current vLLM K3 image (pin the tag the week you book; do not assume `vllm/vllm-openai:kimi-k3` still exists)
* Result pack: load logs, mem, C=1/C=8 tables, 10-prompt greedy token dump

Exit: CR1–CR3 written. Decision: more kernel work vs declare architecture complete.

---

## 9. Atlas engineering map

Follow existing Atlas layout. Plan the split on day 1 — Atlas CI rejects `crates/**/*.rs` over **500 LoC**.

```
docs/k3/
  PRD.md
  UMBRELLA.md      # living index = draft PR body
  LAB.md           # spark1 / spark2 inventory, NIC pins, NCCL proof
  BAKEOFF.md
  RENTAL.md

kernels/gb10/kimi-k3/
  MODEL.toml
  bf16/            # twin + dummy
  mxfp4/           # official experts (fixtures only in lab)

crates/spark-model/src/weight_loader/kimi_k3.rs
crates/spark-model/src/kimi_k3/
  mod.rs
  config.rs
  layer.rs
  kda.rs
  mla.rs
  attnres.rs
  latent_moe.rs
  situ.rs
  cache.rs
```

Register the model factory behind `#[cfg(feature = "kimi-k3")]` or an explicit `model_type = "kimi_k3"` arm so the existing serve matrix stays green.

Later enterprise port is a new hardware dir, not a rewrite. That is P7, not this PRD:

```
kernels/h200/kimi-k3/
kernels/b300/kimi-k3/
```

5090: no `kernels/rtx5090` unless you explicitly add SM120. Use 5090 only if Atlas can run GB10 PTX via compatibility or run tests through a CUDA-arch flag. If 5090 cannot launch SM121 kernels, all GPU kernel work happens on spark1/spark2; 5090 is limited to reference PyTorch / shape debug.

---

## 10. vLLM comparison protocol

Fixed

* Same weights (dummy or official)
* Same tokenizer
* `temperature=0`, `max_tokens` fixed
* Warmup 3 requests, then N=32
* ISL/OSL pairs: 512/128, 2048/256, 8192/256
* Concurrency 1 and 8 (8 only if mem allows)

Reported columns

`engine, hardware, model, isl, osl, concurrency, ttft_p50_ms, ttft_p99_ms, itl_p50_ms, tok_s_per_user, tok_s_system, gpu_mem_gb, notes`

Forbidden

* Comparing spark1/spark2 Atlas dummy to published 16× GB300 vLLM K3 numbers
* Mixing speculative decode on one engine only without a second row
* Writing lab dummy numbers into certified `.benchmarks/*/BASELINE.json`

Lab vLLM on Sparks should reuse the existing Spark / spark-recipes pins, not a K3 image. Pin the rental vLLM image the week you book.

---

## 11. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| KDA ≠ existing GDN kernels | Weeks lost | Separate module; goldens before fuse |
| Hybrid prefix cache bugs at long ctx | NaNs / loops (seen in vLLM DCP >270k) | Cap twin ctx; unshard KDA position cache |
| Two Sparks too slow for dummy TP | False “Atlas is bad” | Label as comms-bound; don’t tune for that number |
| MXFP4 layout mismatch | Silent garbage | Fixture from one real shard header; reuse DeepSeek-V4 path |
| 5090 / Spark arch split (SM120 vs SM121) | Split brain | Decide S1 hour 1 |
| Rental before C0–C7 | Burn rate | Hard gate in this PRD |
| Chat template / reasoning parser | “Model works” but API wrong | Completions-first |
| Umbrella PR treated as mergeable | Pollutes `main`, fails 500 LoC / stamp/seal | Draft only; extract slices |
| “Full K3 on two Sparks” proposal | Wasted week | 1.561 TB vs ~240 GB. Written into LAB.md |

---

## 12. Dependencies

* Atlas source + GB10 image (`Avarok-Cybersecurity/atlas`)
* spark1 and spark2 networked (CX7 QSFP up)
* HF access to twins; later official K3 (~1.56 TB disk **on rental**)
* vLLM Spark image for lab proxies; K3 image pinned at booking time
* Optional: Inferact/RedHat DSpark only after S7 text path works
* CLA signed if you will extract PRs into upstream `main`

---

## 13. Acceptance (v1 done)

1. S0–S6 complete on lab hardware (spark1, spark2, 5090-as-allowed).
2. C0–C7 green.
3. `RENTAL.md` filled with exact Docker/CLI for Atlas and vLLM.
4. One rental run produces the comparison table and greedy token dumps (CR1–CR3).
5. Written decision: ship GB10 dummy + “enterprise pending kernel port” or open P7 (H200/B300 kernel tuple).
6. Umbrella draft PR either closed with pointers to extracted topic PRs, or still open as the leftover log — never merged as-is.

v1 does not require Atlas faster than vLLM.

---

## 14. P7 (after this PRD, optional)

* DSpark
* Vision tower
* 1M context + DCP
* `kernels/b300` / `kernels/h200` tuned GEMM
* PD disaggregation
* 16× Spark native full K3 (only if you want it; not the rental default)

Do not start P7 until S7 artifacts exist.

---

## 15. Immediate next actions (this week)

1. Open the umbrella PR (`wip/k3-bringup`). See §16.
2. Fill `docs/k3/LAB.md` from live spark1/spark2 (`hostname`, `ip -br addr`, `ibdev2netdev`, SSH, NCCL all_gather).
3. Confirm 5090 vs Spark kernel launch (S1 hour 1, can be a one-page note in LAB.md).
4. Stand up bake-off harness on spark1+spark2 with a **shipped** Atlas MoE and vLLM. Produce one JSONL.
5. Vendor `Kimi-K3-0.40B` + `smol-kimi-k3`. Write C0 dry-run + C1 test skeleton. No 1.56 TB download.
6. Sketch `MODEL.toml` and layer enum (KDA vs MLA) from official `config.json` only.

That is the whole path: architecture complete at home on spark1/spark2, weights and headlines on rented Hopper/Blackwell.

---

## 16. Umbrella PR protocol (working log, not a merge)

Goal: one durable Atlas PR that holds the entire K3 bring-up — code, comments, dead-ends, goldens — so you can think on the branch without burning rental budget or polluting `main`. Later, sift winners into small certified PRs.

Atlas CONTRIBUTING expects atomic topic PRs, `/stamp` `/seal` certification, green gates, and `crates/**/*.rs` ≤ 500 LoC. This umbrella is a **draft working log**. It is **never squash-merged as-is**. Never `/stamp` or `/seal` it.

### 16.1 Open it once (day 0)

```bash
# origin = Avarok-Cybersecurity/atlas  (or your fork; then PR fork → upstream main)
git fetch origin
git switch -c wip/k3-bringup origin/main

mkdir -p docs/k3 \
  kernels/gb10/kimi-k3/bf16 \
  kernels/gb10/kimi-k3/mxfp4 \
  crates/spark-model/src/kimi_k3 \
  crates/spark-model/src/weight_loader

# stub so the tree exists and the kernel-shadow check has a legal shape
cat > kernels/gb10/kimi-k3/MODEL.toml <<'EOF'
# K3-WIP: S1 skeleton. Do not serve official moonshotai/Kimi-K3 from this target yet.
[model]
name = "kimi-k3"
hf_id = "inference-optimization/Kimi-K3-0.40B"
params = "0.40B"
active_params = "0.06B"
architecture = "KDA + gated MLA + AttnRes + Stable LatentMoE"
EOF

# copy PRD + UMBRELLA.md + LAB.md stubs into docs/k3/ before this commit
git add docs/k3 kernels/gb10/kimi-k3
git commit -m "docs(k3): umbrella bring-up tree (do not merge)"

git push -u origin wip/k3-bringup

gh pr create --draft --base main --head wip/k3-bringup \
  --title "WIP: Kimi K3 architecture bring-up (do not merge)" \
  --body-file docs/k3/UMBRELLA.md
```

If you do not have write access on `Avarok-Cybersecurity/atlas`, push the branch to your fork and open the draft PR fork → upstream `main`. Same protocol.

Fill the Atlas PR template fields honestly:

* **Summary:** working log for K3 architecture bring-up. Not mergeable.
* **Test plan:** lab gates C0–C7; workspace tests must stay green via feature-flag / ignore, not by disabling existing tests.
* **Notes for reviewers:** do not `/stamp` or `/seal`. Use the conversation as the lab notebook.
* **Authorship:** AI-authored is Atlas-default; say so.
* **CLA:** check it if you intend to extract slices upstream later.

### 16.2 PR body (living index)

`docs/k3/UMBRELLA.md` **is** the PR description. Update it on every phase change:

```bash
gh pr edit --body-file docs/k3/UMBRELLA.md
```

Required sections (template in the companion file):

* Status: S0…S7, last host that ran tests (`spark1` / `spark2` / 5090)
* Rental gate: C0–C7 checkboxes. S7 is forbidden until all are green
* File map
* Decision log (one line per `K3-DECISION`)
* Dead-ends (so we do not re-try on rented silicon)
* Extraction plan (future topic PRs and which umbrella commits they come from)
* Authorship / CLA note: “Working log. Will not /stamp or /seal.”

Use the GitHub PR conversation as the lab notebook. Do not request review for merge.

### 16.3 How to dump code without wrecking `main` or CI

The umbrella must stay *compilable* even when incomplete. Atlas will run fmt / clippy / tests / docs + kernel-shadow + the 500 LoC cap on every push.

1. New code lives only under K3-named paths. Do not edit shipped model targets except to register `kimi_k3` behind a feature / match arm that existing tests do not hit.
2. Register the model factory behind `#[cfg(feature = "kimi-k3")]` or `model_type = "kimi_k3"`. Existing serve matrix stays green.
3. Split early: `kimi_k3/{mod,config,layer,kda,mla,attnres,latent_moe,situ,cache}.rs` from commit 1. Do not discover the 500 LoC cap at extract time.
4. Kernels: `MODEL.toml` plus empty `bf16/` and `mxfp4/` so `scripts/check_kernel_shadows.py` has a legal shape. Do not copy GDN/Mamba `.cu` into this dir and rename them KDA.
5. Graph tests use `ATLAS_SKIP_BUILD`. GPU tests are annotated with the host they require (`spark1`, `spark2`, or 5090).
6. If a push would fail workspace `cargo test` because K3 is half-written, `#[ignore]` or `#[cfg(feature = "kimi-k3")]` the **new** tests. Do not disable existing tests.
7. Never lower `.benchmarks/*/BASELINE.json`. Lab dummy numbers live in `docs/k3/BAKEOFF.md`.

### 16.4 Comment / commit conventions (the searchable log)

Commit subject prefixes:

| Prefix | Meaning |
| --- | --- |
| `feat(k3):` | user-visible graph / loader / cache behavior |
| `fix(k3):` | correctness vs twin |
| `test(k3):` | C0–C7 goldens |
| `docs(k3):` | PRD / bake-off / rental / lab notes |
| `wip(k3):` | incomplete but compiling snapshot (this branch only) |
| `dead(k3):` | revert or quarantine of a failed approach, with why |

Inline comment tags (grep-able):

```
// K3-WIP S1: graph compiles; logits not checked
// K3-GOLDEN C1: token-exact vs inference-optimization/Kimi-K3-0.40B, 128 tok, 8 prompts
// K3-DECISION: KDA is a new backend, not a GDN reuse (2026-09-11)
// K3-DEADEND: fused KDA+MLA state page — NaNs after prefix hit; unshard KDA pos cache
// K3-LAB spark1: NCCL ok on the Up twin; spark2 reachable passwordless
// K3-RENTAL-GATE: blocked on C7
```

Post a session comment on the GitHub PR at the end of each work block:

```
### session YYYY-MM-DD
host: spark1 | spark2 | 5090 | workstation
phase: S1
done: …
next: …
blockers: …
C-gates: C0 ?  C1 ?  C2 ?  C3 ?  C4 ?  C5 ?  C6 ?  C7 ?
```

### 16.5 Later: sift winners into real PRs

When a slice is correct and small, cut it out of the umbrella. Do not rewind the umbrella branch. Do not `/stamp` the umbrella.

```bash
git fetch origin
git switch -c feat/k3-kda origin/main
git checkout wip/k3-bringup -- \
  crates/spark-model/src/kimi_k3/kda.rs \
  crates/spark-model/src/kimi_k3/kda_state.rs \
  tests/k3_kda.rs
# tidy, split files to ≤500 LoC, drop K3-WIP comments that are no longer true
git commit -m "feat(k3): KDA backend + state layout (golden vs twin)"
git push -u origin feat/k3-kda
gh pr create --base main --title "feat(k3): KDA backend" \
  --body "Extracted from #<UMBRELLA_PR>. Gates: C4."
```

Suggested extraction order (one PR each; each goes through `/stamp` `/seal`):

1. `feat/k3-config-loader` — `MODEL.toml` + config + safetensors map (BF16 twin)
2. `feat/k3-kda` — KDA module + state, no GDN reuse
3. `feat/k3-mla-gated` — gated MLA + NoPE
4. `feat/k3-attnres` — block residual mix
5. `feat/k3-situ-latentmoe` — SiTU-GLU + latent down/experts/up
6. `feat/k3-hybrid-cache` — paged MLA KV + KDA state pages + prefix
7. `feat/k3-mxfp4` — packed expert loader + fixture from one real shard header
8. `feat/k3-dual-spark` — TP/EP path + C7
9. `docs/k3-bakeoff-rental` — `BAKEOFF.md` + `RENTAL.md` only after C0–C7 green

After each extraction merges, rebase `wip/k3-bringup` onto `main` so the umbrella shrinks toward empty. When S7 artifacts exist, close the draft PR with a comment pointing at the merged slices and the rental pack.

### 16.6 What this protocol is not

* Not permission to land half-working kernels on `main`
* Not a substitute for C0–C7
* Not a rental ticket
* Not a comparison against published GB300-NVL72 marketing numbers

---

## Appendix A — `docs/k3/LAB.md` stub (fill on P0 day 1)

```markdown
# Lab inventory — Atlas K3

Do not invent these. Fill from the live boxes.

## Hosts
| hostname | IP | role | API | notes |
| --- | --- | --- | --- | --- |
| spark1 |  | head / rank 0 | :8888 | NCCL master-addr |
| spark2 |  | worker / rank 1 | :8889 | no public client API |
| 5090-workstation |  | correctness GPU | local | SM120. Launch test: DATE / PASS or FAIL |

## Fabric
- QSFP cable: spark1:<port> ↔ spark2:<port>
- ibdev2netdev (spark1):
- ibdev2netdev (spark2):
- Up RoCE iface (pin NCCL_SOCKET_IFNAME):
- NCCL_IB_HCA pin:
- passwordless SSH spark1 ↔ spark2: yes/no
- NCCL all_gather log: docs/k3/logs/...

## 5090 SM121 launch test (S1 hour 1)
- Command:
- Result: CAN / CANNOT launch Atlas GB10 PTX
- Consequence: 5090 is GPU-milestone host / PyTorch-only

## Constraint
spark1 + spark2 ≈ 240 GB UMA. Official moonshotai/Kimi-K3 MXFP4 ≈ 1.561 TB.
Two Sparks cannot load official K3. There is no full-model-on-lab milestone.
```

## Appendix B — `docs/k3/UMBRELLA.md` stub (this is the draft PR body)

See companion file `UMBRELLA.md` in this drop. Keep it updated with `gh pr edit --body-file docs/k3/UMBRELLA.md`.
