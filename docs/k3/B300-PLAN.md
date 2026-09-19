# Kimi K3 on 8×B300: integration and rental plan

Research snapshot: 2026-09-19. Research informed the `wip/k3-b300-integration` notebook branch. See [README](README.md) for implemented scope, verification, and remaining blockers. No rental or full-checkpoint download has been started.

## Decision

Target one exclusive x86_64 HGX/DGX-style node with eight B300 GPUs on working NVLink/NVSwitch. Official packed MXFP4 is the initial checkpoint; text-only, one request, short context, no speculation is the first useful milestone. Extend existing K3 tensor parallelism to packed weights with TP=8; pipeline parallelism and a new expert-parallel dispatcher are not prerequisites for this first node.

Do the missing serving connection, packed sharding, shape tests, cross-compilation, and download preparation before starting the expensive eight-GPU clock. A short single-B300 validation session is worth considering before the full node if available. Once rented, overlap checkpoint transfer with hardware admission and small-kernel checks, then prioritize native first-token inference. Do not start with a broad benchmark campaign.

## What the records actually say

- [#1020](https://github.com/Avarok-Cybersecurity/atlas/pull/1020) moved to [#1053](https://github.com/Avarok-Cybersecurity/atlas/pull/1053), branch `wip/k3-bringup`, audited head `940bd4eebf3a049a34fa0f48942a26dd7d5e88d9`. #1053 already is a historical, do-not-merge working notebook. Its old status table and file map are not fully synchronized.
- #1053 documents corrected C0–C7 small-model tests: config/index parsing; 0.40B twin token agreement through EOS; prefill/decode/cache/state checks; AttnRes and router perturbations; and TP=2 on two Sparks agreeing for a 16-token prompt, plus a killed-rank timeout. These are useful lab receipts, not full K3 or TP8 certification. [Corrected evidence](https://github.com/Avarok-Cybersecurity/atlas/pull/1053#issuecomment-5657542189).
- KDA #1076, cache #1077, MLA #1078, AttnRes #1079, SiTU/LatentMoE #1080, expert backend #1081, and TP plan #1082 were extracted into a stack subsequently carried onto main by [#1087](https://github.com/Avarok-Cybersecurity/atlas/pull/1087). Current main `ce2bccd736bb2658552d21a11fd4ba198e842603` has those components. Do not re-add their full historical commits.
- Current main's `crates/spark-model/src/factory.rs` has no Kimi loader dispatch. The umbrella has the missing binding and loader, but `weight_loader/kimi_k3/bf16.rs` explicitly refuses TP>1 when packed weights exist. This is a hard production blocker, not merely an unmeasured optimization.
- The umbrella TP plan handles BF16 `.w1.weight/.w3.weight/.w2.weight`, not official `.weight_packed/.weight_scale`. Its packed expert dispatcher assumes all selected expert IDs are locally available. Generic Atlas EP support does not make that dispatcher EP-ready.
- [#895](https://github.com/Avarok-Cybersecurity/atlas/pull/895) and [#899](https://github.com/Avarok-Cybersecurity/atlas/issues/899) are the Hopper campaign lineage. Actual early serving was Qwen on H100, not K3 on B300. Many useful target, numerical, and memory fixes have since reached main. Original #895 should not be imported wholesale.

Historical research context (not required to follow this plan): `Self Study/Kimi K3 - Atlas Port Gap Analysis - 2026-09-10.md`, `Atlas K3 Lab Inventory — 2026-09-11.md`, `Atlas K3 Bring-up Session — 2026-09-11.md`, `H100 Rental Run - 2026-09-05 Log.md`, and `H100 Session 2 - HANDOFF.md` in the Obsidian vault. The public sources above and source paths below make this plan usable without those notes.

## Hardware, storage, and spend

The live Hugging Face API returned checkpoint revision `f831ab66814297da540d832a5235f8e904f29d06`, 96 safetensors files, total **1,560,936,091,448 bytes** (1.561 TB decimal / 1,453.7 GiB); largest file 16,990,916,912 bytes. [Model metadata](https://huggingface.co/api/models/moonshotai/Kimi-K3?blobs=true). Pin this revision or deliberately replace it and regenerate the manifest before execution.

An ideal eight-way weight split is **181.7 GiB per rank**. That is a lower bound, not the actual allocation budget: embeddings, shared experts, norms, routers, intermediate tensors, conversion copies, KV, recurrent state, and NCCL consume additional memory. The historical routed/non-routed estimate was 1,347.1/106.6 GiB, but its EP-only assumption predates the K3 TP implementation. Do not use that old table as a loader contract.

[NVIDIA's B300 user guide](https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html) describes eight 288-GB GPUs and NVSwitch; other NVIDIA marketing pages use 2.1-TB aggregate figures. Require the provider's actual per-device usable memory and `nvidia-smi` output. Admission is based on the largest planned rank plus measured workspace and startup high-water mark, not product-name arithmetic.

Rental requirements:

- All eight GPUs in one node with full peer connectivity, no MIG partitions, working Fabric Manager/NVSwitch. Separate network-connected GPUs are a different plan.
- Root/container access sufficient to build and profile custom CUDA kernels, launch eight ranks, and run NCCL tests. An inference endpoint or restricted notebook is insufficient.
- A driver compatible with the pinned CUDA 13 toolchain and control-engine image; record actual versions. Require the compiler to support the selected B300 architecture.
- Prefer at least 4 TB **free** fast persistent storage: one shared 1.56-TB snapshot plus images, builds, fixtures, logs, and temporary transfer headroom. Do not allocate one snapshot per rank. If a second converted copy is planned, recalculate disk before downloading.
- Prefer 1–2 TB host RAM with bounded streaming load; do not depend on eight full checkpoint copies in RAM. Compute and enforce actual host high-water estimates.
- Verified download bandwidth, resumable transfers, retention after compute stops, and an export destination tested with a sentinel before the session.

NVIDIA account: check DGX Cloud/Lepton or applicable account credits first, then compare the same node specification against Vast. Account ownership alone does not establish a discount. [NVIDIA Lepton](https://www.nvidia.com/en-us/data-center/dgx-cloud-lepton/). Vast's public B300 page returned “No current offers” during this audit; that is not an authenticated inventory search or a price quote. [Vast B300](https://vast.ai/pricing/gpu/B300).

Compare total session cost: node rate × paid hours + storage + transfer/egress + minimum reservation charge. If quoted per-GPU rate is r, eight GPUs for four hours cost 32r before extras. Obtain a spend/time ceiling before booking; none is specified yet.

At sustained payload throughput of 100 MB/s, 500 MB/s, or 1 GB/s, the 1.561-TB transfer alone takes approximately 4.34 h, 52 min, or 26 min, respectively. These are arithmetic lower bounds excluding stalls and verification. Prestage on provider storage without GPU billing when possible; otherwise overlap transfer with small tests and track ETA.

## Working PR and commit sequence

Create a **new isolated rental integration draft** based on then-current authorized main, linked to #1053 and #895/#899. Suggested title: `WIP: Kimi K3 on B300 — packed TP8 bring-up and rental evidence`. Leave Grok's existing branch, worktree, and Spark jobs intact. Before implementation, identify any unpublished work that should be reused. Do not reset, rebase, or force-push the old notebook.

The new draft is the cross-component experiment record; extract focused landing PRs afterward. Its body should record candidate SHA, checkpoint revision, build/image hashes, hardware, remaining blockers, exact reproductions, measured results, and extraction status. Keep raw large traces in durable artifacts, with checksums and summaries in the PR; preserve required certification records when eventually seeking landing. No stamp/seal/merge of the scratch notebook.

Proposed logical commits, each with a useful result:

| Commit | Change | Evidence required |
| --- | --- | --- |
| 1. `docs(k3): pin B300 bring-up inputs and phase gates` | Source/diff inventory, checkpoint manifest, launch specification, cost checkpoints | Current-main vs #1053 map; no duplicate already-landed slices |
| 2. `k3: connect loader and serving lifecycle` | Adapt missing factory/loader/BoundLayer/config/template connection to current `avarok-*` code | Real twin through the actual server; prefill/decode, EOS, reset, prefix-disabled path, error cleanup |
| 3. `runtime: add B300 target and executable kernel admission` | B300 hardware/architecture, defaults, capability declarations, build/selection checks | Wrong-architecture refusal; strict nvcc+ptxas; actual required symbol launches on B300 |
| 4. `k3: load packed MXFP4 by rank` | Packed and E8M0 scale slicing before GPU allocation, per-rank manifest, bounded staging | Reconstruction and numerical tests at production shapes, malformed-scale refusal, memory estimates |
| 5. `k3: execute packed experts under TP` | Correct local expert dimensions and collectives; shared/replicated contributions counted exactly once | One/two-rank packed expert and full-layer comparison; 4/8-rank checks on node |
| 6. `k3: keep production decode state and projections on GPU` | Reuse main's device-cache work; remove host projection and per-token state round trips; GPU SiTU/router/AttnRes as needed | Per-stage numerical checks and actual inference; profile demonstrates reduced host transfers |
| 7. `campaign: launch and recover eight-rank K3 sessions` | Idempotent download, owned launcher, deadlines, persistent receipts, clean worker failure | Dead-rank and dead-server tests, real generation readiness, export/shutdown rehearsal |
| 8. `k3: batch prefill and tune measured B300 bottleneck` | Chunked/batched prefill and one measured kernel optimization at a time | Correctness unchanged; same-workload before/after timing |
| 9. `docs(k3): record CR1–CR3 and extraction map` | Official load, quality, speed, failures and next work | Reproducible results, no implied certification or unmeasured performance claims |

Dependencies: 2→4→5; 3 can progress independently; 6 builds on the binding and current cache implementation; 7 can progress alongside these. Do not spend paid time polishing unrelated code.

Useful umbrella commits to inspect selectively: `48b9c58a4` config/factory/map; `2009459ca` twin ingest; `e0aab722c` HF semantic corrections; `f5a3b999b` absent KDA lower bound; `9682f4599` twin token/dtype/probe fixes; `089834512` packed GEMM binding; `2ed1455b9` TP binding; `83c0d3415` meaningful tests; `940bd4eeb` AttnRes error cleanup. Determine missing hunks against main; do not blindly cherry-pick this list. Token IDs must come from each checkpoint, not the twin's constants.

Reuse Hopper target/default work now represented by #1045/#1046, numeric fixes #1027/#1030, and memory accounting/residency #1034/#1036/#1037 where present. Evaluate #895's owned launcher/cell tooling separately; the single-process adapter was not proven for eight Atlas ranks.

## TP and B300 details

Primary design is TP=8 with tensor-sharded expert matrices, retaining the existing K3 BF16 semantics. `w1/w3` split output rows; `w2` splits its reduction dimension. Slice packed nibbles and scales together, preserving group-size-32 alignment and logical shapes. Production expert intermediate dimension 3072 gives 384 per rank, which is divisible by 32. Shape divisibility alone is not numerical or loader proof.

Keep expert IDs globally consistent, reduce partial expert output exactly once, and ensure replicated shared-expert/residual contributions are not multiplied by world size. Preserve latent norm placement after routed expert mixing and before the up-projection. Validate current BF16 reduction of FP32 partials against an FP32 oracle before choosing a production reduction dtype.

An overlapping TP8/EP8 implementation is an alternative if measurements justify it, but it needs K3-specific expert ownership/filtering/routing/combine behavior. Do not assume `--ep-size 8` supplies that. Avoid adding pipeline parallelism or multi-node networking to the initial critical path.

B300 uses SM10.3; current B200 target uses architecture-specific `sm_100a` and explicitly says B300 needs its own target. Add `kernels/b300/HARDWARE.toml`, the K3 model targets, and the missing `(10,3)` mapping in `crates/avarok-core/src/arch.rs`. Keep `build_arch.rs` and `spark-runtime/src/cuda_backend/arch_preflight.rs` identity checks intact. Use B300 `sm_103a` with compatible tooling or deliberately validated compatible code generation; do not spoof B300 as B200/GB10 or weaken admission checks. [Current B200 contract](https://github.com/Avarok-Cybersecurity/atlas/blob/ce2bccd736bb2658552d21a11fd4ba198e842603/kernels/b200/HARDWARE.toml), [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus).

Start with known-correct packed W4A16/E8M0 execution and GPU dense projections. Audit datatype/scale/stride/layout at real K3 shapes. Only then evaluate native Blackwell Ultra tensor-core kernels. Do not inherit GB10 warp-level blockscale instructions or Hopper defaults by resemblance. MXFP4 and NVFP4 scale formats are distinct; preserve the official format. The W4A16 path is not automatically the same arithmetic as an MXFP8-activation control engine.

## Before the eight-GPU rental clock

1. Reconcile source and isolate build outputs. One private `CARGO_TARGET_DIR` per branch/target/feature combination; record target arch and binary hash. Compile the server and planned numerical examples with consistent features.
2. Reproduce the small twin on the new integration branch, including an actual generation request. Add a small packed-expert fixture using the official packing contract, then production-dimension one-layer fixtures. A BF16 twin cannot validate packed experts.
3. Generate rank-specific allocation manifests without loading 1.56 TB. Prove no code uploads the complete model then shards it, and count every replicated tensor and transient copy. Read bounded safetensor ranges or shard data once with controlled concurrency.
4. Test numerical paths against an independent reference, plus a deliberately corrupted scale/shard or wrong-rank input that the test must reject. Perform B300 cross-compilation and, if available, one-card device tests before booking all eight.
5. Pin the control engine by image digest and verify its exact command works for K3. Current vLLM recipe lists single-node B300 TP support, but includes old estimate/pre-release text: treat it as a starting point, not a verified local launch. [vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/moonshotai/Kimi-K3.yaml).
6. Build the download manifest, resumable transfer job, status/ETA log, launch kit, and incremental evidence export. Download on destination storage rather than through the Mac or the Sparks. Preserve Deckard and existing Grok work.

## Paid-session sequence

Suggested checkpoints are budget controls, not completion guarantees. Set the actual stop time from the accepted quote.

| Stage | Actions | Continue only when |
| --- | --- | --- |
| First 15–30 minutes | Start resumable checkpoint transfer; check eight GPU identities/free HBM/topology; NCCL pair/all-rank collectives; launch required kernel microtests | All devices and links usable; correct SM target launches; disk/export works |
| During transfer | Twin inference on B300; packed production-shape microtests; TP2→TP4→TP8 fixtures; finish clean build and control image pull | Numerical and failure tests pass; loader memory plan fits actual rank capacity |
| Weights complete | Verify all shards against pinned manifest; run control engine alone with same snapshot, short text requests, speculation off; capture token IDs and timings | Real completions, correct model identity, no lingering worker processes after shutdown |
| Native bring-up | Stop control engine; start Atlas TP8 with C=1 and bounded context; record per-rank load peaks; generate 1, then 32, then 128–256 tokens | Finite/coherent output, stable state, no missing tensor/kernel or collective deadlock |
| Useful inference | Ten fixed prompts, repeat requests, prompt-length sweep; then 2k/256 and 8k/256 at C=1; concurrency only after stable C=1 | Correctness and memory remain stable; report actual active vs queued concurrency |
| Optimization | Profile separate diagnostic runs; target dominant CPU transfer/projection, packed expert GEMM, or prefill bottleneck | Each change has independent numerical proof and same-workload improvement |
| Exit | Export patches, pinned sources, logs, failures, profiles, receipts; stop workers and release rental; confirm billing state | Evidence survives and no unintended compute remains |

Do not run Atlas and the full control model concurrently on the same eight GPUs. CPU builds/downloads may overlap GPU microtests if resources permit; benchmark timing must run without those competitors.

If admission fails, fix the environment briefly or release the node. If official load fails, capture the precise rank/tensor/allocation and reproduce on a fixture; do not burn hours repeatedly loading 96 shards. If the control works but Atlas cannot boot, retain that as a control result, not Atlas success. If a major architecture rewrite is needed, export and return to cheaper hardware.

## Testing and exit criteria

Rapid Software Testing means short, focused investigations, not merely accumulating green tests. Every phase sheet records: what risk we explored, configuration/commit, the independent oracle, a known-bad case the instrument caught, observations, and unresolved questions.

- **CR1 load:** official text weights loaded across eight ranks without OOM; rank-local and replicated bytes plus host/VRAM peaks recorded.
- **CR2 correctness:** independent packed math fixtures, real prompt/completion/usage/termination checks, and fixed-prompt comparison with the pinned control. Investigate the first token/logit divergence; do not dismiss it as a “sampling delta” in greedy mode. Bit-exact cross-engine output is a diagnostic target, not automatically achievable across different accumulation/activation paths. Record any approved numerical tolerance before judging results.
- **CR3 useful performance:** report TTFT, decode tokens/sec, end-to-end time, output token counts and peak memory for C=1 and, when feasible, C=8 at 2k/256 and 8k/256. No promised win over vLLM. Label host fallbacks and disabled features explicitly.
- Check reset/cache isolation, prefill-to-decode boundaries, rank failure with bounded shutdown, and packed scale/shape corruption. Validate thinking/tool protocol after basic text works; defer vision, million-token context, speculation, PP and broad tuning.
- Preserve same-build correctness and performance receipts. Instrumented timings are not benchmark results. A health endpoint, assembled PTX, or a resolved symbol does not prove inference.

Remaining decisions: actual provider/node offer, NVIDIA credit eligibility, spend/time ceiling, achievable transfer rate/storage persistence, current unpublished Grok work, and whether the packed TP8 fixture passes before booking. No production readiness or rental outcome is claimed by this plan.

## Upstream research: llama.cpp, vLLM, and SGLang

### Setup links to check before paying

- [vLLM #50102](https://github.com/vllm-project/vllm/issues/50102): documented CUDA12.9 K3 image tag did not exist. Confirm an actual registry manifest and immutable digest, CUDA/driver compatibility, and successful image pull before booking; the old comment suggesting another tag is not present-day verification of that tag.
- [SGLang #33997](https://github.com/sgl-project/sglang/pull/33997), merged: FlashInfer upgrade removed K3 workarounds. Use a coherent image; do not layer obsolete launch-day patches onto newer dependencies.
- [vLLM #57440](https://github.com/vllm-project/vllm/issues/57440): inspect load/repack memory before increasing context/concurrency. Do not interpret a startup OOM as proof the checkpoint cannot fit.
- [vLLM #50394](https://github.com/vllm-project/vllm/issues/50394): warm actual request shapes, with cold compilation separately timed and bounded.
- [SGLang #37393](https://github.com/sgl-project/sglang/issues/37393): mismatched collectives can look like network failure; inspect per-rank counts and real token progress before tuning NCCL.
- [vLLM #51798](https://github.com/vllm-project/vllm/issues/51798): healthy service can emit nonsense; semantic canaries are required before accepting a control build.

All reports were checked on 2026-09-19. Recheck the chosen image against relevant fixes before execution; do not assume open issue status means that image is affected.

Added 2026-09-19. These findings come from upstream source, PRs, and issue discussions; no upstream reproduction was run in this session. An open report identifies a test to run, not proof that every newer build is affected. Freeze the exact control image and record resolved backends before relying on it.

### llama.cpp: useful graph reference, different execution baseline

[PR #26185](https://github.com/ggml-org/llama.cpp/pull/26185), **merged**, adds the Kimi-K3 text model. Its important distinctions from Kimi Linear are cross-layer attention residuals, latent MoE, SiTU activation, gated MLA output before output projection, and full-rank KDA gating. It reuses a DeepSeek residual weighted-sum operation and supports MXFP4 repacking in conversion. This is a useful independent implementation to compare with the HF reference when inspecting Atlas graph order; Kimi Linear support alone is insufficient.

Do not treat llama.cpp's native MXFP4/“Blackwell” label as a B300 kernel receipt. The inspected [CUDA capability source](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh) restricts its `BLACKWELL_MMA_AVAILABLE` instruction path to SM12.x, explicitly distinguishing that family from datacenter Blackwell SM10.x. B300's SM10.3 requires its own validated instruction path.

The 8×B200 example in the #26185 discussion uses a roughly 1.01-TB Q2_K GGUF and layer splitting. It is neither official MXFP4 nor tensor-parallel B300 inference. Use llama.cpp as a graph/packing reference or supplementary independently quantized behavior check; do not use its tokens or throughput as a same-checkpoint numerical baseline for Atlas MXFP4. Do not spend the first rental creating another terabyte-scale GGUF copy.

### vLLM: concrete correctness, loading, and cache lessons

1. **A healthy service can silently generate garbage.** Open [#51798](https://github.com/vllm-project/vllm/issues/51798) reports incoherent reasoning on 8×B300 with `RedHatAI/Kimi-K3-NVFP4` and v0.27.0, while health and latency remained normal. This is a different checkpoint from official MXFP4. A later comment that v0.28.0 fixed GLM does not establish a K3 fix. Our control admission must require semantic canaries, token IDs, and resolved backend/dtype identity—not just an HTTP response or a version number.

2. **One-token prefill is not necessarily decode.** Merged [#51483](https://github.com/vllm-project/vllm/pull/51483), commit `f8b5c11468f665c75968e3a7c12f16ca074f3a30`, fixes a path that classified a fresh one-token request as decode and could read a recycled request's KDA state. Its description explicitly retains limitations around full CUDA-graph replay and metadata missing a prefill flag. Atlas tests should distinguish fresh first-token, resumed one-token chunk, ordinary decode, and zero-length graph padding, with dirty state slots and both eager/graph execution.

3. **Do not mistake a plausible NaN hypothesis for a proven kernel defect.** Open [#51039](https://github.com/vllm-project/vllm/issues/51039) reports persistent NaN/degenerate output following long-context prefill. Subsequent synthetic tests did not find the originally suspected cross-sequence leak in either KDA backend. They also discovered that a reference KDA call mutates its `v` input: reusing that buffer can fabricate a failing test. Clone mutable inputs independently for each oracle arm. Run chunked-vs-one-shot state tests and poison one sequence to test isolation, but keep the first-NaN cause marked unresolved.

4. **Loading/repacking can strand tens of GiB.** Open [#57440](https://github.com/vllm-project/vllm/issues/57440) reports about 45.6 GiB reserved-but-unallocated after official MXFP4 loading/repacking on B200 with v0.28.0, fastsafetensors and distributed EP. A proposed cleanup restored driver-visible headroom in the reporter's test. This is not a demonstrated B300/Atlas defect, and copying a PyTorch cache-flush call into Atlas makes no sense. Transfer the lifecycle test: record allocator-live, allocator-reserved, and driver-free memory after load, repack, communication initialization, KV reservation, and first forward; explicitly release temporary buffers.

5. **Shared expert replication matters.** Merged [#50656](https://github.com/vllm-project/vllm/pull/50656), commit `5df9999fcfaa72d9eb61348a789058fff805f142`, adds shared-expert sharding. Its source analysis identifies roughly 22.6 GiB/rank in replicated shared-expert weights for the discussed full geometry. Its performance measurements used a pruned development checkpoint, so their gains must not be projected onto our run. Count replicated shared experts explicitly and benchmark the communication tradeoff before sharding them.

6. **TP padding rules must not leak into EP.** Merged [#51131](https://github.com/vllm-project/vllm/pull/51131), commit `beca88e59ea75a7aa1af72a5ae50188fa91d4e3d`, limits intermediate-width TP padding to non-EP operation. Atlas's packed tensor manifests must distinguish logical expert width, rank-local width, padding, and quantization-group layout. Switching topology must not silently reinterpret those dimensions.

7. **Cache boundaries cost correctness and performance work.** Open [#50235](https://github.com/vllm-project/vllm/issues/50235) reports cache misses at a 1536-token physical block boundary in its configuration. Proposed [#50409](https://github.com/vllm-project/vllm/pull/50409) remains open; the reporter found improvement with configuration-dependent retention behavior. Test lengths B−1/B/B+1 and 2B−1/2B/2B+1 using the engine's actual block geometry; do not hardcode 1536 as a K3 architectural constant. Include same-prefix extension, eviction and request-finish lifecycle.

8. **Warmup must cover real workloads.** Open [#50394](https://github.com/vllm-project/vllm/issues/50394) tracks kernels still compiling after readiness. Record cold startup/first-request separately, then warm the specific prompt lengths, batch sizes, and sampling modes before timing them. Keep compilation cache identity tied to the pinned image/build.

9. **Protocol tests can run before renting.** Open [#54273](https://github.com/vllm-project/vllm/issues/54273) reports omitted channel-close markers dropping tool calls and message terminators leaking in streaming. Build tokenizer/parser fixtures for direct think→response/tools transitions, split markers at every streaming boundary, truncation, and streaming/non-streaming equivalence. Use K3's actual token-rendering contract, not a generic Kimi-K2 Jinja assumption. A short completion that contains only reasoning is different from a useful final answer; report both.

Useful optimization ideas after native correctness: merged [#52789](https://github.com/vllm-project/vllm/pull/52789) moves recurrent checkpoints inside prefill instead of requiring a second full-model pass; [#53614](https://github.com/vllm-project/vllm/pull/53614) extends checkpoint handling to partial reuse/speculation; [#55356](https://github.com/vllm-project/vllm/pull/55356) groups MLA cache insertion launches; [#56159](https://github.com/vllm-project/vllm/pull/56159) removes mixed-batch gather/scatter copies. These are design references to profile against, not promised Atlas speedups or drop-in Rust/CUDA patches.

### SGLang: same-node evidence with explicit limits

The [official K3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx) marks 8×B300 unified low-latency/balanced speed configurations verified, while explicitly saying accuracy has not been remeasured. Its published DSPARK numbers use `SGLANG_SIMULATE_ACC_LEN`; simulated acceptance is not an observed speculative speedup. Use a no-speculation baseline and archive every environment switch. Do not compare that speed table directly to Atlas ordinary decode.

Open [#37393](https://github.com/sgl-project/sglang/issues/37393) reports different embedding ALLREDUCE sizes across TP ranks on one 8×B300 node, SGLang v0.5.18 at `71de97b`, with long/multimodal chunked prefill and DCP/DSPARK/HiCache enabled. The process continued answering model/metrics endpoints while generation stalled. This is a report against that combination, not proof against a newer simple TP8 run. For Atlas, log collective sequence, operation, dtype, element count and request/token layout per rank; test odd chunk tails and mixed-length requests with bounded failure. A completed NCCL microbenchmark does not validate scheduler-generated collectives.

Start the control with unified text-only TP8, speculation off, no DCP or external cache tier, and a modest context cap. Enable optional features one at a time with a correctness check. The [SGLang K3 roadmap](https://github.com/sgl-project/sglang/issues/32607) also records custom dependency work; pin a coherent image rather than assembling arbitrary FlashInfer, DeepGEMM and DeepEP versions on the paid node.

### Additional confirmed fixes worth carrying into our tests

- llama.cpp [#28068](https://github.com/ggml-org/llama.cpp/pull/28068), merged September 6, corrects K3 Q/K L2 normalization to `x * rsqrt(sum(x²) + eps)`. The previous denominator formula differs near zero. Add zero/tiny-norm vectors to the independent KDA oracle rather than testing only ordinary random inputs.
- llama.cpp [#28466](https://github.com/ggml-org/llama.cpp/pull/28466), merged September 8, repairs KDA/convolution rollback snapshots. Zero-filled caches concealed the broken path; nonzero-filled buffers exposed it. Test dirty snapshots, multiple rollback positions and sequence isolation. Its reported snapshot-memory expansion reinforces deferring speculation until memory has been measured.
- SGLang [#32477](https://github.com/sgl-project/sglang/pull/32477), merged at `ee678910f7000aa43886f218de0e159bf418f1b5`, prevents writes to a reserved padding KV slot. Masked arithmetic can still propagate NaN (`0 * NaN`), so test padded CUDA-graph rows and guard slots. Open [#32968](https://github.com/sgl-project/sglang/issues/32968) reports additional NaNs after that fix; replacing NaNs with finite logits is not a correctness oracle.
- SGLang [#36859](https://github.com/sgl-project/sglang/issues/36859), open, reports cross-request answer contamination under concurrent multi-turn serving on specified B200/GB300 builds. Root cause/current-main applicability is unproven. Add independent conversations with unique facts, cancellation, cache eviction and slot reuse to the concurrency charter.
- SGLang's K3 configuration notes budget recurrent state separately from MLA KV: DCP can shard the latter without sharding the former. Prefix snapshots and speculation consume additional recurrent-state slots. Log both pools and actual maximum active requests rather than inferring concurrency from free KV bytes.
- For architecture goldens, current [llama.cpp K3 graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/kimi-k3.cpp) banks raw residual inputs, normalizes vectors for residual scores but mixes raw vectors, and keeps shared experts on the original full-width input. Compare these stages with official HF tensors before fusing them in Atlas.

### Changes to the rental gate from this research

- Add CPU-only K3 prompt-renderer and channel-parser parity tests to pre-rental preparation.
- Add first-token-vs-decode and poisoned/recycled KDA-state tests, including graph padding; use separate copies for mutable reference inputs.
- Add rank-collective metadata checks and block-boundary prompt tests to TP8 admission.
- Require memory readings at startup lifecycle boundaries, not just a final weight-residency estimate.
- Pin control-image digest plus FlashInfer/DeepGEMM/DeepEP/CUDA versions and log actual selected backends. Retain the original MXFP4 checkpoint; do not silently substitute NVFP4/GGUF.
- Admit the control only after real semantic canaries. Use no speculation, unified serving and bounded context first; match reasoning effort and rendering between engines. Any throughput derived from simulated acceptance or a pruned/different checkpoint stays out of the comparison.
