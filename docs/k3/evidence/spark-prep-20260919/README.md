# Two-Spark rental preparation — 2026-09-19

This is development evidence for draft PR #1150, not a benchmark certification, merge seal, or claim that the full K3 model runs. Runtime source: `345db2a7c3baaa39baf599e5f5f5509e7d0df2a0`. Tests and tooling are committed separately. The server source bytes were compared with the isolated Spark build tree; binaries were built before the local commits, so an embedded old/unknown revision must not be treated as a certification stamp.

## What passed

- The small original FP32 K3 twin passes the schema-2 launch harness, known completion, endpoint ownership and process cleanup on Spark2.
- A local group32 E2M1/E8M0 packed derivative passes real TP1 and TP2 generation. The expanded **16-case sequence** agrees exactly in text, finish reason and prompt/completion counts: aviation → unrelated request → aviation, then prompt-length boundaries, then aviation again. The expansion adds measured prompt lengths 31, 63 and 127; the complete suite covers 31/32/33/34, 63/64/65/66 and 127/128/129/130. See `generation.json`. The packed derivative is a loader/kernel fixture, not the official quantized checkpoint or a quantization-quality assessment.
- Both ranks record **17,665 matching collective submissions** for the 16-case sequence (13,704 for the earlier 13-case sequence), including shutdown. Successful HTTP generations provide the separate completion evidence. The offline log checker cannot prove device completion, observe graph replay, or detect equal truncation without an independently supplied expected count.
- Enabling prefix caching preserves the same answers. Logs identify prefix matches but **no SSM snapshots**, so Atlas recomputes the prefix and reports zero cached tokens. This covers safe recomputation, not cache-restore acceleration.
- A worker killed after startup causes the probe to fail at its eight-second deadline (8.37 seconds including SSH/controller overhead). The controller then stops its owned process groups; both GPUs return idle. This establishes the external deadline/cleanup path, not a universal timeout on every native NCCL operation.
- The production E8M0 grouped expert GEMM matches independent CPU arithmetic at official TP8 dimensions, including mixed signs/scales, empty experts, multiple routed rows and tile tails. The deliberately dyadic fixture permits exact BF16 output comparison; see `packed-gpu-oracle.txt`.
- Production KDA/MLA CUDA wrappers pass full-model TP8 geometry checks against the core CPU reference, nonzero history, restored continuation, and negative controls. This tests mixer state handoff; it is not a fused-prefill performance test.
- GB10 and B300 K3 release builds complete. The B300 executable built on Spark is **ARM64 host code containing SM103 PTX**; it must not be copied onto an x86_64 rental. Use the native build/container recipe on the destination CPU architecture.

- A clean ARM64 container build passes packed-fixture TP1 and TP2 inference using the identical image on both Sparks. TP2 matches TP1 text/token counts, all 409 collective submissions agree, both containers exit zero and both GPUs return idle. Container TP2 explicitly uses NCCL Socket, not RDMA; see `container.json`. The image includes the inference fixes but predates the final hardware registry/test metadata updates.

## Bugs caught during the rehearsal

1. `extra_cu` in the old K3 manifest was not consumed by the build system. A same-hardware source alias now makes the required E8M0 expert kernel discoverable. The required handle is resolved during model binding, before the boot audit seals kernel lookup.
2. The official `mxfp4-pack-quantized` format was not canonicalized, and a multi-quant build initially chose BF16. K3 now selects its exact MXFP4 target; compatibility checks remain strict.
3. FP32 engine-facing embedding/head/norm conversions had no allocation owner. They are now adopted by the weight store and included in the binding-memory plan.
4. K3 was classified as needing one giant MLA prefill chunk. With a 32-token budget, 128–130-token prompts tried to send 95–97 tokens into a 33-token arena. K3's tokenwise cache path now honors chunking; other MLA guards remain.
5. A short prefill arena did not bound vocabulary-logit reductions on two ranks. Receive capacity now covers both hidden activations and vocabulary logits. The existing overrun refusal remains.
6. Worker exit skipped ordered model teardown and retained sequence handles while freeing the SSM pool. It now releases/drops slots, synchronizes its command stream and tears down the model. The large weight/pool fallback sweep is gone. Eight small pre-existing scratch allocations still reach the backend sweep on each rank; the backend reclaims them, and this is not reported as perfect allocation ownership.
7. The official checkpoint stores padded `A_log`. The [header audit](../official-header-audit-20260919/README.md) validates all 69 zero tails and retains 96 active heads. Nonzero/NaN/infinite padding is refused before upload.

8. B300 was missing from the hardware-ID registry and source-inventory expectations. It now has a distinct key; GB300 is not silently mapped onto B300/B200.

9. The new owned B300 source inherited GB10's known-inconsistent generic expert clamp. B300 now uses plain SiLU there; the unchanged clamp-scope test passes. K3 dispatches its separate SiTU path, so this is source hygiene rather than a changed K3 numerical result. The original snapshot hashes remain provenance, with the later change documented.

## Reproduce the investigation

Build K3 on an idle Spark as described in [the notebook](../../README.md). Prepare the small packed fixture using `scripts/k3/pack_fixture.py --help` and record its checksum. The schema-2 [launch manifest](../../../../scripts/k3/LAUNCH.md) requires explicit rank/GPU identities, binary hash, prefill budget, KV dtype and cache policy. Use separate owned processes and per-rank logs for the two-host Spark run; the packaged launch controller deliberately supports one host only.

Settings used here: TP1 or TP2, EP1, C=1, max sequence 512, max batch/sequence slots 1, prefill budget 32, BF16 KV, prefix caching explicitly off or on, no speculation. KDA/MLA CUDA paths use their existing defaults. Dual-Spark RoCE configuration is lab-specific and must not be copied onto a single-node NVSwitch rental.

Against the owned head endpoint:

```sh
python3 scripts/k3/compare.py --cases scripts/k3/smoke-cases.json \
  --endpoint http://127.0.0.1:18891 --model k3-twin --deadline 20 \
  --output /path/new-tp1-receipts
# Repeat on the TP2 head with the same requests:
python3 scripts/k3/compare.py --cases scripts/k3/smoke-cases.json \
  --endpoint http://127.0.0.1:18891 --model k3-twin --deadline 20 \
  --reference /path/new-tp1-receipts --output /path/new-tp2-receipts
python3 scripts/k3/check_collectives.py --world-size 2 \
  --rank-log 0=/path/rank-0.log --rank-log 1=/path/rank-1.log
```

Keep the K3 build environment and target directory when running the ignored CUDA tests:

```sh
K3_ORACLE_GPU_ORDINAL=0 timeout 180s cargo test --release -p spark-model \
  --test k3_mxfp4_cuda_oracle -- --ignored --nocapture --test-threads=1
K3_ORACLE_GPU_ORDINAL=0 timeout 180s cargo test --release -p spark-model \
  --test k3_mixers_cuda_oracle -- --ignored --nocapture --test-threads=1
```

RST here means asking a focused question, choosing an independent oracle, confirming that a deliberately bad case fails, and preserving observations plus limitations. A healthy endpoint, matching submissions, or compiled PTX alone is insufficient.

## Readiness boundary

This preparation supports a bounded text-completion bring-up session. It does not certify chat-template/channel rendering, streaming/tool calls, concurrent multi-user isolation, CUDA graphs, speculative decoding, or accelerated prefix-cache restore. Those stay separate test charters; the initial rental starts with one request, eager execution, no speculation, and cache reuse disabled. A matching tiny-model completion cannot establish full-model quality or speed.

## Rental admission still required

Use the [official per-rank memory/host-RAM audit](../official-header-audit-20260919/README.md) and [B300 runbook](../../B300-PLAN.md). Require actual usable HBM, 1.5–2 TB host RAM for the current reference projections, adequate persistent storage, a native x86_64 build when applicable, a pinned working control environment, usable NVSwitch, and bounded SM103/TP4/TP8 canaries. Full-weight verification, startup/first-token high-water marks, semantic comparisons and useful throughput remain unmeasured. The host-reference projections are still a substantial performance limitation.

The final Linux workspace test run passed **7,035 tests, zero failures, 120 ignored** across unit/integration/doctest groups. Ignored tests require explicit hardware/weights or remain outside this sweep; the two named K3 GPU tests above were run separately. All 45 Python tooling tests passed. Workspace clippy, rustdoc, formatting, typo and cross-hardware reach checks passed. The exact file-size check passed. The full license checker still reports 14 unchanged Hopper vendor headers already present at the base; no new source-header violations were introduced. Detailed build logs are retained outside the repository; this directory contains compact results rather than benchmark log dumps.
