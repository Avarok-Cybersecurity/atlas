# Kimi K3 integration notebook

Branch: `wip/k3-b300-integration`, based on main `4c00f65a26d4e06d9f42ceece843cf8def48c734`.
This is a development notebook, not a certified merge candidate or a claim that the full model runs.

## What is included

- Missing K3 factory/weight-loader/bound-layer serving connection recovered from the earlier notebook and adapted to current main. Existing main KDA, MLA, AttnRes, MoE and TP components remain the foundation.
- Small-checkpoint CPU reference, golden token fixtures, and explicit opt-in checkpoint tests. The initial serving binding includes host transfers and is a correctness baseline, not the desired final performance path.
- B300-owned `sm_103a` kernel target and architecture detection. See [target details](../../kernels/b300/README.md).
- Pinned checkpoint manifest, resumable staging, full integrity verification, disk admission, and bounded transfer attempts. See [checkpoint commands](../../scripts/k3/CHECKPOINT.md).
- Rank-aware packed loading, allocation ownership, per-rank weight accounting, and pre-upload shape/dtype/inventory checks. EP greater than one remains refused for K3.
- A bounded real-generation probe that checks identity, usage, output and optional reference prefix; a sequential comparison suite; an owned-rank launch harness; and collective submission diagnostics. See [launch](../../scripts/k3/LAUNCH.md) and [collective checks](../../scripts/k3/COLLECTIVES.md).
- [Rental plan and upstream issue research](B300-PLAN.md), including Hopper lessons and debugging order.

## Before renting

The official packed checkpoint is approximately 1.56 TB. TP8 is the intended eight-B300 layout; **packed rank-local loading is now implemented** with weight/scale slicing before GPU allocation and an explicit TP ownership marker. Small-model evidence is not proof of full-model TP8. Treat the rental as bounded bring-up, not a ready production deployment.

Remaining admission gates:

1. Keep packed TP2/TP8 byte reconstruction, independent dequantization, malformed-scale and allocation-failure tests green. Run production-shape GPU numerical checks and real TP2 generation before the rental; repeat TP4/TP8 on the node.
2. Apply the [official header/memory audit](evidence/official-header-audit-20260919/README.md): 214.6 GiB resident weights per TP8 rank, plus explicit runtime reserves. Provision 1.5–2 TB host RAM for the current host-reference projections; measure startup and first-token peaks on the node.
3. Compile the B300 CUDA target and run numerical/module-load checks on actual B300 hardware. A Spark cannot validate SM103 execution.
4. Validate ordinary generation first, then sequential requests, cache boundaries, prefill versus one-token decode, and rank-failure cleanup. Enable graphs, speculative decoding, special attention/cache modes and performance tuning individually afterward.

Use one pinned snapshot shared by all ranks. Prefer staging on persistent storage before starting GPU billing. Keep a working control engine in a separate pinned environment; do not mix vLLM/SGLang dependencies into Atlas builds.

## Local and Spark checks

Offline tooling checks:

```bash
python3 -m unittest discover -s scripts/k3 -p 'test_*.py'
```

The CPU tests that use weights are explicitly ignored by default and fail if selected without the checkpoint:

```bash
AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
K3_TWIN=/models/Kimi-K3-0.40B \
CARGO_TARGET_DIR=target/k3-cpu \
cargo test --release -p avarok-core --lib kimi_k3 -- --ignored --test-threads=1
```

On a CUDA 13 Spark, build into a dedicated target directory:

```bash
AVAROK_TARGET_HW=gb10 AVAROK_TARGET_MODEL=kimi-k3 AVAROK_TARGET_QUANT='*' \
CARGO_TARGET_DIR=target/k3-gb10 cargo build --release -p spark-server
```

Use `AVAROK_TARGET_HW=b300` and a separate `target/k3-b300` directory for B300. Never reuse Spark PTX on B300. Consult the built binary's `serve --help` for current launch options; use a dedicated port and an owned process with a timeout. Check for other users' jobs before using either lab Spark.

Against an already started small-model endpoint:

```bash
python3 scripts/k3/probe.py --endpoint http://127.0.0.1:18888 \
  --model kimi-k3-twin --prompt 'According to all known laws of aviation,' \
  --max-tokens 32 --deadline 120 --output /tmp/k3-generation-unique.json
```

A plain successful probe proves a completed nonempty request, not numerical correctness. Supply `--expected-prefix` from a trusted reference to check semantic agreement, or compare token IDs against the [golden fixture](goldens/kimi-k3-0.40b-greedy.json). Run each request independently, then in a shared server session to detect state contamination. The probe uses plain nonstreaming completions, not a chat-template or streaming-parser certification. Receipts contain prompt/output text; use non-sensitive canaries.

Before a full download, reproduce the [bounded official-header audit](evidence/official-header-audit-20260919/README.md); it catches storage shapes and padding without fetching the 1.56-TB checkpoint.

Store small summaries with exact commit, checkpoint revision, GPU/toolchain, command, return code and limitations. Keep bulk build/benchmark output in external artifacts. No certification record or seal is created by these checks.

See the [2026-09-19 Spark smoke receipts](rst/2026-09-19-spark-smoke.md) for TP1/TP2 generation, compiler checks, identities and limitations.

See the [expanded two-Spark preparation results](evidence/spark-prep-20260919/README.md) for packed TP comparisons, production-shape GPU math, failures found and remaining rental gates.
