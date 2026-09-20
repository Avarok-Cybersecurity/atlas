# K3 next-rental instructions after the B200 rehearsal

The measured rehearsal used four GPUs in **one** NVLink-connected host. The
B200 results establish small-fixture TP1/2/4 generation, numerical kernels and
bounded lifecycle behavior. They do not establish full K3 or B300 readiness.
See [results](evidence/b200-rehearsal-20260919/README.md) and
[measured CPU-path improvement](evidence/b200-rehearsal-20260919/PERFORMANCE.md).

## Prepare before paying for the next node

1. Pin the reviewed Atlas commit, checkpoint revision and dependency versions.
   Transfer committed source using `git archive` or disable macOS copyfile
   metadata when making a tarball (`COPYFILE_DISABLE=1 tar ...`). AppleDouble
   `._*.cu` files are transfer metadata and must not enter kernel compilation.
2. Use the [checkpoint staging harness](../../scripts/k3/CHECKPOINT.md) to
   create the immutable manifest and stage/verify the official snapshot on
   persistent storage before GPU billing where possible. The checkpoint is
   about 1.56 TB; reserve space for builds and receipts. The small 250-GB
   rehearsal disk is insufficient for this step.
3. Prepare the [official tokenizer](../../scripts/k3/TOKENIZER.md) and
   [segmented token-array request](PROTOCOL.md). Hardlink the verified serving
   tree using `stage_serving.py` on the same filesystem. Do not symlink shards
   outside the serving root or silently make another multi-terabyte copy.
4. Select a single eight-B300 node with the actual free memory, full peer
   connectivity, usable NCCL and enough host RAM. The header audit estimates
   214.6 GiB resident weights plus 2.19 GiB staging per TP8 rank, before runtime
   reserves. Current host projections add about 669.5 GiB logical FP32 payload
   across ranks; the plan prefers 1.5–2 TB host RAM and 4 TB free disk.
   These are admission estimates, not observed full-model high-water marks.

## First bounded B300 cell

Record `nvidia-smi`, `nvidia-smi topo -m`, compiler, driver, NCCL and free-disk
information. Check device occupancy and actual peer access. A successful B200
run is not proof that B300 `sm_103a` modules load. Use the repository-pinned Rust
version; Linux builds need the CUDA compiler, NCCL, C/C++ compiler, CMake,
OpenSSL development files and RDMA development headers.

```bash
export AVAROK_TARGET_HW=b300
export AVAROK_TARGET_MODEL=kimi-k3
export AVAROK_TARGET_QUANT=mxfp4
export CARGO_TARGET_DIR=target/k3-b300
cargo build --locked --release -p spark-server --features nccl
cargo test --locked --release -p spark-model \
  --test k3_mxfp4_cuda_oracle --test k3_mixers_cuda_oracle --no-run
K3_ORACLE_GPU_ORDINAL=0 timeout 180s cargo test --locked --release \
  -p spark-model --test k3_mxfp4_cuda_oracle --test k3_mixers_cuda_oracle \
  -- --ignored --nocapture --test-threads=1
```

Start with the verified small packed fixture. Construct explicit TP1/2/8
manifests following [LAUNCH.md](../../scripts/k3/LAUNCH.md), using actual GPU
UUIDs, the executable SHA256 and `sm_103a`. Reserve each selected GPU. Set EP1,
BF16 KV, prefix caching off and an explicit prefill budget. Opt into an
existing writable absolute `CUDA_CACHE_PATH` in `env`; retain that directory
between runs. B200 measured 56.18s cold versus 8.64s warm startup with the same
binary and no concurrent compiler processes. This is not a B300 timing claim.

```bash
python3 scripts/k3/launch.py --manifest /work/tp8-twin.json --dry-run
python3 scripts/k3/launch.py --manifest /work/tp8-twin.json \
  --output /work/evidence/tp8-twin-first
```

The launcher stops its owned ranks after the canary. Do not point the separate
soak tool at that already-stopped endpoint. For extended testing, supervise a
long-lived owned server and run `soak.py` as described in its [guide](../../scripts/k3/SOAK.md),
then use leader-first bounded teardown and audit complete rank logs.

Only after those gates pass, stage the full official serving tree and use a
TP8 manifest whose `request_file` points to the prepared token-array JSON.
Omit `prompt`, match `model_name`/`max_tokens`, and supply a reviewed expected
raw-output prefix from an independent reference. Preserve raw XTML output and
actual token counts; this is not a native chat/tool response. Save allocation
and process-memory peaks, first-token latency, failures and complete logs.

## What is still unproved or slow

- Full official checkpoint loading, measured memory peaks and useful decoding
  throughput on eight B300s. TP8 CPU/shape tests do not replace that experiment.
- Actual B300 execution and TP8/NVSwitch communication under the full model.
- GPU-resident dense projections, router/shared-expert work and removal of
  repeated host/device transfers. Interleaving CPU GEMV rows improved the tiny
  fixture by 16–35%, but leaves the host-reference architecture in place.
- Native XTML chat/reasoning/tool parsing and meaningful official-model quality
  evaluation. Prepared raw-completion requests are the current supported path.
- Multi-host rental operation. The current controller deliberately requires
  `127.0.0.1` and a complete local rank map. Multiple networked machines need
  distributed orchestration, rank-specific network/RDMA admission, cancellation
  and failure propagation, and new end-to-end transport evidence. Four GPUs on
  one host are not a substitute for those tests.

Do not extrapolate the small model's token rate to full K3. The next major
performance step is profiling and moving remaining host projections to the GPU,
with independent numerical comparisons, before treating a costly full-model
session as a throughput campaign.
