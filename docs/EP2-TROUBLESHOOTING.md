# EP=2 Troubleshooting — Common Startup Failures

This doc collects the failure modes we keep seeing for two-node EP=2 runs on DGX Sparks, how to diagnose them, and the currently-known-good recipes.

The typical symptom is a log that ends at:

```
INFO spark_comm::nccl_backend: Received NCCL unique ID from master
```

and nothing after that. Atlas never emits `NCCL initialized` or `Listening on`. The container may hang indefinitely or exit quietly. The true cause is almost always one of the items below — **not** a NCCL network problem.

Run Atlas with `RUST_LOG=info` so the pre-flight checks fire. The `Pre-flight:` lines in `main.rs` before NCCL init catch most misconfigurations with an actionable error rather than a hang.

## 1. Community re-quant with mismatched expert count

**Symptom:** `Pre-flight: checkpoint has experts 0..N but config.num_experts = M` — or, without pre-flight, a panic ~10 minutes into startup when the MoE layer tries to load expert `M` and finds `N`.

**Cause:** Community variants on Hugging Face occasionally re-quantize a *different* base model (e.g. MiniMax M3 with 512 experts) while keeping the config of the smaller variant (256 experts). Atlas's EP-shard math splits the config expert count evenly across ranks; when the checkpoint has more experts than advertised, rank 1 ends up trying to load tensors rank 0 is already loading, or vice versa.

**Fix:** Use the known-good checkpoint below, or patch the HF config to match the real expert count.

## 2. MiniMax M2 / M2.7 with `--speculative` on a checkpoint that includes MTP tensors

**Symptom (with pre-flight):** `Pre-flight: MiniMax checkpoint includes MTP module weights starting at layer 62 but the MiniMax loader does not yet consume them.`

**Symptom (without pre-flight):** Weights load fine, NCCL init completes on both ranks, then `build_model` bails mid-load when the MiniMax loader reaches `load_mtp_weights_multi`. Depending on timing rank 1 sees the crash as "peer disappeared" and hangs for `NCCL_TIMEOUT`.

**Cause:** The MiniMax M2 weight loader's per-module MTP extraction is not wired up yet (see the `anyhow::bail!` in `crates/spark-model/src/weight_loader/minimax.rs` toward the bottom of `load_mtp_weights_multi`). A checkpoint that ships MTP tensors forces the bail.

**Fix:** Drop `--speculative` (and `--mtp-quantization …`, `--num-drafts …`) when serving a MiniMax checkpoint that includes MTP modules. MTP is unsupported for MiniMax today regardless of variant.

## 3. Client forgot the RDMA flags on the docker command

**Symptom:** `ncclCommInitRank` hangs silently. The log sequence is `Initializing NCCL` → `Rank 0: waiting for 1 worker(s)` → `Received NCCL unique ID from master` → nothing. Sometimes also `NCCL WARN Could not find rocep1s0f0 in any device`. Often misdiagnosed as "low GPU memory."

**Cause:** Docker blocks `/dev/infiniband` access by default. Without `--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1`, NCCL fails over from RoCE to TCP, then may pick the wrong NIC (public 1 GbE instead of the 200 GbE `enp1s0f0np0`) and sit there waiting for a handshake that never arrives on that interface.

**Fix:** Always include the full RDMA + NCCL env block from `scripts/start-ep2.sh` on **both** ranks. If you're writing your own compose file, copy the `$RDMA_FLAGS` and `$NCCL_ENV` variables verbatim.

## 4. Mixed Atlas versions on head vs. worker

**Symptom:** `ncclCommInitRank` hangs. Same log trail as #3.

**Cause:** NCCL rejects communicators whose versions differ across ranks (silent wait, not a fast error). If the head runs `alpha-2.43` and the worker runs `alpha-2.16`, the transport layer keeps retrying because one side advertises a capability the other hasn't heard of.

**Fix:** Pin the same image tag on both nodes (the scripts/start-ep2.sh `IMAGE` var) or mount the same spark binary on both sides via `-v $PATH_TO_SPARK_BINARY:/usr/local/bin/spark:ro`.

## 5. Genuinely low memory — after all three above are ruled out

**Symptom:** Container exits after `build_model` with nothing more specific than "OOM" in `dmesg`. If this happens **after** `NCCL initialized` is logged, it's genuine memory pressure.

**Cause:** MiniMax's MoE weight transpose pass is 55–60 GB per rank. With 0.70 GPU memory utilization on a 121 GB Spark (weights ~66 GB), only ~20 GB of transient headroom is available. The transpose fails silently and the container exits.

**Fix:** `--gpu-memory-utilization 0.90`. This leaves the transpose enough slack.

## Known-Good Recipes

### MiniMax-M2.7-NVFP4 (EP=2, no MTP)

Verified against `lukealonso/MiniMax-M2.7-NVFP4` on two DGX Sparks connected via RoCE.

```bash
./scripts/start-ep2.sh lukealonso/MiniMax-M2.7-NVFP4
```

Equivalent manual docker invocation (key points: no `--speculative`, `--gpu-memory-utilization 0.90`, full RDMA flag set from `start-ep2.sh`):

```bash
# On head (rank 0):
sudo docker run -d --name atlas-ep0 --gpus all --ipc=host --network host \
  --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_TIMEOUT=22 -e NCCL_IB_RETRY_CNT=7 \
  -e NCCL_NET_GDR_LEVEL=0 -e NCCL_NET_GDR_C2C=0 \
  -e NCCL_DMABUF_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  -e NCCL_PROTO=Simple -e NCCL_ALGO=Ring \
  -e NCCL_MIN_NCHANNELS=1 -e NCCL_MAX_NCHANNELS=2 \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  avarok/atlas-gb10:latest \
    serve lukealonso/MiniMax-M2.7-NVFP4 \
      --rank 0 --world-size 2 \
      --master-addr <head-ip> --master-port 29500 \
      --port 8888 \
      --max-batch-size 1 \
      --max-seq-len 32768 \
      --gpu-memory-utilization 0.90 \
      --kv-cache-dtype nvfp4 \
      --scheduling-policy slai

# On worker (rank 1): same image, --rank 1, --port 0
```

### Qwen3.5-122B-A10B-NVFP4 (EP=2 + MTP)

```bash
./scripts/start-ep2.sh Sehyo/Qwen3.5-122B-A10B-NVFP4
# with GPU_MEM_UTIL=0.90 in the environment; default 0.70 is too tight.
```

MTP is supported on Qwen3.5-122B, unlike MiniMax. Expect ~40 tok/s decode at EP=2.

### What has NOT been verified to work

* MiniMax with `--speculative` in any flavor. Loader TODO.
* `saricles/MiniMax-M2.7-NVFP4-GB10-AC` — has mixed NVFP4 metadata that Atlas's variant detector does not fully handle. Falls back to FP8→BF16→NVFP4 re-quant at load which costs extra memory; when combined with `--gpu-memory-utilization 0.70` it usually trips failure mode #5 above. Use the `lukealonso` variant until we add a preflight compatibility check.
* Three-or-more-rank EP. All current code paths assume `world_size ≤ 2`.

## If none of the above applies

Capture full docker logs from both ranks (`sudo docker logs atlas-ep0 > ep0.log` on head, same on worker for `ep1`) and paste both into a bug report. The `Pre-flight:` block should appear before any NCCL line; if it says "passed" and the hang is still at the unique-ID line, that's a genuinely new NCCL-transport case and we'll need the logs.
