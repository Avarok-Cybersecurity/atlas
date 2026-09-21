<!-- provenance-id: 526f6e616c6420522e205374657369616b -->
# DeepSeek-V4.1 Flash on a rented B200: the first hour

Written 2026-09-19 on one DGX Spark (Blackbird, GB10), before any B200 has been rented.
**Nothing in this document has run on a B200.** Every number is one of three things, and
each is labelled: measured on the Spark (`docs/perf/DS41_DECODE_RETUNE_2026-09.md`, PR #1156
comments S1/S2), read off a compiler (`kernels/b200/deepseek-v4-flash/PTXAS_RESOURCES.md`), or
arithmetic on those two with the operands shown. The recipe PRs #1140 (one GPU), #1141 (two GPUs,
the sm_100a retune) and #1142 (speculation) say what to do with a B200; section 6 says what in
them the day's numbers have overtaken. Prior art for the rental discipline itself: Turney's
`docs/k3/B300-PLAN.md` (#1150): pin the checkpoint revision, stage on provider storage before the
GPU clock starts, cost = rate x paid hours + storage + transfer, one control run before any
optimisation, export the evidence before releasing the node.

## 0. The tree this runbook assumes

1. Code: the head of PR #1156 (branch `ds41-b200-prep`, 6e0f1ca04 on 2026-09-19), which contains
   the chain #1146 -> #1147 -> #1148 -> #1153 -> #1155 plus S1 (the sm_100a gate) and S2 (device
   routing). `main` at ec9ca1f84 has `kernels/b200/deepseek-v4-flash/` but **not**
   `kernels/b200/deepseek-v4.1-flash/`; that directory arrives with #1146, so a
   `AVAROK_TARGET_HW=b200 AVAROK_TARGET_MODEL=deepseek-v4.1-flash` build of today's `main` panics
   in `build.rs` (`unknown_model`). Build from the #1156 head, or from `main` once the chain lands.
2. The S3 whole-step graph (`ATLAS_DS41_STEP_GRAPH=1`) is **not landed**. Its estimate in section
   2.3 is a design estimate, and the first hour runs the eager path with device routing on.
3. The checkpoint: `vcruz305/DeepSeek-V4.1-Flash-GGUF`, revision
   `58d8ac86298fdf85a2440defee08b1abcad32e45`, the seven Q2_K shards (section 3.3 has the bytes and
   the sha256 of each, from the Spark's Hugging Face download metadata).

## 1. Competing numbers first

| engine | hardware | tok/s | what it is | source |
|---|---|---|---|---|
| EXL3 + DSpark (Turney) | 2x DGX Spark (GB10) | ~30-32 (C=1; logger 28-36; 32.2 from `decode_s`) | engram **dummied** ("Engram cache 0"), DSpark 1.62 accepted tokens per draft, 54% draft-token accept, `ENFORCE_EAGER` | X post 2026-09-14 20:30 EDT (@no_stp_on_snek, status 2099657052935127502), his stats card, verbatim in memory `idea-provenance-ledger` |
| Atlas, this engine, S2 | 1x DGX Spark | **19.19 / 21.87** (MinHeap / Volvo medians; hot probe 23.91) | engram on, 100 GiB device arena, 6/6 texts byte-identical to the phase 1 oracle | #1156 S2 comment, 2026-09-19 |
| Atlas, #1155 head | 1x DGX Spark | 18.57 / 21.14 | same standard, host routing | `DS41_DECODE_RETUNE_2026-09.md` phase 4 |
| llama.cpp (vcruz305 recipe) | 1x GB10, CPU backend | 2.3-2.5 | Q2_K, the recipe Turney pointed at on 09-12 | memory `queue-deepseek-v41-flash-single-spark` |
| Atlas, V4 0731 (the team) | 2x GB10 over RoCE | 15.5 | a different model, CUDA graphs off | #1141 body |
| llama.cpp or vLLM, V4.1 Flash on **B200** | | **none found** | searched the repo docs, the three PR bodies, #1150, the memory notes; #1150 cites an 8x B200 llama.cpp example for Kimi K3 (a different model, ~1 TB Q2_K) and says not to use it as a same-checkpoint baseline | |

The standard behind the Atlas rows (`docs/perf/PERFORMANCE_NUMBERS_STANDARD.md` and the retune
doc): chat endpoint, 300 tokens, temperature 0, one serve, MinHeap x3 then Volvo x3, medians of
three, every text byte-identical to the oracle before any throughput is quoted.

## 2. The arithmetic

### 2.1 The byte floor on one B200

One decode token reads 6.09 GB of weights (measured, retune doc S0): routed experts 240 x
12.22 MiB = 2.93 GB; attention q_b 550 + o_b 550 + o_a 440 + q_a 86 + kv 34 MB; shared experts
512 MB; LM head (Q6_K) 543 MB; router bf16 157 MB; engram wkv 103 MB.

| part | bandwidth | 6.09 GB costs | source of the bandwidth |
|---|---|---|---|
| GB10 (Spark) | 249 GB/s device, 223 GB/s pinned (measured) | 24.5 ms; measured step 42.4 ms wall after #1155 | retune doc S0, phase 4 |
| B200 SXM at the low end | 7.7 TB/s | 6.09 / 7.7 = **0.79 ms** | this row is the conservative end of the vendor figure |
| B200 SXM per the repo | 8.0 TB/s (`memory_bandwidth_gbps = 8000`) | 6.09 / 8.0 = **0.76 ms** | `kernels/b200/HARDWARE.toml` |

So the bytes stop being the step. What is left is the work the Spark hides under its 24.5 ms of
reading.

### 2.2 Without the graph: the step is launch-bound

Per token after S2, measured with nsys on the Spark (#1156 S2 comment): **1,905 kernel launches**
(1,865 before S2 + the 40 x 9 us `moe_v41_route_select`), **186 `cuStreamSynchronize`** (one
read-back per layer, plus the head), 49 D2H, 142 H2D. None of those counts changes with the GPU.

| item | count | cost each (assumed; the box measures it in section 5) | ms |
|---|---|---|---|
| kernel launches | 1,905 | 3-5 us of launch and drain on a launch-bound stream | 5.7-9.5 |
| host waits | 186 | 15-25 us round trip each | 2.8-4.7 |
| the bytes | | section 2.1 | 0.8 |
| **step** | | | **9.3-15.0 ms = 67-107 tok/s** |

The brief's band for the same sum is 8-12 ms of overhead, 80-120 tok/s; the two agree that the
one-B200 eager number is on the order of a hundred tokens a second and is set by the per-launch
and per-wait costs, which are the first two things to read off the profile.

### 2.3 With the whole-step graph (S3, not landed): a design estimate

S3 captures the whole hit step (40 layers, engram, head) as one graph and replays it per token,
falling back to eager on the miss flag. The brief's design estimate is nine segment launches a
token (the host actions that cannot be captured: the engram row uploads before layers 1 and 14,
the miss-flag read-back, the argmax). Two costs remain inside a replay: the graph launches
themselves (9 x ~10 us) and the GPU-side minimum per node (1,905 nodes at ~2-3 us of execution and
drain each even when nothing is read: 3.8-5.7 ms). Estimate: **4.5-6.5 ms a token, 150-220 tok/s**,
and the lever after that is the node count (fusion, #1141's retune), not the bytes. The retune doc's
own phrase for this is "a whole-step graph puts it in the hundreds".

### 2.4 Memory: what one B200 holds

B200 SXM 180 GB HBM3e (`HARDWARE.toml`) = 167.6 GiB. Resident dense weights 2.9 GiB (the recipe and
the #1140 body), plus the KV at a 4,096 cap, the workspace and the 16-slot staging ring (16 x
12.22 MiB = 196 MiB). A 150 GiB arena (`ATLAS_DS41_EXPERT_CACHE_GIB=150`, section 4 step 9) is
12,569 slots of the 15,360 (layer, expert) pairs, 82%; 157 GiB would be 13,200 (86%). The measured
working sets (retune doc, S1 addendum): MinHeap 8,418 pairs, Volvo 8,005, their union 11,223, a
993-token output 10,650 and growing ~3 a token. **On the standard, one B200 holds the working set:
misses only on r1 of each suite (the cold cache), zero on r2/r3.** That is arithmetic on Spark
traces, not a measurement; the serve log's miss counter confirms or refutes it in the first suite.

### 2.5 Two B200s (#1141)

Per GPU per token: half the routed bytes 1.465 GB + the replicated dense 3.16 GB = 4.63 GB =
0.58 ms at 8 TB/s (against 0.76 ms on one). The exchange is one 5,120-wide bf16 row (10 KB) per
layer per direction, 40 layers, over NVLink 5 (~1.8 TB/s per GPU, #1141): bandwidth is not the
question, the 40 extra synchronisations are, on a step that section 2.2 already calls wait-bound.
What two GPUs buy is residency for any prompt length (every expert resident, zero misses) and the
C>1 headroom; at C=1 they do not move the byte floor enough to matter. Rent one first.

## 3. Provider, node, cost, transfer

### 3.1 Provider

Self-serve single-GPU B200 (memory `coreweave-vendor-2026-09-16`, list prices read 2026-09-16,
re-check on the day): **Lambda $6.69/GPU-h, Nebius $7.15, Vultr $8.50.** CoreWeave is sales-led
only (HGX B200 8-GPU node $68.80/h, sold in blocks of eight), Akamai has no B200 (memory
`akamai-outreach-2026-09-16`). Turney's #1150 adds the questions to ask before paying: persistent
storage that survives the GPU being released, verified download bandwidth, a tested export path.

### 3.2 Node

1. x86_64 Linux, one B200 SXM (compute capability 10.0; `arch.rs` maps `(10, 0)` to `b200`; a
   B300 is `sm_103a` and the arch preflight refuses the binary on it).
2. NVIDIA driver: the repo pins **no driver version**; it pins the toolkit at CUDA 13.0
   (`datacenter-binaries.yml` fails any other nvcc). CUDA 13.0 needs the R580 driver branch or
   newer (NVIDIA's CUDA 13.0 release notes, the toolkit-to-driver table). The Spark that produced
   every number here runs driver 595.71.05 with nvcc 13.0.88 (`nvidia-smi`, `nvcc --version`,
   2026-09-19).
3. Runtime libraries on the host (`datacenter-binaries.yml` BUILD-INFO): a CUDA 13 runtime,
   `libnccl2 >= 2.28`, `libibverbs` and `librdmacm`. The CUDA toolkit itself is needed only for
   option B of step 5 (building on the box) and for the ptx gate (step 6).
4. Nsight Systems for section 5 (the Spark has 2025.3.2 under `/usr/local/cuda-13.0/bin/nsys`).
5. Persistent NVMe: at least 400 GB free (264.5 GB of shards + a build tree + logs), on a volume
   that is billed without the GPU.

### 3.3 The shards and the transfer

`~/models/dsv41-q2k` on the Spark: 7 shards, **264,521,661,674 bytes = 264.5 GB = 246.4 GiB**
(`du -sb`, 2026-09-19; "247 GB" in the PR bodies is the GiB figure). Revision
`58d8ac86298fdf85a2440defee08b1abcad32e45` of `vcruz305/DeepSeek-V4.1-Flash-GGUF`; the sha256 is
the LFS hash recorded in each shard's `.metadata`:

| shard | bytes | sha256 |
|---|---:|---|
| `DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf` | 43,072,989,056 | `0bcee934bd4e8350c54410681d9300b0321a300fcb1df1437a20703394d08ee0` |
| `...-00002-of-00007.gguf` | 44,805,480,352 | `124ffa15b6b7ec9630715ad18752a4c3f371749647837b8c9d5f7eb52c4f73c0` |
| `...-00003-of-00007.gguf` | 14,935,107,904 | `4dd35b0b086cc0fb19d3a72afd8aed331317950e2257f37b9672702edc3f2453` |
| `...-00004-of-00007.gguf` | 44,217,904,672 | `d24832f4c2f42340d5d4bf4d9b3277c861e0a0a9ccd25e94bda5581c5351f960` |
| `...-00005-of-00007.gguf` | 44,804,598,176 | `34714c3880bde310207e3d12fa488d8d676f338ff635157fded8ec73e133efad` |
| `...-00006-of-00007.gguf` | 44,803,713,696 | `08dc941d26687cb51f736a4f1c3a2c4843fb1d79573164c645106711081f5faf` |
| `...-00007-of-00007.gguf` | 27,875,485,600 | `550bbdb94a69142abfa4b2f2db86ffe3dd6d4ca45eafbb6728f4fe40506fa6b9` |

Plus `tokenizer.json` (6,367,257 bytes) and `tokenizer_config.json` (801 bytes) beside them.

Transfer time for 264.5 GB, arithmetic lower bounds (no stalls, no verification):

| sustained rate | time |
|---|---|
| 25 MB/s (a residential uplink; the Spark is not the source) | 2 h 56 min |
| 100 MB/s | 44.1 min |
| 500 MB/s | 8.8 min |
| 1 GB/s | 4.4 min |
| 2 GB/s | 2.2 min |

**Recommendation:** pull from Hugging Face onto the provider's persistent volume **before the GPU
clock starts** (the #1150 discipline: prestage on provider storage without GPU billing), pin the
revision, verify the seven sha256 above, and only then rent the GPU. Never route the bytes through
the Spark or a laptop.

```bash
# on the provider's storage VM or the node with the GPU not yet attached
pip install -U "huggingface_hub[cli]"
hf download vcruz305/DeepSeek-V4.1-Flash-GGUF --revision 58d8ac86298fdf85a2440defee08b1abcad32e45 \
  --include 'DeepSeek-V4.1-Flash-Q2_K-*.gguf' 'tokenizer.json' 'tokenizer_config.json' \
  --local-dir /data/dsv41-q2k
cd /data/dsv41-q2k && ls *.gguf | xargs -P 7 -n 1 sha256sum | sort -k2 | tee SHA256SUMS
# every line must match the table above; a mismatch = re-download that shard, nothing else starts
```

### 3.4 Cost

Turney's formula (#1150): node rate x paid hours + storage + transfer/egress + any minimum charge.
One B200: 4 h = **$26.76 (Lambda) / $28.60 (Nebius) / $34.00 (Vultr)**; 8 h = $53.52 / $57.20 /
$68.00; two GPUs double it. Storage for 400 GB and egress of the receipts (megabytes) are the only
other lines. Set the stop time from the accepted quote before booking, and hold to the first-hour
plan below: the numbers that matter are all inside it.

## 4. The first hour, minute by minute

Times are budgets. The clock starts when the GPU is attached; the shards are already on the volume.

1. **0:00 Admission** (Turney's "hardware admission" step, #1150).
   ```bash
   date -u; hostname
   nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.used,pstate --format=csv
   # expect: NVIDIA B200, driver >= 580, compute_cap 10.0, used ~0; memory.total is the SKU's HBM
   # (HARDWARE.toml assumes 180 GB = 171,661 MiB; record what it prints and use it in section 2.4)
   nvidia-smi -q | grep -i -E "cuda version|persistence"
   nvcc --version 2>/dev/null | tail -1 || echo "no toolkit: option A only (step 5), no gate (step 6)"
   nsys --version || echo "no nsys: install nsight-systems before step 12"
   df -h /data; ls -l /data/dsv41-q2k; sha256sum -c /data/dsv41-q2k/SHA256SUMS --quiet && echo shards OK
   dpkg-query -W -f='${Package} ${Version}\n' libnccl2 libibverbs1 librdmacm1 2>/dev/null
   ```
   Record the `nvidia-smi` line verbatim: it is the driver line every receipt carries.

2. **0:03 Clone** (needed for the scripts, the oracle comparison and option B).
   ```bash
   git clone https://github.com/rrstesiak/atlas.git ~/atlas && cd ~/atlas
   git fetch origin ds41-b200-prep && git checkout ds41-b200-prep && git log --oneline -1
   # expect 6e0f1ca04 or a later head of #1156; record the sha
   ```

3. **0:04 The oracle files and the scripts.** Copy from the Spark ahead of time (they are text):
   `~/dflash-logs/ds41_retune_oracle_{minheap,volvo}_r{1,2,3}.txt` (the phase 1 oracle every
   standard since #1147 has been held to) and `~/code/atlas-notes/bin/ds41_suite.sh`,
   `ds41_smoke.sh`, `ds41_nsys_serve.sh`, `ds41_nsys_collect.sh`, `ds41_nsys_stats.sh`,
   `ds41_retune4_run.sh` (these live in atlas-notes, not in the repo). Put them in `~/dflash-logs`
   and `~/bin` on the box; edit the three paths at the top of each script (`BIN`, `MODEL`,
   `~/ds41_bin/...`) to the box's paths.

4. **0:05 The binary, option A (preferred): the CI artifact.** Dispatched from the #1156 head
   **before** the rental, it needs no toolchain on the box (`datacenter-binaries.yml`: "download,
   chmod +x, run").
   ```bash
   # before the rental, from any machine with gh:
   gh workflow run datacenter-binaries.yml -R rrstesiak/atlas --ref ds41-b200-prep \
     -f hw=b200 -f model=deepseek-v4.1-flash
   # on the box:
   gh run download <run-id> -R rrstesiak/atlas -n spark-b200-x86_64 -D ~/ds41_bin
   cat ~/ds41_bin/BUILD-INFO.txt   # sha256, nvcc, libnccl, "carries sm_100a PTX and nothing else"
   sha256sum -c ~/ds41_bin/spark.sha256 && chmod +x ~/ds41_bin/spark && ~/ds41_bin/spark --version
   ```
   The model option `deepseek-v4.1-flash` exists in the workflow from #1146 on (today's `main`
   offers only `deepseek-v4-flash`).

5. **0:05 The binary, option B: build on the box** (only with the CUDA 13.0 toolkit present; the
   Spark's release build of this crate set takes tens of minutes, so start it in the background
   and proceed with step 6 while it runs). Exact names from `crates/avarok-kernels/build.rs`,
   `docker/b200/Dockerfile` and `datacenter-binaries.yml`:
   ```bash
   cd ~/atlas
   export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=13000
   AVAROK_TARGET_HW=b200 AVAROK_TARGET_MODEL=deepseek-v4.1-flash AVAROK_TARGET_QUANT=nvfp4 \
     cargo build --release -p spark-server --no-default-features --features cuda,nccl 2>&1 | tail -20
   ls -l target/release/spark && sha256sum target/release/spark
   ```
   `deepseek-v4.1-flash`'s `MODEL.toml` declares `kernel_source = "deepseek-v4-flash"`, so the
   `.cu` set compiled is `kernels/b200/deepseek-v4-flash/nvfp4/` (the symlink mirror) plus
   `kernels/b200/common/`, exactly the 199 sources of the S1 gate. `nvfp4` is the only quant
   directory the target has (the Dockerfile says so); the checkpoint is Q2_K GGUF regardless.

6. **0:07 The ptx gate on the box** (toolkit required; skip if option A and no nvcc, the CI compile
   is the same gate). Expected: the self-test holds (known-good passes, `known_bad_post_blackwell_dc`
   fails), then **199/199**, 0 rejected entry functions, as S1 measured on the Spark with 13.0.88.
   ```bash
   cd ~/atlas && scripts/hopper_ptx_gate.sh --hw b200 --model deepseek-v4-flash --strict --jobs 16 \
     --out ~/dflash-logs/b200_ptx_gate_on_box.json
   python3 -c "import json; s=json.load(open('$HOME/dflash-logs/b200_ptx_gate_on_box.json'))['summary']; print(s); assert (s['pass'], s['fail'], s['rejected_entries']) == (199, 0, 0)" \
     && echo "gate as on the Spark"
   ```
   Any failure here that S1 did not see is a toolkit difference (compare `nvcc --version` with
   13.0.88), not a kernel regression; note it and continue with the CI binary.

7. **0:09 Check-kernels** (the Dockerfile's "FIRST BOOT SHOULD BE `--check-kernels`"): resolves
   every kernel the model needs against the PTX compiled in, prints the unresolved ones, exits with
   that count; the arch preflight logs the compiled arch against the device's CC on the way.
   ```bash
   export ATLAS_DS41_MAX_SEQ=4096 ATLAS_DS41_MAX_TOKENS=512 RUST_LOG=info
   ~/ds41_bin/spark serve --model-from-path /data/dsv41-q2k --kernel-target deepseek-v4.1-flash \
     --max-seq-len 4096 --max-prefill-tokens 512 --gpu-memory-utilization 0.90 --check-kernels --no-tui \
     2>&1 | tee ~/dflash-logs/b200_check_kernels.log; echo "exit $?"
   ```
   Expect exit 0 and a preflight line naming `sm_100a` on CC 10.0. `MODEL.toml`'s
   `[expected_absent]` table was harvested on GB10 and carried unchanged (its own header says so):
   a non-empty "absent" list here is the first B200 fact to file, not a stop.

8. **0:12 The serve.** The Spark recipe is `ATLAS_DS41_EXPERT_CACHE_GIB=100
   ATLAS_DS41_READER_THREADS=16` (#1148) with the device arena on by default
   (`ATLAS_DS41_ARENA_DEVICE`, #1155) and device routing opt-in (`ATLAS_DS41_DEVICE_ROUTE=1`, S2).
   On the box the arena is device memory sized by section 2.4, `--gpu-memory-utilization` is the
   recipe's 0.90 (the Spark's 0.25 is a unified-memory setting), and the readers face a datacenter
   NVMe (start at 16, it only matters on r1).
   ```bash
   env ATLAS_DS41_MAX_SEQ=4096 ATLAS_DS41_MAX_TOKENS=512 ATLAS_DS41_EXPERT_CACHE_GIB=150 \
       ATLAS_DS41_READER_THREADS=16 ATLAS_DS41_DEVICE_ROUTE=1 RUST_LOG=info \
     setsid nohup ~/ds41_bin/spark serve --model-from-path /data/dsv41-q2k --kernel-target deepseek-v4.1-flash \
       --max-seq-len 4096 --max-prefill-tokens 512 --gpu-memory-utilization 0.90 --port 8899 --no-tui \
       > ~/dflash-logs/b200_serve_r0.log 2>&1 &
   until grep -q "Server live and ready" ~/dflash-logs/b200_serve_r0.log; do
     grep -m1 -E "^Error|panicked|CUDA_ERROR" ~/dflash-logs/b200_serve_r0.log && break; sleep 2; done
   nvidia-smi --query-gpu=memory.used --format=csv   # the arena + resident set; record it
   ```
   The load reads the 2.9 GiB resident set and fills the arena lazily; the first request is the
   cold one. If the arena allocation fails, halve `ATLAS_DS41_EXPERT_CACHE_GIB` and note the free
   memory the log printed; do not lower `--gpu-memory-utilization` first.

9. **0:16 Smoke: the known-answer prompt** (`ds41_smoke.sh ask`; the answer must name Paris in a
   coherent sentence; loops or garbage stop the hour here).
   ```bash
   curl -s http://127.0.0.1:8899/v1/completions -H 'Content-Type: application/json' \
     -d '{"model":"dsv41","prompt":"The capital of France is","max_tokens":16,"temperature":0}' \
     | tee ~/dflash-logs/b200_smoke.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text']); print(d['usage'])"
   ```

10. **0:18 The standard, warm pass.** Same six prompts, same settings, one serve
    (`ds41_suite.sh`: `EP=chat`, 300 tokens, temperature 0, MinHeap x3 then Volvo x3).
    ```bash
    EP=chat PORT=8899 ~/bin/ds41_suite.sh b200_r0 300
    python3 - <<'PY'
    import json,os,statistics; L=os.path.expanduser('~/dflash-logs'); same=0; diff=[]
    for s in ('minheap','volvo'):
        tps=[]
        for r in (1,2,3):
            d=json.load(open(f'{L}/ds41_suite_b200_r0_{s}_r{r}.json')); c=d['choices'][0]
            t=c['message']['content'] if 'message' in c else c['text']; tps.append(d['usage'].get('response_token/s',0))
            if open(f'{L}/ds41_retune_oracle_{s}_r{r}.txt').read()==t: same+=1
            else: diff.append(f'{s}_r{r}')
        print(s, 'median %.2f tok/s' % statistics.median(tps), 'r1-3', ['%.2f'%x for x in tps])
    print(f'oracle: {same}/6 byte-identical to the Spark', diff)
    PY
    grep -c -i "miss" ~/dflash-logs/b200_serve_r0.log   # the miss counter against section 2.4
    ```
    **The oracle clause.** The six Spark texts are the byte-identity reference **only if the B200
    kernels reproduce the same bits**. The decode path is plain CUDA C compiled with
    `--fmad=false`, f32 sums in a fixed order, glibc-exact `expf`/`log1pf` on the device (S2), so
    identity is plausible, and it is not proven until this step prints 6/6. If it prints less:
    (a) run the suite a second time (`b200_r0b`) and compare the box against itself: six
    self-identical texts mean a deterministic B200 oracle, which is then filed as
    `ds41_b200_oracle_{minheap,volvo}_r{1,2,3}.txt` with the first differing character position
    against the Spark for each of the six, and every later B200 number is held to that oracle and
    says so; (b) texts that differ run to run on the box are a bug, and no throughput is quoted.

11. **0:30 The standard, cold pass** (#1140's order: warm, then cold after a cache drop).
    ```bash
    pkill -f "ds41_bin/spark serve"; sleep 5
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
    # restart the serve exactly as in step 8 with the log named b200_serve_cold.log, then:
    EP=chat PORT=8899 ~/bin/ds41_suite.sh b200_cold 300
    ```
    The cold pass measures the NVMe's expert reads (12.22 MiB each; 1.8-2.0 ms on the Spark's
    11 GB/s NVMe) against a datacenter NVMe; r1 of each suite carries them, r2/r3 should not.

12. **0:40 nsys, one warm 60-token request** (the method of every profile in the retune doc:
    `ds41_nsys_serve.sh` armed, one warm 40-token request, collect around one 60-token request).
    ```bash
    pkill -f "ds41_bin/spark serve"; sleep 5
    env ATLAS_DS41_MAX_SEQ=4096 ATLAS_DS41_MAX_TOKENS=512 ATLAS_DS41_EXPERT_CACHE_GIB=150 \
        ATLAS_DS41_READER_THREADS=16 ATLAS_DS41_DEVICE_ROUTE=1 RUST_LOG=info \
      nsys launch --trace=cuda,nvtx,osrt --cuda-graph-trace=node -- \
        ~/ds41_bin/spark serve --model-from-path /data/dsv41-q2k --kernel-target deepseek-v4.1-flash \
        --max-seq-len 4096 --max-prefill-tokens 512 --gpu-memory-utilization 0.90 --port 8899 --no-tui \
        2>&1 | tee ~/dflash-logs/b200_nsys_serve.log &
    # after "Server live and ready", in a second shell:
    ~/bin/ds41_nsys_collect.sh          # warm 40, nsys start, 60 tokens, nsys stop
    pkill -INT -f "spark serve"; sleep 5
    ~/bin/ds41_nsys_stats.sh            # cuda_gpu_kern_sum, cuda_api_sum, osrt_sum as CSV
    ```

13. **0:50 Receipts off the box** (the performance-numbers standard: launch line, commit sha,
    driver line, receipt JSON; plus the nsys CSVs and the `.nsys-rep`).
    ```bash
    cd ~/dflash-logs && tar czf b200_first_hour_$(date -u +%Y%m%dT%H%MZ).tgz b200_* ds41_suite_b200_* \
      ds41_nsys_decode* && ls -l *.tgz
    # scp the tarball to the Spark (or the .38 vault) BEFORE releasing the node; then:
    nvidia-smi --query-gpu=memory.used --format=csv && pgrep -a spark || echo "no spark running"
    ```
    Then stop the GPU billing. Everything past this hour is a second session with a plan written
    from these receipts.

## 5. What to profile first, and what each outcome means

Read these five numbers off the step-12 CSVs (per token = totals divided by the 60 tokens minus the
prefill; the retune doc's method):

1. kernel launches a token (Spark: 1,905 with S2) and `cuStreamSynchronize` a token (Spark: 186);
2. GPU kernel time a token (Spark: 44.5 ms) against wall a token (from `usage.response_token/s`);
3. `cuStreamSynchronize` **blocked** time a token (Spark: 25-28 ms, most of it the GPU finishing
   queued work; on the B200 the GPU is idle in a few ms, so blocked time here is round trips);
4. `cuMemcpyHtoDAsync` a token (Spark: 142 with S2, 260 without);
5. the per-kernel duration distribution: how many kernels run under 5 us.

| what the profile shows | it means | the lever |
|---|---|---|
| wall a token >> kernel time a token; gaps between kernels of 3-10 us | launch-bound, as section 2.2 predicts | the whole-step graph (S3 on #1156); until it lands, `ATLAS_DS41_DEVICE_ROUTE=1` is the most the eager path can do |
| kernel time a token is the step and the big kernels (`kquant_mmvq_*`, `attn_v41_*`) run 20-80 us on grids of 128-320 blocks | under a quarter of a wave on 148 SMs (PTXAS_RESOURCES.md: `wq_a` 320 blocks, `wkv` 128, sized for 48 SMs) | #1141 layer 1, the parameter retune: blocks, rows per block, split-K; no source change to the math |
| `cuStreamSynchronize` blocked time is the step; 186 waits at 20+ us | wait-bound; the host round trip per layer costs more than the layer | confirm `ATLAS_DS41_DEVICE_ROUTE=1` was on (H2D 142, not 260); the remaining wait per layer is the header read-back S3 removes |
| step-10 miss counter non-zero on r2/r3 | the arena is smaller than section 2.4 assumed | raise `ATLAS_DS41_EXPERT_CACHE_GIB` toward what `nvidia-smi` left free; two GPUs (#1141) only if the working set will not fit |
| r1 much slower than r2/r3, r2 = r3 | the cold cache, as on the Spark | the NVMe read rate is the number to file; the standard's medians absorb r1 by design |

Whatever the split, the receipt states the five numbers and the launch line, and nothing is quoted
that the six texts did not pass.

## 6. What the day's numbers supersede in #1140, #1141, #1142 (the PRs are left as written)

**#1140 (the one-GPU campaign)**
1. "hw=b200 model=deepseek-v4.1-flash" from `datacenter-binaries.yml`: the option and the
   `kernels/b200/deepseek-v4.1-flash/` target exist on #1146/#1156, not on `main` (section 0).
2. "The kernels/b200/deepseek-v4-flash target ... passes the sm_100a PTX gate": now 199/199 under
   `--strict`, zero source changes, with the ptxas resource table and `SOURCE_SNAPSHOT.json`
   pinning the bytes (S1, 316b35da6).
3. "About three quarters of the experts resident, the rest streamed from NVMe": 82-86% with a
   150-157 GiB device arena, and the standard's working sets fit (section 2.4).
4. "The same LRU that runs on the Spark" and "`ATLAS_DS41_EXPERT_CACHE_GIB` sized to the box":
   the cache is now a device-memory arena with a pinned staging ring (#1155 P3, default on) and a
   random victim among the oldest 5% (#1148); the recipe's `88` / `READER_THREADS: "8"` are the
   pre-#1148 values; the Spark recipe is 100 / 16, the B200 value is section 2.4.
5. "The single-Spark native figure from #1099": 19.19 / 21.87 (S2) and 18.57 / 21.14 (#1155).
6. "How much of the Spark's disk-bound ceiling the streaming still costs": the B200 step is
   launch- and wait-bound (section 2.2), and on the standard nothing streams after r1.
7. "The llama.cpp CPU recipe on one GB10" as a competing number: 2.3-2.5 tok/s is the only figure
   found (section 1); no B200 llama.cpp or vLLM number for V4.1 exists to cite.

**#1141 (two GPUs, the retune, the port in three layers)**
1. "The CPU-side router staging" and "the top-k downloads" crossing the bus every token: gone with
   S2, which puts the selection, weights and plan on the device and leaves one header read-back
   per layer (H2D 260 -> 142 a token, 6/6 byte-identical).
2. "The async device copies on #1099 are the start of that work": the state is #1155's device
   arena plus S2's device routing; S3's whole-step graph is the next piece and is not landed.
3. Layer 1 of the port, "sweeps", is now quantified: registers 40-48 on the decode GEMVs, no
   spills, grids at 0.67 / 0.27 waves on GB10 and under a quarter of a wave on 148 SMs
   (PTXAS_RESOURCES.md); the phase 3 microbench also says register-prefetch variants of the GEMV
   lose on GB10 (bit-identical, +4 to +55%), so the retune is shape, not unrolling.
4. Layer 3 ("tensor memory and the two-CTA MMA path"): confirmed untouched; the gate needed
   `-DAVAROK_NO_WARP_BLOCKSCALE_MMA` and nothing else.
5. The byte floor per GPU with the split: 4.63 GB = 0.58 ms at 8 TB/s (section 2.5); the split's
   cost is the 40 synchronisations, on a step already wait-bound.
6. "The team's two-Spark run of V4 0731 at 15.5 tok/s" stands as the comparison it names.

**#1142 (speculation)**
1. "N-gram lookup drafts as the floor, on from day one": measured against the suite's own texts
   (retune doc S4), lookup drafts yield 1.03-1.20 tokens a step and the MoE half of a K+1 verify
   grows with K, so on this suite the verify costs more than it returns. Lookup stays out of the
   first hour; DSpark from the shipped modules (layers 37/38/39, from the safetensors index) is
   the line that remains.
2. "The two-Spark EXL3 DSpark run and its accept rate in the same table": 54% draft-token accept,
   1.62 accepted tokens per draft, engram dummied (section 1) is that number.
3. Everything speculative waits for a non-speculative B200 number with the oracle passed (the PR's
   own first line); this runbook produces that number.

## Sources

`docs/perf/DS41_DECODE_RETUNE_2026-09.md` (phases 1-4, S0 byte floor, S1 misses and working
sets, S4 lookup ceiling); PR #1156 body and comments S1 (316b35da6) and S2 (6e0f1ca04);
`kernels/b200/deepseek-v4-flash/{README.md,PTXAS_RESOURCES.md,SOURCE_SNAPSHOT.json}` and
`docs/perf/b200_ptx_gate_deepseek-v4-flash_2026-09-19.md`; `kernels/b200/HARDWARE.toml`;
`crates/avarok-kernels/build.rs`; `crates/avarok-core/src/arch.rs`;
`.github/workflows/datacenter-binaries.yml`; `docker/b200/Dockerfile`; `scripts/hopper_ptx_gate.sh`
and `docs/HARDWARE.md`; PRs #1140, #1141, #1142 (bodies as of 2026-09-19); Turney's
`docs/k3/B300-PLAN.md` on `TheTom/atlas` `wip/k3-b300-integration` (#1150);
`~/code/atlas-notes/bin/ds41_{suite,smoke,nsys_serve,nsys_collect,nsys_stats,retune4_run}.sh`;
`~/models/dsv41-q2k/.cache/huggingface/` (revision and LFS sha256); memory notes
`coreweave-vendor-2026-09-16`, `akamai-outreach-2026-09-16`, `idea-provenance-ledger`,
`queue-deepseek-v41-flash-single-spark`; `nvidia-smi` and `nvcc --version` on Blackbird, 2026-09-19.
