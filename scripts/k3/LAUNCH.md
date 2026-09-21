# Local rank launch plan

This is a single-host controller for one GPU per rank. It requires the complete
rank map and tensor/expert dimensions and refuses partial multi-host configurations.
It reads a strict manifest and verifies executable bytes against its build receipt, model
config, requested device identities, GPU occupancy, architecture declaration,
and free bootstrap/API ports before launching anything. It keeps GPUs visible to
NCCL and pins each rank with `--gpu-ordinal`.

Start worker ranks before rank zero. Own a fresh process group for every rank,
capture separate logs, and stop promptly if any rank exits. A bounded health
check precedes the existing real-completion probe. Require a known expected
prefix rather than accepting arbitrary nonempty output. Always terminate the
owned groups and write the outcome, including failures and timeouts, into a new
evidence directory. This is a bring-up canary, not a persistent deployment or
benchmark certification.

Validate with local fake executables: successful completion, dead worker,
occupied endpoint, missing binary/config, wrong SHA/architecture/rank map,
timeout, and child cleanup. No GPU or model download is needed for these checks.

## Retain the CUDA JIT cache between launches

Do this before the first rental canary. Create a directory owned by the account
running Atlas, on storage retained between server/container restarts:

```bash
mkdir -p /work/cache/atlas-cuda
test -w /work/cache/atlas-cuda
```

Replace `/work/cache/atlas-cuda` with the actual absolute path on the rental.
For containers, mount retained storage at that path; a directory in a deleted
container's writable layer will not survive recreation. Add the path to the
existing manifest `env` object, preserving its other settings:

```json
"CUDA_CACHE_PATH": "/work/cache/atlas-cuda"
```

Setting it only in the launching shell is insufficient: this controller
sanitizes inherited environment variables, including `HOME`. It explicitly
forwards the manifest's `CUDA_CACHE_PATH` to every rank and refuses a relative,
missing or unwritable directory. Keep the directory across subsequent launches;
do not clear it as routine cleanup. The first launch against an empty directory
still pays the cold-start cost. Driver, architecture or kernel changes may
require new cache entries, so do not assume a previous device's warm timing.

For a controlled reproduction, reserve the GPU, stop build jobs, select a new
empty cache directory, and launch once. Launch again with the same binary,
model, GPU, manifest and cache directory, using a new evidence output directory.
Compare start-to-owned-healthy times and require the same successful inference
probe in both runs. Preserve whether each result was cold or warm in its receipt.

## Running it

Save a manifest like this with **real absolute paths, SHA256 from the build
receipt, compiled architecture, and GPU UUIDs**. This one-rank example can be
used on a Spark after replacing its identities. For an eight-GPU B300 run use
`world_size=8`, the supported TP/EP layout, `sm_103a`, and eight rank/device/UUID
entries. No layout is claimed supported merely because the harness accepts it.

```json
{
  "schema": 2,
  "binary": "/absolute/path/to/spark",
  "binary_sha256": "REPLACE_WITH_64_HEX_DIGITS",
  "compiled_arch": "sm_121f",
  "model_dir": "/absolute/path/to/verified/twin",
  "model_name": "k3-twin",
  "world_size": 1,
  "tp_size": 1,
  "ep_size": 1,
  "ranks": [{"rank": 0, "device": 0, "uuid": "GPU-REPLACE"}],
  "master_addr": "127.0.0.1",
  "master_port": 29500,
  "port_base": 18888,
  "env": {"AVAROK_COMM_DIAGNOSTICS": "1", "NCCL_DEBUG": "INFO"},
  "max_seq_len": 1024,
  "max_prefill_tokens": 32,
  "kv_cache_dtype": "bf16",
  "enable_prefix_caching": false,
  "max_batch_size": 1,
  "max_num_seqs": 1,
  "gpu_memory_utilization": 0.8,
  "boot_timeout": 300,
  "probe_timeout": 60,
  "cleanup_timeout": 5,
  "prompt": "REPLACE_WITH_FIXED_REFERENCE_PROMPT",
  "expected_prefix": "REPLACE_WITH_REFERENCE_COMPLETION_PREFIX",
  "max_tokens": 32
}
```

```bash
python3 scripts/k3/launch.py --manifest /path/run.json \
  --output /path/evidence/new-run --dry-run
python3 scripts/k3/launch.py --manifest /path/run.json \
  --output /path/evidence/new-run
python3 -m unittest discover -s scripts/k3 -p test_launch.py
```

Schema 2 requires explicit `max_prefill_tokens`, `kv_cache_dtype`, and
`enable_prefix_caching`; schema 1 manifests are refused. Migrate an old manifest
by selecting these values deliberately and setting `schema` to 2. Prefill tokens
must be a nonnegative integer: `0` requests unchunked prefill, while a positive
value sets the scheduler budget. KV dtype must be `bf16`, `fp8`, or `nvfp4`,
subject to actual model/kernel support. Prefix caching must be JSON `true` or
`false`. All three values are forwarded explicitly, including
`--enable-prefix-caching false`, so the binary's defaults cannot alter the run.
The example uses a small chunk budget for boundary tests; choose a measured
budget for the actual rental workload.

Dry run verifies local config/files/binary hash and renders exact argv without
GPU queries or launches. Live execution requires Linux procfs, `nvidia-smi`,
access to the owned processes' descriptors, and the local verified snapshot.
It records inventory, NVLink topology, each rank's argv/PID/log, probe JSON/log,
and a final summary. Existing output directories are refused. Export the whole
new evidence directory using the rental's configured transfer tool; this
controller has no remote copy or provider shutdown authority.

Startup admission runs three device queries with a 15-second timeout each.
After rank creation, the boot deadline includes all health polling. Health
checks have a separate child deadline to stop trickling headers from extending
the boot wait. Generation has its own deadline, plus a two-second allowance for
the probe to write its receipt. Teardown first signals rank zero, allowing
workers to receive its shutdown collective for up to half the configured grace.
It then signals all owned groups for the remaining grace, kills surviving
groups (including descendants), and reaps leaders (at most one second per
leader). The grace is shared across ranks, not multiplied by rank count. Account for **admission + boot + probe + teardown**, and disk hashing
before admission, when reserving rental time. This is not a hard wall-clock
deadline for an unresponsive filesystem or kernel.

The declared architecture is a build-receipt assertion bound to the executable
hash, compared conservatively with each selected device's CC. It is not extracted
from `nvidia-smi` or proof of device-code coverage. Atlas's own architecture
preflight remains authoritative. `--check-kernels` is deliberately not used as
a single-rank admission step: it loads weights and a full official K3 checkpoint
cannot fit on one GPU. Verify checkpoint bytes separately with `checkpoint.py`;
this launcher checks only `config.json`, not terabytes of weights on each run.

Environment inheritance is restricted. The manifest may explicitly set
`AVAROK_*`, `NCCL_*`, `RUST_LOG`, `LD_LIBRARY_PATH`, and the validated
`CUDA_CACHE_PATH`; secret-like key names are
refused. These values are recorded, so supply configuration only. HF networking
is disabled. `CUDA_VISIBLE_DEVICES` is not inherited or set; device ordinals
must match the full node's inventory. Runtime flags not represented here retain
the binary's defaults; record the binary pin and add a reviewed manifest field
before comparing an additional serving lever.

The existing K3 switches `K3_ALLOW_MXFP4`, `K3_CUDA_KDA`, and `K3_CUDA_MLA` are
also allowed with explicit `"0"` or `"1"` values. Set `K3_ALLOW_MXFP4=1` only for
a reviewed packed fixture or official-checkpoint attempt: it opts into the
loader's experimental packed-expert path, not a promise of full-model support.
The CUDA mixer switches let a controlled test select CPU references. Preserve
their exact values in every comparison; the environment does not inherit them.

This is single-host only. It intentionally refuses partial rank maps and remote
master addresses. For two separate Sparks, use a reviewed multi-host launcher
and the same pinned binary/model/probe contract on both hosts; do not run two
copies of this controller and mistake their independent local rank worlds for
a cross-host NCCL deployment.

## Prepared token-array requests

Schema 2 also accepts `prompt` as a nonempty array of unsigned 32-bit token IDs;
booleans, floats, nested arrays and negative IDs are refused. Text prompts retain
their original meaning, including text that happens to look like JSON.

For the official encoder flow, replace the manifest's `prompt` field with
`"request_file": "/absolute/path/request.json"`. Use the JSON emitted by
`prepare_prompt.py`; the manifest's `model_name` and `max_tokens` must match it.
The request must explicitly declare greedy `temperature: 0` and `stream: false`.
Its optional `stop` array is preserved. Unknown request fields are refused.
`prompt` and `request_file` cannot both be present.

The launcher snapshots the validated payload into the new run's
`probe-request.json` before starting any ranks. Later edits to the input file do
not change the submitted canary. The same bounded probe accepts it directly:

```bash
python3 scripts/k3/probe.py --endpoint http://127.0.0.1:18888 \
  --model kimi-k3 --request-file /absolute/path/request.json \
  --expected-prefix 'REPLACE_WITH_REVIEWED_REFERENCE_PREFIX' \
  --deadline 120 --output /absolute/path/new-probe.json
```

Token IDs and stop strings are sent unchanged to `/v1/completions`. The response
must report the same prompt-token count as the submitted array. This adds raw
completion plumbing; it does not implement native XTML chat or tool parsing.
