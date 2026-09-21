# Rank-local K3 checkpoint loader

`K3SafetensorsLoader` inventories the checkpoint before allocating GPU memory,
then uses the shared tensor-parallel byte planner to upload only each rank's
local bytes. This extraction exposes the loader API; selecting it in model
assembly and serving is a dependent change, not included here.

The loader validates tensor roles, storage dtypes, replicated shapes, packed
weight/scale pairs, required tensor names and index consistency. Index reads
are capped at 128 MiB and duplicate keys are rejected. Shard filenames must
be single path components; canonical shard/index paths must stay under the
checkpoint root. Use an immutable checkpoint directory during loading:
these path and metadata checks are not a sandbox against concurrent file
replacement or modification.

Admission requires:

```
rank-local uploaded bytes + binding copies + largest staging tensor + reserve
    <= initially reported free device memory
```

The caller supplies the reserve for engine, KV cache and runtime allocations.
This is an explicit admission bound, not a complete full-model peak-memory
estimate. Each shard is mapped and rank-local tensor bytes are staged on the
host before upload. No unified-memory fallback is introduced. Failed uploads
release prior allocations; successful stores record `(rank, world_size)` so
consumers can avoid slicing the same weights twice. Existing loaders remain
unchanged.

## Host checks

```sh
AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
  cargo test -p spark-runtime --no-default-features --lib weights::k3
```

The nine host tests use real temporary safetensors files and a recording GPU
backend. They cover exact packed rank bytes, allocation rollback, topology and
memory rejection, malformed scale pairs and replicated shapes, checkpoint
completeness, the binding-copy/reserve boundary, duplicate index keys and
oversized indexes. The mock allows verifying that invalid inputs fail before
allocation; it does not establish CUDA transfer behavior or device memory
availability. No new GPU execution, full-checkpoint inference, distributed
collective or performance certification is claimed by this extraction.

The runtime depends on the shared core K3 graph, geometry, binding-memory and
tensor-parallel byte-planning contracts. Merge the core/planner dependencies
first. `DeferredTensor` and resource release move into separate files only to
keep the weight-store source within the repository file-size limit.
