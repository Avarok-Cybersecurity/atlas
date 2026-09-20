# K3 tensor-parallel storage planning

The shared core planner describes row/column/replicated placement, byte ranges
for dense and packed MXFP4 tensors, required text-decoder names and replicated
shapes. The spark-model adapter delegates to this planner instead of maintaining
a second projection-name table. Device allocation and serving are separate PRs.

## Memory estimates

```sh
AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo run -p avarok-core --example k3_rank_memory -- 4
python3 scripts/k3/audit_headers.py --output /data/k3-header-audit
AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo run -p avarok-core --example k3_rank_memory -- 4 /data/k3-header-audit/headers /data/k3-header-audit/a-log
```

The header reader performs bounded HTTP Range requests against one pinned
checkpoint and rejects full-body responses. It does not download complete
shards. Its output directory must not already exist, so stale header JSON cannot
contaminate a new receipt. A failed run preserves partial evidence; use a new
output path for the next attempt. Offline config estimates and actual header accounting differ: preserve
which mode produced a result. Neither predicts exact GPU peak memory, allocator
fragmentation, context, NCCL buffers, KV cache or inference scratch. Binding-copy
accounting describes the extracted binder contract, not every future loader.

## Small packed test fixture

With numpy and safetensors installed, `pack_fixture.py` generates a synthetic
MXFP4 expert fixture from the small FP32 checkpoint. Set an explicit input-size
limit. This is a bounded test-data generator, not a production quantizer:

```sh
python3 scripts/k3/pack_fixture.py --source /data/k3-small --output /data/k3-packed-fixture --max-source-bytes 2000000000
python3 -m unittest discover -s scripts/k3 -p 'test_*.py'
```

## Evidence boundary

The combined core stack has 107 passing host K3 tests and 11 ignored real-model
cases. Planner checks exercise tensor aliases, bad layouts, packed byte sizes,
rank slicing and overflow; the header harness rejects wrong range responses.
No full checkpoint or distributed execution is claimed. The rank-memory example
was run locally in config mode only. Existing observed rental evidence stays in
#1150. Review the numerical and allocation gates before any hardware run.
