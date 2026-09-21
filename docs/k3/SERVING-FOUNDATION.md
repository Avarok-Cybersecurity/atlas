# K3 serving foundation

This layer connects the separate host reference graph, tensor-parallel plan and
rank-local safetensors loader to model assembly and serving. It is development
bring-up, not a declaration of full-model or chat support.

The K3 loader binds BF16/FP32 weights and opt-in packed E8M0 experts. Packed
multi-rank weights must carry the rank-aware loader's partition marker; the
loader refuses to reinterpret an unmarked upload as an already sliced model.
FP32 engine projection conversions are adopted by the derived-weight owner.
The shared tensor plan controls slicing instead of a second model-local plan.

Bound layers copy activations to the host reference graph and back. Existing
KDA and MLA CUDA mixer callbacks can be disabled with `K3_CUDA_KDA=0` and
`K3_CUDA_MLA=0`. Packed experts require `K3_ALLOW_MXFP4=1` and their E8M0 kernel.
This foundation retains the original per-call KDA state transfer. Resident
recurrence and optional resident dense/shared MLP execution are a later slice;
this implementation sets the core dense callback to `None`.

Serving selects the rank-local safetensors loader before device allocation,
refuses expert parallelism and GGUF for that path, and pins packed K3 to an
actually compiled MXFP4 kernel target. K3 per-token prefill keeps the scheduler's
chunk budget; it does not inherit the unrelated single-chunk MLA restriction.

## Dependencies and integration

The review base is the clean runtime-weight-loader stack, which already
contains the host reference and shared planner. Separately integrate:

- GB10 expert source registration (#1165), or the B200 target (#1177).
- Idle command receive and bounded broadcast completion (#1171).
- Vocabulary-aware receive capacity (#1173).
- Plain-completion whitespace (#1174) and ordered worker teardown (#1175).
- Official XTML chat guard and raw-marker handling (#1178).

Packed expert gate/up batching (#1163) is a separate optimization. None of these
independent changes are silently duplicated here. Real integration validation
must run against their combined final tree, with exact source/binary identity.

## Validation limits

On this extracted tree, 107 core K3 tests pass (11 fixture-dependent tests
ignored), and Metal-feature production-library checks for model/server plus
the server binary check pass on macOS. These do not execute CUDA. Existing ungated CUDA-only tests
prevent the full model test target from compiling with that feature selection.
Linux/CUDA tests, launch, shutdown and numerical regressions remain required on
the new extracted head. Historical twin evidence belongs to the original
integration tree; it does not certify this reconstruction. Full K3 weights,
B300, TP8, multiple hosts and XTML chat/tools are not validated here.
