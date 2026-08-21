// SPDX-License-Identifier: AGPL-3.0-only
//
// Tensor-core prefill attention at HDIM=512, for Gemma-4's global layers.
//
// This is `kernels/gb10/common/inferspark_prefill.cu` instantiated at a second
// shape — NOT a copy. The body is unchanged; only BC and HDIM differ, and the
// four constants that made that possible were parameterised separately so the
// change could be shown SASS-identical at the original shape.
//
// WHY IT EXISTS. Gemma-4-26B-A4B has 5 global (512-wide) and 25 sliding
// (256-wide) attention layers. The sliding ones take the tensor-core kernel at
// 2.0 ms/layer; the global ones fell through to `inferspark_prefill_512`, 114
// lines of scalar code with no tensor cores, at 3296.9 ms/layer — 16,484 ms of
// an 18,078 ms 4096-token prefill, 91% of the model's cold TTFT.
//
// SHAPE. BC=16, not 32: at HDIM=512 a 32-row K/V tile needs 132.8 KB of shared
// memory against a 101,376 B cap. BC=16 fits in 84,992 B. BR stays 32 because
// the warp mapping splits a 32-row Q tile across two warp pairs.
//
// MEASURED standalone against `inferspark_prefill_512` (S=1024, 4 q-heads,
// 2 kv-heads, causal): cosine 0.999998, max abs 0.0039, 64.7x faster
// (22.02 ms -> 0.34 ms). The residual is bf16 accumulation order, not error.
#define HDIM 512
#define BC 16
#define ATLAS_PREFILL_ENTRY inferspark_prefill_512tc
#define ATLAS_SKIP_PREFILL_64 1
#include "../../common/inferspark_prefill.cu"
