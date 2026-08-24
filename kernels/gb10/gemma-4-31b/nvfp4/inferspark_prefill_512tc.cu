// SPDX-License-Identifier: AGPL-3.0-only
//
// Tensor-core prefill attention at HDIM=512, for Gemma-4-31B's global layers.
//
// Identical instantiation to the gemma-4-26b-a4b one: this model has the same
// `global_head_dim = 512` and the same 256-wide sliding layers, so it was on the
// same scalar `inferspark_prefill_512` path.
//
// ★ IT IS ALSO REQUIRED, not merely desirable. `ops::wide_prefill_kernel` probes
// for this entry, and `--check-kernels` refuses to serve a target where a lookup
// resolves to handle 0 — "its dispatch site is on a silent fallback path". Adding
// the instantiation is the fix that error asks for; the alternative is declaring
// it `expected_absent` and leaving this model on the scalar kernel.
#define HDIM 512
#define BC 16
#define ATLAS_PREFILL_ENTRY inferspark_prefill_512tc
#define ATLAS_SKIP_PREFILL_64 1
#include "../../common/inferspark_prefill.cu"
