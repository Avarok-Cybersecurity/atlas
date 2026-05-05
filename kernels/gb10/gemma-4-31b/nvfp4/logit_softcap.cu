// SPDX-License-Identifier: AGPL-3.0-only

// Logit softcapping for Gemma-4: logits = cap * tanh(logits / cap)
//
// Applied after LM head to bound logits magnitude.
// Gemma-4 uses cap=30.0. In-place operation on BF16 logits.
//
// Grid: (ceil(N/256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void logit_softcap_bf16(
    __nv_bfloat16* __restrict__ logits,
    unsigned int N,
    float inv_cap,   // 1.0 / cap
    float cap        // softcap value (e.g. 30.0)
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = __bfloat162float(logits[idx]);
    float y = cap * tanhf(x * inv_cap);
    logits[idx] = __float2bfloat16(y);
}

// FP32 in/out variant. Used for Gemma-4-31B dense whose 0.125-logit
// tiebreak at decode step 1 sits on a BF16 representable boundary; the
// pre-softcap rounding to BF16 flips the greedy argmax. Keeping
// logits FP32 from LM head through softcap and into the sampler
// avoids the boundary-rounding flip.
extern "C" __global__ void logit_softcap_fp32(
    float* __restrict__ logits,
    unsigned int N,
    float inv_cap,
    float cap
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = logits[idx];
    logits[idx] = cap * tanhf(x * inv_cap);
}
