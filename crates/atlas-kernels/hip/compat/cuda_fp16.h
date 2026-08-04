#pragma once
// CUDA→HIP compat shim for cuda_fp16.h. HIP's hip_fp16.h provides the
// CUDA-compatible __half / __half2 types and __float2half / __half2float /
// __halves2half2 conversions, so most kernels need only the include redirect
// (mirrors cuda_bf16.h). Bare `half` alias added for kernels that use it.
#include <hip/hip_fp16.h>
#ifndef ATLAS_CUDA_FP16_COMPAT
#define ATLAS_CUDA_FP16_COMPAT
typedef __half half;
#endif
#ifndef ATLAS_CVTA_COMPAT
#define ATLAS_CVTA_COMPAT
#define __cvta_generic_to_shared(p) ((unsigned long long)(size_t)(p))
#endif
