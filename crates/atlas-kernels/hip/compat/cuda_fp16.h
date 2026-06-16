#pragma once
// CUDA cuda_fp16.h -> HIP fp16 compat shim. HIP's hip_fp16.h provides __half,
// __half2, and the __float2half/__half2float/__hadd/... family under the same
// names CUDA uses, so a straight include suffices for kernels that only need
// half storage + scalar conversions (e.g. vision_encoder.cu).
#include <hip/hip_fp16.h>
#ifndef ATLAS_CVTA_COMPAT
#define ATLAS_CVTA_COMPAT
#define __cvta_generic_to_shared(p) ((unsigned long long)(size_t)(p))
#endif
