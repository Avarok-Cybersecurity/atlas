#pragma once
#include <hip/hip_fp8.h>
typedef __hip_fp8_storage_t __nv_fp8_storage_t;
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
typedef __hip_fp8x2_storage_t __nv_fp8x2_storage_t;
#define __NV_E4M3 __HIP_E4M3
#define __NV_E5M2 __HIP_E5M2
#define __NV_SATFINITE __HIP_SATFINITE
#define __NV_NOSAT __HIP_NOSAT
#define __nv_cvt_float2_to_fp8x2 __hip_cvt_float2_to_fp8x2
#define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
#define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw
#define __nv_cvt_bfloat16raw2_to_fp8x2 __hip_cvt_bfloat16raw2_to_fp8x2
#ifndef ATLAS_CVTA_COMPAT
#define ATLAS_CVTA_COMPAT
#define __cvta_generic_to_shared(p) ((unsigned long long)(size_t)(p))
#endif
