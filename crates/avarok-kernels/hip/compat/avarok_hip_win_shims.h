#pragma once
// SPDX-License-Identifier: AGPL-3.0-only
//
// Windows HIP SDK (ROCm 6.4 clang) compatibility shims.
//
// The Windows HIP headers do NOT declare the CUDA-style, mask-argument warp
// intrinsics (`__shfl_*_sync`, `__any_sync`, `__all_sync`, `__activemask`)
// that Linux ROCm provides, so Atlas's unmodified CUDA kernels fail to compile
// there with hundreds of `use of undeclared identifier '__shfl_down_sync'`.
// Each is mapped onto the base HIP intrinsic (`__shfl*`, `__any`, `__all`,
// `__ballot`); the warp mask is advisory on AMD's single-wavefront execution
// and is dropped. Two overloads per shuffle (with/without an explicit `width`)
// so we never lean on `warpSize` in a default-argument expression.
//
// Force-included ONLY on the Windows HIP build (see build_target.rs's
// `cfg!(windows)` branch). Linux ROCm already declares these, so it never sees
// this header and there is no redefinition. NVIDIA (nvcc) and SCALE builds do
// not use it at all.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)

template <typename T>
__device__ __forceinline__ T __shfl_sync(unsigned long long, T v, int src_lane) {
    return __shfl(v, src_lane);
}
template <typename T>
__device__ __forceinline__ T __shfl_sync(unsigned long long, T v, int src_lane, int width) {
    return __shfl(v, src_lane, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_up_sync(unsigned long long, T v, unsigned int delta) {
    return __shfl_up(v, delta);
}
template <typename T>
__device__ __forceinline__ T __shfl_up_sync(unsigned long long, T v, unsigned int delta, int width) {
    return __shfl_up(v, delta, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_down_sync(unsigned long long, T v, unsigned int delta) {
    return __shfl_down(v, delta);
}
template <typename T>
__device__ __forceinline__ T __shfl_down_sync(unsigned long long, T v, unsigned int delta, int width) {
    return __shfl_down(v, delta, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned long long, T v, int lane_mask) {
    return __shfl_xor(v, lane_mask);
}
template <typename T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned long long, T v, int lane_mask, int width) {
    return __shfl_xor(v, lane_mask, width);
}

__device__ __forceinline__ int __any_sync(unsigned long long, int pred) { return __any(pred); }
__device__ __forceinline__ int __all_sync(unsigned long long, int pred) { return __all(pred); }

// CUDA returns a 32-bit lane mask; AMD wavefronts are 64-wide, so __ballot(1)
// (the full active-lane mask) is the faithful analogue.
__device__ __forceinline__ unsigned long long __activemask() { return __ballot(1); }

#endif  // __HIP_PLATFORM_AMD__ || __HIP__
