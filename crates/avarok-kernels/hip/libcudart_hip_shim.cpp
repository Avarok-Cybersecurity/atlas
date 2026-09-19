// SPDX-License-Identifier: AGPL-3.0-only
//
// cudart → HIP shim. spark-runtime/spark-storage link a handful of CUDA
// *runtime* API symbols (distinct from the cu* driver API in
// libcuda_hip_shim.cpp) for host-side allocation and copies. Each maps 1:1
// onto HIP. Built to libcudart.so on Linux and archived into cudart.lib on
// Windows so `-lcudart` resolves to a real HIP-backed implementation.
//
// cudaMemcpyKind is ABI-compatible with hipMemcpyKind (HIP mirrors the CUDA
// enum values on purpose), so the `kind` int passes straight through.
#include <hip/hip_runtime.h>

extern "C" {

int cudaMalloc(void** ptr, size_t size) { return hipMalloc(ptr, size); }
int cudaFree(void* ptr) { return hipFree(ptr); }

int cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return hipHostMalloc(ptr, size, flags);
}

int cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
    return hipMemcpy(dst, src, count, (hipMemcpyKind)kind);
}

int cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                      size_t width, size_t height, int kind, void* stream) {
    return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                            (hipMemcpyKind)kind, (hipStream_t)stream);
}

int cudaDeviceSynchronize() { return hipDeviceSynchronize(); }

}  // extern "C"
