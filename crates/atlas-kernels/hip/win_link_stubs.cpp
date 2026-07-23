// SPDX-License-Identifier: AGPL-3.0-only
//
// Windows HIP build: link-only stub definitions.
//
// spark-runtime/spark-storage emit `-lcuda`, `-lcudart` and `-lcublasLt` for
// their direct CUDA driver/runtime/cuBLASLt FFI. On Linux the libcuda->HIP
// shim (libcuda_hip_shim.cpp) plus SCALE's CUDA-compat libs satisfy these; on
// Windows there is no such provider. Hosted CI runners have NO AMD GPU, so the
// windows/amd-hip target is COMPILE-ONLY — the binary is never executed. These
// stubs exist purely so the link resolves. Each is an `extern "C"` symbol with
// C linkage (x64 MSVC: undecorated), matched by NAME at link time; arguments
// are irrelevant to symbol resolution. cu*/cuda* return 0 (CUDA_SUCCESS);
// cublasLt* return 1 (a non-success status) so any accidental runtime call
// fails loudly rather than silently "succeeding".
//
// If the build ever needs a RUNNABLE Windows AMD binary, replace these with
// real HIP / hipBLASLt mappings (and link amdhip64) — see README-HIP.md.
extern "C" {
int cuInit() { return 0; }
int cuDriverGetVersion() { return 0; }
int cuCtxGetCurrent() { return 0; }
int cuCtxSetCurrent() { return 0; }
int cuCtxGetDevice() { return 0; }
int cuCtxSynchronize() { return 0; }
int cuCtxCreate() { return 0; }
int cuCtxCreate_v2() { return 0; }
int cuCtxDestroy_v2() { return 0; }
int cuDeviceGet() { return 0; }
int cuDeviceGetAttribute() { return 0; }
int cuDeviceGetCount() { return 0; }
int cuDeviceGetName() { return 0; }
int cuDevicePrimaryCtxRelease_v2() { return 0; }
int cuDevicePrimaryCtxRetain() { return 0; }
int cuDeviceTotalMem_v2() { return 0; }
int cuGetErrorName() { return 0; }
int cuGetErrorString() { return 0; }
int cuEventCreate() { return 0; }
int cuEventDestroy_v2() { return 0; }
int cuEventElapsedTime() { return 0; }
int cuEventRecord() { return 0; }
int cuEventSynchronize() { return 0; }
int cuFuncSetAttribute() { return 0; }
int cuGraphDestroy() { return 0; }
int cuGraphExecDestroy() { return 0; }
int cuGraphInstantiate() { return 0; }
int cuGraphInstantiateWithFlags() { return 0; }
int cuGraphLaunch() { return 0; }
int cuLaunchKernel() { return 0; }
int cuMemAlloc() { return 0; }
int cuMemAlloc_v2() { return 0; }
int cuMemAllocHost() { return 0; }
int cuMemAllocHost_v2() { return 0; }
int cuMemAllocManaged() { return 0; }
int cuMemFree() { return 0; }
int cuMemFree_v2() { return 0; }
int cuMemFreeHost() { return 0; }
int cuMemGetInfo() { return 0; }
int cuMemGetInfo_v2() { return 0; }
int cuMemHostAlloc() { return 0; }
int cuMemHostGetDevicePointer_v2() { return 0; }
int cuMemcpyDtoD_v2() { return 0; }
int cuMemcpyDtoDAsync_v2() { return 0; }
int cuMemcpyDtoH_v2() { return 0; }
int cuMemcpyDtoHAsync_v2() { return 0; }
int cuMemcpyHtoD() { return 0; }
int cuMemcpyHtoD_v2() { return 0; }
int cuMemcpyHtoDAsync() { return 0; }
int cuMemcpyHtoDAsync_v2() { return 0; }
int cuMemsetD8_v2() { return 0; }
int cuMemsetD8Async() { return 0; }
int cuMemsetD32_v2() { return 0; }
int cuMemsetD32Async() { return 0; }
int cuModuleGetFunction() { return 0; }
int cuModuleGetGlobal_v2() { return 0; }
int cuModuleLoadData() { return 0; }
int cuModuleUnload() { return 0; }
int cuStreamCreate() { return 0; }
int cuStreamDestroy() { return 0; }
int cuStreamDestroy_v2() { return 0; }
int cuStreamSynchronize() { return 0; }
int cuStreamWaitEvent() { return 0; }
int cuStreamBeginCapture() { return 0; }
int cuStreamBeginCapture_v2() { return 0; }
int cuStreamEndCapture() { return 0; }
int cuStreamIsCapturing() { return 0; }
int cudaMalloc() { return 0; }
int cudaFree() { return 0; }
int cudaHostAlloc() { return 0; }
int cudaMemcpy() { return 0; }
int cudaMemcpy2DAsync() { return 0; }
int cudaDeviceSynchronize() { return 0; }
int cublasLtCreate() { return 1; }
int cublasLtDestroy() { return 1; }
int cublasLtMatmul() { return 1; }
int cublasLtMatmulAlgoGetHeuristic() { return 1; }
int cublasLtMatmulDescCreate() { return 1; }
int cublasLtMatmulDescDestroy() { return 1; }
int cublasLtMatmulDescSetAttribute() { return 1; }
int cublasLtMatmulPreferenceCreate() { return 1; }
int cublasLtMatmulPreferenceDestroy() { return 1; }
int cublasLtMatmulPreferenceSetAttribute() { return 1; }
int cublasLtMatrixLayoutCreate() { return 1; }
int cublasLtMatrixLayoutDestroy() { return 1; }
}  // extern "C"
