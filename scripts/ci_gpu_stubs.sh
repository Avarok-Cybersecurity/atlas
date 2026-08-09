#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Generate fail-fast GPU library stubs for no-GPU CI runners.
#
# spark-storage uses raw `extern "C"` FFI to libcuda (cuStreamCreate etc.)
# and spark-comm links libnccl directly, so linking the workspace needs
# these libraries present even when `ATLAS_SKIP_BUILD=1` avoids nvcc.
# Every stub symbol returns a non-success code (CUDA_ERROR_NO_DEVICE = 100,
# ncclSystemError = 2, cudart/cublasLt = 1) so any code path that actually
# invokes a GPU call fails-fast at runtime — matching a real no-GPU host.
# GPU tests are `#[ignore]`-gated, so the stubs exist purely to satisfy the
# linker.
set -euo pipefail

DEST=/usr/lib/x86_64-linux-gnu
CUDA_STUBS=/usr/local/cuda/targets/x86_64-linux/lib/stubs

cat > /tmp/libcuda_stub.c <<'EOF'
/* Stub for every CUDA driver API symbol that any Atlas crate
 * links against (spark-storage, atlas-core::registry, cudarc).
 * Each returns CUDA_ERROR_NO_DEVICE (100) so callers fail-fast.
 * Generated for CI link-time only; never exercised at runtime
 * because every test that touches these is #[ignore]-gated. */
/* Init / device / context */
int cuInit(unsigned int x) { (void)x; return 100; }
int cuDeviceGet(void *a, int b) { (void)a; (void)b; return 100; }
int cuCtxCreate(void **a, unsigned int b, int c) { (void)a; (void)b; (void)c; return 100; }
int cuCtxCreate_v2(void **a, unsigned int b, int c) { (void)a; (void)b; (void)c; return 100; }
int cuCtxDestroy_v2(void *a) { (void)a; return 100; }
int cuCtxGetCurrent(void **a) { (void)a; return 100; }
int cuCtxSetCurrent(void *a) { (void)a; return 100; }
int cuCtxGetDevice(int *a) { (void)a; return 100; }
int cuDeviceGetAttribute(int *a, unsigned int b, int c) { (void)a; (void)b; (void)c; return 100; }
/* Errors */
int cuGetErrorName(int code, const char **out) { (void)code; (void)out; return 100; }
int cuGetErrorString(int code, const char **out) { (void)code; (void)out; return 100; }
/* Streams */
int cuStreamCreate(void **a, unsigned int b) { (void)a; (void)b; return 100; }
int cuStreamDestroy_v2(void *a) { (void)a; return 100; }
int cuStreamSynchronize(void *a) { (void)a; return 100; }
int cuStreamQuery(void *a) { (void)a; return 100; }
int cuStreamBeginCapture(void *a, int b) { (void)a; (void)b; return 100; }
int cuStreamBeginCapture_v2(void *a, int b) { (void)a; (void)b; return 100; }
int cuStreamEndCapture(void *a, void **b) { (void)a; (void)b; return 100; }
int cuStreamIsCapturing(void *a, unsigned int *b) { (void)a; (void)b; return 100; }
int cuStreamWaitEvent(void *a, void *b, unsigned int c) { (void)a; (void)b; (void)c; return 100; }
/* Modules + launch */
int cuModuleLoadData(void **a, const void *b) { (void)a; (void)b; return 100; }
int cuModuleUnload(void *a) { (void)a; return 100; }
int cuModuleGetFunction(void **a, void *b, const char *c) { (void)a; (void)b; (void)c; return 100; }
int cuModuleGetGlobal_v2(unsigned long long *dptr, unsigned long *bytes,
                         void *hmod, const char *name) {
    (void)dptr; (void)bytes; (void)hmod; (void)name; return 100;
}
int cuFuncSetAttribute(void *f, int attr, int val) { (void)f; (void)attr; (void)val; return 100; }
int cuLaunchKernel(void *f, unsigned int gx, unsigned int gy, unsigned int gz,
                   unsigned int bx, unsigned int by, unsigned int bz,
                   unsigned int sm, void *s, void **p, void **e) {
    (void)f; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
    (void)sm; (void)s; (void)p; (void)e; return 100;
}
/* Events */
int cuEventCreate(void **a, unsigned int b) { (void)a; (void)b; return 100; }
int cuEventDestroy_v2(void *a) { (void)a; return 100; }
int cuEventRecord(void *a, void *b) { (void)a; (void)b; return 100; }
int cuEventSynchronize(void *a) { (void)a; return 100; }
int cuEventElapsedTime(float *a, void *b, void *c) { (void)a; (void)b; (void)c; return 100; }
/* Memory — device */
int cuMemAlloc(unsigned long long *a, unsigned long b) { (void)a; (void)b; return 100; }
int cuMemAlloc_v2(unsigned long long *a, unsigned long b) { (void)a; (void)b; return 100; }
int cuMemAllocManaged(unsigned long long *a, unsigned long b, unsigned int c) {
    (void)a; (void)b; (void)c; return 100;
}
int cuMemFree_v2(unsigned long long a) { (void)a; return 100; }
int cuMemGetInfo(unsigned long *a, unsigned long *b) { (void)a; (void)b; return 100; }
int cuMemGetInfo_v2(unsigned long *a, unsigned long *b) { (void)a; (void)b; return 100; }
/* Memory — host (pinned) */
int cuMemAllocHost(void **a, unsigned long b) { (void)a; (void)b; return 100; }
int cuMemAllocHost_v2(void **a, unsigned long b) { (void)a; (void)b; return 100; }
int cuMemFreeHost(void *a) { (void)a; return 100; }
int cuMemHostGetDevicePointer_v2(unsigned long long *a, void *b, unsigned int c) { (void)a; (void)b; (void)c; return 100; }
/* Memcpy / memset */
int cuMemcpyHtoD(unsigned long long a, const void *b, unsigned long c) {
    (void)a; (void)b; (void)c; return 100;
}
int cuMemcpyHtoD_v2(unsigned long long a, const void *b, unsigned long c) {
    (void)a; (void)b; (void)c; return 100;
}
int cuMemcpyHtoDAsync(unsigned long long a, const void *b, unsigned long c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 100;
}
int cuMemcpyHtoDAsync_v2(unsigned long long a, const void *b, unsigned long c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 100;
}
int cuMemcpyDtoHAsync_v2(void *a, unsigned long long b, unsigned long c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 100;
}
int cuMemcpyDtoDAsync_v2(unsigned long long a, unsigned long long b, unsigned long c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 100;
}
int cuMemsetD8Async(unsigned long long a, unsigned char b, unsigned long c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 100;
}
int cuMemsetD32Async(unsigned long long a, unsigned int b, unsigned long c, void *d) {
    (void)a; (void)b; (void)c; (void)d; return 100;
}
/* Graphs */
int cuGraphInstantiateWithFlags(void **a, void *b, unsigned long long c) {
    (void)a; (void)b; (void)c; return 100;
}
int cuGraphLaunch(void *a, void *b) { (void)a; (void)b; return 100; }
int cuGraphExecDestroy(void *a) { (void)a; return 100; }
int cuGraphDestroy(void *a) { (void)a; return 100; }
EOF
cc -shared -fPIC -nostdlib -o /tmp/libcuda.so /tmp/libcuda_stub.c
sudo install -m 0644 /tmp/libcuda.so "$DEST/libcuda.so"
# Some build scripts also probe /usr/local/cuda/.../stubs
sudo mkdir -p "$CUDA_STUBS"
sudo install -m 0644 /tmp/libcuda.so "$CUDA_STUBS/libcuda.so"

cat > /tmp/libnccl_stub.c <<'EOF'
/* Stub for NCCL collective-communication symbols spark-comm
 * links against. Each returns ncclSystemError (2) so any
 * caller fails-fast; never invoked at runtime in default
 * `cargo test` because every NCCL test is `#[ignore]`-gated. */
int ncclGetUniqueId(void *uid) { (void)uid; return 2; }
int ncclGetErrorString(int code) { (void)code; return 2; }
int ncclCommInitRank(void **comm, int n, void *uid, int rank) {
    (void)comm; (void)n; (void)uid; (void)rank; return 2;
}
int ncclCommInitRankConfig(void **comm, int n, void *uid, int rank, void *config) {
    (void)comm; (void)n; (void)uid; (void)rank; (void)config; return 2;
}
int ncclCommDestroy(void *comm) { (void)comm; return 2; }
int ncclCommAbort(void *comm) { (void)comm; return 2; }
int ncclCommGetAsyncError(void *comm, int *err) { (void)comm; (void)err; return 2; }
int ncclCommRegister(void *comm, void *p, unsigned long n, void **h) {
    (void)comm; (void)p; (void)n; (void)h; return 2;
}
int ncclCommDeregister(void *comm, void *h) { (void)comm; (void)h; return 2; }
int ncclMemAlloc(void **p, unsigned long n) { (void)p; (void)n; return 2; }
int ncclMemFree(void *p) { (void)p; return 2; }
int ncclGroupStart(void) { return 2; }
int ncclGroupEnd(void) { return 2; }
int ncclSend(const void *p, unsigned long n, int dt, int peer, void *comm, void *s) {
    (void)p; (void)n; (void)dt; (void)peer; (void)comm; (void)s; return 2;
}
int ncclRecv(void *p, unsigned long n, int dt, int peer, void *comm, void *s) {
    (void)p; (void)n; (void)dt; (void)peer; (void)comm; (void)s; return 2;
}
int ncclAllReduce(const void *a, void *b, unsigned long n, int dt, int op, void *comm, void *s) {
    (void)a; (void)b; (void)n; (void)dt; (void)op; (void)comm; (void)s; return 2;
}
int ncclAllGather(const void *a, void *b, unsigned long n, int dt, void *comm, void *s) {
    (void)a; (void)b; (void)n; (void)dt; (void)comm; (void)s; return 2;
}
int ncclReduceScatter(const void *a, void *b, unsigned long n, int dt, int op, void *comm, void *s) {
    (void)a; (void)b; (void)n; (void)dt; (void)op; (void)comm; (void)s; return 2;
}
int ncclBroadcast(const void *a, void *b, unsigned long n, int dt, int op, void *comm, void *s) {
    (void)a; (void)b; (void)n; (void)dt; (void)op; (void)comm; (void)s; return 2;
}
EOF
cc -shared -fPIC -nostdlib -o /tmp/libnccl.so /tmp/libnccl_stub.c
sudo install -m 0644 /tmp/libnccl.so "$DEST/libnccl.so"

# The Holo-3.1/Ornith enablement added cublaslt.rs (ATLAS_CUBLAS_GEMM FFI)
# and a cudaMemcpy2DAsync path; build.rs emits -lcublasLt/-lcudart even
# under ATLAS_SKIP_BUILD. Every symbol returns 1 so a real invocation
# fails-fast; GPU paths are #[ignore]-gated.
cat > /tmp/libcublaslt_stub.c <<'EOF'
int cublasLtCreate(void){return 1;}
int cublasLtDestroy(void){return 1;}
int cublasLtMatmul(void){return 1;}
int cublasLtMatmulAlgoGetHeuristic(void){return 1;}
int cublasLtMatmulDescCreate(void){return 1;}
int cublasLtMatmulDescDestroy(void){return 1;}
int cublasLtMatmulDescSetAttribute(void){return 1;}
int cublasLtMatmulPreferenceCreate(void){return 1;}
int cublasLtMatmulPreferenceDestroy(void){return 1;}
int cublasLtMatmulPreferenceSetAttribute(void){return 1;}
int cublasLtMatrixLayoutCreate(void){return 1;}
int cublasLtMatrixLayoutDestroy(void){return 1;}
EOF
cc -shared -fPIC -nostdlib -o /tmp/libcublasLt.so /tmp/libcublaslt_stub.c
sudo install -m 0644 /tmp/libcublasLt.so "$DEST/libcublasLt.so"

cat > /tmp/libcudart_stub.c <<'EOF'
int cudaMalloc(void){return 1;}
int cudaFree(void){return 1;}
int cudaMemcpy(void){return 1;}
int cudaMemcpy2DAsync(void){return 1;}
int cudaDeviceSynchronize(void){return 1;}
int cudaHostAlloc(void){return 1;}
EOF
cc -shared -fPIC -nostdlib -o /tmp/libcudart.so /tmp/libcudart_stub.c
sudo install -m 0644 /tmp/libcudart.so "$DEST/libcudart.so"
