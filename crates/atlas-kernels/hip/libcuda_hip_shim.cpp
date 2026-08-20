// SPDX-License-Identifier: AGPL-3.0-only
//
// libcuda → HIP shim. Re-exports the exact 33 CUDA driver-API symbols the
// Atlas spark binary imports (via cudarc) and implements each over HIP/ROCm.
// Built as `libcuda.so` and placed FIRST on the loader path so the unchanged
// cudarc runtime drives AMD GPUs natively — no SCALE.
//
// CUDA driver ABI ↔ HIP type compatibility:
//   CUdeviceptr (u64) ↔ hipDeviceptr_t (void*)   — cast
//   CUstream/CUmodule/CUfunction/CUevent/CUgraph(Exec) ↔ hip* — opaque ptrs
//   CUresult ↔ hipError_t — success==0 matches; error enums differ but cudarc
//   checks success and formats via cuGetErrorString (mapped to hipGetErrorString).
#include <hip/hip_runtime.h>
#include <cstring>
#include <mutex>
#include <unordered_map>

typedef unsigned long long CUdeviceptr;

// Windows device-memory accounting.
//
// hipMemGetInfo is BROKEN on the Windows HIP runtime: it returns
// hipErrorInvalidValue ("invalid argument") standalone, and reports free==0
// under a live context. Atlas sizes its KV cache from cuMemGetInfo, so a bogus
// 0-free makes serve fail with "No memory left for KV cache" even with tens of
// GB genuinely available (measured: 64 GB allocatable via a hipMalloc ladder).
//
// So track what we hand out and synthesise a truthful answer when HIP won't
// give one. Only engages when hipMemGetInfo actually fails or returns zeros,
// so Linux/ROCm behaviour is byte-identical to before.
static std::mutex g_mem_mu;
static std::unordered_map<void *, size_t> g_mem_sizes;
static size_t g_mem_used = 0;

static void atlas_track_alloc(void *p, size_t n) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_mem_mu);
    g_mem_sizes[p] = n;
    g_mem_used += n;
}

static void atlas_track_free(void *p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_mem_mu);
    auto it = g_mem_sizes.find(p);
    if (it == g_mem_sizes.end()) return;
    g_mem_used = (g_mem_used > it->second) ? g_mem_used - it->second : 0;
    g_mem_sizes.erase(it);
}

extern "C" {

// ── init / context ────────────────────────────────────────────────────
int cuCtxGetCurrent(void** pctx)            { return hipCtxGetCurrent((hipCtx_t*)pctx); }
int cuCtxSetCurrent(void* ctx)              { return hipCtxSetCurrent((hipCtx_t)ctx); }
int cuCtxCreate_v2(void** pctx, unsigned f, int dev)
                                            { return hipCtxCreate((hipCtx_t*)pctx, f, dev); }
int cuCtxDestroy_v2(void* ctx)              { return hipCtxDestroy((hipCtx_t)ctx); }

// ── errors ────────────────────────────────────────────────────────────
int cuGetErrorName(int err, const char** s)   { *s = hipGetErrorName((hipError_t)err);   return 0; }
int cuGetErrorString(int err, const char** s) { *s = hipGetErrorString((hipError_t)err); return 0; }

// ── memory ────────────────────────────────────────────────────────────
int cuMemAlloc_v2(CUdeviceptr* dptr, size_t n)      {
    int r = hipMalloc((void**)dptr, n);
    if (r == hipSuccess && dptr) atlas_track_alloc((void*)*dptr, n);
    return r;
}
int cuMemFree_v2(CUdeviceptr dptr)                  {
    atlas_track_free((void*)dptr);
    return hipFree((void*)dptr);
}
int cuMemAllocHost_v2(void** pp, size_t n)          { return hipHostMalloc(pp, n, 0); }
int cuMemFreeHost(void* p)                          { return hipHostFree(p); }
int cuMemAllocManaged(CUdeviceptr* dptr, size_t n, unsigned flags)
                                                    {
    int r = hipMallocManaged((void**)dptr, n, flags);
    if (r == hipSuccess && dptr) atlas_track_alloc((void*)*dptr, n);
    return r;
}
// See the g_mem_used comment above: fall back to totalGlobalMem minus tracked
// allocations whenever hipMemGetInfo errors or hands back a zero.
int cuMemGetInfo_v2(size_t* free, size_t* total)    {
    size_t f = 0, t = 0;
    hipError_t e = hipMemGetInfo(&f, &t);
    if (e == hipSuccess && t != 0 && f != 0) {
        if (free)  *free  = f;
        if (total) *total = t;
        return hipSuccess;
    }
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return (int)e;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return (int)e;
    size_t used;
    { std::lock_guard<std::mutex> lk(g_mem_mu); used = g_mem_used; }
    const size_t tot = prop.totalGlobalMem;
    if (total) *total = tot;
    if (free)  *free  = (used < tot) ? (tot - used) : 0;
    return hipSuccess;
}

int cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src, size_t n, void* s)
                              { return hipMemcpyHtoDAsync((hipDeviceptr_t)dst, (void*)src, n, (hipStream_t)s); }
int cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr src, size_t n, void* s)
                              { return hipMemcpyDtoHAsync(dst, (hipDeviceptr_t)src, n, (hipStream_t)s); }
int cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t n, void* s)
                              { return hipMemcpyDtoDAsync((hipDeviceptr_t)dst, (hipDeviceptr_t)src, n, (hipStream_t)s); }
int cuMemsetD8Async(CUdeviceptr dst, unsigned char uc, size_t n, void* s)
                              { return hipMemsetD8Async((hipDeviceptr_t)dst, uc, n, (hipStream_t)s); }
int cuMemsetD32Async(CUdeviceptr dst, unsigned int ui, size_t n, void* s)
                              { return hipMemsetD32Async((hipDeviceptr_t)dst, ui, n, (hipStream_t)s); }

// ── modules / kernels ─────────────────────────────────────────────────
int cuModuleLoadData(void** m, const void* image)          { return hipModuleLoadData((hipModule_t*)m, image); }
int cuModuleGetFunction(void** f, void* m, const char* nm) { return hipModuleGetFunction((hipFunction_t*)f, (hipModule_t)m, nm); }
// Fetch a __constant__/global symbol's device addr+size (registry::device_symbol).
int cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, void* m, const char* nm)
                              { return hipModuleGetGlobal((hipDeviceptr_t*)dptr, bytes, (hipModule_t)m, nm); }
int cuModuleUnload(void* m)                                { return hipModuleUnload((hipModule_t)m); }
// On NVIDIA this opts a module function into >48KB dynamic shared mem. On AMD
// the LDS is sized from hipModuleLaunchKernel's shared-mem arg (up to the
// RDNA 64KB cap), so no opt-in is needed — success no-op.
int cuFuncSetAttribute(void* f, int attr, int val)         { (void)f;(void)attr;(void)val; return 0; }
int cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz,
                   unsigned shmem, void* stream, void** params, void** extra) {
  return hipModuleLaunchKernel((hipFunction_t)f, gx, gy, gz, bx, by, bz,
                               shmem, (hipStream_t)stream, params, extra);
}

// ── streams ───────────────────────────────────────────────────────────
int cuStreamCreate(void** s, unsigned flags)        { return hipStreamCreateWithFlags((hipStream_t*)s, flags); }
int cuStreamSynchronize(void* s)                    { return hipStreamSynchronize((hipStream_t)s); }
// Re-added on the merge to main: main introduced a cuStreamQuery call in
// spark-runtime (copy_d2h_on_stream) after this shim was written.
int cuStreamQuery(void* s)                          { return hipStreamQuery((hipStream_t)s); }
int cuStreamWaitEvent(void* s, void* e, unsigned f) { return hipStreamWaitEvent((hipStream_t)s, (hipEvent_t)e, f); }
int cuStreamBeginCapture(void* s, int mode)         { return hipStreamBeginCapture((hipStream_t)s, (hipStreamCaptureMode)mode); }
int cuStreamEndCapture(void* s, void** pgraph)      { return hipStreamEndCapture((hipStream_t)s, (hipGraph_t*)pgraph); }

// ── events ────────────────────────────────────────────────────────────
int cuEventCreate(void** e, unsigned flags) { return hipEventCreateWithFlags((hipEvent_t*)e, flags); }
int cuEventDestroy_v2(void* e)              { return hipEventDestroy((hipEvent_t)e); }
int cuEventRecord(void* e, void* s)         { return hipEventRecord((hipEvent_t)e, (hipStream_t)s); }
int cuEventSynchronize(void* e)             { return hipEventSynchronize((hipEvent_t)e); }
int cuEventElapsedTime(float* ms, void* a, void* b) { return hipEventElapsedTime(ms, (hipEvent_t)a, (hipEvent_t)b); }

// ── CUDA graphs ───────────────────────────────────────────────────────
// cudarc's cuGraphInstantiate (legacy arity): (exec*, graph, errNode*, logBuf, bufSize)
int cuGraphInstantiate(void** pexec, void* graph, void** errNode, char* logBuf, size_t bufSize) {
  (void)errNode; (void)logBuf; (void)bufSize;
  return hipGraphInstantiate((hipGraphExec_t*)pexec, (hipGraph_t)graph, nullptr, nullptr, 0);
}
int cuGraphLaunch(void* exec, void* s)  { return hipGraphLaunch((hipGraphExec_t)exec, (hipStream_t)s); }
int cuGraphExecDestroy(void* exec)      { return hipGraphExecDestroy((hipGraphExec_t)exec); }
int cuGraphDestroy(void* graph)         { return hipGraphDestroy((hipGraph_t)graph); }

int cuGraphInstantiateWithFlags(void** pexec, void* graph, unsigned long long flags){ return hipGraphInstantiateWithFlags((hipGraphExec_t*)pexec,(hipGraph_t)graph,flags); }

int cuMemcpyHtoD_v2(unsigned long long d,const void*s,size_t n){return hipMemcpyHtoD((hipDeviceptr_t)d,(void*)s,n);}
int cuMemcpyDtoH_v2(void*d,unsigned long long s,size_t n){return hipMemcpyDtoH(d,(hipDeviceptr_t)s,n);}
int cuMemcpyDtoD_v2(unsigned long long d,unsigned long long s,size_t n){return hipMemcpyDtoD((hipDeviceptr_t)d,(hipDeviceptr_t)s,n);}
int cuMemsetD8_v2(unsigned long long d,unsigned char v,size_t n){return hipMemsetD8((hipDeviceptr_t)d,v,n);}
int cuMemsetD32_v2(unsigned long long d,unsigned int v,size_t n){return hipMemsetD32((hipDeviceptr_t)d,v,n);}
int cuMemHostAlloc(void**p,size_t n,unsigned int f){return hipHostMalloc(p,n,f);}
int cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int f)
                                            { return hipHostGetDevicePointer((void**)pdptr, p, f); }

// --- CudaContext::new path (cuDeviceGetAttribute maps CUDA enum NUMBERS to sane values, bypassing HIP enum mismatch) ---
int cuInit(unsigned int f){return hipInit(f);}
int cuDriverGetVersion(int*v){return hipDriverGetVersion(v);}
int cuDeviceGet(int*d,int o){return hipDeviceGet(d,o);}
int cuDeviceGetCount(int*c){return hipGetDeviceCount(c);}
int cuDeviceTotalMem_v2(size_t*b,int d){return hipDeviceTotalMem(b,d);}
int cuDevicePrimaryCtxRetain(void**c,int d){return hipDevicePrimaryCtxRetain((hipCtx_t*)c,d);}
int cuDevicePrimaryCtxRelease_v2(int d){return hipDevicePrimaryCtxRelease(d);}
int cuCtxSynchronize(void){return hipDeviceSynchronize();}
int cuCtxGetDevice(int*d){return hipGetDevice(d);}
int cuDeviceGetName(char*n,int len,int d){ if(len>0){const char*s="AMD-gfx1151"; int i=0; for(;i<len-1 && s[i];i++) n[i]=s[i]; n[i]=0;} return 0;}
int cuDeviceGetAttribute(int*v,int attr,int dev){ (void)dev; switch(attr){ case 75:*v=12;break; case 76:*v=1;break; case 16:*v=40;break; case 1:*v=1024;break; case 10:*v=32;break; case 8:*v=65536;break; case 18:*v=1;break; case 19:*v=1;break; case 41:*v=1;break; case 36:*v=1500;break; default:*v=0;break;} return 0;}

int cuStreamDestroy_v2(void*s){return hipStreamDestroy((hipStream_t)s);}
int cuStreamDestroy(void*s){return hipStreamDestroy((hipStream_t)s);}

} // extern "C"
