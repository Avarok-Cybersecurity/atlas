# SCALE 1.7.0 — trivial CUDA program fails on gfx1151 (Radeon 8060S / Strix Halo)

**Box:** AMD Ryzen AI MAX+ 395, Radeon 8060S iGPU (gfx1151), native
Ubuntu 24.04, kernel 6.17.0-1023-oem, ROCm 7.13 (amdrocm channel).
`/dev/kfd` + `/dev/dri/renderD128` present; user in `render`+`video`;
`rocminfo` cleanly enumerates Agent gfx1151 (40 CUs).

**SCALE:** 1.7.0-Linux, `targets/gfx1151/` toolchain.

## Repro

```cpp
// tiny.cu
#include <cstdio>
extern "C" __global__ void tinykernel(int* p) { if (p) *p = 42; }
int main() {
    int* d = nullptr;
    cudaMalloc(&d, sizeof(int));
    tinykernel<<<1,1>>>(d);
    cudaDeviceSynchronize();
    int h = 0; cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("result: %d\n", h);
}
```

```
nvcc tiny.cu -o tiny_full     # SCALE targets/gfx1151/bin/nvcc — compiles clean
./tiny_full
```

## Failure (depends on which HSA runtime is loaded)

- `LD_LIBRARY_PATH=/opt/rocm/lib` (ROCm 7.13 HSA): throws
  `scale::SimpleException — cudaMalloc: No usable CUDA devices found.,
  CUDA error: "no device"` — despite rocminfo seeing the agent.
- `LD_LIBRARY_PATH=<scale>/targets/gfx1151/lib` (SCALE's bundled
  libhsa-runtime64 + libhsakmt): **SIGSEGV** (core dumped).

Also seen on every run: `Warning: Could not enable debug trap`.

## Questions for SCALE/Spectral

1. Is gfx1151 (RDNA3.5 Strix Halo iGPU) supported by the SCALE 1.7.0
   *runtime* (libredscale), or only the compiler toolchain?
2. Which HSA runtime is SCALE 1.7.0 built/tested against — does it
   require a specific ROCm version rather than ROCm 7.13?
3. What does `cuModuleLoadData` accept as a code-object format (relevant
   once device init works — Atlas loads kernels via `cuModuleLoadData`)?

## Backtrace — the SIGSEGV case (SCALE's bundled libhsa-runtime64)

SIGSEGV inside the HSA runtime during GPU queue creation:

```
#0  rocr::AMD::GpuAgent::ReleaseQueueMainScratch(ScratchCache::ScratchInfo&)  libhsa-runtime64.so.1
#1  rocr::AMD::GpuAgent::QueueCreate(...)                                     libhsa-runtime64.so.1
#2  rocr::AMD::GpuAgent::InitDma()                                            libhsa-runtime64.so.1
#5  rocr::HSA::hsa_queue_create(...)                                          libhsa-runtime64.so.1
#6-#13  libredscale.so
#14 cudaMalloc                                                                libredscale.so
#15 main
```

The crash PC is inside SCALE's bundled `libhsa-runtime64` (ROCr) — DMA
queue / scratch-cache setup. SCALE's runtime cannot create a GPU queue
on gfx1151. This strongly indicates SCALE 1.7.0's bundled ROCr predates
or lacks gfx1151 (Strix Halo / RDNA3.5) queue+scratch support — the
compiler toolchain targets gfx1151 fine, but the runtime does not.

## Bottom line

Atlas builds and links cleanly on Strix Halo via SCALE 1.7.0 and runs to
GPU init. It is blocked entirely on SCALE's runtime failing to bring up
the gfx1151 device — a 5-line CUDA program reproduces it. Needs a SCALE
build whose ROCr supports gfx1151.

## Native ROCm works — isolates the bug to SCALE

`amdrocm-blas-test-gfx1151` `rocblas-bench -f gemm -r f32_r -m 512 -n 512 -k 512`
runs a native rocBLAS SGEMM on gfx1151: **1296.7 GFLOPS, 207 us, exit 0.**
So the GPU, amdkfd driver, and ROCm 7.13 are all healthy for native
compute — the failure is specific to SCALE 1.7.0's runtime.
