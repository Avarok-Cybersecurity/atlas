// SPDX-License-Identifier: AGPL-3.0-only
//
// Vendored replacement for the 8 symbols gdn_holo_0.o imports from NVIDIA's
// proprietary 37 MB libcute_dsl_runtime.so.
//
// The CuTe-DSL AOT export (`compiled_fn.export_to_c`) emits host launch code
// that calls `_cuda*` / `_cu*` prefixed wrappers instead of the CUDA runtime
// directly. In libcute_dsl_runtime.so those wrappers are thin error-checked
// trampolines over the corresponding cudart/driver entry points (verified by
// disassembly: each `_cudaX` export at 0x17c8a0..0x17cc38 is a single `b`
// branch into a dispatcher that forwards the SAME registers to `cudaX`).
// Forwarding directly to cudart therefore preserves semantics exactly, and
// deletes the proprietary-runtime link dependency plus its RUNPATH coupling.
//
// SIGNATURES ARE LOAD-BEARING. They were confirmed two ways:
//   1. disassembly of the call sites in gdn_holo_0.o (objdump -dr): register
//      assignments match the cudart/driver prototypes argument-for-argument
//      (e.g. `_cudaDeviceGetAttribute(&v, 97 /*MaxSharedMemoryPerBlockOptin*/,
//      dev)`, `_cuKernelGetAttribute(&v, 1 /*SHARED_SIZE_BYTES*/, kernel, dev)`,
//      `_cudaLibraryLoadData(&lib, code, 0,0,0, 0,0,0)` — all 8 args);
//   2. the bit-parity gate (tests/gdn_aot_parity.rs) runs the SAME kernel via
//      this stub and via the original libcute_dsl_runtime.so-linked .so on
//      identical inputs and requires identical output.
// If a re-export ever changes what the .o imports, the static link fails loud
// at build time (undefined symbol) — that is the point of link-time vendoring.
//
// `_cuKernelGetAttribute` is a DRIVER symbol. It is resolved lazily through
// dlopen("libcuda.so.1") rather than link time so the binary carries no
// DT_NEEDED on libcuda: the rest of Atlas loads the driver dynamically
// (cudarc), and a hard driver dependency would break CPU-only hosts that
// merely build/link this target.

#include <cuda_runtime.h>
#include <dlfcn.h>

extern "C" {

int _cudaLibraryLoadData(cudaLibrary_t* library, const void* code,
                         cudaJitOption* jit_options, void** jit_option_values,
                         unsigned int num_jit_options,
                         cudaLibraryOption* library_options,
                         void** library_option_values,
                         unsigned int num_library_options) {
  return (int)cudaLibraryLoadData(library, code, jit_options, jit_option_values,
                                  num_jit_options, library_options,
                                  library_option_values, num_library_options);
}

int _cudaLibraryGetKernel(cudaKernel_t* kernel, cudaLibrary_t library,
                          const char* name) {
  return (int)cudaLibraryGetKernel(kernel, library, name);
}

int _cudaGetDevice(int* device) { return (int)cudaGetDevice(device); }

int _cudaGetDeviceCount(int* count) { return (int)cudaGetDeviceCount(count); }

int _cudaDeviceGetAttribute(int* value, int attr, int device) {
  return (int)cudaDeviceGetAttribute(value, (cudaDeviceAttr)attr, device);
}

// `func` is a cudaKernel_t from _cudaLibraryGetKernel; cudart accepts kernel
// handles here (documented since CUDA 12.1).
int _cudaFuncSetAttribute(const void* func, int attr, int value) {
  return (int)cudaFuncSetAttribute(func, (cudaFuncAttribute)attr, value);
}

int _cudaKernelSetAttributeForDevice(cudaKernel_t kernel, int attr, int value,
                                     int device) {
  return (int)cudaKernelSetAttributeForDevice(kernel, (cudaFuncAttribute)attr,
                                              value, device);
}

int _cudaLaunchKernelEx(const cudaLaunchConfig_t* config, const void* func,
                        void** args) {
  return (int)cudaLaunchKernelExC(config, func, args);
}

// Driver API, lazily resolved (see file header). 999 = CUDA_ERROR_UNKNOWN —
// the .o checks the return and skips the dependent attribute writes, and the
// shim surfaces the nonzero wrapper result to Rust, which bails.
int _cuKernelGetAttribute(int* pi, int attrib, void* kernel, int dev) {
  typedef int (*cu_kernel_get_attribute_t)(int*, int, void*, int);
  static cu_kernel_get_attribute_t fn = []() -> cu_kernel_get_attribute_t {
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) return nullptr;
    return (cu_kernel_get_attribute_t)dlsym(h, "cuKernelGetAttribute");
  }();
  if (!fn) return 999;
  return fn(pi, attrib, kernel, dev);
}

}  // extern "C"
