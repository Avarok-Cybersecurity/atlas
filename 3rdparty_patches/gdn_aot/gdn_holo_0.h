
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} gdn_holo_0_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_gdn_holo_0_cuda_init(void **);
void _mlir_gdn_holo_0_cuda_load_to_device(void **);
static inline void gdn_holo_0_Kernel_Module_Load(gdn_holo_0_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_gdn_holo_0_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_gdn_holo_0_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void gdn_holo_0_Kernel_Module_Unload(gdn_holo_0_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} gdn_holo_0_Tensor_g_q_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} gdn_holo_0_Tensor_g_k_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} gdn_holo_0_Tensor_g_v_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} gdn_holo_0_Tensor_g_o_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
} gdn_holo_0_Tensor_g_alpha_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
} gdn_holo_0_Tensor_g_beta_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
} gdn_holo_0_Tensor_g_state_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
} gdn_holo_0_Tensor_g_init_state_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
} gdn_holo_0_Tensor_g_tensormaps_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
} gdn_holo_0_Tensor_cu_seqlens_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_gdn_holo_0__mlir_ciface_cutlass___call___flashinfergdn_kernelsdelta_rule_dsldelta_rule_sm120_FullyFusedDeltaRuleSm120_object_at__Tensorgmemoi641i64_Tensorgmemo1i64i64_Tensorgmemo1i64i64_Tensorgmemo1i64i64(void **args, int32_t num_args);

static inline int32_t cute_dsl_gdn_holo_0_wrapper(gdn_holo_0_Kernel_Module_t *module, gdn_holo_0_Tensor_g_q_t *g_q, gdn_holo_0_Tensor_g_k_t *g_k, gdn_holo_0_Tensor_g_v_t *g_v, gdn_holo_0_Tensor_g_o_t *g_o, gdn_holo_0_Tensor_g_alpha_t *g_alpha, gdn_holo_0_Tensor_g_beta_t *g_beta, gdn_holo_0_Tensor_g_state_t *g_state, gdn_holo_0_Tensor_g_init_state_t *g_init_state, gdn_holo_0_Tensor_g_tensormaps_t *g_tensormaps, gdn_holo_0_Tensor_cu_seqlens_t *cu_seqlens, float scale, int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads, int32_t num_sab_heads, int32_t num_seqs, int32_t total_checkpoints, int32_t checkpoint_every_n_tokens, int32_t grid_x, cudaStream_t stream) {
    int32_t ret;
    void *args[21] = {
        g_q, g_k, g_v, g_o, g_alpha, g_beta, g_state, g_init_state, g_tensormaps, cu_seqlens, &scale, &num_q_heads, &num_k_heads, &num_v_heads, &num_sab_heads, &num_seqs, &total_checkpoints, &checkpoint_every_n_tokens, &grid_x, &stream,
        &ret
    };
    _mlir_gdn_holo_0__mlir_ciface_cutlass___call___flashinfergdn_kernelsdelta_rule_dsldelta_rule_sm120_FullyFusedDeltaRuleSm120_object_at__Tensorgmemoi641i64_Tensorgmemo1i64i64_Tensorgmemo1i64i64_Tensorgmemo1i64i64(args, 21);
    return ret;
}
