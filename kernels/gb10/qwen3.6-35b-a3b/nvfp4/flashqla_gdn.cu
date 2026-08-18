// SPDX-License-Identifier: MIT
// FlashQLA 0.1.2 device kernels frozen for Atlas source integration.
// Generated device code only; TVM/TileLang host wrappers are intentionally excluded.
#define ENABLE_BF16
#include <cuda_runtime.h>
#include <cuda.h>
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>

// Atlas-native GPU gate unpack. The production BA+gate kernel writes log-space values.
extern "C" __global__ void flashqla_unpack_gate_beta(const float* src, float* gate, float* beta, int total, int nv, int stride, int input_is_log) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int count = total * nv;
  if (idx >= count) return;
  int token = idx / nv;
  int head = idx - token * nv;
  float value = src[(size_t)token * stride + head];
  if (!input_is_log) value = logf(fmaxf(value, 1.0e-30f));
  gate[idx] = value;
  beta[idx] = src[(size_t)token * stride + nv + head];
}
// ---- flashqla_kkt_solve ----
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void flashqla_kkt_solve(bfloat16_t* __restrict__ a, const float* __restrict__ b, const int64_t* __restrict__ chunk_indices, const int64_t* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ k, int data_batch_size, int num_chunks, int num_tokens, int real_batch_size);
extern "C" __global__ void __launch_bounds__(256, 1) flashqla_kkt_solve(bfloat16_t* __restrict__ a, const float* __restrict__ b, const int64_t* __restrict__ chunk_indices, const int64_t* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ k, int data_batch_size, int num_chunks, int num_tokens, int real_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t k_is_ready_mem[1];
  auto k_is_ready = reinterpret_cast<Barrier*>(k_is_ready_mem);
  __shared__ __align__(16) uint64_t a_is_ready_mem[1];
  auto a_is_ready = reinterpret_cast<Barrier*>(a_is_ready_mem);
  int batch_idx = 0;
  int chunk_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  __shared__ __align__(16) float b_shared[32];
  float a32_fragment[8];
  float a16o_fragment[2];
  float a16i_row[16];
  float a16i_sum[1];
  float a32_shared_local_cast_1[4];
  bfloat16_t a_local_cast[4];
  float a32_shared_local_cast_3[4];
  bfloat16_t a_local_cast_2[4];
  if (tl::tl_shuffle_elect<0>()) {
    k_is_ready[0].init(32);
    a_is_ready[0].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = ((int)chunk_indices[((((int64_t)((int)blockIdx.x)) >> (int64_t)5) * (int64_t)2)]);
  chunk_idx = ((int)chunk_indices[(((((int64_t)((int)blockIdx.x)) >> (int64_t)5) * (int64_t)2) + (int64_t)1)]);
  int64_t condval;
  if (((0 <= batch_idx) && (batch_idx <= real_batch_size))) {
    condval = cu_seqlens[batch_idx];
  } else {
    condval = (int64_t)0;
  }
  seq_start_idx = ((int)condval);
  int64_t condval_1;
  if (((-1 <= batch_idx) && (batch_idx < real_batch_size))) {
    condval_1 = cu_seqlens[(((int64_t)batch_idx) + (int64_t)1)];
  } else {
    condval_1 = (int64_t)0;
  }
  seq_end_idx = ((int)condval_1);
  int chunk_idx_1 = chunk_idx;
  int seq_start_idx_1 = seq_start_idx;
  int seq_end_idx_1 = seq_end_idx;
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<64>();
    if ((((chunk_idx_1 * 32) + seq_start_idx_1) + 32) <= seq_end_idx_1) {
      if (((int)threadIdx.x) < 32) {
        float condval_2;
        if (((0 <= (((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x))) && ((((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens))) {
          condval_2 = b[((((((int64_t)chunk_idx_1) * (int64_t)1024) + (((int64_t)seq_start_idx_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) & (int64_t)31))];
        } else {
          condval_2 = 0x0p+0f/*0.000000e+00*/;
        }
        b_shared[((int)threadIdx.x)] = condval_2;
      }
    } else {
      if (((int)threadIdx.x) < 32) {
        if ((((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x)) < seq_end_idx_1) {
          float condval_3;
          if (((0 <= (((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x))) && ((((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens))) {
            condval_3 = b[((((((int64_t)chunk_idx_1) * (int64_t)1024) + (((int64_t)seq_start_idx_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) & (int64_t)31))];
          } else {
            condval_3 = 0x0p+0f/*0.000000e+00*/;
          }
          b_shared[((int)threadIdx.x)] = condval_3;
        } else {
          b_shared[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
        }
      }
    }
    k_is_ready[0].wait(0);
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[8];
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        float broadcast_var = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(a32_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      }
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki >> 2) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a32_fragment + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a32_fragment + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      float2 __1;
        float2 v_ = *(float2*)(a32_fragment + (i_1 * 2));
        float2 v__1 = make_float2(b_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_1 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], b_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_1 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      *(float2*)(a32_fragment + (i_1 * 2)) = __1;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      if ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_2 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < (((((((int)threadIdx.x) >> 6) * 16) + ((i_2 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_2 & 1))) {
        a32_fragment[i_2] = 0x0p+0f/*0.000000e+00*/;
      } else {
        if ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_2 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == (((((((int)threadIdx.x) >> 6) * 16) + ((i_2 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_2 & 1))) {
          a32_fragment[i_2] = 0x1p+0f/*1.000000e+00*/;
        }
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      if (((((int)threadIdx.x) & 63) >> 5) == ((((int)threadIdx.x) >> 6) + 1)) {
        float broadcast_var_1 = -0x1p+0f/*-1.000000e+00*/;
        float2 __2;
          float2 v__2 = *(float2*)(a32_fragment + (i_3 * 2));
          float2 v__3 = make_float2(broadcast_var_1, broadcast_var_1);
          *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
        *(float2*)(((float*)buf_dyn_shmem) + ((((((i_3 & 1) * 128) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 3616)) = __2;
      } else {
        if (((((int)threadIdx.x) & 63) >> 5) == (((int)threadIdx.x) >> 6)) {
          *(float2*)(((float*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 63) >> 5) * 272) + ((i_3 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 3072)) = *(float2*)(a32_fragment + (i_3 * 2));
        }
      }
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int k_s = 1; k_s < 16; ++k_s) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        for (int vec_s = 0; vec_s < 4; ++vec_s) {
          if (((i_4 * 4) + vec_s) < k_s) {
            a16i_row[((i_4 * 4) + vec_s)] = ((float*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_s * 16)) + (i_4 * 4)) + vec_s) + 3072)];
          }
        }
      }
      a16i_sum[0] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int k_r = 0; k_r < k_s; ++k_r) {
        a16i_sum[0] = (a16i_sum[0] - (((float*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_r * 16)) + (((int)threadIdx.x) & 15)) + 3072)] * a16i_row[k_r]));
      }
      tl::__sync_thread_partial<3, 128>();
      if ((((int)threadIdx.x) >> 5) == 0) {
        if ((((int)threadIdx.x) & 15) < k_s) {
          ((float*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_s * 16)) + (((int)threadIdx.x) & 15)) + 3072)] = a16i_sum[0];
        }
      }
    }
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(a16o_fragment + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int k_r_1 = 0; k_r_1 < 16; ++k_r_1) {
      float2 __3;
        float2 v__4 = make_float2(((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 16) + k_r_1) + 3344)], ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 16) + k_r_1) + 3344)]);
        float2 v__5 = *(float2*)(((float*)buf_dyn_shmem) + (((k_r_1 * 16) + ((((int)threadIdx.x) & 7) * 2)) + 3616));
        float2 v__6 = *(float2*)(a16o_fragment + 0);
        *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(a16o_fragment + 0) = __3;
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 7) * 32) + (i_5 * 16)) + (((int)threadIdx.x) >> 3)) + 3616)] = a16o_fragment[i_5];
    }
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(a16o_fragment + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int k_r_2 = 0; k_r_2 < 16; ++k_r_2) {
      float2 __4;
        float2 v__7 = make_float2(((float*)buf_dyn_shmem)[(((k_r_2 * 16) + (((int)threadIdx.x) >> 3)) + 3616)], ((float*)buf_dyn_shmem)[(((k_r_2 * 16) + (((int)threadIdx.x) >> 3)) + 3616)]);
        float2 v__8 = *(float2*)(((float*)buf_dyn_shmem) + (((k_r_2 * 16) + ((((int)threadIdx.x) & 7) * 2)) + 3072));
        float2 v__9 = *(float2*)(a16o_fragment + 0);
        *(float2*)(&(__4.x)) = tl::fma2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)), *(float2*)(&(v__9.x)));
      *(float2*)(a16o_fragment + 0) = __4;
    }
    tl::__sync_thread_partial<3, 128>();
    *(float2*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 3616)) = *(float2*)(a16o_fragment + 0);
    *(float2*)(((float*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 2048)) = *(float2*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 3072));
    float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(((float*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 2064)) = make_float2(broadcast_var_4, broadcast_var_4);
    *(float2*)(((float*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 2560)) = *(float2*)(a16o_fragment + 0);
    *(float2*)(((float*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 2576)) = *(float2*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 3344));
    a_is_ready[0].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    if (((int)threadIdx.x) < 160) {
      tl::__sync_thread_partial<4, 32>();
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_4;
        if (((((((chunk_idx_1 * 32) + (i_6 * 2)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < (num_tokens + 8)) && (1 <= ((chunk_idx_1 * 4) + ((((i_6 * 2) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) >> 3))))) {
          condval_4 = *(uint4*)(k + (((((((((int64_t)chunk_idx_1) * (int64_t)65536) + (((int64_t)i_6) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)seq_start_idx_1) * (int64_t)2048)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)31) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)16384));
        } else {
          condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((i_6 >> 2) * 512)) + ((((i_6 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_6) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_6) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_4;
      }
      k_is_ready[0].arrive();
    } else {
      if (((int)threadIdx.x) < 192) {
        a_is_ready[0].wait(0);
        if ((((chunk_idx_1 * 32) + seq_start_idx_1) + 32) <= seq_end_idx_1) {
          #pragma unroll
          for (int i_7 = 0; i_7 < 8; ++i_7) {
            if ((((((chunk_idx_1 * 32) + (i_7 * 4)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < (num_tokens + 20)) && (20 <= ((((chunk_idx_1 * 32) + (i_7 * 4)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1))) {
              *(float4*)(a32_shared_local_cast_1 + 0) = *(float4*)(((float*)buf_dyn_shmem) + (((i_7 * 128) + (((int)threadIdx.x) * 4)) + 1408));
              uint2 __5;
              float4 v__10 = *(float4*)(a32_shared_local_cast_1 + 0);
              (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__10))[0]);
              (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__10))[1]);
              *(uint2*)(a_local_cast + 0) = __5;
              *(uint2*)(a + (((((((((int64_t)chunk_idx_1) * (int64_t)32768) + (((int64_t)i_7) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)1024)) + (((int64_t)seq_start_idx_1) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)31) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4)) - (int64_t)20480)) = *(uint2*)(a_local_cast + 0);
            }
          }
        }
      } else {
        a_is_ready[0].wait(0);
        if (seq_end_idx_1 < (((chunk_idx_1 * 32) + seq_start_idx_1) + 32)) {
          #pragma unroll
          for (int i_8 = 0; i_8 < 4; ++i_8) {
            if (((((chunk_idx_1 * 32) + (i_8 * 8)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < (seq_end_idx_1 + 24)) {
              *(float4*)(a32_shared_local_cast_3 + 0) = *(float4*)(((float*)buf_dyn_shmem) + (((i_8 * 256) + (((int)threadIdx.x) * 4)) + 1280));
              uint2 __6;
              float4 v__11 = *(float4*)(a32_shared_local_cast_3 + 0);
              (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
              (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&v__11))[1]);
              *(uint2*)(a_local_cast_2 + 0) = __6;
              if (24 <= ((((chunk_idx_1 * 32) + (i_8 * 8)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1)) {
                if (((((chunk_idx_1 * 32) + (i_8 * 8)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < (num_tokens + 24)) {
                  *(uint2*)(a + (((((((((int64_t)chunk_idx_1) * (int64_t)32768) + (((int64_t)i_8) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)1024)) + (((int64_t)seq_start_idx_1) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)31) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4)) - (int64_t)24576)) = *(uint2*)(a_local_cast_2 + 0);
                }
              }
            }
          }
        }
      }
    }
  }
}

// ---- flashqla_fused_nocp ----
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void flashqla_fused_nocp(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cp_seq_map, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const int64_t* __restrict__ raw_cu_seqlens, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(512, 1) flashqla_fused_nocp(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cp_seq_map, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const int64_t* __restrict__ raw_cu_seqlens, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t data_is_ready_mem[2];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[2];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_o_mem[1];
  auto bar_o = reinterpret_cast<Barrier*>(bar_o_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t _bar_2_mem[1];
  auto _bar_2 = reinterpret_cast<Barrier*>(_bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  __shared__ __align__(16) uint64_t bar_4_mem[1];
  auto bar_4 = reinterpret_cast<Barrier*>(bar_4_mem);
  __shared__ __align__(16) uint64_t bar_5_mem[1];
  auto bar_5 = reinterpret_cast<Barrier*>(bar_5_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int raw_batch_idx = 0;
  int raw_seq_end_idx = 0;
  signed char need_store_final_state = (signed char)0;
  int num_iters = 0;
  int num_unmasked_iters = 0;
  float h_fragment[64];
  __shared__ __align__(16) float g_exp_shared[32];
  __shared__ __align__(16) float g_shared[64];
  __shared__ __align__(16) float b_shared[64];
  int seq_split_idx = 0;
  int chunk_split_idx = 0;
  float g_last_local[1];
  __shared__ __align__(16) float g_rev_exp_shared[32];
  float u_fragment[16];
  bfloat16_t v_shared_local_cast[2];
  bfloat16_t v_shared_local_cast_1[2];
  float v_fragment[16];
  float p_fragment[8];
  float g_fragment[8];
  float a_fragment[8];
  bfloat16_t a_shared_local_cast_2[2];
  bfloat16_t a_shared_local_cast_3[2];
  float o_fragment[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(o_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    bar_o[0].init(128);
    bar_0[0].init(416);
    bar_1[0].init(256);
    _bar_2[0].init(128);
    bar_3[0].init(128);
    bar_4[0].init(128);
    bar_5[0].init(416);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = 0;
  seq_start_idx = ((int)cu_seqlens[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)]);
  seq_end_idx = ((int)cu_seqlens[((((int64_t)((int)blockIdx.x)) >> (int64_t)6) + (int64_t)1)]);
  chunk_start_idx = ((int)chunk_offsets[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)]);
  raw_batch_idx = ((int)cp_seq_map[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)]);
  int64_t condval;
  if (((-1 <= raw_batch_idx) && (raw_batch_idx < raw_batch_size))) {
    condval = raw_cu_seqlens[(((int64_t)raw_batch_idx) + (int64_t)1)];
  } else {
    condval = (int64_t)0;
  }
  raw_seq_end_idx = ((int)condval);
  need_store_final_state = ((signed char)((bool)1 & (raw_seq_end_idx == seq_end_idx)));
  num_iters = (((seq_end_idx + 31) - seq_start_idx) >> 5);
  num_unmasked_iters = ((seq_end_idx - seq_start_idx) >> 5);
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<160>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(h0 + ((((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    }
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      data_is_ready[(i_s & 1)].wait(((i_s & 3) >> 1));
      bar_0[0].arrive();
      bar_0[0].wait((i_s & 1));
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_1 >> 1) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)h_fragment[(i_1 * 8)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 7)])));
      }
      bar_1[0].arrive();
      bar_1[0].wait((i_s & 1));
      g_last_local[0] = g_exp_shared[31];
      #pragma unroll
      for (int i_2 = 0; i_2 < 64; ++i_2) {
        h_fragment[i_2] = (h_fragment[i_2] * g_last_local[0]);
      }
      bar_5[0].arrive();
      bar_5[0].wait((i_s & 1));
      {
        bfloat16_t A_local[32];
        bfloat16_t B_local[16];
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((i_s & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A_local[(i_3 * 8)])));
          }
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 34816)])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + ((i_5 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + (((i_5 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      data_is_free[(i_s & 1)].arrive();
    }
    if ((bool)need_store_final_state) {
      if (0 <= raw_batch_idx) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          if (raw_batch_idx < raw_batch_size) {
            *(float2*)(ht + ((((((((((((int64_t)raw_batch_idx) * (int64_t)524288) + (((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_6) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_6) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_fragment + (i_6 * 2));
          }
        }
      }
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<128>();
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial<3, 128>();
        if (((int)threadIdx.x) < 160) {
          g_exp_shared[(((int)threadIdx.x) - 128)] = exp2f((g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          float condval_1;
          if (((((i_s_1 * 32) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 128))) {
            condval_1 = exp2f(((g_shared[(((i_s_1 & 1) * 32) + 31)] - g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          } else {
            condval_1 = 0x0p+0f/*0.000000e+00*/;
          }
          g_rev_exp_shared[(((int)threadIdx.x) - 128)] = condval_1;
        }
        bar_1[0].arrive();
        bar_1[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_1[8];
          bfloat16_t B_local_1[16];
          #pragma unroll
          for (int i_7 = 0; i_7 < 4; ++i_7) {
            float broadcast_var = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(u_fragment + (i_7 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          }
          for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 4096) + ((ki_1 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(A_local_1[0])));
            for (int i_8 = 0; i_8 < 2; ++i_8) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((ki_1 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_8) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_8 * 8)])));
            }
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial<3, 128>();
        #pragma unroll
        for (int i_9 = 0; i_9 < 8; ++i_9) {
          float2 __1;
            float2 v_ = *(float2*)(u_fragment + (i_9 * 2));
            float2 v__1 = make_float2((g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
            *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(u_fragment + (i_9 * 2)) = __1;
        }
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          *(uint1*)(v_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_10 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576));
          float2 __2;
            float2 v__2 = *(float2*)(u_fragment + (i_10 * 2));
            float2 __3;
            uint1 v__3 = *(uint1*)(v_shared_local_cast + 0);
            ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
            *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(v__2.x)), *(float2*)(&(__3.x)));
          *(float2*)(u_fragment + (i_10 * 2)) = __2;
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_11 = 0; i_11 < 8; ++i_11) {
          uint1 __4;
          float2 v__4 = *(float2*)(u_fragment + (i_11 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
          *(uint1*)(v_shared_local_cast_1 + 0) = __4;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_11 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_11 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576)) = *(uint1*)(v_shared_local_cast_1 + 0);
        }
        bar_3[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_2[8];
          bfloat16_t B_local_2[16];
          #pragma unroll
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(v_fragment + (i_12 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
          }
          tl::__sync_thread_partial<4, 128>();
          for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 28672)])), (&(A_local_2[0])));
            for (int i_13 = 0; i_13 < 2; ++i_13) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_s_1 & 1) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])), (&(B_local_2[(i_13 * 8)])));
            }
            for (int j_2 = 0; j_2 < 2; ++j_2) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 32768)])), __pack_half2(((bfloat16_t)v_fragment[(i_14 * 8)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 7)])));
        }
        bar_4[0].arrive();
        #pragma unroll
        for (int i_15 = 0; i_15 < 8; ++i_15) {
          float2 __5;
            float2 v__5 = *(float2*)(v_fragment + (i_15 * 2));
            float2 v__6 = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
          *(float2*)(v_fragment + (i_15 * 2)) = __5;
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 34816)])), __pack_half2(((bfloat16_t)v_fragment[(i_16 * 8)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 7)])));
        }
        bar_5[0].arrive();
        bar_5[0].wait((i_s_1 & 1));
        data_is_free[(i_s_1 & 1)].arrive();
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<128>();
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 & 1)].wait(((i_s_2 & 3) >> 1));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_3[8];
            bfloat16_t B_local_3[8];
            #pragma unroll
            for (int i_17 = 0; i_17 < 2; ++i_17) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(p_fragment + (i_17 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
            }
            for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_3[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_3[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 4), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 4));
            }
          }
          #pragma unroll
          for (int i_18 = 0; i_18 < 4; ++i_18) {
            float2 __6;
              float2 v__7 = make_float2(g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              float2 v__8 = *(float2*)(g_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
            *(float2*)(g_fragment + (i_18 * 2)) = __6;
          }
          #pragma unroll
          for (int i_19 = 0; i_19 < 8; ++i_19) {
            if ((((((((int)threadIdx.x) >> 6) * 16) + ((i_19 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_19 & 1)) <= ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_19 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64)) {
              g_fragment[i_19] = exp2f((g_fragment[i_19] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
            } else {
              g_fragment[i_19] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_20 = 0; i_20 < 4; ++i_20) {
            *(uint1*)(a_shared_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_20 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_20 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 28672));
            float2 __7;
            uint1 v__9 = *(uint1*)(a_shared_local_cast_2 + 0);
            ((float2*)(&__7))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__9))[0]);
            *(float2*)(a_fragment + (i_20 * 2)) = __7;
          }
          #pragma unroll
          for (int i_21 = 0; i_21 < 8; ++i_21) {
            a_fragment[i_21] = (a_fragment[i_21] * g_fragment[i_21]);
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 4; ++i_22) {
            float2 __8;
              float2 v__10 = *(float2*)(a_fragment + (i_22 * 2));
              float2 v__11 = *(float2*)(b_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_22 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)));
            *(float2*)(a_fragment + (i_22 * 2)) = __8;
          }
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_23 = 0; i_23 < 4; ++i_23) {
            uint1 __9;
            float2 v__12 = *(float2*)(a_fragment + (i_23 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__12))[0]);
            *(uint1*)(a_shared_local_cast_3 + 0) = __9;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_23 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_23 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 28672)) = *(uint1*)(a_shared_local_cast_3 + 0);
          }
          bar_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_4[8];
            bfloat16_t B_local_4[16];
            #pragma unroll
            for (int i_24 = 0; i_24 < 4; ++i_24) {
              float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(o_fragment + (i_24 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            }
            tl::__sync_thread_partial<5, 128>();
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_4 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_4[0])));
              for (int i_25 = 0; i_25 < 2; ++i_25) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((ki_4 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_25) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_25 * 8)])));
              }
              for (int j_3 = 0; j_3 < 2; ++j_3) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_3 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_3 * 8) + 4)));
              }
            }
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 8; ++i_26) {
            p_fragment[i_26] = (p_fragment[i_26] * (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_fragment[i_26]));
          }
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((((((((int)threadIdx.x) >> 7) * 32) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 8)) - 64) >> 8) * 256)) + (((((((((int)threadIdx.x) >> 7) * 32) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 8)) + 192) & 255)) + 36864)])), __pack_half2(((bfloat16_t)p_fragment[0]), ((bfloat16_t)p_fragment[1])), __pack_half2(((bfloat16_t)p_fragment[2]), ((bfloat16_t)p_fragment[3])), __pack_half2(((bfloat16_t)p_fragment[4]), ((bfloat16_t)p_fragment[5])), __pack_half2(((bfloat16_t)p_fragment[6]), ((bfloat16_t)p_fragment[7])));
          bar_3[0].arrive();
          #pragma unroll
          for (int i_27 = 0; i_27 < 8; ++i_27) {
            float2 __10;
              float2 v__13 = *(float2*)(o_fragment + (i_27 * 2));
              float2 v__14 = make_float2((0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]), (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
              *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
            *(float2*)(o_fragment + (i_27 * 2)) = __10;
          }
          bar_4[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_5[8];
            bfloat16_t B_local_5[16];
            tl::__sync_thread_partial<5, 128>();
            for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_5) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 36864)])), (&(A_local_5[0])));
              for (int i_28 = 0; i_28 < 2; ++i_28) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki_5 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_28) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_5[(i_28 * 8)])));
              }
              for (int j_4 = 0; j_4 < 2; ++j_4) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_4 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_4 * 8) + 4)));
              }
            }
          }
          bar_5[0].arrive();
          bar_5[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_29 = 0; i_29 < 2; ++i_29) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_29 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) - 128) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_29 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) + 384) & 511)) + 30720)])), __pack_half2(((bfloat16_t)o_fragment[(i_29 * 8)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 1)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 2)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 3)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 4)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 5)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 6)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 7)])));
          }
          data_is_free[(i_s_2 & 1)].arrive();
        }
        bar_o[0].arrive();
      } else {
        tl::warpgroup_reg_dealloc<32>();
        if (((int)threadIdx.x) < 416) {
          tl::__sync_thread_partial<6, 32>();
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 & 1)].wait((((i_s_3 >> 1) + 1) & 1));
            int left = ((i_s_3 * 32) + seq_start_idx);
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 16384)])), 0, ((((int)blockIdx.x) & 63) >> 2), left, batch_idx);
                tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 18432)])), 64, ((((int)blockIdx.x) & 63) >> 2), left, batch_idx);
              }
            } else {
              tl::__sync_thread_partial<6, 32>();
              #pragma unroll
              for (int i_30 = 0; i_30 < 16; ++i_30) {
                if ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_2;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_30)) && ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_2 = *(uint4*)(q + (((((((((int64_t)i_30) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)49152));
                  } else {
                    condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
                  }
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_30 >> 2) * 512)) + ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_30) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_30) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = condval_2;
                } else {
                  bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_30 >> 2) * 512)) + ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_30) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_30) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
                }
              }
            }
            tl::__sync_thread_partial<6, 32>();
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 8192)])), 0, ((((int)blockIdx.x) & 63) >> 2), left, batch_idx);
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 10240)])), 64, ((((int)blockIdx.x) & 63) >> 2), left, batch_idx);
              }
            } else {
              tl::__sync_thread_partial<6, 32>();
              #pragma unroll
              for (int i_31 = 0; i_31 < 16; ++i_31) {
                if ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_3;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_31)) && ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_3 = *(uint4*)(k + (((((((((int64_t)i_31) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)left) * (int64_t)2048)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)2048)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)49152));
                  } else {
                    condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
                  }
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_31 >> 2) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_31) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_31) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = condval_3;
                } else {
                  bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_31 >> 2) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_31) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_31) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
                }
              }
            }
            data_is_ready[(i_s_3 & 1)].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::__sync_thread_partial<7, 32>();
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              data_is_free[(i_s_4 & 1)].wait((((i_s_4 >> 1) + 1) & 1));
              int left_1 = ((i_s_4 * 32) + seq_start_idx);
              if ((left_1 + 32) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 & 1)].expect_transaction(4096);
                  tl::fence_proxy_async();
                  tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_4 & 1) * 2048) + 24576)])), ((((int)blockIdx.x) & 1) * 64), ((((int)blockIdx.x) & 63) >> 1), left_1, batch_idx);
                }
              } else {
                tl::__sync_thread_partial<7, 32>();
                #pragma unroll
                for (int i_32 = 0; i_32 < 8; ++i_32) {
                  if ((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (seq_end_idx + 52)) {
                    bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_4;
                    if (((((13 <= ((((((int)threadIdx.x) >> 3) + left_1) >> 2) + i_32)) && ((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (num_tokens + 52))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_4 = *(uint4*)(v + (((((((((int64_t)i_32) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)left_1) * (int64_t)4096)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)212992));
                    } else {
                      condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
                    }
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_s_4 & 1) * 2048) + ((i_32 >> 1) * 512)) + (((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_32) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = condval_4;
                  } else {
                    bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_s_4 & 1) * 2048) + ((i_32 >> 1) * 512)) + (((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_32) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
                  }
                }
              }
              if ((left_1 + 32) <= seq_end_idx) {
                float condval_5;
                if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                  condval_5 = b[(((((((int64_t)left_1) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)13312)];
                } else {
                  condval_5 = 0x0p+0f/*0.000000e+00*/;
                }
                b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_5;
              } else {
                if ((left_1 + ((int)threadIdx.x)) < (seq_end_idx + 416)) {
                  float condval_6;
                  if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_6 = b[(((((((int64_t)left_1) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)13312)];
                  } else {
                    condval_6 = 0x0p+0f/*0.000000e+00*/;
                  }
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_6;
                } else {
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = 0x0p+0f/*0.000000e+00*/;
                }
              }
              data_is_ready[(i_s_4 & 1)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              tl::__sync_thread_partial<8, 32>();
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 & 1)].wait((((i_s_5 >> 1) + 1) & 1));
                int left_2 = ((i_s_5 * 32) + seq_start_idx);
                if ((left_2 + 32) <= seq_end_idx) {
                  if (tl::tl_shuffle_elect<32>()) {
                    data_is_ready[(i_s_5 & 1)].expect_transaction(2048);
                    tl::fence_proxy_async();
                    tl::tma_load(a_desc, data_is_ready[(i_s_5 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_5 & 1) * 1024) + 28672)])), 0, ((((int)blockIdx.x) & 63) >> 1), left_2, batch_idx);
                  }
                } else {
                  #pragma unroll
                  for (int i_33 = 0; i_33 < 4; ++i_33) {
                    if ((((i_33 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (seq_end_idx + 112)) {
                      bfloat16_t broadcast_var_10 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      uint4 condval_7;
                      if (((((14 <= ((((((int)threadIdx.x) >> 2) + left_2) >> 3) + i_33)) && ((((i_33 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (num_tokens + 112))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_7 = *(uint4*)(a + (((((((((int64_t)i_33) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)1024)) + (((int64_t)left_2) * (int64_t)1024)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) - (int64_t)114688));
                      } else {
                        condval_7 = make_uint4(__pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10));
                      }
                      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_5 & 1) * 1024) + (i_33 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 25088)) = condval_7;
                    } else {
                      bfloat16_t broadcast_var_11 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_5 & 1) * 1024) + (i_33 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 25088)) = make_uint4(__pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11));
                    }
                  }
                }
                if ((left_2 + 32) <= seq_end_idx) {
                  float condval_8;
                  if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_8 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)14336)];
                  } else {
                    condval_8 = 0x0p+0f/*0.000000e+00*/;
                  }
                  g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_8;
                } else {
                  if ((left_2 + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                    float condval_9;
                    if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_9 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)14336)];
                    } else {
                      condval_9 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_9;
                  } else {
                    float condval_10;
                    if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_10 = g[((((((int64_t)seq_end_idx) * (int64_t)32) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)32)];
                    } else {
                      condval_10 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_10;
                  }
                }
                data_is_ready[(i_s_5 & 1)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_unmasked_iters; ++i_s_6) {
                int right = ((i_s_6 * 32) + seq_start_idx);
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                if (0 < i_s_6) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)buf_dyn_shmem)[30720])), ((((int)blockIdx.x) & 1) * 64), ((((int)blockIdx.x) & 63) >> 1), (right - 32), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((i_s_6 & 1));
              }
              if (num_unmasked_iters < num_iters) {
                seq_split_idx = ((num_unmasked_iters * 32) + seq_start_idx);
                chunk_split_idx = (chunk_start_idx + num_unmasked_iters);
                int right_1 = seq_split_idx;
                bar_0[0].arrive();
                bar_0[0].wait((num_unmasked_iters & 1));
                if (0 < num_unmasked_iters) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)buf_dyn_shmem)[30720])), ((((int)blockIdx.x) & 1) * 64), ((((int)blockIdx.x) & 63) >> 1), (right_1 - 32), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((num_unmasked_iters & 1));
              }
              seq_split_idx = (((num_iters * 32) + seq_start_idx) - 32);
              bar_o[0].wait(0);
              if (0 < num_iters) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_34 = 0; i_34 < 8; ++i_34) {
                    if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (seq_end_idx + 60)) {
                      if (15 <= ((((((int)threadIdx.x) >> 3) + seq_split_idx) >> 2) + i_34)) {
                        if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60)) {
                          if (batch_idx < 1) {
                            *(uint4*)(o + (((((((((int64_t)i_34) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)seq_split_idx) * (int64_t)4096)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)245760)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_34 >> 1) * 512) + (((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_34) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 30720));
                          }
                        }
                      }
                    } else {
                      if (((((int)blockIdx.x) >> 6) == (batch_size - 1)) && ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60))) {
                        if (15 <= ((((((int)threadIdx.x) >> 3) + seq_split_idx) >> 2) + i_34)) {
                          if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60)) {
                            if (batch_idx < 1) {
                              bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                              *(uint4*)(o + (((((((((int64_t)i_34) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)seq_split_idx) * (int64_t)4096)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)245760)) = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


// ---- flashqla_kkt_packed_strided ----
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void flashqla_kkt_packed_strided(bfloat16_t* __restrict__ a, const float* __restrict__ b, const int64_t* __restrict__ chunk_indices, const int64_t* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ k, int data_batch_size, int num_chunks, int num_tokens, int real_batch_size);
extern "C" __global__ void __launch_bounds__(256, 1) flashqla_kkt_packed_strided(bfloat16_t* __restrict__ a, const float* __restrict__ b, const int64_t* __restrict__ chunk_indices, const int64_t* __restrict__ cu_seqlens, const bfloat16_t* __restrict__ k, int data_batch_size, int num_chunks, int num_tokens, int real_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t k_is_ready_mem[1];
  auto k_is_ready = reinterpret_cast<Barrier*>(k_is_ready_mem);
  __shared__ __align__(16) uint64_t a_is_ready_mem[1];
  auto a_is_ready = reinterpret_cast<Barrier*>(a_is_ready_mem);
  int batch_idx = 0;
  int chunk_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  __shared__ __align__(16) float b_shared[32];
  float a32_fragment[8];
  float a16o_fragment[2];
  float a16i_row[16];
  float a16i_sum[1];
  float a32_shared_local_cast_1[4];
  bfloat16_t a_local_cast[4];
  float a32_shared_local_cast_3[4];
  bfloat16_t a_local_cast_2[4];
  if (tl::tl_shuffle_elect<0>()) {
    k_is_ready[0].init(32);
    a_is_ready[0].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = ((int)chunk_indices[(((int64_t)((int)blockIdx.y)) * (int64_t)2)]);
  chunk_idx = ((int)chunk_indices[((((int64_t)((int)blockIdx.y)) * (int64_t)2) + (int64_t)1)]);
  int64_t condval;
  if (((0 <= batch_idx) && (batch_idx <= real_batch_size))) {
    condval = cu_seqlens[batch_idx];
  } else {
    condval = (int64_t)0;
  }
  seq_start_idx = ((int)condval);
  int64_t condval_1;
  if (((-1 <= batch_idx) && (batch_idx < real_batch_size))) {
    condval_1 = cu_seqlens[(((int64_t)batch_idx) + (int64_t)1)];
  } else {
    condval_1 = (int64_t)0;
  }
  seq_end_idx = ((int)condval_1);
  int chunk_idx_1 = chunk_idx;
  int seq_start_idx_1 = seq_start_idx;
  int seq_end_idx_1 = seq_end_idx;
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<64>();
    if ((((chunk_idx_1 * 32) + seq_start_idx_1) + 32) <= seq_end_idx_1) {
      if (((int)threadIdx.x) < 32) {
        float condval_2;
        if (((0 <= (((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x))) && ((((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens))) {
          condval_2 = b[((((((int64_t)chunk_idx_1) * (int64_t)1024) + (((int64_t)seq_start_idx_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((int64_t)((int)blockIdx.x)))];
        } else {
          condval_2 = 0x0p+0f/*0.000000e+00*/;
        }
        b_shared[((int)threadIdx.x)] = condval_2;
      }
    } else {
      if (((int)threadIdx.x) < 32) {
        if ((((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x)) < seq_end_idx_1) {
          float condval_3;
          if (((0 <= (((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x))) && ((((chunk_idx_1 * 32) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens))) {
            condval_3 = b[((((((int64_t)chunk_idx_1) * (int64_t)1024) + (((int64_t)seq_start_idx_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((int64_t)((int)blockIdx.x)))];
          } else {
            condval_3 = 0x0p+0f/*0.000000e+00*/;
          }
          b_shared[((int)threadIdx.x)] = condval_3;
        } else {
          b_shared[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
        }
      }
    }
    k_is_ready[0].wait(0);
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[8];
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        float broadcast_var = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(a32_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      }
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki >> 2) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a32_fragment + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(a32_fragment + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      float2 __1;
        float2 v_ = *(float2*)(a32_fragment + (i_1 * 2));
        float2 v__1 = make_float2(b_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_1 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], b_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_1 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      *(float2*)(a32_fragment + (i_1 * 2)) = __1;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      if ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_2 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < (((((((int)threadIdx.x) >> 6) * 16) + ((i_2 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_2 & 1))) {
        a32_fragment[i_2] = 0x0p+0f/*0.000000e+00*/;
      } else {
        if ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_2 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == (((((((int)threadIdx.x) >> 6) * 16) + ((i_2 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_2 & 1))) {
          a32_fragment[i_2] = 0x1p+0f/*1.000000e+00*/;
        }
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      if (((((int)threadIdx.x) & 63) >> 5) == ((((int)threadIdx.x) >> 6) + 1)) {
        float broadcast_var_1 = -0x1p+0f/*-1.000000e+00*/;
        float2 __2;
          float2 v__2 = *(float2*)(a32_fragment + (i_3 * 2));
          float2 v__3 = make_float2(broadcast_var_1, broadcast_var_1);
          *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
        *(float2*)(((float*)buf_dyn_shmem) + ((((((i_3 & 1) * 128) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 3616)) = __2;
      } else {
        if (((((int)threadIdx.x) & 63) >> 5) == (((int)threadIdx.x) >> 6)) {
          *(float2*)(((float*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 63) >> 5) * 272) + ((i_3 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 3072)) = *(float2*)(a32_fragment + (i_3 * 2));
        }
      }
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int k_s = 1; k_s < 16; ++k_s) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        for (int vec_s = 0; vec_s < 4; ++vec_s) {
          if (((i_4 * 4) + vec_s) < k_s) {
            a16i_row[((i_4 * 4) + vec_s)] = ((float*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_s * 16)) + (i_4 * 4)) + vec_s) + 3072)];
          }
        }
      }
      a16i_sum[0] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int k_r = 0; k_r < k_s; ++k_r) {
        a16i_sum[0] = (a16i_sum[0] - (((float*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_r * 16)) + (((int)threadIdx.x) & 15)) + 3072)] * a16i_row[k_r]));
      }
      tl::__sync_thread_partial<3, 128>();
      if ((((int)threadIdx.x) >> 5) == 0) {
        if ((((int)threadIdx.x) & 15) < k_s) {
          ((float*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 31) >> 4) * 272) + (k_s * 16)) + (((int)threadIdx.x) & 15)) + 3072)] = a16i_sum[0];
        }
      }
    }
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(a16o_fragment + 0) = make_float2(broadcast_var_2, broadcast_var_2);
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int k_r_1 = 0; k_r_1 < 16; ++k_r_1) {
      float2 __3;
        float2 v__4 = make_float2(((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 16) + k_r_1) + 3344)], ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 16) + k_r_1) + 3344)]);
        float2 v__5 = *(float2*)(((float*)buf_dyn_shmem) + (((k_r_1 * 16) + ((((int)threadIdx.x) & 7) * 2)) + 3616));
        float2 v__6 = *(float2*)(a16o_fragment + 0);
        *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
      *(float2*)(a16o_fragment + 0) = __3;
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 7) * 32) + (i_5 * 16)) + (((int)threadIdx.x) >> 3)) + 3616)] = a16o_fragment[i_5];
    }
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(a16o_fragment + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int k_r_2 = 0; k_r_2 < 16; ++k_r_2) {
      float2 __4;
        float2 v__7 = make_float2(((float*)buf_dyn_shmem)[(((k_r_2 * 16) + (((int)threadIdx.x) >> 3)) + 3616)], ((float*)buf_dyn_shmem)[(((k_r_2 * 16) + (((int)threadIdx.x) >> 3)) + 3616)]);
        float2 v__8 = *(float2*)(((float*)buf_dyn_shmem) + (((k_r_2 * 16) + ((((int)threadIdx.x) & 7) * 2)) + 3072));
        float2 v__9 = *(float2*)(a16o_fragment + 0);
        *(float2*)(&(__4.x)) = tl::fma2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)), *(float2*)(&(v__9.x)));
      *(float2*)(a16o_fragment + 0) = __4;
    }
    tl::__sync_thread_partial<3, 128>();
    *(float2*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 3616)) = *(float2*)(a16o_fragment + 0);
    *(float2*)(((float*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 2048)) = *(float2*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 3072));
    float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(((float*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 2064)) = make_float2(broadcast_var_4, broadcast_var_4);
    *(float2*)(((float*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 2560)) = *(float2*)(a16o_fragment + 0);
    *(float2*)(((float*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 3) * 32) + ((((int)threadIdx.x) & 7) * 2)) + 2576)) = *(float2*)(((float*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 3344));
    a_is_ready[0].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    if (((int)threadIdx.x) < 160) {
      tl::__sync_thread_partial<4, 32>();
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_4;
        if (((((((chunk_idx_1 * 32) + (i_6 * 2)) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) < (num_tokens + 8)) && (1 <= ((chunk_idx_1 * 4) + ((((i_6 * 2) + (((int)threadIdx.x) >> 4)) + seq_start_idx_1) >> 3))))) {
          condval_4 = *(uint4*)(k + (((((((((int64_t)chunk_idx_1) * (int64_t)262144) + (((int64_t)i_6) * (int64_t)16384)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)seq_start_idx_1) * (int64_t)8192)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)65536));
        } else {
          condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((i_6 >> 2) * 512)) + ((((i_6 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_6) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_6) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_4;
      }
      k_is_ready[0].arrive();
    } else {
      if (((int)threadIdx.x) < 192) {
        a_is_ready[0].wait(0);
        if ((((chunk_idx_1 * 32) + seq_start_idx_1) + 32) <= seq_end_idx_1) {
          #pragma unroll
          for (int i_7 = 0; i_7 < 8; ++i_7) {
            if ((((((chunk_idx_1 * 32) + (i_7 * 4)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < (num_tokens + 20)) && (20 <= ((((chunk_idx_1 * 32) + (i_7 * 4)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1))) {
              *(float4*)(a32_shared_local_cast_1 + 0) = *(float4*)(((float*)buf_dyn_shmem) + (((i_7 * 128) + (((int)threadIdx.x) * 4)) + 1408));
              uint2 __5;
              float4 v__10 = *(float4*)(a32_shared_local_cast_1 + 0);
              (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__10))[0]);
              (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__10))[1]);
              *(uint2*)(a_local_cast + 0) = __5;
              *(uint2*)(a + (((((((((int64_t)chunk_idx_1) * (int64_t)32768) + (((int64_t)i_7) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)1024)) + (((int64_t)seq_start_idx_1) * (int64_t)1024)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4)) - (int64_t)20480)) = *(uint2*)(a_local_cast + 0);
            }
          }
        }
      } else {
        a_is_ready[0].wait(0);
        if (seq_end_idx_1 < (((chunk_idx_1 * 32) + seq_start_idx_1) + 32)) {
          #pragma unroll
          for (int i_8 = 0; i_8 < 4; ++i_8) {
            if (((((chunk_idx_1 * 32) + (i_8 * 8)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < (seq_end_idx_1 + 24)) {
              *(float4*)(a32_shared_local_cast_3 + 0) = *(float4*)(((float*)buf_dyn_shmem) + (((i_8 * 256) + (((int)threadIdx.x) * 4)) + 1280));
              uint2 __6;
              float4 v__11 = *(float4*)(a32_shared_local_cast_3 + 0);
              (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
              (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&v__11))[1]);
              *(uint2*)(a_local_cast_2 + 0) = __6;
              if (24 <= ((((chunk_idx_1 * 32) + (i_8 * 8)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1)) {
                if (((((chunk_idx_1 * 32) + (i_8 * 8)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < (num_tokens + 24)) {
                  *(uint2*)(a + (((((((((int64_t)chunk_idx_1) * (int64_t)32768) + (((int64_t)i_8) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)1024)) + (((int64_t)seq_start_idx_1) * (int64_t)1024)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4)) - (int64_t)24576)) = *(uint2*)(a_local_cast_2 + 0);
                }
              }
            }
          }
        }
      }
    }
  }
}


// ---- flashqla_prepare_h_packed_strided ----
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void flashqla_prepare_h_packed_strided(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, bfloat16_t* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ mt, const int64_t* __restrict__ num_warmup_chunks, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens);
extern "C" __global__ void __launch_bounds__(512, 1) flashqla_prepare_h_packed_strided(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, bfloat16_t* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ mt, const int64_t* __restrict__ num_warmup_chunks, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t data_is_ready_mem[2];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[2];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t bar_2_mem[1];
  auto bar_2 = reinterpret_cast<Barrier*>(bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int num_iters = 0;
  signed char calc_mt = (signed char)0;
  float h_fragment[128];
  __shared__ __align__(16) float g_shared[64];
  float m_fragment_R[64];
  float g_prod_X[1];
  __shared__ __align__(16) float b_shared[64];
  float g_last_local_X[1];
  float m_fragment_L[64];
  float g_prod_Y[1];
  float g_last_local_Y[1];
  float g_last_local_S[1];
  bfloat16_t ht_local_cast[2];
  float x_fragment[32];
  float z_fragment_R[16];
  bfloat16_t h_shared_local_cast_1[2];
  bfloat16_t mt_local_cast_2[2];
  bfloat16_t mt_local_cast_3[2];
  __shared__ __align__(16) float g_rev_exp_shared[32];
  float y_fragment[32];
  bfloat16_t v_shared_local_cast_4[2];
  float g_rev_exp_shared_local_cast_5[2];
  float z_fragment_L[16];
  bfloat16_t h_shared_local_cast_6[2];
  bfloat16_t mt_local_cast_7[2];
  bfloat16_t mt_local_cast_8[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(h_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    bar_0[0].init(416);
    bar_1[0].init(256);
    bar_2[0].init(416);
    bar_3[0].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = 0;
  seq_start_idx = ((int)cu_seqlens[((int64_t)((int)blockIdx.y))]);
  seq_end_idx = ((int)cu_seqlens[(((int64_t)((int)blockIdx.y)) + (int64_t)1)]);
  chunk_start_idx = ((int)chunk_offsets[((int64_t)((int)blockIdx.y))]);
  num_iters = ((int)num_warmup_chunks[((((int64_t)((int)blockIdx.y)) * (int64_t)32) + ((int64_t)((int)blockIdx.x)))]);
  calc_mt = ((signed char)((((seq_end_idx + 31) - seq_start_idx) >> 5) <= num_iters));
  if (seq_start_idx < (seq_end_idx - (num_iters * 32))) {
    seq_start_idx = (seq_end_idx - (num_iters * 32));
  }
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<168>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(h0 + (((((((((((int64_t)((int)blockIdx.y)) * (int64_t)524288) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i) >> (int64_t)4) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((((int64_t)i) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    }
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      data_is_ready[(i_s & 1)].wait(((i_s & 3) >> 1));
      bar_0[0].arrive();
      bar_0[0].wait((i_s & 1));
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 4096) + ((i_1 >> 2) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)h_fragment[(i_1 * 8)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 7)])));
      }
      bar_1[0].arrive();
      bar_1[0].wait((i_s & 1));
      g_last_local_S[0] = exp2f((g_shared[(((i_s & 1) * 32) + 31)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
      #pragma unroll
      for (int i_2 = 0; i_2 < 128; ++i_2) {
        h_fragment[i_2] = (h_fragment[i_2] * g_last_local_S[0]);
      }
      bar_2[0].arrive();
      bar_2[0].wait((i_s & 1));
      {
        bfloat16_t A_local[32];
        bfloat16_t B_local[32];
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (ki * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])), (&(A_local[(i_3 * 8)])));
          }
          for (int i_4 = 0; i_4 < 4; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 36864)])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            for (int j = 0; j < 4; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + ((i_5 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + (((i_5 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      bar_3[0].arrive();
      data_is_free[(i_s & 1)].arrive();
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 64; ++i_6) {
      uint1 __1;
      float2 v_ = *(float2*)(h_fragment + (i_6 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      *(uint1*)(ht_local_cast + 0) = __1;
      *(uint1*)(ht + (((((((((((int64_t)((int)blockIdx.y)) * (int64_t)524288) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_6) >> (int64_t)4) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((((int64_t)i_6) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(ht_local_cast + 0);
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<160>();
      if ((bool)calc_mt) {
        #pragma unroll
        for (int i_7 = 0; i_7 < 64; ++i_7) {
          if (((((((((int)threadIdx.x) & 63) >> 5) * 64) + ((i_7 >> 4) * 16)) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == (((((((int)threadIdx.x) >> 6) * 32) + (((i_7 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))) {
            m_fragment_R[i_7] = 0x1p+0f/*1.000000e+00*/;
          } else {
            m_fragment_R[i_7] = 0x0p+0f/*0.000000e+00*/;
          }
        }
        g_prod_X[0] = 0x0p+0f/*0.000000e+00*/;
      }
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_1[8];
          bfloat16_t B_local_1[32];
          #pragma unroll
          for (int i_8 = 0; i_8 < 8; ++i_8) {
            float broadcast_var = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(x_fragment + (i_8 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          }
          for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((i_s_1 & 1) * 1024) + (ki_1 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 40960)])), (&(A_local_1[0])));
            for (int i_9 = 0; i_9 < 4; ++i_9) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 4096) + (((((int)threadIdx.x) & 127) >> 6) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(B_local_1[(i_9 * 8)])));
            }
            for (int j_1 = 0; j_1 < 4; ++j_1) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(x_fragment + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(x_fragment + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
        #pragma unroll
        for (int i_10 = 0; i_10 < 16; ++i_10) {
          float2 __2;
            float2 v__1 = *(float2*)(x_fragment + (i_10 * 2));
            float2 v__2 = make_float2((b_shared[(((((i_s_1 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (b_shared[(((((i_s_1 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
            *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__1.x)), *(float2*)(&(v__2.x)));
          *(float2*)(x_fragment + (i_10 * 2)) = __2;
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_11 = 0; i_11 < 4; ++i_11) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 32768)])), __pack_half2(((bfloat16_t)x_fragment[(i_11 * 8)]), ((bfloat16_t)x_fragment[((i_11 * 8) + 1)])), __pack_half2(((bfloat16_t)x_fragment[((i_11 * 8) + 2)]), ((bfloat16_t)x_fragment[((i_11 * 8) + 3)])), __pack_half2(((bfloat16_t)x_fragment[((i_11 * 8) + 4)]), ((bfloat16_t)x_fragment[((i_11 * 8) + 5)])), __pack_half2(((bfloat16_t)x_fragment[((i_11 * 8) + 6)]), ((bfloat16_t)x_fragment[((i_11 * 8) + 7)])));
        }
        bar_2[0].arrive();
        if ((bool)calc_mt) {
          g_prod_X[0] = (g_prod_X[0] + g_shared[(((i_s_1 & 1) * 32) + 31)]);
          bar_2[0].wait((i_s_1 & 1));
          tl::__sync_thread_partial<4, 128>();
          #pragma unroll
          for (int i_12 = 0; i_12 < 32; ++i_12) {
            uint1 __3;
            float2 v__3 = *(float2*)(m_fragment_R + (i_12 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
            *(uint1*)(h_shared_local_cast_1 + 0) = __3;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((((((int)threadIdx.x) >> 6) * 32) + (((i_12 & 7) >> 1) * 8)) >> 6) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_12 >> 3) * 1024)) + ((i_12 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_12 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_12 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(h_shared_local_cast_1 + 0);
          }
          bar_3[0].wait((i_s_1 & 1));
          {
            bfloat16_t A_local_2[8];
            bfloat16_t B_local_2[16];
            #pragma unroll
            for (int i_13 = 0; i_13 < 4; ++i_13) {
              float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(z_fragment_R + (i_13 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
            }
            tl::__sync_thread_partial<4, 128>();
            for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 4096) + ((ki_2 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_2[0])));
              for (int i_14 = 0; i_14 < 2; ++i_14) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki_2 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(B_local_2[(i_14 * 8)])));
              }
              for (int j_2 = 0; j_2 < 2; ++j_2) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(z_fragment_R + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(z_fragment_R + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
              }
            }
          }
          tl::__sync_thread_partial<4, 128>();
          #pragma unroll
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_15) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_15) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 45056)])), __pack_half2(((bfloat16_t)z_fragment_R[(i_15 * 8)]), ((bfloat16_t)z_fragment_R[((i_15 * 8) + 1)])), __pack_half2(((bfloat16_t)z_fragment_R[((i_15 * 8) + 2)]), ((bfloat16_t)z_fragment_R[((i_15 * 8) + 3)])), __pack_half2(((bfloat16_t)z_fragment_R[((i_15 * 8) + 4)]), ((bfloat16_t)z_fragment_R[((i_15 * 8) + 5)])), __pack_half2(((bfloat16_t)z_fragment_R[((i_15 * 8) + 6)]), ((bfloat16_t)z_fragment_R[((i_15 * 8) + 7)])));
          }
          {
            bfloat16_t A_local_3[32];
            bfloat16_t B_local_3[16];
            tl::__sync_thread_partial<4, 128>();
            for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
              for (int i_16 = 0; i_16 < 4; ++i_16) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])), (&(A_local_3[(i_16 * 8)])));
              }
              for (int i_17 = 0; i_17 < 2; ++i_17) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki_3 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_17) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 45056)])), (&(B_local_3[(i_17 * 8)])));
              }
              for (int i_18 = 0; i_18 < 4; ++i_18) {
                for (int j_3 = 0; j_3 < 2; ++j_3) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(m_fragment_R + ((i_18 * 16) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(m_fragment_R + (((i_18 * 16) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
                }
              }
            }
          }
        }
        data_is_free[(i_s_1 & 1)].arrive();
      }
      if ((bool)calc_mt) {
        g_last_local_X[0] = exp2f((g_prod_X[0] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
        #pragma unroll
        for (int i_19 = 0; i_19 < 64; ++i_19) {
          m_fragment_R[i_19] = (m_fragment_R[i_19] * g_last_local_X[0]);
        }
        #pragma unroll
        for (int i_20 = 0; i_20 < 32; ++i_20) {
          uint1 __4;
          float2 v__4 = *(float2*)(m_fragment_R + (i_20 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
          *(uint1*)(mt_local_cast_2 + 0) = __4;
          *(uint1*)(mt + (((((((((((int64_t)((int)blockIdx.y)) * (int64_t)524288) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_20) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_20) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_20) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(mt_local_cast_2 + 0);
        }
      } else {
        #pragma unroll
        for (int i_21 = 0; i_21 < 64; ++i_21) {
          m_fragment_R[i_21] = 0x0p+0f/*0.000000e+00*/;
        }
        #pragma unroll
        for (int i_22 = 0; i_22 < 32; ++i_22) {
          uint1 __5;
          float2 v__5 = *(float2*)(m_fragment_R + (i_22 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__5))[0]);
          *(uint1*)(mt_local_cast_3 + 0) = __5;
          *(uint1*)(mt + (((((((((((int64_t)((int)blockIdx.y)) * (int64_t)524288) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_22) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_22) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_22) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(mt_local_cast_3 + 0);
        }
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<160>();
        if ((bool)calc_mt) {
          #pragma unroll
          for (int i_23 = 0; i_23 < 64; ++i_23) {
            if (((((((((int)threadIdx.x) & 63) >> 5) * 64) + ((i_23 >> 4) * 16)) + (((i_23 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == ((((((((int)threadIdx.x) >> 6) * 32) + (((i_23 & 15) >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_23 & 1)) - 128)) {
              m_fragment_L[i_23] = 0x1p+0f/*1.000000e+00*/;
            } else {
              m_fragment_L[i_23] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          g_prod_Y[0] = 0x0p+0f/*0.000000e+00*/;
        }
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 & 1)].wait(((i_s_2 & 3) >> 1));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          g_last_local_Y[0] = g_shared[(((i_s_2 & 1) * 32) + 31)];
          tl::__sync_thread_partial<3, 128>();
          if (((int)threadIdx.x) < 288) {
            g_rev_exp_shared[(((int)threadIdx.x) - 256)] = exp2f(((g_last_local_Y[0] - g_shared[((((i_s_2 & 1) * 32) + ((int)threadIdx.x)) - 256)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          }
          g_last_local_Y[0] = exp2f((g_last_local_Y[0] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          bar_1[0].arrive();
          bar_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_4[8];
            bfloat16_t B_local_4[32];
            #pragma unroll
            for (int i_24 = 0; i_24 < 8; ++i_24) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(y_fragment + (i_24 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
            }
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_4 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_4[0])));
              for (int i_25 = 0; i_25 < 4; ++i_25) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 127) >> 6) * 8192) + (ki_4 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_25 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_25 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_25 * 8)])));
              }
              for (int j_4 = 0; j_4 < 4; ++j_4) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(y_fragment + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(y_fragment + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
              }
            }
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 32; ++i_26) {
            y_fragment[i_26] = (y_fragment[i_26] * g_last_local_Y[0]);
          }
          tl::__sync_thread_partial<3, 128>();
          #pragma unroll
          for (int i_27 = 0; i_27 < 16; ++i_27) {
            *(uint1*)(v_shared_local_cast_4 + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_2 & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + ((i_27 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + ((i_27 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24320));
            *(float2*)(g_rev_exp_shared_local_cast_5 + 0) = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            float2 __6;
              float2 v__6 = *(float2*)(y_fragment + (i_27 * 2));
              float2 __7;
                float2 __8;
                uint1 v__7 = *(uint1*)(v_shared_local_cast_4 + 0);
                ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__7))[0]);
                float2 v__8 = *(float2*)(g_rev_exp_shared_local_cast_5 + 0);
                *(float2*)(&(__7.x)) = tl::mul2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
              *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(v__6.x)), *(float2*)(&(__7.x)));
            *(float2*)(y_fragment + (i_27 * 2)) = __6;
          }
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_28 = 0; i_28 < 4; ++i_28) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2) + (i_28 >> 1)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1) + (i_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) - 128) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2) + (i_28 >> 1)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1) + (i_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) + 384) & 511)) + 36864)])), __pack_half2(((bfloat16_t)y_fragment[(i_28 * 8)]), ((bfloat16_t)y_fragment[((i_28 * 8) + 1)])), __pack_half2(((bfloat16_t)y_fragment[((i_28 * 8) + 2)]), ((bfloat16_t)y_fragment[((i_28 * 8) + 3)])), __pack_half2(((bfloat16_t)y_fragment[((i_28 * 8) + 4)]), ((bfloat16_t)y_fragment[((i_28 * 8) + 5)])), __pack_half2(((bfloat16_t)y_fragment[((i_28 * 8) + 6)]), ((bfloat16_t)y_fragment[((i_28 * 8) + 7)])));
          }
          bar_2[0].arrive();
          if ((bool)calc_mt) {
            g_prod_Y[0] = (g_prod_Y[0] + g_shared[(((i_s_2 & 1) * 32) + 31)]);
            bar_2[0].wait((i_s_2 & 1));
            tl::__sync_thread_partial<5, 128>();
            #pragma unroll
            for (int i_29 = 0; i_29 < 32; ++i_29) {
              uint1 __9;
              float2 v__9 = *(float2*)(m_fragment_L + (i_29 * 2));
              (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__9))[0]);
              *(uint1*)(h_shared_local_cast_6 + 0) = __9;
              *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((((((int)threadIdx.x) >> 6) * 32) + (((i_29 & 7) >> 1) * 8)) >> 6) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_29 >> 3) * 1024)) + ((i_29 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_29 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_29 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 16384)) = *(uint1*)(h_shared_local_cast_6 + 0);
            }
            bar_3[0].wait((i_s_2 & 1));
            {
              bfloat16_t A_local_5[8];
              bfloat16_t B_local_5[16];
              #pragma unroll
              for (int i_30 = 0; i_30 < 4; ++i_30) {
                float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
                *(float4*)(z_fragment_L + (i_30 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
              }
              tl::__sync_thread_partial<5, 128>();
              for (int ki_5 = 0; ki_5 < 8; ++ki_5) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_5 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_5 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_5[0])));
                for (int i_31 = 0; i_31 < 2; ++i_31) {
                  tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((ki_5 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_31) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_5[(i_31 * 8)])));
                }
                for (int j_5 = 0; j_5 < 2; ++j_5) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(z_fragment_L + (j_5 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(z_fragment_L + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
                }
              }
            }
            tl::__sync_thread_partial<5, 128>();
            #pragma unroll
            for (int i_32 = 0; i_32 < 2; ++i_32) {
              tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_32 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) - 128) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_32 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) + 384) & 511)) + 43008)])), __pack_half2(((bfloat16_t)z_fragment_L[(i_32 * 8)]), ((bfloat16_t)z_fragment_L[((i_32 * 8) + 1)])), __pack_half2(((bfloat16_t)z_fragment_L[((i_32 * 8) + 2)]), ((bfloat16_t)z_fragment_L[((i_32 * 8) + 3)])), __pack_half2(((bfloat16_t)z_fragment_L[((i_32 * 8) + 4)]), ((bfloat16_t)z_fragment_L[((i_32 * 8) + 5)])), __pack_half2(((bfloat16_t)z_fragment_L[((i_32 * 8) + 6)]), ((bfloat16_t)z_fragment_L[((i_32 * 8) + 7)])));
            }
            {
              bfloat16_t A_local_6[32];
              bfloat16_t B_local_6[16];
              tl::__sync_thread_partial<5, 128>();
              for (int ki_6 = 0; ki_6 < 2; ++ki_6) {
                for (int i_33 = 0; i_33 < 4; ++i_33) {
                  tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (ki_6 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_33 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_33 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])), (&(A_local_6[(i_33 * 8)])));
                }
                for (int i_34 = 0; i_34 < 2; ++i_34) {
                  tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki_6 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_34) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 43008)])), (&(B_local_6[(i_34 * 8)])));
                }
                for (int i_35 = 0; i_35 < 4; ++i_35) {
                  for (int j_6 = 0; j_6 < 2; ++j_6) {
                    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(m_fragment_L + ((i_35 * 16) + (j_6 * 8))), reinterpret_cast<const unsigned*>(A_local_6 + (i_35 * 8)), reinterpret_cast<const unsigned*>(B_local_6 + (j_6 * 8)));
                    tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(m_fragment_L + (((i_35 * 16) + (j_6 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_6 + (i_35 * 8)), reinterpret_cast<const unsigned*>(B_local_6 + ((j_6 * 8) + 4)));
                  }
                }
              }
            }
          }
          data_is_free[(i_s_2 & 1)].arrive();
        }
        if ((bool)calc_mt) {
          g_last_local_Y[0] = exp2f((g_prod_Y[0] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          #pragma unroll
          for (int i_36 = 0; i_36 < 64; ++i_36) {
            m_fragment_L[i_36] = (m_fragment_L[i_36] * g_last_local_Y[0]);
          }
          #pragma unroll
          for (int i_37 = 0; i_37 < 32; ++i_37) {
            uint1 __10;
            float2 v__10 = *(float2*)(m_fragment_L + (i_37 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__10))[0] = __float22bfloat162_rn(((float2*)(&v__10))[0]);
            *(uint1*)(mt_local_cast_7 + 0) = __10;
            *(uint1*)(mt + ((((((((((((int64_t)((int)blockIdx.y)) * (int64_t)524288) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_37) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_37) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_37) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)128)) = *(uint1*)(mt_local_cast_7 + 0);
          }
        } else {
          #pragma unroll
          for (int i_38 = 0; i_38 < 64; ++i_38) {
            m_fragment_L[i_38] = 0x0p+0f/*0.000000e+00*/;
          }
          #pragma unroll
          for (int i_39 = 0; i_39 < 32; ++i_39) {
            uint1 __11;
            float2 v__11 = *(float2*)(m_fragment_L + (i_39 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__11))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
            *(uint1*)(mt_local_cast_8 + 0) = __11;
            *(uint1*)(mt + ((((((((((((int64_t)((int)blockIdx.y)) * (int64_t)524288) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_39) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_39) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_39) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)128)) = *(uint1*)(mt_local_cast_8 + 0);
          }
        }
      } else {
        tl::warpgroup_reg_dealloc<24>();
        if (((int)threadIdx.x) < 416) {
          tl::__sync_thread_partial<6, 32>();
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 & 1)].wait((((i_s_3 >> 1) + 1) & 1));
            int left = ((i_s_3 * 32) + seq_start_idx);
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 16384)])), 0, (((int)blockIdx.x) >> 1), left, batch_idx);
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 18432)])), 64, (((int)blockIdx.x) >> 1), left, batch_idx);
              }
            } else {
              tl::__sync_thread_partial<6, 32>();
              #pragma unroll
              for (int i_40 = 0; i_40 < 16; ++i_40) {
                if ((((i_40 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_40)) && ((((i_40 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval = *(uint4*)(k + (((((((((int64_t)i_40) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)left) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)196608));
                  } else {
                    condval = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
                  }
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_40 >> 2) * 512)) + ((((i_40 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_40) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_40) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = condval;
                } else {
                  bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_40 >> 2) * 512)) + ((((i_40 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_40) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_40) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
                }
              }
            }
            data_is_ready[(i_s_3 & 1)].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::__sync_thread_partial<7, 32>();
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              data_is_free[(i_s_4 & 1)].wait((((i_s_4 >> 1) + 1) & 1));
              int left_1 = ((i_s_4 * 32) + seq_start_idx);
              if ((left_1 + 32) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 & 1)].expect_transaction(8192);
                  tl::fence_proxy_async();
                  tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_4 & 1) * 4096) + 24576)])), 0, ((int)blockIdx.x), left_1, batch_idx);
                }
              } else {
                #pragma unroll
                for (int i_41 = 0; i_41 < 16; ++i_41) {
                  if ((((i_41 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (seq_end_idx + 26)) {
                    bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_1;
                    if (((((13 <= ((((((int)threadIdx.x) >> 4) + left_1) >> 1) + i_41)) && ((((i_41 * 2) + (((int)threadIdx.x) >> 4)) + left_1) < (num_tokens + 26))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_1 = *(uint4*)(v + (((((((((int64_t)i_41) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)left_1) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)212992));
                    } else {
                      condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
                    }
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((i_s_4 & 1) * 4096) + (i_41 * 256)) + (((int)threadIdx.x) * 8)) + 21248)) = condval_1;
                  } else {
                    bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((i_s_4 & 1) * 4096) + (i_41 * 256)) + (((int)threadIdx.x) * 8)) + 21248)) = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
                  }
                }
              }
              tl::__sync_thread_partial<7, 32>();
              if ((left_1 + 32) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 & 1)].expect_transaction(2048);
                  tl::fence_proxy_async();
                  tl::tma_load(a_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_4 & 1) * 1024) + 40960)])), 0, ((int)blockIdx.x), left_1, batch_idx);
                }
              } else {
                #pragma unroll
                for (int i_42 = 0; i_42 < 4; ++i_42) {
                  if ((((i_42 * 8) + (((int)threadIdx.x) >> 2)) + left_1) < (seq_end_idx + 104)) {
                    bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_2;
                    if (((((13 <= ((((((int)threadIdx.x) >> 2) + left_1) >> 3) + i_42)) && ((((i_42 * 8) + (((int)threadIdx.x) >> 2)) + left_1) < (num_tokens + 104))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_2 = *(uint4*)(a + (((((((((int64_t)i_42) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)1024)) + (((int64_t)left_1) * (int64_t)1024)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)1024)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) - (int64_t)106496));
                    } else {
                      condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
                    }
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_4 & 1) * 1024) + (i_42 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 37632)) = condval_2;
                  } else {
                    bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_4 & 1) * 1024) + (i_42 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 37632)) = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
                  }
                }
              }
              data_is_ready[(i_s_4 & 1)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 & 1)].wait((((i_s_5 >> 1) + 1) & 1));
                int left_2 = ((i_s_5 * 32) + seq_start_idx);
                if ((left_2 + 32) <= seq_end_idx) {
                  float condval_3;
                  if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_3 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((int64_t)((int)blockIdx.x))) - (int64_t)14336)];
                  } else {
                    condval_3 = 0x0p+0f/*0.000000e+00*/;
                  }
                  g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_3;
                } else {
                  if ((left_2 + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                    float condval_4;
                    if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_4 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((int64_t)((int)blockIdx.x))) - (int64_t)14336)];
                    } else {
                      condval_4 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_4;
                  } else {
                    float condval_5;
                    if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_5 = g[((((((int64_t)seq_end_idx) * (int64_t)32) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((int64_t)((int)blockIdx.x))) - (int64_t)32)];
                    } else {
                      condval_5 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_5;
                  }
                }
                if ((left_2 + 32) <= seq_end_idx) {
                  float condval_6;
                  if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_6 = b[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((int64_t)((int)blockIdx.x))) - (int64_t)14336)];
                  } else {
                    condval_6 = 0x0p+0f/*0.000000e+00*/;
                  }
                  b_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_6;
                } else {
                  if ((left_2 + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                    float condval_7;
                    if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_7 = b[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((int64_t)((int)blockIdx.x))) - (int64_t)14336)];
                    } else {
                      condval_7 = 0x0p+0f/*0.000000e+00*/;
                    }
                    b_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_7;
                  } else {
                    b_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = 0x0p+0f/*0.000000e+00*/;
                  }
                }
                data_is_ready[(i_s_5 & 1)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_iters; ++i_s_6) {
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                bar_1[0].wait((i_s_6 & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  tl::tma_store(h_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), 0, 0, ((int)blockIdx.x), (chunk_start_idx + i_s_6), batch_idx);
                  tl::tma_store(h_desc, (&(((bfloat16_t*)buf_dyn_shmem)[8192])), 64, 0, ((int)blockIdx.x), (chunk_start_idx + i_s_6), batch_idx);
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0>();
                }
                bar_2[0].arrive();
              }
            }
          }
        }
      }
    }
  }
}


// ---- flashqla_fused_cp_packed_strided ----
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void flashqla_fused_cp_packed_strided(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cp_seq_map, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const int64_t* __restrict__ raw_cu_seqlens, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(512, 1) flashqla_fused_cp_packed_strided(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cp_seq_map, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const int64_t* __restrict__ raw_cu_seqlens, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t data_is_ready_mem[2];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[2];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_o_mem[1];
  auto bar_o = reinterpret_cast<Barrier*>(bar_o_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t _bar_2_mem[1];
  auto _bar_2 = reinterpret_cast<Barrier*>(_bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  __shared__ __align__(16) uint64_t bar_4_mem[1];
  auto bar_4 = reinterpret_cast<Barrier*>(bar_4_mem);
  __shared__ __align__(16) uint64_t bar_5_mem[1];
  auto bar_5 = reinterpret_cast<Barrier*>(bar_5_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int raw_batch_idx = 0;
  int raw_seq_end_idx = 0;
  signed char need_store_final_state = (signed char)0;
  int num_iters = 0;
  int num_unmasked_iters = 0;
  float h_fragment[64];
  __shared__ __align__(16) float g_exp_shared[32];
  __shared__ __align__(16) float g_shared[64];
  __shared__ __align__(16) float b_shared[64];
  int seq_split_idx = 0;
  int chunk_split_idx = 0;
  float g_last_local[1];
  __shared__ __align__(16) float g_rev_exp_shared[32];
  float u_fragment[16];
  bfloat16_t v_shared_local_cast[2];
  bfloat16_t v_shared_local_cast_1[2];
  float v_fragment[16];
  float p_fragment[8];
  float g_fragment[8];
  float a_fragment[8];
  bfloat16_t a_shared_local_cast_2[2];
  bfloat16_t a_shared_local_cast_3[2];
  float o_fragment[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(o_desc);
    tl::prefetch_tma_descriptor(h_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    bar_o[0].init(128);
    bar_0[0].init(416);
    bar_1[0].init(256);
    _bar_2[0].init(128);
    bar_3[0].init(128);
    bar_4[0].init(128);
    bar_5[0].init(416);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = 0;
  seq_start_idx = ((int)cu_seqlens[((int64_t)((int)blockIdx.y))]);
  seq_end_idx = ((int)cu_seqlens[(((int64_t)((int)blockIdx.y)) + (int64_t)1)]);
  chunk_start_idx = ((int)chunk_offsets[((int64_t)((int)blockIdx.y))]);
  raw_batch_idx = ((int)cp_seq_map[((int64_t)((int)blockIdx.y))]);
  int64_t condval;
  if (((-1 <= raw_batch_idx) && (raw_batch_idx < raw_batch_size))) {
    condval = raw_cu_seqlens[(((int64_t)raw_batch_idx) + (int64_t)1)];
  } else {
    condval = (int64_t)0;
  }
  raw_seq_end_idx = ((int)condval);
  need_store_final_state = ((signed char)((bool)1 & (raw_seq_end_idx == seq_end_idx)));
  num_iters = (((seq_end_idx + 31) - seq_start_idx) >> 5);
  num_unmasked_iters = ((seq_end_idx - seq_start_idx) >> 5);
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<160>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(h0 + ((((((((((((int64_t)((int)blockIdx.y)) * (int64_t)524288) + ((((int64_t)((int)blockIdx.x)) & (int64_t)31) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)5) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    }
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      data_is_ready[(i_s & 1)].wait(((i_s & 3) >> 1));
      bar_0[0].arrive();
      bar_0[0].wait((i_s & 1));
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_1 >> 1) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)h_fragment[(i_1 * 8)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 7)])));
      }
      bar_1[0].arrive();
      bar_1[0].wait((i_s & 1));
      g_last_local[0] = g_exp_shared[31];
      #pragma unroll
      for (int i_2 = 0; i_2 < 64; ++i_2) {
        h_fragment[i_2] = (h_fragment[i_2] * g_last_local[0]);
      }
      bar_5[0].arrive();
      bar_5[0].wait((i_s & 1));
      {
        bfloat16_t A_local[32];
        bfloat16_t B_local[16];
        tl::__sync_thread_partial<3, 128>();
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((i_s & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A_local[(i_3 * 8)])));
          }
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 34816)])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + ((i_5 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + (((i_5 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      data_is_free[(i_s & 1)].arrive();
    }
    if ((bool)need_store_final_state) {
      if (0 <= raw_batch_idx) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          if (raw_batch_idx < raw_batch_size) {
            *(float2*)(ht + ((((((((((((int64_t)raw_batch_idx) * (int64_t)524288) + ((((int64_t)((int)blockIdx.x)) & (int64_t)31) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_6) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)5) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_6) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_fragment + (i_6 * 2));
          }
        }
      }
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<128>();
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial<3, 128>();
        if (((int)threadIdx.x) < 160) {
          g_exp_shared[(((int)threadIdx.x) - 128)] = exp2f((g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          float condval_1;
          if (((((i_s_1 * 32) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 128))) {
            condval_1 = exp2f(((g_shared[(((i_s_1 & 1) * 32) + 31)] - g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          } else {
            condval_1 = 0x0p+0f/*0.000000e+00*/;
          }
          g_rev_exp_shared[(((int)threadIdx.x) - 128)] = condval_1;
        }
        bar_1[0].arrive();
        bar_1[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_1[8];
          bfloat16_t B_local_1[16];
          #pragma unroll
          for (int i_7 = 0; i_7 < 4; ++i_7) {
            float broadcast_var = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(u_fragment + (i_7 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          }
          for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 4096) + ((ki_1 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(A_local_1[0])));
            for (int i_8 = 0; i_8 < 2; ++i_8) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((ki_1 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_8) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_8 * 8)])));
            }
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial<3, 128>();
        #pragma unroll
        for (int i_9 = 0; i_9 < 8; ++i_9) {
          float2 __1;
            float2 v_ = *(float2*)(u_fragment + (i_9 * 2));
            float2 v__1 = make_float2((g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
            *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(u_fragment + (i_9 * 2)) = __1;
        }
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          *(uint1*)(v_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_10 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576));
          float2 __2;
            float2 v__2 = *(float2*)(u_fragment + (i_10 * 2));
            float2 __3;
            uint1 v__3 = *(uint1*)(v_shared_local_cast + 0);
            ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
            *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(v__2.x)), *(float2*)(&(__3.x)));
          *(float2*)(u_fragment + (i_10 * 2)) = __2;
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_11 = 0; i_11 < 8; ++i_11) {
          uint1 __4;
          float2 v__4 = *(float2*)(u_fragment + (i_11 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
          *(uint1*)(v_shared_local_cast_1 + 0) = __4;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_11 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_11 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576)) = *(uint1*)(v_shared_local_cast_1 + 0);
        }
        bar_3[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_2[8];
          bfloat16_t B_local_2[16];
          #pragma unroll
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(v_fragment + (i_12 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
          }
          tl::__sync_thread_partial<4, 128>();
          for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 28672)])), (&(A_local_2[0])));
            for (int i_13 = 0; i_13 < 2; ++i_13) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_s_1 & 1) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])), (&(B_local_2[(i_13 * 8)])));
            }
            for (int j_2 = 0; j_2 < 2; ++j_2) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 32768)])), __pack_half2(((bfloat16_t)v_fragment[(i_14 * 8)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 7)])));
        }
        bar_4[0].arrive();
        #pragma unroll
        for (int i_15 = 0; i_15 < 8; ++i_15) {
          float2 __5;
            float2 v__5 = *(float2*)(v_fragment + (i_15 * 2));
            float2 v__6 = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
          *(float2*)(v_fragment + (i_15 * 2)) = __5;
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 34816)])), __pack_half2(((bfloat16_t)v_fragment[(i_16 * 8)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 7)])));
        }
        bar_5[0].arrive();
        bar_5[0].wait((i_s_1 & 1));
        data_is_free[(i_s_1 & 1)].arrive();
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<128>();
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 & 1)].wait(((i_s_2 & 3) >> 1));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_3[8];
            bfloat16_t B_local_3[8];
            #pragma unroll
            for (int i_17 = 0; i_17 < 2; ++i_17) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(p_fragment + (i_17 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
            }
            for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_3[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_3[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 4), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 4));
            }
          }
          #pragma unroll
          for (int i_18 = 0; i_18 < 4; ++i_18) {
            float2 __6;
              float2 v__7 = make_float2(g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              float2 v__8 = *(float2*)(g_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
            *(float2*)(g_fragment + (i_18 * 2)) = __6;
          }
          #pragma unroll
          for (int i_19 = 0; i_19 < 8; ++i_19) {
            if ((((((((int)threadIdx.x) >> 6) * 16) + ((i_19 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_19 & 1)) <= ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_19 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64)) {
              g_fragment[i_19] = exp2f((g_fragment[i_19] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
            } else {
              g_fragment[i_19] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_20 = 0; i_20 < 4; ++i_20) {
            *(uint1*)(a_shared_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_20 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_20 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 28672));
            float2 __7;
            uint1 v__9 = *(uint1*)(a_shared_local_cast_2 + 0);
            ((float2*)(&__7))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__9))[0]);
            *(float2*)(a_fragment + (i_20 * 2)) = __7;
          }
          #pragma unroll
          for (int i_21 = 0; i_21 < 8; ++i_21) {
            a_fragment[i_21] = (a_fragment[i_21] * g_fragment[i_21]);
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 4; ++i_22) {
            float2 __8;
              float2 v__10 = *(float2*)(a_fragment + (i_22 * 2));
              float2 v__11 = *(float2*)(b_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_22 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)));
            *(float2*)(a_fragment + (i_22 * 2)) = __8;
          }
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_23 = 0; i_23 < 4; ++i_23) {
            uint1 __9;
            float2 v__12 = *(float2*)(a_fragment + (i_23 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__12))[0]);
            *(uint1*)(a_shared_local_cast_3 + 0) = __9;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_23 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_23 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 28672)) = *(uint1*)(a_shared_local_cast_3 + 0);
          }
          bar_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_4[8];
            bfloat16_t B_local_4[16];
            #pragma unroll
            for (int i_24 = 0; i_24 < 4; ++i_24) {
              float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(o_fragment + (i_24 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            }
            tl::__sync_thread_partial<5, 128>();
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_4 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_4[0])));
              for (int i_25 = 0; i_25 < 2; ++i_25) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((ki_4 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_25) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_25 * 8)])));
              }
              for (int j_3 = 0; j_3 < 2; ++j_3) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_3 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_3 * 8) + 4)));
              }
            }
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 8; ++i_26) {
            p_fragment[i_26] = (p_fragment[i_26] * (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_fragment[i_26]));
          }
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((((((((int)threadIdx.x) >> 7) * 32) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 8)) - 64) >> 8) * 256)) + (((((((((int)threadIdx.x) >> 7) * 32) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 8)) + 192) & 255)) + 36864)])), __pack_half2(((bfloat16_t)p_fragment[0]), ((bfloat16_t)p_fragment[1])), __pack_half2(((bfloat16_t)p_fragment[2]), ((bfloat16_t)p_fragment[3])), __pack_half2(((bfloat16_t)p_fragment[4]), ((bfloat16_t)p_fragment[5])), __pack_half2(((bfloat16_t)p_fragment[6]), ((bfloat16_t)p_fragment[7])));
          bar_3[0].arrive();
          #pragma unroll
          for (int i_27 = 0; i_27 < 8; ++i_27) {
            float2 __10;
              float2 v__13 = *(float2*)(o_fragment + (i_27 * 2));
              float2 v__14 = make_float2((0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]), (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
              *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
            *(float2*)(o_fragment + (i_27 * 2)) = __10;
          }
          bar_4[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_5[8];
            bfloat16_t B_local_5[16];
            tl::__sync_thread_partial<5, 128>();
            for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_5) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 36864)])), (&(A_local_5[0])));
              for (int i_28 = 0; i_28 < 2; ++i_28) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki_5 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_28) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_5[(i_28 * 8)])));
              }
              for (int j_4 = 0; j_4 < 2; ++j_4) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_4 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_4 * 8) + 4)));
              }
            }
          }
          bar_5[0].arrive();
          bar_5[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_29 = 0; i_29 < 2; ++i_29) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_29 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) - 128) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_29 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) + 384) & 511)) + 30720)])), __pack_half2(((bfloat16_t)o_fragment[(i_29 * 8)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 1)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 2)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 3)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 4)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 5)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 6)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 7)])));
          }
          data_is_free[(i_s_2 & 1)].arrive();
        }
        bar_o[0].arrive();
      } else {
        tl::warpgroup_reg_dealloc<32>();
        if (((int)threadIdx.x) < 416) {
          tl::__sync_thread_partial<6, 32>();
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 & 1)].wait((((i_s_3 >> 1) + 1) & 1));
            int left = ((i_s_3 * 32) + seq_start_idx);
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 16384)])), 0, ((((int)blockIdx.x) & 31) >> 1), left, batch_idx);
                tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 18432)])), 64, ((((int)blockIdx.x) & 31) >> 1), left, batch_idx);
              }
            } else {
              tl::__sync_thread_partial<6, 32>();
              #pragma unroll
              for (int i_30 = 0; i_30 < 16; ++i_30) {
                if ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_2;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_30)) && ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_2 = *(uint4*)(q + (((((((((int64_t)i_30) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)left) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)31) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)196608));
                  } else {
                    condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
                  }
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_30 >> 2) * 512)) + ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_30) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_30) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = condval_2;
                } else {
                  bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_30 >> 2) * 512)) + ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_30) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_30) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
                }
              }
            }
            tl::__sync_thread_partial<6, 32>();
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 8192)])), 0, ((((int)blockIdx.x) & 31) >> 1), left, batch_idx);
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 10240)])), 64, ((((int)blockIdx.x) & 31) >> 1), left, batch_idx);
              }
            } else {
              tl::__sync_thread_partial<6, 32>();
              #pragma unroll
              for (int i_31 = 0; i_31 < 16; ++i_31) {
                if ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_3;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_31)) && ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_3 = *(uint4*)(k + (((((((((int64_t)i_31) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)left) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)31) >> (int64_t)1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)196608));
                  } else {
                    condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
                  }
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_31 >> 2) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_31) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_31) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = condval_3;
                } else {
                  bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_31 >> 2) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_31) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_31) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
                }
              }
            }
            data_is_ready[(i_s_3 & 1)].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::__sync_thread_partial<7, 32>();
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              data_is_free[(i_s_4 & 1)].wait((((i_s_4 >> 1) + 1) & 1));
              int left_1 = ((i_s_4 * 32) + seq_start_idx);
              if ((left_1 + 32) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 & 1)].expect_transaction(4096);
                  tl::fence_proxy_async();
                  tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_4 & 1) * 2048) + 24576)])), ((((int)blockIdx.x) >> 5) * 64), (((int)blockIdx.x) & 31), left_1, batch_idx);
                }
              } else {
                tl::__sync_thread_partial<7, 32>();
                #pragma unroll
                for (int i_32 = 0; i_32 < 8; ++i_32) {
                  if ((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (seq_end_idx + 52)) {
                    bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_4;
                    if (((((13 <= ((((((int)threadIdx.x) >> 3) + left_1) >> 2) + i_32)) && ((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (num_tokens + 52))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_4 = *(uint4*)(v + ((((((((((int64_t)i_32) * (int64_t)32768) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)8192)) + (((int64_t)left_1) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)5) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)425984));
                    } else {
                      condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
                    }
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_s_4 & 1) * 2048) + ((i_32 >> 1) * 512)) + (((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_32) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = condval_4;
                  } else {
                    bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_s_4 & 1) * 2048) + ((i_32 >> 1) * 512)) + (((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_32) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
                  }
                }
              }
              if ((left_1 + 32) <= seq_end_idx) {
                float condval_5;
                if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                  condval_5 = b[(((((((int64_t)left_1) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) & (int64_t)31)) - (int64_t)13312)];
                } else {
                  condval_5 = 0x0p+0f/*0.000000e+00*/;
                }
                b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_5;
              } else {
                if ((left_1 + ((int)threadIdx.x)) < (seq_end_idx + 416)) {
                  float condval_6;
                  if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_6 = b[(((((((int64_t)left_1) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) & (int64_t)31)) - (int64_t)13312)];
                  } else {
                    condval_6 = 0x0p+0f/*0.000000e+00*/;
                  }
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_6;
                } else {
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = 0x0p+0f/*0.000000e+00*/;
                }
              }
              data_is_ready[(i_s_4 & 1)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              tl::__sync_thread_partial<8, 32>();
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 & 1)].wait((((i_s_5 >> 1) + 1) & 1));
                int left_2 = ((i_s_5 * 32) + seq_start_idx);
                if ((left_2 + 32) <= seq_end_idx) {
                  if (tl::tl_shuffle_elect<32>()) {
                    data_is_ready[(i_s_5 & 1)].expect_transaction(2048);
                    tl::fence_proxy_async();
                    tl::tma_load(a_desc, data_is_ready[(i_s_5 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_5 & 1) * 1024) + 28672)])), 0, (((int)blockIdx.x) & 31), left_2, batch_idx);
                  }
                } else {
                  #pragma unroll
                  for (int i_33 = 0; i_33 < 4; ++i_33) {
                    if ((((i_33 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (seq_end_idx + 112)) {
                      bfloat16_t broadcast_var_10 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      uint4 condval_7;
                      if (((((14 <= ((((((int)threadIdx.x) >> 2) + left_2) >> 3) + i_33)) && ((((i_33 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (num_tokens + 112))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_7 = *(uint4*)(a + (((((((((int64_t)i_33) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)1024)) + (((int64_t)left_2) * (int64_t)1024)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)31) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) - (int64_t)114688));
                      } else {
                        condval_7 = make_uint4(__pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10));
                      }
                      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_5 & 1) * 1024) + (i_33 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 25088)) = condval_7;
                    } else {
                      bfloat16_t broadcast_var_11 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_5 & 1) * 1024) + (i_33 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 25088)) = make_uint4(__pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11));
                    }
                  }
                }
                if ((left_2 + 32) <= seq_end_idx) {
                  float condval_8;
                  if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_8 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) & (int64_t)31)) - (int64_t)14336)];
                  } else {
                    condval_8 = 0x0p+0f/*0.000000e+00*/;
                  }
                  g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_8;
                } else {
                  if ((left_2 + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                    float condval_9;
                    if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_9 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) & (int64_t)31)) - (int64_t)14336)];
                    } else {
                      condval_9 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_9;
                  } else {
                    float condval_10;
                    if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_10 = g[((((((int64_t)seq_end_idx) * (int64_t)32) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) & (int64_t)31)) - (int64_t)32)];
                    } else {
                      condval_10 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_10;
                  }
                }
                data_is_ready[(i_s_5 & 1)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_unmasked_iters; ++i_s_6) {
                int right = ((i_s_6 * 32) + seq_start_idx);
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                if (0 < i_s_6) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)buf_dyn_shmem)[30720])), ((((int)blockIdx.x) >> 5) * 64), (((int)blockIdx.x) & 31), (right - 32), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((i_s_6 & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  tl::tma_store(h_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), ((((int)blockIdx.x) >> 5) * 64), 0, (((int)blockIdx.x) & 31), (chunk_start_idx + i_s_6), batch_idx);
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0>();
                }
              }
              if (num_unmasked_iters < num_iters) {
                seq_split_idx = ((num_unmasked_iters * 32) + seq_start_idx);
                chunk_split_idx = (chunk_start_idx + num_unmasked_iters);
                int right_1 = seq_split_idx;
                bar_0[0].arrive();
                bar_0[0].wait((num_unmasked_iters & 1));
                if (0 < num_unmasked_iters) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)buf_dyn_shmem)[30720])), ((((int)blockIdx.x) >> 5) * 64), (((int)blockIdx.x) & 31), (right_1 - 32), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((num_unmasked_iters & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  tl::tma_store(h_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), ((((int)blockIdx.x) >> 5) * 64), 0, (((int)blockIdx.x) & 31), chunk_split_idx, batch_idx);
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0>();
                }
              }
              seq_split_idx = (((num_iters * 32) + seq_start_idx) - 32);
              bar_o[0].wait(0);
              if (0 < num_iters) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_34 = 0; i_34 < 8; ++i_34) {
                    if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (seq_end_idx + 60)) {
                      if (15 <= ((((((int)threadIdx.x) >> 3) + seq_split_idx) >> 2) + i_34)) {
                        if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60)) {
                          if (batch_idx < 1) {
                            *(uint4*)(o + ((((((((((int64_t)i_34) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)seq_split_idx) * (int64_t)4096)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)5) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)245760)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_34 >> 1) * 512) + (((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_34) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 30720));
                          }
                        }
                      }
                    } else {
                      if ((((int)blockIdx.y) == (batch_size - 1)) && ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60))) {
                        if (15 <= ((((((int)threadIdx.x) >> 3) + seq_split_idx) >> 2) + i_34)) {
                          if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60)) {
                            if (batch_idx < 1) {
                              bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                              *(uint4*)(o + ((((((((((int64_t)i_34) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)seq_split_idx) * (int64_t)4096)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)31) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)5) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)245760)) = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


// ---- flashqla_cp_warmup ----
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void flashqla_cp_warmup(const int64_t* __restrict__ cu_seqlens, signed char* __restrict__ fallback_mask, const float* __restrict__ g, const signed char* __restrict__ ht_mask, int64_t* __restrict__ num_warmup_chunks, int batch_size, int num_tokens);
extern "C" __global__ void __launch_bounds__(32, 1) flashqla_cp_warmup(const int64_t* __restrict__ cu_seqlens, signed char* __restrict__ fallback_mask, const float* __restrict__ g, const signed char* __restrict__ ht_mask, int64_t* __restrict__ num_warmup_chunks, int batch_size, int num_tokens) {
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int num_iters = 0;
  float g_cumsum[1];
  int64_t n_fragment[1];
  signed char f_fragment[1];
  float g_fragment[1];
  if ((bool)ht_mask[((int64_t)((int)blockIdx.x))]) {
    num_warmup_chunks[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)threadIdx.x)))] = (int64_t)0;
  } else {
    seq_start_idx = ((int)cu_seqlens[((int64_t)((int)blockIdx.x))]);
    seq_end_idx = ((int)cu_seqlens[(((int64_t)((int)blockIdx.x)) + (int64_t)1)]);
    num_iters = ((seq_end_idx - seq_start_idx) >> 5);
    g_cumsum[0] = 0x0p+0f/*0.000000e+00*/;
    n_fragment[0] = ((int64_t)num_iters);
    f_fragment[0] = (signed char)1;
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      float condval;
      if (((1 <= (seq_end_idx - (i_s * 32))) && ((seq_end_idx - (i_s * 32)) <= num_tokens))) {
        condval = g[((((((int64_t)seq_end_idx) * (int64_t)32) + ((int64_t)((int)threadIdx.x))) - (((int64_t)i_s) * (int64_t)1024)) - (int64_t)32)];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      g_fragment[0] = condval;
      g_cumsum[0] = (g_cumsum[0] + g_fragment[0]);
      if ((g_cumsum[0] < -0x1.4p+3f/*-1.000000e+01*/) && (n_fragment[0] == ((int64_t)num_iters))) {
        n_fragment[0] = (((int64_t)i_s) + (int64_t)1);
        f_fragment[0] = (signed char)0;
      }
    }
    num_warmup_chunks[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)threadIdx.x)))] = n_fragment[0];
    fallback_mask[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)threadIdx.x)))] = ((signed char)((bool)f_fragment[0]));
  }
}


// ---- flashqla_cp_correct_h0 ----
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void flashqla_cp_correct_h0(float* __restrict__ cp_h0, const signed char* __restrict__ fallback_mask, __grid_constant__ const CUtensorMap ht_buffer_desc, __grid_constant__ const CUtensorMap mt_buffer_desc, const float* __restrict__ raw_h0, const int64_t* __restrict__ seq_map_r2c, int cp_batch_size, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(256, 1) flashqla_cp_correct_h0(float* __restrict__ cp_h0, const signed char* __restrict__ fallback_mask, __grid_constant__ const CUtensorMap ht_buffer_desc, __grid_constant__ const CUtensorMap mt_buffer_desc, const float* __restrict__ raw_h0, const int64_t* __restrict__ seq_map_r2c, int cp_batch_size, int raw_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t mbarrier_mem[8];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int num_iters = 0;
  float h_fragment[32];
  bfloat16_t h_shared_local_cast[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(ht_buffer_desc);
    tl::prefetch_tma_descriptor(mt_buffer_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
    mbarrier[6].init(128);
    mbarrier[7].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  seq_start_idx = ((int)seq_map_r2c[(((int64_t)((int)blockIdx.x)) >> (int64_t)7)]);
  seq_end_idx = ((int)seq_map_r2c[((((int64_t)((int)blockIdx.x)) >> (int64_t)7) + (int64_t)1)]);
  num_iters = (seq_end_idx - seq_start_idx);
  int seq_start_idx_1 = seq_start_idx;
  int num_iters_1 = num_iters;
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int i_s = 0; i_s < (num_iters_1 - 1); ++i_s) {
      mbarrier[((i_s & 1) + 4)].wait((((i_s & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(i_s & 1)].arrive_and_expect_tx(8192);
        tl::tma_load(ht_buffer_desc, mbarrier[(i_s & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s & 1) * 4096) + 32768)])), ((((int)blockIdx.x) & 3) * 32), 0, ((((int)blockIdx.x) & 127) >> 2), (seq_start_idx_1 + i_s));
      }
      mbarrier[((i_s & 1) + 6)].wait((((i_s & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((i_s & 1) + 2)].arrive_and_expect_tx(32768);
        tl::tma_load(mt_buffer_desc, mbarrier[((i_s & 1) + 2)], (&(((bfloat16_t*)buf_dyn_shmem)[((i_s & 1) * 16384)])), 0, 0, ((((int)blockIdx.x) & 127) >> 2), (seq_start_idx_1 + i_s));
        tl::tma_load(mt_buffer_desc, mbarrier[((i_s & 1) + 2)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s & 1) * 16384) + 8192)])), 64, 0, ((((int)blockIdx.x) & 127) >> 2), (seq_start_idx_1 + i_s));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(raw_h0 + ((((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)2) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i) >> (int64_t)2) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)16)) + (((((int64_t)i) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    }
    for (int i_s_1 = 0; i_s_1 < (num_iters_1 - 1); ++i_s_1) {
      if (0 <= (seq_start_idx_1 + i_s_1)) {
        #pragma unroll
        for (int i_1 = 0; i_1 < 16; ++i_1) {
          if ((seq_start_idx_1 + i_s_1) < cp_batch_size) {
            *(float2*)(cp_h0 + (((((((((((((int64_t)seq_start_idx_1) * (int64_t)524288) + (((int64_t)i_s_1) * (int64_t)524288)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)127) >> (int64_t)2) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_1) >> (int64_t)2) * (int64_t)2048)) + ((((int64_t)i_1) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)16)) + (((((int64_t)i_1) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_fragment + (i_1 * 2));
          }
        }
      }
      bool condval;
      if (((0 <= (seq_start_idx_1 + i_s_1)) && ((seq_start_idx_1 + i_s_1) < cp_batch_size))) {
        condval = ((bool)fallback_mask[(((((int64_t)seq_start_idx_1) * (int64_t)32) + (((int64_t)i_s_1) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)127) >> (int64_t)2))]);
      } else {
        condval = (bool)0;
      }
      if (condval) {
        tl::__sync_thread_partial<3, 128>();
        #pragma unroll
        for (int i_2 = 0; i_2 < 4; ++i_2) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 40960)])), __pack_half2(((bfloat16_t)h_fragment[(i_2 * 8)]), ((bfloat16_t)h_fragment[((i_2 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_2 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_2 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_2 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_2 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_2 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_2 * 8) + 7)])));
        }
      }
      mbarrier[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        *(uint1*)(h_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((i_s_1 & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + ((i_3 >> 2) * 512)) + ((i_3 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_3 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 32768));
        float2 __1;
        uint1 v_ = *(uint1*)(h_shared_local_cast + 0);
        ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
        *(float2*)(h_fragment + (i_3 * 2)) = __1;
      }
      mbarrier[((i_s_1 & 1) + 4)].arrive();
      mbarrier[((i_s_1 & 1) + 2)].wait(((i_s_1 & 3) >> 1));
      bool condval_1;
      if (((0 <= (seq_start_idx_1 + i_s_1)) && ((seq_start_idx_1 + i_s_1) < cp_batch_size))) {
        condval_1 = ((bool)fallback_mask[(((((int64_t)seq_start_idx_1) * (int64_t)32) + (((int64_t)i_s_1) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)127) >> (int64_t)2))]);
      } else {
        condval_1 = (bool)0;
      }
      if (condval_1) {
        {
          bfloat16_t A_local[32];
          bfloat16_t B_local[8];
          for (int ki = 0; ki < 8; ++ki) {
            for (int i_4 = 0; i_4 < 4; ++i_4) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 16384) + ((ki >> 2) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (i_4 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_4 * 8)])));
            }
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 40960)])), (&(B_local[0])));
            for (int i_5 = 0; i_5 < 4; ++i_5) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + (i_5 * 8)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + ((i_5 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + 4));
            }
          }
        }
      }
      mbarrier[((i_s_1 & 1) + 6)].arrive();
    }
    if (1 <= (seq_start_idx_1 + num_iters_1)) {
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        if ((seq_start_idx_1 + num_iters_1) <= cp_batch_size) {
          *(float2*)(cp_h0 + ((((((((((((((int64_t)seq_start_idx_1) * (int64_t)524288) + (((int64_t)num_iters_1) * (int64_t)524288)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)127) >> (int64_t)2) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_6) >> (int64_t)2) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)16)) + (((((int64_t)i_6) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)524288)) = *(float2*)(h_fragment + (i_6 * 2));
        }
      }
    }
  }
}


// ---- flashqla_chunk_local_cumsum ----
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void flashqla_chunk_local_cumsum(const int64_t* __restrict__ chunk_indices, const int64_t* __restrict__ cu_seqlens, float* __restrict__ g_cumsum, const float* __restrict__ g_raw, int data_batch_size, int num_chunks, int num_tokens, int real_batch_size);
extern "C" __global__ void __launch_bounds__(128, 1) flashqla_chunk_local_cumsum(const int64_t* __restrict__ chunk_indices, const int64_t* __restrict__ cu_seqlens, float* __restrict__ g_cumsum, const float* __restrict__ g_raw, int data_batch_size, int num_chunks, int num_tokens, int real_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float gT_fragment[8];
  float src_buffer[8];
  float g_raw_local_cast[8];
  float g_raw_local_cast_1[8];
  float g_cumsum_local_cast_2[8];
  float g_cumsum_local_cast_3[8];
  float g_cumsum_local_cast_4[8];
  int64_t batch_idx = chunk_indices[(((int64_t)((int)blockIdx.x)) * (int64_t)2)];
  int64_t chunk_idx = chunk_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)2) + (int64_t)1)];
  int64_t condval;
  if ((((int64_t)0 <= batch_idx) && (batch_idx <= ((int64_t)real_batch_size)))) {
    condval = cu_seqlens[batch_idx];
  } else {
    condval = (int64_t)0;
  }
  int64_t seq_start_idx = condval;
  int64_t condval_1;
  if ((((int64_t)-1 <= batch_idx) && (batch_idx < ((int64_t)real_batch_size)))) {
    condval_1 = cu_seqlens[(batch_idx + (int64_t)1)];
  } else {
    condval_1 = (int64_t)0;
  }
  int64_t seq_end_idx = condval_1;
  if ((((chunk_idx * (int64_t)32) + seq_start_idx) + (int64_t)32) <= seq_end_idx) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    ulonglong4 condval_2;
    if ((((int64_t)0 <= (((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx)) && ((((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx) < ((int64_t)num_tokens)))) {
      condval_2 = tl::load_global_256(&(*(ulonglong4*)(g_raw + (((chunk_idx * (int64_t)1024) + (seq_start_idx * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))));
    } else {
      condval_2 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var), *(unsigned long long*)&make_float2(broadcast_var, broadcast_var));
    }
    *(ulonglong4*)(g_raw_local_cast + 0) = condval_2;
    for (int i = 0; i < 4; ++i) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      float2 condval_3;
      if ((((((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx) < ((int64_t)num_tokens)) && ((int64_t)0 <= (((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx)))) {
        condval_3 = *(float2*)(g_raw_local_cast + (i * 2));
      } else {
        condval_3 = make_float2(broadcast_var_1, broadcast_var_1);
      }
      *(float2*)(gT_fragment + (i * 2)) = condval_3;
    }
  } else {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    ulonglong4 condval_4;
    if ((((int64_t)0 <= (((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx)) && ((((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx) < ((int64_t)num_tokens)))) {
      condval_4 = tl::load_global_256(&(*(ulonglong4*)(g_raw + (((chunk_idx * (int64_t)1024) + (seq_start_idx * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))));
    } else {
      condval_4 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2));
    }
    *(ulonglong4*)(g_raw_local_cast_1 + 0) = condval_4;
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      if ((((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx) < seq_end_idx) {
        *(float2*)(gT_fragment + (i_1 * 2)) = *(float2*)(g_raw_local_cast_1 + (i_1 * 2));
      } else {
        float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(gT_fragment + (i_1 * 2)) = make_float2(broadcast_var_3, broadcast_var_3);
      }
    }
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 2) * 33) + ((((int)threadIdx.x) & 3) * 8)) + i_2)] = gT_fragment[i_2];
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    src_buffer[i_3] = ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 31) * 33) + (i_3 * 4)) + (((int)threadIdx.x) >> 5))];
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    ((float*)buf_dyn_shmem)[(((i_4 * 128) + ((int)threadIdx.x)) + 1056)] = src_buffer[i_4];
  }
  __syncthreads();
  tl::CumSum2D<128, 1, false>::run((&(((float*)buf_dyn_shmem)[1056])), (&(((float*)buf_dyn_shmem)[1056])), 32, 32);
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    src_buffer[i_5] = ((float*)buf_dyn_shmem)[(((i_5 * 128) + ((int)threadIdx.x)) + 1056)];
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 31) * 33) + (i_6 * 4)) + (((int)threadIdx.x) >> 5))] = src_buffer[i_6];
  }
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 8; ++i_7) {
    gT_fragment[i_7] = ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 2) * 33) + ((((int)threadIdx.x) & 3) * 8)) + i_7)];
  }
  if ((((chunk_idx * (int64_t)32) + seq_start_idx) + (int64_t)32) <= seq_end_idx) {
    if (((((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx) < ((int64_t)num_tokens)) && ((int64_t)0 <= (((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx))) {
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        *(float2*)(g_cumsum_local_cast_2 + (i_8 * 2)) = *(float2*)(gT_fragment + (i_8 * 2));
      }
      tl::store_global_256(&(*(ulonglong4*)(g_cumsum + (((chunk_idx * (int64_t)1024) + (seq_start_idx * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))), *(ulonglong4*)(g_cumsum_local_cast_2 + 0));
    }
  } else {
    if ((((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx) < seq_end_idx) {
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        *(float2*)(g_cumsum_local_cast_3 + (i_9 * 2)) = *(float2*)(gT_fragment + (i_9 * 2));
      }
      if ((int64_t)0 <= (((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx)) {
        if ((((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx) < ((int64_t)num_tokens)) {
          tl::store_global_256(&(*(ulonglong4*)(g_cumsum + (((chunk_idx * (int64_t)1024) + (seq_start_idx * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))), *(ulonglong4*)(g_cumsum_local_cast_3 + 0));
        }
      }
    }
  }
  if (batch_idx == (((int64_t)real_batch_size) - (int64_t)1)) {
    if ((seq_end_idx <= (((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx)) && ((((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx) < ((int64_t)num_tokens))) {
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
        *(float2*)(g_cumsum_local_cast_4 + (i_10 * 2)) = make_float2(broadcast_var_4, broadcast_var_4);
      }
      if ((int64_t)0 <= (((chunk_idx * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) + seq_start_idx)) {
        tl::store_global_256(&(*(ulonglong4*)(g_cumsum + (((chunk_idx * (int64_t)1024) + (seq_start_idx * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))), *(ulonglong4*)(g_cumsum_local_cast_4 + 0));
      }
    }
  }
}

// ---- flashqla_fused_nocp_packed_strided ----
// Frozen from FlashQLA blackwell_sm120 fused_fwd.py with T.StridedTensor Q/K/V
// (token stride = packed 8192).  Fixes the masked-fallback stride bug in the
// original contiguous flashqla_fused_nocp (Q/K 2048, V 4096 elems/token) when
// Atlas feeds the packed [1,T,8192] QKV buffer: both the TMA path and the
// masked tail reads now use the 8192-element packed pitch.
// Source SHA-256 (generated .cu before symbol rename): 44708d78d69ccd49...

#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void flashqla_fused_nocp_packed_strided(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(512, 1) flashqla_fused_nocp_packed_strided(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t data_is_ready_mem[2];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[2];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_o_mem[1];
  auto bar_o = reinterpret_cast<Barrier*>(bar_o_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t _bar_2_mem[1];
  auto _bar_2 = reinterpret_cast<Barrier*>(_bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  __shared__ __align__(16) uint64_t bar_4_mem[1];
  auto bar_4 = reinterpret_cast<Barrier*>(bar_4_mem);
  __shared__ __align__(16) uint64_t bar_5_mem[1];
  auto bar_5 = reinterpret_cast<Barrier*>(bar_5_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int raw_batch_idx = 0;
  int raw_seq_end_idx = 0;
  signed char need_store_final_state = (signed char)0;
  int num_iters = 0;
  int num_unmasked_iters = 0;
  float h_fragment[64];
  __shared__ __align__(16) float g_exp_shared[32];
  __shared__ __align__(16) float g_shared[64];
  __shared__ __align__(16) float b_shared[64];
  int seq_split_idx = 0;
  int chunk_split_idx = 0;
  float g_last_local[1];
  __shared__ __align__(16) float g_rev_exp_shared[32];
  float u_fragment[16];
  bfloat16_t v_shared_local_cast[2];
  bfloat16_t v_shared_local_cast_1[2];
  float v_fragment[16];
  float p_fragment[8];
  float g_fragment[8];
  float a_fragment[8];
  bfloat16_t a_shared_local_cast_2[2];
  bfloat16_t a_shared_local_cast_3[2];
  float o_fragment[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(o_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    bar_o[0].init(128);
    bar_0[0].init(416);
    bar_1[0].init(256);
    _bar_2[0].init(128);
    bar_3[0].init(128);
    bar_4[0].init(128);
    bar_5[0].init(416);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = 0;
  seq_start_idx = ((int)cu_seqlens[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)]);
  seq_end_idx = ((int)cu_seqlens[((((int64_t)((int)blockIdx.x)) >> (int64_t)6) + (int64_t)1)]);
  chunk_start_idx = ((int)chunk_offsets[(((int64_t)((int)blockIdx.x)) >> (int64_t)6)]);
  raw_batch_idx = (((int)blockIdx.x) >> 6);
  raw_seq_end_idx = seq_end_idx;
  need_store_final_state = ((signed char)((bool)1 & (raw_seq_end_idx == seq_end_idx)));
  num_iters = (((seq_end_idx + 31) - seq_start_idx) >> 5);
  num_unmasked_iters = ((seq_end_idx - seq_start_idx) >> 5);
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<160>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(h0 + ((((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    }
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      data_is_ready[(i_s & 1)].wait(((i_s & 3) >> 1));
      bar_0[0].arrive();
      bar_0[0].wait((i_s & 1));
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_1 >> 1) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)h_fragment[(i_1 * 8)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 7)])));
      }
      bar_1[0].arrive();
      bar_1[0].wait((i_s & 1));
      g_last_local[0] = g_exp_shared[31];
      #pragma unroll
      for (int i_2 = 0; i_2 < 64; ++i_2) {
        h_fragment[i_2] = (h_fragment[i_2] * g_last_local[0]);
      }
      bar_5[0].arrive();
      bar_5[0].wait((i_s & 1));
      {
        bfloat16_t A_local[32];
        bfloat16_t B_local[16];
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((i_s & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A_local[(i_3 * 8)])));
          }
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 34816)])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + ((i_5 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + (((i_5 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      data_is_free[(i_s & 1)].arrive();
    }
    if ((bool)need_store_final_state) {
      if (0 <= raw_batch_idx) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          if (raw_batch_idx < raw_batch_size) {
            *(float2*)(ht + ((((((((((((int64_t)raw_batch_idx) * (int64_t)524288) + (((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_6) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_6) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_fragment + (i_6 * 2));
          }
        }
      }
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<128>();
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial<3, 128>();
        if (((int)threadIdx.x) < 160) {
          g_exp_shared[(((int)threadIdx.x) - 128)] = exp2f((g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          float condval;
          if (((((i_s_1 * 32) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 128))) {
            condval = exp2f(((g_shared[(((i_s_1 & 1) * 32) + 31)] - g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          } else {
            condval = 0x0p+0f/*0.000000e+00*/;
          }
          g_rev_exp_shared[(((int)threadIdx.x) - 128)] = condval;
        }
        bar_1[0].arrive();
        bar_1[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_1[8];
          bfloat16_t B_local_1[16];
          #pragma unroll
          for (int i_7 = 0; i_7 < 4; ++i_7) {
            float broadcast_var = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(u_fragment + (i_7 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          }
          for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 4096) + ((ki_1 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(A_local_1[0])));
            for (int i_8 = 0; i_8 < 2; ++i_8) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((ki_1 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_8) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_8 * 8)])));
            }
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial<3, 128>();
        #pragma unroll
        for (int i_9 = 0; i_9 < 8; ++i_9) {
          float2 __1;
            float2 v_ = *(float2*)(u_fragment + (i_9 * 2));
            float2 v__1 = make_float2((g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
            *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(u_fragment + (i_9 * 2)) = __1;
        }
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          *(uint1*)(v_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_10 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576));
          float2 __2;
            float2 v__2 = *(float2*)(u_fragment + (i_10 * 2));
            float2 __3;
            uint1 v__3 = *(uint1*)(v_shared_local_cast + 0);
            ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
            *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(v__2.x)), *(float2*)(&(__3.x)));
          *(float2*)(u_fragment + (i_10 * 2)) = __2;
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_11 = 0; i_11 < 8; ++i_11) {
          uint1 __4;
          float2 v__4 = *(float2*)(u_fragment + (i_11 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
          *(uint1*)(v_shared_local_cast_1 + 0) = __4;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_11 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_11 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576)) = *(uint1*)(v_shared_local_cast_1 + 0);
        }
        bar_3[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_2[8];
          bfloat16_t B_local_2[16];
          #pragma unroll
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(v_fragment + (i_12 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
          }
          tl::__sync_thread_partial<4, 128>();
          for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 28672)])), (&(A_local_2[0])));
            for (int i_13 = 0; i_13 < 2; ++i_13) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_s_1 & 1) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])), (&(B_local_2[(i_13 * 8)])));
            }
            for (int j_2 = 0; j_2 < 2; ++j_2) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 32768)])), __pack_half2(((bfloat16_t)v_fragment[(i_14 * 8)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 7)])));
        }
        bar_4[0].arrive();
        #pragma unroll
        for (int i_15 = 0; i_15 < 8; ++i_15) {
          float2 __5;
            float2 v__5 = *(float2*)(v_fragment + (i_15 * 2));
            float2 v__6 = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
          *(float2*)(v_fragment + (i_15 * 2)) = __5;
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 34816)])), __pack_half2(((bfloat16_t)v_fragment[(i_16 * 8)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 7)])));
        }
        bar_5[0].arrive();
        bar_5[0].wait((i_s_1 & 1));
        data_is_free[(i_s_1 & 1)].arrive();
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<128>();
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 & 1)].wait(((i_s_2 & 3) >> 1));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_3[8];
            bfloat16_t B_local_3[8];
            #pragma unroll
            for (int i_17 = 0; i_17 < 2; ++i_17) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(p_fragment + (i_17 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
            }
            for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_3[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_3[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 4), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 4));
            }
          }
          #pragma unroll
          for (int i_18 = 0; i_18 < 4; ++i_18) {
            float2 __6;
              float2 v__7 = make_float2(g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              float2 v__8 = *(float2*)(g_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
            *(float2*)(g_fragment + (i_18 * 2)) = __6;
          }
          #pragma unroll
          for (int i_19 = 0; i_19 < 8; ++i_19) {
            if ((((((((int)threadIdx.x) >> 6) * 16) + ((i_19 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_19 & 1)) <= ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_19 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64)) {
              g_fragment[i_19] = exp2f((g_fragment[i_19] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
            } else {
              g_fragment[i_19] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_20 = 0; i_20 < 4; ++i_20) {
            *(uint1*)(a_shared_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_20 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_20 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 28672));
            float2 __7;
            uint1 v__9 = *(uint1*)(a_shared_local_cast_2 + 0);
            ((float2*)(&__7))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__9))[0]);
            *(float2*)(a_fragment + (i_20 * 2)) = __7;
          }
          #pragma unroll
          for (int i_21 = 0; i_21 < 8; ++i_21) {
            a_fragment[i_21] = (a_fragment[i_21] * g_fragment[i_21]);
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 4; ++i_22) {
            float2 __8;
              float2 v__10 = *(float2*)(a_fragment + (i_22 * 2));
              float2 v__11 = *(float2*)(b_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_22 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)));
            *(float2*)(a_fragment + (i_22 * 2)) = __8;
          }
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_23 = 0; i_23 < 4; ++i_23) {
            uint1 __9;
            float2 v__12 = *(float2*)(a_fragment + (i_23 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__12))[0]);
            *(uint1*)(a_shared_local_cast_3 + 0) = __9;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_23 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_23 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 28672)) = *(uint1*)(a_shared_local_cast_3 + 0);
          }
          bar_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_4[8];
            bfloat16_t B_local_4[16];
            #pragma unroll
            for (int i_24 = 0; i_24 < 4; ++i_24) {
              float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(o_fragment + (i_24 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            }
            tl::__sync_thread_partial<5, 128>();
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_4 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_4[0])));
              for (int i_25 = 0; i_25 < 2; ++i_25) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((ki_4 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_25) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_25 * 8)])));
              }
              for (int j_3 = 0; j_3 < 2; ++j_3) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_3 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_3 * 8) + 4)));
              }
            }
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 8; ++i_26) {
            p_fragment[i_26] = (p_fragment[i_26] * (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_fragment[i_26]));
          }
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((((((((int)threadIdx.x) >> 7) * 32) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 8)) - 64) >> 8) * 256)) + (((((((((int)threadIdx.x) >> 7) * 32) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 8)) + 192) & 255)) + 36864)])), __pack_half2(((bfloat16_t)p_fragment[0]), ((bfloat16_t)p_fragment[1])), __pack_half2(((bfloat16_t)p_fragment[2]), ((bfloat16_t)p_fragment[3])), __pack_half2(((bfloat16_t)p_fragment[4]), ((bfloat16_t)p_fragment[5])), __pack_half2(((bfloat16_t)p_fragment[6]), ((bfloat16_t)p_fragment[7])));
          bar_3[0].arrive();
          #pragma unroll
          for (int i_27 = 0; i_27 < 8; ++i_27) {
            float2 __10;
              float2 v__13 = *(float2*)(o_fragment + (i_27 * 2));
              float2 v__14 = make_float2((0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]), (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
              *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
            *(float2*)(o_fragment + (i_27 * 2)) = __10;
          }
          bar_4[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_5[8];
            bfloat16_t B_local_5[16];
            tl::__sync_thread_partial<5, 128>();
            for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_5) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 36864)])), (&(A_local_5[0])));
              for (int i_28 = 0; i_28 < 2; ++i_28) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki_5 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_28) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_5[(i_28 * 8)])));
              }
              for (int j_4 = 0; j_4 < 2; ++j_4) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_4 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_4 * 8) + 4)));
              }
            }
          }
          bar_5[0].arrive();
          bar_5[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_29 = 0; i_29 < 2; ++i_29) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_29 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) - 128) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_29 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) + 384) & 511)) + 30720)])), __pack_half2(((bfloat16_t)o_fragment[(i_29 * 8)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 1)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 2)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 3)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 4)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 5)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 6)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 7)])));
          }
          data_is_free[(i_s_2 & 1)].arrive();
        }
        bar_o[0].arrive();
      } else {
        tl::warpgroup_reg_dealloc<32>();
        if (((int)threadIdx.x) < 416) {
          tl::__sync_thread_partial<6, 32>();
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 & 1)].wait((((i_s_3 >> 1) + 1) & 1));
            int left = ((i_s_3 * 32) + seq_start_idx);
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 16384)])), 0, ((((int)blockIdx.x) & 63) >> 2), left, batch_idx);
                tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 18432)])), 64, ((((int)blockIdx.x) & 63) >> 2), left, batch_idx);
              }
            } else {
              tl::__sync_thread_partial<6, 32>();
              #pragma unroll
              for (int i_30 = 0; i_30 < 16; ++i_30) {
                if ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_1;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_30)) && ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_1 = *(uint4*)(q + (((((((((int64_t)i_30) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)left) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)196608));
                  } else {
                    condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
                  }
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_30 >> 2) * 512)) + ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_30) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_30) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = condval_1;
                } else {
                  bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_30 >> 2) * 512)) + ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_30) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_30) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
                }
              }
            }
            tl::__sync_thread_partial<6, 32>();
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 8192)])), 0, ((((int)blockIdx.x) & 63) >> 2), left, batch_idx);
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 10240)])), 64, ((((int)blockIdx.x) & 63) >> 2), left, batch_idx);
              }
            } else {
              tl::__sync_thread_partial<6, 32>();
              #pragma unroll
              for (int i_31 = 0; i_31 < 16; ++i_31) {
                if ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_2;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_31)) && ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_2 = *(uint4*)(k + (((((((((int64_t)i_31) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)left) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)196608));
                  } else {
                    condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
                  }
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_31 >> 2) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_31) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_31) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = condval_2;
                } else {
                  bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_31 >> 2) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_31) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_31) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
                }
              }
            }
            data_is_ready[(i_s_3 & 1)].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::__sync_thread_partial<7, 32>();
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              data_is_free[(i_s_4 & 1)].wait((((i_s_4 >> 1) + 1) & 1));
              int left_1 = ((i_s_4 * 32) + seq_start_idx);
              if ((left_1 + 32) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 & 1)].expect_transaction(4096);
                  tl::fence_proxy_async();
                  tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_4 & 1) * 2048) + 24576)])), ((((int)blockIdx.x) & 1) * 64), ((((int)blockIdx.x) & 63) >> 1), left_1, batch_idx);
                }
              } else {
                tl::__sync_thread_partial<7, 32>();
                #pragma unroll
                for (int i_32 = 0; i_32 < 8; ++i_32) {
                  if ((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (seq_end_idx + 52)) {
                    bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_3;
                    if (((((13 <= ((((((int)threadIdx.x) >> 3) + left_1) >> 2) + i_32)) && ((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (num_tokens + 52))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_3 = *(uint4*)(v + (((((((((int64_t)i_32) * (int64_t)32768) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)8192)) + (((int64_t)left_1) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)425984));
                    } else {
                      condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
                    }
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_s_4 & 1) * 2048) + ((i_32 >> 1) * 512)) + (((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_32) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = condval_3;
                  } else {
                    bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_s_4 & 1) * 2048) + ((i_32 >> 1) * 512)) + (((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_32) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
                  }
                }
              }
              if ((left_1 + 32) <= seq_end_idx) {
                float condval_4;
                if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                  condval_4 = b[(((((((int64_t)left_1) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)13312)];
                } else {
                  condval_4 = 0x0p+0f/*0.000000e+00*/;
                }
                b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_4;
              } else {
                if ((left_1 + ((int)threadIdx.x)) < (seq_end_idx + 416)) {
                  float condval_5;
                  if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_5 = b[(((((((int64_t)left_1) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)13312)];
                  } else {
                    condval_5 = 0x0p+0f/*0.000000e+00*/;
                  }
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_5;
                } else {
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = 0x0p+0f/*0.000000e+00*/;
                }
              }
              data_is_ready[(i_s_4 & 1)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              tl::__sync_thread_partial<8, 32>();
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 & 1)].wait((((i_s_5 >> 1) + 1) & 1));
                int left_2 = ((i_s_5 * 32) + seq_start_idx);
                if ((left_2 + 32) <= seq_end_idx) {
                  if (tl::tl_shuffle_elect<32>()) {
                    data_is_ready[(i_s_5 & 1)].expect_transaction(2048);
                    tl::fence_proxy_async();
                    tl::tma_load(a_desc, data_is_ready[(i_s_5 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_5 & 1) * 1024) + 28672)])), 0, ((((int)blockIdx.x) & 63) >> 1), left_2, batch_idx);
                  }
                } else {
                  #pragma unroll
                  for (int i_33 = 0; i_33 < 4; ++i_33) {
                    if ((((i_33 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (seq_end_idx + 112)) {
                      bfloat16_t broadcast_var_10 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      uint4 condval_6;
                      if (((((14 <= ((((((int)threadIdx.x) >> 2) + left_2) >> 3) + i_33)) && ((((i_33 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (num_tokens + 112))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_6 = *(uint4*)(a + (((((((((int64_t)i_33) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)1024)) + (((int64_t)left_2) * (int64_t)1024)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) - (int64_t)114688));
                      } else {
                        condval_6 = make_uint4(__pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10));
                      }
                      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_5 & 1) * 1024) + (i_33 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 25088)) = condval_6;
                    } else {
                      bfloat16_t broadcast_var_11 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_5 & 1) * 1024) + (i_33 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 25088)) = make_uint4(__pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11));
                    }
                  }
                }
                if ((left_2 + 32) <= seq_end_idx) {
                  float condval_7;
                  if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_7 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)14336)];
                  } else {
                    condval_7 = 0x0p+0f/*0.000000e+00*/;
                  }
                  g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_7;
                } else {
                  if ((left_2 + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                    float condval_8;
                    if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_8 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)14336)];
                    } else {
                      condval_8 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_8;
                  } else {
                    float condval_9;
                    if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_9 = g[((((((int64_t)seq_end_idx) * (int64_t)32) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) >> (int64_t)1)) - (int64_t)32)];
                    } else {
                      condval_9 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_9;
                  }
                }
                data_is_ready[(i_s_5 & 1)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_unmasked_iters; ++i_s_6) {
                int right = ((i_s_6 * 32) + seq_start_idx);
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                if (0 < i_s_6) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)buf_dyn_shmem)[30720])), ((((int)blockIdx.x) & 1) * 64), ((((int)blockIdx.x) & 63) >> 1), (right - 32), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((i_s_6 & 1));
              }
              if (num_unmasked_iters < num_iters) {
                seq_split_idx = ((num_unmasked_iters * 32) + seq_start_idx);
                chunk_split_idx = (chunk_start_idx + num_unmasked_iters);
                int right_1 = seq_split_idx;
                bar_0[0].arrive();
                bar_0[0].wait((num_unmasked_iters & 1));
                if (0 < num_unmasked_iters) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)buf_dyn_shmem)[30720])), ((((int)blockIdx.x) & 1) * 64), ((((int)blockIdx.x) & 63) >> 1), (right_1 - 32), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((num_unmasked_iters & 1));
              }
              seq_split_idx = (((num_iters * 32) + seq_start_idx) - 32);
              bar_o[0].wait(0);
              if (0 < num_iters) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_34 = 0; i_34 < 8; ++i_34) {
                    if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (seq_end_idx + 60)) {
                      if (15 <= ((((((int)threadIdx.x) >> 3) + seq_split_idx) >> 2) + i_34)) {
                        if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60)) {
                          if (batch_idx < 1) {
                            *(uint4*)(o + (((((((((int64_t)i_34) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)seq_split_idx) * (int64_t)4096)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)245760)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_34 >> 1) * 512) + (((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_34) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 30720));
                          }
                        }
                      }
                    } else {
                      if (((((int)blockIdx.x) >> 6) == (batch_size - 1)) && ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60))) {
                        if (15 <= ((((((int)threadIdx.x) >> 3) + seq_split_idx) >> 2) + i_34)) {
                          if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60)) {
                            if (batch_idx < 1) {
                              bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                              *(uint4*)(o + (((((((((int64_t)i_34) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)seq_split_idx) * (int64_t)4096)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)63) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)245760)) = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// ---- flashqla_fused_cp_packed_strided_qkg_pair ----
// qkg_pair fused candidate (NCU-targeted L2 improvement): blockIdx.y = CP
// segment; blockIdx.x groups the 4 blocks sharing one Q/K head (2 V heads x
// 2 DV tiles) contiguously: qk_head = x/4, v_head = qk_head*2 + (x%4)/2,
// dv_tile = x%2 (block_DV=64).  Improves Q/K L2 reuse vs the baseline fused
// (native_source fused L2 27% vs native ref 50%; this candidate is ~9-12%
// faster at T=2048/4096/8192 with numeric parity <= 1e-2).
// Source SHA-256 (generated device.cu): see result/native_source_ncu.

#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void flashqla_fused_cp_packed_strided_qkg_pair(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cp_seq_map, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const int64_t* __restrict__ raw_cu_seqlens, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size);
extern "C" __global__ void __launch_bounds__(512, 1) flashqla_fused_cp_packed_strided_qkg_pair(const bfloat16_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const float* __restrict__ b, const int64_t* __restrict__ chunk_offsets, const int64_t* __restrict__ cp_seq_map, const int64_t* __restrict__ cu_seqlens, const float* __restrict__ g, const float* __restrict__ h0, __grid_constant__ const CUtensorMap h_desc, float* __restrict__ ht, const bfloat16_t* __restrict__ k, __grid_constant__ const CUtensorMap k_desc, bfloat16_t* __restrict__ o, __grid_constant__ const CUtensorMap o_desc, const bfloat16_t* __restrict__ q, __grid_constant__ const CUtensorMap q_desc, const int64_t* __restrict__ raw_cu_seqlens, const bfloat16_t* __restrict__ v, __grid_constant__ const CUtensorMap v_desc, int batch_size, int num_tokens, int raw_batch_size) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t data_is_ready_mem[2];
  auto data_is_ready = reinterpret_cast<Barrier*>(data_is_ready_mem);
  __shared__ __align__(16) uint64_t data_is_free_mem[2];
  auto data_is_free = reinterpret_cast<Barrier*>(data_is_free_mem);
  __shared__ __align__(16) uint64_t bar_o_mem[1];
  auto bar_o = reinterpret_cast<Barrier*>(bar_o_mem);
  __shared__ __align__(16) uint64_t bar_0_mem[1];
  auto bar_0 = reinterpret_cast<Barrier*>(bar_0_mem);
  __shared__ __align__(16) uint64_t bar_1_mem[1];
  auto bar_1 = reinterpret_cast<Barrier*>(bar_1_mem);
  __shared__ __align__(16) uint64_t _bar_2_mem[1];
  auto _bar_2 = reinterpret_cast<Barrier*>(_bar_2_mem);
  __shared__ __align__(16) uint64_t bar_3_mem[1];
  auto bar_3 = reinterpret_cast<Barrier*>(bar_3_mem);
  __shared__ __align__(16) uint64_t bar_4_mem[1];
  auto bar_4 = reinterpret_cast<Barrier*>(bar_4_mem);
  __shared__ __align__(16) uint64_t bar_5_mem[1];
  auto bar_5 = reinterpret_cast<Barrier*>(bar_5_mem);
  int batch_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  int chunk_start_idx = 0;
  int raw_batch_idx = 0;
  int raw_seq_end_idx = 0;
  signed char need_store_final_state = (signed char)0;
  int num_iters = 0;
  int num_unmasked_iters = 0;
  float h_fragment[64];
  __shared__ __align__(16) float g_exp_shared[32];
  __shared__ __align__(16) float g_shared[64];
  __shared__ __align__(16) float b_shared[64];
  int seq_split_idx = 0;
  int chunk_split_idx = 0;
  float g_last_local[1];
  __shared__ __align__(16) float g_rev_exp_shared[32];
  float u_fragment[16];
  bfloat16_t v_shared_local_cast[2];
  bfloat16_t v_shared_local_cast_1[2];
  float v_fragment[16];
  float p_fragment[8];
  float g_fragment[8];
  float a_fragment[8];
  bfloat16_t a_shared_local_cast_2[2];
  bfloat16_t a_shared_local_cast_3[2];
  float o_fragment[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(o_desc);
    tl::prefetch_tma_descriptor(h_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    data_is_ready[0].init(96);
    data_is_ready[1].init(96);
    data_is_free[0].init(384);
    data_is_free[1].init(384);
    bar_o[0].init(128);
    bar_0[0].init(416);
    bar_1[0].init(256);
    _bar_2[0].init(128);
    bar_3[0].init(128);
    bar_4[0].init(128);
    bar_5[0].init(416);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = 0;
  seq_start_idx = ((int)cu_seqlens[((int64_t)((int)blockIdx.y))]);
  seq_end_idx = ((int)cu_seqlens[(((int64_t)((int)blockIdx.y)) + (int64_t)1)]);
  chunk_start_idx = ((int)chunk_offsets[((int64_t)((int)blockIdx.y))]);
  raw_batch_idx = ((int)cp_seq_map[((int64_t)((int)blockIdx.y))]);
  int64_t condval;
  if (((-1 <= raw_batch_idx) && (raw_batch_idx < raw_batch_size))) {
    condval = raw_cu_seqlens[(((int64_t)raw_batch_idx) + (int64_t)1)];
  } else {
    condval = (int64_t)0;
  }
  raw_seq_end_idx = ((int)condval);
  need_store_final_state = ((signed char)((bool)1 & (raw_seq_end_idx == seq_end_idx)));
  num_iters = (((seq_end_idx + 31) - seq_start_idx) >> 5);
  num_unmasked_iters = ((seq_end_idx - seq_start_idx) >> 5);
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<160>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      *(float2*)(h_fragment + (i * 2)) = *(float2*)(h0 + ((((((((((((int64_t)((int)blockIdx.y)) * (int64_t)524288) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    }
    for (int i_s = 0; i_s < num_iters; ++i_s) {
      data_is_ready[(i_s & 1)].wait(((i_s & 3) >> 1));
      bar_0[0].arrive();
      bar_0[0].wait((i_s & 1));
      tl::__sync_thread_partial<3, 128>();
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_1 >> 1) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)h_fragment[(i_1 * 8)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 2)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 4)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)h_fragment[((i_1 * 8) + 6)]), ((bfloat16_t)h_fragment[((i_1 * 8) + 7)])));
      }
      bar_1[0].arrive();
      bar_1[0].wait((i_s & 1));
      g_last_local[0] = g_exp_shared[31];
      #pragma unroll
      for (int i_2 = 0; i_2 < 64; ++i_2) {
        h_fragment[i_2] = (h_fragment[i_2] * g_last_local[0]);
      }
      bar_5[0].arrive();
      bar_5[0].wait((i_s & 1));
      {
        bfloat16_t A_local[32];
        bfloat16_t B_local[16];
        tl::__sync_thread_partial<3, 128>();
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((i_s & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A_local[(i_3 * 8)])));
          }
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 34816)])), (&(B_local[(i_4 * 8)])));
          }
          for (int i_5 = 0; i_5 < 4; ++i_5) {
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + ((i_5 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(h_fragment + (((i_5 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_5 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
      }
      data_is_free[(i_s & 1)].arrive();
    }
    if ((bool)need_store_final_state) {
      if (0 <= raw_batch_idx) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          if (raw_batch_idx < raw_batch_size) {
            *(float2*)(ht + ((((((((((((int64_t)raw_batch_idx) * (int64_t)524288) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)8192)) + ((((int64_t)i_6) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((int64_t)i_6) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_fragment + (i_6 * 2));
          }
        }
      }
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<128>();
      for (int i_s_1 = 0; i_s_1 < num_iters; ++i_s_1) {
        data_is_ready[(i_s_1 & 1)].wait(((i_s_1 & 3) >> 1));
        bar_0[0].arrive();
        bar_0[0].wait((i_s_1 & 1));
        tl::__sync_thread_partial<3, 128>();
        if (((int)threadIdx.x) < 160) {
          g_exp_shared[(((int)threadIdx.x) - 128)] = exp2f((g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          float condval_1;
          if (((((i_s_1 * 32) + seq_start_idx) + ((int)threadIdx.x)) < (seq_end_idx + 128))) {
            condval_1 = exp2f(((g_shared[(((i_s_1 & 1) * 32) + 31)] - g_shared[((((i_s_1 & 1) * 32) + ((int)threadIdx.x)) - 128)]) * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
          } else {
            condval_1 = 0x0p+0f/*0.000000e+00*/;
          }
          g_rev_exp_shared[(((int)threadIdx.x) - 128)] = condval_1;
        }
        bar_1[0].arrive();
        bar_1[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_1[8];
          bfloat16_t B_local_1[16];
          #pragma unroll
          for (int i_7 = 0; i_7 < 4; ++i_7) {
            float broadcast_var = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(u_fragment + (i_7 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          }
          for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 4096) + ((ki_1 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(A_local_1[0])));
            for (int i_8 = 0; i_8 < 2; ++i_8) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((ki_1 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_8) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_8 * 8)])));
            }
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_fragment + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial<3, 128>();
        #pragma unroll
        for (int i_9 = 0; i_9 < 8; ++i_9) {
          float2 __1;
            float2 v_ = *(float2*)(u_fragment + (i_9 * 2));
            float2 v__1 = make_float2((g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/), (g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] * -0x1p+0f/*-1.000000e+00*/));
            *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
          *(float2*)(u_fragment + (i_9 * 2)) = __1;
        }
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          *(uint1*)(v_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_10 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576));
          float2 __2;
            float2 v__2 = *(float2*)(u_fragment + (i_10 * 2));
            float2 __3;
            uint1 v__3 = *(uint1*)(v_shared_local_cast + 0);
            ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
            *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(v__2.x)), *(float2*)(&(__3.x)));
          *(float2*)(u_fragment + (i_10 * 2)) = __2;
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_11 = 0; i_11 < 8; ++i_11) {
          uint1 __4;
          float2 v__4 = *(float2*)(u_fragment + (i_11 * 2));
          (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
          *(uint1*)(v_shared_local_cast_1 + 0) = __4;
          *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((i_s_1 & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_11 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_11 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24576)) = *(uint1*)(v_shared_local_cast_1 + 0);
        }
        bar_3[0].wait((i_s_1 & 1));
        {
          bfloat16_t A_local_2[8];
          bfloat16_t B_local_2[16];
          #pragma unroll
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(v_fragment + (i_12 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
          }
          tl::__sync_thread_partial<4, 128>();
          for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_1 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 28672)])), (&(A_local_2[0])));
            for (int i_13 = 0; i_13 < 2; ++i_13) {
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_s_1 & 1) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])), (&(B_local_2[(i_13 * 8)])));
            }
            for (int j_2 = 0; j_2 < 2; ++j_2) {
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(v_fragment + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
            }
          }
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_14 = 0; i_14 < 2; ++i_14) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_14) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 32768)])), __pack_half2(((bfloat16_t)v_fragment[(i_14 * 8)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_14 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_14 * 8) + 7)])));
        }
        bar_4[0].arrive();
        #pragma unroll
        for (int i_15 = 0; i_15 < 8; ++i_15) {
          float2 __5;
            float2 v__5 = *(float2*)(v_fragment + (i_15 * 2));
            float2 v__6 = make_float2(g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_rev_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
            *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)));
          *(float2*)(v_fragment + (i_15 * 2)) = __5;
        }
        tl::__sync_thread_partial<4, 128>();
        #pragma unroll
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) - 64) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 7) & 7) >> 2)) & 1) * 32)) + ((((((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 3) & 3) >> 1) + i_16) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 1) & 1)) & 1) * 8)) + 448) & 511)) + 34816)])), __pack_half2(((bfloat16_t)v_fragment[(i_16 * 8)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 1)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 2)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 3)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 4)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 5)])), __pack_half2(((bfloat16_t)v_fragment[((i_16 * 8) + 6)]), ((bfloat16_t)v_fragment[((i_16 * 8) + 7)])));
        }
        bar_5[0].arrive();
        bar_5[0].wait((i_s_1 & 1));
        data_is_free[(i_s_1 & 1)].arrive();
      }
    } else {
      if (((int)threadIdx.x) < 384) {
        tl::warpgroup_reg_alloc<128>();
        for (int i_s_2 = 0; i_s_2 < num_iters; ++i_s_2) {
          data_is_ready[(i_s_2 & 1)].wait(((i_s_2 & 3) >> 1));
          bar_0[0].arrive();
          bar_0[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_3[8];
            bfloat16_t B_local_3[8];
            #pragma unroll
            for (int i_17 = 0; i_17 < 2; ++i_17) {
              float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(p_fragment + (i_17 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
            }
            for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_3[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((i_s_2 & 1) * 4096) + ((ki_3 >> 2) * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_3[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(p_fragment + 4), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + 4));
            }
          }
          #pragma unroll
          for (int i_18 = 0; i_18 < 4; ++i_18) {
            float2 __6;
              float2 v__7 = make_float2(g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], g_shared[(((((i_s_2 & 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
              float2 v__8 = *(float2*)(g_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(v__7.x)), *(float2*)(&(v__8.x)));
            *(float2*)(g_fragment + (i_18 * 2)) = __6;
          }
          #pragma unroll
          for (int i_19 = 0; i_19 < 8; ++i_19) {
            if ((((((((int)threadIdx.x) >> 6) * 16) + ((i_19 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_19 & 1)) <= ((((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_19 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 64)) {
              g_fragment[i_19] = exp2f((g_fragment[i_19] * 0x1.715475a31a4bep+0f/*1.442695e+00*/));
            } else {
              g_fragment[i_19] = 0x0p+0f/*0.000000e+00*/;
            }
          }
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_20 = 0; i_20 < 4; ++i_20) {
            *(uint1*)(a_shared_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_20 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_20 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 28672));
            float2 __7;
            uint1 v__9 = *(uint1*)(a_shared_local_cast_2 + 0);
            ((float2*)(&__7))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__9))[0]);
            *(float2*)(a_fragment + (i_20 * 2)) = __7;
          }
          #pragma unroll
          for (int i_21 = 0; i_21 < 8; ++i_21) {
            a_fragment[i_21] = (a_fragment[i_21] * g_fragment[i_21]);
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 4; ++i_22) {
            float2 __8;
              float2 v__10 = *(float2*)(a_fragment + (i_22 * 2));
              float2 v__11 = *(float2*)(b_shared + ((((((i_s_2 & 1) * 32) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_22 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 64));
              *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)));
            *(float2*)(a_fragment + (i_22 * 2)) = __8;
          }
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_23 = 0; i_23 < 4; ++i_23) {
            uint1 __9;
            float2 v__12 = *(float2*)(a_fragment + (i_23 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__12))[0]);
            *(uint1*)(a_shared_local_cast_3 + 0) = __9;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_2 & 1) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_23 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_23 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 28672)) = *(uint1*)(a_shared_local_cast_3 + 0);
          }
          bar_1[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_4[8];
            bfloat16_t B_local_4[16];
            #pragma unroll
            for (int i_24 = 0; i_24 < 4; ++i_24) {
              float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(o_fragment + (i_24 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            }
            tl::__sync_thread_partial<5, 128>();
            for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_s_2 & 1) * 4096) + ((ki_4 >> 2) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_4[0])));
              for (int i_25 = 0; i_25 < 2; ++i_25) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((ki_4 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_25) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_25 * 8)])));
              }
              for (int j_3 = 0; j_3 < 2; ++j_3) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_3 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_3 * 8) + 4)));
              }
            }
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 8; ++i_26) {
            p_fragment[i_26] = (p_fragment[i_26] * (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_fragment[i_26]));
          }
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((((((((int)threadIdx.x) >> 7) * 32) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 8)) - 64) >> 8) * 256)) + (((((((((int)threadIdx.x) >> 7) * 32) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 8)) + 192) & 255)) + 36864)])), __pack_half2(((bfloat16_t)p_fragment[0]), ((bfloat16_t)p_fragment[1])), __pack_half2(((bfloat16_t)p_fragment[2]), ((bfloat16_t)p_fragment[3])), __pack_half2(((bfloat16_t)p_fragment[4]), ((bfloat16_t)p_fragment[5])), __pack_half2(((bfloat16_t)p_fragment[6]), ((bfloat16_t)p_fragment[7])));
          bar_3[0].arrive();
          #pragma unroll
          for (int i_27 = 0; i_27 < 8; ++i_27) {
            float2 __10;
              float2 v__13 = *(float2*)(o_fragment + (i_27 * 2));
              float2 v__14 = make_float2((0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]), (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * g_exp_shared[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
              *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
            *(float2*)(o_fragment + (i_27 * 2)) = __10;
          }
          bar_4[0].wait((i_s_2 & 1));
          {
            bfloat16_t A_local_5[8];
            bfloat16_t B_local_5[16];
            tl::__sync_thread_partial<5, 128>();
            for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_5) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 36864)])), (&(A_local_5[0])));
              for (int i_28 = 0; i_28 < 2; ++i_28) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((ki_5 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_28) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 32768)])), (&(B_local_5[(i_28 * 8)])));
              }
              for (int j_4 = 0; j_4 < 2; ++j_4) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + (j_4 * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(o_fragment + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_5 + ((j_4 * 8) + 4)));
              }
            }
          }
          bar_5[0].arrive();
          bar_5[0].wait((i_s_2 & 1));
          tl::__sync_thread_partial<5, 128>();
          #pragma unroll
          for (int i_29 = 0; i_29 < 2; ++i_29) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_29 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) - 128) >> 9) * 512)) + ((((((((((int)threadIdx.x) >> 7) * 64) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) + 6) & 7) >> 2)) & 1) * 32)) + (((i_29 + (((((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) >> 1) + 1) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((((int)threadIdx.x) >> 7) + (((int)threadIdx.x) & 7)) & 1)) & 1) * 8)) + 384) & 511)) + 30720)])), __pack_half2(((bfloat16_t)o_fragment[(i_29 * 8)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 1)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 2)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 3)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 4)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 5)])), __pack_half2(((bfloat16_t)o_fragment[((i_29 * 8) + 6)]), ((bfloat16_t)o_fragment[((i_29 * 8) + 7)])));
          }
          data_is_free[(i_s_2 & 1)].arrive();
        }
        bar_o[0].arrive();
      } else {
        tl::warpgroup_reg_dealloc<32>();
        if (((int)threadIdx.x) < 416) {
          tl::__sync_thread_partial<6, 32>();
          for (int i_s_3 = 0; i_s_3 < num_iters; ++i_s_3) {
            data_is_free[(i_s_3 & 1)].wait((((i_s_3 >> 1) + 1) & 1));
            int left = ((i_s_3 * 32) + seq_start_idx);
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 16384)])), 0, (((int)blockIdx.x) >> 2), left, batch_idx);
                tl::tma_load(q_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 18432)])), 64, (((int)blockIdx.x) >> 2), left, batch_idx);
              }
            } else {
              tl::__sync_thread_partial<6, 32>();
              #pragma unroll
              for (int i_30 = 0; i_30 < 16; ++i_30) {
                if ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_2;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_30)) && ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_2 = *(uint4*)(q + (((((((((int64_t)i_30) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)left) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)196608));
                  } else {
                    condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
                  }
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_30 >> 2) * 512)) + ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_30) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_30) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = condval_2;
                } else {
                  bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_30 >> 2) * 512)) + ((((i_30 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_30) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_30) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
                }
              }
            }
            tl::__sync_thread_partial<6, 32>();
            if ((left + 32) <= seq_end_idx) {
              if (tl::tl_shuffle_elect<32>()) {
                data_is_ready[(i_s_3 & 1)].expect_transaction(8192);
                tl::fence_proxy_async();
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 8192)])), 0, (((int)blockIdx.x) >> 2), left, batch_idx);
                tl::tma_load(k_desc, data_is_ready[(i_s_3 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_3 & 1) * 4096) + 10240)])), 64, (((int)blockIdx.x) >> 2), left, batch_idx);
              }
            } else {
              tl::__sync_thread_partial<6, 32>();
              #pragma unroll
              for (int i_31 = 0; i_31 < 16; ++i_31) {
                if ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (seq_end_idx + 24)) {
                  bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  uint4 condval_3;
                  if (((((12 <= ((((((int)threadIdx.x) >> 4) + left) >> 1) + i_31)) && ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) + left) < (num_tokens + 24))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_3 = *(uint4*)(k + (((((((((int64_t)i_31) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)8192)) + (((int64_t)left) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)2) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) - (int64_t)196608));
                  } else {
                    condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
                  }
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_31 >> 2) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_31) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_31) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = condval_3;
                } else {
                  bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_s_3 & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + ((i_31 >> 2) * 512)) + ((((i_31 * 2) + (((int)threadIdx.x) >> 4)) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_31) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (((((int)threadIdx.x) >> 5) + i_31) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
                }
              }
            }
            data_is_ready[(i_s_3 & 1)].arrive();
          }
        } else {
          if (((int)threadIdx.x) < 448) {
            tl::__sync_thread_partial<7, 32>();
            for (int i_s_4 = 0; i_s_4 < num_iters; ++i_s_4) {
              data_is_free[(i_s_4 & 1)].wait((((i_s_4 >> 1) + 1) & 1));
              int left_1 = ((i_s_4 * 32) + seq_start_idx);
              if ((left_1 + 32) <= seq_end_idx) {
                if (tl::tl_shuffle_elect<32>()) {
                  data_is_ready[(i_s_4 & 1)].expect_transaction(4096);
                  tl::fence_proxy_async();
                  tl::tma_load(v_desc, data_is_ready[(i_s_4 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_4 & 1) * 2048) + 24576)])), ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), left_1, batch_idx);
                }
              } else {
                #pragma unroll
                for (int i_32 = 0; i_32 < 8; ++i_32) {
                  if ((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (seq_end_idx + 52)) {
                    bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    uint4 condval_4;
                    if (((((13 <= ((((((int)threadIdx.x) >> 3) + left_1) >> 2) + i_32)) && ((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + left_1) < (num_tokens + 52))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_4 = *(uint4*)(v + (((((((((int64_t)i_32) * (int64_t)32768) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)8192)) + (((int64_t)left_1) * (int64_t)8192)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)8192)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)425984));
                    } else {
                      condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
                    }
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_s_4 & 1) * 2048) + ((i_32 >> 1) * 512)) + (((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_32) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = condval_4;
                  } else {
                    bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_s_4 & 1) * 2048) + ((i_32 >> 1) * 512)) + (((((i_32 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_32) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)) = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
                  }
                }
              }
              if ((left_1 + 32) <= seq_end_idx) {
                float condval_5;
                if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                  condval_5 = b[(((((((int64_t)left_1) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) >> (int64_t)1)) - (int64_t)13312)];
                } else {
                  condval_5 = 0x0p+0f/*0.000000e+00*/;
                }
                b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_5;
              } else {
                if ((left_1 + ((int)threadIdx.x)) < (seq_end_idx + 416)) {
                  float condval_6;
                  if (((((416 <= (left_1 + ((int)threadIdx.x))) && ((left_1 + ((int)threadIdx.x)) < (num_tokens + 416))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_6 = b[(((((((int64_t)left_1) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) >> (int64_t)1)) - (int64_t)13312)];
                  } else {
                    condval_6 = 0x0p+0f/*0.000000e+00*/;
                  }
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = condval_6;
                } else {
                  b_shared[((((i_s_4 & 1) * 32) + ((int)threadIdx.x)) - 416)] = 0x0p+0f/*0.000000e+00*/;
                }
              }
              data_is_ready[(i_s_4 & 1)].arrive();
            }
          } else {
            if (((int)threadIdx.x) < 480) {
              tl::__sync_thread_partial<8, 32>();
              for (int i_s_5 = 0; i_s_5 < num_iters; ++i_s_5) {
                data_is_free[(i_s_5 & 1)].wait((((i_s_5 >> 1) + 1) & 1));
                int left_2 = ((i_s_5 * 32) + seq_start_idx);
                if ((left_2 + 32) <= seq_end_idx) {
                  if (tl::tl_shuffle_elect<32>()) {
                    data_is_ready[(i_s_5 & 1)].expect_transaction(2048);
                    tl::fence_proxy_async();
                    tl::tma_load(a_desc, data_is_ready[(i_s_5 & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((i_s_5 & 1) * 1024) + 28672)])), 0, (((int)blockIdx.x) >> 1), left_2, batch_idx);
                  }
                } else {
                  #pragma unroll
                  for (int i_33 = 0; i_33 < 4; ++i_33) {
                    if ((((i_33 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (seq_end_idx + 112)) {
                      bfloat16_t broadcast_var_10 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      uint4 condval_7;
                      if (((((14 <= ((((((int)threadIdx.x) >> 2) + left_2) >> 3) + i_33)) && ((((i_33 * 8) + (((int)threadIdx.x) >> 2)) + left_2) < (num_tokens + 112))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                        condval_7 = *(uint4*)(a + (((((((((int64_t)i_33) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)1024)) + (((int64_t)left_2) * (int64_t)1024)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) - (int64_t)114688));
                      } else {
                        condval_7 = make_uint4(__pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10), __pack_nv_bfloat162(broadcast_var_10, broadcast_var_10));
                      }
                      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_5 & 1) * 1024) + (i_33 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 25088)) = condval_7;
                    } else {
                      bfloat16_t broadcast_var_11 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_s_5 & 1) * 1024) + (i_33 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 25088)) = make_uint4(__pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11), __pack_nv_bfloat162(broadcast_var_11, broadcast_var_11));
                    }
                  }
                }
                if ((left_2 + 32) <= seq_end_idx) {
                  float condval_8;
                  if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                    condval_8 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) >> (int64_t)1)) - (int64_t)14336)];
                  } else {
                    condval_8 = 0x0p+0f/*0.000000e+00*/;
                  }
                  g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_8;
                } else {
                  if ((left_2 + ((int)threadIdx.x)) < (seq_end_idx + 448)) {
                    float condval_9;
                    if (((((448 <= (left_2 + ((int)threadIdx.x))) && ((left_2 + ((int)threadIdx.x)) < (num_tokens + 448))) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_9 = g[(((((((int64_t)left_2) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)32)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) >> (int64_t)1)) - (int64_t)14336)];
                    } else {
                      condval_9 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_9;
                  } else {
                    float condval_10;
                    if (((((1 <= seq_end_idx) && (seq_end_idx <= num_tokens)) && (0 <= batch_idx)) && (batch_idx < 1))) {
                      condval_10 = g[((((((int64_t)seq_end_idx) * (int64_t)32) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) >> (int64_t)1)) - (int64_t)32)];
                    } else {
                      condval_10 = 0x0p+0f/*0.000000e+00*/;
                    }
                    g_shared[((((i_s_5 & 1) * 32) + ((int)threadIdx.x)) - 448)] = condval_10;
                  }
                }
                data_is_ready[(i_s_5 & 1)].arrive();
              }
            } else {
              for (int i_s_6 = 0; i_s_6 < num_unmasked_iters; ++i_s_6) {
                int right = ((i_s_6 * 32) + seq_start_idx);
                bar_0[0].arrive();
                bar_0[0].wait((i_s_6 & 1));
                if (0 < i_s_6) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)buf_dyn_shmem)[30720])), ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), (right - 32), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((i_s_6 & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  tl::tma_store(h_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), ((((int)blockIdx.x) & 1) * 64), 0, (((int)blockIdx.x) >> 1), (chunk_start_idx + i_s_6), batch_idx);
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0>();
                }
              }
              if (num_unmasked_iters < num_iters) {
                seq_split_idx = ((num_unmasked_iters * 32) + seq_start_idx);
                chunk_split_idx = (chunk_start_idx + num_unmasked_iters);
                int right_1 = seq_split_idx;
                bar_0[0].arrive();
                bar_0[0].wait((num_unmasked_iters & 1));
                if (0 < num_unmasked_iters) {
                  if (tl::tl_shuffle_elect<32>()) {
                    tl::tma_store(o_desc, (&(((bfloat16_t*)buf_dyn_shmem)[30720])), ((((int)blockIdx.x) & 1) * 64), (((int)blockIdx.x) >> 1), (right_1 - 32), batch_idx);
                    tl::tma_store_arrive();
                    tl::tma_store_wait<0>();
                  }
                }
                bar_5[0].arrive();
                bar_1[0].wait((num_unmasked_iters & 1));
                if (tl::tl_shuffle_elect<32>()) {
                  tl::tma_store(h_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), ((((int)blockIdx.x) & 1) * 64), 0, (((int)blockIdx.x) >> 1), chunk_split_idx, batch_idx);
                  tl::tma_store_arrive();
                  tl::tma_store_wait<0>();
                }
              }
              seq_split_idx = (((num_iters * 32) + seq_start_idx) - 32);
              bar_o[0].wait(0);
              if (0 < num_iters) {
                if (0 <= batch_idx) {
                  #pragma unroll
                  for (int i_34 = 0; i_34 < 8; ++i_34) {
                    if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (seq_end_idx + 60)) {
                      if (15 <= ((((((int)threadIdx.x) >> 3) + seq_split_idx) >> 2) + i_34)) {
                        if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60)) {
                          if (batch_idx < 1) {
                            *(uint4*)(o + (((((((((int64_t)i_34) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)seq_split_idx) * (int64_t)4096)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4096)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)245760)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_34 >> 1) * 512) + (((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + 4) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((((int)threadIdx.x) >> 5) + i_34) + 1) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 30720));
                          }
                        }
                      }
                    } else {
                      if ((((int)blockIdx.y) == (batch_size - 1)) && ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60))) {
                        if (15 <= ((((((int)threadIdx.x) >> 3) + seq_split_idx) >> 2) + i_34)) {
                          if ((((i_34 * 4) + (((int)threadIdx.x) >> 3)) + seq_split_idx) < (num_tokens + 60)) {
                            if (batch_idx < 1) {
                              bfloat16_t broadcast_var_12 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
                              *(uint4*)(o + (((((((((int64_t)i_34) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)seq_split_idx) * (int64_t)4096)) + ((((int64_t)batch_idx) * ((int64_t)num_tokens)) * (int64_t)4096)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)245760)) = make_uint4(__pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12), __pack_nv_bfloat162(broadcast_var_12, broadcast_var_12));
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
