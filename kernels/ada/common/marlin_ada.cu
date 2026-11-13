// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08):
// the instantiation set of csrc/quantization/marlin/generate_kernels.py:36-180 (THREAD_CONFIGS,
// THREAD_M_BLOCKS and the per-thread-count filter, the AWQ-INT4 and GPTQ-INT4 entries of
// QUANT_CONFIGS) and the kernel parameter list of kernel.h:9-16, Copyright contributors to
// the vLLM project, Apache-2.0; the shared-memory budget of marlin.cu:96-147
// (get_scales_cache_size, get_kernel_cache_size), Copyright (C) Marlin.2024 Elias Frantar,
// Apache-2.0, adapted from https://github.com/IST-DASLab/marlin.
// Changes (by Atlas contributors), listed in VLLM-MARLIN-ATTRIBUTION.md: one hand-written
// translation unit instead of generated per-type files and a kernel_selector.h; every
// instantiation is exported under an unmangled `extern "C"` name (below), because Atlas binds
// kernels by PTX entry name; the three bool parameters are `int` in the exported ABI; the set
// is pruned to BF16 activations, GPTQ int4 (kU4B8) and AWQ int4 with zero points (kU4), group
// size 128, 4 stages; the shared-memory budget is a compile-time check here, not a launch-time
// choice; no torch.

// Ada (sm_89) Marlin W4A16 GEMM entry points (port sprint lane L18a; host side L18b,
// crates/spark-model/src/port/marlin/). Nothing in Atlas looks these up unless the
// `marlin_w4a16` row (off on every target) routes a layer to Marlin; the lookups are lazy
// (`try_target_kernel`), so an Ada build without this module keeps every current path.
//
// ── EXPORTED API (binding for L18b) ────────────────────────────────────────────────────
//
// Module: `marlin_ada` (this file's stem). 30 entry points, all with ONE signature:
//
//   extern "C" __global__ void <name>(
//       const int4* A,               // [m, k] BF16, row stride `lda` elements (lda % 8 == 0)
//       const int4* B,               // Marlin-tiled int4 weights (L18b's repack.rs layout)
//       int4* C,                     // [m, n] BF16 output, row-major, contiguous
//       int4* C_tmp,                 // FP32 reduce scratch (use_fp32_reduce != 0), see below
//       const int4* b_bias_ptr,      // [n] BF16 bias, read only when has_bias != 0
//       const int4* scales_ptr,      // [k / 128, n] BF16 group scales, Marlin-permuted
//       const float* global_scale_ptr,  // unused by these types; pass 0
//       const int4* zp_ptr,          // kU4 only: [k / 128, n / 8] packed int4 zero points,
//                                    // Marlin-permuted; kU4B8: pass 0
//       int prob_m, int prob_n, int prob_k,
//       int lda,                     // A row stride in elements (prob_k when contiguous)
//       int* locks,                  // workspace, >= grid.x ints, ZERO before the first
//                                    // launch; every launch leaves it zero again
//       int has_bias,                // 0 or 1
//       int use_atomic_add,          // 0 or 1 (vLLM: only when ceil(m/64) * n <= 2048)
//       int use_fp32_reduce,         // 0 or 1 (vLLM default 1)
//       int max_shared_mem);         // the dynamic shared-memory size of THIS launch
//
// Launch: grid (sm_count, 1, 1) (vLLM: `sms * blocks_per_sm`, blocks_per_sm = 1), block
// (threads, 1, 1) with `threads` from the name, dynamic shared memory = `max_shared_mem`
// bytes, which must be at least the configuration's cache size (marlin.cu:114-147; checked
// for every entry at compile time below) and at most the device opt-in (101,376 bytes on
// sm_89). Above 48 KiB the Atlas registry raises the function attribute itself
// (avarok_core::registry::launch_on_stream). vLLM passes `max_shared_mem - 512` for a
// single block per SM (marlin.cu:217, 385-396). One launch covers prob_m <= 16 *
// thread_m_blocks * max_par rows in parallel slices; the host splits larger m exactly as
// marlin.cu:296-401 does (`max_thread_m_blocks = 4`, `max_par` 16, or 128 when n <= 4096).
//
// C_tmp (FP32 reduce): sm_count * 64 * max_thread_n (256) floats is enough for every
// configuration here (vLLM sizes it `sms * max_m_block_size * max_thread_n`). Each slice
// that more than one block shares reduces through `locks[locks_off]`, a per-launch counter
// the last block resets to zero.
//
// Shape rules (vLLM's is_valid_config, marlin.cu:149-179): prob_k % thread_k == 0,
// prob_n % thread_n == 0, prob_k % 128 == 0 (one scale group per 128 inputs). No act-order:
// a GPTQ checkpoint with `desc_act` needs its K permutation applied to A and B on the host
// (the template asserts group_blocks != 0).
//
// Names: marlin_ada_bf16_<w>_g128_m<M>_k<thread_k>_n<thread_n>_t<threads>, with
//   <w>  u4b8  GPTQ int4, symmetric, stored value = q + 8   (vLLM kU4B8, has_zp = false)
//        u4    AWQ int4 with int4 zero points              (vLLM kU4,   has_zp = true)
//   <M>  m8  = thread_m_blocks 1 with m_block_size_8 (prob_m <= 8; vLLM's 0.5)
//        m16 / m32 / m48 / m64 = thread_m_blocks 1 / 2 / 3 / 4
//   (thread_k, thread_n, threads) per M, vLLM's small_batch_thread_configs for m8 and m16
//   and large_batch_thread_configs for m32 to m64 (marlin.cu:75-89), in priority order:
//        m8, m16:        (128, 128, 256), (64, 128, 128), (128, 64, 128)
//        m32, m48, m64:  (64, 256, 256),  (64, 128, 128), (128, 64, 128)
//
// thread_m_blocks 3 (m48) is instantiated although the lane brief lists {0.5, 1, 2, 4}:
// marlin.cu:307 picks `min(ceil(m / 16), 4)` blocks, so m in 33..48 asks for 3, and a
// faithful translation of the host logic (L18b) must find it.
//
// The 30 exported names:
//   marlin_ada_bf16_u4b8_g128_m8_k128_n128_t256    marlin_ada_bf16_u4_g128_m8_k128_n128_t256
//   marlin_ada_bf16_u4b8_g128_m8_k64_n128_t128     marlin_ada_bf16_u4_g128_m8_k64_n128_t128
//   marlin_ada_bf16_u4b8_g128_m8_k128_n64_t128     marlin_ada_bf16_u4_g128_m8_k128_n64_t128
//   marlin_ada_bf16_u4b8_g128_m16_k128_n128_t256   marlin_ada_bf16_u4_g128_m16_k128_n128_t256
//   marlin_ada_bf16_u4b8_g128_m16_k64_n128_t128    marlin_ada_bf16_u4_g128_m16_k64_n128_t128
//   marlin_ada_bf16_u4b8_g128_m16_k128_n64_t128    marlin_ada_bf16_u4_g128_m16_k128_n64_t128
//   marlin_ada_bf16_u4b8_g128_m32_k64_n256_t256    marlin_ada_bf16_u4_g128_m32_k64_n256_t256
//   marlin_ada_bf16_u4b8_g128_m32_k64_n128_t128    marlin_ada_bf16_u4_g128_m32_k64_n128_t128
//   marlin_ada_bf16_u4b8_g128_m32_k128_n64_t128    marlin_ada_bf16_u4_g128_m32_k128_n64_t128
//   marlin_ada_bf16_u4b8_g128_m48_k64_n256_t256    marlin_ada_bf16_u4_g128_m48_k64_n256_t256
//   marlin_ada_bf16_u4b8_g128_m48_k64_n128_t128    marlin_ada_bf16_u4_g128_m48_k64_n128_t128
//   marlin_ada_bf16_u4b8_g128_m48_k128_n64_t128    marlin_ada_bf16_u4_g128_m48_k128_n64_t128
//   marlin_ada_bf16_u4b8_g128_m64_k64_n256_t256    marlin_ada_bf16_u4_g128_m64_k64_n256_t256
//   marlin_ada_bf16_u4b8_g128_m64_k64_n128_t128    marlin_ada_bf16_u4_g128_m64_k64_n128_t128
//   marlin_ada_bf16_u4b8_g128_m64_k128_n64_t128    marlin_ada_bf16_u4_g128_m64_k128_n64_t128
//
// Not instantiated (each is one more MARLIN_ADA_ENTRY line plus, for a new weight type, its
// dequant specialization): FP16 activations, per-channel scales (group -1), group 32 and 64,
// GPTQ int8 (kU8B128), FP8 weights, NVFP4 / MXFP4 (kFE2M1f) and MXFP8, float zero points
// (HQQ), stages 2 (Turing).
//
// ── NUMERICS ───────────────────────────────────────────────────────────────────────────
//
// FP32 accumulation in mma.sync m16n8k16 BF16; the dequantized weight is
// `(q - 8) * scale` (kU4B8) or `(q - z) * scale` (kU4, with the zero point subtracted in BF16
// before the scale multiply), both rounded to BF16 before the MMA, exactly as in vLLM. The
// Ada tree's `--fmad=false` does not touch the explicit `lop3`, `__hsub2`, `__hfma2` and
// `__hmul2` steps. With use_atomic_add = 0 a launch is deterministic, and WITHIN one entry
// (same grid, same use_fp32_reduce) row r of C has the same bits for every prob_m up to the
// entry's m: the k split across blocks, `iters = ceil(k_tiles * n_tiles / grid)`, does not
// depend on m while one launch covers m (`parallel` 1), and rows past prob_m are predicated
// off. ACROSS entries the bits differ (another thread_k splits k differently; m8 uses the
// transposed MMA), so a host that switches entries with m, as vLLM's chooser does, is not
// row-invariant; use_atomic_add = 1 makes the order run-dependent. L18b's microtest owns the
// numerics gate (within 2 BF16 ulps plus a term-magnitude bound of its CPU reference, M = 1,
// 8, 16, 17, 32, 64).

#include "marlin_template_ada.cuh"

// The Marlin pipeline depth for sm_80 and newer (marlin.cu:286).
#define MARLIN_ADA_STAGES 4
// group size 128 = 8 blocks of 16 inputs (marlin.cu:256).
#define MARLIN_ADA_GROUP_BLOCKS 8

namespace marlin_ada_check {

// Shared-memory bytes a configuration needs: marlin.cu:96-147 for 4-bit weights, group size
// 128 and integer zero points, as a compile-time function.
constexpr int cache_bytes(int thread_k, int thread_n, int thread_m_blocks, bool has_zp) {
  const int num_bits = 4;
  const int group_size = 128;
  const int stages = MARLIN_ADA_STAGES;
  const int pack_factor = 32 / num_bits;
  const int tb_m = thread_m_blocks * 16;
  const int sh_a_size = stages * (tb_m * thread_k) * 2;
  const int sh_b_size = stages * (thread_k * thread_n / pack_factor) * 4;
  const int sh_red_size = tb_m * (thread_n + 8) * 2;
  const int sh_bias_size = thread_n * 2;
  const int small = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  const int big = sh_b_size > sh_red_size ? sh_b_size : sh_red_size;
  const int tmp_size = big > small ? big : small;
  const int tb_groups = (thread_k + group_size - 1) / group_size;
  const int sh_s_size = tb_groups * thread_n * 2 * stages;
  const int sh_zp_size = has_zp ? sh_s_size / 4 : 0;
  return tmp_size + sh_a_size + sh_s_size + sh_zp_size;
}

// The sm_86 / sm_89 per-block opt-in (cudaDevAttrMaxSharedMemoryPerBlockOptin), less the
// 512 bytes vLLM keeps back when it validates a configuration (marlin.cu:217).
constexpr int kAdaSmemBudget = 101376 - 512;

// Every configuration below must fit, with and without zero points; the largest is
// m64 with (64, 256, 256) and zero points.
static_assert(cache_bytes(128, 128, 1, true) <= kAdaSmemBudget, "m8/m16 k128 n128");
static_assert(cache_bytes(64, 128, 1, true) <= kAdaSmemBudget, "m8/m16 k64 n128");
static_assert(cache_bytes(128, 64, 1, true) <= kAdaSmemBudget, "m8/m16 k128 n64");
static_assert(cache_bytes(64, 256, 4, true) <= kAdaSmemBudget, "m32..m64 k64 n256");
static_assert(cache_bytes(64, 128, 4, true) <= kAdaSmemBudget, "m32..m64 k64 n128");
static_assert(cache_bytes(128, 64, 4, true) <= kAdaSmemBudget, "m32..m64 k128 n64");

// The weight types are the 4-bit ones the budget above assumes.
static_assert(vllm::scalar_size_bits(vllm::kU4B8Id) == 4, "kU4B8 is 4-bit");
static_assert(vllm::scalar_size_bits(vllm::kU4Id) == 4, "kU4 is 4-bit");
static_assert(128 == 16 * MARLIN_ADA_GROUP_BLOCKS, "group size 128");

}  // namespace marlin_ada_check

// The exported parameter list: kernel.h's MARLIN_KERNEL_PARAMS with `int` for the three bool
// flags, so every argument is a 4- or 8-byte value on the host side.
#define MARLIN_ADA_ENTRY_PARAMS                                                         \
  const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C,          \
      int4* __restrict__ C_tmp, const int4* __restrict__ b_bias_ptr,                     \
      const int4* __restrict__ scales_ptr, const float* __restrict__ global_scale_ptr,   \
      const int4* __restrict__ zp_ptr, int prob_m, int prob_n, int prob_k, int lda,      \
      int* locks, int has_bias, int use_atomic_add, int use_fp32_reduce, int max_shared_mem

// One exported entry: BF16 activations, scales and output; weight type W_ID; group 128;
// 4 stages; integer (not float) zero points.
#define MARLIN_ADA_ENTRY(NAME, W_ID, THREADS, M_BLOCKS, N_BLOCKS, K_BLOCKS, M8)                  \
  extern "C" __global__ void NAME(MARLIN_ADA_ENTRY_PARAMS) {                                     \
    MARLIN_NAMESPACE_NAME::Marlin<vllm::kBFloat16Id, W_ID, vllm::kBFloat16Id, vllm::kBFloat16Id, \
                                  THREADS, M_BLOCKS, N_BLOCKS, K_BLOCKS, M8, MARLIN_ADA_STAGES,  \
                                  MARLIN_ADA_GROUP_BLOCKS, false>(                               \
        A, B, C, C_tmp, b_bias_ptr, scales_ptr, global_scale_ptr, zp_ptr, prob_m, prob_n,         \
        prob_k, lda, locks, has_bias != 0, use_atomic_add != 0, use_fp32_reduce != 0,             \
        max_shared_mem);                                                                         \
  }

// ── GPTQ int4 (kU4B8), group 128 ──
// NAME, weight, threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m8_k128_n128_t256, vllm::kU4B8Id, 256, 1, 8, 8, true)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m8_k64_n128_t128, vllm::kU4B8Id, 128, 1, 8, 4, true)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m8_k128_n64_t128, vllm::kU4B8Id, 128, 1, 4, 8, true)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m16_k128_n128_t256, vllm::kU4B8Id, 256, 1, 8, 8, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m16_k64_n128_t128, vllm::kU4B8Id, 128, 1, 8, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m16_k128_n64_t128, vllm::kU4B8Id, 128, 1, 4, 8, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m32_k64_n256_t256, vllm::kU4B8Id, 256, 2, 16, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m32_k64_n128_t128, vllm::kU4B8Id, 128, 2, 8, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m32_k128_n64_t128, vllm::kU4B8Id, 128, 2, 4, 8, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m48_k64_n256_t256, vllm::kU4B8Id, 256, 3, 16, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m48_k64_n128_t128, vllm::kU4B8Id, 128, 3, 8, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m48_k128_n64_t128, vllm::kU4B8Id, 128, 3, 4, 8, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m64_k64_n256_t256, vllm::kU4B8Id, 256, 4, 16, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m64_k64_n128_t128, vllm::kU4B8Id, 128, 4, 8, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4b8_g128_m64_k128_n64_t128, vllm::kU4B8Id, 128, 4, 4, 8, false)

// ── AWQ int4 with int4 zero points (kU4), group 128 ──
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m8_k128_n128_t256, vllm::kU4Id, 256, 1, 8, 8, true)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m8_k64_n128_t128, vllm::kU4Id, 128, 1, 8, 4, true)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m8_k128_n64_t128, vllm::kU4Id, 128, 1, 4, 8, true)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m16_k128_n128_t256, vllm::kU4Id, 256, 1, 8, 8, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m16_k64_n128_t128, vllm::kU4Id, 128, 1, 8, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m16_k128_n64_t128, vllm::kU4Id, 128, 1, 4, 8, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m32_k64_n256_t256, vllm::kU4Id, 256, 2, 16, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m32_k64_n128_t128, vllm::kU4Id, 128, 2, 8, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m32_k128_n64_t128, vllm::kU4Id, 128, 2, 4, 8, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m48_k64_n256_t256, vllm::kU4Id, 256, 3, 16, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m48_k64_n128_t128, vllm::kU4Id, 128, 3, 8, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m48_k128_n64_t128, vllm::kU4Id, 128, 3, 4, 8, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m64_k64_n256_t256, vllm::kU4Id, 256, 4, 16, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m64_k64_n128_t128, vllm::kU4Id, 128, 4, 8, 4, false)
MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m64_k128_n64_t128, vllm::kU4Id, 128, 4, 4, 8, false)
