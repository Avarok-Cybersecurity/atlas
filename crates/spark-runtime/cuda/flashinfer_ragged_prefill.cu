// flashinfer_ragged_prefill.cu
//
// Host-callable C-ABI wrapper around FlashInfer's FA2 ragged batched-prefill
// attention kernel, specialized for BF16 Q/K/V/O, head_dim = 256, GQA, causal
// (or non-causal). Built for GB10 / sm_121f, CUDA 13.
//
// This file mirrors the canonical caller `csrc/batch_prefill.cu`
// (`BatchPrefillWithRaggedKVCacheRun`), stripped of the torch / TensorView
// wrapping. It uses:
//   - flashinfer::PrefillPlan        (scheduler.cuh)      -> scheduler metadata
//   - flashinfer::PrefillPlanInfo    (scheduler.cuh)      -> plan offsets
//   - flashinfer::BatchPrefillRaggedParams (default_prefill_params.cuh)
//   - flashinfer::BatchPrefillWithRaggedKVCacheDispatched (prefill.cuh)
//   - flashinfer::DefaultAttention<false,false,false,false> (variants.cuh)
//
// The wrapper owns NO persistent state; all scratch buffers are passed in.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstddef>
#include <vector>

#include <flashinfer/allocator.h>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>

namespace flashinfer {
// Forward declaration of the dispatched ragged-prefill entry point. It is also
// defined (as a template) in prefill.cuh which we include above; declaring it
// here keeps parity with csrc/batch_prefill.cu.
template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, bool enable_pdl,
                                                    cudaStream_t stream);
}  // namespace flashinfer

using flashinfer::BatchPrefillRaggedParams;
using flashinfer::DefaultAttention;
using flashinfer::GetPtrFromBaseOffset;
using flashinfer::MaskMode;
using flashinfer::PosEncodingMode;
using flashinfer::PrefillPlan;
using flashinfer::PrefillPlanInfo;

// Compile-time configuration. head_dim is always 256 here.
namespace {
constexpr uint32_t kHeadDim = 256;
constexpr PosEncodingMode kPosEnc = PosEncodingMode::kNone;
constexpr bool kUseFp16QkReduction = false;

// Standard (composable default) attention variant: no custom mask, no sliding
// window, no logits soft-cap, no alibi. This is exactly what the JIT codegen
// emits for plain causal attention:
//   DefaultAttention<false, false, false, false>
// (see flashinfer/jit/attention/modules.py and
//  csrc/batch_decode_mla_config.jinja).
using StandardAttention = DefaultAttention</*use_custom_mask=*/false,
                                           /*use_sliding_window=*/false,
                                           /*use_logits_soft_cap=*/false,
                                           /*use_alibi=*/false>;

using Params = BatchPrefillRaggedParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;
}  // namespace

// ---------------------------------------------------------------------------
// Workspace sizing.
//
// PrefillPlan carves its scheduler metadata out of caller-provided int/float
// workspaces using an AlignedAllocator. The exact byte counts depend on
// `padded_batch_size`, which is computed at plan time from the indptr arrays
// and the device's SM count. We therefore return generous upper bounds.
//
// int_ws layout (per PrefillPlan):
//   request_indices : padded_batch_size * 4
//   qo_tile_indices : padded_batch_size * 4
//   kv_tile_indices : padded_batch_size * 4
//   o_indptr        : (batch+1) * 4
//   kv_chunk_size   : 4
//   [cuda_graph]    : total_num_rows (uint32) * 4  (we never enable cuda graph)
//   [split_kv]      : merge_indptr (total_rows+1)*4 + block_valid_mask (padded_batch)*1
//   + 16B alignment slack per allocation.
//
// padded_batch_size is bounded by the kernel grid: <= 2 * num_sm rounded up.
// We use a safe ceiling of 4096 CTAs.
//
// float_ws layout (only used when split_kv): tmp_v + tmp_s, sized
//   num_qo_heads * padded_batch_size * cta_tile_q * (head_dim_vo + 1) * 4 bytes.
// ---------------------------------------------------------------------------
extern "C" int atlas_fi_ragged_prefill_workspace_sizes(
    uint32_t max_batch, uint32_t max_total_qo_rows, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t head_dim,
    size_t* float_ws_bytes_out, size_t* int_ws_bytes_out, size_t* pinned_int_ws_bytes_out) {
  if (float_ws_bytes_out == nullptr || int_ws_bytes_out == nullptr ||
      pinned_int_ws_bytes_out == nullptr) {
    return -1;
  }
  if (head_dim != kHeadDim) {
    return -2;
  }

  // Upper bound on the scheduler's padded batch size. PrefillPlan uses
  // 2 * num_sm CTAs as the grid ceiling; query the device but clamp to a
  // safe minimum so callers can size buffers before any device is selected.
  int num_sm = 0;
  int dev_id = 0;
  if (cudaGetDevice(&dev_id) == cudaSuccess) {
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
  }
  const uint32_t grid_ceiling = (num_sm > 0) ? static_cast<uint32_t>(2 * num_sm) : 512u;
  // padded_batch_size can be at most the grid ceiling, but also at least the
  // number of qo tiles which is bounded by total_qo_rows. Take a safe max.
  uint32_t padded_batch = grid_ceiling;
  if (max_batch > padded_batch) padded_batch = max_batch;
  if (max_total_qo_rows > padded_batch) padded_batch = max_total_qo_rows;
  // Add headroom.
  padded_batch += 256;

  const size_t align_slack = 16;

  // int workspace (device) and pinned int workspace (host) have the SAME
  // layout (PrefillPlan fills the pinned buffer then memcpys to device).
  size_t int_bytes = 0;
  int_bytes += static_cast<size_t>(padded_batch) * sizeof(int32_t) + align_slack;  // request_indices
  int_bytes += static_cast<size_t>(padded_batch) * sizeof(int32_t) + align_slack;  // qo_tile_indices
  int_bytes += static_cast<size_t>(padded_batch) * sizeof(int32_t) + align_slack;  // kv_tile_indices
  int_bytes += (static_cast<size_t>(max_batch) + 1) * sizeof(int32_t) + align_slack;  // o_indptr
  int_bytes += sizeof(int32_t) + align_slack;                                       // kv_chunk_size
  int_bytes += sizeof(uint32_t) + align_slack;                                      // total_num_rows (cuda graph; unused but cheap)
  // split_kv extras:
  int_bytes += (static_cast<size_t>(max_total_qo_rows) + 1) * sizeof(int32_t) + align_slack;  // merge_indptr
  int_bytes += static_cast<size_t>(padded_batch) * sizeof(bool) + align_slack;                // block_valid_mask

  // float workspace (device), only used when split_kv. Size for the largest
  // cta_tile_q (128).
  const uint32_t cta_tile_q_max = 128;
  size_t float_bytes = 0;
  float_bytes += static_cast<size_t>(num_qo_heads) * padded_batch * cta_tile_q_max *
                     head_dim * sizeof(float) + align_slack;  // tmp_v
  float_bytes += static_cast<size_t>(num_qo_heads) * padded_batch * cta_tile_q_max *
                     sizeof(float) + align_slack;             // tmp_s

  *int_ws_bytes_out = int_bytes;
  *pinned_int_ws_bytes_out = int_bytes;
  *float_ws_bytes_out = float_bytes;
  return 0;
}

// ---------------------------------------------------------------------------
// One-shot ragged batched prefill attention.
// ---------------------------------------------------------------------------
namespace {

// Populate the scheduler-derived params fields from a decoded plan_info, and
// resolve split-KV scratch pointers (tmp_v/tmp_s). Mirrors the
// post-ADDITIONAL_PARAMS_SETTER block of csrc/batch_prefill.cu.
inline void apply_plan_info(Params& params, const PrefillPlanInfo& plan_info,
                            void* int_buffer_ptr, void* float_buffer_ptr,
                            __nv_bfloat16** tmp_v_out, float** tmp_s_out) {
  params.request_indices =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.request_indices_offset);
  params.qo_tile_indices =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
  params.kv_tile_indices =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);

  __nv_bfloat16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan_info.split_kv) {
    params.merge_indptr =
        GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.merge_indptr_offset);
    tmp_v = GetPtrFromBaseOffset<__nv_bfloat16>(float_buffer_ptr, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask =
          GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
    }
  }
  params.padded_batch_size = plan_info.padded_batch_size;
  params.max_total_num_rows = plan_info.total_num_rows;
  if (plan_info.enable_cuda_graph) {
    params.total_num_rows =
        GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
  }
  *tmp_v_out = tmp_v;
  *tmp_s_out = tmp_s;
}

// Run the dispatched kernel for a fixed MaskMode, dispatching on cta_tile_q.
template <MaskMode MASK_MODE>
inline cudaError_t run_dispatched(Params& params, __nv_bfloat16* tmp_v, float* tmp_s,
                                  int64_t cta_tile_q, cudaStream_t stream) {
  cudaError_t status = cudaSuccess;
  DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    status = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
        CTA_TILE_Q, kHeadDim, kHeadDim, kPosEnc, kUseFp16QkReduction, MASK_MODE,
        StandardAttention, Params>(params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
  });
  return status;
}

}  // namespace

extern "C" int atlas_fi_ragged_prefill_bf16_hd256(
    const void* q, const void* k, const void* v, void* o,
    const int32_t* qo_indptr_h, const int32_t* kv_indptr_h,
    const int32_t* qo_indptr_d, const int32_t* kv_indptr_d,
    uint32_t batch, uint32_t total_qo_rows, uint32_t total_kv_rows,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
    float sm_scale, int causal,
    void* float_ws, size_t float_ws_bytes,
    void* int_ws, size_t int_ws_bytes,
    void* pinned_int_ws, size_t pinned_int_ws_bytes,
    void* stream_raw) {
  if (head_dim != kHeadDim) return -2;
  if (num_kv_heads == 0 || num_qo_heads % num_kv_heads != 0) return -3;

  cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);

  // ------------------------------------------------------------------
  // Stage 1: plan. Fill scheduler metadata into int_ws / pinned_int_ws and
  // produce a PrefillPlanInfo. PrefillPlan reads the HOST indptr arrays.
  // ------------------------------------------------------------------
  PrefillPlanInfo plan_info;
  cudaError_t status = PrefillPlan<int32_t>(
      float_ws, float_ws_bytes,
      int_ws, pinned_int_ws, int_ws_bytes,
      plan_info,
      const_cast<int32_t*>(qo_indptr_h), const_cast<int32_t*>(kv_indptr_h),
      /*total_num_rows=*/total_qo_rows,
      /*batch_size=*/batch,
      num_qo_heads, num_kv_heads,
      /*head_dim_qk=*/head_dim, /*head_dim_vo=*/head_dim,
      /*page_size=*/1,
      /*enable_cuda_graph=*/false,
      /*sizeof_dtype_o=*/sizeof(__nv_bfloat16),
      /*window_left=*/-1,
      /*fixed_split_size=*/-1,
      /*disable_split_kv=*/false,
      /*num_colocated_ctas=*/0,
      stream);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  // ------------------------------------------------------------------
  // Stage 2: build params. Contiguous row-major [rows, heads, head_dim] so
  //   stride_n = num_*_heads * head_dim, stride_h = head_dim.
  // ------------------------------------------------------------------
  Params params;  // zero/default-initialized by the host ctor

  params.q = static_cast<__nv_bfloat16*>(const_cast<void*>(q));
  params.k = static_cast<__nv_bfloat16*>(const_cast<void*>(k));
  params.v = static_cast<__nv_bfloat16*>(const_cast<void*>(v));
  params.o = static_cast<__nv_bfloat16*>(o);
  params.lse = nullptr;

  // The kernel reads the DEVICE indptr copies.
  params.q_indptr = const_cast<int32_t*>(qo_indptr_d);
  params.kv_indptr = const_cast<int32_t*>(kv_indptr_d);

  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.group_size = flashinfer::uint_fastdiv(num_qo_heads / num_kv_heads);

  params.q_stride_n = num_qo_heads * head_dim;
  params.q_stride_h = head_dim;
  params.k_stride_n = num_kv_heads * head_dim;
  params.k_stride_h = head_dim;
  params.v_stride_n = num_kv_heads * head_dim;
  params.v_stride_h = head_dim;

  params.window_left = -1;
  params.logits_soft_cap = 0.0f;
  params.sm_scale = sm_scale;
  // rope_* unused (PosEncodingMode::kNone); leave at ctor defaults (0).

  // Scheduler-derived pointers + split-KV scratch.
  __nv_bfloat16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  apply_plan_info(params, plan_info, int_ws, float_ws, &tmp_v, &tmp_s);

  // ------------------------------------------------------------------
  // Stage 3: dispatch the kernel. MaskMode is a compile-time template param,
  // selected at runtime from `causal`.
  // ------------------------------------------------------------------
  if (causal) {
    status = run_dispatched<MaskMode::kCausal>(params, tmp_v, tmp_s, plan_info.cta_tile_q, stream);
  } else {
    status = run_dispatched<MaskMode::kNone>(params, tmp_v, tmp_s, plan_info.cta_tile_q, stream);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return 0;
}
