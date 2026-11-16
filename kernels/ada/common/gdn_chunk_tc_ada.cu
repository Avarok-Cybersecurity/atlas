// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
//
// Portions derived from Cinference (https://github.com/satellitedown/cinference,
// commit 383e5db), itself derived from NInfer by Neroued and contributors
// (https://github.com/Neroued/ninfer), Apache License 2.0. Source:
//   src/ops/linear_attention/gated_delta_net/chunked/common.cuh:1-21,
//   chunked/prepare_wy_wu.cuh:1-724 and prepare_wy_wu.cu:1-46,
//   chunked/state_passing.cuh:1-538 and state_passing.cu:1-52,
//   chunked/output.cuh:1-370 and output.cu:1-57,
//   gated_delta_net/common.cuh:52-104 (SmemTile, bf16 -> f32 panel load),
//   src/ops/common/mma.cuh:10-43,91-104 and memory.cuh:9-137 (ldmatrix,
//   mma.sync, cp.async wrappers; those two files carry satellitedown's
//   modification notice), src/ops/common/bf16_vector.cuh:13-26.
// The chunked/*.cuh files carry no Cinference modification line, so they are
// NInfer's original code under the root Apache-2.0 LICENSE.
// Changes (by Atlas contributors), detailed in CINFERENCE-GDN-CHUNK-ATTRIBUTION.md:
//   * one translation unit with three `extern "C"` entries and Atlas's
//     plain-argument ABI; each template is bound to one sm_89 geometry, and
//     the C++20 template lambdas are rewritten as template functions;
//   * Atlas layouts: q/k/v read from the packed conv output at `qk_stride` /
//     `v_stride` token pitches, gate/beta at `gb_stride`; the LINEAR gate is
//     taken to log space in-kernel as logf(fmaxf(gate, 1e-30)) and summed by one
//     thread in token order, exactly as Atlas's FLA path does
//     (gated_delta_rule_fla.cu), instead of upstream's Hillis-Steele scan of a
//     log-space input; the FP32 state is Atlas's [k][v] layout, updated in
//     place, contiguous or through a per-request pointer table; the workspace is
//     blocked per (chunk, value head) like Atlas's FLA scratch;
//   * a partial last chunk is handled in-kernel (zero-filled rows, zero beta,
//     flat log decay, masked stores) instead of requiring T % 64 == 0;
//   * a batch dimension for co-dispatched prefill;
//   * expf instead of ex2.approx for every decay, as Atlas's FLA path;
//   * TF32 operands are rounded to nearest (cvt.rna.tf32.f32) where they are
//     produced instead of truncated by the MMA;
//   * fixed geometries for sm_89 and 34 SMs: prepare 4 warps (23.75 KiB, 4
//     CTAs per SM), state passing 16-column value strips (8 warps, 44.25 KiB,
//     2 CTAs per SM), output 4 warps (24 KiB, 4 CTAs per SM) with a
//     grid-stride chunk loop the host sizes from the device's SM count;
//   * shared memory through prefill_dyn_smem.cuh with a launch floor per entry.
//
// ── What this computes ──────────────────────────────────────────────────────
// The gated delta rule over a whole prompt, per value head (k_dim = v_dim = 128):
//     S_t = g_t S_{t-1} + k_t ((v_t - g_t S_{t-1}^T k_t) beta_t)^T
//     o_t = S_t^T q_t / sqrt(k_dim)
// in chunks of 64 tokens (the WY form, as Atlas's FLA path):
//   1. prepare: gc = cumsum(log g) per chunk, T_inv = (I + strict_lower(beta_i
//      exp(gc_i - gc_l) k_l.k_i))^-1, U = T_inv (beta V), W = T_inv (beta exp(gc) K)
//      (BF16 m16n8k16 for K K^T, TF32 m16n8k8 for the triangular products);
//   2. state passing, serial over chunks, state in FP32 MMA accumulators:
//      v_new = U - W S_c, S_{c+1} = exp(gc_last) S_c + K^T (v_new exp(gc_last - gc));
//      writes bf16(S_c) and bf16(v_new) for stage 3;
//   3. output: O = (exp(gc) Q S_c + tril(exp(gc_i - gc_l) Q K^T) v_new) / sqrt(k_dim).
// The scalar spine Atlas runs today (`gated_delta_rule_chunk_delta_h_vfused`) is
// stage 2 on FP32 FMAs; on an H100 it issues zero MMAs and is latency-bound on
// its dependent chain (GDN-PREFILL-ATTRIBUTION.md). This file puts every
// per-chunk product on mma.sync, which sm_80 and later all have.
//
// Selected by the host only when the `[defaults] gdn_prefill_tc` row resolves
// on AND this module is in the image (only kernels/ada compiles it), so no
// other target can reach it (`layers/qwen3_ssm/gdn_chunk_tc_ada.rs`).

#include <cuda_bf16.h>

#include <cstdint>

#include "prefill_dyn_smem.cuh"

namespace gdn_tc_ada {

constexpr int kWarpSize = 32;
constexpr int kStateDim = 128;
constexpr int BT = 64;
constexpr int BC = 16;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;
constexpr int BF16_MMA_K = 16;
// Atlas's FLA gate floor (gated_delta_rule_fla.cu GATE_FLOOR): log(0) would
// make exp(gc_i - gc_l) a NaN; 1e-30 is full decay for any real gate.
constexpr float kGateFloor = 1e-30f;

static_assert(BT % BC == 0, "BT must be a multiple of BC");
static_assert(BT % MMA_M == 0, "BT must be a multiple of MMA_M");

// ── PTX wrappers (NInfer ops/common/mma.cuh, memory.cuh) ────────────────────

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

// Atlas change: round an FP32 value to the nearest TF32 (10 explicit mantissa
// bits) once, where the operand is produced. The MMA itself ignores the low 13
// bits, which is a truncation with a consistent bias toward zero.
__device__ __forceinline__ unsigned tf32_rna_bits_of(float x) {
    unsigned y;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float tf32_round(float x) { return __uint_as_float(tf32_rna_bits_of(x)); }

__device__ __forceinline__ unsigned tf32_round_bits(unsigned bits) {
    return tf32_rna_bits_of(__uint_as_float(bits));
}

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 4 || sizeof(V) == 8 || sizeof(V) == 16, "vector width");
    return *reinterpret_cast<const V*>(ptr);
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 4 || sizeof(V) == 8 || sizeof(V) == 16, "vector width");
    *reinterpret_cast<V*>(ptr) = value;
}

// 16-byte asynchronous global -> shared copies. `src_bytes` < 16 zero-fills the
// rest of the destination (0 = pure zero fill, no global read is issued); the
// source address is kept inside the tensor regardless.
__device__ __forceinline__ void cp_async16_ca(void* smem_dst, const void* gmem_src, int src_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async16_cg(void* smem_dst, const void* gmem_src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

struct alignas(8) Bf16x4Pack {
    __nv_bfloat162 pair[2];
};
static_assert(sizeof(Bf16x4Pack) == 8, "Bf16x4Pack is 8 bytes");

__device__ __forceinline__ int valid_rows(unsigned seq_len, int chunk) {
    const int rest = static_cast<int>(seq_len) - chunk * BT;
    return rest < BT ? rest : BT;
}

// ── Swizzled shared-memory views (NInfer gated_delta_net/common.cuh) ────────

template <int STRIDE>
struct SmemTile {
    float* __restrict__ base;
    static_assert(STRIDE == 16 || STRIDE >= 32, "SmemTile: STRIDE in {16, 32, 64, 128}");

    __device__ __forceinline__ int swz_xor(int row) const {
        if constexpr (STRIDE >= 32) {
            return ((row & 3) << 3) | (row & 4);
        } else {
            return ((row >> 1) & 3) << 2;
        }
    }

    __device__ __forceinline__ float& at(int row, int col) const {
        return base[row * STRIDE + (col ^ swz_xor(row))];
    }

    __device__ __forceinline__ float4& vec4_at(int row, int col) const {
        return *reinterpret_cast<float4*>(&base[row * STRIDE + (col ^ swz_xor(row))]);
    }
};

template <int STRIDE>
struct Bf16Tile {
    __nv_bfloat16* __restrict__ base;
    static_assert(STRIDE == 16 || STRIDE == 32 || STRIDE == 64 || STRIDE == 128, "Bf16Tile stride");

    __device__ __forceinline__ int swizzled_col(int row, int col) const {
        return col ^ ((row & (STRIDE / 8 - 1)) << 3);
    }

    __device__ __forceinline__ __nv_bfloat16* ptr(int row, int col) const {
        return base + row * STRIDE + swizzled_col(row, col);
    }
};

// ROWS x STRIDE BF16 rows at `gmem_row_stride` -> FP32 swizzled tile (all rows
// valid: the callers read the blocked workspace, which holds whole chunks).
template <int ROWS, int STRIDE, int THREADS, class View>
__device__ __forceinline__ void issue_load_bf16_to_float_vec4(View view,
                                                              const __nv_bfloat16* __restrict__ row0,
                                                              std::int64_t gmem_row_stride, int tid) {
    static_assert(STRIDE % 4 == 0, "STRIDE must be a multiple of 4");
    constexpr int VEC_PER_ROW = STRIDE / 4;
    constexpr int N_VEC = ROWS * VEC_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VEC; v += THREADS) {
        const int row = v / VEC_PER_ROW;
        const int col4 = v - row * VEC_PER_ROW;
        const Bf16x4Pack packed =
            load_vec<Bf16x4Pack>(row0 + static_cast<std::int64_t>(row) * gmem_row_stride + col4 * 4);
        const float2 lo = __bfloat1622float2(packed.pair[0]);
        const float2 hi = __bfloat1622float2(packed.pair[1]);
        view.vec4_at(row, col4 * 4) = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
}

// ═══ Stage 1: prepare_wy_wu (NInfer chunked/prepare_wy_wu.cuh) ═════════════

namespace prep {

constexpr int N_SUB = BT / BC;          // 4
constexpr int WY_WARPS = N_SUB;         // 4 warps own the triangular construction
constexpr int N_K_TILES = BT / MMA_K;   // 8
constexpr int SCR_STRIDE = 20;          // Phase D scratch row stride (2-way write, 0-way read)
constexpr int K_PANEL_COLS = 32;
constexpr int WU_PANEL_COLS = 16;
constexpr int BLOCK_WARPS = 4;
constexpr int BLOCK_THREADS = BLOCK_WARPS * kWarpSize;
constexpr int N_K_PANELS = kStateDim / K_PANEL_COLS;          // 4
constexpr int K_TILES_PER_PANEL = K_PANEL_COLS / BF16_MMA_K;  // 2
constexpr int N_WU_PANELS = kStateDim / WU_PANEL_COLS;        // 8
constexpr int WU_N_TILES = WU_PANEL_COLS / MMA_N;             // 2
constexpr int T_INV_FLOATS = BT * BT;
constexpr int SCRATCH_FLOATS = WY_WARPS * BC * SCR_STRIDE;
constexpr int WU_STAGE_FLOATS = BT * WU_PANEL_COLS;
constexpr int K_STAGE_FLOATS = BT * K_PANEL_COLS / 2;
constexpr int STAGE_FLOATS = SCRATCH_FLOATS > WU_STAGE_FLOATS ? SCRATCH_FLOATS : WU_STAGE_FLOATS;
constexpr int OUTPUT_STAGE_FLOATS = BT * WU_PANEL_COLS / 2;
constexpr int SMEM_FLOATS = T_INV_FLOATS + STAGE_FLOATS + OUTPUT_STAGE_FLOATS + 3 * BT;

static_assert(MMA_M == BC, "kernel assumes MMA m == BC");
static_assert(STAGE_FLOATS >= K_STAGE_FLOATS, "K stage aliases the scratch stage");
static_assert(SMEM_FLOATS == 6080, "mirrored by PREPARE_SMEM in gdn_chunk_tc_ada.rs");

struct Tiles {
    float f[SMEM_FLOATS];
};

__device__ __forceinline__ int wu_output_swizzled_col(int row, int col) {
    return (((col >> 3) ^ (row & (WU_PANEL_COLS / 8 - 1))) << 3) | (col & 7);
}

__device__ __forceinline__ void scatter_frag_to_scr(const float frag[8], float* __restrict__ scr_smem,
                                                    int warp, int lane) {
    float* Sptr = scr_smem + warp * BC * SCR_STRIDE;
    const int lane_g = lane >> 2;
    const int col_2t = (lane & 3) << 1;
    Sptr[lane_g * SCR_STRIDE + col_2t] = frag[0];
    Sptr[lane_g * SCR_STRIDE + col_2t + 1] = frag[1];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t] = frag[2];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 1] = frag[3];
    Sptr[lane_g * SCR_STRIDE + col_2t + 8] = frag[4];
    Sptr[lane_g * SCR_STRIDE + col_2t + 9] = frag[5];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 8] = frag[6];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 9] = frag[7];
}

// 16x16x16: A from raw row-major scratch, B from the swizzled M view.
__device__ __forceinline__ void mma16_raw_x_swiz(float D[8], int lane, const float* __restrict__ A_buf,
                                                 SmemTile<BT> M_view, int M_row_off, int M_col_off) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0 = A_buf[lane_g * SCR_STRIDE + (k_off + lane_t)];
        const float a1 = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t)];
        const float a2 = A_buf[lane_g * SCR_STRIDE + (k_off + lane_t + 4)];
        const float a3 = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t + 4)];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0 = M_view.at(M_row_off + k_off + lane_t, M_col_off + n_off + lane_g);
            const float b1 = M_view.at(M_row_off + k_off + lane_t + 4, M_col_off + n_off + lane_g);
            mma_tf32(D[nt * 4 + 0], D[nt * 4 + 1], D[nt * 4 + 2], D[nt * 4 + 3], a0, a1, a2, a3, b0, b1);
        }
    }
}

// 16x16x16: A from the swizzled M view, B from raw row-major scratch.
__device__ __forceinline__ void mma16_swiz_x_raw(float D[8], int lane, SmemTile<BT> M_view,
                                                 int M_row_off, int M_col_off,
                                                 const float* __restrict__ B_buf) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0 = M_view.at(M_row_off + lane_g, M_col_off + k_off + lane_t);
        const float a1 = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t);
        const float a2 = M_view.at(M_row_off + lane_g, M_col_off + k_off + lane_t + 4);
        const float a3 = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t + 4);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0 = B_buf[(k_off + lane_t) * SCR_STRIDE + (n_off + lane_g)];
            const float b1 = B_buf[(k_off + lane_t + 4) * SCR_STRIDE + (n_off + lane_g)];
            mma_tf32(D[nt * 4 + 0], D[nt * 4 + 1], D[nt * 4 + 2], D[nt * 4 + 3], a0, a1, a2, a3, b0, b1);
        }
    }
}

// Templated on (MY_W, MY_J) so `A_reg[k]` resolves to compile-time indices.
template <int MY_W, int MY_J>
__device__ __forceinline__ void compute_off_diag(float out[8], int warp, int lane,
                                                 const float A_reg[N_SUB][8],
                                                 float* __restrict__ scr_smem, SmemTile<BT> M_view) {
    static_assert(0 <= MY_J && MY_J < MY_W && MY_W <= N_SUB, "off-diagonal block index");
    float sum[8] = {};
#pragma unroll
    for (int k = MY_J; k < MY_W; ++k) {
        scatter_frag_to_scr(A_reg[k], scr_smem, warp, lane);
        __syncwarp();
        const float* A_buf = scr_smem + warp * BC * SCR_STRIDE;
        mma16_raw_x_swiz(sum, lane, A_buf, M_view, k * BC, MY_J * BC);
        __syncwarp();
        if (k == MY_J) {
#pragma unroll
            for (int e = 0; e < 8; ++e) sum[e] += A_reg[k][e];
        }
    }
    scatter_frag_to_scr(sum, scr_smem, warp, lane);
    __syncwarp();
    const float* B_buf = scr_smem + warp * BC * SCR_STRIDE;
    float prod[8] = {};
    mma16_swiz_x_raw(prod, lane, M_view, MY_W * BC, MY_W * BC, B_buf);
#pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = prod[e] + sum[e];
}

__device__ __forceinline__ void store_frag_to_M(const float frag[8], int my_w, int my_j, int lane_g,
                                                int lane_t, SmemTile<BT> M_view) {
    const int row_g0 = my_w * BC + lane_g;
    const int row_g1 = row_g0 + 8;
    const int col_base = my_j * BC + 2 * lane_t;
    M_view.at(row_g0, col_base) = frag[0];
    M_view.at(row_g0, col_base + 1) = frag[1];
    M_view.at(row_g1, col_base) = frag[2];
    M_view.at(row_g1, col_base + 1) = frag[3];
    M_view.at(row_g0, col_base + 8) = frag[4];
    M_view.at(row_g0, col_base + 9) = frag[5];
    M_view.at(row_g1, col_base + 8) = frag[6];
    M_view.at(row_g1, col_base + 9) = frag[7];
}

template <int DIAG_BLOCK>
__device__ __forceinline__ void solve_diag_block(int lane, SmemTile<BT> M_view) {
    constexpr int diag_off = DIAG_BLOCK * BC;
    const int wcol = lane & 15;
    for (int i = 1; i < BC; ++i) {
        const int row_i = diag_off + i;
        const int col = diag_off + wcol;
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < BC - 1; ++j) {
            if (j < i) { sum += M_view.at(row_i, diag_off + j) * M_view.at(diag_off + j, col); }
        }
        __syncwarp();
        if (lane < 16 && wcol < i) { M_view.at(row_i, col) += sum; }
        __syncwarp();
    }
}

// Scaled BF16 rows -> one FP32 WU panel, rows >= valid zero-filled without a
// global read (Atlas change: partial last chunk), values rounded to TF32.
__device__ __forceinline__ void load_scaled_wu_panel(SmemTile<WU_PANEL_COLS> panel,
                                                     const __nv_bfloat16* __restrict__ input_row0,
                                                     std::int64_t input_row_stride,
                                                     const float* __restrict__ scale, int panel_col,
                                                     int valid, int tid) {
    constexpr int VEC_PER_ROW = WU_PANEL_COLS / 4;
    constexpr int N_VEC = BT * VEC_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VEC; v += BLOCK_THREADS) {
        const int row = v / VEC_PER_ROW;
        const int col4 = (v - row * VEC_PER_ROW) * 4;
        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row < valid) {
            const Bf16x4Pack packed = load_vec<Bf16x4Pack>(
                input_row0 + static_cast<std::int64_t>(row) * input_row_stride + panel_col + col4);
            const float2 lo = __bfloat1622float2(packed.pair[0]);
            const float2 hi = __bfloat1622float2(packed.pair[1]);
            const float s = scale[row];
            val = make_float4(tf32_round(lo.x * s), tf32_round(lo.y * s), tf32_round(hi.x * s),
                              tf32_round(hi.y * s));
        }
        panel.vec4_at(row, col4) = val;
    }
}

// D = T_inv @ panel (64 x 16 per panel), stored BF16 to the blocked workspace
// (row stride kStateDim). W is stored with each group of eight columns in the
// order {0,4,1,5,2,6,3,7}, so stage 2's ldmatrix.x2 forms its TF32 A fragment.
template <bool InterleaveOutput>
__device__ __forceinline__ void compute_store_wu_panel(SmemTile<BT> T_view,
                                                       SmemTile<WU_PANEL_COLS> panel,
                                                       __nv_bfloat16* __restrict__ output_smem,
                                                       __nv_bfloat16* __restrict__ output_row0,
                                                       int panel_col, int warp, int lane) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    const int row_g0 = warp * MMA_M + lane_g;
    const int row_g1 = row_g0 + 8;
    const int col_pair = lane_t << 1;

    float D[WU_N_TILES][4] = {};
#pragma unroll
    for (int k_tile = 0; k_tile < N_K_TILES; ++k_tile) {
        const int k_off = k_tile * MMA_K;
        const int col_t0 = k_off + lane_t;
        const int col_t1 = col_t0 + 4;
        const float a0 = T_view.at(row_g0, col_t0);
        const float a1 = T_view.at(row_g1, col_t0);
        const float a2 = T_view.at(row_g0, col_t1);
        const float a3 = T_view.at(row_g1, col_t1);
#pragma unroll
        for (int n = 0; n < WU_N_TILES; ++n) {
            const int col = n * MMA_N + lane_g;
            const float b0 = panel.at(k_off + lane_t, col);
            const float b1 = panel.at(k_off + lane_t + 4, col);
            mma_tf32(D[n][0], D[n][1], D[n][2], D[n][3], a0, a1, a2, a3, b0, b1);
        }
    }

    __nv_bfloat16* const warp_output = output_smem + warp * MMA_M * WU_PANEL_COLS;
#pragma unroll
    for (int n = 0; n < WU_N_TILES; ++n) {
        const int col = n * MMA_N + col_pair;
        store_vec(&warp_output[lane_g * WU_PANEL_COLS + wu_output_swizzled_col(lane_g, col)],
                  __floats2bfloat162_rn(D[n][0], D[n][1]));
        store_vec(&warp_output[(lane_g + 8) * WU_PANEL_COLS + wu_output_swizzled_col(lane_g + 8, col)],
                  __floats2bfloat162_rn(D[n][2], D[n][3]));
    }
    __syncwarp();

    constexpr int STORE_ELEMS = 8;
    constexpr int STORE_PER_ROW = WU_PANEL_COLS / STORE_ELEMS;
    constexpr int STORE_PER_WARP = MMA_M * STORE_PER_ROW;
#pragma unroll
    for (int v = lane; v < STORE_PER_WARP; v += kWarpSize) {
        const int row = v / STORE_PER_ROW;
        const int col8 = (v - row * STORE_PER_ROW) * STORE_ELEMS;
        uint4 packed = load_vec<uint4>(&warp_output[row * WU_PANEL_COLS + wu_output_swizzled_col(row, col8)]);
        if constexpr (InterleaveOutput) {
            packed = {
                (packed.x & 0x0000ffffU) | (packed.z << 16),
                (packed.x >> 16) | (packed.z & 0xffff0000U),
                (packed.y & 0x0000ffffU) | (packed.w << 16),
                (packed.y >> 16) | (packed.w & 0xffff0000U),
            };
        }
        store_vec(output_row0 + static_cast<std::int64_t>(warp * MMA_M + row) * kStateDim + panel_col +
                      col8,
                  packed);
    }
}

// Native BF16 K K^T on the lower-triangular 4x4 sub-block grid: warp w owns
// row strip w and the N_OWNED = w + 1 column blocks at or left of the diagonal.
template <int N_OWNED>
__device__ __forceinline__ void kkt_strip(float (&A_reg)[N_SUB][8], Bf16Tile<K_PANEL_COLS> K_view,
                                          int warp, int lane) {
    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
#pragma unroll
    for (int k_tile = 0; k_tile < K_TILES_PER_PANEL; ++k_tile) {
        const int k_off = k_tile * BF16_MMA_K;
        unsigned af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3],
                    smem_addr(K_view.ptr(warp * BC + a_rowoff, k_off + a_coloff)));
#pragma unroll
        for (int j_sub = 0; j_sub < N_OWNED; ++j_sub) {
#pragma unroll
            for (int n_tile = 0; n_tile < 2; ++n_tile) {
                const int row_b = j_sub * BC + n_tile * MMA_N + b_rin;
                unsigned bf[2];
                ldmatrix_x2(bf[0], bf[1], smem_addr(K_view.ptr(row_b, k_off + b_koff)));
                mma_bf16(A_reg[j_sub][n_tile * 4 + 0], A_reg[j_sub][n_tile * 4 + 1],
                         A_reg[j_sub][n_tile * 4 + 2], A_reg[j_sub][n_tile * 4 + 3], af[0], af[1],
                         af[2], af[3], bf[0], bf[1]);
            }
        }
    }
}

template <int MY_W, int MY_J>
__device__ __forceinline__ void wave_compute_store(const float (&A_reg)[N_SUB][8],
                                                   float* __restrict__ scr_smem, SmemTile<BT> M_view,
                                                   int warp, int lane) {
    float out[8];
    compute_off_diag<MY_W, MY_J>(out, warp, lane, A_reg, scr_smem, M_view);
    store_frag_to_M(out, MY_W, MY_J, lane >> 2, lane & 3, M_view);
}

__device__ __forceinline__ void prepare_body(const __nv_bfloat16* __restrict__ k_in,
                                             const __nv_bfloat16* __restrict__ v_in,
                                             const float* __restrict__ gate,
                                             const float* __restrict__ beta,
                                             __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ U,
                                             float* __restrict__ g_out, unsigned seq_len,
                                             unsigned num_chunks, unsigned num_k_heads,
                                             unsigned num_v_heads, unsigned qk_stride,
                                             unsigned v_stride, unsigned gb_stride, float* smem) {
    const int chunk = static_cast<int>(blockIdx.x);
    const int h_v = static_cast<int>(blockIdx.y);
    const int b = static_cast<int>(blockIdx.z);
    if (chunk >= static_cast<int>(num_chunks) || h_v >= static_cast<int>(num_v_heads)) return;
    const int h_qk = h_v / static_cast<int>(num_v_heads / num_k_heads);
    const int ce = valid_rows(seq_len, chunk);
    const std::int64_t tok0 = static_cast<std::int64_t>(b) * seq_len + static_cast<std::int64_t>(chunk) * BT;
    const std::int64_t blk =
        (static_cast<std::int64_t>(b) * num_chunks + chunk) * num_v_heads + h_v;

    float* const T_inv_smem = smem;
    float* const stage_smem = smem + T_INV_FLOATS;
    auto* const output_smem = reinterpret_cast<__nv_bfloat16*>(stage_smem + STAGE_FLOATS);
    float* const g_smem = stage_smem + STAGE_FLOATS + OUTPUT_STAGE_FLOATS;
    float* const beta_smem = g_smem + BT;
    float* const bg_smem = beta_smem + BT;
    // stage_smem aliases BF16 K (WY-B), FP32 Schur scratch (WY-D) and one
    // pre-scaled FP32 V/K panel (WU); every handover is barrier guarded.
    auto* const K_smem = reinterpret_cast<__nv_bfloat16*>(stage_smem);
    float* const scr_smem = stage_smem;

    SmemTile<BT> M_view{T_inv_smem};
    Bf16Tile<K_PANEL_COLS> K_view{K_smem};
    SmemTile<WU_PANEL_COLS> WU_view{stage_smem};

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    // === Phase WY-A: beta, and the log-space gate cumsum in Atlas's order ===
    if (tid < BT) {
        beta_smem[tid] = (tid < ce) ? beta[(tok0 + tid) * gb_stride + h_v] : 0.0f;
    }
    if (warp == 0) {
        const int t0 = 2 * lane;
        const int t1 = t0 + 1;
        g_smem[t0] = (t0 < ce) ? logf(fmaxf(gate[(tok0 + t0) * gb_stride + h_v], kGateFloor)) : 0.0f;
        g_smem[t1] = (t1 < ce) ? logf(fmaxf(gate[(tok0 + t1) * gb_stride + h_v], kGateFloor)) : 0.0f;
        __syncwarp();
        // Sequential, in token order, on one thread: Atlas's FLA path proved a
        // re-associated (tree) scan of this sum costs task accuracy (the note at
        // gated_delta_rule_recompute_wu). Pad rows add 0, so they hold the last
        // real value: no extra decay, and a defined exp() everywhere.
        if (lane == 0) {
            float acc = 0.0f;
            for (int i = 0; i < BT; ++i) {
                acc += g_smem[i];
                g_smem[i] = acc;
            }
        }
        __syncwarp();
        g_out[blk * BT + t0] = g_smem[t0];
        g_out[blk * BT + t1] = g_smem[t1];
    }

    // === Phase WY-B: native BF16 K K^T, one 64 x 32 K panel at a time ===
    float A_reg[N_SUB][8];
#pragma unroll
    for (int j = 0; j < N_SUB; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) A_reg[j][e] = 0.0f;
    }
    const __nv_bfloat16* const k_rows = k_in + tok0 * qk_stride + static_cast<std::int64_t>(h_qk) * kStateDim;
    constexpr int K_VECS_PER_ROW = K_PANEL_COLS / 8;
    constexpr int K_STAGE_VECS = BT * K_VECS_PER_ROW;
#pragma unroll
    for (int kp = 0; kp < N_K_PANELS; ++kp) {
        const int panel_col = kp * K_PANEL_COLS;
#pragma unroll
        for (int v = tid; v < K_STAGE_VECS; v += BLOCK_THREADS) {
            const int row = v / K_VECS_PER_ROW;
            const int col8 = (v - row * K_VECS_PER_ROW) * 8;
            const bool ok = row < ce;
            const int src_row = ok ? row : 0;
            cp_async16_cg(K_view.ptr(row, col8),
                          k_rows + static_cast<std::int64_t>(src_row) * qk_stride + panel_col + col8,
                          ok ? 16 : 0);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        switch (warp) {
        case 0: kkt_strip<1>(A_reg, K_view, warp, lane); break;
        case 1: kkt_strip<2>(A_reg, K_view, warp, lane); break;
        case 2: kkt_strip<3>(A_reg, K_view, warp, lane); break;
        case 3: kkt_strip<4>(A_reg, K_view, warp, lane); break;
        default: break;
        }
        if (kp + 1 < N_K_PANELS) { __syncthreads(); }
    }

    // === Phase WY-C: -beta_r exp(g_r - g_c) K K^T, strict lower on the diagonal ===
    // The select (not a 0/1 mask) keeps an overflowing exp on the strict upper
    // triangle from producing inf * 0 = NaN.
    if (warp < WY_WARPS) {
        const int r_g0 = warp * BC + lane_g;
        const int r_g1 = r_g0 + 8;
        const float nbeta_r0 = -beta_smem[r_g0];
        const float nbeta_r1 = -beta_smem[r_g1];
        const float g_r0 = g_smem[r_g0];
        const float g_r1 = g_smem[r_g1];
#pragma unroll
        for (int j_sub = 0; j_sub < N_SUB; ++j_sub) {
            if (j_sub > warp) continue;
            const int c0 = j_sub * BC + 2 * lane_t;
            const int c1 = c0 + 1;
            const int c2 = c0 + 8;
            const int c3 = c0 + 9;
            const float g_c0 = g_smem[c0];
            const float g_c1 = g_smem[c1];
            const float g_c2 = g_smem[c2];
            const float g_c3 = g_smem[c3];
            const bool is_diag = (j_sub == warp);
            A_reg[j_sub][0] = (!is_diag || r_g0 > c0) ? nbeta_r0 * A_reg[j_sub][0] * expf(g_r0 - g_c0) : 0.0f;
            A_reg[j_sub][1] = (!is_diag || r_g0 > c1) ? nbeta_r0 * A_reg[j_sub][1] * expf(g_r0 - g_c1) : 0.0f;
            A_reg[j_sub][2] = (!is_diag || r_g1 > c0) ? nbeta_r1 * A_reg[j_sub][2] * expf(g_r1 - g_c0) : 0.0f;
            A_reg[j_sub][3] = (!is_diag || r_g1 > c1) ? nbeta_r1 * A_reg[j_sub][3] * expf(g_r1 - g_c1) : 0.0f;
            A_reg[j_sub][4] = (!is_diag || r_g0 > c2) ? nbeta_r0 * A_reg[j_sub][4] * expf(g_r0 - g_c2) : 0.0f;
            A_reg[j_sub][5] = (!is_diag || r_g0 > c3) ? nbeta_r0 * A_reg[j_sub][5] * expf(g_r0 - g_c3) : 0.0f;
            A_reg[j_sub][6] = (!is_diag || r_g1 > c2) ? nbeta_r1 * A_reg[j_sub][6] * expf(g_r1 - g_c2) : 0.0f;
            A_reg[j_sub][7] = (!is_diag || r_g1 > c3) ? nbeta_r1 * A_reg[j_sub][7] * expf(g_r1 - g_c3) : 0.0f;
        }
    }
    __syncthreads();

    for (int idx = tid; idx < T_INV_FLOATS; idx += BLOCK_THREADS) T_inv_smem[idx] = 0.0f;
    __syncthreads();

    switch (warp) {
    case 0: store_frag_to_M(A_reg[0], 0, 0, lane_g, lane_t, M_view); break;
    case 1: store_frag_to_M(A_reg[1], 1, 1, lane_g, lane_t, M_view); break;
    case 2: store_frag_to_M(A_reg[2], 2, 2, lane_g, lane_t, M_view); break;
    case 3: store_frag_to_M(A_reg[3], 3, 3, lane_g, lane_t, M_view); break;
    default: break;
    }
    __syncwarp();
    switch (warp) {
    case 0: solve_diag_block<0>(lane, M_view); break;
    case 1: solve_diag_block<1>(lane, M_view); break;
    case 2: solve_diag_block<2>(lane, M_view); break;
    case 3: solve_diag_block<3>(lane, M_view); break;
    default: break;
    }
    __syncthreads();

    // === Phase WY-D: block-Schur off-diagonal completion (3 waves) ===
    switch (warp) {
    case 1: wave_compute_store<1, 0>(A_reg, scr_smem, M_view, warp, lane); break;
    case 2: wave_compute_store<2, 1>(A_reg, scr_smem, M_view, warp, lane); break;
    case 3: wave_compute_store<3, 2>(A_reg, scr_smem, M_view, warp, lane); break;
    default: break;
    }
    __syncthreads();
    switch (warp) {
    case 2: wave_compute_store<2, 0>(A_reg, scr_smem, M_view, warp, lane); break;
    case 3: wave_compute_store<3, 1>(A_reg, scr_smem, M_view, warp, lane); break;
    default: break;
    }
    __syncthreads();
    if (warp == 3) { wave_compute_store<3, 0>(A_reg, scr_smem, M_view, warp, lane); }
    __syncthreads();

    // === Phase WY-E: + I on the diagonal; beta exp(g) row scale for W ===
    if (tid < BT) {
        M_view.at(tid, tid) += 1.0f;
        bg_smem[tid] = beta_smem[tid] * expf(g_smem[tid]);
    }
    __syncthreads();
    // Atlas change: T_inv is an A operand of every W/U product; round it once.
    for (int idx = tid; idx < T_INV_FLOATS; idx += BLOCK_THREADS) {
        T_inv_smem[idx] = tf32_round(T_inv_smem[idx]);
    }

    // === Phase WU-A/B: U = T_inv @ (beta V), one 64 x 16 panel at a time ===
    const __nv_bfloat16* const v_rows = v_in + tok0 * v_stride + static_cast<std::int64_t>(h_v) * kStateDim;
    __nv_bfloat16* const U_blk = U + blk * BT * kStateDim;
    __nv_bfloat16* const W_blk = W + blk * BT * kStateDim;
#pragma unroll 1
    for (int panel = 0; panel < N_WU_PANELS; ++panel) {
        __syncthreads();
        const int panel_col = panel * WU_PANEL_COLS;
        load_scaled_wu_panel(WU_view, v_rows, v_stride, beta_smem, panel_col, ce, tid);
        __syncthreads();
        compute_store_wu_panel<false>(M_view, WU_view, output_smem, U_blk, panel_col, warp, lane);
    }
    // === Phase WU-C/D: W = T_inv @ (beta exp(g) K), interleaved for stage 2 ===
#pragma unroll 1
    for (int panel = 0; panel < N_WU_PANELS; ++panel) {
        __syncthreads();
        const int panel_col = panel * WU_PANEL_COLS;
        load_scaled_wu_panel(WU_view, k_rows, qk_stride, bg_smem, panel_col, ce, tid);
        __syncthreads();
        compute_store_wu_panel<true>(M_view, WU_view, output_smem, W_blk, panel_col, warp, lane);
    }
}

}  // namespace prep

// ═══ Stage 2: state passing (NInfer chunked/state_passing.cuh) ═════════════

namespace sp {

constexpr int N_STRIP = 16;                         // value columns per CTA
constexpr int D_STRIPS = kStateDim / N_STRIP;       // 8
constexpr int DT_TILES = N_STRIP / MMA_N;           // 2
constexpr int BT_SPLITS = 4;
constexpr int N_WARPS = DT_TILES * BT_SPLITS;       // 8
constexpr int THREADS = N_WARPS * kWarpSize;        // 256
constexpr int BT_PER_WARP = BT / BT_SPLITS;         // 16
constexpr int S_PER_WARP = kStateDim / BT_SPLITS;   // 32
constexpr int M_TILES_MM1_PW = BT_PER_WARP / MMA_M; // 1
constexpr int M_TILES_H_PW = S_PER_WARP / MMA_M;    // 2
constexpr int M_TILES_H_GLOB = kStateDim / MMA_M;   // 8
constexpr int K_TILES_MM2 = BT / MMA_K;             // 8
constexpr int W_STORAGE_FLT = BT * kStateDim / 2;
constexpr int UVD_FLT = BT * N_STRIP;
constexpr int K_STORAGE_FLT = BT * kStateDim / 2;
constexpr int SNAP_K_ROWS = MMA_M * M_TILES_H_PW;   // 32
constexpr int SNAP_FLT = SNAP_K_ROWS * N_STRIP;     // 512
constexpr int N_SNAP_ITERS = M_TILES_H_GLOB / M_TILES_H_PW; // 4
constexpr int K_TILES_PER_SNAP_ITER = SNAP_K_ROWS / MMA_K;  // 4
constexpr int SMEM_FLOATS = W_STORAGE_FLT + UVD_FLT + K_STORAGE_FLT + SNAP_FLT * N_SNAP_ITERS + BT;

static_assert(M_TILES_H_GLOB == BT_SPLITS * M_TILES_H_PW, "state rows split exactly");
static_assert(N_SNAP_ITERS == BT_SPLITS, "one snapshot buffer per state split");
static_assert(SMEM_FLOATS == 11328, "mirrored by STATE_SMEM in gdn_chunk_tc_ada.rs");

struct Tiles {
    float f[SMEM_FLOATS];
};

// FP32 state snapshot as [value, state] with a stride-32 swizzle shared by the
// scatter, the h_chunk write and the MM1 ldmatrix read.
struct SnapView {
    float* __restrict__ base;
    static constexpr int kStride = kStateDim / 4;
    __device__ __forceinline__ int swz_xor(int row) const { return (row & 7) << 2; }
    __device__ __forceinline__ float& at(int row, int col) const {
        return base[row * kStride + (col ^ swz_xor(row))];
    }
    __device__ __forceinline__ float4& vec4_at(int row, int col) const {
        return *reinterpret_cast<float4*>(&base[row * kStride + (col ^ swz_xor(row))]);
    }
};
static_assert(SnapView::kStride == 32, "snapshot stride");

// W arrives in stage 1's {0,4,1,5,2,6,3,7} column order.
struct Bf16WView {
    __nv_bfloat16* __restrict__ base;
    __device__ __forceinline__ int swizzled_col(int row, int col) const {
        return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
    }
    __device__ __forceinline__ __nv_bfloat16* ptr(int row, int col) const {
        return &base[row * kStateDim + swizzled_col(row, col)];
    }
};

// K in native BF16; within each group of eight token rows the physical order
// is {0,4,1,5,2,6,3,7}, so ldmatrix.x2.trans packs token columns {t, t+4} into
// the two halves of each register, the TF32 A fragment.
struct Bf16KView {
    __nv_bfloat16* __restrict__ base;
    __device__ __forceinline__ int physical_row(int logical_row) const {
        const int row8 = logical_row & 7;
        return (logical_row & ~7) + ((row8 & 3) << 1) + (row8 >> 2);
    }
    __device__ __forceinline__ int swizzled_col(int physical_row_, int col) const {
        return (col & ~63) + ((((col & 63) >> 3) ^ (physical_row_ & 7)) << 3) + (col & 7);
    }
    __device__ __forceinline__ __nv_bfloat16* logical_ptr(int row, int col) const {
        const int row_phys = physical_row(row);
        return &base[row_phys * kStateDim + swizzled_col(row_phys, col)];
    }
    __device__ __forceinline__ __nv_bfloat16* physical_ptr(int row, int col) const {
        return &base[row * kStateDim + swizzled_col(row, col)];
    }
};

__device__ __forceinline__ void issue_load_w_bf16(Bf16WView view, const __nv_bfloat16* gmem_row0, int tid) {
    constexpr int COPIES_PER_ROW = kStateDim / 8;
    constexpr int COPIES = BT * COPIES_PER_ROW;
#pragma unroll
    for (int copy = tid; copy < COPIES; copy += THREADS) {
        const int row = copy / COPIES_PER_ROW;
        const int col = (copy - row * COPIES_PER_ROW) * 8;
        cp_async16_ca(view.ptr(row, col), gmem_row0 + static_cast<std::int64_t>(row) * kStateDim + col, 16);
    }
}

// K rows from the packed input; rows >= valid are zero-filled (Atlas change).
__device__ __forceinline__ void issue_load_k_bf16(Bf16KView view, const __nv_bfloat16* gmem_row0,
                                                  std::int64_t gmem_row_stride, int valid, int tid) {
    constexpr int COPIES_PER_ROW = kStateDim / 8;
    constexpr int COPIES = BT * COPIES_PER_ROW;
#pragma unroll
    for (int copy = tid; copy < COPIES; copy += THREADS) {
        const int row = copy / COPIES_PER_ROW;
        const int col = (copy - row * COPIES_PER_ROW) * 8;
        const bool ok = row < valid;
        const int src_row = ok ? row : 0;
        cp_async16_ca(view.logical_ptr(row, col),
                      gmem_row0 + static_cast<std::int64_t>(src_row) * gmem_row_stride + col, ok ? 16 : 0);
    }
}

__device__ __forceinline__ void unpack_bf16x2_to_fp32_bits(unsigned packed, unsigned& low, unsigned& high) {
    low = packed << 16;
    high = packed & 0xffff0000U;
}

__device__ __forceinline__ void state_body(float* h_state, const __nv_bfloat16* __restrict__ W_in,
                                           const __nv_bfloat16* __restrict__ U_in,
                                           const __nv_bfloat16* __restrict__ k_in,
                                           const float* __restrict__ g_in,
                                           __nv_bfloat16* __restrict__ v_new,
                                           __nv_bfloat16* __restrict__ h_chunk, unsigned seq_len,
                                           unsigned num_chunks, unsigned num_k_heads,
                                           unsigned num_v_heads, unsigned qk_stride,
                                           unsigned h_state_is_table, float* smem) {
    auto* const W_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    float* const uvd_smem = smem + W_STORAGE_FLT;
    auto* const k_smem = reinterpret_cast<__nv_bfloat16*>(uvd_smem + UVD_FLT);
    float* const snap_smem = uvd_smem + UVD_FLT + K_STORAGE_FLT;
    float* const g_smem = snap_smem + SNAP_FLT * N_SNAP_ITERS;

    Bf16WView W_view{W_smem};
    SmemTile<N_STRIP> vd_view{uvd_smem};   // U and v_decay alias in disjoint phases
    SmemTile<N_STRIP> U_view{uvd_smem};
    Bf16KView k_view{k_smem};

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    const int s_idx = warp % BT_SPLITS;
    const int dt_idx = warp / BT_SPLITS;
    const int warp_d_local = dt_idx * MMA_N;

    const int hd = static_cast<int>(blockIdx.x);
    const int h_v = hd / D_STRIPS;
    const int strip_idx = hd - h_v * D_STRIPS;
    const int d_off = strip_idx * N_STRIP;
    const int warp_d_global = d_off + warp_d_local;
    const int b = static_cast<int>(blockIdx.y);
    if (h_v >= static_cast<int>(num_v_heads)) return;
    const int h_qk = h_v / static_cast<int>(num_v_heads / num_k_heads);

    constexpr std::int64_t kHeadElems = static_cast<std::int64_t>(kStateDim) * kStateDim;
    float* const H = h_state_is_table
                         ? reinterpret_cast<float* const*>(h_state)[b] + static_cast<std::int64_t>(h_v) * kHeadElems
                         : h_state + (static_cast<std::int64_t>(b) * num_v_heads + h_v) * kHeadElems;

    // === Phase 0: FP32 state [k][v] -> per-warp accumulator fragments ===
    float h_frag[M_TILES_H_PW][4];
    {
        const int row_off = s_idx * S_PER_WARP;
        const int col_d0 = warp_d_global + 2 * lane_t;
#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
            const int row_g0 = row_off + m * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const float2 top = load_vec<float2>(H + static_cast<std::int64_t>(row_g0) * kStateDim + col_d0);
            const float2 bot = load_vec<float2>(H + static_cast<std::int64_t>(row_g1) * kStateDim + col_d0);
            h_frag[m][0] = top.x;
            h_frag[m][1] = top.y;
            h_frag[m][2] = bot.x;
            h_frag[m][3] = bot.y;
        }
    }

    // Blocked workspace: block (b, c, h_v) = (b * num_chunks + c) * H_v + h_v.
    const std::int64_t blk0 = static_cast<std::int64_t>(b) * num_chunks * num_v_heads + h_v;
    const std::int64_t k_tok0 = static_cast<std::int64_t>(b) * seq_len;
    const __nv_bfloat16* const k_head = k_in + k_tok0 * qk_stride + static_cast<std::int64_t>(h_qk) * kStateDim;
    const std::int64_t k_chunk_stride = static_cast<std::int64_t>(BT) * qk_stride;

    // Chunk 0 staged before the loop: W first (MM1), then K (MM2), U synchronously.
    issue_load_w_bf16(W_view, W_in + blk0 * BT * kStateDim, tid);
    cp_commit();
    issue_load_k_bf16(k_view, k_head, qk_stride, valid_rows(seq_len, 0), tid);
    cp_commit();
    issue_load_bf16_to_float_vec4<BT, N_STRIP, THREADS>(U_view, U_in + blk0 * BT * kStateDim + d_off,
                                                        kStateDim, tid);

    for (int chunk = 0; chunk < static_cast<int>(num_chunks); ++chunk) {
        const std::int64_t blk = blk0 + static_cast<std::int64_t>(chunk) * num_v_heads;

        // === Phase A: g, and every warp scatters its state fragments ===
        if (tid < BT) { g_smem[tid] = g_in[blk * BT + tid]; }
        {
            SnapView snap{snap_smem + s_idx * SNAP_FLT};
#pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int k_g0 = m * MMA_M + lane_g;
                const int k_g1 = k_g0 + 8;
                const int d0 = warp_d_local + 2 * lane_t;
                const int d1 = d0 + 1;
                snap.at(d0, k_g0) = h_frag[m][0];
                snap.at(d1, k_g0) = h_frag[m][1];
                snap.at(d0, k_g1) = h_frag[m][2];
                snap.at(d1, k_g1) = h_frag[m][3];
            }
        }
        cp_wait<1>();     // W landed; K may still be in flight
        __syncthreads();  // W/U/g and every snapshot visible

        // === Phase B: bf16(S_c) to h_chunk, and MM1 = W S_c ===
        const std::int64_t hc_base = blk * kHeadElems;
        float vnew_frag[M_TILES_MM1_PW][4] = {};
#pragma unroll
        for (int sit = 0; sit < N_SNAP_ITERS; ++sit) {
            const int k_row_off = sit * SNAP_K_ROWS;
            SnapView snap{snap_smem + sit * SNAP_FLT};
            {
                constexpr int K_VEC_PER_D = SNAP_K_ROWS / 4;
                constexpr int N_VEC_SNAP = SNAP_FLT / 4;
#pragma unroll
                for (int v = tid; v < N_VEC_SNAP; v += THREADS) {
                    const int d_local = v / K_VEC_PER_D;
                    const int k_off = (v - d_local * K_VEC_PER_D) * 4;
                    const float4 val = snap.vec4_at(d_local, k_off);
                    __nv_bfloat16* out =
                        &h_chunk[hc_base + static_cast<std::int64_t>(d_off + d_local) * kStateDim + k_row_off + k_off];
                    store_vec(out, __floats2bfloat162_rn(val.x, val.y));
                    store_vec(out + 2, __floats2bfloat162_rn(val.z, val.w));
                }
            }
            const int lane_in_8 = lane & 7;
            const int matrix_half = (lane >> 3) & 1;
#pragma unroll
            for (int kt = 0; kt < K_TILES_PER_SNAP_ITER; ++kt) {
                const int W_k_local = k_row_off + kt * MMA_K;
                const int snap_k = kt * MMA_K;
                unsigned ub0, ub1;
                ldmatrix_x2(ub0, ub1, smem_addr(&snap.at(warp_d_local + lane_in_8, snap_k + matrix_half * 4)));
                ub0 = tf32_round_bits(ub0);
                ub1 = tf32_round_bits(ub1);
#pragma unroll
                for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
                    const int row_base = s_idx * BT_PER_WARP + m_mm1 * MMA_M;
                    const int a_row = row_base + matrix_half * 8 + lane_in_8;
                    unsigned packed0, packed1;
                    ldmatrix_x2(packed0, packed1, smem_addr(W_view.ptr(a_row, W_k_local)));
                    unsigned ua0, ua1, ua2, ua3;
                    unpack_bf16x2_to_fp32_bits(packed0, ua0, ua2);
                    unpack_bf16x2_to_fp32_bits(packed1, ua1, ua3);
                    mma_tf32_bits(vnew_frag[m_mm1][0], vnew_frag[m_mm1][1], vnew_frag[m_mm1][2],
                                  vnew_frag[m_mm1][3], ua0, ua1, ua2, ua3, ub0, ub1);
                }
            }
        }

        // === Phase C: v_new = U - W S_c ===
#pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_local + 2 * lane_t;
            const float2 u_top = load_vec<float2>(&U_view.at(row_g0, col_d0));
            const float2 u_bot = load_vec<float2>(&U_view.at(row_g1, col_d0));
            vnew_frag[m_mm1][0] = u_top.x - vnew_frag[m_mm1][0];
            vnew_frag[m_mm1][1] = u_top.y - vnew_frag[m_mm1][1];
            vnew_frag[m_mm1][2] = u_bot.x - vnew_frag[m_mm1][2];
            vnew_frag[m_mm1][3] = u_bot.y - vnew_frag[m_mm1][3];
        }

        // === Phase D: store v_new (undecayed), v_decay -> vd_view, decay S ===
        const float g_C = g_smem[BT - 1];
        float gamma_C = 0.0f;
        if (lane == 0) { gamma_C = expf(g_C); }
        gamma_C = __shfl_sync(0xffffffffU, gamma_C, 0);
        const std::int64_t vn_base = blk * BT * kStateDim;
#pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            float dec_top = 0.0f;
            float dec_bot = 0.0f;
            if (lane_t == 0) {
                dec_top = expf(g_C - g_smem[row_g0]);
                dec_bot = expf(g_C - g_smem[row_g1]);
            }
            const int decay_src_lane = lane & ~3;
            dec_top = __shfl_sync(0xffffffffU, dec_top, decay_src_lane);
            dec_bot = __shfl_sync(0xffffffffU, dec_bot, decay_src_lane);
            const float v0 = vnew_frag[m_mm1][0];
            const float v1 = vnew_frag[m_mm1][1];
            const float v2 = vnew_frag[m_mm1][2];
            const float v3 = vnew_frag[m_mm1][3];
            const int col_g = warp_d_global + 2 * lane_t;
            store_vec(&v_new[vn_base + static_cast<std::int64_t>(row_g0) * kStateDim + col_g],
                      __floats2bfloat162_rn(v0, v1));
            store_vec(&v_new[vn_base + static_cast<std::int64_t>(row_g1) * kStateDim + col_g],
                      __floats2bfloat162_rn(v2, v3));
            const int col_l = warp_d_local + 2 * lane_t;
            store_vec(&vd_view.at(row_g0, col_l),
                      make_float2(tf32_round(v0 * dec_top), tf32_round(v1 * dec_top)));
            store_vec(&vd_view.at(row_g1, col_l),
                      make_float2(tf32_round(v2 * dec_bot), tf32_round(v3 * dec_bot)));
        }
#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { h_frag[m][e] *= gamma_C; }
        }

        cp_wait<0>();     // K landed
        __syncthreads();  // K and v_decay visible to MM2

        // === Phase E: S += K^T v_decay over the whole chunk ===
#pragma unroll
        for (int kt = 0; kt < K_TILES_MM2; ++kt) {
            const int k_off_local = kt * MMA_K;
            const int col_g = warp_d_local + lane_g;
            const float b0 = vd_view.at(k_off_local + lane_t, col_g);
            const float b1 = vd_view.at(k_off_local + lane_t + 4, col_g);
#pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int col_a_base = s_idx * S_PER_WARP + m * MMA_M;
                const int a_row = k_off_local + (lane & 7);
                const int a_col = col_a_base + ((lane >> 3) & 1) * 8;
                unsigned packed0, packed1;
                ldmatrix_x2_t(packed0, packed1, smem_addr(k_view.physical_ptr(a_row, a_col)));
                unsigned ua0, ua1, ua2, ua3;
                unpack_bf16x2_to_fp32_bits(packed0, ua0, ua2);
                unpack_bf16x2_to_fp32_bits(packed1, ua1, ua3);
                mma_tf32_bits(h_frag[m][0], h_frag[m][1], h_frag[m][2], h_frag[m][3], ua0, ua1, ua2,
                              ua3, __float_as_uint(b0), __float_as_uint(b1));
            }
        }
        __syncthreads();  // MM2 done before the next chunk overwrites W/K/U

        if (chunk + 1 < static_cast<int>(num_chunks)) {
            const std::int64_t blk_next = blk + num_v_heads;
            issue_load_w_bf16(W_view, W_in + blk_next * BT * kStateDim, tid);
            cp_commit();
            issue_load_k_bf16(k_view, k_head + static_cast<std::int64_t>(chunk + 1) * k_chunk_stride, qk_stride,
                              valid_rows(seq_len, chunk + 1), tid);
            cp_commit();
            issue_load_bf16_to_float_vec4<BT, N_STRIP, THREADS>(
                U_view, U_in + blk_next * BT * kStateDim + d_off, kStateDim, tid);
        }
    }

    // === Phase Z: accumulator fragments -> FP32 state [k][v], in place ===
    {
        const int col_d0 = warp_d_global + 2 * lane_t;
#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
            const int k_g0 = s_idx * S_PER_WARP + m * MMA_M + lane_g;
            const int k_g1 = k_g0 + 8;
            store_vec(H + static_cast<std::int64_t>(k_g0) * kStateDim + col_d0, make_float2(h_frag[m][0], h_frag[m][1]));
            store_vec(H + static_cast<std::int64_t>(k_g1) * kStateDim + col_d0, make_float2(h_frag[m][2], h_frag[m][3]));
        }
    }
}

}  // namespace sp

// ═══ Stage 3: chunk output (NInfer chunked/output.cuh) ══════════════════════

namespace out {

constexpr int N_WARPS = 4;
constexpr int THREADS = N_WARPS * kWarpSize;
constexpr int K_PANEL = 32;
constexpr int D_PANEL = 16;
constexpr int N_K_PANELS = kStateDim / K_PANEL;
constexpr int N_D_PANELS = kStateDim / D_PANEL;
constexpr int N_TILES_BT = BT / MMA_N;
constexpr int K_TILES_BT = BT / MMA_K;
constexpr int Q_BF16 = BT * kStateDim;
constexpr int K_PANEL_BF16 = BT * K_PANEL;
constexpr int H_PANEL_BF16 = D_PANEL * kStateDim;
constexpr int V_PANEL_BF16 = BT * D_PANEL;
constexpr int STAGE_BF16 = K_PANEL_BF16 > H_PANEL_BF16 ? K_PANEL_BF16 : H_PANEL_BF16;
constexpr int BF16_SMEM_ELEMS = Q_BF16 + 2 * STAGE_BF16;

static_assert(BT == N_WARPS * MMA_M, "one 16-row strip per warp");
static_assert(STAGE_BF16 >= V_PANEL_BF16, "V panel fits a stage");
static_assert(BF16_SMEM_ELEMS * 2 == 24576, "mirrored by OUTPUT_SMEM in gdn_chunk_tc_ada.rs");

struct Tiles {
    __nv_bfloat16 h[BF16_SMEM_ELEMS];
};

// ROWS x COLS BF16 rows -> swizzled tile; rows >= valid zero-filled.
template <int ROWS, int COLS>
__device__ __forceinline__ void issue_cp_bf16(Bf16Tile<COLS> dst, const __nv_bfloat16* __restrict__ src_row0,
                                              std::int64_t src_row_stride, int valid, int tid) {
    static_assert(COLS % 8 == 0, "16-byte copies");
    constexpr int VECS_PER_ROW = COLS / 8;
    constexpr int N_VECS = ROWS * VECS_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VECS; v += THREADS) {
        const int row = v / VECS_PER_ROW;
        const int col8 = (v - row * VECS_PER_ROW) * 8;
        const bool ok = row < valid;
        const int src_row = ok ? row : 0;
        cp_async16_cg(dst.ptr(row, col8), src_row0 + static_cast<std::int64_t>(src_row) * src_row_stride + col8,
                      ok ? 16 : 0);
    }
}

template <int N_TILES, int K_TILES, int A_STRIDE, int B_STRIDE>
__device__ __forceinline__ void mma_bf16_panel(float (&D)[N_TILES][4], Bf16Tile<A_STRIDE> A,
                                               Bf16Tile<B_STRIDE> B, int a_row_base, int a_col_base,
                                               int b_col_base, int lane) {
    const int lane_in_8 = lane & 7;
    const int lane_q = lane >> 3;
    const int a_row = a_row_base + lane_in_8 + ((lane_q & 1) << 3);
    const int a_col = a_col_base + ((lane_q >> 1) << 3);
    const int b_col = b_col_base + ((lane_q & 1) << 3);
#pragma unroll
    for (int kt = 0; kt < K_TILES; ++kt) {
        const int k_off = kt * BF16_MMA_K;
        unsigned af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(A.ptr(a_row, a_col + k_off)));
#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            unsigned bf[2];
            ldmatrix_x2(bf[0], bf[1], smem_addr(B.ptr(nt * MMA_N + lane_in_8, b_col + k_off)));
            mma_bf16(D[nt][0], D[nt][1], D[nt][2], D[nt][3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }
    }
}

template <int N_TILES>
__device__ __forceinline__ void mma_av_panel(float (&D)[N_TILES][4], const float (&A_a)[K_TILES_BT][4],
                                             Bf16Tile<D_PANEL> V, int lane_g, int lane_t) {
#pragma unroll
    for (int kt = 0; kt < K_TILES_BT; ++kt) {
        const int k_off = kt * MMA_K;
#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            const int col = nt * MMA_N + lane_g;
            const float b0 = __bfloat162float(*V.ptr(k_off + lane_t, col));
            const float b1 = __bfloat162float(*V.ptr(k_off + lane_t + 4, col));
            mma_tf32(D[nt][0], D[nt][1], D[nt][2], D[nt][3], A_a[kt][0], A_a[kt][1], A_a[kt][2],
                     A_a[kt][3], b0, b1);
        }
    }
}

__device__ __forceinline__ void output_job(const __nv_bfloat16* __restrict__ q_in,
                                           const __nv_bfloat16* __restrict__ k_in,
                                           const __nv_bfloat16* __restrict__ v_new_in,
                                           const float* __restrict__ g_in,
                                           const __nv_bfloat16* __restrict__ h_chunk_in,
                                           __nv_bfloat16* __restrict__ attn_out, int chunk, int h_v,
                                           int b, unsigned seq_len, unsigned num_chunks,
                                           unsigned num_k_heads, unsigned num_v_heads,
                                           unsigned qk_stride, float* smem) {
    auto* const bf16_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    auto* const q_smem = bf16_smem;
    auto* const stage0 = q_smem + Q_BF16;
    auto* const stage1 = stage0 + STAGE_BF16;
    // g lives in stage0's tail once the last K panel moved to stage1.
    float* const g_smem = reinterpret_cast<float*>(stage0 + V_PANEL_BF16);

    Bf16Tile<kStateDim> q_view{q_smem};
    Bf16Tile<K_PANEL> k_stage0{stage0};
    Bf16Tile<K_PANEL> k_stage1{stage1};
    Bf16Tile<kStateDim> h_view{stage0};
    Bf16Tile<D_PANEL> v_view{stage1};

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const int ce = valid_rows(seq_len, chunk);
    const int h_qk = h_v / static_cast<int>(num_v_heads / num_k_heads);
    const std::int64_t tok0 = static_cast<std::int64_t>(b) * seq_len + static_cast<std::int64_t>(chunk) * BT;
    const std::int64_t blk = (static_cast<std::int64_t>(b) * num_chunks + chunk) * num_v_heads + h_v;
    const __nv_bfloat16* const q_rows = q_in + tok0 * qk_stride + static_cast<std::int64_t>(h_qk) * kStateDim;
    const __nv_bfloat16* const k_rows = k_in + tok0 * qk_stride + static_cast<std::int64_t>(h_qk) * kStateDim;
    const __nv_bfloat16* const vn_blk = v_new_in + blk * BT * kStateDim;
    const __nv_bfloat16* const hc_blk = h_chunk_in + blk * kStateDim * kStateDim;
    const std::int64_t value_row_stride = static_cast<std::int64_t>(num_v_heads) * kStateDim;
    __nv_bfloat16* const out_rows = attn_out + tok0 * value_row_stride + static_cast<std::int64_t>(h_v) * kStateDim;
    const float scale = rsqrtf(static_cast<float>(kStateDim));

    // Q is resident; K streams through two 64 x 32 panels.
    issue_cp_bf16<BT, kStateDim>(q_view, q_rows, qk_stride, ce, tid);
    issue_cp_bf16<BT, K_PANEL>(k_stage0, k_rows, qk_stride, ce, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    float A_strip[N_TILES_BT][4] = {};
#pragma unroll
    for (int panel = 0; panel < N_K_PANELS; ++panel) {
        Bf16Tile<K_PANEL> current = (panel & 1) == 0 ? k_stage0 : k_stage1;
        if (panel + 1 < N_K_PANELS) {
            Bf16Tile<K_PANEL> next = (panel & 1) == 0 ? k_stage1 : k_stage0;
            issue_cp_bf16<BT, K_PANEL>(next, k_rows + static_cast<std::int64_t>(panel + 1) * K_PANEL, qk_stride,
                                       ce, tid);
            cp_commit();
        } else if (tid < BT) {
            g_smem[tid] = g_in[blk * BT + tid];
        }
        mma_bf16_panel<N_TILES_BT, K_PANEL / BF16_MMA_K>(A_strip, q_view, current, warp * MMA_M,
                                                         panel * K_PANEL, 0, lane);
        if (panel + 1 < N_K_PANELS) {
            cp_wait<0>();
            __syncthreads();
        }
    }
    __syncthreads();

    // Causal decay in FP32; the select replaces an overflowing upper product.
    const int row_g0 = warp * MMA_M + lane_g;
    const int row_g1 = row_g0 + 8;
    const float g_r0 = g_smem[row_g0];
    const float g_r1 = g_smem[row_g1];
    const float gamma0 = expf(g_r0);
    const float gamma1 = expf(g_r1);
#pragma unroll
    for (int nt = 0; nt < N_TILES_BT; ++nt) {
        const int s0 = nt * MMA_N + 2 * lane_t;
        const int s1 = s0 + 1;
        const float g_s0 = g_smem[s0];
        const float g_s1 = g_smem[s1];
        A_strip[nt][0] = (s0 <= row_g0) ? A_strip[nt][0] * expf(g_r0 - g_s0) : 0.0f;
        A_strip[nt][1] = (s1 <= row_g0) ? A_strip[nt][1] * expf(g_r0 - g_s1) : 0.0f;
        A_strip[nt][2] = (s0 <= row_g1) ? A_strip[nt][2] * expf(g_r1 - g_s0) : 0.0f;
        A_strip[nt][3] = (s1 <= row_g1) ? A_strip[nt][3] * expf(g_r1 - g_s1) : 0.0f;
    }

    // m16n8 accumulator layout -> TF32 A-operand layout for A @ V (rounded once).
    float A_a[K_TILES_BT][4];
    {
        const int src_lo = (lane_g << 2) | (lane_t >> 1);
        const int src_hi = src_lo + 2;
        const bool t_odd = (lane_t & 1) != 0;
        constexpr unsigned mask = 0xFFFFFFFFu;
#pragma unroll
        for (int kt = 0; kt < K_TILES_BT; ++kt) {
            const float d0_lo = __shfl_sync(mask, A_strip[kt][0], src_lo);
            const float d1_lo = __shfl_sync(mask, A_strip[kt][1], src_lo);
            const float d2_lo = __shfl_sync(mask, A_strip[kt][2], src_lo);
            const float d3_lo = __shfl_sync(mask, A_strip[kt][3], src_lo);
            const float d0_hi = __shfl_sync(mask, A_strip[kt][0], src_hi);
            const float d1_hi = __shfl_sync(mask, A_strip[kt][1], src_hi);
            const float d2_hi = __shfl_sync(mask, A_strip[kt][2], src_hi);
            const float d3_hi = __shfl_sync(mask, A_strip[kt][3], src_hi);
            A_a[kt][0] = tf32_round(t_odd ? d1_lo : d0_lo);
            A_a[kt][1] = tf32_round(t_odd ? d3_lo : d2_lo);
            A_a[kt][2] = tf32_round(t_odd ? d1_hi : d0_hi);
            A_a[kt][3] = tf32_round(t_odd ? d3_hi : d2_hi);
        }
    }

    // Every K read finished before stage0 becomes H panel 0.
    __syncthreads();
    issue_cp_bf16<D_PANEL, kStateDim>(h_view, hc_blk, kStateDim, D_PANEL, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll 1
    for (int panel = 0; panel < N_D_PANELS; ++panel) {
        const int d_off = panel * D_PANEL;
        issue_cp_bf16<BT, D_PANEL>(v_view, vn_blk + d_off, kStateDim, BT, tid);
        cp_commit();

        float D_frag[D_PANEL / MMA_N][4] = {};
        mma_bf16_panel<D_PANEL / MMA_N, kStateDim / BF16_MMA_K>(D_frag, q_view, h_view, warp * MMA_M, 0, 0,
                                                               lane);
#pragma unroll
        for (int nt = 0; nt < D_PANEL / MMA_N; ++nt) {
            D_frag[nt][0] *= gamma0;
            D_frag[nt][1] *= gamma0;
            D_frag[nt][2] *= gamma1;
            D_frag[nt][3] *= gamma1;
        }
        cp_wait<0>();
        __syncthreads();
        if (panel + 1 < N_D_PANELS) {
            issue_cp_bf16<D_PANEL, kStateDim>(h_view, hc_blk + static_cast<std::int64_t>(panel + 1) * D_PANEL * kStateDim,
                                              kStateDim, D_PANEL, tid);
            cp_commit();
        }
        mma_av_panel(D_frag, A_a, v_view, lane_g, lane_t);
#pragma unroll
        for (int nt = 0; nt < D_PANEL / MMA_N; ++nt) {
            const int d_global = d_off + nt * MMA_N + 2 * lane_t;
            if (row_g0 < ce) {
                store_vec(&out_rows[static_cast<std::int64_t>(row_g0) * value_row_stride + d_global],
                          __floats2bfloat162_rn(scale * D_frag[nt][0], scale * D_frag[nt][1]));
            }
            if (row_g1 < ce) {
                store_vec(&out_rows[static_cast<std::int64_t>(row_g1) * value_row_stride + d_global],
                          __floats2bfloat162_rn(scale * D_frag[nt][2], scale * D_frag[nt][3]));
            }
        }
        if (panel + 1 < N_D_PANELS) {
            cp_wait<0>();
            __syncthreads();
        }
    }
}

}  // namespace out

}  // namespace gdn_tc_ada

// ═══ Entry points ════════════════════════════════════════════════════════════
//
// The workspace is Atlas's FLA scratch, carved by the host in the FLA order
// (W, U, S, uc, gc), every region blocked by (b * num_chunks + c) * H_v + h_v:
//   w_out, u_out   [blocks][64][128] bf16      (W interleaved, see stage 1)
//   h_chunk        [blocks][128 v][128 k] bf16 (entry state of each chunk)
//   v_new          [blocks][64][128] bf16
//   gc             [blocks][64] f32
// The rows of a partial last chunk past `seq_len` hold zeros (W, U, v_new) and
// the last real cumulative gate (gc).

// grid (num_chunks, num_v_heads, batch), block 128.
extern "C" __global__ void __launch_bounds__(gdn_tc_ada::prep::BLOCK_THREADS)
gdn_chunk_tc_ada_prepare(const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value,
                         const float* __restrict__ gate, const float* __restrict__ beta,
                         __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ u_out,
                         float* __restrict__ gc_out, unsigned int seq_len, unsigned int num_chunks,
                         unsigned int num_k_heads, unsigned int num_v_heads, unsigned int qk_stride,
                         unsigned int v_stride, unsigned int gb_stride) {
    float* smem = reinterpret_cast<float*>(avarok_prefill_smem);
    gdn_tc_ada::prep::prepare_body(key, value, gate, beta, w_out, u_out, gc_out, seq_len, num_chunks,
                                   num_k_heads, num_v_heads, qk_stride, v_stride, gb_stride, smem);
}
AVAROK_EXPORT_DYN_SMEM(gdn_chunk_tc_ada_prepare, gdn_tc_ada::prep::Tiles);

// grid (num_v_heads * 8, batch, 1), block 256. `h_state` is [batch][H_v][128 k][128 v]
// FP32, or with `h_state_is_table` a device array of `batch` pointers to
// [H_v][128][128] each; updated in place.
extern "C" __global__ void __launch_bounds__(gdn_tc_ada::sp::THREADS, 2)
gdn_chunk_tc_ada_state(float* h_state, const __nv_bfloat16* __restrict__ w_in,
                       const __nv_bfloat16* __restrict__ u_in, const __nv_bfloat16* __restrict__ key,
                       const float* __restrict__ gc_in, __nv_bfloat16* __restrict__ v_new_out,
                       __nv_bfloat16* __restrict__ h_chunk_out, unsigned int seq_len,
                       unsigned int num_chunks, unsigned int num_k_heads, unsigned int num_v_heads,
                       unsigned int qk_stride, unsigned int h_state_is_table) {
    float* smem = reinterpret_cast<float*>(avarok_prefill_smem);
    gdn_tc_ada::sp::state_body(h_state, w_in, u_in, key, gc_in, v_new_out, h_chunk_out, seq_len,
                               num_chunks, num_k_heads, num_v_heads, qk_stride, h_state_is_table, smem);
}
AVAROK_EXPORT_DYN_SMEM(gdn_chunk_tc_ada_state, gdn_tc_ada::sp::Tiles);

// grid (grid_chunks, num_v_heads, batch), block 128; each CTA loops over chunks
// blockIdx.x, blockIdx.x + gridDim.x, ... Output rows are [batch * seq_len][H_v][128].
extern "C" __global__ void __launch_bounds__(gdn_tc_ada::out::THREADS, 4)
gdn_chunk_tc_ada_output(const __nv_bfloat16* __restrict__ query, const __nv_bfloat16* __restrict__ key,
                        const __nv_bfloat16* __restrict__ v_new_in, const float* __restrict__ gc_in,
                        const __nv_bfloat16* __restrict__ h_chunk_in, __nv_bfloat16* __restrict__ output,
                        unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
                        unsigned int num_v_heads, unsigned int qk_stride) {
    float* smem = reinterpret_cast<float*>(avarok_prefill_smem);
    const int h_v = static_cast<int>(blockIdx.y);
    const int b = static_cast<int>(blockIdx.z);
    if (h_v >= static_cast<int>(num_v_heads)) return;
    const int stride = static_cast<int>(gridDim.x);
    for (int chunk = static_cast<int>(blockIdx.x); chunk < static_cast<int>(num_chunks); chunk += stride) {
        gdn_tc_ada::out::output_job(query, key, v_new_in, gc_in, h_chunk_in, output, chunk, h_v, b, seq_len,
                                    num_chunks, num_k_heads, num_v_heads, qk_stride, smem);
        if (chunk + stride < static_cast<int>(num_chunks)) { __syncthreads(); }
    }
}
AVAROK_EXPORT_DYN_SMEM(gdn_chunk_tc_ada_output, gdn_tc_ada::out::Tiles);
