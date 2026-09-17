// SPDX-License-Identifier: AGPL-3.0-only
//
// GLM-5.3-Flash KDA (Kimi Delta Attention) CHUNKED PREFILL.
//
// Two cooperating kernels. The verified formulation separates naturally into a phase that
// is independent per chunk and a phase that is sequential across chunks, and forcing them
// into one launch would serialise the parallel half for nothing:
//
//   kda_chunk_prepare   grid=(num_chunks, H)  — per-chunk, fully parallel
//                       cumulative decay, the WY matrix A (build + forward substitution),
//                       and the two A-projected tensors u and w.
//   kda_chunk_scan      grid=(H)              — sequential over chunks, carries the state
//                       inter/intra-chunk output and the recurrent state update.
//
// ─────────────────────────────────────────────────────────────────────────────
// SOURCE-DERIVED EQUATIONS  (HF 5.16.1 chunk_kimi_delta_attention)
// ─────────────────────────────────────────────────────────────────────────────
// Within a chunk of C positions, per head, with D the key/value dim:
//
//  (1) cumulative per-CHANNEL decay, NOT per-head:
//        gc[i,d] = SUM_{p<=i} g[p,d]                    g is the LOG-decay from kda_gate
//      and the cross-position relationship is
//        decay(i,j,d) = exp(gc[i,d] - gc[j,d])
//      Sign matters: it is gc[i] MINUS gc[j], i the query row, j the key column.
//
//  (2) WY / forward substitution:
//        A[i,j] = -SUM_d (k[i,d]*beta[i]) * k[j,d] * decay(i,j,d)      for j <  i
//        A[i,j] = 0                                                    for j >= i
//        for i = 1..C-1:  A[i,:i] <- A[i,:i] + A[i,:i] @ A[:i,:i]
//        A[i,i] <- 1
//      (HF: `masked_fill(triu(diagonal=0), 0)`, the substitution loop, then `+ eye`.)
//      Row i of the substitution reads only rows m < i, which are already final.
//
//        u[i,d] = SUM_j A[i,j] * v[j,d]*beta[j]
//        w[i,d] = SUM_j A[i,j] * k[j,d]*beta[j] * exp(gc[j,d])
//
//  (3) intra- and inter-chunk propagation, S the [D_k, D_v] recurrent state:
//        v_new[i,v]  = u[i,v] - SUM_k w[i,k] * S[k,v]
//        intra[i,j]  = SUM_d q[i,d]*k[j,d]*decay(i,j,d)   for j <= i   (diagonal INCLUDED,
//                                                                       decay(i,i,d) = 1)
//        out[i,v]    = SUM_k q[i,k]*exp(gc[i,k])*S[k,v] + SUM_j intra[i,j]*v_new[j,v]
//        S[k,v]     <- S[k,v]*exp(gc[C-1,k])
//                      + SUM_i k[i,k]*exp(gc[C-1,k] - gc[i,k]) * v_new[i,v]
//      Note the two masks differ: A drops the diagonal (`triu diagonal=0`), intra keeps it
//      (`triu diagonal=1`).
//
//  (4) ragged tail, T % C != 0: positions past T are ZERO-PADDED in q, k, v, g and beta.
//      That is self-cleaning and it is worth seeing why, because it is the difference
//      between a correct tail and a silently poisoned state:
//        g=0    -> gc stops advancing, so gc[C-1] still equals the last REAL cumulative
//                  decay and the state update below is unaffected;
//        beta=0 -> k_beta and v_beta vanish, so A's row and column for a pad position are
//                  zero, the substitution leaves them zero, and `+eye` gives A[i,i]=1;
//        hence u[i]=w[i]=0 and v_new[i]=0, so a pad position contributes NOTHING to the
//        state sum, and its `out` row is simply never written back.
//
// `q` is pre-scaled by `scale` = 1/sqrt(D) exactly once, matching HF's
// `query = F.pad(query, ...) * scale`; both `intra` and the inter-chunk term therefore see
// the scaled q, and A (which uses k, never q) does not.
//
// ─────────────────────────────────────────────────────────────────────────────
// HF vs vLLM
// ─────────────────────────────────────────────────────────────────────────────
// vLLM's chunk path is a TILED REFORMULATION of the same recurrence — sub-chunk `BC`
// blocking, a fused inter+solve_tril kernel, GLA-style output, and cumulative sums carried
// in log2 space (`cumsum_scale = RCP_LN2` with `exp2`). It is not textually comparable to
// HF's reference loop and should not be diffed line by line. The semantics both implement
// are pinned by the recurrence, so the load-bearing check is chunk == recurrent, verified
// here on GPU and already verified inside HF itself (Slice 2 self-check: chunked vs
// recurrent 2.98e-8, chunked-padded vs recurrent 8.94e-8).
//
// ─────────────────────────────────────────────────────────────────────────────
// SHARED MEMORY — a correctness blocker on this stack, not a tuning knob
// ─────────────────────────────────────────────────────────────────────────────
// Atlas's CUDA backend has NO `cuFuncSetAttribute` /
// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` path, so a block cannot opt in past the
// DEFAULT 48 KiB (49152 B). GB10's 101376 B ceiling is unreachable from here today.
//
//   kda_chunk_prepare : (C*D + C*C + C) * 4 bytes
//   kda_chunk_scan    : (2*C*D + C*C)   * 4 bytes
//
// At D=128 that gives, for the scan (the larger of the two):
//   C=16 ->  9216 B     C=32 -> 36864 B     C=64 -> 81920 B  ✗ EXCEEDS 49152
// so C <= 32 at production D=128. The chunk size is a tiling parameter, not a model
// parameter — the chunked algorithm is algebraically equal to the recurrence for ANY C —
// so capping it costs nothing semantically. The caller MUST size the dynamic allocation;
// the microtest asserts the requirement against 49152 before launching.

#include <cuda_bf16.h>
#include <math.h>

// ── phase 1 ─────────────────────────────────────────────────────────────────
// grid = (num_chunks, H)   block = (128,1,1)
// smem = C*D (gc) + C*C (A) + C (row scratch) floats
extern "C" __global__ void kda_chunk_prepare(
    const float* __restrict__ k,      // [T_pad, H, D] fp32, ALREADY L2-normalised
    const float* __restrict__ v,      // [T_pad, H, D] fp32
    const float* __restrict__ gate,   // [T_pad, H, D] fp32, LOG-decay
    const float* __restrict__ beta,   // [T_pad, H]    fp32, ALREADY sigmoided
    float* __restrict__ gc_out,       // [T_pad, H, D] fp32, cumulative decay
    float* __restrict__ u_out,        // [T_pad, H, D] fp32
    float* __restrict__ w_out,        // [T_pad, H, D] fp32
    unsigned int H,
    unsigned int D,
    unsigned int C,
    unsigned int T                    // real token count; positions >= T are FORCED to zero
) {
    extern __shared__ float sh[];
    float* gc = sh;               // [C, D]
    float* A = sh + (size_t)C * D; // [C, C]
    float* row = A + (size_t)C * C; // [C]

    const unsigned int c = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const size_t base = ((size_t)c * C * H + h) * D; // position c*C, head h
    const size_t stride = (size_t)H * D;             // one position

    // Pad positions are forced to zero HERE rather than trusted to be zero in the caller's
    // buffer. Guarding only the `out` write is NOT enough: a non-zero pad still reaches the
    // recurrent state through gc / u / w, and the state is carried forward, so the corruption
    // is invisible in this prefill's output and surfaces later. Caught by microtest D5.
    #define KDA_AT(buf, i, d) (((c) * C + (i)) < T ? (buf)[base + (size_t)(i) * stride + (d)] : 0.0f)
    #define KDA_BETA(i) (((c) * C + (i)) < T ? beta[((size_t)((c) * C + (i)) * H) + h] : 0.0f)

    // (1) cumulative per-channel decay
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < C; ++i) {
            acc += KDA_AT(gate, i, d);
            gc[(size_t)i * D + d] = acc;
        }
    }
    __syncthreads();

    // (2) A[i,j] for j < i, zero elsewhere
    for (unsigned int idx = threadIdx.x; idx < C * C; idx += blockDim.x) {
        const unsigned int i = idx / C, j = idx % C;
        float a = 0.0f;
        if (j < i) {
            const float bi = KDA_BETA(i);
            float acc = 0.0f;
            for (unsigned int d = 0; d < D; ++d) {
                acc += KDA_AT(k, i, d) * bi * KDA_AT(k, j, d)
                     * expf(gc[(size_t)i * D + d] - gc[(size_t)j * D + d]);
            }
            a = -acc;
        }
        A[idx] = a;
    }
    __syncthreads();

    // forward substitution: row i sees only rows m < i, already final
    for (unsigned int i = 1; i < C; ++i) {
        for (unsigned int j = threadIdx.x; j < i; j += blockDim.x) {
            float acc = 0.0f;
            for (unsigned int m = 0; m < i; ++m) {
                acc += A[(size_t)i * C + m] * A[(size_t)m * C + j];
            }
            row[j] = A[(size_t)i * C + j] + acc;
        }
        __syncthreads();
        for (unsigned int j = threadIdx.x; j < i; j += blockDim.x) {
            A[(size_t)i * C + j] = row[j];
        }
        __syncthreads();
    }
    for (unsigned int i = threadIdx.x; i < C; i += blockDim.x) {
        A[(size_t)i * C + i] = 1.0f;
    }
    __syncthreads();

    // u and w
    for (unsigned int idx = threadIdx.x; idx < C * D; idx += blockDim.x) {
        const unsigned int i = idx / D, d = idx % D;
        float au = 0.0f, aw = 0.0f;
        for (unsigned int j = 0; j <= i; ++j) {
            const float a = A[(size_t)i * C + j];
            if (a == 0.0f) continue;
            const float bj = KDA_BETA(j);
            au += a * KDA_AT(v, j, d) * bj;
            aw += a * KDA_AT(k, j, d) * bj * expf(gc[(size_t)j * D + d]);
        }
        const size_t o = base + (size_t)i * stride + d;
        u_out[o] = au;
        w_out[o] = aw;
        gc_out[o] = gc[(size_t)i * D + d];
    }
    #undef KDA_AT
    #undef KDA_BETA
}

// ── phase 2 ─────────────────────────────────────────────────────────────────
// grid = (H,1,1)   block = (128,1,1)
// smem = C*D (gc) + C*D (v_new) + C*C (intra) floats
extern "C" __global__ void kda_chunk_scan(
    const float* __restrict__ q,      // [T_pad, H, D] fp32, ALREADY L2-normalised
    const float* __restrict__ k,      // [T_pad, H, D] fp32, ALREADY L2-normalised
    const float* __restrict__ gc_in,  // [T_pad, H, D] fp32
    const float* __restrict__ u_in,   // [T_pad, H, D] fp32
    const float* __restrict__ w_in,   // [T_pad, H, D] fp32
    float* __restrict__ state,        // [H, D, D] fp32, K-major, read-modify-write
    float* __restrict__ out,          // [T_pad, H, D] fp32 (pad rows left untouched)
    unsigned int H,
    unsigned int D,
    unsigned int C,
    unsigned int num_chunks,
    unsigned int T,                   // real token count; rows >= T are not written
    float scale                       // 1/sqrt(D), applied to q
) {
    extern __shared__ float sh[];
    float* gc = sh;                      // [C, D]
    float* vnew = sh + (size_t)C * D;    // [C, D]
    float* intra = vnew + (size_t)C * D; // [C, C]

    const unsigned int h = blockIdx.x;
    const size_t stride = (size_t)H * D;
    float* S = state + (size_t)h * D * D;

    for (unsigned int c = 0; c < num_chunks; ++c) {
        const size_t base = ((size_t)c * C * H + h) * D;

        for (unsigned int idx = threadIdx.x; idx < C * D; idx += blockDim.x) {
            gc[idx] = gc_in[base + (size_t)(idx / D) * stride + (idx % D)];
        }
        __syncthreads();

        // v_new = u - w @ S
        for (unsigned int idx = threadIdx.x; idx < C * D; idx += blockDim.x) {
            const unsigned int i = idx / D, vi = idx % D;
            float acc = 0.0f;
            for (unsigned int kk = 0; kk < D; ++kk) {
                acc += w_in[base + (size_t)i * stride + kk] * S[(size_t)kk * D + vi];
            }
            vnew[idx] = u_in[base + (size_t)i * stride + vi] - acc;
        }
        // intra[i,j], j <= i (diagonal included)
        for (unsigned int idx = threadIdx.x; idx < C * C; idx += blockDim.x) {
            const unsigned int i = idx / C, j = idx % C;
            float a = 0.0f;
            if (j <= i) {
                for (unsigned int d = 0; d < D; ++d) {
                    a += q[base + (size_t)i * stride + d] * scale
                       * k[base + (size_t)j * stride + d]
                       * expf(gc[(size_t)i * D + d] - gc[(size_t)j * D + d]);
                }
            }
            intra[idx] = a;
        }
        __syncthreads();

        // out = (q*exp(gc)) @ S + intra @ v_new
        for (unsigned int idx = threadIdx.x; idx < C * D; idx += blockDim.x) {
            const unsigned int i = idx / D, vi = idx % D;
            const unsigned int t = c * C + i;
            if (t >= T) continue;
            float acc = 0.0f;
            for (unsigned int kk = 0; kk < D; ++kk) {
                acc += q[base + (size_t)i * stride + kk] * scale
                     * expf(gc[(size_t)i * D + kk]) * S[(size_t)kk * D + vi];
            }
            for (unsigned int j = 0; j <= i; ++j) {
                acc += intra[(size_t)i * C + j] * vnew[(size_t)j * D + vi];
            }
            out[base + (size_t)i * stride + vi] = acc;
        }
        __syncthreads();

        // S <- S*exp(gc_last) + SUM_i k[i]*exp(gc_last - gc[i]) (x) v_new[i]
        for (unsigned int idx = threadIdx.x; idx < D * D; idx += blockDim.x) {
            const unsigned int kk = idx / D, vi = idx % D;
            const float gl = gc[(size_t)(C - 1) * D + kk];
            float acc = S[idx] * expf(gl);
            for (unsigned int i = 0; i < C; ++i) {
                // Same defence as in `prepare`: a pad position must contribute nothing to the
                // carried state even if the caller's buffer holds garbage there.
                if (c * C + i >= T) continue;
                acc += k[base + (size_t)i * stride + kk]
                     * expf(gl - gc[(size_t)i * D + kk])
                     * vnew[(size_t)i * D + vi];
            }
            S[idx] = acc;
        }
        __syncthreads();
    }
}
