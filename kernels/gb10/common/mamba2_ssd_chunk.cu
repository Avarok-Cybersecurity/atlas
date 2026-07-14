// SPDX-License-Identifier: AGPL-3.0-only

// Mamba-2 SSD (state-space duality) CHUNKED prefill scan.
//
// Replaces the token-sequential recurrence (`mamba2_ssm_prefill*`) with the chunked
// formulation from "Transformers are SSMs" (Dao & Gu, 2024), i.e. what mamba-ssm's
// `_mamba_chunk_scan_combined` and vLLM's Triton `ssd_*` kernels compute.
//
// The sequential kernel walks 1023 dependent steps of scalar FMAs, so it is latency
// bound and cannot use the tensor cores at all. Because the decay `dA_t` is a SCALAR
// per (head, token) — the defining restriction of Mamba-2 vs Mamba-1 — the product
// prod_{r=s+1..t} dA_r factorises as exp(cs_t - cs_s). That lets the whole scan be
// rewritten as matmuls, with only ceil(T/L) sequential links instead of T.
//
// Per head h, chunk c, local indices t,s in [0,L):
//   a_t     = -exp(A_log[h]) * dt_t          (log-decay, <= 0)
//   cs_t    = inclusive cumsum of a over the chunk
//   h_t     = exp(cs_t) * h0_c + Hlocal_t
//   y_t[p]  = exp(cs_t) * sum_n C_t[n]*h0_c[p][n]                       (Y_off)
//           + sum_{s<=t} (C_t . B_s) * exp(cs_t-cs_s) * dt_s * x_s[p]   (Y_diag)
//           + D[h] * x_t[p]
//   Hlocal_c[p][n] = sum_s exp(cs_last-cs_s) * dt_s * x_s[p] * B_s[n]
//   h0_{c+1}       = exp(cs_last) * h0_c + Hlocal_c                     (only T/L steps)
//
// All decay maths is fp32 in log space (never multiply dA values together), and every
// exponent is clamped to <= 0 so factors stay in (0,1] — mirroring vLLM's
// `fast_exp(tl.minimum(dA_cs_m - dA_cs_k, 0.0))`.
//
// Padded tail tokens get dt = 0 => a = 0 => they contribute nothing and cs stays flat.

#include <cuda_bf16.h>

#define SSD_L 64          // chunk length
// head_dim rows per block. 32, not 16: sB/sCM/CB and all the exp() decay work are
// independent of the head_dim slice, so every p-block redoes them. At PT=16 (4
// p-blocks per head) that is 4x redundant global traffic and 4x redundant SFU work
// for only ~18 MMAs per warp per chunk. PT=32 halves both.
#define SSD_PT 64         // head_dim rows per block (head_dim must be a multiple)

// ---------------------------------------------------------------------------
// K1: per-chunk dt (softplus+clamp) and inclusive cumsum of the log-decay.
// Grid: (nchunks, num_heads, batch)   Block: (SSD_L)
// ---------------------------------------------------------------------------
extern "C" __global__ void mamba2_ssd_cumsum(
    const __nv_bfloat16* __restrict__ dt_raw,   // [t][b*num_heads + h]
    const float* __restrict__ A_log,            // [num_heads]
    const float* __restrict__ dt_bias,          // [num_heads]
    float* __restrict__ dt_out,                 // [b][h][nchunks][L]
    float* __restrict__ dA_cs,                  // [b][h][nchunks][L]
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int nchunks,
    unsigned int dt_stride,
    float dt_min,
    float dt_max
) {
    const unsigned int c = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int t = threadIdx.x;
    const unsigned int gt = c * SSD_L + t;

    float dtv = 0.0f;                       // padded tail => dt = 0 (contributes nothing)
    if (gt < seq_len) {
        dtv = (float)dt_raw[(unsigned long long)gt * dt_stride + b * num_heads + h]
            + dt_bias[h];
        dtv = (dtv > 20.0f) ? dtv : logf(1.0f + expf(dtv));
        dtv = fminf(fmaxf(dtv, dt_min), dt_max);
    }
    const float a = -expf(A_log[h]) * dtv;  // <= 0

    // Inclusive scan across the 64 lanes (2 warps).
    __shared__ float warp_tail[2];
    const unsigned int lane = t & 31u;
    const unsigned int warp = t >> 5;
    float v = a;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        float n = __shfl_up_sync(0xFFFFFFFFu, v, off);
        if (lane >= (unsigned int)off) v += n;
    }
    if (lane == 31u) warp_tail[warp] = v;
    __syncthreads();
    if (warp == 1u) v += warp_tail[0];

    const unsigned long long base =
        (((unsigned long long)b * num_heads + h) * nchunks + c) * SSD_L;
    dt_out[base + t] = dtv;
    dA_cs[base + t]  = v;
}

// ---------------------------------------------------------------------------
// K2: CB[c][g][t][s] = sum_n C[c*L+t][g][n] * B[c*L+s][g][n]   (raw C.B^T, fp32)
// The per-head dt/decay scaling is applied later by K3 (it depends on h, not g).
// Grid: (nchunks, n_groups, batch)   Block: (128) = 4 warps
// ---------------------------------------------------------------------------
extern "C" __global__ void mamba2_ssd_bmm(
    const __nv_bfloat16* __restrict__ B_in,     // [t][b*n_groups*N + g*N + n]
    const __nv_bfloat16* __restrict__ C_in,
    float* __restrict__ CB,                     // [b][c][g][L][L]
    unsigned int seq_len,
    unsigned int nchunks,
    unsigned int n_groups,
    unsigned int state_size,                    // N
    unsigned int bc_stride
) {
    const unsigned int c = blockIdx.x;
    const unsigned int g = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int N = state_size;

    extern __shared__ __nv_bfloat16 smem_bmm[];
    __nv_bfloat16* sC = smem_bmm;               // [L][N]
    __nv_bfloat16* sB = sC + (unsigned long long)SSD_L * N;

    for (unsigned int i = threadIdx.x; i < SSD_L * N; i += blockDim.x) {
        const unsigned int t = i / N, n = i - t * N;
        const unsigned int gt = c * SSD_L + t;
        const unsigned long long off =
            (unsigned long long)gt * bc_stride + b * n_groups * N + g * N + n;
        const bool ok = gt < seq_len;
        sC[i] = ok ? C_in[off] : __float2bfloat16(0.0f);
        sB[i] = ok ? B_in[off] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // m16n8k16: M = t (64), N = s (64), K = n (state).  A = sC[t][n], B = sB[s][n]
    // (B is naturally N-by-K, i.e. exactly the .col operand.)
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int gid = lane >> 2;         // 0..7
    const unsigned int tid = lane & 3u;         // 0..3
    const unsigned int wm = warp * 16u;         // this warp's 16 rows of t

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    const unsigned short* A16 = (const unsigned short*)sC;
    const unsigned short* B16 = (const unsigned short*)sB;

    for (unsigned int k = 0; k < N; k += 16u) {
        const unsigned int r0 = wm + gid, r1 = wm + gid + 8u;
        const unsigned int c0 = k + tid * 2u,  c1 = k + tid * 2u + 8u;
        if (c0 >= N) break;

        unsigned int a0 = ((unsigned int)A16[r0 * N + c0 + 1] << 16) | A16[r0 * N + c0];
        unsigned int a1 = ((unsigned int)A16[r1 * N + c0 + 1] << 16) | A16[r1 * N + c0];
        unsigned int a2 = ((unsigned int)A16[r0 * N + c1 + 1] << 16) | A16[r0 * N + c1];
        unsigned int a3 = ((unsigned int)A16[r1 * N + c1 + 1] << 16) | A16[r1 * N + c1];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const unsigned int s = nt * 8u + gid;           // the B row (== MMA N index)
            unsigned int b0 = ((unsigned int)B16[s * N + c0 + 1] << 16) | B16[s * N + c0];
            unsigned int b1 = ((unsigned int)B16[s * N + c1 + 1] << 16) | B16[s * N + c1];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }

    float* out = CB + ((((unsigned long long)b * nchunks + c) * n_groups + g)
                       * SSD_L * SSD_L);
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const unsigned int s0 = nt * 8u + tid * 2u;
        const unsigned int t0 = wm + gid, t1 = wm + gid + 8u;
        out[t0 * SSD_L + s0]      = acc[nt][0];
        out[t0 * SSD_L + s0 + 1]  = acc[nt][1];
        out[t1 * SSD_L + s0]      = acc[nt][2];
        out[t1 * SSD_L + s0 + 1]  = acc[nt][3];
    }
}

// ---------------------------------------------------------------------------
// K3: fused chunk_state + state_passing + chunk_scan.
//
// Fusing them keeps h0 in shared memory, so the [nchunks][H][P][N] "states" tensor
// that vLLM round-trips through DRAM (~50 MB/layer) never exists.
//
// Grid: (num_heads, head_dim/SSD_PT, batch)   Block: (128) = 4 warps
// ---------------------------------------------------------------------------
extern "C" __global__ void mamba2_ssd_scan(
    float* __restrict__ h_state,                // [b][h][P][N] fp32  (in/out)
    const __nv_bfloat16* __restrict__ x,        // [t][(b*H+h)*P + p]
    const __nv_bfloat16* __restrict__ B_in,     // [t][b*G*N + g*N + n]
    const __nv_bfloat16* __restrict__ C_in,
    const float* __restrict__ D_param,          // [H]
    const float* __restrict__ dt_f32,           // [b][h][nchunks][L]
    const float* __restrict__ dA_cs,            // [b][h][nchunks][L]
    const float* __restrict__ CB,               // [b][c][g][L][L]
    __nv_bfloat16* __restrict__ output,         // [t][(b*H+h)*P + p]
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,                      // P
    unsigned int state_size,                    // N
    unsigned int n_groups,
    unsigned int nchunks,
    unsigned int x_stride,
    unsigned int bc_stride,
    unsigned int y_stride
) {
    const unsigned int h  = blockIdx.x;
    const unsigned int pt = blockIdx.y;         // which SSD_PT slice of head_dim
    const unsigned int b  = blockIdx.z;
    const unsigned int P  = head_dim;
    const unsigned int N  = state_size;
    const unsigned int g  = h / (num_heads / n_groups);
    const unsigned int p0 = pt * SSD_PT;

    const float D_val = D_param[h];

    extern __shared__ char smem_raw[];
    float*         sH  = (float*)smem_raw;                          // [PT][N+1] fp32 (h0)
    __nv_bfloat16* sHb = (__nv_bfloat16*)(sH + SSD_PT * (N + 1));   // [PT][N]  h0 bf16
    __nv_bfloat16* sB  = sHb + SSD_PT * N;                          // [L][N]
    __nv_bfloat16* sCM = sB  + SSD_L * N;                           // [L][N] as C, then [L][L] as M
    __nv_bfloat16* sX  = sCM + SSD_L * N;                           // [L][PT]
    __nv_bfloat16* sXt = sX  + SSD_L * SSD_PT;                      // [PT][L]
    float*         sdA = (float*)(sXt + SSD_PT * SSD_L);            // [L]
    float*         sdt = sdA + SSD_L;                               // [L]

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int gid  = lane >> 2;
    const unsigned int tid  = lane & 3u;

    float* Hg = h_state + ((unsigned long long)(b * num_heads + h) * P + p0) * N;

    // h0 starts from the incoming state (0 on a fresh prefill).
    for (unsigned int i = threadIdx.x; i < SSD_PT * N; i += blockDim.x) {
        const unsigned int p = i / N, n = i - p * N;
        sH[p * (N + 1) + n] = Hg[p * N + n];
    }
    __syncthreads();

    for (unsigned int c = 0; c < nchunks; c++) {
        const unsigned long long dbase =
            (((unsigned long long)b * num_heads + h) * nchunks + c) * SSD_L;

        for (unsigned int i = threadIdx.x; i < SSD_L; i += blockDim.x) {
            sdA[i] = dA_cs[dbase + i];
            sdt[i] = dt_f32[dbase + i];
        }
        // x tile: [L][PT] and its transpose [PT][L]
        for (unsigned int i = threadIdx.x; i < SSD_L * SSD_PT; i += blockDim.x) {
            const unsigned int t = i / SSD_PT, p = i - t * SSD_PT;
            const unsigned int gt = c * SSD_L + t;
            __nv_bfloat16 v = __float2bfloat16(0.0f);
            if (gt < seq_len) {
                v = x[(unsigned long long)gt * x_stride
                      + (unsigned long long)(b * num_heads + h) * P + p0 + p];
            }
            sX[t * SSD_PT + p] = v;
            sXt[p * SSD_L + t] = v;
        }
        // B and C tiles: [L][N]
        for (unsigned int i = threadIdx.x; i < SSD_L * N; i += blockDim.x) {
            const unsigned int t = i / N, n = i - t * N;
            const unsigned int gt = c * SSD_L + t;
            const unsigned long long off =
                (unsigned long long)gt * bc_stride + b * n_groups * N + g * N + n;
            const bool ok = gt < seq_len;
            sB[i]  = ok ? B_in[off] : __float2bfloat16(0.0f);
            sCM[i] = ok ? C_in[off] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        const float cs_last = sdA[SSD_L - 1];

        // h0 -> bf16 (B-operand of Y_off), laid out [p][n] so k is contiguous.
        for (unsigned int i = threadIdx.x; i < SSD_PT * N; i += blockDim.x) {
            const unsigned int p = i / N, n = i - p * N;
            sHb[p * N + n] = __float2bfloat16(sH[p * (N + 1) + n]);
        }
        // Cd[t][n] = C[t][n] * exp(cs_t)   (row scaling folds the decay into the operand)
        for (unsigned int i = threadIdx.x; i < SSD_L * N; i += blockDim.x) {
            const unsigned int t = i / N;
            sCM[i] = __float2bfloat16((float)sCM[i] * __expf(fminf(sdA[t], 0.0f)));
        }
        __syncthreads();

        // ---- (a) Y_off = Cd (L x N) @ h0^T (N x PT) ----
        // PT == head_dim (64) => ONE block per head, so sB/sCM/CB and every exp() decay
        // pass is loaded/computed exactly once instead of once per head_dim slice.
        //
        // That needs 4 m-blocks x 8 n-tiles = 32 warp-tasks. Running them as 32 warps
        // (1024 threads) does NOT launch: the register budget collapses to 64/thread and
        // CUDA reports LAUNCH_OUT_OF_RESOURCES. So keep 16 warps (512 thr) and give each
        // warp TWO adjacent n-tiles; the A fragments are shared between them, only the B
        // fragments differ, so the second tile is nearly free.
        const unsigned int wm  = (warp >> 2) * 16u;   // m-block (4)
        const unsigned int wnb = (warp & 3u) * 2u;    // first of this warp's 2 n-tiles
        float acc[2][4];
        #pragma unroll
        for (int q = 0; q < 2; q++) { acc[q][0]=0.f; acc[q][1]=0.f; acc[q][2]=0.f; acc[q][3]=0.f; }
        {
            const unsigned short* A16 = (const unsigned short*)sCM;   // [L][N]
            const unsigned short* B16 = (const unsigned short*)sHb;   // [PT][N] (n contiguous)
            for (unsigned int k = 0; k < N; k += 16u) {
                const unsigned int r0 = wm + gid, r1 = wm + gid + 8u;
                const unsigned int c0 = k + tid * 2u, c1 = k + tid * 2u + 8u;
                if (c0 >= N) break;
                unsigned int a0 = ((unsigned int)A16[r0*N + c0+1] << 16) | A16[r0*N + c0];
                unsigned int a1 = ((unsigned int)A16[r1*N + c0+1] << 16) | A16[r1*N + c0];
                unsigned int a2 = ((unsigned int)A16[r0*N + c1+1] << 16) | A16[r0*N + c1];
                unsigned int a3 = ((unsigned int)A16[r1*N + c1+1] << 16) | A16[r1*N + c1];
                #pragma unroll
                for (unsigned int q = 0; q < 2u; q++) {
                    const unsigned int p = (wnb + q) * 8u + gid;      // MMA N index = p
                    unsigned int b0 = ((unsigned int)B16[p*N + c0+1] << 16) | B16[p*N + c0];
                    unsigned int b1 = ((unsigned int)B16[p*N + c1+1] << 16) | B16[p*N + c1];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                        : "=f"(acc[q][0]), "=f"(acc[q][1]), "=f"(acc[q][2]), "=f"(acc[q][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                          "f"(acc[q][0]), "f"(acc[q][1]), "f"(acc[q][2]), "f"(acc[q][3]));
                }
            }
        }
        __syncthreads();   // sCM is about to be reused as the decay-masked M tile

        // ---- build M[t][s] = CB[t][s] * exp(min(cs_t-cs_s,0)) * dt_s, strictly causal ----
        {
            const float* cb = CB + ((((unsigned long long)b * nchunks + c) * n_groups + g)
                                    * SSD_L * SSD_L);
            for (unsigned int i = threadIdx.x; i < SSD_L * SSD_L; i += blockDim.x) {
                const unsigned int t = i / SSD_L, s = i - t * SSD_L;
                float v = 0.0f;
                if (s <= t) {
                    v = cb[i] * __expf(fminf(sdA[t] - sdA[s], 0.0f)) * sdt[s];
                }
                sCM[i] = __float2bfloat16(v);
            }
        }
        __syncthreads();

        // ---- (b) Y_diag += M (L x L) @ X (L x PT) ----
        {
            const unsigned short* A16 = (const unsigned short*)sCM;   // [L][L]
            const unsigned short* B16 = (const unsigned short*)sX;    // [L][PT]  (k = t-row)
            for (unsigned int k = 0; k < SSD_L; k += 16u) {
                const unsigned int r0 = wm + gid, r1 = wm + gid + 8u;
                const unsigned int c0 = k + tid * 2u, c1 = k + tid * 2u + 8u;
                unsigned int a0 = ((unsigned int)A16[r0*SSD_L + c0+1] << 16) | A16[r0*SSD_L + c0];
                unsigned int a1 = ((unsigned int)A16[r1*SSD_L + c0+1] << 16) | A16[r1*SSD_L + c0];
                unsigned int a2 = ((unsigned int)A16[r0*SSD_L + c1+1] << 16) | A16[r0*SSD_L + c1];
                unsigned int a3 = ((unsigned int)A16[r1*SSD_L + c1+1] << 16) | A16[r1*SSD_L + c1];
                #pragma unroll
                for (unsigned int q = 0; q < 2u; q++) {
                    const unsigned int p = (wnb + q) * 8u + gid;
                    // B[n=p][k=s] = sX[s][p]  -> strided by SSD_PT
                    unsigned int b0 = ((unsigned int)B16[(c0+1)*SSD_PT + p] << 16)
                                    | B16[c0*SSD_PT + p];
                    unsigned int b1 = ((unsigned int)B16[(c1+1)*SSD_PT + p] << 16)
                                    | B16[c1*SSD_PT + p];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                        : "=f"(acc[q][0]), "=f"(acc[q][1]), "=f"(acc[q][2]), "=f"(acc[q][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                          "f"(acc[q][0]), "f"(acc[q][1]), "f"(acc[q][2]), "f"(acc[q][3]));
                }
            }
        }

        // ---- (c) store y = Y_off + Y_diag + D*x ----
        // m16n8k16 accumulator layout: the M index comes from group_id (rows gid, gid+8)
        // and the N index from threadID_in_group*2 (cols tid*2, tid*2+1). This is NOT the
        // same mapping the B *operand* uses (which is group_id on N) -- swapping them is
        // the classic way to get a silently wrong MMA kernel.
        #pragma unroll
        for (unsigned int q = 0; q < 2u; q++) {
            const unsigned int pa = (wnb + q) * 8u + tid * 2u;   // N index
            const unsigned int ta = wm + gid;                     // M index
            const unsigned int tb = wm + gid + 8u;
            const unsigned int ts[4] = { ta, ta, tb, tb };
            const unsigned int ps[4] = { pa, pa + 1u, pa, pa + 1u };
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const unsigned int t  = ts[r];
                const unsigned int p  = ps[r];
                const unsigned int gt = c * SSD_L + t;
                if (gt >= seq_len || p >= SSD_PT) continue;
                const float xv = (float)sX[t * SSD_PT + p];
                output[(unsigned long long)gt * y_stride
                       + (unsigned long long)(b * num_heads + h) * P + p0 + p] =
                    __float2bfloat16(acc[q][r] + D_val * xv);
            }
        }
        __syncthreads();

        // ---- (d) Bd[s][n] = B[s][n] * exp(min(cs_last-cs_s,0)) * dt_s ----
        for (unsigned int i = threadIdx.x; i < SSD_L * N; i += blockDim.x) {
            const unsigned int s = i / N;
            sB[i] = __float2bfloat16((float)sB[i]
                        * __expf(fminf(cs_last - sdA[s], 0.0f)) * sdt[s]);
        }
        __syncthreads();

        // ---- (e) Hlocal[p][n] = sum_s Xt[p][s] * Bd[s][n]   (M=PT, N=N, K=L) ----
        // PT=64 -> 4 m-tiles of 16, x 8 n-lanes = 32 tasks over 16 warps, so each warp
        // takes 2 m-tiles (mt and mt+2). Each lane still covers up to 2 of the 12 n-tiles.
        const unsigned int NT = N / 8u;                 // 12 n-tiles
        const unsigned int emt0 = (warp >> 3);          // 0 or 1
        const unsigned int enb  = warp & 7u;            // n-lane
        float hacc[2][2][4];
        #pragma unroll
        for (int u = 0; u < 2; u++)
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                hacc[u][i][0]=0.f; hacc[u][i][1]=0.f; hacc[u][i][2]=0.f; hacc[u][i][3]=0.f;
            }
        {
            const unsigned short* A16 = (const unsigned short*)sXt;  // [PT][L]
            const unsigned short* B16 = (const unsigned short*)sB;   // [L][N] -> B[n][k=s] = sB[s][n]
            for (unsigned int k = 0; k < SSD_L; k += 16u) {
                const unsigned int c0 = k + tid * 2u, c1 = k + tid * 2u + 8u;
                #pragma unroll
                for (unsigned int u = 0; u < 2u; u++) {
                    const unsigned int emt = (emt0 + u * 2u) * 16u;
                    const unsigned int r0 = emt + gid, r1 = emt + gid + 8u;
                    unsigned int a0 = ((unsigned int)A16[r0*SSD_L + c0+1] << 16) | A16[r0*SSD_L + c0];
                    unsigned int a1 = ((unsigned int)A16[r1*SSD_L + c0+1] << 16) | A16[r1*SSD_L + c0];
                    unsigned int a2 = ((unsigned int)A16[r0*SSD_L + c1+1] << 16) | A16[r0*SSD_L + c1];
                    unsigned int a3 = ((unsigned int)A16[r1*SSD_L + c1+1] << 16) | A16[r1*SSD_L + c1];
                    #pragma unroll
                    for (unsigned int j = 0; j < 2u; j++) {
                        const unsigned int ntile = enb + j * 8u;
                        if (ntile >= NT) break;
                        const unsigned int n = ntile * 8u + gid;
                        unsigned int b0 = ((unsigned int)B16[(c0+1)*N + n] << 16) | B16[c0*N + n];
                        unsigned int b1 = ((unsigned int)B16[(c1+1)*N + n] << 16) | B16[c1*N + n];
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                            : "=f"(hacc[u][j][0]), "=f"(hacc[u][j][1]),
                              "=f"(hacc[u][j][2]), "=f"(hacc[u][j][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                              "f"(hacc[u][j][0]), "f"(hacc[u][j][1]),
                              "f"(hacc[u][j][2]), "f"(hacc[u][j][3]));
                    }
                }
            }
        }

        // ---- (f) state passing: h0 <- exp(cs_last)*h0 + Hlocal   (the only serial link) ----
        const float decay = __expf(fminf(cs_last, 0.0f));
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < SSD_PT * N; i += blockDim.x) {
            const unsigned int p = i / N, n = i - p * N;
            sH[p * (N + 1) + n] *= decay;
        }
        __syncthreads();
        {
            #pragma unroll
            for (unsigned int u = 0; u < 2u; u++) {
                const unsigned int emt = (emt0 + u * 2u) * 16u;
                #pragma unroll
                for (unsigned int j = 0; j < 2u; j++) {
                    const unsigned int ntile = enb + j * 8u;
                    if (ntile >= NT) break;
                    const unsigned int n0 = ntile * 8u + tid * 2u;
                    const unsigned int pA = emt + gid, pB = emt + gid + 8u;
                    atomicAdd(&sH[pA * (N + 1) + n0],     hacc[u][j][0]);
                    atomicAdd(&sH[pA * (N + 1) + n0 + 1], hacc[u][j][1]);
                    atomicAdd(&sH[pB * (N + 1) + n0],     hacc[u][j][2]);
                    atomicAdd(&sH[pB * (N + 1) + n0 + 1], hacc[u][j][3]);
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int i = threadIdx.x; i < SSD_PT * N; i += blockDim.x) {
        const unsigned int p = i / N, n = i - p * N;
        Hg[p * N + n] = sH[p * (N + 1) + n];
    }
}
