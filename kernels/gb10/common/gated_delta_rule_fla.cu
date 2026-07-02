// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN prefill — FLA-style MULTI-KERNEL decomposition (the path to beat
// vLLM). The single-fused chunk64 kernel was boxed in (serial-per-chunk in one
// CTA → 0.38-0.69x). FLA's speed comes from splitting into passes where the BIG
// matmuls are PARALLEL over all chunks (full 48-SM occupancy) and only a small
// state-passing is serial:
//   recompute_w_u (THIS, parallel over chunks×heads, H-independent):
//       solve (I+L)U = βV, (I+L)W = β·exp(gc)·K  via forward-substitution.
//       L[i][l] = β_i·exp(gc_i-gc_l)·<k_l,k_i> (l<i), Gram built on tensor cores.
//   chunk_delta_h (serial over chunks): S_{c+1}=exp(gc_last)S_c + K̃ᵀU - K̃ᵀ(WS).
//   chunk_fwd_o (parallel over chunks): O = Q̃·S + tril(Q̃·Kᵀ)·U.
// Math validated == recurrent SSOT by the CPU oracle
// (crates/spark-runtime/tests/gdn_chunk64_oracle.rs :: fla_decomposed_ref).
//
// gate[] LINEAR decay (NO exp, NO clamp on prefill); chunk decay in LOG space.
// GB10 sm_121: mma.sync.m16n8k16 BF16 (no wgmma/TMA, ldmatrix broken).

#include <cuda_bf16.h>

#define K_DIM 128
#define V_DIM 128
#define CHUNK 64
// Floor for the linear gate before log-space cumsum. Deep-layer gates can
// underflow to exactly 0.0 (or tiny negatives), and log(0)=-inf → exp(gc_i-gc_l)
// = exp(-inf - -inf) = NaN. The per-token recurrence (h=g·h) tolerates g≈0; the
// chunked log-space form does not. 1e-30 ⇒ log≈-69 (effectively full decay) and
// is a no-op for any normal gate. (GATE-B used g∈[0.8,0.999], never hitting this.)
#define GATE_FLOOR 1e-30f

// GB10 sm_121 has cp.async.cg (NO TMA). 16-byte async global→shared copy + group
// commit/wait — used to double-buffer the per-chunk W/U/K loads in chunk_delta_h
// so the serial state spine overlaps the next chunk's load with the current
// chunk's compute (it was global-load-LATENCY bound at 4 warps, not FLOP bound).
__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// C[m][n] = Σ_k A[m][k]·B[n][k], M=64, K=K_DIM, N=NTC*8. A/B row-major bf16 smem;
// 128 threads = 4 warps (16 M-rows each). NSTRIDE = C row-stride. (SSOT helper.)
template <int NTC, int NSTRIDE, bool OutBf16>
__device__ __forceinline__ void mma_gram(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, void* __restrict__ C
) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    const unsigned grp = lane >> 2;
    const unsigned q = lane & 3;
    const unsigned warp_m = warp * 16;
    const unsigned short* sA = (const unsigned short*)A;
    const unsigned short* sB = (const unsigned short*)B;
    float acc[NTC][4];
    #pragma unroll
    for (int nt = 0; nt < NTC; nt++) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
    #pragma unroll
    for (unsigned ks = 0; ks < K_DIM; ks += 16) {
        unsigned fr0 = warp_m + grp, fr1 = fr0 + 8;
        unsigned fc0 = ks + q * 2, fc1 = fc0 + 8;
        unsigned a0 = *(const unsigned*)&sA[fr0 * K_DIM + fc0];
        unsigned a1 = *(const unsigned*)&sA[fr1 * K_DIM + fc0];
        unsigned a2 = *(const unsigned*)&sA[fr0 * K_DIM + fc1];
        unsigned a3 = *(const unsigned*)&sA[fr1 * K_DIM + fc1];
        #pragma unroll
        for (int nt = 0; nt < NTC; nt++) {
            unsigned nc = nt * 8 + grp;
            unsigned k0 = ks + q * 2, k1 = k0 + 8;
            unsigned b0 = ((unsigned)sB[nc * K_DIM + k0 + 1] << 16) | (unsigned)sB[nc * K_DIM + k0];
            unsigned b1 = ((unsigned)sB[nc * K_DIM + k1 + 1] << 16) | (unsigned)sB[nc * K_DIM + k1];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }
    #pragma unroll
    for (int nt = 0; nt < NTC; nt++) {
        unsigned n0 = nt * 8 + q * 2, n1 = n0 + 1;
        unsigned m0 = warp_m + grp, m1 = m0 + 8;
        if (OutBf16) {
            __nv_bfloat16* Cb = (__nv_bfloat16*)C;
            Cb[m0 * NSTRIDE + n0] = __float2bfloat16(acc[nt][0]);
            Cb[m0 * NSTRIDE + n1] = __float2bfloat16(acc[nt][1]);
            Cb[m1 * NSTRIDE + n0] = __float2bfloat16(acc[nt][2]);
            Cb[m1 * NSTRIDE + n1] = __float2bfloat16(acc[nt][3]);
        } else {
            float* Cf = (float*)C;
            Cf[m0 * NSTRIDE + n0] = acc[nt][0];
            Cf[m0 * NSTRIDE + n1] = acc[nt][1];
            Cf[m1 * NSTRIDE + n0] = acc[nt][2];
            Cf[m1 * NSTRIDE + n1] = acc[nt][3];
        }
    }
}

// ── KERNEL 1: recompute_w_u ──────────────────────────────────────────────
// Grid: (NT, num_v_heads, batch)  Block: (128,1,1).  One CTA per (chunk, head).
// Outputs (f32, layout [(b*NT+c)*nv+vh][CHUNK][·]):
//   U_out: [.. ][CHUNK][V_DIM]   = T·(βV)
//   W_out: [.. ][CHUNK][K_DIM]   = T·(β·exp(gc)·K)
// where T=(I+L)⁻¹ applied by forward-substitution (parallel over the V/K cols).
// smem: sk_bf(16K) + kk(16K f32) + L(16K f32) + gc(256) ≈ 48.25KB.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_recompute_wu(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ W_out,   // bf16 — per-chunk intermediate, fed to TC matmuls in #2/#3
    __nv_bfloat16* __restrict__ U_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,   // NT = ceil(seq_len/CHUNK)
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int c = blockIdx.x;     // chunk index
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (c >= num_chunks || vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sk = (__nv_bfloat16*)smem_raw;       // [CHUNK*K_DIM] bf16
    float* kk = (float*)(sk + CHUNK * K_DIM);           // [CHUNK*CHUNK] f32 Gram
    float* L = kk + CHUNK * CHUNK;                      // [CHUNK*CHUNK] f32 decay-weighted strict-lower
    float* gc = L + CHUNK * CHUNK;                      // [CHUNK]

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        sk[i * K_DIM + j] = (i < ce)
            ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + j]
            : __float2bfloat16(0.0f);
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            acc += logf(fmaxf(gate[(unsigned long long)(cs + i) * gb_stride + vh], GATE_FLOOR));
            gc[i] = acc;
        }
    }
    __syncthreads();

    mma_gram<8, CHUNK, false>(sk, sk, kk);   // kk[l][i] = <k_l,k_i>
    __syncthreads();

    // L[i][l] = β_i·exp(gc_i-gc_l)·<k_l,k_i>  for l<i ; 0 otherwise.  (kk symmetric)
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += 128) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l < i) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            L[i * CHUNK + l] = bi * expf(gc[i] - gc[l]) * kk[l * CHUNK + i];
        } else {
            L[i * CHUNK + l] = 0.0f;
        }
    }
    __syncthreads();

    const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

    // Pass 1: U[:,v] forward-sub (one thread per v-element).  U_i = β_i·V_i - Σ_{l<i} L[i][l]·U_l
    if (tid < v_dim) {
        float u[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            float ui = bi * (float)value[(unsigned long long)(cs + i) * v_stride + vh * v_dim + tid];
            for (unsigned int l = 0; l < i; l++) ui -= L[i * CHUNK + l] * u[l];
            u[i] = ui;
            U_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(ui);
        }
    }
    // Pass 2: W[:,k] forward-sub (one thread per k-element).  W_i = β_i·exp(gc_i)·K_i - Σ_{l<i} L[i][l]·W_l
    if (tid < k_dim) {
        float w[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            float wi = bi * expf(gc[i]) * (float)sk[i * K_DIM + tid];
            for (unsigned int l = 0; l < i; l++) wi -= L[i * CHUNK + l] * w[l];
            w[i] = wi;
            W_out[base * CHUNK * K_DIM + i * k_dim + tid] = __float2bfloat16(wi);
        }
    }
}

// chunk_delta_h double-buffer: per-buffer smem holds {W,K,U} bf16 for one chunk.
#define CDH_BUFSZ (CHUNK * (2 * K_DIM + V_DIM))   // 24576 bf16 = 48KB

// Prefetch chunk c's W/U/K into buffer slot p via cp.async, and compute its gc
// (the cumulative log-decay) on tid 0. K is loaded per-row and bounded to i<ce
// (rows cs+i ≥ seq_len would be OOB); W/U load the full block (in-bounds, and the
// i≥ce tail is never read since the compute loops are bounded by ce).
__device__ __forceinline__ void cdh_prefetch(
    __nv_bfloat16* buf, float* gcb, unsigned int p,
    const __nv_bfloat16* __restrict__ W_in, const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key, const float* __restrict__ gate,
    unsigned int c, unsigned int b, unsigned int vh, unsigned int seq_len,
    unsigned int num_chunks, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int kh, unsigned int qk_stride, unsigned int gb_stride
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);
    __nv_bfloat16* Wp = buf + (unsigned long long)p * CDH_BUFSZ;
    __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
    __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
    const unsigned int nthr = blockDim.x;   // 128 (scalar/TC) or 256 (k-split)
    const __nv_bfloat16* Wsrc = W_in + base * CHUNK * K_DIM;
    for (unsigned int e = tid * 8; e < CHUNK * K_DIM; e += nthr * 8) cp_async16(&Wp[e], &Wsrc[e]);
    const __nv_bfloat16* Usrc = U_in + base * CHUNK * V_DIM;
    for (unsigned int e = tid * 8; e < CHUNK * V_DIM; e += nthr * 8) cp_async16(&Up[e], &Usrc[e]);
    for (unsigned int j = tid; j < CHUNK * 16; j += nthr) {
        unsigned int i = j >> 4, c16 = (j & 15) * 8;
        if (i < ce)
            cp_async16(&Kp[i * K_DIM + c16],
                       key + (unsigned long long)(cs + i) * qk_stride + kh * k_dim + c16);
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            acc += logf(fmaxf(gate[(unsigned long long)(cs + i) * gb_stride + vh], GATE_FLOOR));
            gcb[p * CHUNK + i] = acc;
        }
    }
    cp_commit();
}

// ── KERNEL 2: chunk_delta_h ──────────────────────────────────────────────
// The SERIAL state-passing spine — PRECISION-CRITICAL, so S stays f32 and its
// matmuls are fp32-FFMA (NOT bf16-TC: bf16-S drift fails token-equality).
// Grid: (num_v_heads, batch). One CTA per head, serial over chunks. 128 threads
// = v-columns; thread tid owns the WHOLE state column S[:,tid] RESIDENT IN
// REGISTERS (Sreg[K_DIM]) across all chunks — loaded once, updated in-register
// per chunk, stored once. This kills the per-chunk smem read/write of the 64KB
// f32 state, and the freed smem is spent on a DOUBLE BUFFER so the per-chunk
// W/U/K loads (the real bottleneck: global-load latency unhidden at 4 warps)
// are cp.async-prefetched for chunk c+1 while chunk c computes.
// (V-tiling for occupancy REGRESSED — 2 CTAs/head redundantly reload W + re-run
// the serial loop. bf16-TC for the matmuls is precision-SAFE (oracle probe @28k:
// +0.05% output drift) but architecturally BLOCKED on GB10: the 64KB f32 state
// must persist across chunks — in registers (128/thread → collides with TC's
// ~128 fragment-accumulator regs → spills) or in smem (64KB → no room for bf16
// TC operands + f32 outputs under the 99KB cap; V-tiling S to fit = the redundant
// reload that regressed). So scalar register-S + cp.async double-buffer is the
// achieved optimum here; TC needs a ground-up smem-state-tiling rewrite.)
// Per chunk c (entry S_c): store S_c → S_out; uc = U_c - W_c·S_c → uc_out;
// S_{c+1} = exp(gc_last)·S_c + Σ_i exp(gc_last-gc_i)·uc_i·k_i.
// smem: 2×{W(16K)+K(16K)+U(16K)} bf16 + gc[2][CHUNK] = 98816 B.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_delta_h(
    float* __restrict__ h_state,          // [nv][K][V] per (b,vh): entry state IN, final state OUT
    const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    float* __restrict__ S_out,            // [(b*NT+c)*nv+vh][K][V] per-chunk ENTRY states
    __nv_bfloat16* __restrict__ uc_out,   // [(b*NT+c)*nv+vh][C][V] corrected values
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw;          // buf[2][CDH_BUFSZ]
    float* gcb = (float*)(buf + 2 * CDH_BUFSZ);             // gcb[2][CHUNK]

    // State column S[:,tid] resident in registers for this thread's whole lifetime.
    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[K_DIM];
    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) Sreg[k] = H[k * V_DIM + tid];

    // Prologue: kick off chunk 0's loads.
    cdh_prefetch(buf, gcb, 0, W_in, U_in, key, gate, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride);

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

        if (c + 1 < num_chunks) {
            cdh_prefetch(buf, gcb, (c + 1) & 1u, W_in, U_in, key, gate, c + 1, b, vh, seq_len,
                         num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride);
            cp_wait<1>();   // chunk c's loads (older group) complete; keep c+1's in flight
        } else {
            cp_wait<0>();
        }
        __syncthreads();    // make buf[cur] visible CTA-wide

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* gcc = gcb + cur * CHUNK;

        // Store entry state S_c (thread tid owns column tid; fire-and-forget global write).
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = Sreg[k];

        // uc_i = U_i - W_i·S   (W·S contracts over k against the register state column)
        float duc[CHUNK];
        const float dl = gcc[ce - 1];
        const float edl = expf(dl);
        for (unsigned int i = 0; i < ce; i++) {
            float ws = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < K_DIM; k++)
                ws += (float)Wp[i * K_DIM + k] * Sreg[k];
            float uci = (float)Up[i * V_DIM + tid] - ws;
            uc_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(uci);
            duc[i] = expf(dl - gcc[i]) * uci;   // decayed corrected-value, once per i
        }
        // S_{c+1} = edl·S + Σ_i duc_i·k_i   (in-register update, no smem state traffic)
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) {
            float hv = edl * Sreg[k];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kp[i * K_DIM + k];   // pure MAC inner loop
            Sreg[k] = hv;
        }
        __syncthreads();   // before buf[cur] is overwritten by the chunk-(c+2) prefetch
    }

    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) H[k * V_DIM + tid] = Sreg[k];
}

// ── KERNEL 2-TC: chunk_delta_h_tc ────────────────────────────────────────
// State-tiling tensor-core variant of the serial spine (A/B candidate vs the
// scalar register-S kernel above). Same math, same outputs. register-S stays the
// f32 MASTER state (no 64KB smem state); each chunk a bf16 SNAPSHOT Sᵀ[v][k]=
// bf16(S[k][v]) is staged to smem PURELY as an mma operand — the f32 master in
// registers is undamaged, so accumulation precision is unchanged (the snapshot
// is only a per-chunk read, like a bf16 GEMM input; oracle probe @28k = +0.05%).
//   Phase A (TC):  ws[i][v] = Σ_k W[i][k]·Sᵀ[v][k]  via mma_gram → uc=U-ws → duc.
//   Phase B (scalar, increment-1):  S[k][v] = edl·S[k][v] + Σ_i duc_i·K[i][k].
// smem: Sᵀ(32K) + W(16K) + ws(32K f32) + U(16K) = 96.25KB (K reuses Sᵀ for phase B).
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_delta_h_tc(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    float* __restrict__ S_out,
    __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* St = (__nv_bfloat16*)smem_raw;          // [V_DIM*K_DIM] bf16 snapshot Sᵀ
    __nv_bfloat16* Wb = St + V_DIM * K_DIM;                // [CHUNK*K_DIM] bf16
    float* ws = (float*)(Wb + CHUNK * K_DIM);              // [CHUNK*V_DIM] f32 (W·S output)
    __nv_bfloat16* Ub = (__nv_bfloat16*)(ws + CHUNK * V_DIM); // [CHUNK*V_DIM] bf16
    float* gc = (float*)(Ub + CHUNK * V_DIM);              // [CHUNK]
    __nv_bfloat16* Kb = St;                                // phase B reuses Sᵀ region for K

    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[K_DIM];
    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) Sreg[k] = H[k * V_DIM + tid];

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

        // Entry state S_c → S_out (thread tid owns column tid).
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = Sreg[k];

        // Stage bf16 snapshot Sᵀ[v][k] = S[k][v] (thread tid=v writes row v) + load W, gc.
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) St[tid * K_DIM + k] = __float2bfloat16(Sreg[k]);
        for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
            unsigned int i = idx / k_dim, k = idx % k_dim;
            Wb[i * K_DIM + k] = (i < ce) ? W_in[base * CHUNK * K_DIM + i * k_dim + k] : __float2bfloat16(0.0f);
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (unsigned int i = 0; i < ce; i++) {
                acc += logf(fmaxf(gate[(unsigned long long)(cs + i) * gb_stride + vh], GATE_FLOOR));
                gc[i] = acc;
            }
        }
        __syncthreads();

        // Phase A: ws[i][v] = Σ_k W[i][k]·Sᵀ[v][k] = <W_i, S[:,v]>  on tensor cores.
        mma_gram<16, V_DIM, false>(Wb, St, ws);
        __syncthreads();

        // uc = U - ws ; duc = decay·uc  (read ws from smem; no per-element matmul)
        float duc[CHUNK];
        const float dl = gc[ce - 1];
        const float edl = expf(dl);
        for (unsigned int idx = tid; idx < CHUNK * v_dim; idx += 128) {
            unsigned int i = idx / v_dim, v = idx % v_dim;
            Ub[i * V_DIM + v] = (i < ce) ? U_in[base * CHUNK * V_DIM + i * v_dim + v] : __float2bfloat16(0.0f);
        }
        __syncthreads();
        if (tid < v_dim) {
            for (unsigned int i = 0; i < ce; i++) {
                float uci = (float)Ub[i * V_DIM + tid] - ws[i * V_DIM + tid];
                uc_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(uci);
                duc[i] = expf(dl - gc[i]) * uci;
            }
        }
        __syncthreads();   // before Sᵀ region is reused for K

        // Load K into the (freed) Sᵀ region; Phase B scalar S-update (register-S).
        for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
            unsigned int i = idx / k_dim, k = idx % k_dim;
            Kb[i * K_DIM + k] = (i < ce)
                ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + k]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) {
            float hv = edl * Sreg[k];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kb[i * K_DIM + k];
            Sreg[k] = hv;
        }
        __syncthreads();   // before St/Wb/ws reused next chunk
    }

    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) H[k * V_DIM + tid] = Sreg[k];
}

// ── KERNEL 2-KSPLIT: chunk_delta_h_ksplit<SPLIT> ─────────────────────────
// OCCUPANCY variant of the serial spine (A/B vs the scalar/TC kernels above).
// chunk_delta_h is occupancy/latency bound (32 heads = 32 CTAs, only 4 warps each
// → can't hide smem-load/FFMA latency; TC made it WORSE because staging latency is
// also unhidden). Fix: split the K dimension of the state across SPLIT threads per
// v-column → 128·SPLIT threads = 4·SPLIT warps/CTA (more warps to hide latency) on
// the SAME 32 SMs, NO redundant work. Thread (v,sub) owns S[sub·KH .. +KH][v] in
// registers (Sreg[KH], KH=K_DIM/SPLIT). W·S needs the full-k sum → a log2(SPLIT)
// __shfl_xor butterfly across the aligned SPLIT-group of lanes. Same f32 math/output.
// smem: 2×{W,K,U} bf16 double-buffer + 2×gc = 98816 B (same as scalar).
template <int SPLIT>
__device__ __forceinline__ void cdh_ksplit_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, float* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride
) {
    constexpr int KH = K_DIM / SPLIT;            // per-thread slice of the state column
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    const unsigned int t = threadIdx.x;          // 0..128·SPLIT-1
    const unsigned int v = t / SPLIT;            // v-column 0..127
    const unsigned int sub = t % SPLIT;          // which k-slice
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw;          // buf[2][CDH_BUFSZ]
    float* gcb = (float*)(buf + 2 * CDH_BUFSZ);             // gcb[2][CHUNK]

    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    cdh_prefetch(buf, gcb, 0, W_in, U_in, key, gate, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride);

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

        if (c + 1 < num_chunks) {
            cdh_prefetch(buf, gcb, (c + 1) & 1u, W_in, U_in, key, gate, c + 1, b, vh, seq_len,
                         num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* gcc = gcb + cur * CHUNK;

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v] = Sreg[kk];

        const float dl = gcc[ce - 1];
        const float edl = expf(dl);
        float duc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float wsp = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++)
                wsp += (float)Wp[i * K_DIM + k0 + kk] * Sreg[kk];
            #pragma unroll
            for (int s = 1; s < SPLIT; s <<= 1) wsp += __shfl_xor_sync(0xffffffffu, wsp, s);
            float uci = (float)Up[i * V_DIM + v] - wsp;   // wsp == full <W_i, S[:,v]>
            if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v] = __float2bfloat16(uci);
            duc[i] = expf(dl - gcc[i]) * uci;
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++) {
            float hv = edl * Sreg[kk];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kp[i * K_DIM + k0 + kk];
            Sreg[kk] = hv;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++) H[(k0 + kk) * V_DIM + v] = Sreg[kk];
}

// SPLIT=2 (8 warps/CTA) is the chosen production variant: chunk_delta_h 34→26ms,
// FLA total 1.55→1.75x, cos=1.0 vs scalar. SPLIT=4 (16 warps) was tested and gave
// NO further gain (26.3 vs 26.5ms) — 8 warps already saturates the latency hiding,
// so the kernel is no longer occupancy-bound past that. Template kept for the record.
extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_ksplit(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, float* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride
) {
    cdh_ksplit_core<2>(h_state, W_in, U_in, key, gate, S_out, uc_out, seq_len, num_chunks,
                       num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, gb_stride);
}

// ── KERNEL 3: chunk_fwd_o ────────────────────────────────────────────────
// The PARALLEL output pass. Grid: (NT, num_v_heads, batch). One CTA per (chunk,head).
// O_i = (exp(gc_i)·<S_c[:,v],q_i> + Σ_{l<=i} exp(gc_i-gc_l)·<k_l,q_i>·uc_l[v])·rsqrt(d).
// BOTH inner products are tensor-core Gram matmuls (full occupancy → compute bound):
//   kq[i][l] = <q_i,k_l>          (mma_gram, decay folded in)
//   o1[i][v] = <q_i, S_c[:,v]>    (mma_gram with S_c read TRANSPOSED → [V][K])
// S_c read bf16 + o1 bf16 (TERMINAL output → no compounding, like wy4's bf16
// output rounding → precision-safe). o1 reuses the freed sk region. Layout matches wy4.
// smem: sq(16K)+sk/o1(16K)+kq(16K f32)+ucb(16K)+Sbᵀ(32K bf16)+gc = 96.25KB.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_fwd_o(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ S_in,       // [(b*NT+c)*nv+vh][K][V] entry states (from #2)
    const __nv_bfloat16* __restrict__ uc_in,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride
) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (c >= num_chunks || vh >= num_v_heads || b >= batch_size) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);
    const unsigned long long out_base = ((unsigned long long)(b * seq_len) * num_v_heads + vh) * v_dim;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sq = (__nv_bfloat16*)smem_raw;          // [CHUNK*K_DIM]
    __nv_bfloat16* sk = sq + CHUNK * K_DIM;                // [CHUNK*K_DIM]
    float* kq = (float*)(sk + CHUNK * K_DIM);              // [CHUNK*CHUNK]
    __nv_bfloat16* ucb = (__nv_bfloat16*)(kq + CHUNK * CHUNK); // [CHUNK*V_DIM]
    __nv_bfloat16* Sb = ucb + CHUNK * V_DIM;               // [K_DIM*V_DIM] bf16 (S_c)
    float* gc = (float*)(Sb + K_DIM * V_DIM);              // [CHUNK]

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        if (i < ce) {
            unsigned long long off = (unsigned long long)(cs + i) * qk_stride + kh * k_dim + j;
            sq[i * K_DIM + j] = query[off];
            sk[i * K_DIM + j] = key[off];
        } else {
            sq[i * K_DIM + j] = __float2bfloat16(0.0f);
            sk[i * K_DIM + j] = __float2bfloat16(0.0f);
        }
    }
    for (unsigned int idx = tid; idx < CHUNK * v_dim; idx += 128) {
        unsigned int i = idx / v_dim, v = idx % v_dim;
        ucb[i * V_DIM + v] = (i < ce) ? uc_in[base * CHUNK * V_DIM + i * v_dim + v] : __float2bfloat16(0.0f);
    }
    // S_c read TRANSPOSED → Sbᵀ[v][k] = S_c[k][v], so mma_gram(q, Sbᵀ) = <q_i,S_c[:,v]>.
    for (unsigned int idx = tid; idx < K_DIM * V_DIM; idx += 128) {
        unsigned int v = idx / K_DIM, k = idx % K_DIM;
        Sb[idx] = __float2bfloat16(S_in[base * K_DIM * V_DIM + k * V_DIM + v]);
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            acc += logf(fmaxf(gate[(unsigned long long)(cs + i) * gb_stride + vh], GATE_FLOOR));
            gc[i] = acc;
        }
    }
    __syncthreads();

    mma_gram<8, CHUNK, false>(sq, sk, kq);   // kq[i][l] = <q_i, k_l>
    __syncthreads();

    // Fold the intra-chunk decay into the Gram ONCE: kq[i][l] ← exp(gc_i-gc_l)·<q_i,k_l>
    // (was: expf(gc_i-gc_l) recomputed per v-column = v_dim× redundant transcendentals).
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += 128) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l <= i) kq[p] = expf(gc[i] - gc[l]) * kq[p];
    }
    __syncthreads();   // sk is free past mma1 → reuse its region for the o1 = q·Sᵀ result

    // o1[i][v] = <q_i, S_c[:,v]>  on tensor cores (bf16 out → terminal, precision-safe).
    __nv_bfloat16* o1 = sk;                   // [CHUNK*V_DIM] bf16, reuses sk's 16KB
    mma_gram<16, V_DIM, true>(sq, Sb, o1);
    __syncthreads();

    if (tid < v_dim) {
        for (unsigned int i = 0; i < ce; i++) {
            float t1 = expf(gc[i]) * (float)o1[i * V_DIM + tid];
            float t2 = 0.0f;
            for (unsigned int l = 0; l <= i; l++)
                t2 += kq[i * CHUNK + l] * (float)ucb[l * V_DIM + tid];   // pure MAC inner loop
            output[out_base + (unsigned long long)(cs + i) * num_v_heads * v_dim + tid] =
                __float2bfloat16((t1 + t2) * inv_sqrt_d);
        }
    }
}
