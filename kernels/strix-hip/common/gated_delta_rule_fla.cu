// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN prefill — FLA-style MULTI-KERNEL decomposition, AMD HIP/WMMA port
// for gfx1151 (Strix Halo, RDNA 3.5, wave32, 64 KB LDS/workgroup).
//
// Ported from the NVIDIA SM121 source kernels/gb10/common/gated_delta_rule_fla.cu
// (read that file for the full algorithm derivation; the math here is UNCHANGED).
// The CHUNK-64 gb10 kernel cannot run on gfx1151 because (a) chunk_fwd_o's smem is
// ~96 KB (> the 64 KB LDS cap), (b) it uses cp.async.cg PTX (no AMD equivalent),
// and (c) it uses mma.sync.m16n8k16 PTX (no AMD equivalent). This port applies the
// three mechanical transforms below at CHUNK=32; the decomposition is exact for ANY
// CHUNK (per-chunk log-space decay, f32 state spine preserved), so token-equality
// (cos≈1.0 vs the recurrent SSOT) holds identically to the gb10 build.
//
//   TRANSFORM 1 — CHUNK 64 → 32 (gb10 line 23 `#define CHUNK 64`).
//     Halves every chunk-dependent LDS term. At C=32 all three kernels fit < 64 KB
//     (see per-kernel budgets at each kernel header). chunk_delta_h's cp.async
//     DOUBLE buffer collapses to a SINGLE buffer (no `cur=c&1` toggle) because the
//     synchronous copy below makes prefetch == compute-blocking.
//
//   TRANSFORM 2 — cp.async.cg → synchronous 16-byte HIP smem copy.
//     The only cp.async usage is chunk_delta_h's `cdh_prefetch` driven by the
//     `cp_async16`/`cp_commit`/`cp_wait` helpers (gb10 lines 35-41, 206-240,
//     300-316, 514-530). Replaced with the validated `cp16()` idiom from
//     kernels/strix-hip/common/inferspark_prefill.cu (line 62-64:
//     `*(uint4*)smem_dst = *(const uint4*)gmem_src`) + `__syncthreads()`: prefetch
//     ONE chunk into the single buffer, __syncthreads, compute, __syncthreads,
//     repeat. No commit/wait groups, no double-buffer toggle.
//
//   TRANSFORM 3 — mma.sync.m16n8k16 → AMD WMMA m16n16k16.
//     The only MMA primitive is `mma_gram<>` (gb10 lines 45-98). Ported to one
//     `wmma_gram<>` helper using `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`
//     (wave32), following the PROVEN fragment idiom in
//     kernels/strix-hip/common/dense_gemm_tc.cu (lines 80-97) and
//     inferspark_prefill.cu (lines 26-32). See the wmma_gram header for the full
//     n=8(NVIDIA m16n8k16) → n=16(AMD m16n16k16) N-tiling derivation.
//
// GUARDS unchanged: GATE_FLOOR=1e-30, all f32 state/accumulation exactly as gb10.
// gate[] LINEAR decay on prefill; chunk decay in LOG space.

#include <cuda_bf16.h>

// AMD WMMA fragment vector types (wave32). Same as dense_gemm_tc.cu / inferspark.
#if defined(__HIP_PLATFORM_AMD__) || defined(__SCALE__)
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
#endif

#define K_DIM 128
#define V_DIM 128
// TRANSFORM 1: gb10 had `#define CHUNK 64`. Halving to 32 lands every kernel's
// chunk-dependent LDS under the gfx1151 64 KB/workgroup cap (budgets per kernel).
#define CHUNK 32
// Floor for the linear gate before log-space cumsum. Deep-layer gates can underflow
// to exactly 0.0 (or tiny negatives) → log(0)=-inf → exp(gc_i-gc_l)=NaN. 1e-30 ⇒
// log≈-69 (≈full decay), no-op for any normal gate. (gb10 line 29, UNCHANGED.)
#define GATE_FLOOR 1e-30f

// ── TRANSFORM 2: synchronous 16-byte smem copy (replaces cp_async16) ─────────
// Validated idiom from inferspark_prefill.cu line 62-64. `dst`/`src` MUST be
// 16-byte aligned (callers copy 8-bf16 = 16-byte spans on 16-aligned offsets,
// exactly as the gb10 cp.async loops did). No commit/wait — the copy completes
// before the next instruction; a `__syncthreads()` after the prefetch loop makes
// the staged buffer CTA-visible (mirrors gb10's cp_wait + __syncthreads).
__device__ __forceinline__ void cp16(void* smem_dst, const void* gmem_src) {
    *(uint4*)smem_dst = *(const uint4*)gmem_src;
}

// ── TRANSFORM 3: AMD WMMA Gram helper (replaces mma_gram) ────────────────────
// Computes C[m][n] = Σ_k A[m][k]·B[n][k], M=CHUNK, K=K_DIM(128), N=NN16*16.
// A is [M][K_DIM] row-major bf16; B is [N][K_DIM] row-major bf16 (so the op is
// C = A · Bᵀ — B's rows are the N-columns of the output, contracted over K).
//
// gb10's `mma_gram<NTC,NSTRIDE,OutBf16>` issued NVIDIA m16n8k16 ops, tiling N in
// units of 8 (NTC = N/8). AMD's `wmma_f32_16x16x16` issues ONE m16n16k16 op that
// produces a full 16(M)×16(N) tile — i.e. ONE AMD N-tile == TWO NVIDIA n8 tiles.
// So an `mma_gram<NTC,..>` maps to `wmma_gram<NTC/2,..>` (NTC is even — 8 or 16 —
// at every call site, so NTC/2 AMD N-tiles cover N exactly). N = NN16*16 = NTC*8. ✓
//   gb10 mma_gram<8, CHUNK,..>  (N=CHUNK=64 @C64; N=32 @C32) → here NN16=2, N=32. ✓ (C=32)
//   gb10 mma_gram<16,V_DIM,..>  (N=128)                       → here NN16=8, N=128.   ✓
//
// 128 threads = 4 warps; each warp owns 16 M-rows (warp_m = warp*16). With M=CHUNK
// =32 only warps 0,1 hold valid M-rows; warps 2,3 (warp_m=32,48 ≥ CHUNK) are guarded
// off. (gb10 ran M=64 across all 4 warps; the M-guard is new and C=32-specific.)
//
// Fragment idiom — VALIDATED in dense_gemm_tc.cu (lines 80-97) + inferspark (26-32):
//   A (M×K row-major): lane l → a[i] = A[(warp_m + (l&15))*K_DIM + ks + i], i=0..15
//   B (N×K row-major): lane l → b[k] = B[(nt*16 + (l&15))*K_DIM + ks + k], k=0..15
//   C/D store:         lane l, elem e(0..7) → row = warp_m + 2*e + (l>>4),
//                                             col = nt*16 + (l&15)
//   (the A/B "full-K-row/col into lane (l&15)" load + the "row=2*e+(l>>4), col=l&15"
//    store map are the exact dense_gemm_tc / inferspark fragment maps.)
template <int NN16, int NSTRIDE, bool OutBf16>
__device__ __forceinline__ void wmma_gram(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, void* __restrict__ C
) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__SCALE__)
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    const unsigned ll = lane & 15;       // 0..15 → WMMA A-row / B-col / store-col
    const unsigned lh = lane >> 4;       // 0 or 1 → store-row parity
    const unsigned warp_m = warp * 16;
    if (warp_m >= CHUNK) return;         // M=CHUNK=32 → only warps 0,1 active

    v8f acc[NN16];
    #pragma unroll
    for (int nt = 0; nt < NN16; nt++) acc[nt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll
    for (unsigned ks = 0; ks < K_DIM; ks += 16) {
        // A-fragment: M-row (warp_m+ll), 16 contiguous K elems from ks.
        v16bf a;
        #pragma unroll
        for (int i = 0; i < 16; i++)
            a[i] = (__bf16)A[(warp_m + ll) * K_DIM + ks + i];

        #pragma unroll
        for (int nt = 0; nt < NN16; nt++) {
            // B-fragment: N-col (nt*16+ll), 16 contiguous K elems from ks
            // (B is [N][K_DIM] row-major; row N-col, cols ks..ks+15).
            v16bf b;
            #pragma unroll
            for (int k = 0; k < 16; k++)
                b[k] = (__bf16)B[(nt * 16 + ll) * K_DIM + ks + k];
            acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nt]);
        }
    }

    // Store: C[m][n] with row-stride NSTRIDE. (row=warp_m+2*e+lh, col=nt*16+ll)
    #pragma unroll
    for (int nt = 0; nt < NN16; nt++) {
        const unsigned col = nt * 16 + ll;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned row = warp_m + 2 * e + lh;
            if (OutBf16) ((__nv_bfloat16*)C)[row * NSTRIDE + col] = __float2bfloat16(acc[nt][e]);
            else         ((float*)C)[row * NSTRIDE + col] = acc[nt][e];
        }
    }
#else
    // Non-AMD fallback: this file is the gfx1151 port; the gb10 build uses
    // kernels/gb10/common/gated_delta_rule_fla.cu. Keep the symbol resolvable for
    // host-side type checks but never reached on a non-HIP/SCALE target.
    (void)A; (void)B; (void)C;
#endif
}

// ── KERNEL 1: recompute_w_u ──────────────────────────────────────────────
// (gb10 lines 100-197.) Grid: (NT, num_v_heads, batch)  Block: (128,1,1).
// One CTA per (chunk, head). Outputs (f32, layout [(b*NT+c)*nv+vh][CHUNK][·]):
//   U_out: T·(βV) ; W_out: T·(β·exp(gc)·K) ; T=(I+L)⁻¹ via forward-substitution.
//
// LDS BUDGET @ C=32:  sk(bf16 CHUNK*K_DIM = 32*128*2 = 8192)
//                   + kk(f32  CHUNK*CHUNK = 32*32*4   = 4096)
//                   + L (f32  CHUNK*CHUNK = 32*32*4   = 4096)
//                   + gc(f32  CHUNK       = 32*4      =  128)  = 16512 B (~16.1 KB). ✓
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_recompute_wu(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ W_out,
    __nv_bfloat16* __restrict__ U_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int c = blockIdx.x;
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
    float* L = kk + CHUNK * CHUNK;                      // [CHUNK*CHUNK] f32 strict-lower
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

    // kk[l][i] = <k_l,k_i>. gb10: mma_gram<8,CHUNK,false> (N=CHUNK). C=32 ⇒ NN16=CHUNK/16=2.
    wmma_gram<CHUNK / 16, CHUNK, false>(sk, sk, kk);
    __syncthreads();

    // L[i][l] = β_i·exp(gc_i-gc_l)·<k_l,k_i> for l<i ; 0 otherwise. (kk symmetric)
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

    // Pass 1: U[:,v] forward-sub. U_i = β_i·V_i - Σ_{l<i} L[i][l]·U_l
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
    // Pass 2: W[:,k] forward-sub. W_i = β_i·exp(gc_i)·K_i - Σ_{l<i} L[i][l]·W_l
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

// chunk_delta_h SINGLE-buffer: smem holds {W,K,U} bf16 for one chunk.
// (gb10 line 200 `CDH_BUFSZ` was per-buffer of a DOUBLE buffer; here it is the
// single live buffer — TRANSFORM 1+2 collapse.)
#define CDH_BUFSZ (CHUNK * (2 * K_DIM + V_DIM))   // @C32: 32*384 = 12288 bf16 = 24576 B

// Prefetch chunk c's W/U/K into the single buffer via synchronous cp16, and compute
// its gc on tid 0. (gb10 cdh_prefetch lines 206-240, with cp_async16→cp16, no commit.)
// `p` is retained for call-site symmetry but is always 0 (single buffer).
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
    for (unsigned int e = tid * 8; e < CHUNK * K_DIM; e += nthr * 8) cp16(&Wp[e], &Wsrc[e]);
    const __nv_bfloat16* Usrc = U_in + base * CHUNK * V_DIM;
    for (unsigned int e = tid * 8; e < CHUNK * V_DIM; e += nthr * 8) cp16(&Up[e], &Usrc[e]);
    for (unsigned int j = tid; j < CHUNK * 16; j += nthr) {
        unsigned int i = j >> 4, c16 = (j & 15) * 8;
        if (i < ce)
            cp16(&Kp[i * K_DIM + c16],
                 key + (unsigned long long)(cs + i) * qk_stride + kh * k_dim + c16);
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            acc += logf(fmaxf(gate[(unsigned long long)(cs + i) * gb_stride + vh], GATE_FLOOR));
            gcb[p * CHUNK + i] = acc;
        }
    }
    // (gb10 cp_commit() removed — cp16 is synchronous; CTA-visibility comes from
    // the __syncthreads() the caller issues after this returns.)
}

// ── KERNEL 2: chunk_delta_h (scalar register-S, single-buffer) ───────────────
// (gb10 lines 242-354.) The SERIAL state-passing spine — PRECISION-CRITICAL, so S
// stays f32 and its matmuls are fp32-FFMA. Grid: (num_v_heads, batch). 128 threads
// = v-columns; thread tid owns the WHOLE state column S[:,tid] in registers
// (Sreg[K_DIM]) across all chunks. TRANSFORM 1+2: double buffer → single buffer,
// cp.async → cp16; per chunk: prefetch → __syncthreads → compute → __syncthreads.
//
// LDS BUDGET @ C=32:  buf({W,K,U} bf16 CHUNK*(2K+V) = 12288*2 = 24576 B)
//                   + gcb(f32 CHUNK = 128 B)                    = 24704 B (~24.1 KB). ✓
//   (gb10 was 2× buffers + 2× gc = 98816 B; single-buffer halves the buffer term.)
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_delta_h(
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
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw;          // buf[CDH_BUFSZ] (single)
    float* gcb = (float*)(buf + CDH_BUFSZ);                 // gcb[CHUNK]

    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[K_DIM];
    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) Sreg[k] = H[k * V_DIM + tid];

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

        // SINGLE-buffer synchronous prefetch of THIS chunk (p=0), then make visible.
        cdh_prefetch(buf, gcb, 0, W_in, U_in, key, gate, c, b, vh, seq_len,
                     num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride);
        __syncthreads();

        __nv_bfloat16* Wp = buf;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* gcc = gcb;

        // Store entry state S_c (thread tid owns column tid).
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = Sreg[k];

        // uc_i = U_i - W_i·S (W·S contracts over k against the register state column)
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
            duc[i] = expf(dl - gcc[i]) * uci;
        }
        // S_{c+1} = edl·S + Σ_i duc_i·k_i (in-register update, no smem state traffic)
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) {
            float hv = edl * Sreg[k];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kp[i * K_DIM + k];
            Sreg[k] = hv;
        }
        __syncthreads();   // before buf is overwritten by the next chunk's prefetch
    }

    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) H[k * V_DIM + tid] = Sreg[k];
}

// ── KERNEL 2-TC: chunk_delta_h_tc ────────────────────────────────────────
// (gb10 lines 356-473.) State-tiling tensor-core variant of the serial spine.
// register-S stays the f32 MASTER state; each chunk a bf16 SNAPSHOT Sᵀ[v][k] is
// staged to smem PURELY as a wmma operand (f32 master undamaged → precision
// unchanged). Phase A (WMMA): ws[i][v] = Σ_k W[i][k]·Sᵀ[v][k]; Phase B (scalar):
// S[k][v] = edl·S[k][v] + Σ_i duc_i·K[i][k].
//
// LDS BUDGET @ C=32:  St(bf16 V_DIM*K_DIM = 128*128*2 = 32768)
//                   + Wb(bf16 CHUNK*K_DIM = 32*128*2  =  8192)
//                   + ws(f32  CHUNK*V_DIM = 32*128*4  = 16384)
//                   + Ub(bf16 CHUNK*V_DIM = 32*128*2  =  8192)
//                   + gc(f32  CHUNK = 128)                       = 65664 B (~64.1 KB).
//   ⚠ This MARGINALLY EXCEEDS 64 KB (65664 > 65536). St (the V*K snapshot) is
//     CHUNK-INDEPENDENT (32 KB fixed), so CHUNK=32 does not shrink it. The
//     production path is chunk_delta_h_ksplit (24 KB, below) — this TC variant is
//     an A/B candidate that does NOT fit gfx1151 LDS as written. Kept for parity
//     with the e2e harness; see report note (d). Compile may fail the LDS cap.
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

        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = Sreg[k];

        // Stage bf16 snapshot Sᵀ[v][k] = S[k][v] (thread tid=v writes row v) + W, gc.
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

        // Phase A: ws[i][v] = Σ_k W[i][k]·Sᵀ[v][k]. gb10: mma_gram<16,V_DIM,false> → NN16=8.
        wmma_gram<V_DIM / 16, V_DIM, false>(Wb, St, ws);
        __syncthreads();

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

// ── KERNEL 2-KSPLIT: chunk_delta_h_ksplit<SPLIT> (PRODUCTION serial spine) ────
// (gb10 lines 475-584.) OCCUPANCY variant: split the K dim of the state across
// SPLIT threads per v-column → 128·SPLIT threads = more warps to hide latency.
// Thread (v,sub) owns S[sub·KH .. +KH][v] in registers (Sreg[KH], KH=K_DIM/SPLIT).
// W·S needs the full-k sum → a log2(SPLIT) __shfl_xor butterfly across the aligned
// SPLIT-group of lanes (the build mirror widens the 32-bit mask to 64-bit for the
// wavefront; see crates/atlas-kernels/build.rs widen_warp_masks — SPLIT=2 partners
// are adjacent lanes 2v,2v+1, correct on wave32 regardless). Same f32 math/output.
// TRANSFORM 1+2: double buffer → single buffer, cp.async → cp16.
//
// LDS BUDGET @ C=32:  buf({W,K,U} bf16 CHUNK*(2K+V) = 24576 B)
//                   + gcb(f32 CHUNK = 128 B)                     = 24704 B (~24.1 KB). ✓
//   (gb10 was 2×buffer + 2×gc = 98816 B; single-buffer → 24.1 KB. Identical to the
//    scalar kernel above — ksplit's extra warps don't add smem.)
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
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw;          // buf[CDH_BUFSZ] (single)
    float* gcb = (float*)(buf + CDH_BUFSZ);                 // gcb[CHUNK]

    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

        // SINGLE-buffer synchronous prefetch of THIS chunk (p=0), then make visible.
        cdh_prefetch(buf, gcb, 0, W_in, U_in, key, gate, c, b, vh, seq_len,
                     num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride);
        __syncthreads();

        __nv_bfloat16* Wp = buf;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* gcc = gcb;

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
            for (int s = 1; s < SPLIT; s <<= 1) wsp += __shfl_xor_sync(0xffffffffULL, wsp, s);
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
        __syncthreads();   // before buf is overwritten by the next chunk's prefetch
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++) H[(k0 + kk) * V_DIM + v] = Sreg[kk];
}

// SPLIT=2 (8 warps/CTA) production variant. (gb10 lines 573-584.)
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
// (gb10 lines 586-689.) The PARALLEL output pass. Grid: (NT, num_v_heads, batch).
// One CTA per (chunk,head).
// O_i = (exp(gc_i)·<S_c[:,v],q_i> + Σ_{l<=i} exp(gc_i-gc_l)·<k_l,q_i>·uc_l[v])·rsqrt(d).
// Both inner products are WMMA Gram matmuls. S_c read bf16 + o1 bf16 (terminal →
// precision-safe). o1 reuses the freed sk region.
//
// LDS BUDGET @ C=32:  sq(bf16 CHUNK*K_DIM = 32*128*2 = 8192)
//                   + sk(bf16 CHUNK*K_DIM = 32*128*2 = 8192)   (o1 reuses this region)
//                   + kq(f32  CHUNK*CHUNK = 32*32*4  = 4096)
//                   + ucb(bf16 CHUNK*V_DIM = 32*128*2 = 8192)
//                   + Sb(bf16 K_DIM*V_DIM = 128*128*2 = 32768)  (CHUNK-INDEPENDENT)
//                   + gc(f32  CHUNK = 32*4 = 128)               = 61568 B (~60.1 KB). ✓
//   (The fixed K_DIM*V_DIM*2 = 32 KB Sb term dominates; the C=32 halving of the
//    other terms is what brings the total from gb10's ~96 KB under 64 KB.)
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_fwd_o(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ S_in,
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
    // S_c read TRANSPOSED → Sbᵀ[v][k] = S_c[k][v], so wmma_gram(q, Sbᵀ) = <q_i,S_c[:,v]>.
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

    // kq[i][l] = <q_i, k_l>. gb10: mma_gram<8,CHUNK,false> (N=CHUNK). C=32 ⇒ NN16=2.
    wmma_gram<CHUNK / 16, CHUNK, false>(sq, sk, kq);
    __syncthreads();

    // Fold intra-chunk decay into the Gram ONCE: kq[i][l] ← exp(gc_i-gc_l)·<q_i,k_l>.
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += 128) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l <= i) kq[p] = expf(gc[i] - gc[l]) * kq[p];
    }
    __syncthreads();   // sk free past mma1 → reuse its region for the o1 = q·Sᵀ result

    // o1[i][v] = <q_i, S_c[:,v]>. gb10: mma_gram<16,V_DIM,true> (N=V_DIM) → NN16=8.
    __nv_bfloat16* o1 = sk;                   // [CHUNK*V_DIM] bf16, reuses sk's region
    wmma_gram<V_DIM / 16, V_DIM, true>(sq, Sb, o1);
    __syncthreads();

    if (tid < v_dim) {
        for (unsigned int i = 0; i < ce; i++) {
            float t1 = expf(gc[i]) * (float)o1[i * V_DIM + tid];
            float t2 = 0.0f;
            for (unsigned int l = 0; l <= i; l++)
                t2 += kq[i * CHUNK + l] * (float)ucb[l * V_DIM + tid];
            output[out_base + (unsigned long long)(cs + i) * num_v_heads * v_dim + tid] =
                __float2bfloat16((t1 + t2) * inv_sqrt_d);
        }
    }
}
