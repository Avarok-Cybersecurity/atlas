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
#include <cuda.h>
#include <mma.h>   // nvcuda::wmma fragment types (tc_vblock variant; mma_gram itself
                   // stays raw mma.sync PTX — ldmatrix is broken on sm_121).

#define K_DIM 128
#define V_DIM 128
#define CHUNK 64
// Floor for the linear gate before log-space cumsum. Deep-layer gates can
// underflow to exactly 0.0 (or tiny negatives), and log(0)=-inf → exp(gc_i-gc_l)
// = exp(-inf - -inf) = NaN. The per-token recurrence (h=g·h) tolerates g≈0; the
// chunked log-space form does not. 1e-30 ⇒ log≈-69 (effectively full decay) and
// is a no-op for any normal gate. (GATE-B used g∈[0.8,0.999], never hitting this.)
#define GATE_FLOOR 1e-30f

// Per-stream prefill geometry. VARLEN (ragged co-dispatch batch) reads
// cu_seqlens[b]/cu_chunks[b]; uniform/single-stream (is_varlen=0) reduces to
// b*seq_len / b*num_chunks (byte-identical to the pre-varlen path). Requires
// b, seq_len, num_chunks, cu_seqlens, cu_chunks, is_varlen in scope. `tokoff` is
// the token offset into the [Σ seqlen_b] packed inputs; `choff` the chunk offset
// into the [Σ nchunks_b] scratch; in varlen `num_chunks` (grid x) = MAX nchunks.
struct GdnGeom { unsigned int seqlen, nchunks, choff; unsigned long long tokoff; };
#define GDN_GEOM(g)                                                            \
    GdnGeom g;                                                                 \
    (void)cu_chunks;                                                          \
    if (is_varlen) {                                                           \
        unsigned int _s0 = (unsigned int)cu_seqlens[b];                       \
        g.seqlen  = (unsigned int)cu_seqlens[b + 1] - _s0;                    \
        g.tokoff  = (unsigned long long)_s0;                                  \
        unsigned int _co = 0; /* choff = Σ_{i<b} ceil(len_i/64), in-kernel */ \
        for (unsigned int _i = 0; _i < b; _i++)                               \
            _co += ((unsigned int)(cu_seqlens[_i + 1] - cu_seqlens[_i])       \
                    + CHUNK - 1) / CHUNK;                                      \
        g.choff   = _co;                                                       \
        g.nchunks = (g.seqlen + CHUNK - 1) / CHUNK;                           \
    } else {                                                                  \
        g.seqlen  = seq_len;                                                   \
        g.tokoff  = (unsigned long long)b * seq_len;                          \
        g.choff   = b * num_chunks;                                            \
        g.nchunks = num_chunks;                                                \
    }

// 16-byte async global→shared copy + group
// ★ The line that used to be here said "GB10 sm_121 has cp.async.cg (NO TMA)".
// That is FALSE and it cost real time: `cp.async.bulk.tensor` + `mbarrier`
// COMPILE AND RUN on sm_121a with our own CUDA 13.0 nvcc (probe: a 2D tile load
// with `__grid_constant__ CUtensorMap`, no external headers), and the FlashQLA
// GDN spine measured on THIS box at 15.0 ms uses exactly that. Our whole
// `kernels/` tree still uses zero TMA and zero mbarrier — that gap, not tile
// shape, is why our spine is 119.8 ms against its 15.0 ms. Do not re-derive
// "no TMA" from this file.
// commit/wait — used to double-buffer the per-chunk W/U/K loads in chunk_delta_h
// so the serial state spine overlaps the next chunk's load with the current
// chunk's compute (it was global-load-LATENCY bound at 4 warps, not FLOP bound).
// ── TMA (cp.async.bulk.tensor) + mbarrier primitives ─────────────────────────
// The copy engine generates addresses, tiles, AND bounds-checks: a tile that
// hangs off the end of the tensor is ZERO-FILLED in hardware, so the per-row
// `i < ce` predication `cdh_prefetch` needs disappears, and nothing round-trips
// through the register file the way a `cp.async` still does.
//
// Sequencing per stage: `bar_init` once, then per tile
//   expect_tx(bytes) -> tma_load_2d(...) -> bar_wait(parity)
// The barrier counts BYTES, not arrivals: `expect_tx` states the transaction
// size up front and the copy engine completes it, which is why the consumer can
// wait without the producer arriving explicitly.
//
// `__grid_constant__ const CUtensorMap` is a 128-byte BY-VALUE parameter — see
// `KernelLaunch::arg_tensormap`, which had to learn to pass an argument wider
// than one 8-byte slot for this to be expressible at all.
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::
                 "r"((unsigned int)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::
                 "r"((unsigned int)__cvta_generic_to_shared(bar)), "r"(bytes) : "memory");
}
// Spin until the barrier flips to `parity`. `try_wait.parity` is the documented
// pairing for a transaction barrier; a plain `test_wait` races the phase flip.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned int parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra DONE_%=;\n"
        "bra WAIT_%=;\n"
        "DONE_%=:\n"
        "}\n" ::
        "r"((unsigned int)__cvta_generic_to_shared(bar)), "r"(parity) : "memory");
}
// The descriptor lives in the (read-only) param space while the barrier lives in
// shared; the async proxy needs a fence between a generic-space write and a TMA
// that observes it.
__device__ __forceinline__ void tma_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
// One 2-D tile. `c0` is the fastest-varying coordinate (columns), `c1` the row —
// the same fastest-first ordering `TensorMap::tiled_2d_bf16` encodes, and
// reversing it silently transposes rather than failing.
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* desc, uint64_t* bar, void* dst_smem, int c0, int c1
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n" ::
        "r"((unsigned int)__cvta_generic_to_shared(dst_smem)),
        "l"((const void*)desc),
        "r"((unsigned int)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

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
// 256 threads: mma_gram is fenced to the first 4 warps, and the two independent
// forward-subs run concurrently on the two thread halves instead of back-to-back.
extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_recompute_wu(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ W_out,   // bf16 — per-chunk intermediate, fed to TC matmuls in #2/#3
    __nv_bfloat16* __restrict__ U_out,
    float* __restrict__ gc_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,   // NT = ceil(seq_len/CHUNK)
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int c = blockIdx.x;     // chunk index (grid x = MAX num_chunks)
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDN_GEOM(g);
    if (c >= g.nchunks) return;            // per-stream chunk bound

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

    // Per-stream input offset (cu_seqlens[b] in varlen, else b*seq_len; b=0 → +0).
    key   += g.tokoff * qk_stride;
    value += g.tokoff * v_stride;
    gate  += g.tokoff * gb_stride;
    beta  += g.tokoff * gb_stride;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sk = (__nv_bfloat16*)smem_raw;       // [CHUNK*K_DIM] bf16
    float* kk = (float*)(sk + CHUNK * K_DIM);           // [CHUNK*CHUNK] f32 Gram
    // L ALIASES kk. Safe because the two live in disjoint triangles: the build below
    // writes L only at (i,l) with l<i (strictly LOWER) and reads kk only at (l,i)
    // with l<i (strictly UPPER, kk being symmetric), and both forward-subs consume
    // L only for l<i. Saves 16 KB, which takes this kernel from 2 to 3 CTAs/SM.
    float* L = kk;                                     // [CHUNK*CHUNK] f32 strict-lower, aliased
    float* gc = L + CHUNK * CHUNK;                      // [CHUNK]

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += blockDim.x) {
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
            gc_out[base * CHUNK + i] = acc;
        }
    }
    __syncthreads();

    if (tid < 128) mma_gram<8, CHUNK, false>(sk, sk, kk);   // kk[l][i] = <k_l,k_i>
    __syncthreads();

    // L[i][l] = β_i·exp(gc_i-gc_l)·<k_l,k_i>  for l<i ; 0 otherwise.  (kk symmetric)
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += blockDim.x) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        // Only the strict-lower half is written — the old `else` branch zeroed the
        // rest, which is both unread (every consumer loops l<i) and, under the
        // alias, would clobber the upper-triangle kk values still being read.
        if (i < ce && l < i) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            L[i * CHUNK + l] = bi * expf(gc[i] - gc[l]) * kk[l * CHUNK + i];
        }
    }
    __syncthreads();

    // Passes 1 and 2 are INDEPENDENT: both only read L/gc/beta and they write
    // disjoint outputs (U_out vs W_out). At 128 threads they ran back-to-back;
    // at 256 they run concurrently on disjoint thread halves, which halves the
    // serial triangular-solve section that dominates this kernel.
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
    const unsigned int wtid = tid - 128u;   // Pass 2 runs on the upper half
    if (tid >= 128u && wtid < k_dim) {
        float w[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            float wi = bi * expf(gc[i]) * (float)sk[i * K_DIM + wtid];
            for (unsigned int l = 0; l < i; l++) wi -= L[i * CHUNK + l] * w[l];
            w[i] = wi;
            W_out[base * CHUNK * K_DIM + i * k_dim + wtid] = __float2bfloat16(wi);
        }
    }
}

// chunk_delta_h double-buffer: per-buffer smem holds {W,K,U} bf16 for one chunk.
#define CDH_BUFSZ (CHUNK * (2 * K_DIM + V_DIM))   // 24576 bf16 = 48KB

// Prefetch chunk c's W/U/K into buffer slot p via cp.async, and load its gc
// plus decays on tid 0. K is loaded per-row and bounded to i<ce
// (rows cs+i ≥ seq_len would be OOB); W/U load the full block (in-bounds, and the
// i≥ce tail is never read since the compute loops are bounded by ce).
__device__ __forceinline__ void cdh_prefetch(
    __nv_bfloat16* buf, float* gcb, float* decb, unsigned int p,
    const __nv_bfloat16* __restrict__ W_in, const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key, const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    unsigned int c, unsigned int b, unsigned int vh, unsigned int seq_len,
    unsigned int num_chunks, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int kh, unsigned int qk_stride, unsigned int gb_stride,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int tid = threadIdx.x;
    GDN_GEOM(g);
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    key += g.tokoff * qk_stride;   // per-stream token offset (cu_seqlens / uniform b*seq_len)
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
    // Decay table, ONE THREAD PER ENTRY. This used to be a `tid == 0` loop of up
    // to CHUNK serial `expf` calls while every other thread idled at the next
    // barrier — per chunk, on the serial state spine. Each entry depends only on
    // its own `gc_in`, so there was never a reason to serialize it.
    {
        const float dl = gc_in[base * CHUNK + ce - 1];
        if (tid == 0) decb[p * (CHUNK + 1)] = expf(dl);
        for (unsigned int i = tid; i < ce; i += nthr) {
            const float gv = gc_in[base * CHUNK + i];
            gcb[p * CHUNK + i] = gv;
            decb[p * (CHUNK + 1) + 1 + i] = expf(dl - gv);
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
// Per chunk c (entry S_c): store bf16(S_c) → S_out; uc = U_c - W_c·S_c → uc_out;
// S_{c+1} = exp(gc_last)·S_c + Σ_i exp(gc_last-gc_i)·uc_i·k_i.
// smem: 2×{W(16K)+K(16K)+U(16K)} bf16 + gc[2][CHUNK] + decay[2][CHUNK+1].
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_delta_h(
    float* __restrict__ h_state,          // [nv][K][V] per (b,vh): entry state IN, final state OUT
    const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out,    // [(b*NT+c)*nv+vh][K][V] per-chunk ENTRY states
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
    float* decb = gcb + 2 * CHUNK;                          // decb[2][CHUNK+1], [0]=exp(gc_last)

    // State column S[:,tid] resident in registers for this thread's whole lifetime.
    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[K_DIM];
    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) Sreg[k] = H[k * V_DIM + tid];

    // Prologue: kick off chunk 0's loads.
    cdh_prefetch(buf, gcb, decb, 0, W_in, U_in, key, gate, gc_in, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 nullptr, nullptr, 0);  // scalar chunk_delta_h: uniform only

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

        if (c + 1 < num_chunks) {
            cdh_prefetch(buf, gcb, decb, (c + 1) & 1u, W_in, U_in, key, gate, gc_in, c + 1, b, vh, seq_len,
                         num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 nullptr, nullptr, 0);  // scalar chunk_delta_h: uniform only
            cp_wait<1>();   // chunk c's loads (older group) complete; keep c+1's in flight
        } else {
            cp_wait<0>();
        }
        __syncthreads();    // make buf[cur] visible CTA-wide

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* dec = decb + cur * (CHUNK + 1);

        // Store entry state S_c for the output pass. The recurrent master stays f32 in Sreg/H;
        // chunk_fwd_o consumes S_c as a bf16 MMA operand, so f32 scratch only adds traffic.
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = __float2bfloat16(Sreg[k]);

        // uc_i = U_i - W_i·S   (W·S contracts over k against the register state column)
        float duc[CHUNK];
        const float edl = dec[0];
        for (unsigned int i = 0; i < ce; i++) {
            float ws = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < K_DIM; k++)
                ws += (float)Wp[i * K_DIM + k] * Sreg[k];
            float uci = (float)Up[i * V_DIM + tid] - ws;
            uc_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(uci);
            duc[i] = dec[1 + i] * uci;   // decayed corrected-value, once per i
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
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out,
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

        // Entry state bf16(S_c) → S_out (thread tid owns column tid).
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = __float2bfloat16(Sreg[k]);

        // Stage bf16 snapshot Sᵀ[v][k] = S[k][v] (thread tid=v writes row v) + load W, gc.
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) St[tid * K_DIM + k] = __float2bfloat16(Sreg[k]);
        for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
            unsigned int i = idx / k_dim, k = idx % k_dim;
            Wb[i * K_DIM + k] = (i < ce) ? W_in[base * CHUNK * K_DIM + i * k_dim + k] : __float2bfloat16(0.0f);
        }
        for (unsigned int i = tid; i < ce; i += 128) {
            gc[i] = gc_in[base * CHUNK + i];
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
// smem: 2×{W,K,U} bf16 double-buffer + 2×gc + 2×decay (same shape as scalar).
template <int SPLIT>
__device__ __forceinline__ void cdh_ksplit_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;            // per-thread slice of the state column
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;          // 0..128·SPLIT-1
    const unsigned int v = t / SPLIT;            // v-column 0..127
    const unsigned int sub = t % SPLIT;          // which k-slice
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // Distinct name from the sibling `extern __shared__ char smem_raw[]` in this
    // same kernel (~L445): two decls of one dynamic-smem symbol in a function trip
    // nvcc #1556-D. All `extern __shared__` views alias the same base — byte-identical.
    extern __shared__ char smem_raw_dhc[];
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw_dhc;          // buf[2][CDH_BUFSZ]
    float* gcb = (float*)(buf + 2 * CDH_BUFSZ);             // gcb[2][CHUNK]
    float* decb = gcb + 2 * CHUNK;                          // decb[2][CHUNK+1], [0]=exp(gc_last)

    // Batched co-dispatch passes h_state as a device POINTER TABLE (one contiguous
    // [nv,kd,vd] h_state per request, the same table wy64 uses); single-stream
    // passes a contiguous base (is_table=0 → byte-identical to the original).
    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    cdh_prefetch(buf, gcb, decb, 0, W_in, U_in, key, gate, gc_in, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 cu_seqlens, cu_chunks, is_varlen);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) {
            cdh_prefetch(buf, gcb, decb, (c + 1) & 1u, W_in, U_in, key, gate, gc_in, c + 1, b, vh, seq_len,
                         num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                         cu_seqlens, cu_chunks, is_varlen);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* dec = decb + cur * (CHUNK + 1);

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v] = __float2bfloat16(Sreg[kk]);

        const float edl = dec[0];
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
            duc[i] = dec[1 + i] * uci;
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
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_ksplit_core<2>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len, num_chunks,
                       num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, gb_stride, h_state_is_table,
                       cu_seqlens, cu_chunks, is_varlen);
}

// ── KERNEL 2-VTILE: chunk_delta_h_vtile<SPLIT,VT> ────────────────────────────
// TRAFFIC variant. An ncu memory-pipeline profile of ksplit at the 35B GDN shape
// says the spine is neither FLOP- nor occupancy-bound:
//
//   lts__throughput (L2)        60.6%   <- highest utilisation
//   l1tex__throughput (L1/smem) 56.2%   <- second
//   smsp__warps_active          16.4%
//   inst/cycle/scheduler         0.31   (issue-starved BY memory, not by warps)
//   avg warp latency/inst        6.35 cycles
//
// It is MEMORY-PIPELINE bound, and that one fact explains the two failed attempts:
// tc_vblock and dvsplit both split the DV axis across CTAs, which DUPLICATES the
// K-dimensioned W and K loads (they are not v-indexed) — paying L2, the saturated
// resource, to buy occupancy, which was never the constraint. dvsplit measured 0.97x
// and tc_vblock 0.89x for exactly that reason.
//
// This variant instead REDUCES traffic, and keeps one CTA per head so W/K are loaded
// exactly once:
//
//   1. Register-tile the v axis (VT columns per thread). Every W and K smem read is
//      reused across VT outputs, so L1 read volume drops by VT.
//   2. Fuse the two per-chunk passes. ksplit computes duc[i] for all i (a 64-float
//      per-thread array, 64 of its 165 registers) and only then updates S. Keeping
//      S_old and S_new separately lets duc collapse to a scalar per v: the update is
//      applied inside the same i-loop that produced it. That deletes the array, and
//      with it the register pressure that capped ksplit at 1 CTA/SM.
//   3. Stage only W and K (32 KB). U is streamed from global — each element is read
//      once per thread anyway, so smem buys nothing and costs 16 KB.
//
// Per-thread smem reads per chunk fall from 8192 (ksplit: 64i x 64kk twice) to 2048
// (64i x 16kk twice) — a 4x cut in L1 traffic, against a 56%-utilised L1.
//
// Correctness is unchanged and exact: for every token i the wsp reduction still uses
// the chunk-start state, because S_old is never written during the loop; only S_new
// accumulates. Same f32 accumulation, same shfl butterfly over the SPLIT group.
// Drop-in ABI == ksplit, so chunk_fwd_o is untouched.
template <int SPLIT, int VT>
__device__ __forceinline__ void cdh_vtile_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;          // per-thread slice of the state column
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;        // 0 .. (V_DIM/VT)*SPLIT - 1
    const unsigned int v0 = (t / SPLIT) * VT;  // first of this thread's VT v-columns
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw_vt[];
    __nv_bfloat16* Wp = (__nv_bfloat16*)smem_raw_vt;   // [CHUNK*K_DIM]
    __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;            // [CHUNK*K_DIM]
    __nv_bfloat16* Up = Kp + CHUNK * K_DIM;            // [CHUNK*V_DIM]
    float* decs = (float*)(Up + CHUNK * V_DIM);        // [CHUNK+1], [0]=exp(gc_last)

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sold[KH][VT];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = H[(k0 + kk) * V_DIM + v0 + vt];

    const __nv_bfloat16* key_b = key + g.tokoff * qk_stride;
    const unsigned int nthr = blockDim.x;

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        __syncthreads();
        const __nv_bfloat16* Wsrc = W_in + base * CHUNK * K_DIM;
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) Wp[e] = Wsrc[e];
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) {
            unsigned int i = e / K_DIM, kx = e % K_DIM;
            Kp[e] = (i < ce)
                ? key_b[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + kx]
                : __float2bfloat16(0.0f);
        }
        {
            const __nv_bfloat16* Usrc0 = U_in + base * CHUNK * V_DIM;
            for (unsigned int e = t; e < CHUNK * V_DIM; e += nthr) Up[e] = Usrc0[e];
        }
        if (t == 0) {
            float dl = gc_in[base * CHUNK + ce - 1];
            decs[0] = expf(dl);
            for (unsigned int i = 0; i < ce; i++) decs[1 + i] = expf(dl - gc_in[base * CHUNK + i]);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v0 + vt] =
                    __float2bfloat16(Sold[kk][vt]);

        // S_new starts at edl*S_old; the i-loop adds duc_i·K_i on top, so duc never
        // needs to outlive one iteration (this is the register saving vs ksplit).
        const float edl = decs[0];
        float Snew[KH][VT];
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Snew[kk][vt] = edl * Sold[kk][vt];

        for (unsigned int i = 0; i < ce; i++) {
            float wsp[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) wsp[vt] = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float w = (float)Wp[i * K_DIM + k0 + kk];   // read ONCE, used VT times
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) wsp[vt] += w * Sold[kk][vt];
            }
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                #pragma unroll
                for (int s = 1; s < SPLIT; s <<= 1)
                    wsp[vt] += __shfl_xor_sync(0xffffffffu, wsp[vt], s);
            const float dc = decs[1 + i];
            float d[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) {
                const float uci = (float)Up[i * V_DIM + v0 + vt] - wsp[vt];
                if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v0 + vt] = __float2bfloat16(uci);
                d[vt] = dc * uci;
            }
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float kv = (float)Kp[i * K_DIM + k0 + kk];  // read ONCE, used VT times
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) Snew[kk][vt] += d[vt] * kv;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = Snew[kk][vt];
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) H[(k0 + kk) * V_DIM + v0 + vt] = Sold[kk][vt];
}

// 512 threads = (V_DIM/VT=64) v-groups x SPLIT(8). KH=16, so Sold+Snew is 2*16*2=64
// state registers — the budget that lets minBlocksPerMultiprocessor=2 hold without
// spilling. Grid is UNCHANGED at [nv, batch]: one CTA per head, so W/K are fetched
// once per head per chunk and L2 sees no duplication.
extern "C" __global__ void __launch_bounds__(512, 1)
gated_delta_rule_chunk_delta_h_vtile(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_vtile_core<4, 1>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
                         num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride,
                         gb_stride, h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// Identical geometry to `..._vtile` above (512 threads, SPLIT=4, grid [nv, batch]) with
// the k-reduction accumulated in Neumaier-compensated form. `..._vtile` is 2.10x on the
// spine but costs -0.60 BFCL on both models, because SPLIT=4 reassociates a sum that
// SPLIT=2 folds in one round; this variant keeps the warp density and pays a handful of
// pure-ALU ops to make the sum order-independent. Same ABI, same smem — the launcher
// picks between them, nothing else changes.
// ── KERNEL 2-VFUSED: chunk_delta_h_vfused = cdh_vtile_core<2,1> ──────────────
// THE SHIPPING SPINE. Same fused single-pass core as `..._vtile` below, but at
// ksplit's own thread geometry: SPLIT=2, 256 threads.
//
// The fusion is where the speed is. Folding the two per-chunk passes into one
// collapses `duc` from a CHUNK-long array to a scalar, which drops smem from
// ksplit's 99,336 B to 49,412 B — and that, not warp count, is what buys 2.01x
// on the spine (measured, gdn_chunk_shapetest, t in {2048,8192,16384}).
//
// `..._vtile` raises this to SPLIT=4 / 512 threads for 2.15x — only 7% more —
// and that 7% costs -0.60 BFCL on both models. Bisected on the ssm-poisoning
// tripwire, which is decisive and takes 4 minutes:
//     ksplit  (unfused, SPLIT=2)  12/12 replays byte-identical
//     vfused  (fused,   SPLIT=2)  12/12   <- this kernel
//     vtile   (fused,   SPLIT=4)   1/12
// So the fusion is innocent and SPLIT=4 is implicated. The mechanism is NOT
// reassociation of the k-sum: compensating the SPLIT-way butterfly fold in
// Neumaier form scored 0/12, i.e. no better, so that hypothesis is refuted and
// the real cause of SPLIT=4's drift is still unknown. Do not promote SPLIT=4
// on the strength of a cosine check — it reads 1.0000 while failing two
// accuracy gates.
extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_vfused(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_vtile_core<2, 1>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
                         num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride,
                         gb_stride, h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// ── KERNEL 2-PIPE: chunk_delta_h_pipe<SPLIT,VT> ──────────────────────────────
// `cdh_vtile_core`'s fused single-pass compute, but with the ASYNC DOUBLE BUFFER
// that core dropped.
//
// The shipping `vfused` spine loads each chunk with plain scalar copies
// (`Wp[e] = Wsrc[e]`) behind a bare `__syncthreads()`. That routes every byte
// global→register→shared and, worse, leaves the load of chunk c+1 strictly after
// the compute of chunk c on a spine that is SERIAL in c. The original
// `chunk_delta_h` already had the fix — `cdh_prefetch`, a 16-byte `cp.async`
// double buffer — and it was simply not carried over when the fused core was
// written. This reuses that same helper rather than growing a second one.
//
// Two buffers of {W,K,U} = 2 * CDH_BUFSZ bf16 = 98,304 B, plus the gc/decay
// tables = 99,336 B, which is exactly the footprint `ksplit` already proved fits
// at 1 CTA/SM on GB10. Occupancy is therefore unchanged; only the overlap is new.
//
// SPLIT is deliberately 2, NOT 4. SPLIT=4 measures faster and fails the
// ssm-state-poisoning tripwire 1/12 (see the vfused note); that verdict is about
// the thread geometry, which this kernel does not change, so it carries over.
//
// This is a STAGING POST, not the destination. The measured reference
// (FlashQLA, 15.0 ms vs our 119.8 ms on the same box) gets its speed from TMA
// (`cp.async.bulk.tensor` + `CUtensorMap`) and an `mbarrier` producer/consumer
// pipeline, neither of which appears anywhere in this tree yet. `cp.async` +
// double buffering is the part reachable without that machinery.
template <int SPLIT, int VT>
__device__ __forceinline__ void cdh_pipe_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;
    const unsigned int v0 = (t / SPLIT) * VT;
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw_pipe[];
    __nv_bfloat16* bufs = (__nv_bfloat16*)smem_raw_pipe;      // 2 x CDH_BUFSZ
    float* gcb  = (float*)(bufs + 2 * CDH_BUFSZ);             // 2 x CHUNK
    float* decb = gcb + 2 * CHUNK;                            // 2 x (CHUNK+1)

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sold[KH][VT];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = H[(k0 + kk) * V_DIM + v0 + vt];

    if (g.nchunks == 0) return;
    // Prologue: chunk 0 into slot 0, then each iteration issues c+1 before
    // consuming c, so the copy engine is busy across the whole compute body.
    cdh_prefetch(bufs, gcb, decb, 0, W_in, U_in, key, gate, gc_in, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 cu_seqlens, cu_chunks, is_varlen);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int p = c & 1u;
        const unsigned int ce = (g.seqlen - c * CHUNK) < CHUNK ? (g.seqlen - c * CHUNK) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) {
            cdh_prefetch(bufs, gcb, decb, p ^ 1u, W_in, U_in, key, gate, gc_in, c + 1, b, vh,
                         seq_len, num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                         cu_seqlens, cu_chunks, is_varlen);
            cp_wait<1>();   // slot p landed; slot p^1 may still be in flight
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        const __nv_bfloat16* Wp = bufs + (unsigned long long)p * CDH_BUFSZ;
        const __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        const __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* decs = decb + p * (CHUNK + 1);

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v0 + vt] =
                    __float2bfloat16(Sold[kk][vt]);

        const float edl = decs[0];
        float Snew[KH][VT];
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Snew[kk][vt] = edl * Sold[kk][vt];

        for (unsigned int i = 0; i < ce; i++) {
            float wsp[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) wsp[vt] = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float w = (float)Wp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) wsp[vt] += w * Sold[kk][vt];
            }
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                #pragma unroll
                for (int sft = 1; sft < SPLIT; sft <<= 1)
                    wsp[vt] += __shfl_xor_sync(0xffffffffu, wsp[vt], sft);
            const float dc = decs[1 + i];
            float d[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) {
                const float uci = (float)Up[i * V_DIM + v0 + vt] - wsp[vt];
                if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v0 + vt] = __float2bfloat16(uci);
                d[vt] = dc * uci;
            }
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float kv = (float)Kp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) Snew[kk][vt] += d[vt] * kv;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = Snew[kk][vt];
        // The next iteration overwrites slot p; nothing may still be reading it.
        __syncthreads();
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) H[(k0 + kk) * V_DIM + v0 + vt] = Sold[kk][vt];
}

// 256 threads, SPLIT=2 — the geometry the ssm-poisoning tripwire cleared.
extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_pipe(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_pipe_core<2, 1>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
                        num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride,
                        gb_stride, h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// ── KERNEL 2-TMA: chunk_delta_h_tma<SPLIT,VT> ────────────────────────────────
// `cdh_pipe_core`'s compute and double buffer, with the staging moved from
// `cp.async` to TMA + `mbarrier`.
//
// This is the first TMA in this tree. The measured reference (FlashQLA, 15.0 ms
// against our 105.7 ms on the same box) gets its speed from TMA plus a fused,
// warp-specialised spine; this kernel takes only the first half, so it isolates
// "TMA vs cp.async" as ONE variable against `..._pipe`. Fusing the three FLA
// kernels is the other half and is not attempted here.
//
// What the copy engine takes over:
//   * address generation and tiling for all three tensors;
//   * the K gather. `cdh_prefetch` walks K row by row (`i < ce` predicated,
//     16 `cp_async16` per row) because K lives at a `qk_stride` pitch with a
//     `kh * k_dim` column offset. That is exactly a strided 2-D tile, so one
//     `tma_load_2d` at (col = kh*k_dim, row = tokoff + cs) replaces the loop;
//   * the ragged tail. A tile overhanging the tensor is ZERO-FILLED in
//     hardware, which is the same value the predicated path wrote by hand.
//
// The barrier counts BYTES: one `expect_tx` per stage covering all three tiles,
// so the consumer waits once rather than per tensor. Stage p is reused every
// second chunk, so its phase parity is `(c >> 1) & 1`.
//
// REQUIRES k_dim == K_DIM and v_dim == V_DIM: the descriptors are encoded from
// the compile-time tile, and a narrower runtime head would silently load the
// wrong columns. The launcher checks this and falls back rather than trusting it.
template <int SPLIT, int VT>
__device__ __forceinline__ void cdh_tma_core(
    float* __restrict__ h_state, const CUtensorMap* w_desc, const CUtensorMap* u_desc,
    const CUtensorMap* k_desc, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int v_dim, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;
    constexpr unsigned int STAGE_BYTES =
        (unsigned int)(CHUNK * K_DIM * 2 + CHUNK * K_DIM * 2 + CHUNK * V_DIM * 2);
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;
    const unsigned int v0 = (t / SPLIT) * VT;
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int nthr = blockDim.x;

    // TMA writes into shared, so the destinations must be 128-byte aligned.
    extern __shared__ __align__(128) char smem_raw_tma[];
    __nv_bfloat16* bufs = (__nv_bfloat16*)smem_raw_tma;   // 2 x {W,K,U}
    float* decb = (float*)(bufs + 2 * CDH_BUFSZ);         // 2 x (CHUNK+1)
    __shared__ __align__(8) uint64_t bar[2];

    if (t == 0) {
        mbar_init(&bar[0], 1);
        mbar_init(&bar[1], 1);
    }
    __syncthreads();

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sold[KH][VT];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = H[(k0 + kk) * V_DIM + v0 + vt];

    if (g.nchunks == 0) return;

    // Issue every tile for chunk `c` into slot `p` behind one transaction count.
    auto issue = [&](unsigned int c, unsigned int p) {
        const unsigned long long base =
            ((unsigned long long)(g.choff + c) * num_v_heads + vh);
        __nv_bfloat16* Wp = bufs + (unsigned long long)p * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        if (t == 0) {
            mbar_expect_tx(&bar[p], STAGE_BYTES);
            tma_fence();
            tma_load_2d(w_desc, &bar[p], Wp, 0, (int)(base * CHUNK));
            tma_load_2d(u_desc, &bar[p], Up, 0, (int)(base * CHUNK));
            tma_load_2d(k_desc, &bar[p], Kp, (int)(kh * K_DIM),
                        (int)(g.tokoff + (unsigned long long)c * CHUNK));
        }
        // The decay table is a scalar-load reduction, not a tile; it stays on
        // the ordinary path, one thread per entry.
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const float dl = gc_in[base * CHUNK + ce - 1];
        if (t == 0) decb[p * (CHUNK + 1)] = expf(dl);
        for (unsigned int i = t; i < ce; i += nthr)
            decb[p * (CHUNK + 1) + 1 + i] = expf(dl - gc_in[base * CHUNK + i]);
    };

    issue(0, 0);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int p = c & 1u;
        const unsigned int ce = (g.seqlen - c * CHUNK) < CHUNK ? (g.seqlen - c * CHUNK) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) issue(c + 1, p ^ 1u);
        mbar_wait(&bar[p], (c >> 1) & 1u);
        __syncthreads();

        const __nv_bfloat16* Wp = bufs + (unsigned long long)p * CDH_BUFSZ;
        const __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        const __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* decs = decb + p * (CHUNK + 1);

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v0 + vt] =
                    __float2bfloat16(Sold[kk][vt]);

        const float edl = decs[0];
        float Snew[KH][VT];
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Snew[kk][vt] = edl * Sold[kk][vt];

        for (unsigned int i = 0; i < ce; i++) {
            float wsp[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) wsp[vt] = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float w = (float)Wp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) wsp[vt] += w * Sold[kk][vt];
            }
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                #pragma unroll
                for (int sft = 1; sft < SPLIT; sft <<= 1)
                    wsp[vt] += __shfl_xor_sync(0xffffffffu, wsp[vt], sft);
            const float dc = decs[1 + i];
            float d[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) {
                const float uci = (float)Up[i * V_DIM + v0 + vt] - wsp[vt];
                if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v0 + vt] = __float2bfloat16(uci);
                d[vt] = dc * uci;
            }
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float kv = (float)Kp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) Snew[kk][vt] += d[vt] * kv;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = Snew[kk][vt];
        __syncthreads();
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) H[(k0 + kk) * V_DIM + v0 + vt] = Sold[kk][vt];
}

extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_tma(
    float* __restrict__ h_state,
    const __grid_constant__ CUtensorMap w_desc,
    const __grid_constant__ CUtensorMap u_desc,
    const __grid_constant__ CUtensorMap k_desc,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int v_dim,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_tma_core<2, 1>(h_state, &w_desc, &u_desc, &k_desc, gc_in, S_out, uc_out, seq_len,
                       num_chunks, num_k_heads, num_v_heads, v_dim, h_state_is_table,
                       cu_seqlens, cu_chunks, is_varlen);
}

// ── KERNEL 2-DVSPLIT: chunk_delta_h_dvsplit<SPLIT,DVB> ───────────────────────
// SCALAR DV-block split. Same DV factorization tc_vblock uses (column block
// [dv0,dv0+DVB) of S_{c+1} depends only on the same column block of S_c — the W·S
// contraction reduces over k, never v), but WITHOUT wmma and WITHOUT the double
// buffer. Motivation is an ncu profile of ksplit at the 35B GDN shape:
//
//   Grid 32 blocks on 48 SMs  →  Waves Per SM 0.67   (a third of the GPU idle)
//   Registers/thread     165  →  Block Limit Registers   = 1
//   Dynamic smem      99.34KB →  Block Limit Shared Mem  = 1
//   Achieved occupancy 16.68% (8 active warps/SM)
//   Memory 62.1% / Compute 26.5%  → neither bandwidth- nor FLOP-bound
//
// ksplit is triply capped: too few CTAs to fill the machine, and 1 CTA/SM from BOTH
// registers and smem so the few it has cannot hide latency. tc_vblock fixed only the
// first (64 CTAs) while keeping 1 CTA/SM, converting 0.67 waves into 1.33 — a second
// nearly-empty wave — which is why it measured 0.89-0.90x (slower), not faster.
//
// This variant attacks all three:
//   * grid.y gains the DV axis  → 64 CTAs (2 per head) instead of 32
//   * SPLIT=4 over DVB=64 v-cols → Sreg[KH=32] instead of [64]: -32 registers
//   * single-buffered smem       → 41,476 B instead of 99,336: ≥2 CTAs/SM
// The double buffer is deliberately dropped: with ≥2 resident CTAs the SM hides load
// latency by switching blocks, which is cheaper than paying 2x smem to prefetch
// within one block. That is the whole bet, and it is why this is worth measuring
// rather than assuming.
//
// Cost accepted: W and K are K-dimensioned, so both DV-block CTAs load them — ~2x the
// W/K global traffic (~5 ms/prefill at this shape) against a kernel that is only 62%
// bandwidth-utilised. Drop-in ABI == ksplit/tc_vblock (same 21 args, same
// S_out/uc_out/h_state layout), so chunk_fwd_o is unchanged.
template <int SPLIT, int DVB>
__device__ __forceinline__ void cdh_dvsplit_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;          // per-thread slice of the state column
    constexpr int NDVB = V_DIM / DVB;          // DV blocks per head
    const unsigned int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    const unsigned int dvb = blockIdx.y % (unsigned int)NDVB;
    const unsigned int b = blockIdx.y / (unsigned int)NDVB;
    const unsigned int dv0 = dvb * (unsigned int)DVB;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;        // 0 .. DVB*SPLIT-1
    const unsigned int vloc = t / SPLIT;       // 0 .. DVB-1
    const unsigned int v = dv0 + vloc;         // absolute v-column
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw_dvs[];
    __nv_bfloat16* Wp = (__nv_bfloat16*)smem_raw_dvs;      // [CHUNK*K_DIM]
    __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;                // [CHUNK*K_DIM]
    __nv_bfloat16* Up = Kp + CHUNK * K_DIM;                // [CHUNK*DVB]  (this DV block only)
    float* decs = (float*)(Up + CHUNK * DVB);              // [CHUNK+1], [0]=exp(gc_last)

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    const __nv_bfloat16* key_b = key + g.tokoff * qk_stride;
    const unsigned int nthr = blockDim.x;

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        __syncthreads();   // previous iteration's readers are done with smem
        const __nv_bfloat16* Wsrc = W_in + base * CHUNK * K_DIM;
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) Wp[e] = Wsrc[e];
        const __nv_bfloat16* Usrc = U_in + base * CHUNK * V_DIM;
        for (unsigned int e = t; e < CHUNK * DVB; e += nthr)
            Up[e] = Usrc[(e / DVB) * V_DIM + dv0 + (e % DVB)];
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) {
            unsigned int i = e / K_DIM, kx = e % K_DIM;
            Kp[e] = (i < ce)
                ? key_b[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + kx]
                : __float2bfloat16(0.0f);
        }
        if (t == 0) {
            float dl = gc_in[base * CHUNK + ce - 1];
            decs[0] = expf(dl);
            for (unsigned int i = 0; i < ce; i++) decs[1 + i] = expf(dl - gc_in[base * CHUNK + i]);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v] = __float2bfloat16(Sreg[kk]);

        const float edl = decs[0];
        float duc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float wsp = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++)
                wsp += (float)Wp[i * K_DIM + k0 + kk] * Sreg[kk];
            #pragma unroll
            for (int s = 1; s < SPLIT; s <<= 1) wsp += __shfl_xor_sync(0xffffffffu, wsp, s);
            float uci = (float)Up[i * DVB + vloc] - wsp;   // wsp == full <W_i, S[:,v]>
            if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v] = __float2bfloat16(uci);
            duc[i] = decs[1 + i] * uci;
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++) {
            float hv = edl * Sreg[kk];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kp[i * K_DIM + k0 + kk];
            Sreg[kk] = hv;
        }
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++) H[(k0 + kk) * V_DIM + v] = Sreg[kk];
}

// 256 threads = DVB(64) v-columns x SPLIT(4) k-slices. minBlocksPerMultiprocessor=2
// is load-bearing, not a hint: it caps the compiler at 128 regs/thread so two CTAs
// are actually resident, which is the entire point of the variant.
extern "C" __global__ void __launch_bounds__(256, 2)
gated_delta_rule_chunk_delta_h_dvsplit(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_dvsplit_core<4, 64>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
                            num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride,
                            gb_stride, h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// ── KERNEL 2-TC-VBLOCK: chunk_delta_h_tc_vblock ──────────────────────────────
// wmma Phase-A (W·S on tensor cores via mma_gram) + DV-block split (DV 128→2×64)
// + double-buffered cp.async. Grafts chunk_delta_h_tc's Phase A (line ~479) but the
// DV split halves every DV-dimensioned smem buffer, so the wmma Sᵀ+ws buffers AND a
// double buffer fit under 99KB (81KB used) — the thing the shelved TC (96KB, single-
// buffered) couldn't. State stays f32 in registers (precision-critical). Grid adds a
// dv-block axis folded into blockIdx.y: more CTAs on the same 48 SMs. Drop-in ABI ==
// ksplit (SPEC C): same 21 args, same S_out/uc_out/h_state layout for chunk_fwd_o.
//
// DV-block factorization (the #1 correctness fact): column block [dv_off,dv_off+64)
// of S_{c+1} depends ONLY on the same column block of S_c. The W·S contraction
// reduces over k (never v); duc_i[v]=exp(gc_last-gc_i)·uc_i[v] is per-column; edl·S_c
// is element-wise in v. DV is purely an OUTPUT/state axis, never a contraction axis →
// two DV-block CTAs carry disjoint, non-interacting column-slices (no reduction, no
// cross-block comms). Same split croll83 prepare_h uses; same reason ksplit_vblock is
// bit-parity. (This split is on DV/output; ksplit's is on K/contraction + a shfl.)
//
// Thread layout (256 threads = 8 warps): a DV_BLK(64)-wide v-block × KSPLIT(4)-way
// K-split of the state column IN REGISTERS — like ksplit SPLIT=4 but WITHOUT the
// __shfl_xor butterfly (wmma performs the full-K W·S sum in Phase A). The other 4
// warps idle ONLY during the Phase-A mma_gram call (which tiles M by warp*16 and
// would overrun CHUNK=64 past 4 warps); they participate in all loads + Phases B.
//
// smem (DV_BLK=64): St[64*128]bf16(16K) + ws[64*64]f32(16K) + buf[2][64*128+64*64]
// bf16(48K) + gcb[2][64]f32 + decb[2][65]f32 = 82952 B (~81KB), double-buffered AND
// wmma. Kb aliases St (Phase A consumes St→ws; St then dead, reused to stage K for
// Phase B), recovering the 16KB that lets the double buffer coexist with wmma.
// NOTE: launcher MUST opt in via CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=82952.
#define DV_BLK 64
#define NUM_DV_BLK (V_DIM / DV_BLK)                 // 2
// double-buffer slot: {W[CHUNK*K_DIM], Uslice[CHUNK*DV_BLK]} bf16
#define TCVB_SLOT (CHUNK * K_DIM + CHUNK * DV_BLK)  // 8192 + 4096 = 12288 bf16

extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_tc_vblock(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out,
    __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    const unsigned int dv_blk = blockIdx.y / batch_size;   // 0..NUM_DV_BLK-1
    const unsigned int b      = blockIdx.y % batch_size;
    const unsigned int dv_off = dv_blk * DV_BLK;           // 0 or 64
    GDN_GEOM(g);
    key += g.tokoff * qk_stride;   // per-stream token offset (b*seq_len / cu_seqlens) —
                                   // the raw key tensor is token-major; W/U/gc fold b via
                                   // choff but key did not → batch>1 read batch-0's tokens.

    const unsigned int tid = threadIdx.x;                  // 0..255  (8 warps)
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // ── smem (82952 B, double-buffered) ──
    extern __shared__ char smem_raw[];
    __nv_bfloat16* St  = (__nv_bfloat16*)smem_raw;             // [DV_BLK*K_DIM] Sᵀ snapshot   16KB
    float*         ws  = (float*)(St + DV_BLK * K_DIM);        // [CHUNK*DV_BLK] Phase-A out   16KB
    __nv_bfloat16* buf = (__nv_bfloat16*)(ws + CHUNK * DV_BLK);// buf[2][TCVB_SLOT]            48KB
    float*         gcb = (float*)(buf + 2 * TCVB_SLOT);        // gcb[2][CHUNK]
    float*         decb = gcb + 2 * CHUNK;                     // decb[2][CHUNK+1] [0]=exp(gc_last)
    __nv_bfloat16* Kb  = St;                                   // Phase B reuses St for K (alias)

    // ── resident f32 state slice: this CTA owns v-columns [dv_off, dv_off+DV_BLK).
    //    256 threads → DV_BLK(64) v-columns × KSPLIT(4) k-strips of KH=32. Thread
    //    holds Sreg[KH] for one v-column and one k-strip (4-way K-split in registers,
    //    like ksplit SPLIT=4 but WITHOUT shfl — wmma does the W·S sum).
    constexpr int KSPLIT = 256 / DV_BLK;                      // 4
    constexpr int KH     = K_DIM / KSPLIT;                    // 32
    const unsigned int vloc = tid / KSPLIT;                   // 0..DV_BLK-1  local v
    const unsigned int ksub = tid % KSPLIT;                   // 0..3
    const unsigned int k0   = ksub * KH;
    const unsigned int v    = dv_off + vloc;                  // absolute v-column

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    // ── inline DV-block-aware double-buffer prefetch (W full-K, U dv-sliced, gc/dec) ──
    // (cdh_prefetch stages full-V U; here U is sliced to [dv_off,dv_off+DV_BLK) so the
    //  U buffer is DV_BLK-wide — hence an inline prefetch instead of the shared helper.)
    auto prefetch = [&](unsigned int p, unsigned int c) {
        __nv_bfloat16* Wp = buf + (unsigned long long)p * TCVB_SLOT;
        __nv_bfloat16* Up = Wp + CHUNK * K_DIM;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
        // W [CHUNK][K_DIM] full (DV-independent): cp.async 16B = 8 bf16.
        for (unsigned int idx = tid * 8; idx < CHUNK * K_DIM; idx += 256 * 8) {
            unsigned int i = idx / K_DIM, k = idx % K_DIM;
            if (i < ce) cp_async16(&Wp[i * K_DIM + k], &W_in[base * CHUNK * K_DIM + i * k_dim + k]);
        }
        // U [CHUNK][DV_BLK] dv-sliced.
        for (unsigned int idx = tid * 8; idx < CHUNK * DV_BLK; idx += 256 * 8) {
            unsigned int i = idx / DV_BLK, vv = idx % DV_BLK;
            if (i < ce) cp_async16(&Up[i * DV_BLK + vv],
                                   &U_in[base * CHUNK * V_DIM + i * v_dim + (dv_off + vv)]);
        }
        cp_commit();
        if (tid == 0) {            // gc / decay computed scalar (cheap, no cp.async).
            float* gcp = gcb + p * CHUNK; float* dp = decb + p * (CHUNK + 1);
            float gl = gc_in[base * CHUNK + (ce - 1)];
            dp[0] = expf(gl);
            for (unsigned int i = 0; i < ce; i++) {
                float gi = gc_in[base * CHUNK + i];
                gcp[i] = gi; dp[1 + i] = expf(gl - gi);
            }
        }
    };

    prefetch(0, 0);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs  = c * CHUNK;
        const unsigned int ce  = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) { prefetch((c + 1) & 1u, c + 1); cp_wait<1>(); }
        else                   { cp_wait<0>(); }
        __syncthreads();

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * TCVB_SLOT;
        __nv_bfloat16* Up = Wp + CHUNK * K_DIM;
        const float*   dec = decb + cur * (CHUNK + 1);
        const float    edl = dec[0];

        // (1) entry state bf16(S_c) → S_out  [(choff+c)*nv+vh][K_DIM][V_DIM].
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v] = __float2bfloat16(Sreg[kk]);

        // (2) stage Sᵀ snapshot for this dv-block: St[vloc][k] = bf16(S[k][v]). The 4
        //     ksub lanes of a v-column together fill its full 128-wide K row.
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            St[vloc * K_DIM + (k0 + kk)] = __float2bfloat16(Sreg[kk]);
        __syncthreads();

        // (3) PHASE A (TENSOR CORE): ws[i][vloc] = Σ_k W[i][k]·Sᵀ[vloc][k] = <W_i,S[:,v]>.
        //     mma_gram M=CHUNK(64) over 4 warps, K=K_DIM(128), N=DV_BLK(64)=NTC*8 → NTC=8,
        //     NSTRIDE=DV_BLK. Guard to first 4 warps: mma_gram tiles M by warp*16 and
        //     warps 4–7 would address M-rows 64–127 (past CHUNK). Each warp writes disjoint
        //     ws rows (no internal __syncthreads) → the warp guard is safe.
        if ((tid >> 5) < 4)
            mma_gram<DV_BLK / 8, DV_BLK, false>(Wp, St, ws);   // NTC = 64/8 = 8
        __syncthreads();

        // (4) uc = U - ws ; duc = exp(gc_last-gc_i)·uc   (scalar; ksub==0 writes uc_out).
        float duc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float uci = (float)Up[i * DV_BLK + vloc] - ws[i * DV_BLK + vloc];
            if (ksub == 0)
                uc_out[base * CHUNK * V_DIM + i * v_dim + v] = __float2bfloat16(uci);
            duc[i] = dec[1 + i] * uci;
        }
        __syncthreads();   // before St region reused for K

        // (5) load K into the freed St region (Kb=St), full K_DIM (DV-independent).
        for (unsigned int idx = tid * 8; idx < CHUNK * K_DIM; idx += 256 * 8) {
            unsigned int i = idx / K_DIM, k = idx % K_DIM;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                Kb[i * K_DIM + k + j] = (i < ce)
                    ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + (k + j)]
                    : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // (6) PHASE B (SCALAR, f32 register-S): S_{c+1}[k][v]=edl·S+Σ_i duc_i·K_i[k].
        #pragma unroll
        for (int kk = 0; kk < KH; kk++) {
            float hv = edl * Sreg[kk];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kb[i * K_DIM + (k0 + kk)];
            Sreg[kk] = hv;
        }
        __syncthreads();   // before St/ws/buf reused next chunk
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++) H[(k0 + kk) * V_DIM + v] = Sreg[kk];
}

// ── KERNEL 2-KSPLIT-VBLOCK: chunk_delta_h_ksplit_vblock<SPLIT,VTILES> ─────
// V-block (DV) split of the ksplit spine for the BATCH=1 single-stream wall:
// ksplit fills only nv·batch = 32 CTAs at batch=1 (< GB10's ~48 SMs), so the
// SMs are starved and more warps/CTA (SPLIT) no longer help (8 warps already
// saturate latency hiding — see line 627). The only math-preserving way to add
// CTAs at batch=1 is to PARTITION the v-columns across CTAs: blockIdx.y = vtile
// owns v in [vtile·VW, (vtile+1)·VW), VW = V_DIM/VTILES. That lifts CTAs to
// nv·VTILES·batch (128 at VTILES=4, batch=1). Each v-column's state evolves
// INDEPENDENTLY (the W·S contraction reduces over k, never over v; S_{c+1}[k,v]
// = edl·S_c[k,v] + Σ duc_i·k_i[k] has no cross-v term), so the per-CTA work
// shrinks by VTILES while the output is byte-identical to ksplit — bit-parity.
// COST/RISK: each vtile CTA still cdh_prefetches the FULL W/U/K chunk (W/K are
// needed fully; only 1/VTILES of U is used) — VTILES× the chunk DRAM reads, but
// the VTILES CTAs of a head are co-resident so L2 serves the re-reads (W+U+K ≈
// 48KB/chunk). The bet: L2 reuse + the shrunk per-CTA v-loop beat the idle-SM
// loss. The OLD naive V-tiling regressed (line 298) by re-running the serial
// loop redundantly AND dropping the double-buffer; this keeps both. The microtest
// (gdn_cdh_vblock_microtest) is the gate — bit-parity vs ksplit + per-(t,batch)
// ms/iter A/B.
//
// VERDICT (2026-06-25, gdn_cdh_vblock_microtest on GB10, bit-parity 18/18 PASS):
// the clean double-buffered V-split STILL REGRESSES — 0.71x/0.65x/0.34x at
// batch=1 for VTILES=2/4/8, slower at batch=2/4 too, monotonically worse with
// more tiles. So the prior comment (line 298) stands with hard numbers: adding
// CTAs does NOT help here. chunk_delta_h is NOT occupancy-starved at batch=1 —
// it is bound by the SERIAL CHUNK DEPENDENCY (S_{c+1} needs S_c) + per-CTA
// latency hiding (already saturated at SPLIT=2 → 8 warps). More CTAs only add
// redundant W/U/K re-reads and SHRINK warps/CTA, hurting latency hiding. NOT
// wired into any dispatch — kept for the record + the microtest A/B. The real
// single-stream GDN lever is the serial recurrence itself (parallel/associative
// chunk scan or the wmma rewrite), not occupancy.
template <int SPLIT, int VTILES>
__device__ __forceinline__ void cdh_ksplit_vblock_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;            // per-thread slice of the state column
    constexpr int VW = V_DIM / VTILES;           // v-columns this CTA owns
    const unsigned int vh = blockIdx.x;
    const unsigned int vtile = blockIdx.y;       // 0..VTILES-1  (NEW axis)
    const unsigned int b = blockIdx.z;           // batch (was blockIdx.y in ksplit)
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;          // 0..VW·SPLIT-1
    const unsigned int v = vtile * VW + (t / SPLIT);   // absolute v-column
    const unsigned int sub = t % SPLIT;          // which k-slice
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // Distinct name from the sibling `extern __shared__ char smem_raw[]` in this
    // same kernel (~L719): two decls of one dynamic-smem symbol in a function trip
    // nvcc #1556-D. All `extern __shared__` views alias the same base — byte-identical.
    extern __shared__ char smem_raw_dhc[];
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw_dhc;          // buf[2][CDH_BUFSZ]
    float* gcb = (float*)(buf + 2 * CDH_BUFSZ);             // gcb[2][CHUNK]
    float* decb = gcb + 2 * CHUNK;                          // decb[2][CHUNK+1]

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    cdh_prefetch(buf, gcb, decb, 0, W_in, U_in, key, gate, gc_in, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 cu_seqlens, cu_chunks, is_varlen);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) {
            cdh_prefetch(buf, gcb, decb, (c + 1) & 1u, W_in, U_in, key, gate, gc_in, c + 1, b, vh, seq_len,
                         num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                         cu_seqlens, cu_chunks, is_varlen);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* dec = decb + cur * (CHUNK + 1);

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v] = __float2bfloat16(Sreg[kk]);

        const float edl = dec[0];
        float duc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float wsp = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++)
                wsp += (float)Wp[i * K_DIM + k0 + kk] * Sreg[kk];
            #pragma unroll
            for (int s = 1; s < SPLIT; s <<= 1) wsp += __shfl_xor_sync(0xffffffffu, wsp, s);
            float uci = (float)Up[i * V_DIM + v] - wsp;
            if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v] = __float2bfloat16(uci);
            duc[i] = dec[1 + i] * uci;
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

// Block = (V_DIM/VTILES)·SPLIT threads: VTILES=2→128, 4→64, 8→32 (all warp-mult).
// Grid: [num_v_heads, VTILES, batch]. Same smem as ksplit (full double-buffer).
extern "C" __global__ void __launch_bounds__(128, 2)
gated_delta_rule_chunk_delta_h_ksplit_vblock2(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_ksplit_vblock_core<2, 2>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
        num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, gb_stride,
        h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}
extern "C" __global__ void __launch_bounds__(64, 4)
gated_delta_rule_chunk_delta_h_ksplit_vblock4(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_ksplit_vblock_core<2, 4>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
        num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, gb_stride,
        h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}
extern "C" __global__ void __launch_bounds__(32, 8)
gated_delta_rule_chunk_delta_h_ksplit_vblock8(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_ksplit_vblock_core<2, 8>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
        num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, gb_stride,
        h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// ── KERNEL 3: chunk_fwd_o ────────────────────────────────────────────────
// The PARALLEL output pass. Grid: (NT, num_v_heads, batch). One CTA per (chunk,head).
// O_i = (exp(gc_i)·<S_c[:,v],q_i> + Σ_{l<=i} exp(gc_i-gc_l)·<k_l,q_i>·uc_l[v])·rsqrt(d).
// BOTH inner products are tensor-core Gram matmuls (full occupancy → compute bound):
//   kq[i][l] = <q_i,k_l>          (mma_gram, decay folded in)
//   o1[i][v] = <q_i, S_c[:,v]>    (mma_gram with S_c read TRANSPOSED → [V][K])
// S_c read bf16 + o1 bf16 (TERMINAL output → no compounding, like wy4's bf16
// output rounding → precision-safe). o1 reuses the freed sk region. Layout matches wy4.
// smem: sq(16K)+sk/o1(16K)+kq(16K f32)+ucb(16K)+Sbᵀ(32K bf16)+gc+egc.
// 512 threads: mma_gram is fenced to the first 4 warps (it hardwires M=64), while
// the ~36K element staging/compute loops below use all 16 — an nsys steady-state
// profile put this kernel at 9.8% of GPU time, the largest single GDN kernel once
// the vtile spine landed, and it stages 96.5 KB of smem through what used to be
// only 4 warps. Same grid, same traffic: only the warp count changes.
extern "C" __global__ void __launch_bounds__(512, 1)
gated_delta_rule_chunk_fwd_o(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    const __nv_bfloat16* __restrict__ S_in, // [(b*NT+c)*nv+vh][K][V] entry states (from #2)
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
    unsigned int gb_stride,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDN_GEOM(g);
    if (c >= g.nchunks) return;            // per-stream chunk bound
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    const unsigned long long out_base = (g.tokoff * num_v_heads + vh) * v_dim;
    // Per-stream input offset (cu_seqlens[b] in varlen, else b*seq_len; b=0 → +0).
    query += g.tokoff * qk_stride;
    key   += g.tokoff * qk_stride;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sq = (__nv_bfloat16*)smem_raw;          // [CHUNK*K_DIM]
    __nv_bfloat16* sk = sq + CHUNK * K_DIM;                // [CHUNK*K_DIM]
    float* kq = (float*)(sk + CHUNK * K_DIM);              // [CHUNK*CHUNK]
    __nv_bfloat16* ucb = (__nv_bfloat16*)(kq + CHUNK * CHUNK); // [CHUNK*V_DIM]
    __nv_bfloat16* Sb = ucb + CHUNK * V_DIM;               // [K_DIM*V_DIM] bf16 (S_c)
    float* gc = (float*)(Sb + K_DIM * V_DIM);              // [CHUNK]
    float* egc = gc + CHUNK;                               // [CHUNK] exp(gc)

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += blockDim.x) {
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
    for (unsigned int idx = tid; idx < CHUNK * v_dim; idx += blockDim.x) {
        unsigned int i = idx / v_dim, v = idx % v_dim;
        ucb[i * V_DIM + v] = (i < ce) ? uc_in[base * CHUNK * V_DIM + i * v_dim + v] : __float2bfloat16(0.0f);
    }
    // S_c read TRANSPOSED → Sbᵀ[v][k] = S_c[k][v], so mma_gram(q, Sbᵀ) = <q_i,S_c[:,v]>.
    for (unsigned int idx = tid; idx < K_DIM * V_DIM; idx += blockDim.x) {
        unsigned int v = idx / K_DIM, k = idx % K_DIM;
        Sb[idx] = S_in[base * K_DIM * V_DIM + k * V_DIM + v];
    }
    for (unsigned int i = tid; i < ce; i += blockDim.x) {
        float g = gc_in[base * CHUNK + i];
        gc[i] = g;
        egc[i] = expf(g);
    }
    __syncthreads();

    if (tid < 128) mma_gram<8, CHUNK, false>(sq, sk, kq);   // kq[i][l] = <q_i, k_l>
    __syncthreads();

    // Fold the intra-chunk decay into the Gram ONCE: kq[i][l] ← exp(gc_i-gc_l)·<q_i,k_l>
    // (was: expf(gc_i-gc_l) recomputed per v-column = v_dim× redundant transcendentals).
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += blockDim.x) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l <= i) kq[p] = expf(gc[i] - gc[l]) * kq[p];
    }
    __syncthreads();   // sk is free past mma1 → reuse its region for the o1 = q·Sᵀ result

    // o1[i][v] = <q_i, S_c[:,v]>  on tensor cores (bf16 out → terminal, precision-safe).
    __nv_bfloat16* o1 = sk;                   // [CHUNK*V_DIM] bf16, reuses sk's 16KB
    if (tid < 128) mma_gram<16, V_DIM, true>(sq, Sb, o1);
    __syncthreads();

    if (tid < v_dim) {
        for (unsigned int i = 0; i < ce; i++) {
            float t1 = egc[i] * (float)o1[i * V_DIM + tid];
            float t2 = 0.0f;
            for (unsigned int l = 0; l <= i; l++)
                t2 += kq[i * CHUNK + l] * (float)ucb[l * V_DIM + tid];   // pure MAC inner loop
            output[out_base + (unsigned long long)(cs + i) * num_v_heads * v_dim + tid] =
                __float2bfloat16((t1 + t2) * inv_sqrt_d);
        }
    }
}
