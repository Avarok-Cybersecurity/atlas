// SPDX-License-Identifier: AGPL-3.0-only

// Per-Q-head paged Flash Attention compute.
//
// Grid: (num_q_heads, ceil(q_len/BR), 1) — one block per Q head.
// Same pipeline as contiguous kernel (V async, K double-buffered),
// just with paged K/V loads via LOAD_KV_TILE macro.
//
// Fixes 3 bottlenecks from old GQA-grouped kernel:
//   1. GQA serialization: was ~14 syncs/tile → now 3 syncs/tile
//   2. V blocking: V now loads async, overlaps with QK^T (BF16 cp.async)
//   3. Q reload: Q loaded once at start, not GQA_RATIO× per tile
//
// Expects the including file to define:
//   LOAD_KV_TILE(cache, block_table, smem, kv_start, kv_len, kv_head, tid, stride)
//   KERNEL_NAME, K_CACHE_TYPE, V_CACHE_TYPE, KERNEL_EXTRA_PARAMS, KERNEL_PREAMBLE

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Async global→shared 16-byte copy helpers (cp.async on NVIDIA + SCALE).
// Portable counterparts are defined in the strix-hip copy of this header,
// where they degrade to synchronous uint4 copies (AMD has no cp.async).
__device__ __forceinline__ void atlas_cp16(void* smem_dst, const void* gmem_src) {
    unsigned _s = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(_s), "l"(gmem_src));
}
__device__ __forceinline__ void atlas_cp16_pred(void* smem_dst, const void* gmem_src, bool pred) {
    unsigned _s = __cvta_generic_to_shared(smem_dst);
    unsigned _b = pred ? 16u : 0u;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(_s), "l"(gmem_src), "r"(_b));
}
__device__ __forceinline__ void atlas_cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void atlas_cp_wait()   { asm volatile("cp.async.wait_group 0;"); }

// Phase 2c precision upgrade (2026-05-24): P*V MMA now uses FP16 inputs
// instead of BF16. FP16 has 10-bit mantissa vs BF16's 7-bit → 8× finer
// precision on softmax probabilities, which is the largest remaining
// source of attention output drift vs the PyTorch reference (which keeps
// P at full FP32 internally on CPU). Q*K MMA stays BF16 because Q and
// K are already BF16 from the cache; converting them to FP16 wouldn't
// add information.
//
// `smem_V` stays BF16 so the LOAD_KV_TILE macros (BF16 cp.async, FP8
// dequant, NVFP4 dequant) don't need rewriting. V is converted to FP16
// per-MMA in registers via this helper. ~10% prefill+decode slowdown
// from the extra conversions, but eliminates the BF16-P precision loss
// that was driving FP8-induced token-margin flips (mid-word `</think>`,
// `parameter>\n` and `.method().method()` chain attractors).
//
// SSOT: one helper used at every P*V MMA call site in this header.
__device__ __forceinline__ unsigned int bf16x2_to_f16x2_bits(
    __nv_bfloat16 lo, __nv_bfloat16 hi
) {
    __half2 h2 = __floats2half2_rn(__bfloat162float(lo), __bfloat162float(hi));
    return *reinterpret_cast<const unsigned int*>(&h2);
}

#ifdef ATLAS_ATTN_FP8_SMEM
#include <cuda_fp8.h>
// FP8-smem occupancy variant — NVIDIA GB10 (sm_121), FP8-KV cache, HDIM=256.
// K and V live in shared memory as raw E4M3 bytes (1 B) instead of dequantized
// BF16 (2 B), halving smem_K + smem_V (~25 KB at BR=32 -> ~45 KB/CTA total) so
// 2 CTAs/SM fit and the QK/PV MMA latency is hidden. The bytes are dequantized
// to BF16/FP16 in-register right before each MMA via the same `fp8_to_bf16`
// (per-tensor k_scale/v_scale) the load-time path used — so the MMA operands,
// and therefore the kernel output, are bit-identical to the BF16-smem kernel.
// Only the smem *storage* changes (deferred dequant). `fp8_to_bf16` is defined
// by the including FP8 wrapper (inferspark_prefill_paged_fp8*.cu).
__device__ __forceinline__ unsigned int fp8x2_to_bf16x2_bits(
    __nv_fp8_storage_t lo, __nv_fp8_storage_t hi, float scale
) {
    unsigned short l = __bfloat16_as_ushort(fp8_to_bf16(lo, scale));
    unsigned short h = __bfloat16_as_ushort(fp8_to_bf16(hi, scale));
    return ((unsigned int)h << 16) | (unsigned int)l;
}
__device__ __forceinline__ unsigned int fp8x2_to_f16x2_bits(
    __nv_fp8_storage_t lo, __nv_fp8_storage_t hi, float scale
) {
    return bf16x2_to_f16x2_bits(fp8_to_bf16(lo, scale), fp8_to_bf16(hi, scale));
}
#endif

// Softmax exponential. Phase 2b precision fix (2026-05-24): the prior
// degree-3 Taylor polynomial was advertised as "max err ~1e-4" but
// numerical verification (against torch.exp on x in [-20, 0]) showed
// **max relative error 5.1e-3 (~0.5%)** — concentrated at tf near 1.0.
// Across 18920-token attention rows and 10 full-attention layers, this
// compounds to ~5% cosine drift vs PyTorch reference softmax. Linear-
// attention layers (GDN) don't use softmax and were unaffected,
// matching the per-layer drift pattern.
//
// Default path: `__expf` — CUDA SFU exp, ~2 ULP accuracy, ~10 cycles.
// Opt-in fast path: `ATLAS_FAST_SOFTMAX_EXP` — the original FA4-style
// polynomial. Use only when the ~0.5% softmax-row drift is acceptable.
__device__ __forceinline__ float sw_exp(float x) {
#ifdef ATLAS_FAST_SOFTMAX_EXP
    // FA4-style: degree-3 polynomial for 2^tf, max err ~0.5% at tf~1.
    float t = x * 1.4426950408889634f;
    float ti = floorf(t);
    float tf = t - ti;
    float p = 1.0f + tf * (0.6931471805599453f +
              tf * (0.2402265069591007f +
              tf * 0.05550410866482158f));
    return ldexpf(p, (int)ti);
#else
    // SSOT for prefill-attention softmax exp. Matches PyTorch reference.
    return __expf(x);
#endif
}

#define BR 32
// B5 (chunked-prefill audit fix 2026-05-27): tried BC=64 → PTX JIT
// failure in `inferspark_prefill_paged_batched` (smem/register budget
// blown under PREFILL_BATCHED). Tried BC=48 → compiles but
// non-power-of-2 doesn't align cleanly to m16n8k16 MMA fragments.
// Reverted to BC=32 baseline; B5 deferred to a smem redesign pass.
#define BC 32
#ifndef HDIM
#define HDIM 256
#endif
// PAD_KV may be overridden by a kernel before including this header (e.g. the
// FP8-smem cp.async path needs a 16-aligned row stride for 16-byte cp.async /
// uint4 stores; FP8 elements are 1 byte so PAD_KV=8 → 264 = only 8-aligned,
// whereas PAD_KV=16 → 272 = 16-aligned). Default 8 keeps the BF16 row stride
// (256+8)*2 = 528 bytes 16-aligned, as before.
#ifndef PAD_KV
#define PAD_KV 8
#endif
#define HDIM_PAD (HDIM + PAD_KV)
#define PAD_P 8

// ATLAS_ATTN_LDMATRIX: use ldmatrix.x4 for the A-operand (Q for QK^T, P for PV)
// smem loads instead of 4 scalar unsigned-int loads. One PTX instruction
// replaces 4 manual loads, shortening the load→MMA dependency chain this
// latency-bound prefill kernel is gated on. PROVEN on GB10/SM121 (ldmatrix_probe.cu
// cosine 1.0; commit 7dbdfe41 "bit-identical, +2%"). Default ON — opt OUT with
// -DATLAS_DISABLE_ATTN_LDMATRIX (the prior default-off) if a regression appears.
#ifndef ATLAS_DISABLE_ATTN_LDMATRIX
#define ATLAS_ATTN_LDMATRIX
#endif
#define N_TILES_PER_WARP ((HDIM / 8) / 2)
#define TILE_CHUNKS (BR * (HDIM / 8))

// SCALE/gfx1151: RDNA3.5 has a hard 64 KB/workgroup LDS cap. The
// double-buffered smem_K[2] (33,792 B at HDIM=256) pushes this kernel's
// __shared__ to 70,400 B > 65,536. Single-buffer smem_K under SCALE
// (70,400 -> 53,504 B, fits with margin). Correct-by-construction: the
// existing __syncthreads() before the K prefetch and after the K-wait
// already bracket the QK^T read and the prefetch write of smem_K, and the
// PV stage never reads smem_K — so collapsing to one buffer is race-free
// (it only reduces load/compute overlap). NVIDIA #else keeps the original
// double buffer verbatim (byte-identical codegen, zero regression).
#if defined(__SCALE__)
#define ATLAS_KBUFN 1
#define ATLAS_KB(x) 0u
#else
#define ATLAS_KBUFN 2
#define ATLAS_KB(x) (x)
#endif

extern "C" __global__ void KERNEL_NAME(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
#ifdef PREFILL_BATCHED
    // Q12 Phase 3: batched paged prefill.
    // - block_table_ptrs[b] is the per-stream paged-KV block table.
    // - Q and O are stacked: [batch, q_len, num_q_heads, head_dim] flattened
    //   contiguously. Each stream's Q/O lands at `b * q_len * q_seq_stride`.
    // - All other parameters are SHARED across streams (same q_len, kv_len,
    //   q_offset etc.). The scheduler enforces same-chunk-len batching.
    // - Grid extended to (num_q_heads, q_chunks, batch_size); blockIdx.z = b.
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
#else
    const int* __restrict__ block_table,
#endif
    const unsigned int q_len,
    unsigned int kv_len,
    unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,  // 0 = full attn; >0 = mask K positions where (Q - K) >= window
    const unsigned int causal_mask_enabled  // 1 = causal (default); 0 = bidirectional (DFlash γ-block)
    KERNEL_EXTRA_PARAMS
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
#ifdef PREFILL_BATCHED
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const int* const __restrict__ block_table = block_table_ptrs[b];
#endif
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
#ifdef PREFILL_BATCHED
    // Per-batch Q/O offsets — stacked [batch, q_len, num_q_heads, head_dim].
    const unsigned long long q_batch_off = (unsigned long long)b * q_len * q_seq_stride;
#endif

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
#ifdef ATLAS_ATTN_FP8_SMEM
    // FP8-smem variant: K/V kept as raw E4M3 bytes, dequantized in-register
    // before each MMA (see fp8x2_to_*_bits). Halves smem_K + smem_V.
    __shared__ __nv_fp8_storage_t smem_K[ATLAS_KBUFN][BC][HDIM_PAD];  // double-buffered
    __shared__ __nv_fp8_storage_t smem_V[BC][HDIM_PAD];
#else
    __shared__ __nv_bfloat16 smem_K[ATLAS_KBUFN][BC][HDIM_PAD];  // double-buffered (single under SCALE)
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
#endif
    // Phase 2c: smem_P FP16 (10-bit mantissa) vs BF16 (7-bit).
    // Read back as 2x packed FP16 per .b32 register for the .f16.f16 MMA.
    // Bisect: `ATLAS_DISABLE_FP16_PV` reverts the Phase 2c FP16 P×V path
    // to the pre-Phase-2b BF16 P×V (smem_P=BF16, store via
    // __float2bfloat16_rn, .bf16.bf16 MMA, direct V read).
#ifdef ATLAS_DISABLE_FP16_PV
    __shared__ __nv_bfloat16 smem_P[BR][BC + PAD_P];
#else
    __shared__ __half smem_P[BR][BC + PAD_P];
#endif
    __shared__ float smem_ml[BR][2];

    KERNEL_PREAMBLE

    // q_rope_pos: absolute position used to rotate the query block. Indirect
    // (DFlash) declares it in KERNEL_PREAMBLE from a device u32 (= true decode
    // position, decoupled from cache-slot base). All other variants: equals
    // q_offset (correct for causal attention where RoPE pos == cache base).
#ifndef Q_ROPE_POS_OVERRIDE
    unsigned int q_rope_pos = q_offset;
#endif

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP;
    const unsigned int p_smem_stride = BC + PAD_P;

    // Single-head accumulators (no GQA array — 4× fewer registers)
    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }

    // === Merged Q + K[0] load (single commit group) ===
    // Q via cp.async, K[0] via LOAD_KV_TILE (cp.async for BF16, sync for FP8/NVFP4).
    // For FP8/NVFP4, Q async copies overlap with K synchronous dequant work.
    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += blockDim.x) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            if (q_start + row < q_len) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                atlas_cp16(&smem_Q[row][col], gm);
            } else { *((uint4*)&smem_Q[row][col]) = make_uint4(0,0,0,0); }
        }
        if (num_kv_blocks > 0) {
            LOAD_KV_TILE(K_cache, block_table, smem_K[0], 0, kv_len, kv_head, tid, blockDim.x);
        }
        atlas_cp_commit();
        atlas_cp_wait();
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // === Start V load (overlaps with QK^T for BF16 cp.async) ===
        LOAD_KV_TILE(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, blockDim.x);
        atlas_cp_commit();

        // === QK^T (warps 0-1, register-based) ===
        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < 4; i++) { acc_s[i][0]=0; acc_s[i][1]=0; acc_s[i][2]=0; acc_s[i][3]=0; }

            const unsigned short* sQ = (const unsigned short*)smem_Q;
#ifdef ATLAS_ATTN_FP8_SMEM
            const __nv_fp8_storage_t* sK = (const __nv_fp8_storage_t*)smem_K[ATLAS_KB(buf)];
#else
            const unsigned short* sK = (const unsigned short*)smem_K[ATLAS_KB(buf)];
#endif

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM/16); ks++) {
                unsigned int kb = ks*16;
                unsigned int a0,a1,a2,a3;
#ifdef ATLAS_ATTN_LDMATRIX
                // SM121 ldmatrix.x4 NON-trans for the Q A-fragment (v47-proven on
                // GB10): one instr replaces 4 manual smem loads, shortening the
                // load->MMA dependency chain this latency-bound kernel is gated on.
                // sQ-relative addressing so the same code serves BR32/BR64.
                { unsigned int qb=__cvta_generic_to_shared(&sQ[(qk_warp_m+(lane_id&15))*HDIM_PAD+(lane_id>>4)*8+kb]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(qb)); }
#else
                unsigned int ar0=qk_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=kb+tid_in_group*2, ac1=ac0+8;
                a0=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac0];
                a1=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac0];
                a2=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac1];
                a3=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac1];
#endif

                #pragma unroll
                for (int nt=0; nt<4; nt++) {
                    unsigned int nc=nt*8+group_id, k0=kb+tid_in_group*2, k1=k0+8;
#ifdef ATLAS_ATTN_FP8_SMEM
                    unsigned int b0=fp8x2_to_bf16x2_bits(sK[nc*HDIM_PAD+k0],sK[nc*HDIM_PAD+k0+1],k_scale);
                    unsigned int b1=fp8x2_to_bf16x2_bits(sK[nc*HDIM_PAD+k1],sK[nc*HDIM_PAD+k1+1],k_scale);
#else
                    unsigned int b0=((unsigned int)sK[nc*HDIM_PAD+k0+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k0];
                    unsigned int b1=((unsigned int)sK[nc*HDIM_PAD+k1+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k1];
#endif
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_s[nt][0]),"=f"(acc_s[nt][1]),"=f"(acc_s[nt][2]),"=f"(acc_s[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_s[nt][0]),"f"(acc_s[nt][1]),"f"(acc_s[nt][2]),"f"(acc_s[nt][3]));
                }
            }

            // === Register-based softmax with causal mask ===
            unsigned int row0=qk_warp_m+group_id, row1=row0+8;
            #pragma unroll
            for (int nt=0; nt<4; nt++) {
                acc_s[nt][0]*=inv_sqrt_d; acc_s[nt][1]*=inv_sqrt_d;
                acc_s[nt][2]*=inv_sqrt_d; acc_s[nt][3]*=inv_sqrt_d;
                unsigned int c0=nt*8+tid_in_group*2, c1=c0+1;
                unsigned int qr0=q_rope_pos+q_start+row0, qr1=q_rope_pos+q_start+row1;
                // Causal mask: only enforce when causal_mask_enabled (default 1).
                // DFlash γ-block runs with causal_mask_enabled=0 so the γ
                // queries attend bidirectionally within their block; the prefix
                // KV positions are still strictly < q_offset so they need no
                // mask in the non-causal mode.
                if(causal_mask_enabled){
                    if(kv_start+c0>qr0) acc_s[nt][0]=-1e30f; if(kv_start+c1>qr0) acc_s[nt][1]=-1e30f;
                    if(kv_start+c0>qr1) acc_s[nt][2]=-1e30f; if(kv_start+c1>qr1) acc_s[nt][3]=-1e30f;
                }
                if(c0>=kv_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][2]=-1e30f;}
                if(c1>=kv_tile_len){acc_s[nt][1]=-1e30f;acc_s[nt][3]=-1e30f;}
                if(row0>=q_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][1]=-1e30f;}
                if(row1>=q_tile_len){acc_s[nt][2]=-1e30f;acc_s[nt][3]=-1e30f;}
                // Sliding window mask: K positions outside [Q-window+1, Q]. Only
                // evaluate after causal mask so (qr - kv_pos) is non-negative.
                if(sliding_window>0){
                    if(qr0>=kv_start+c0 && qr0-(kv_start+c0)>=sliding_window) acc_s[nt][0]=-1e30f;
                    if(qr0>=kv_start+c1 && qr0-(kv_start+c1)>=sliding_window) acc_s[nt][1]=-1e30f;
                    if(qr1>=kv_start+c0 && qr1-(kv_start+c0)>=sliding_window) acc_s[nt][2]=-1e30f;
                    if(qr1>=kv_start+c1 && qr1-(kv_start+c1)>=sliding_window) acc_s[nt][3]=-1e30f;
                }
            }

            float rmax0=-1e30f, rmax1=-1e30f;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                rmax0=fmaxf(rmax0,fmaxf(acc_s[nt][0],acc_s[nt][1]));
                rmax1=fmaxf(rmax1,fmaxf(acc_s[nt][2],acc_s[nt][3]));
            }
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,1));
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,2));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,1));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,2));

            // Online softmax: conditional rescaling (FA4-style)
            float mn0=fmaxf(m_r0,rmax0);
            if (mn0 != m_r0) {
                float eo0=sw_exp(m_r0-mn0); l_r0*=eo0;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=eo0;acc_o[i][1]*=eo0;}
                m_r0=mn0;
            }
            float mn1=fmaxf(m_r1,rmax1);
            if (mn1 != m_r1) {
                float eo1=sw_exp(m_r1-mn1); l_r1*=eo1;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=eo1;acc_o[i][3]*=eo1;}
                m_r1=mn1;
            }

            float sum0=0, sum1=0;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                float p00=sw_exp(acc_s[nt][0]-m_r0),p01=sw_exp(acc_s[nt][1]-m_r0);
                float p10=sw_exp(acc_s[nt][2]-m_r1),p11=sw_exp(acc_s[nt][3]-m_r1);
                sum0+=p00+p01; sum1+=p10+p11;
                unsigned int c0=nt*8+tid_in_group*2;
#ifdef ATLAS_DISABLE_FP16_PV
                smem_P[row0][c0]=__float2bfloat16_rn(p00); smem_P[row0][c0+1]=__float2bfloat16_rn(p01);
                smem_P[row1][c0]=__float2bfloat16_rn(p10); smem_P[row1][c0+1]=__float2bfloat16_rn(p11);
#else
                smem_P[row0][c0]=__float2half_rn(p00); smem_P[row0][c0+1]=__float2half_rn(p01);
                smem_P[row1][c0]=__float2half_rn(p10); smem_P[row1][c0+1]=__float2half_rn(p11);
#endif
            }
            sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,1); sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,2);
            sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,1); sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,2);
            l_r0+=sum0; l_r1+=sum1;

            if(tid_in_group==0){
                smem_ml[row0][0]=m_r0; smem_ml[row0][1]=l_r0;
                smem_ml[row1][0]=m_r1; smem_ml[row1][1]=l_r1;
            }
        }

        // Wait for V tile load (was loading during QK^T+softmax for BF16)
        atlas_cp_wait();
        __syncthreads();

        // Warps 2-3: rescale accumulators to match current m
        if(warp_id>=2){
            unsigned int r0=pv_warp_m+group_id, r1=r0+8;
            float cm0=smem_ml[r0][0], cm1=smem_ml[r1][0];
            if (cm0 != m_r0) {
                float er0=sw_exp(m_r0-cm0);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=er0;acc_o[i][1]*=er0;}
                m_r0=cm0;
            }
            if (cm1 != m_r1) {
                float er1=sw_exp(m_r1-cm1);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=er1;acc_o[i][3]*=er1;}
                m_r1=cm1;
            }
        }

        // === Preload K[i+1] (paged, overlaps with PV for BF16 cp.async) ===
        if(kv_block+1<num_kv_blocks){
            LOAD_KV_TILE(K_cache, block_table, smem_K[ATLAS_KB(1-buf)], (kv_block+1)*BC, kv_len, kv_head, tid, blockDim.x);
            atlas_cp_commit();
        }

        // === PV MMA (all 4 warps) ===
        // Phase 2c: FP16 inputs (vs prior BF16) — 8× finer P precision,
        // same MMA shape and throughput. V converted from BF16 to FP16
        // in registers per-MMA via bf16x2_to_f16x2_bits.
        // Bisect: ATLAS_DISABLE_FP16_PV reverts to the pre-Phase-2b BF16
        // P×V MMA (direct smem_V read, .bf16.bf16 MMA op).
        {
            const unsigned short* sP=(const unsigned short*)smem_P;
            #pragma unroll
            for(unsigned int ks=0;ks<2;ks++){
                unsigned int ko=ks*16;
                unsigned int a0,a1,a2,a3;
#ifdef ATLAS_ATTN_LDMATRIX
                // ldmatrix.x4 for the P (softmax-prob) A-fragment — same lever as QK.
                { unsigned int pb=__cvta_generic_to_shared(&sP[(pv_warp_m+(lane_id&15))*p_smem_stride+(lane_id>>4)*8+ko]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(pb)); }
#else
                unsigned int ar0=pv_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=ko+tid_in_group*2, ac1=ac0+8;
                a0=*(const unsigned int*)&sP[ar0*p_smem_stride+ac0];
                a1=*(const unsigned int*)&sP[ar1*p_smem_stride+ac0];
                a2=*(const unsigned int*)&sP[ar0*p_smem_stride+ac1];
                a3=*(const unsigned int*)&sP[ar1*p_smem_stride+ac1];
#endif
                #pragma unroll
                for(int nt=0;nt<N_TILES_PER_WARP;nt++){
                    unsigned int nc=(pv_n_start+nt)*8+group_id, k0=ko+tid_in_group*2, k1=k0+8;
#ifdef ATLAS_DISABLE_FP16_PV
                    const unsigned short* sV=(const unsigned short*)smem_V;
                    unsigned int b0=((unsigned int)sV[(k0+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k0*HDIM_PAD+nc];
                    unsigned int b1=((unsigned int)sV[(k1+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k1*HDIM_PAD+nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#else
#ifdef ATLAS_ATTN_FP8_SMEM
                    unsigned int b0=fp8x2_to_f16x2_bits(smem_V[k0][nc], smem_V[k0+1][nc], v_scale);
                    unsigned int b1=fp8x2_to_f16x2_bits(smem_V[k1][nc], smem_V[k1+1][nc], v_scale);
#else
                    unsigned int b0=bf16x2_to_f16x2_bits(
                        smem_V[k0][nc], smem_V[k0+1][nc]);
                    unsigned int b1=bf16x2_to_f16x2_bits(
                        smem_V[k1][nc], smem_V[k1+1][nc]);
#endif
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#endif
                }
            }
        }

        // Wait for K[i+1] prefetch to complete before next iteration
        if(kv_block+1<num_kv_blocks){
            atlas_cp_wait();
        }
        __syncthreads();
    }

    // === Final normalization and store ===
    {
        unsigned int r0=pv_warp_m+group_id, r1=r0+8;
        float il0,il1;
        if(warp_id<2){
            il0=(l_r0>0)?(1.f/l_r0):0;
            il1=(l_r1>0)?(1.f/l_r1):0;
        } else {
            float lv0=smem_ml[r0][1], lv1=smem_ml[r1][1];
            il0=(lv0>0)?(1.f/lv0):0;
            il1=(lv1>0)?(1.f/lv1):0;
        }

#ifdef PREFILL_BATCHED
        __nv_bfloat16* ob=O+q_batch_off+q_head*head_dim;
#else
        __nv_bfloat16* ob=O+q_head*head_dim;
#endif
        #pragma unroll
        for(int nt=0;nt<N_TILES_PER_WARP;nt++){
            unsigned int c0=(pv_n_start+nt)*8+tid_in_group*2;
            unsigned int gr0=q_start+r0, gr1=q_start+r1;
            if(gr0<q_len&&r0<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0]*il0));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1]*il0));
                *(unsigned int*)&ob[gr0*q_seq_stride+c0]=lo|(hi<<16);
            }
            if(gr1<q_len&&r1<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2]*il1));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3]*il1));
                *(unsigned int*)&ob[gr1*q_seq_stride+c0]=lo|(hi<<16);
            }
        }
    }
}

// ============================================================================
// BR=64 variant: 8 warps (256 threads) for longer sequences (q_len >= 256).
//
// Key differences from BR=32:
//   - 64 Q rows per CTA (halves Q-block count, halves causal iterations)
//   - 256 threads → 2× faster K tile loads (critical for NVFP4 dequant)
//   - Warp-specialized V loading: warps 4-7 load V during QK^T (warps 0-3)
//   - QK^T: warps 0-3, each owns 16 M-rows
//   - PV:   all 8 warps in 4 pairs:
//           (0,4)→rows 0-15, (1,5)→rows 16-31,
//           (2,6)→rows 32-47, (3,7)→rows 48-63
//           Each warp handles 16 of 32 N-tiles (128 of 256 head_dim columns)
//
// Shared memory (~88 KB, within 228 KB/SM on GB10):
//   Q:   [64][264] = 33.0 KB
//   K:   [2][32][264] = 33.0 KB  (double-buffered)
//   V:   [32][264] = 16.5 KB
//   P:   [64][40]  =  5.0 KB
//   m/l: [64][2]   =  0.5 KB
// ============================================================================

// Under SCALE/gfx1151 the BR64=64 large-chunk prefill kernels are
// COMPILE-ONLY (force_br32_prefill routes all dispatch to the BR=32
// kernel — see HARDWARE.toml / paged_attn.rs). BR64=32 here only needs
// to make them fit RDNA3.5's 64 KB LDS so the binary builds; they are
// never launched on AMD, so the host grid (still BR64=64) is irrelevant.
// NVIDIA keeps BR64=64 verbatim.
#if defined(__SCALE__)
#define BR64 32
#else
#define BR64 64
#endif
#define TILE_CHUNKS_Q64 (BR64 * (HDIM / 8))

#define _PAGED_CONCAT(a, b) a##b
#define PAGED_CONCAT(a, b) _PAGED_CONCAT(a, b)

extern "C" __global__ void PAGED_CONCAT(KERNEL_NAME, _64)(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
#ifdef PREFILL_BATCHED
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
#else
    const int* __restrict__ block_table,
#endif
    const unsigned int q_len,
    unsigned int kv_len,
    unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,
    const unsigned int causal_mask_enabled
    KERNEL_EXTRA_PARAMS
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
#ifdef PREFILL_BATCHED
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const int* const __restrict__ block_table = block_table_ptrs[b];
#endif
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR64;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR64, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
#ifdef PREFILL_BATCHED
    const unsigned long long q_batch_off = (unsigned long long)b * q_len * q_seq_stride;
#endif

    __shared__ __nv_bfloat16 smem_Q64[BR64][HDIM_PAD];
#ifdef ATLAS_ATTN_FP8_SMEM
    __shared__ __nv_fp8_storage_t smem_K64[ATLAS_KBUFN][BC][HDIM_PAD];
    __shared__ __nv_fp8_storage_t smem_V64[BC][HDIM_PAD];
#else
    __shared__ __nv_bfloat16 smem_K64[ATLAS_KBUFN][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V64[BC][HDIM_PAD];
#endif
    // Phase 2c: smem_P64 FP16 — same rationale as smem_P above.
#ifdef ATLAS_DISABLE_FP16_PV
    __shared__ __nv_bfloat16 smem_P64[BR64][BC + PAD_P];
#else
    __shared__ __half smem_P64[BR64][BC + PAD_P];
#endif
    __shared__ float smem_ml64[BR64][2];

    KERNEL_PREAMBLE

    // q_rope_pos: absolute position used to rotate the query block. Indirect
    // (DFlash) declares it in KERNEL_PREAMBLE from a device u32 (= true decode
    // position, decoupled from cache-slot base). All other variants: equals
    // q_offset (correct for causal attention where RoPE pos == cache base).
#ifndef Q_ROPE_POS_OVERRIDE
    unsigned int q_rope_pos = q_offset;
#endif

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = warp_id * 16;           // warps 0-3, each 16 rows
    const unsigned int pv_warp_m = (warp_id & 3) * 16;     // pairs (0,4),(1,5),(2,6),(3,7)
    const unsigned int pv_n_start = (warp_id >> 2) * N_TILES_PER_WARP;
    const unsigned int p_smem_stride64 = BC + PAD_P;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }

    // === Merged Q(64 rows) + K[0](32 rows) load ===
    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 256) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            if (q_start + row < q_len) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                atlas_cp16(&smem_Q64[row][col], gm);
            } else { *((uint4*)&smem_Q64[row][col]) = make_uint4(0,0,0,0); }
        }
        if (num_kv_blocks > 0) {
            LOAD_KV_TILE(K_cache, block_table, smem_K64[0], 0, kv_len, kv_head, tid, blockDim.x);
        }
        atlas_cp_commit();
        atlas_cp_wait();
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // === Warp-specialized: QK^T (warps 0-3) || V load (warps 4-7) ===
        // Warps 4-7 load V tile with 128 threads while warps 0-3 compute QK^T.
        // For FP8/NVFP4 (sync dequant): true overlap of ALU (dequant) with MMA (QK^T).
        // For BF16 (cp.async): async copies issued by 128 threads, DMA bandwidth unchanged.
        float acc_s[4][4];
        if (warp_id < 4) {
            #pragma unroll
            for (int i = 0; i < 4; i++) { acc_s[i][0]=0; acc_s[i][1]=0; acc_s[i][2]=0; acc_s[i][3]=0; }

            const unsigned short* sQ = (const unsigned short*)smem_Q64;
#ifdef ATLAS_ATTN_FP8_SMEM
            const __nv_fp8_storage_t* sK = (const __nv_fp8_storage_t*)smem_K64[ATLAS_KB(buf)];
#else
            const unsigned short* sK = (const unsigned short*)smem_K64[ATLAS_KB(buf)];
#endif

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM/16); ks++) {
                unsigned int kb = ks*16;
                unsigned int a0,a1,a2,a3;
#ifdef ATLAS_ATTN_LDMATRIX
                // SM121 ldmatrix.x4 NON-trans for the Q A-fragment (v47-proven on
                // GB10): one instr replaces 4 manual smem loads, shortening the
                // load->MMA dependency chain this latency-bound kernel is gated on.
                // sQ-relative addressing so the same code serves BR32/BR64.
                { unsigned int qb=__cvta_generic_to_shared(&sQ[(qk_warp_m+(lane_id&15))*HDIM_PAD+(lane_id>>4)*8+kb]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(qb)); }
#else
                unsigned int ar0=qk_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=kb+tid_in_group*2, ac1=ac0+8;
                a0=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac0];
                a1=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac0];
                a2=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac1];
                a3=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac1];
#endif

                #pragma unroll
                for (int nt=0; nt<4; nt++) {
                    unsigned int nc=nt*8+group_id, k0=kb+tid_in_group*2, k1=k0+8;
#ifdef ATLAS_ATTN_FP8_SMEM
                    unsigned int b0=fp8x2_to_bf16x2_bits(sK[nc*HDIM_PAD+k0],sK[nc*HDIM_PAD+k0+1],k_scale);
                    unsigned int b1=fp8x2_to_bf16x2_bits(sK[nc*HDIM_PAD+k1],sK[nc*HDIM_PAD+k1+1],k_scale);
#else
                    unsigned int b0=((unsigned int)sK[nc*HDIM_PAD+k0+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k0];
                    unsigned int b1=((unsigned int)sK[nc*HDIM_PAD+k1+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k1];
#endif
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_s[nt][0]),"=f"(acc_s[nt][1]),"=f"(acc_s[nt][2]),"=f"(acc_s[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_s[nt][0]),"f"(acc_s[nt][1]),"f"(acc_s[nt][2]),"f"(acc_s[nt][3]));
                }
            }

            // === Register-based softmax with causal mask ===
            unsigned int row0=qk_warp_m+group_id, row1=row0+8;
            #pragma unroll
            for (int nt=0; nt<4; nt++) {
                acc_s[nt][0]*=inv_sqrt_d; acc_s[nt][1]*=inv_sqrt_d;
                acc_s[nt][2]*=inv_sqrt_d; acc_s[nt][3]*=inv_sqrt_d;
                unsigned int c0=nt*8+tid_in_group*2, c1=c0+1;
                unsigned int qr0=q_rope_pos+q_start+row0, qr1=q_rope_pos+q_start+row1;
                // Causal mask gated for DFlash γ-block (causal_mask_enabled=0).
                if(causal_mask_enabled){
                    if(kv_start+c0>qr0) acc_s[nt][0]=-1e30f; if(kv_start+c1>qr0) acc_s[nt][1]=-1e30f;
                    if(kv_start+c0>qr1) acc_s[nt][2]=-1e30f; if(kv_start+c1>qr1) acc_s[nt][3]=-1e30f;
                }
                if(c0>=kv_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][2]=-1e30f;}
                if(c1>=kv_tile_len){acc_s[nt][1]=-1e30f;acc_s[nt][3]=-1e30f;}
                if(row0>=q_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][1]=-1e30f;}
                if(row1>=q_tile_len){acc_s[nt][2]=-1e30f;acc_s[nt][3]=-1e30f;}
                // Sliding window mask: K positions outside [Q-window+1, Q]. Only
                // evaluate after causal mask so (qr - kv_pos) is non-negative.
                if(sliding_window>0){
                    if(qr0>=kv_start+c0 && qr0-(kv_start+c0)>=sliding_window) acc_s[nt][0]=-1e30f;
                    if(qr0>=kv_start+c1 && qr0-(kv_start+c1)>=sliding_window) acc_s[nt][1]=-1e30f;
                    if(qr1>=kv_start+c0 && qr1-(kv_start+c0)>=sliding_window) acc_s[nt][2]=-1e30f;
                    if(qr1>=kv_start+c1 && qr1-(kv_start+c1)>=sliding_window) acc_s[nt][3]=-1e30f;
                }
            }

            float rmax0=-1e30f, rmax1=-1e30f;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                rmax0=fmaxf(rmax0,fmaxf(acc_s[nt][0],acc_s[nt][1]));
                rmax1=fmaxf(rmax1,fmaxf(acc_s[nt][2],acc_s[nt][3]));
            }
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,1));
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,2));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,1));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,2));

            float mn0=fmaxf(m_r0,rmax0);
            if (mn0 != m_r0) {
                float eo0=sw_exp(m_r0-mn0); l_r0*=eo0;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=eo0;acc_o[i][1]*=eo0;}
                m_r0=mn0;
            }
            float mn1=fmaxf(m_r1,rmax1);
            if (mn1 != m_r1) {
                float eo1=sw_exp(m_r1-mn1); l_r1*=eo1;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=eo1;acc_o[i][3]*=eo1;}
                m_r1=mn1;
            }

            float sum0=0, sum1=0;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                float p00=sw_exp(acc_s[nt][0]-m_r0),p01=sw_exp(acc_s[nt][1]-m_r0);
                float p10=sw_exp(acc_s[nt][2]-m_r1),p11=sw_exp(acc_s[nt][3]-m_r1);
                sum0+=p00+p01; sum1+=p10+p11;
                unsigned int c0=nt*8+tid_in_group*2;
#ifdef ATLAS_DISABLE_FP16_PV
                smem_P64[row0][c0]=__float2bfloat16_rn(p00); smem_P64[row0][c0+1]=__float2bfloat16_rn(p01);
                smem_P64[row1][c0]=__float2bfloat16_rn(p10); smem_P64[row1][c0+1]=__float2bfloat16_rn(p11);
#else
                smem_P64[row0][c0]=__float2half_rn(p00); smem_P64[row0][c0+1]=__float2half_rn(p01);
                smem_P64[row1][c0]=__float2half_rn(p10); smem_P64[row1][c0+1]=__float2half_rn(p11);
#endif
            }
            sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,1); sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,2);
            sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,1); sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,2);
            l_r0+=sum0; l_r1+=sum1;

            if(tid_in_group==0){
                smem_ml64[row0][0]=m_r0; smem_ml64[row0][1]=l_r0;
                smem_ml64[row1][0]=m_r1; smem_ml64[row1][1]=l_r1;
            }
            // Warps 0-3: commit empty cp.async group (balance with warps 4-7)
            atlas_cp_commit();
        } else {
            // Warps 4-7: load V tile (128 threads, overlaps with QK^T above)
            LOAD_KV_TILE(V_cache, block_table, smem_V64, kv_start, kv_len, kv_head, tid - 128, 128);
            atlas_cp_commit();
        }

        // Wait for V loads to complete (warps 0-3: no-op, warps 4-7: wait for copies)
        atlas_cp_wait();
        __syncthreads();

        // Warps 4-7: rescale accumulators to match current m
        if(warp_id>=4){
            unsigned int r0=pv_warp_m+group_id, r1=r0+8;
            float cm0=smem_ml64[r0][0], cm1=smem_ml64[r1][0];
            if (cm0 != m_r0) {
                float er0=sw_exp(m_r0-cm0);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=er0;acc_o[i][1]*=er0;}
                m_r0=cm0;
            }
            if (cm1 != m_r1) {
                float er1=sw_exp(m_r1-cm1);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=er1;acc_o[i][3]*=er1;}
                m_r1=cm1;
            }
        }

        // === Preload K[i+1] (256 threads = 2× faster) ===
        if(kv_block+1<num_kv_blocks){
            LOAD_KV_TILE(K_cache, block_table, smem_K64[ATLAS_KB(1-buf)], (kv_block+1)*BC, kv_len, kv_head, tid, blockDim.x);
            atlas_cp_commit();
        }

        // === PV MMA (all 8 warps) ===
        {
            // Phase 2c: FP16 PV MMA — see BR=32 path above for rationale.
            const unsigned short* sP=(const unsigned short*)smem_P64;
            #pragma unroll
            for(unsigned int ks=0;ks<2;ks++){
                unsigned int ko=ks*16;
                unsigned int a0,a1,a2,a3;
#ifdef ATLAS_ATTN_LDMATRIX
                // ldmatrix.x4 for the P (softmax-prob) A-fragment — same lever as QK.
                { unsigned int pb=__cvta_generic_to_shared(&sP[(pv_warp_m+(lane_id&15))*p_smem_stride64+(lane_id>>4)*8+ko]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(pb)); }
#else
                unsigned int ar0=pv_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=ko+tid_in_group*2, ac1=ac0+8;
                a0=*(const unsigned int*)&sP[ar0*p_smem_stride64+ac0];
                a1=*(const unsigned int*)&sP[ar1*p_smem_stride64+ac0];
                a2=*(const unsigned int*)&sP[ar0*p_smem_stride64+ac1];
                a3=*(const unsigned int*)&sP[ar1*p_smem_stride64+ac1];
#endif
                #pragma unroll
                for(int nt=0;nt<N_TILES_PER_WARP;nt++){
                    unsigned int nc=(pv_n_start+nt)*8+group_id, k0=ko+tid_in_group*2, k1=k0+8;
#ifdef ATLAS_DISABLE_FP16_PV
                    const unsigned short* sV=(const unsigned short*)smem_V64;
                    unsigned int b0=((unsigned int)sV[(k0+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k0*HDIM_PAD+nc];
                    unsigned int b1=((unsigned int)sV[(k1+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k1*HDIM_PAD+nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#else
#ifdef ATLAS_ATTN_FP8_SMEM
                    unsigned int b0=fp8x2_to_f16x2_bits(smem_V64[k0][nc], smem_V64[k0+1][nc], v_scale);
                    unsigned int b1=fp8x2_to_f16x2_bits(smem_V64[k1][nc], smem_V64[k1+1][nc], v_scale);
#else
                    unsigned int b0=bf16x2_to_f16x2_bits(
                        smem_V64[k0][nc], smem_V64[k0+1][nc]);
                    unsigned int b1=bf16x2_to_f16x2_bits(
                        smem_V64[k1][nc], smem_V64[k1+1][nc]);
#endif
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#endif
                }
            }
        }

        if(kv_block+1<num_kv_blocks){
            atlas_cp_wait();
        }
        __syncthreads();
    }

    // === Final normalization and store ===
    {
        unsigned int r0=pv_warp_m+group_id, r1=r0+8;
        float il0,il1;
        if(warp_id<4){
            il0=(l_r0>0)?(1.f/l_r0):0;
            il1=(l_r1>0)?(1.f/l_r1):0;
        } else {
            float lv0=smem_ml64[r0][1], lv1=smem_ml64[r1][1];
            il0=(lv0>0)?(1.f/lv0):0;
            il1=(lv1>0)?(1.f/lv1):0;
        }

#ifdef PREFILL_BATCHED
        __nv_bfloat16* ob=O+q_batch_off+q_head*head_dim;
#else
        __nv_bfloat16* ob=O+q_head*head_dim;
#endif
        #pragma unroll
        for(int nt=0;nt<N_TILES_PER_WARP;nt++){
            unsigned int c0=(pv_n_start+nt)*8+tid_in_group*2;
            unsigned int gr0=q_start+r0, gr1=q_start+r1;
            if(gr0<q_len&&r0<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0]*il0));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1]*il0));
                *(unsigned int*)&ob[gr0*q_seq_stride+c0]=lo|(hi<<16);
            }
            if(gr1<q_len&&r1<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2]*il1));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3]*il1));
                *(unsigned int*)&ob[gr1*q_seq_stride+c0]=lo|(hi<<16);
            }
        }
    }
}
