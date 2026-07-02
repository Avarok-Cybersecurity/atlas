// SPDX-License-Identifier: AGPL-3.0-only

// Per-Q-head paged Flash Attention compute — ASYMMETRIC K/V variant.
//
// Fork of prefill_paged_compute.cuh that takes TWO tile-loader macros
// (`LOAD_K_TILE` for K loads, `LOAD_V_TILE` for V loads). Enables kernels
// where K and V have different on-disk byte layouts — for example Bf16K +
// Turbo3V where K is BF16 and V is 3-bit packed Lloyd-Max + per-group FP8.
//
// Grid: (num_q_heads, ceil(q_len/BR), 1) — one block per Q head.
// Same pipeline as the symmetric kernel; only the loader macros differ.
//
// Expects the including file to define:
//   LOAD_K_TILE(cache, block_table, smem, kv_start, kv_len, kv_head, tid, stride)
//   LOAD_V_TILE(cache, block_table, smem, kv_start, kv_len, kv_head, tid, stride)
//   KERNEL_NAME, K_CACHE_TYPE, V_CACHE_TYPE, KERNEL_EXTRA_PARAMS, KERNEL_PREAMBLE

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Async global→shared 16-byte copy helpers (cp.async on NVIDIA + SCALE).
// The strix-hip copy of this header degrades these to synchronous uint4
// copies (AMD has no cp.async). Per-tree behavior comes purely from which
// header is compiled — no #if at the call sites.
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
// `smem_V` stays BF16 so the LOAD_V_TILE macros (BF16 cp.async, FP8
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
#define BC 32
#ifndef HDIM
#define HDIM 256
#endif
#define PAD_KV 8
#define HDIM_PAD (HDIM + PAD_KV)
#define PAD_P 8
#define N_TILES_PER_WARP ((HDIM / 8) / 2)
#define TILE_CHUNKS (BR * (HDIM / 8))

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
    const unsigned int kv_len,
    const unsigned int q_offset,
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
    __shared__ __nv_bfloat16 smem_K[2][BC][HDIM_PAD];  // double-buffered
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    // Phase 2c: smem_P now FP16 (10-bit mantissa) vs BF16 (7-bit).
    // Read back as 2x packed FP16 per .b32 register for the .f16.f16 MMA.
    __shared__ __half smem_P[BR][BC + PAD_P];
    __shared__ float smem_ml[BR][2];

    KERNEL_PREAMBLE

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
    // Q via cp.async, K[0] via LOAD_K_TILE (path varies by K dtype).
    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += blockDim.x) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q[row][col]);
            if (q_start + row < q_len) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gm));
            } else { *((uint4*)&smem_Q[row][col]) = make_uint4(0,0,0,0); }
        }
        if (num_kv_blocks > 0) {
            LOAD_K_TILE(K_cache, block_table, smem_K[0], 0, kv_len, kv_head, tid, blockDim.x);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // === Start V load (overlaps with QK^T) ===
        LOAD_V_TILE(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, blockDim.x);
        asm volatile("cp.async.commit_group;");

        // === QK^T (warps 0-1, register-based) ===
        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < 4; i++) { acc_s[i][0]=0; acc_s[i][1]=0; acc_s[i][2]=0; acc_s[i][3]=0; }

            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)smem_K[buf];

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM/16); ks++) {
                unsigned int kb = ks*16;
                unsigned int ar0=qk_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=kb+tid_in_group*2, ac1=ac0+8;
                unsigned int a0=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac0];
                unsigned int a1=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac0];
                unsigned int a2=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac1];
                unsigned int a3=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac1];

                #pragma unroll
                for (int nt=0; nt<4; nt++) {
                    unsigned int nc=nt*8+group_id, k0=kb+tid_in_group*2, k1=k0+8;
                    unsigned int b0=((unsigned int)sK[nc*HDIM_PAD+k0+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k0];
                    unsigned int b1=((unsigned int)sK[nc*HDIM_PAD+k1+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k1];
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
                unsigned int qr0=q_offset+q_start+row0, qr1=q_offset+q_start+row1;
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
                smem_P[row0][c0]=__float2half_rn(p00); smem_P[row0][c0+1]=__float2half_rn(p01);
                smem_P[row1][c0]=__float2half_rn(p10); smem_P[row1][c0+1]=__float2half_rn(p11);
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
        asm volatile("cp.async.wait_group 0;");
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
            LOAD_K_TILE(K_cache, block_table, smem_K[1-buf], (kv_block+1)*BC, kv_len, kv_head, tid, blockDim.x);
            asm volatile("cp.async.commit_group;");
        }

        // === PV MMA (all 4 warps) ===
        // Phase 2c: FP16 inputs (vs prior BF16) — 8× finer P precision,
        // same MMA shape and throughput. V converted from BF16 to FP16
        // in registers per-MMA via bf16x2_to_f16x2_bits.
        {
            const unsigned short* sP=(const unsigned short*)smem_P;
            #pragma unroll
            for(unsigned int ks=0;ks<2;ks++){
                unsigned int ko=ks*16;
                unsigned int ar0=pv_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=ko+tid_in_group*2, ac1=ac0+8;
                unsigned int a0=*(const unsigned int*)&sP[ar0*p_smem_stride+ac0];
                unsigned int a1=*(const unsigned int*)&sP[ar1*p_smem_stride+ac0];
                unsigned int a2=*(const unsigned int*)&sP[ar0*p_smem_stride+ac1];
                unsigned int a3=*(const unsigned int*)&sP[ar1*p_smem_stride+ac1];
                #pragma unroll
                for(int nt=0;nt<N_TILES_PER_WARP;nt++){
                    unsigned int nc=(pv_n_start+nt)*8+group_id, k0=ko+tid_in_group*2, k1=k0+8;
                    unsigned int b0=bf16x2_to_f16x2_bits(
                        smem_V[k0][nc], smem_V[k0+1][nc]);
                    unsigned int b1=bf16x2_to_f16x2_bits(
                        smem_V[k1][nc], smem_V[k1+1][nc]);
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
                }
            }
        }

        // Wait for K[i+1] prefetch to complete before next iteration
        if(kv_block+1<num_kv_blocks){
            asm volatile("cp.async.wait_group 0;");
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

#define BR64 64
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
    const unsigned int kv_len,
    const unsigned int q_offset,
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
    __shared__ __nv_bfloat16 smem_K64[2][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V64[BC][HDIM_PAD];
    // Phase 2c: smem_P64 now FP16 — same rationale as smem_P above.
    __shared__ __half smem_P64[BR64][BC + PAD_P];
    __shared__ float smem_ml64[BR64][2];

    KERNEL_PREAMBLE

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
            unsigned int sa = __cvta_generic_to_shared(&smem_Q64[row][col]);
            if (q_start + row < q_len) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gm));
            } else { *((uint4*)&smem_Q64[row][col]) = make_uint4(0,0,0,0); }
        }
        if (num_kv_blocks > 0) {
            LOAD_K_TILE(K_cache, block_table, smem_K64[0], 0, kv_len, kv_head, tid, blockDim.x);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
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
            const unsigned short* sK = (const unsigned short*)smem_K64[buf];

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM/16); ks++) {
                unsigned int kb = ks*16;
                unsigned int ar0=qk_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=kb+tid_in_group*2, ac1=ac0+8;
                unsigned int a0=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac0];
                unsigned int a1=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac0];
                unsigned int a2=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac1];
                unsigned int a3=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac1];

                #pragma unroll
                for (int nt=0; nt<4; nt++) {
                    unsigned int nc=nt*8+group_id, k0=kb+tid_in_group*2, k1=k0+8;
                    unsigned int b0=((unsigned int)sK[nc*HDIM_PAD+k0+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k0];
                    unsigned int b1=((unsigned int)sK[nc*HDIM_PAD+k1+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k1];
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
                unsigned int qr0=q_offset+q_start+row0, qr1=q_offset+q_start+row1;
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
                smem_P64[row0][c0]=__float2half_rn(p00); smem_P64[row0][c0+1]=__float2half_rn(p01);
                smem_P64[row1][c0]=__float2half_rn(p10); smem_P64[row1][c0+1]=__float2half_rn(p11);
            }
            sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,1); sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,2);
            sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,1); sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,2);
            l_r0+=sum0; l_r1+=sum1;

            if(tid_in_group==0){
                smem_ml64[row0][0]=m_r0; smem_ml64[row0][1]=l_r0;
                smem_ml64[row1][0]=m_r1; smem_ml64[row1][1]=l_r1;
            }
            // Warps 0-3: commit empty cp.async group (balance with warps 4-7)
            asm volatile("cp.async.commit_group;");
        } else {
            // Warps 4-7: load V tile (128 threads, overlaps with QK^T above)
            LOAD_V_TILE(V_cache, block_table, smem_V64, kv_start, kv_len, kv_head, tid - 128, 128);
            asm volatile("cp.async.commit_group;");
        }

        // Wait for V loads to complete (warps 0-3: no-op, warps 4-7: wait for copies)
        asm volatile("cp.async.wait_group 0;");
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
            LOAD_K_TILE(K_cache, block_table, smem_K64[1-buf], (kv_block+1)*BC, kv_len, kv_head, tid, blockDim.x);
            asm volatile("cp.async.commit_group;");
        }

        // === PV MMA (all 8 warps) ===
        {
            // Phase 2c: FP16 PV MMA — see BR=32 path above for rationale.
            const unsigned short* sP=(const unsigned short*)smem_P64;
            #pragma unroll
            for(unsigned int ks=0;ks<2;ks++){
                unsigned int ko=ks*16;
                unsigned int ar0=pv_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=ko+tid_in_group*2, ac1=ac0+8;
                unsigned int a0=*(const unsigned int*)&sP[ar0*p_smem_stride64+ac0];
                unsigned int a1=*(const unsigned int*)&sP[ar1*p_smem_stride64+ac0];
                unsigned int a2=*(const unsigned int*)&sP[ar0*p_smem_stride64+ac1];
                unsigned int a3=*(const unsigned int*)&sP[ar1*p_smem_stride64+ac1];
                #pragma unroll
                for(int nt=0;nt<N_TILES_PER_WARP;nt++){
                    unsigned int nc=(pv_n_start+nt)*8+group_id, k0=ko+tid_in_group*2, k1=k0+8;
                    unsigned int b0=bf16x2_to_f16x2_bits(
                        smem_V64[k0][nc], smem_V64[k0+1][nc]);
                    unsigned int b1=bf16x2_to_f16x2_bits(
                        smem_V64[k1][nc], smem_V64[k1+1][nc]);
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
                }
            }
        }

        if(kv_block+1<num_kv_blocks){
            asm volatile("cp.async.wait_group 0;");
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
