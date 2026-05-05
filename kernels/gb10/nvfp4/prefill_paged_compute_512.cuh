// SPDX-License-Identifier: AGPL-3.0-only

// HDIM=512 paged Flash Attention compute (Gemma-4 full-attention layers).
//
// Constraints (NVIDIA GB10 / sm_121f):
//   - Per-block opt-in dynamic smem cap = 99 KB (101,376 bytes,
//     queried via cuDeviceGetAttribute(attrib=97))
//   - Standard 4-warp BR=32 BC=32 template at HDIM=512 needs ~135 KB → fails
//
// This template's design (fits 101,120 bytes):
//   - BR=32, BC=32, PAD_KV=0 (saves 1.5 KB; bank conflicts in K/V smem
//     reads accepted as a v1 perf cost — correctness first)
//   - Single-buffered K (saves 33 KB vs double-buffer)
//   - 8 warps (256 threads):
//       * QK^T:    warps 0-1 (each owns 16 Q rows)
//       * V load:  warps 2-7 (192 threads, async cp.async)
//       * PV:      all 8 warps split (col_group, row_group) over HDIM=512
//                  pv_warp_m=(wid&1)*16, pv_n_start=(wid>>1)*16
//                  → each warp 16 col-tiles × 16 rows = 64 acc_o floats/thread
//
// Smem layout (dynamic, total 101,120 B):
//   Q   [32][512]  = 32,768
//   K   [32][512]  = 32,768  (single-buffered)
//   V   [32][512]  = 32,768
//   P   [32][40]   =  2,560  (PAD_P=8 to mitigate bank conflict on PV read)
//   m/l [32][2]f32 =    256
//
// CALLER MUST: KernelLaunch::shared_mem(101120). The registry auto-issues
// cuFuncSetAttribute(MAX_DYNAMIC_SHARED, 101120) when shared_mem > 48 KB.
//
// Expects the including .cu file to define:
//   LOAD_KV_TILE_512(cache, bt, smem_ptr, kv_s, kv_l, kvh, t, stride)
//   KERNEL_NAME, K_CACHE_TYPE, V_CACHE_TYPE,
//   KERNEL_EXTRA_PARAMS, KERNEL_PREAMBLE

#include <cuda_bf16.h>

__device__ __forceinline__ float sw_exp_512(float x) {
    float t = x * 1.4426950408889634f;
    float ti = floorf(t);
    float tf = t - ti;
    float p = 1.0f + tf * (0.6931471805599453f +
              tf * (0.2402265069591007f +
              tf * 0.05550410866482158f));
    return ldexpf(p, (int)ti);
}

#define BR_512   32
#define BC_512   32
#define HDIM_512 512
#define PAD_P_512 8
#define N_TILES_PER_WARP_512 16  // (HDIM/8) / 4 col-groups
#define TILE_CHUNKS_512 (BR_512 * (HDIM_512 / 8))

extern "C" __global__ void KERNEL_NAME(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_table,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,  // 0 = full attn; >0 = mask K where (Q - K) >= window
    const unsigned int causal_mask_enabled  // 1 = causal (default); 0 = bidirectional (DFlash γ-block)
    KERNEL_EXTRA_PARAMS
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR_512;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR_512, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    extern __shared__ __align__(16) unsigned char smem_dyn_512[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_dyn_512);
    __nv_bfloat16* smem_K = smem_Q + (unsigned int)BR_512 * HDIM_512;
    __nv_bfloat16* smem_V = smem_K + (unsigned int)BC_512 * HDIM_512;
    __nv_bfloat16* smem_P = smem_V + (unsigned int)BC_512 * HDIM_512;
    float*         smem_ml = reinterpret_cast<float*>(
                       smem_P + (unsigned int)BR_512 * (BC_512 + PAD_P_512));

    KERNEL_PREAMBLE

    const unsigned int group_id     = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m    = (warp_id & 1) * 16;          // warps 0-1
    const unsigned int pv_warp_m    = (warp_id & 1) * 16;          // pair: rows 0-15/16-31
    const unsigned int pv_n_start   = (warp_id >> 1) * N_TILES_PER_WARP_512;
    const unsigned int p_smem_stride = BC_512 + PAD_P_512;

    float acc_o[N_TILES_PER_WARP_512][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP_512; i++) {
        acc_o[i][0]=0.f; acc_o[i][1]=0.f; acc_o[i][2]=0.f; acc_o[i][3]=0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f,    l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + BC_512 - 1) / BC_512;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC_512;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }

    // === Initial Q + K[0] load (256 threads) ===
    {
        const unsigned int cpr = HDIM_512 / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_512; idx += blockDim.x) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * HDIM_512 + col]);
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gm));
            } else { *((uint4*)&smem_Q[row * HDIM_512 + col]) = make_uint4(0,0,0,0); }
        }
        if (num_kv_blocks > 0) {
            LOAD_KV_TILE_512(K_cache, block_table, smem_K, 0, kv_len, kv_head, tid, blockDim.x);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC_512;
        unsigned int kv_end   = min(kv_start + BC_512, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        // === Warp-specialized: warps 0-1 QK^T  ||  warps 2-7 V load ===
        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i=0;i<4;i++){acc_s[i][0]=0;acc_s[i][1]=0;acc_s[i][2]=0;acc_s[i][3]=0;}
            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)smem_K;

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM_512/16); ks++) {
                unsigned int kb = ks*16;
                unsigned int ar0=qk_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=kb+tid_in_group*2, ac1=ac0+8;
                unsigned int a0=*(const unsigned int*)&sQ[ar0*HDIM_512+ac0];
                unsigned int a1=*(const unsigned int*)&sQ[ar1*HDIM_512+ac0];
                unsigned int a2=*(const unsigned int*)&sQ[ar0*HDIM_512+ac1];
                unsigned int a3=*(const unsigned int*)&sQ[ar1*HDIM_512+ac1];
                #pragma unroll
                for (int nt=0; nt<4; nt++) {
                    unsigned int nc=nt*8+group_id, k0=kb+tid_in_group*2, k1=k0+8;
                    unsigned int b0=((unsigned int)sK[nc*HDIM_512+k0+1]<<16)|(unsigned int)sK[nc*HDIM_512+k0];
                    unsigned int b1=((unsigned int)sK[nc*HDIM_512+k1+1]<<16)|(unsigned int)sK[nc*HDIM_512+k1];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_s[nt][0]),"=f"(acc_s[nt][1]),"=f"(acc_s[nt][2]),"=f"(acc_s[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_s[nt][0]),"f"(acc_s[nt][1]),"f"(acc_s[nt][2]),"f"(acc_s[nt][3]));
                }
            }

            unsigned int row0=qk_warp_m+group_id, row1=row0+8;
            #pragma unroll
            for (int nt=0;nt<4;nt++) {
                acc_s[nt][0]*=inv_sqrt_d; acc_s[nt][1]*=inv_sqrt_d;
                acc_s[nt][2]*=inv_sqrt_d; acc_s[nt][3]*=inv_sqrt_d;
                unsigned int c0=nt*8+tid_in_group*2, c1=c0+1;
                unsigned int qr0=q_offset+q_start+row0, qr1=q_offset+q_start+row1;
                if(causal_mask_enabled){
                    if(kv_start+c0>qr0) acc_s[nt][0]=-1e30f; if(kv_start+c1>qr0) acc_s[nt][1]=-1e30f;
                    if(kv_start+c0>qr1) acc_s[nt][2]=-1e30f; if(kv_start+c1>qr1) acc_s[nt][3]=-1e30f;
                }
                if(c0>=kv_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][2]=-1e30f;}
                if(c1>=kv_tile_len){acc_s[nt][1]=-1e30f;acc_s[nt][3]=-1e30f;}
                if(row0>=q_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][1]=-1e30f;}
                if(row1>=q_tile_len){acc_s[nt][2]=-1e30f;acc_s[nt][3]=-1e30f;}
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
                float eo0=sw_exp_512(m_r0-mn0); l_r0*=eo0;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP_512;i++){acc_o[i][0]*=eo0;acc_o[i][1]*=eo0;}
                m_r0=mn0;
            }
            float mn1=fmaxf(m_r1,rmax1);
            if (mn1 != m_r1) {
                float eo1=sw_exp_512(m_r1-mn1); l_r1*=eo1;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP_512;i++){acc_o[i][2]*=eo1;acc_o[i][3]*=eo1;}
                m_r1=mn1;
            }

            float sum0=0, sum1=0;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                float p00=sw_exp_512(acc_s[nt][0]-m_r0),p01=sw_exp_512(acc_s[nt][1]-m_r0);
                float p10=sw_exp_512(acc_s[nt][2]-m_r1),p11=sw_exp_512(acc_s[nt][3]-m_r1);
                sum0+=p00+p01; sum1+=p10+p11;
                unsigned int c0=nt*8+tid_in_group*2;
                smem_P[row0*p_smem_stride+c0]   = __float2bfloat16(p00);
                smem_P[row0*p_smem_stride+c0+1] = __float2bfloat16(p01);
                smem_P[row1*p_smem_stride+c0]   = __float2bfloat16(p10);
                smem_P[row1*p_smem_stride+c0+1] = __float2bfloat16(p11);
            }
            sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,1); sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,2);
            sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,1); sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,2);
            l_r0+=sum0; l_r1+=sum1;

            if(tid_in_group==0){
                smem_ml[row0*2  ]=m_r0; smem_ml[row0*2+1]=l_r0;
                smem_ml[row1*2  ]=m_r1; smem_ml[row1*2+1]=l_r1;
            }
            asm volatile("cp.async.commit_group;");  // empty: balance with V load
        } else {
            // Warps 2-7 (192 threads): cooperative V tile load.
            // Renumber threads 0..191 within this group; share LOAD_KV_TILE_512 macro.
            LOAD_KV_TILE_512(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid - 64, 192);
            asm volatile("cp.async.commit_group;");
        }

        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 2-7 rescale to current m before PV
        if(warp_id>=2){
            unsigned int r0=pv_warp_m+group_id, r1=r0+8;
            float cm0=smem_ml[r0*2], cm1=smem_ml[r1*2];
            if (cm0 != m_r0) {
                float er0=sw_exp_512(m_r0-cm0);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP_512;i++){acc_o[i][0]*=er0;acc_o[i][1]*=er0;}
                m_r0=cm0;
            }
            if (cm1 != m_r1) {
                float er1=sw_exp_512(m_r1-cm1);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP_512;i++){acc_o[i][2]*=er1;acc_o[i][3]*=er1;}
                m_r1=cm1;
            }
        }

        // === PV MMA (all 8 warps) ===
        {
            const unsigned short* sP=(const unsigned short*)smem_P;
            const unsigned short* sV=(const unsigned short*)smem_V;
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
                for(int nt=0;nt<N_TILES_PER_WARP_512;nt++){
                    unsigned int nc=(pv_n_start+nt)*8+group_id, k0=ko+tid_in_group*2, k1=k0+8;
                    unsigned int b0=((unsigned int)sV[(k0+1)*HDIM_512+nc]<<16)|(unsigned int)sV[k0*HDIM_512+nc];
                    unsigned int b1=((unsigned int)sV[(k1+1)*HDIM_512+nc]<<16)|(unsigned int)sV[k1*HDIM_512+nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
                }
            }
        }

        __syncthreads();

        // === Sequential K[next] load (single-buffered, all 256 threads) ===
        if(kv_block+1 < num_kv_blocks){
            LOAD_KV_TILE_512(K_cache, block_table, smem_K, (kv_block+1)*BC_512, kv_len, kv_head, tid, blockDim.x);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
    }

    // === Final normalization and store ===
    {
        unsigned int r0=pv_warp_m+group_id, r1=r0+8;
        float il0,il1;
        if(warp_id<2){
            il0=(l_r0>0)?(1.f/l_r0):0;
            il1=(l_r1>0)?(1.f/l_r1):0;
        } else {
            float lv0=smem_ml[r0*2+1], lv1=smem_ml[r1*2+1];
            il0=(lv0>0)?(1.f/lv0):0;
            il1=(lv1>0)?(1.f/lv1):0;
        }

        __nv_bfloat16* ob = O + q_head*head_dim;
        #pragma unroll
        for(int nt=0;nt<N_TILES_PER_WARP_512;nt++){
            unsigned int c0 = (pv_n_start+nt)*8 + tid_in_group*2;
            unsigned int gr0 = q_start+r0, gr1 = q_start+r1;
            if(gr0<q_len && r0<q_tile_len && c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0]*il0));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1]*il0));
                *(unsigned int*)&ob[gr0*q_seq_stride+c0]=lo|(hi<<16);
            }
            if(gr1<q_len && r1<q_tile_len && c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2]*il1));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3]*il1));
                *(unsigned int*)&ob[gr1*q_seq_stride+c0]=lo|(hi<<16);
            }
        }
    }
}
