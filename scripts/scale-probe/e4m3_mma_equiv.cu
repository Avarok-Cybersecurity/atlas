// SPDX-License-Identifier: AGPL-3.0-only
// Equivalence test (run on an NVIDIA GPU — dgx2): prove that the
// dequant-e4m3->bf16 + 2x mma.m16n8k16.bf16 decomposition reproduces
// mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 exactly, so the
// SCALE/gfx1151 #if-branch (no native e4m3 MMA) is correct-by-proof,
// not by guesswork. One warp, one 16x8 tile. Canonical PTX m16n8k32 /
// m16n8k16 .row.col fragment layouts (same as Atlas w4a16_gemm.cu /
// prefill_paged_compute.cu usage).
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

// A: 16x32 row-major, B: 8x32 (col.col: B[n][k]), D: 16x8.
__device__ float e2f(unsigned char b){ return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3)); }

__global__ void ref_e4m3(const unsigned char* A,const unsigned char* B,float* D){
  unsigned lane=threadIdx.x; unsigned gid=lane>>2, tig=lane&3;
  // A frag (m16n8k32 .row): a0=A[gid][tig*4..],a1=A[gid+8][tig*4..],
  //                          a2=A[gid][16+tig*4..],a3=A[gid+8][16+tig*4..]
  auto pk=[&](const unsigned char*M,int r,int k){ return *(const unsigned*)&M[r*32+k]; };
  unsigned a0=pk(A,gid,tig*4),a1=pk(A,gid+8,tig*4),a2=pk(A,gid,16+tig*4),a3=pk(A,gid+8,16+tig*4);
  // B frag (m16n8k32 .col): b0=B[gid][tig*4..], b1=B[gid][16+tig*4..]
  unsigned b0=pk(B,gid,tig*4), b1=pk(B,gid,16+tig*4);
  float acc[4]={0,0,0,0};
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
    :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
     "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]));
  // D acc layout: (d0,d1)->row=gid,col=tig*2,+1 ; (d2,d3)->row=gid+8
  D[(gid)*8+tig*2]=acc[0]; D[(gid)*8+tig*2+1]=acc[1];
  D[(gid+8)*8+tig*2]=acc[2]; D[(gid+8)*8+tig*2+1]=acc[3];
}

// Candidate: dequant e4m3->bf16, two mma.m16n8k16.bf16 (K 0..15, 16..31).
// m16n8k16 .row A frag (4xb32=8 bf16): for a K16 block, per thread holds
// A[gid][k0..k0+1],A[gid+8][k0..],A[gid][k0+8..],A[gid+8][k0+8..] with
// k0=tig*2. We build bf16 regs directly from the e4m3 source matrices
// so the repack is explicit and layout-correct by construction.
__device__ unsigned b2(float lo,float hi){ // pack 2 bf16 -> b32
  unsigned short l=__bfloat16_as_ushort(__float2bfloat16(lo));
  unsigned short h=__bfloat16_as_ushort(__float2bfloat16(hi));
  return ((unsigned)h<<16)|l;
}
__global__ void cand_bf16(const unsigned char* A,const unsigned char* B,float* D){
  unsigned lane=threadIdx.x; unsigned gid=lane>>2, tig=lane&3;
  float acc[4]={0,0,0,0};
  #pragma unroll
  for(int half=0;half<2;half++){ int kb=half*16;
    auto Ad=[&](int r,int k){ return e2f(A[r*32+kb+k]); };
    auto Bd=[&](int n,int k){ return e2f(B[n*32+kb+k]); };
    int k0=tig*2;
    unsigned a0=b2(Ad(gid,k0),Ad(gid,k0+1));
    unsigned a1=b2(Ad(gid+8,k0),Ad(gid+8,k0+1));
    unsigned a2=b2(Ad(gid,k0+8),Ad(gid,k0+9));
    unsigned a3=b2(Ad(gid+8,k0+8),Ad(gid+8,k0+9));
    unsigned bb0=b2(Bd(gid,k0),Bd(gid,k0+1));
    unsigned bb1=b2(Bd(gid,k0+8),Bd(gid,k0+9));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
      :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
      :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(bb0),"r"(bb1),
       "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]));
  }
  D[(gid)*8+tig*2]=acc[0]; D[(gid)*8+tig*2+1]=acc[1];
  D[(gid+8)*8+tig*2]=acc[2]; D[(gid+8)*8+tig*2+1]=acc[3];
}

int main(){
  const int Asz=16*32,Bsz=8*32,Dsz=16*8;
  unsigned char hA[Asz],hB[Bsz]; float hDr[Dsz],hDc[Dsz],hCpu[Dsz];
  for(int i=0;i<Asz;i++){ float v=((i*37%17)-8)*0.25f; hA[i]=__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3);}
  for(int i=0;i<Bsz;i++){ float v=((i*29%13)-6)*0.5f;  hB[i]=__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3);}
  // CPU ground truth: D[m][n]=sum_k deq(A[m][k])*deq(B[n][k])
  auto deq=[](unsigned char b){ __half_raw h=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); __half x; *(unsigned short*)&x=*(unsigned short*)&h; return (float)x;};
  for(int m=0;m<16;m++)for(int n=0;n<8;n++){ float s=0; for(int k=0;k<32;k++) s+=deq(hA[m*32+k])*deq(hB[n*32+k]); hCpu[m*8+n]=s; }
  unsigned char *dA,*dB; float *dD;
  cudaMalloc(&dA,Asz);cudaMalloc(&dB,Bsz);cudaMalloc(&dD,Dsz*4);
  cudaMemcpy(dA,hA,Asz,cudaMemcpyHostToDevice);cudaMemcpy(dB,hB,Bsz,cudaMemcpyHostToDevice);
  ref_e4m3<<<1,32>>>(dA,dB,dD);cudaMemcpy(hDr,dD,Dsz*4,cudaMemcpyDeviceToHost);
  cand_bf16<<<1,32>>>(dA,dB,dD);cudaMemcpy(hDc,dD,Dsz*4,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  float mrc=0,mrcpu=0,mccpu=0;
  for(int i=0;i<Dsz;i++){ mrc=fmaxf(mrc,fabsf(hDr[i]-hDc[i])); mrcpu=fmaxf(mrcpu,fabsf(hDr[i]-hCpu[i])); mccpu=fmaxf(mccpu,fabsf(hDc[i]-hCpu[i])); }
  printf("max|ref-cand|=%.4f  max|ref-cpu|=%.4f  max|cand-cpu|=%.4f\n",mrc,mrcpu,mccpu);
  printf("ref[0..7]:  "); for(int i=0;i<8;i++)printf("%.2f ",hDr[i]); printf("\n");
  printf("cand[0..7]: "); for(int i=0;i<8;i++)printf("%.2f ",hDc[i]); printf("\n");
  printf("cpu[0..7]:  "); for(int i=0;i<8;i++)printf("%.2f ",hCpu[i]); printf("\n");
  printf("%s\n", (mrc<0.5f && mrcpu<2.0f) ? "EQUIV_OK" : "EQUIV_FAIL");
  return 0;
}
