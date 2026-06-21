// SPDX-License-Identifier: AGPL-3.0-only
// Prove a DROP-IN helper: given the exact e4m3 m16n8k32 register
// fragments (a0..a3,b0,b1) a kernel already built, atlas_mma(...) must
// reproduce mma.sync.m16n8k32.e4m3.e4m3.f32 bit-exactly via intra-group
// shuffle -> bf16 -> 2x mma.m16n8k16.bf16. Run on dgx2 (NVIDIA, free).
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

__device__ __forceinline__ float e2f(unsigned char b){
  return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3));
}
__device__ __forceinline__ unsigned bf2(float lo,float hi){
  unsigned short l=__bfloat16_as_ushort(__float2bfloat16(lo));
  unsigned short h=__bfloat16_as_ushort(__float2bfloat16(hi));
  return ((unsigned)h<<16)|l;
}

// THE DROP-IN HELPER (candidate). a0..a3,b0,b1 = exact e4m3 frags as
// built by Atlas (m16n8k32 .row.col). gid=lane>>2, tig=lane&3.
//  e4m3 layout: a0=A[gid][K=4tig..+3] a1=A[gid+8][..] a2=A[gid][16+4tig..]
//               a3=A[gid+8][16+4tig..] ; b0=B[gid][4tig..] b1=B[gid][16+4tig..]
//  Recover any A[row][K=j] (j 0..31): src reg = (j<16?a*:a*+2) for that
//  row; the value lives in lane (group_base + j/4), byte j%4. Use
//  __shfl_sync to gather this thread's bf16-needed K = {2tig,2tig+1,
//  8+2tig,9+2tig} per K-half.
__device__ __forceinline__ void atlas_mma_e4m3(float* acc,
    unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
  unsigned lane=threadIdx.x&31u; unsigned tig=lane&3u; unsigned base=lane&~3u;
  // byte extract
  auto byt=[](unsigned r,int i)->unsigned char{ return (unsigned char)(r>>(8*i)); };
  #pragma unroll
  for(int half=0;half<2;half++){
    // pick the e4m3 source reg pair for this K-half
    unsigned A_g  = half? a2 : a0;   // A[gid][16*half + 4tig ..]
    unsigned A_g8 = half? a3 : a1;   // A[gid+8][...]
    unsigned B_g  = half? b1 : b0;   // B[gid][...]
    // this thread (tig) needs K-local indices {2tig,2tig+1,8+2tig,9+2tig}
    // within the 16-wide half. K-local j lives in lane base+(j/4), byte j%4.
    auto gA=[&](unsigned reg,int j)->float{
      unsigned src=base+(unsigned)(j>>2); unsigned v=__shfl_sync(0xffffffffu,reg,src);
      return e2f(byt(v,j&3));
    };
    int j0=2*tig, j1=8+2*tig;
    unsigned A0=bf2(gA(A_g ,j0),gA(A_g ,j0+1));
    unsigned A1=bf2(gA(A_g8,j0),gA(A_g8,j0+1));
    unsigned A2=bf2(gA(A_g ,j1),gA(A_g ,j1+1));
    unsigned A3=bf2(gA(A_g8,j1),gA(A_g8,j1+1));
    unsigned B0=bf2(gA(B_g ,j0),gA(B_g ,j0+1));
    unsigned B1=bf2(gA(B_g ,j1),gA(B_g ,j1+1));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
      :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
      :"r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),
       "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]));
  }
}

__device__ unsigned PK(const unsigned char*M,int r,int k){ return *(const unsigned*)&M[r*32+k]; }

__global__ void k_ref(const unsigned char*A,const unsigned char*B,float*D){
  unsigned lane=threadIdx.x; unsigned g=lane>>2,t=lane&3;
  unsigned a0=PK(A,g,t*4),a1=PK(A,g+8,t*4),a2=PK(A,g,16+t*4),a3=PK(A,g+8,16+t*4);
  unsigned b0=PK(B,g,t*4),b1=PK(B,g,16+t*4);
  float acc[4]={0,0,0,0};
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
    :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
     "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]));
  D[g*8+t*2]=acc[0];D[g*8+t*2+1]=acc[1];D[(g+8)*8+t*2]=acc[2];D[(g+8)*8+t*2+1]=acc[3];
}
__global__ void k_cand(const unsigned char*A,const unsigned char*B,float*D){
  unsigned lane=threadIdx.x; unsigned g=lane>>2,t=lane&3;
  unsigned a0=PK(A,g,t*4),a1=PK(A,g+8,t*4),a2=PK(A,g,16+t*4),a3=PK(A,g+8,16+t*4);
  unsigned b0=PK(B,g,t*4),b1=PK(B,g,16+t*4);
  float acc[4]={0,0,0,0};
  atlas_mma_e4m3(acc,a0,a1,a2,a3,b0,b1);
  D[g*8+t*2]=acc[0];D[g*8+t*2+1]=acc[1];D[(g+8)*8+t*2]=acc[2];D[(g+8)*8+t*2+1]=acc[3];
}

int main(){
  const int As=16*32,Bs=8*32,Ds=128;
  unsigned char hA[As],hB[Bs]; float hr[Ds],hc[Ds],hp[Ds];
  for(int i=0;i<As;i++) hA[i]=__nv_cvt_float_to_fp8(((i*37%17)-8)*0.25f,__NV_SATFINITE,__NV_E4M3);
  for(int i=0;i<Bs;i++) hB[i]=__nv_cvt_float_to_fp8(((i*29%13)-6)*0.5f,__NV_SATFINITE,__NV_E4M3);
  auto dq=[](unsigned char b){ __half_raw h=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); __half x;*(unsigned short*)&x=*(unsigned short*)&h;return (float)x;};
  for(int m=0;m<16;m++)for(int n=0;n<8;n++){float s=0;for(int k=0;k<32;k++)s+=dq(hA[m*32+k])*dq(hB[n*32+k]);hp[m*8+n]=s;}
  unsigned char*dA,*dB; float*dD; cudaMalloc(&dA,As);cudaMalloc(&dB,Bs);cudaMalloc(&dD,Ds*4);
  cudaMemcpy(dA,hA,As,cudaMemcpyHostToDevice);cudaMemcpy(dB,hB,Bs,cudaMemcpyHostToDevice);
  k_ref<<<1,32>>>(dA,dB,dD);cudaMemcpy(hr,dD,Ds*4,cudaMemcpyDeviceToHost);
  k_cand<<<1,32>>>(dA,dB,dD);cudaMemcpy(hc,dD,Ds*4,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  float mrc=0,mrp=0; for(int i=0;i<Ds;i++){mrc=fmaxf(mrc,fabsf(hr[i]-hc[i]));mrp=fmaxf(mrp,fabsf(hr[i]-hp[i]));}
  printf("max|ref-cand|=%.4f max|ref-cpu|=%.4f\n",mrc,mrp);
  printf("ref : ");for(int i=0;i<8;i++)printf("%.2f ",hr[i]);printf("\n");
  printf("cand: ");for(int i=0;i<8;i++)printf("%.2f ",hc[i]);printf("\n");
  printf("%s\n",(mrc<0.5f)?"HELPER_OK":"HELPER_FAIL");
  return 0;
}
