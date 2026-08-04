#include "gdn_holo_0.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" void atlas_gdn_load();
extern "C" int atlas_gdn_prefill_packed(void*,void*,void*,void*,void*,void*,void*,float,int,int,int,int,int,int,int,int,void*);
static std::pair<void*,long> rd(const char*p){ FILE*f=fopen(p,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); void*h=malloc(n); fread(h,1,n,f); fclose(f); return {h,n}; }
int main(){
  const int T=2048,nk=16,nv=32,kd=128,vd=128;
  const int key_dim=nk*kd, value_dim=nv*vd, conv_dim=2*key_dim+value_dim, gb=2*nv;
  atlas_gdn_load();
  auto [hq,_q]=rd("/tmp/gdn_ref/q.bin"); auto [hk,_k]=rd("/tmp/gdn_ref/k.bin"); auto [hv,_v]=rd("/tmp/gdn_ref/v.bin");
  auto [hg,_g]=rd("/tmp/gdn_ref/g.bin"); auto [hb,_b]=rd("/tmp/gdn_ref/beta.bin");
  // pack QKV [T,conv_dim] bf16
  __half* qkv=(__half*)malloc((size_t)T*conv_dim*2);
  __half *Q=(__half*)hq,*K=(__half*)hk,*V=(__half*)hv;
  for(int t=0;t<T;t++){ __half* row=qkv+(size_t)t*conv_dim;
    for(int i=0;i<key_dim;i++) row[i]=Q[(size_t)t*key_dim+i];
    for(int i=0;i<key_dim;i++) row[key_dim+i]=K[(size_t)t*key_dim+i];
    for(int i=0;i<value_dim;i++) row[2*key_dim+i]=V[(size_t)t*value_dim+i]; }
  // interleave gate_beta [T,2nv] fp32
  float* gbuf=(float*)malloc((size_t)T*gb*4); float *G=(float*)hg,*B=(float*)hb;
  for(int t=0;t<T;t++){ for(int h=0;h<nv;h++){ gbuf[(size_t)t*gb+h]=G[(size_t)t*nv+h]; gbuf[(size_t)t*gb+nv+h]=B[(size_t)t*nv+h]; } }
  void *dqkv,*dgb,*dout,*dst,*dini,*dtm,*dcu;
  cudaMalloc(&dqkv,(size_t)T*conv_dim*2); cudaMemcpy(dqkv,qkv,(size_t)T*conv_dim*2,cudaMemcpyHostToDevice);
  cudaMalloc(&dgb,(size_t)T*gb*4); cudaMemcpy(dgb,gbuf,(size_t)T*gb*4,cudaMemcpyHostToDevice);
  cudaMalloc(&dout,(size_t)T*value_dim*2); cudaMemset(dout,0,(size_t)T*value_dim*2);
  cudaMalloc(&dst,(size_t)nv*kd*vd*4); cudaMemset(dst,0,(size_t)nv*kd*vd*4);
  cudaMalloc(&dini,(size_t)nv*kd*vd*4); cudaMemset(dini,0,(size_t)nv*kd*vd*4);
  cudaMalloc(&dtm,6144*128); cudaMemset(dtm,0,6144*128);
  long long cu_host[2]={0,T}; cudaMalloc(&dcu,16); cudaMemcpy(dcu,cu_host,16,cudaMemcpyHostToDevice);
  cudaStream_t s; cudaStreamCreate(&s);
  int ret=atlas_gdn_prefill_packed(dqkv,dgb,dout,dst,dini,dtm,dcu, 0.08838834764831843f, T,nk,nv,kd,vd,conv_dim,gb,1, s);
  cudaStreamSynchronize(s);
  printf("packed ret=%d cuda=%s\n", ret, cudaGetErrorString(cudaGetLastError()));
  size_t n=(size_t)T*value_dim; __half* ho=(__half*)malloc(n*2); cudaMemcpy(ho,dout,n*2,cudaMemcpyDeviceToHost);
  auto [hr,_r]=rd("/tmp/gdn_ref/o_ref.bin"); float* ref=(float*)hr;
  double maxe=0,so=0,dot=0,no=0,nr=0; int nan=0;
  for(size_t i=0;i<n;i++){ float ov=__half2float(ho[i]); if(isnan(ov))nan++; double e=fabs(ov-ref[i]); if(e>maxe)maxe=e; so+=fabs(ov); dot+=(double)ov*ref[i]; no+=(double)ov*ov; nr+=(double)ref[i]*ref[i]; }
  double cos=dot/(sqrt(no)*sqrt(nr)+1e-12);
  printf("max_abs_err=%.6f |o|mean=%.6f cos=%.6f nan=%d\n", maxe, so/n, cos, nan);
  printf("RESULT: %s\n",(cos>0.99&&nan==0&&so/n>1e-4)?"PASS - Atlas-native packed layout matches JIT reference":"MISMATCH");
  return 0;
}
