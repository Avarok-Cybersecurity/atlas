#include "gdn_holo_0.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

static void* load_bin(const char* path, size_t* nbytes){
  FILE* f=fopen(path,"rb"); if(!f){printf("missing %s\n",path);exit(1);}
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  void* h=malloc(n); if(fread(h,1,n,f)!=(size_t)n){printf("read fail\n");exit(1);} fclose(f);
  void* d; cudaMalloc(&d,n); cudaMemcpy(d,h,n,cudaMemcpyHostToDevice);
  if(nbytes)*nbytes=n; free(h); return d;
}
int main(){
  const int T=2048,Hv=32,D=128;
  size_t nb;
  void* dq=load_bin("/tmp/gdn_ref/q.bin",&nb);
  void* dk=load_bin("/tmp/gdn_ref/k.bin",&nb);
  void* dv=load_bin("/tmp/gdn_ref/v.bin",&nb);
  void* dg=load_bin("/tmp/gdn_ref/g.bin",&nb);
  void* dbeta=load_bin("/tmp/gdn_ref/beta.bin",&nb);
  int64_t cu_host[2]={0,(int64_t)T}; void* dcu; cudaMalloc(&dcu,16); cudaMemcpy(dcu,cu_host,16,cudaMemcpyHostToDevice);
  void *dout,*dstate,*dinit,*dtm;
  cudaMalloc(&dout,(size_t)T*Hv*D*2); cudaMemset(dout,0,(size_t)T*Hv*D*2);
  cudaMalloc(&dstate,(size_t)Hv*D*D*4); cudaMemset(dstate,0,(size_t)Hv*D*D*4);
  cudaMalloc(&dinit,(size_t)Hv*D*D*4); cudaMemset(dinit,0,(size_t)Hv*D*D*4);
  cudaMalloc(&dtm,(size_t)6144*128); cudaMemset(dtm,0,(size_t)6144*128);

  gdn_holo_0_Kernel_Module_t module; gdn_holo_0_Kernel_Module_Load(&module);

  gdn_holo_0_Tensor_g_q_t       g_q   ={dq,    {2048,128,16},{2048,128}};
  gdn_holo_0_Tensor_g_k_t       g_k   ={dk,    {128,2048,16},{2048,128}};
  gdn_holo_0_Tensor_g_v_t       g_v   ={dv,    {128,2048,32},{4096,128}};
  gdn_holo_0_Tensor_g_o_t       g_o   ={dout,  {128,2048,32},{4096,128}};
  gdn_holo_0_Tensor_g_alpha_t   g_al  ={dg,    {65536}};
  gdn_holo_0_Tensor_g_beta_t    g_be  ={dbeta, {65536}};
  gdn_holo_0_Tensor_g_state_t   g_st  ={dstate,{524288}};
  gdn_holo_0_Tensor_g_init_state_t g_in={dinit,{524288}};
  gdn_holo_0_Tensor_g_tensormaps_t g_tm={dtm, {6144}};
  gdn_holo_0_Tensor_cu_seqlens_t g_cu ={dcu,  {2}};

  cudaStream_t stream; cudaStreamCreate(&stream);
  int ret=cute_dsl_gdn_holo_0_wrapper(&module,&g_q,&g_k,&g_v,&g_o,&g_al,&g_be,&g_st,&g_in,&g_tm,&g_cu,
      0.08838834764831843f, 16,16,32,32,1,1,0, 32, stream);
  cudaStreamSynchronize(stream);
  printf("wrapper ret=%d  cuda=%s\n", ret, cudaGetErrorString(cudaGetLastError()));

  // INSTRUMENT: did the kernel write state? sample output?
  { float* hs=(float*)malloc((size_t)Hv*D*D*4); cudaMemcpy(hs,dstate,(size_t)Hv*D*D*4,cudaMemcpyDeviceToHost);
    double sm=0; int snz=0; for(size_t i=0;i<(size_t)Hv*D*D;i++){ sm+=fabs(hs[i]); if(hs[i]!=0)snz++; }
    printf("STATE |mean|=%.6f nonzero=%d/%zu\n", sm/((size_t)Hv*D*D), snz, (size_t)Hv*D*D); }
  { __half* ho2=(__half*)malloc(16*2); cudaMemcpy(ho2,dout,16*2,cudaMemcpyDeviceToHost);
    printf("o[0..7]="); for(int i=0;i<8;i++) printf("%.4f ", __half2float(ho2[i])); printf("\n"); }

  size_t on=(size_t)T*Hv*D;
  __half* ho=(__half*)malloc(on*2); cudaMemcpy(ho,dout,on*2,cudaMemcpyDeviceToHost);
  FILE* fr=fopen("/tmp/gdn_ref/o_ref.bin","rb"); float* ref=(float*)malloc(on*4);
  if(fread(ref,4,on,fr)!=on){printf("ref read fail\n");return 1;} fclose(fr);
  double maxerr=0,so=0,sr=0,dot=0,no=0,nr=0; int nan=0;
  for(size_t i=0;i<on;i++){ float ov=__half2float(ho[i]); if(isnan(ov))nan++;
    double e=fabs(ov-ref[i]); if(e>maxerr)maxerr=e; so+=fabs(ov); sr+=fabs(ref[i]);
    dot+=(double)ov*ref[i]; no+=(double)ov*ov; nr+=(double)ref[i]*ref[i]; }
  double cos=dot/(sqrt(no)*sqrt(nr)+1e-12);
  printf("max_abs_err=%.6f  |o|mean=%.6f  |ref|mean=%.6f  cos=%.6f  nan=%d\n",
         maxerr, so/on, sr/on, cos, nan);
  printf("RESULT: %s\n",(cos>0.99 && nan==0 && so/on>1e-4)?"PASS - AOT C-ABI kernel matches JIT reference":"MISMATCH");
  return 0;
}
