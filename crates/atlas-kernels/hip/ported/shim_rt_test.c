#include <stdio.h>
#include <stdlib.h>
typedef unsigned long long CUdeviceptr;
extern int cuMemAlloc_v2(CUdeviceptr*, size_t);
extern int cuMemcpyHtoDAsync_v2(CUdeviceptr, const void*, size_t, void*);
extern int cuMemcpyDtoHAsync_v2(void*, CUdeviceptr, size_t, void*);
extern int cuModuleLoadData(void**, const void*);
extern int cuModuleGetFunction(void**, void*, const char*);
extern int cuLaunchKernel(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**);
extern int cuStreamSynchronize(void*);
int main(){
  int n=256; float h[256]; for(int i=0;i<n;i++) h[i]=(float)i;
  FILE*f=fopen("addone.co","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  void*img=malloc(sz); fread(img,1,sz,f); fclose(f);
  CUdeviceptr d; int e;
  e=cuMemAlloc_v2(&d,n*4); printf("cuMemAlloc=%d\n",e);
  e=cuMemcpyHtoDAsync_v2(d,h,n*4,0); printf("cuMemcpyHtoD=%d\n",e);
  void*mod; e=cuModuleLoadData(&mod,img); printf("cuModuleLoadData=%d\n",e);
  void*fn; e=cuModuleGetFunction(&fn,mod,"add_one"); printf("cuModuleGetFunction=%d\n",e);
  void*args[2]; args[0]=&d; args[1]=&n;
  e=cuLaunchKernel(fn,1,1,1,256,1,1,0,0,args,0); printf("cuLaunchKernel=%d\n",e);
  cuStreamSynchronize(0);
  cuMemcpyDtoHAsync_v2(h,d,n*4,0); cuStreamSynchronize(0);
  int ok=1; for(int i=0;i<n;i++) if(h[i]!=(float)i+1.0f){ok=0;printf("MISMATCH i=%d got=%f\n",i,h[i]);break;}
  printf("%s add_one via shim: h[0]=%.1f h[255]=%.1f\n", ok?"PASS":"FAIL", h[0], h[255]);
  return !ok;
}
