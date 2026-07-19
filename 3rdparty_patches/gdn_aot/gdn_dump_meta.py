import os, torch, numpy as np
import flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache as cc
import flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 as dr
_orig=cc.cached_compile
class Proxy:
    def __init__(s,cf): s._cf=cf; s.last=None
    def __call__(s,*a,**k): s.last=(a,k); return s._cf(*a,**k)
    def __getattr__(s,n): return getattr(s._cf,n)
caps=[]
def _wrap(f,*a,**k):
    p=Proxy(_orig(f,*a,**k)); caps.append(p); return p
cc.cached_compile=_wrap; dr.cached_compile=_wrap
from flashinfer.gdn_prefill import chunk_gated_delta_rule
torch.manual_seed(0)
T,Hqk,Hv,D=2048,16,32,128
dt,dev=torch.float16,"cuda"
q=torch.randn(T,Hqk,D,dtype=dt,device=dev)
k=torch.nn.functional.normalize(torch.randn(T,Hqk,D,dtype=torch.float32,device=dev),dim=-1).to(dt)
v=torch.randn(T,Hv,D,dtype=dt,device=dev)
g=torch.rand(T,Hv,dtype=torch.float32,device=dev)
beta=torch.rand(T,Hv,dtype=torch.float32,device=dev).sigmoid()
cu=torch.tensor([0,T],dtype=torch.int64,device=dev)
h0=torch.zeros(1,Hv,D,D,dtype=torch.float32,device=dev)
out=torch.zeros(T,Hv,D,dtype=dt,device=dev); so=torch.zeros_like(h0)
chunk_gated_delta_rule(q,k,v,g,beta,None,h0,True,cu,False,out,so); torch.cuda.synchronize()
args=caps[-1].last[0]
print(f"=== {len(args)} kernel args (the C wrapper contract) ===")
for i,a in enumerate(args):
    t=type(a).__name__
    extra=""
    for attr in ("shape","stride","element_type","value","dtype"):
        try:
            val=getattr(a,attr)
            extra+=f" {attr}={val() if callable(val) else val}"
        except Exception: pass
    print(f"  [{i:2}] {t}{extra[:120]}")
# save reference I/O for the C harness
os.makedirs("/tmp/gdn_ref",exist_ok=True)
def sv(name,t): t.detach().contiguous().cpu().numpy().tofile(f"/tmp/gdn_ref/{name}.bin")
sv("q",q); sv("k",k); sv("v",v); sv("g",g); sv("beta",beta)
cu.detach().cpu().numpy().tofile("/tmp/gdn_ref/cu.bin")
sv("o_ref",out.float())
print("saved q,k,v,g,beta,cu,o_ref -> /tmp/gdn_ref/ ; o_ref |mean|=",out.float().abs().mean().item())
