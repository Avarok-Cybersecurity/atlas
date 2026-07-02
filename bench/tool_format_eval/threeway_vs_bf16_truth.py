import numpy as np, pathlib
AT=pathlib.Path("/tmp/atlas_fp8_dump"); VL=pathlib.Path("/tmp/vllm_ref"); VL2=pathlib.Path("/tmp/vllm_ref2"); BF=pathlib.Path("/tmp/bf16_truth")
def rd(p): p=pathlib.Path(p); return np.fromfile(p,dtype="<f4") if pathlib.Path(p).exists() else None
def cd(a,b):
    if a is None or b is None: return None
    n=min(a.size,b.size);x,y=a[:n].astype(np.float64),b[:n].astype(np.float64)
    nx,ny=np.linalg.norm(x),np.linalg.norm(y)
    cos=float(np.dot(x,y)/(nx*ny)) if nx and ny else float('nan')
    return (cos, float(np.linalg.norm(x-y)/(ny+1e-9)))
def fmt(r): return f"{r[0]:.5f}/{r[1]:.4f}" if r else "  -  "
FULL=[3,7,11,15]; ATTN_REL={3:0,7:1,11:2,15:3}
print("cos/rel_l2 vs BF16 GROUND TRUTH   (lower rel_l2 = closer to correct)")
print(f"{'op':>22} {'layer':>6} {'Atlas-FP8':>16} {'vLLM-FP8':>16}   winner")
print('-'*82)
# per-layer residual (all L0..15)
for i in range(16):
    a=cd(rd(AT/f"atlas_L{i}.bin"), rd(BF/f"hf_op_L{i}_layer_out.bin"))
    v=cd(rd(VL/f"vllm_L{i}.bin"),  rd(BF/f"hf_op_L{i}_layer_out.bin"))
    w = "" if (a is None or v is None) else ("ATLAS closer" if a[1]<v[1] else "vLLM closer")
    print(f"{'residual':>22} {i:>6} {fmt(a):>16} {fmt(v):>16}   {w}")
print('-'*82)
# attention-layer ops: o_proj, q/k/v proj, k_after_norm
for op,af,bf,vf in [("o_proj","atlas_op_L{r}_o_proj.bin","hf_op_L{a}_o_proj.bin","vllm_op_L{a}_o_proj.bin"),
                    ("q_proj","atlas_op_L{r}_q_proj_full.bin","hf_op_L{a}_q_proj_full.bin",None),
                    ("k_proj","atlas_op_L{r}_k_proj.bin","hf_op_L{a}_k_proj.bin",None),
                    ("v_proj","atlas_op_L{r}_v_proj.bin","hf_op_L{a}_v_proj.bin",None),
                    ("k_after_norm","atlas_op_L{r}_k_post_norm.bin","hf_op_L{a}_k_after_norm.bin",None),
                    ("post_attn_norm","atlas_op_L{r}_post_attn_norm_out.bin","hf_op_L{a}_post_attn_norm_out.bin",None)]:
    for L in FULL:
        r=ATTN_REL[L]
        a=cd(rd(AT/(af.format(r=r))), rd(BF/(bf.format(a=L))))
        if vf: v=cd(rd(VL2/(vf.format(a=L))), rd(BF/(bf.format(a=L))))
        else:  v=None
        w = "" if a is None else ("" if v is None else ("ATLAS closer" if a[1]<v[1] else "vLLM closer"))
        print(f"{op:>22} {L:>6} {fmt(a):>16} {fmt(v):>16}   {w}")
