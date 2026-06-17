import numpy as np, pathlib
A=pathlib.Path("/tmp/atlas_fp8_dump"); V=pathlib.Path("/tmp/vllm_ref")
def rd(p):
    p=pathlib.Path(p); return np.fromfile(p,dtype="<f4") if p.exists() else None
def cos(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64);n=min(a.size,b.size);a,b=a[:n],b[:n]
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na and nb else float('nan')
def rl2(a,b):
    n=min(a.size,b.size);a,b=a[:n].astype(np.float64),b[:n].astype(np.float64)
    return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-9))
FULL=[3,7,11,15,19,23,27,31,35,39]
print("=== PER-LAYER residual cosine: Atlas-FP8 vs vLLM-FP8 (149-tok probe, current build) ===")
print(f"{'L':>3} {'type':>5} {'cos':>9} {'rel_l2':>8} {'|vllm|':>8} {'|atlas|':>8}")
onset=None
for i in range(40):
    a=rd(A/f"atlas_L{i}.bin"); v=rd(V/f"vllm_L{i}.bin")
    if a is None or v is None: print(f"{i:>3}  missing"); continue
    c=cos(a,v); typ="attn" if i in FULL else "ssm"; flag=""
    if c<0.999 and onset is None: onset=i; flag="  <-- ONSET"
    print(f"{i:>3} {typ:>5} {c:>9.5f} {rl2(a,v):>8.4f} {np.linalg.norm(v):>8.2f} {np.linalg.norm(a):>8.2f}{flag}")
print(f"\nDIVERGENCE ONSET: L{onset}")
print("\n=== PER-OP o_proj cosine (attn layers): Atlas vs vLLM ===")
for rel,abs_i in enumerate(FULL):
    a=rd(A/f"atlas_op_L{rel}_o_proj.bin"); v=rd(V/f"vllm_op_L{abs_i}_o_proj.bin")
    if a is None or v is None: print(f"  L{abs_i}: missing (a={a is not None} v={v is not None})"); continue
    print(f"  L{abs_i:>2}  o_proj  cos={cos(a,v):.5f}  rel_l2={rl2(a,v):.4f}")
