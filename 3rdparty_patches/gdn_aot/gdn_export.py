import os, torch
# Capture the compiled CuTe kernel by wrapping cached_compile.
import flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache as cc
import flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 as dr
captured = []
_orig = cc.cached_compile
def _wrap(func, *a, **k):
    cf = _orig(func, *a, **k)
    captured.append(cf)
    return cf
cc.cached_compile = _wrap
dr.cached_compile = _wrap

from flashinfer.gdn_prefill import chunk_gated_delta_rule
T, Hqk, Hv, D = 2048, 16, 32, 128
dt, dev = torch.bfloat16, "cuda"
q = torch.randn(T, Hqk, D, dtype=dt, device=dev)
k = torch.nn.functional.normalize(torch.randn(T, Hqk, D, dtype=torch.float32, device=dev), dim=-1).to(dt)
v = torch.randn(T, Hv, D, dtype=dt, device=dev)
g = torch.rand(T, Hv, dtype=torch.float32, device=dev)
beta = torch.rand(T, Hv, dtype=torch.float32, device=dev).sigmoid()
cu = torch.tensor([0, T], dtype=torch.int64, device=dev)
h0 = torch.zeros(1, Hv, D, D, dtype=torch.float32, device=dev)
so = torch.zeros_like(h0)
_ = chunk_gated_delta_rule(q, k, v, g, beta, None, h0, True, cu, False, None, so)
torch.cuda.synchronize()
print(f"GDN ran at Holo shape (T={T} Hqk={Hqk} Hv={Hv} D={D}); captured {len(captured)} compiled kernel(s)")
for i, cf in enumerate(captured):
    print(f"  [{i}] {type(cf).__name__}  export_to_c={hasattr(cf,'export_to_c')}")

os.makedirs("/tmp/gdn_aot", exist_ok=True)
ok = 0
for i, cf in enumerate(captured):
    try:
        cf.export_to_c("/tmp/gdn_aot", f"gdn_holo_{i}", f"gdn_holo_{i}")
        print(f"  EXPORTED [{i}] -> /tmp/gdn_aot/gdn_holo_{i}.{{h,o}}")
        ok += 1
    except Exception as e:
        import traceback; print(f"  EXPORT FAIL [{i}]:", str(e)[:300])
print(f"=== exported {ok}/{len(captured)} ===")
