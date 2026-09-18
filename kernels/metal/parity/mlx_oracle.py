import struct, numpy as np, mlx.core as mx
def bits(x): return struct.unpack('<I', struct.pack('<f', np.float32(x)))[0]
def hexf(x): return f"0x{bits(x):08x}"
A = lambda v: mx.array(np.asarray(v, dtype=np.float32))
print(f"mlx {getattr(mx,'__version__','?')}  device {mx.default_device()}")

# ---- 1. FMA: SEARCH for inputs where the two behaviours actually differ.
rng = np.random.default_rng(7)
cands = []
for _ in range(200000):
    a, b, c = rng.standard_normal(3).astype(np.float32)
    two = np.float32(np.float32(a) * np.float32(b)) + np.float32(c)
    fma = np.float32(np.float64(a) * np.float64(b) + np.float64(c))
    if bits(two) != bits(fma):
        cands.append((a, b, c, two, fma))
        if len(cands) >= 64: break
print(f"\n[1] FMA contraction — {len(cands)} DISCRIMINATING input(s) found "
      f"(inputs where two-roundings != contracted differ at all)")
if cands:
    av = np.array([x[0] for x in cands], dtype=np.float32)
    bv = np.array([x[1] for x in cands], dtype=np.float32)
    cv = np.array([x[2] for x in cands], dtype=np.float32)
    got = np.array(A(av) * A(bv) + A(cv), copy=False)
    n_fma = sum(bits(g) == bits(x[4]) for g, x in zip(got, cands))
    n_two = sum(bits(g) == bits(x[3]) for g, x in zip(got, cands))
    print(f"    matches contracted-fma : {n_fma}/{len(cands)}")
    print(f"    matches two-roundings  : {n_two}/{len(cands)}")
    verdict = ("CONTRACTS" if n_fma == len(cands) else
               "does NOT contract" if n_two == len(cands) else "MIXED — not a single rule")
    print(f"    => MLX {verdict}")
    a,b,c,two,fma = cands[0]
    print(f"    example a={hexf(a)} b={hexf(b)} c={hexf(c)} -> two {hexf(two)} fma {hexf(fma)} MLX {hexf(got[0])}")

# ---- 2. transcendentals over a wide sweep
xs = np.concatenate([np.linspace(0.01, 20, 2000), np.linspace(20, 88, 500)]).astype(np.float32)
for name, mxf, npf in (("exp", mx.exp, np.exp), ("rsqrt", mx.rsqrt, lambda v: np.float32(1.0)/np.sqrt(v))):
    g = np.array(mxf(A(xs)), copy=False); r = npf(xs).astype(np.float32)
    gb = np.array([bits(v) for v in g]); rb = np.array([bits(v) for v in r])
    diff = np.abs(gb.astype(np.int64) - rb.astype(np.int64))
    print(f"\n[2] {name}: bit-identical to numpy on {int((diff==0).sum())}/{len(xs)}; "
          f"max |ulp delta| {int(diff.max())}")

# ---- 3. reduction: MANY permutations, not one
v = rng.standard_normal(4096).astype(np.float32)
base = np.array(mx.sum(A(v)), copy=False).item()
same = 0; trials = 32
for _ in range(trials):
    p = rng.permutation(len(v))
    s = np.array(mx.sum(A(v[p])), copy=False).item()
    same += (bits(s) == bits(base))
print(f"\n[3] mx.sum over 4096 f32, {trials} random permutations")
print(f"    permutation-invariant in {same}/{trials}")
print(f"    => {'ORDER-INDEPENDENT on this shape' if same==trials else 'ORDER-DEPENDENT — bit result depends on input order'}")
# run-to-run determinism on identical input
r1 = np.array(mx.sum(A(v)), copy=False).item(); r2 = np.array(mx.sum(A(v)), copy=False).item()
print(f"    run-to-run on identical input: {'deterministic' if bits(r1)==bits(r2) else 'NON-deterministic'}")
