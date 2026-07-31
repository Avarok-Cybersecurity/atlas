"""Compare two logit_gate.py dumps: exact-match, max drift, and KL.

usage: python3 logit_diff.py base.json cand.json
"""
import json, math, sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

print("%-12s %8s %9s %11s %11s %11s %9s" % (
    "prompt", "positions", "tok_same", "identical", "max|dlp|", "mean|dlp|", "max_KL"))

worst_dlp = 0.0
worst_kl = 0.0
all_identical = True

for name in a:
    if name not in b:
        print("%-12s MISSING in candidate" % name); all_identical = False; continue
    ta, tb = a[name]["tokens"] or [], b[name]["tokens"] or []
    la, lb = a[name]["token_logprobs"] or [], b[name]["token_logprobs"] or []
    n = min(len(la), len(lb))
    tok_same = sum(1 for i in range(min(len(ta), len(tb))) if ta[i] == tb[i])
    tok_tot = min(len(ta), len(tb))

    diffs = []
    for i in range(n):
        x, y = la[i], lb[i]
        if x is None or y is None:
            continue
        diffs.append(abs(x - y))
    mx = max(diffs) if diffs else 0.0
    mean = (sum(diffs) / len(diffs)) if diffs else 0.0

    # KL over the top-k distributions (renormalized on the shared support).
    kls = []
    tla, tlb = a[name]["top_logprobs"] or [], b[name]["top_logprobs"] or []
    for i in range(min(len(tla), len(tlb))):
        da, db = tla[i], tlb[i]
        if not da or not db:
            continue
        keys = set(da) & set(db)
        if not keys:
            continue
        pa = {k: math.exp(da[k]) for k in keys}
        pb = {k: math.exp(db[k]) for k in keys}
        sa, sb = sum(pa.values()), sum(pb.values())
        if sa <= 0 or sb <= 0:
            continue
        kl = sum((pa[k] / sa) * math.log((pa[k] / sa) / (pb[k] / sb))
                 for k in keys if pa[k] > 0 and pb[k] > 0)
        kls.append(abs(kl))
    mkl = max(kls) if kls else 0.0

    ident = (mx == 0.0 and tok_same == tok_tot and mkl == 0.0)
    all_identical &= ident
    worst_dlp = max(worst_dlp, mx); worst_kl = max(worst_kl, mkl)
    print("%-12s %8d %6d/%-5d %11s %11.3e %11.3e %9.2e" % (
        name, n, tok_same, tok_tot, "YES" if ident else "no", mx, mean, mkl))

print()
if all_identical:
    print("VERDICT: BIT-IDENTICAL across all prompts (max|dlp|=0, max KL=0) -> zero drift.")
else:
    print("VERDICT: NOT bit-identical. worst max|dlp|=%.4e  worst KL=%.4e" % (worst_dlp, worst_kl))
    print("  (bf16 logprob quantum ~ 0.008 near lp=-1; drift below that is representational,")
    print("   but ANY nonzero drift means the coherence + BFCL gates are mandatory.)")
