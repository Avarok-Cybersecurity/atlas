#!/usr/bin/env python3
"""Decompose temp-0.3 divergence into INTRINSIC sampling noise vs FP8-EXCESS.

The right metric is NOT "do two draws differ" (that's just sampling), but
"does the FP8 distribution shift make the two engines diverge MORE than two
draws from the SAME engine would?" That excess is what FP8 + temp contributes
beyond what vLLM-vs-vLLM at the same temp already has.

For a 2-token decision with gap g (vLLM) and g+delta (atlas):
  intrinsic  = 1 - sum pv_i^2          (vLLM self-collision complement)
  cross      = 1 - sum pv_i * pa_i     (vLLM-vs-atlas)
  excess     = cross - intrinsic       (FP8-attributable extra divergence)

We also report the TVD between pv and pa (pure distribution shift, no sampling
stochasticity) as the cleanest "FP8 moved the distribution" measure, and how it
scales with T. This directly tests the angle: does lowering effective entropy
(temp<1) make the SAME FP8 logit shift produce a LARGER fractional distribution
move?
"""
from __future__ import annotations
import numpy as np

mean_abs_logit_diff = 0.1586
sigma_logit = mean_abs_logit_diff*np.sqrt(np.pi/2)
sigma_gap = sigma_logit*np.sqrt(2)

def softmax2(g,T):
    x=np.array([0.0,-g])/T; x-=x.max(); e=np.exp(x); return e/e.sum()

def metrics(g,T,n_mc=4000,rng=None):
    if rng is None: rng=np.random.default_rng(0)
    pv=softmax2(g,T)
    intrinsic=1-np.sum(pv*pv)
    deltas=rng.normal(0,sigma_gap,n_mc)
    cross=[]; tvd=[]
    for d in deltas:
        pa=softmax2(g+d,T)
        cross.append(1-np.sum(pv*pa))
        tvd.append(0.5*np.abs(pv-pa).sum())
    cross=np.mean(cross); tvd=np.mean(tvd)
    return intrinsic, cross, cross-intrinsic, tvd

print(f"sigma_gap={sigma_gap:.4f} logits\n")
print(f"{'gap':>5} {'T':>5} {'intrinsic':>10} {'cross':>8} {'excess':>8} {'TVD':>8}")
rng=np.random.default_rng(0)
for g in (0.1,0.28,0.5,1.0):
    for T in (1.0,0.6,0.3,0.1):
        i,c,e,t=metrics(g,T,3000,rng)
        print(f"{g:5.2f} {T:5.2f} {i:10.4f} {c:8.4f} {e:8.4f} {t:8.4f}")
    print()

# TVD vs T at fixed gap shows whether lower T amplifies the *distribution shift*
print("=== TVD (pure distribution move, no sampling noise) vs T, gap=0.28 ===")
for T in (1.5,1.0,0.6,0.3,0.2,0.1):
    _,_,_,t=metrics(0.28,T,4000,rng)
    print(f"  T={T:4.2f}: TVD={t:.4f}")

# Population-weighted EXCESS divergence (the FP8-attributable part)
print("\n=== population EXCESS divergence (long-ctx, 23.7% low-margin U(0,1.5)) ===")
gs=np.linspace(0.02,1.5,30)
for T in (1.0,0.3,0.1):
    exc=np.mean([metrics(g,T,1500,rng)[2] for g in gs])
    pop=0.237*exc
    print(f"  T={T}: mean excess(low-margin)={exc:.4f}  pop E[excess/token]={pop:.4f}  "
          f"P(>=1 excess-div in 200 tok)={1-(1-pop)**200:.3f}")
