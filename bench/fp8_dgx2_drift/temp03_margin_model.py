#!/usr/bin/env python3
"""Greedy-vs-temp divergence amplification, parameterized by logit margin.

Core question for the ANGLE: at temp 0.3, how much MORE do Atlas and vLLM
diverge per token than at greedy, and is that the dominant gap driver?

Two-token model: at a decision the relevant competition is top-1 vs top-2 with
logit gap g (vLLM frame). FP8 shifts the gap by delta ~ N(0, s) where s is the
observed per-logit perturbation. We use the MEASURED per-logit diff stats from
the L39 dump (mean|diff|=0.159, and the top-2 gap delta at the canonical pos).

GREEDY divergence at gap g:  the engines pick different tokens iff FP8 flips the
sign of the gap => P_flip(g) = P(delta > g) for a one-sided shift, ~ erfc.
TEMP-T divergence at gap g: even with the SAME argmax, the two softmax-over-{1,2}
distributions differ; collision-divergence
  P_div(g,T) = 1 - [pv1*pa1 + pv2*pa2]
where pv = softmax([0,-g]/T), pa = softmax([0,-(g+delta)]/T).

We integrate both over the EMPIRICAL margin distribution from C1
(23.7% of long-ctx positions have gap<1.5 logprobs; we model the gap CDF from
the reported buckets) to get expected per-token divergence, then compound over
a realistic agentic generation length.
"""
from __future__ import annotations
import numpy as np

# --- measured FP8 perturbation on the top-2 logit GAP ---------------------
# From temp03_sampling_amp.py at the canonical position: vLLM gap=0.2822;
# Atlas's gap at the same position (recompute below from raw dumps would be
# ideal, but we use the measured per-logit diff distribution as the noise model).
# mean|per-logit diff| = 0.1586; the gap is a difference of two logits, so its
# perturbation std ~ sqrt(2)*sigma_logit. Estimate sigma_logit from mean|diff|:
# for a zero-mean Gaussian, E|x| = sigma*sqrt(2/pi) => sigma = E|x|*sqrt(pi/2).
mean_abs_logit_diff = 0.1586
sigma_logit = mean_abs_logit_diff * np.sqrt(np.pi/2)   # ~0.1988
sigma_gap = sigma_logit * np.sqrt(2)                    # perturbation on a gap ~0.281
print(f"sigma_logit ~ {sigma_logit:.4f}, sigma_gap ~ {sigma_gap:.4f} (logits)")

def softmax2(g, T):
    # logits [0, -g] (top-1 ref 0, runner-up at -g), temperature T
    x = np.array([0.0, -g])/T
    x -= x.max()
    e = np.exp(x); return e/e.sum()

def p_greedy_flip(g):
    # argmax flips iff FP8 noise pushes runner-up above top-1: delta_gap < -g
    # delta_gap ~ N(0, sigma_gap). P(N < -g) = 0.5*erfc(g/(sqrt2*sigma_gap))
    from math import erfc, sqrt
    return 0.5*erfc(g/(np.sqrt(2)*sigma_gap))

def p_temp_div(g, T, n_mc=4000, rng=None):
    # Monte-Carlo over the FP8 gap perturbation; for each, compute collision-
    # divergence between the vLLM(g) and Atlas(g+delta) 2-way softmaxes.
    if rng is None: rng = np.random.default_rng(0)
    deltas = rng.normal(0, sigma_gap, n_mc)
    pv = softmax2(g, T)
    accs = []
    for d in deltas:
        pa = softmax2(g + d, T)
        # collision prob that two independent draws agree:
        coll = pv[0]*pa[0] + pv[1]*pa[1]
        accs.append(1.0 - coll)
    return float(np.mean(accs))

# --- empirical margin distribution from C1 --------------------------------
# C1: 23.7% of long-ctx positions gap<1.5 logprobs. Short prompts: ~0% (gaps 10-18).
# Model the long-context gap distribution: a mass of low-margin positions plus a
# high-margin bulk. Use a mixture: 23.7% Uniform(0,1.5) + 76.3% with large gaps.
# For the high-margin bulk both flip and temp-div are ~0, so they don't matter;
# we focus the integral on the low-margin sub-population and report per-regime.
def expected_div_lowmargin(T):
    # integrate over Uniform(0,1.5) gap for the low-margin 23.7%
    gs = np.linspace(0.02, 1.5, 40)
    rng = np.random.default_rng(1)
    flip = np.mean([p_greedy_flip(g) for g in gs])
    tdiv = np.mean([p_temp_div(g, T, 2000, rng) for g in gs])
    return flip, tdiv

print("\n=== per-position divergence by gap (greedy flip vs temp-T collision) ===")
print(f"{'gap':>5} {'greedy_flip':>12} {'div@T1.0':>10} {'div@T0.3':>10} {'div@T0.1':>10}")
rng = np.random.default_rng(0)
for g in (0.1, 0.28, 0.5, 1.0, 1.5, 3.0, 5.0):
    print(f"{g:5.2f} {p_greedy_flip(g):12.4f} "
          f"{p_temp_div(g,1.0,2000,rng):10.4f} "
          f"{p_temp_div(g,0.3,2000,rng):10.4f} "
          f"{p_temp_div(g,0.1,2000,rng):10.4f}")

print("\n=== expected per-token divergence over LOW-MARGIN sub-population ===")
for T in (1.0, 0.3, 0.1):
    flip, tdiv = expected_div_lowmargin(T)
    print(f"  T={T}: greedy_flip={flip:.4f}  temp_div={tdiv:.4f}  "
          f"amplification(temp/greedy)={tdiv/max(flip,1e-9):.2f}x")

# --- population-level: 23.7% low-margin, rest negligible -------------------
print("\n=== population expected per-token divergence (long-context) ===")
frac_low = 0.237
for T in (1.0, 0.3, 0.1):
    flip, tdiv = expected_div_lowmargin(T)
    pop_greedy = frac_low*flip
    pop_temp = frac_low*tdiv
    print(f"  T={T}: E[greedy div/token]={pop_greedy:.4f}  E[temp div/token]={pop_temp:.4f}")
    for N in (50, 200, 500):
        pg = 1-(1-pop_greedy)**N
        pt = 1-(1-pop_temp)**N
        print(f"      N={N:4d}: P(>=1 greedy-divergence)={pg:.3f}  P(>=1 temp-divergence)={pt:.3f}")

# --- short-prompt (cargo-like, focused) regime: gaps 10-18, ~0 low-margin --
print("\n=== short/focused-prompt regime (cargo) gaps>=10 ===")
for T in (0.3,):
    for g in (10.0, 14.0, 18.0):
        print(f"  gap={g}: greedy_flip={p_greedy_flip(g):.2e}  temp_div@T0.3={p_temp_div(g,T,2000,rng):.2e}")
