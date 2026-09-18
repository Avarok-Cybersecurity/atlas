# SPDX-License-Identifier: AGPL-3.0-only

"""The three training objectives, and why a proper scoring rule is the point.

THE PROBLEM
-----------
Training sees one sampled outcome per example, not the distribution that
produced it. A loss that merely wants the right ARGMAX (accuracy, hinge) is
free to output 1.0 on a genuine 70/30 event -- it scores better for doing so.
That is exactly the miscalibration a decision model must not have, and it is
why "train with cross-entropy and read the softmax as a probability" is a
folk method rather than a guarantee.

A PROPER scoring rule fixes this: it is uniquely minimised, in expectation
over outcomes, by reporting the TRUE distribution. Both cross-entropy
(logarithmic rule) and Brier (quadratic rule) are proper, so with enough
samples per context either recovers the truth. They differ in how they get
there and in how they behave under noise, which is the comparison this file
exists to run.

  observed_ce    -log p[Y].  Proper. Unbounded: a confident miss costs
                 infinitely much, so it punishes overconfidence hard and can
                 destabilise on noisy labels.

  direct_brier   sum_k (p_k - 1[Y=k])^2.  Proper, bounded in [0, 2]. Gentler
                 on outliers, which on sampled outcomes is usually a virtue.

  paired_reward  A REINFORCE estimator of the same quadratic objective, using
                 M sampled actions instead of the closed-form gradient. This
                 is the RLCD-shaped arm: it is what you must use when the
                 reward is only observable by ACTING, and it should match
                 direct Brier when both are available.

THE PAIRED ESTIMATOR
--------------------
From NanoJev's write-up, for M independent draws A_1..A_M from p, with c_k the
count of draws landing on k:

    R = (2/M) * sum_i 1[A_i = Y]  -  sum_k c_k(c_k - 1) / (M(M-1))

The first term rewards agreeing with the outcome. The second is an unbiased
estimator of ||p||^2 built from PAIRS of distinct draws -- hence "paired" --
and it penalises confidence. Together E[R|x] = 2 p.q - ||p||^2, whose maximum
over p is q: the same optimum as Brier, up to a constant.

Implemented as a policy gradient, R multiplies the log-probabilities of the
drawn actions, with a leave-one-out baseline to cut variance. The baseline
must not depend on the action whose log-prob it scales, or it biases the
estimator -- a mistake that still trains, just to the wrong place.
"""

from __future__ import annotations

import torch


def observed_ce(logp: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """-log p[Y], the logarithmic proper scoring rule."""
    return -logp.gather(1, outcome.unsqueeze(1)).squeeze(1).mean()


def direct_brier(logp: torch.Tensor, outcome: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum_k (p_k - 1[Y=k])^2, the quadratic proper scoring rule."""
    p = logp.exp()
    onehot = torch.zeros_like(p).scatter_(1, outcome.unsqueeze(1), 1.0)
    return (((p - onehot) ** 2) * mask).sum(dim=1).mean()


def paired_reward(
    logp: torch.Tensor,
    outcome: torch.Tensor,
    mask: torch.Tensor,
    m: int = 32,
    generator: torch.Generator | None = None,
    exact_pair_term: bool = True,
) -> torch.Tensor:
    """Policy-gradient surrogate whose optimum is the true distribution.

    Maximises E[R|x] = 2 p.q - ||p||^2, whose unique maximiser over the simplex
    is p = q. Returned as a LOSS (negated) so every arm minimises.

    The two terms are estimated differently on purpose:

      2 p.q   Only reachable by sampling, because q is never observed -- one
              outcome Y is. REINFORCE gives it: E_{A~p}[2*1[A=Y] dlog p(A)]
              = 2 dp_Y, which in expectation over Y~q is the gradient of
              2 p.q. A leave-one-out baseline (the mean hit over the OTHER
              draws) cuts variance; it must exclude draw i itself, or it
              correlates with the action and biases the gradient -- which
              still trains, just to the wrong place.

      ||p||^2 Computable in closed form here, since p is right there. The
              sampled alternative is the pair statistic
              sum_k c_k(c_k-1)/(M(M-1)), which is what makes the rule
              "paired"; it is unbiased but adds variance for nothing when the
              exact value is available. `exact_pair_term=False` selects it, for
              the setting where only actions are observable.
    """
    p = logp.exp()
    b, k = p.shape

    actions = torch.multinomial(p.clamp_min(1e-12), m, replacement=True, generator=generator)
    hit = (actions == outcome.unsqueeze(1)).float()                     # [B, M]

    if m > 1:
        loo = (hit.sum(dim=1, keepdim=True) - hit) / (m - 1)            # [B, M]
    else:
        loo = torch.zeros_like(hit)
    advantage = (2.0 * hit - 2.0 * loo).detach()                        # [B, M]
    surrogate = (advantage * logp.gather(1, actions)).mean(dim=1)       # [B]

    if exact_pair_term:
        penalty = (p * p * mask).sum(dim=1)                             # [B]
    else:
        counts = torch.zeros_like(p).scatter_add_(
            1, actions, torch.ones_like(actions, dtype=p.dtype)
        )
        pair_stat = (counts * (counts - 1.0)).sum(dim=1) / (m * (m - 1))
        # Sampled value, routed through log-probs so it carries a gradient.
        penalty = (pair_stat.detach().unsqueeze(1) * logp.gather(1, actions)).mean(dim=1)

    return -(surrogate - penalty).mean()


OBJECTIVES = {
    "ce": "observed cross-entropy (logarithmic rule)",
    "brier": "direct Brier (quadratic rule)",
    "paired": "paired proper-reward policy gradient (RLCD-shaped)",
}


def compute_loss(name: str, logp, outcome, mask, m: int = 32, generator=None):
    if name == "ce":
        return observed_ce(logp, outcome)
    if name == "brier":
        return direct_brier(logp, outcome, mask)
    if name == "paired":
        return paired_reward(logp, outcome, mask, m=m, generator=generator)
    raise ValueError(f"unknown objective {name!r}; known: {sorted(OBJECTIVES)}")
