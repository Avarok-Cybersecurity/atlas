# SPDX-License-Identifier: AGPL-3.0-only

"""Positive control: do the objectives actually recover the true distribution?

WHY THIS TEST EXISTS FIRST
--------------------------
A wrong policy-gradient estimator does not crash and does not look wrong. It
trains, the loss falls, and the model converges -- to the wrong distribution.
Once an LLM backbone is in the loop there is no way to tell that apart from
"the task is hard", because there is no reference to compare against.

So before any of this touches a model, each objective is pointed at a problem
whose answer is known exactly: a free logit vector, sampled outcomes drawn
from a fixed q, and nothing else to learn. A proper scoring rule must drive
the logits to q and stay there. If `paired` does not land where `brier` lands,
the estimator is broken, not the model.

This mirrors the CPU benchmark in NanoJev's RLCD write-up (small model,
synthetic data, K = 2/3/5, M = 32, a few hundred Adam steps, several seeds),
which serves the same purpose there.

Run: python test_objectives.py
"""

from __future__ import annotations

import torch

from objectives import compute_loss


def recover(q: list[float], objective: str, steps: int = 4000, lr: float = 0.05,
            batch: int = 512, seed: int = 0, m: int = 32) -> torch.Tensor:
    """Fit a free logit vector to sampled outcomes from `q`."""
    g = torch.Generator().manual_seed(seed)
    qt = torch.tensor(q)
    k = len(q)

    logits = torch.zeros(k, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    mask = torch.ones(batch, k, dtype=torch.bool)

    for _ in range(steps):
        # Fresh outcomes every step: the model must learn the DISTRIBUTION,
        # not a fixed sample of it.
        outcome = torch.multinomial(qt.expand(batch, k), 1, replacement=True, generator=g).squeeze(1)
        logp = torch.log_softmax(logits.unsqueeze(0).expand(batch, k), dim=-1)
        loss = compute_loss(objective, logp, outcome, mask, m=m, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return torch.softmax(logits.detach(), dim=-1)


def main():
    cases = {
        "K=2 balanced": [0.7, 0.3],
        "K=2 extreme": [0.95, 0.05],
        "K=3": [0.5, 0.3, 0.2],
        "K=5 flat-ish": [0.3, 0.25, 0.2, 0.15, 0.1],
    }
    objectives = ["ce", "brier", "paired"]

    print(f"{'case':16} {'objective':9} {'recovered':38} {'max |err|':>10} {'sq-L2':>9}")
    print("-" * 88)
    worst = {}
    for name, q in cases.items():
        for obj in objectives:
            p = recover(q, obj)
            err = max(abs(float(p[i]) - q[i]) for i in range(len(q)))
            l2 = sum((float(p[i]) - q[i]) ** 2 for i in range(len(q)))
            shown = "[" + ", ".join(f"{float(x):.3f}" for x in p) + "]"
            print(f"{name:16} {obj:9} {shown:38} {err:>10.4f} {l2:>9.6f}")
            worst[obj] = max(worst.get(obj, 0.0), err)
        print()

    print("worst max-|err| across all cases:")
    ok = True
    for obj in objectives:
        # 0.03 is loose enough for a sampled estimator at this step count and
        # tight enough that a sign error or a biased baseline cannot pass.
        verdict = "PASS" if worst[obj] < 0.03 else "FAIL"
        if worst[obj] >= 0.03:
            ok = False
        print(f"  {obj:9} {worst[obj]:.4f}  {verdict}")
    print("\nAll three are proper scoring rules, so all three must land on q.")
    print("If `paired` alone misses, the policy-gradient estimator is wrong -- not the task.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
