# SPDX-License-Identifier: AGPL-3.0-only

"""Train the invariant head, then calibrate it, then check both claims.

This puts together the four things the other implementations do that the first
version of this testbed did not:

  1. PER-CANDIDATE ENCODING (NanoJev). Each candidate is its own sequence, so
     the head is permutation-invariant by construction rather than by
     augmentation. `--check-invariance` verifies it exactly, by permuting the
     cached candidate axis -- which is what permuting the options IS under this
     encoding.

  2. COMPOSITE CE + BRIER LOSS (kotobalabs, Verdict). Both are proper scoring
     rules and the 8-seed comparison in `train.py` could not separate them, so
     summing them is a reasonable response to a genuine tie: the log rule
     punishes confident misses hard, the quadratic rule is bounded and steadier
     on noisy labels, and neither dominates.

  3. OPTION SHUFFLING during training. Free here -- it is a permutation of the
     cached K axis. Under an architecture that is already invariant it should
     change nothing, and `--shuffle` exists so that "should" can be tested
     rather than assumed.

  4. POST-HOC TEMPERATURE SCALING on a held-out calibration split. One scalar,
     fitted by minimising NLL, applied to the logits. It cannot change any
     argmax, so accuracy is untouched by construction and only calibration
     moves -- which is exactly why it is safe to apply and easy to check.

The calibration split is separate from both train and test. Fitting a
temperature on the test set would improve the number being reported by tuning
on it, which is the quiet version of training on test.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from evaluate import evaluate_by_kind, format_report, reference_points
from heads import DecisionHeadV2, masked_log_softmax


def load(path, device):
    z = np.load(path, allow_pickle=True)
    return {
        "H": torch.tensor(z["H"], device=device),
        "mask": torch.tensor(z["mask"], device=device),
        "outcome": torch.tensor(z["outcome"], device=device),
        "truth": z["truth"],
    }


def composite_loss(logp, outcome, mask, brier_weight: float = 1.0):
    """Logarithmic + quadratic proper scoring rules, summed."""
    ce = -logp.gather(1, outcome.unsqueeze(1)).squeeze(1)
    p = logp.exp()
    onehot = torch.zeros_like(p).scatter_(1, outcome.unsqueeze(1), 1.0)
    brier = (((p - onehot) ** 2) * mask).sum(dim=1)
    return (ce + brier_weight * brier).mean()


def shuffled(H, mask, outcome, gen):
    """Permute each row's candidate axis, carrying the outcome index with it.

    Padded slots must stay padded, so each row is permuted only within its own
    valid width -- a global permutation would move real candidates into pad
    positions and quietly corrupt the batch.
    """
    b, k, _ = H.shape
    idx = torch.arange(k, device=H.device).unsqueeze(0).repeat(b, 1)
    widths = mask.sum(1)
    for w in widths.unique():
        w = int(w)
        rows = (widths == w).nonzero(as_tuple=True)[0]
        if len(rows) == 0:
            continue
        perm = torch.argsort(torch.rand(len(rows), w, device=H.device, generator=gen), dim=1)
        idx[rows, :w] = perm
    Hs = torch.gather(H, 1, idx.unsqueeze(-1).expand(-1, -1, H.shape[-1]))
    ms = torch.gather(mask, 1, idx)
    # Where did the outcome land? invert the permutation.
    pos = torch.argsort(idx, dim=1)
    oc = pos.gather(1, outcome.unsqueeze(1)).squeeze(1)
    return Hs, ms, oc


def predict(head, cache, temperature=1.0, batch=2048):
    head.eval()
    out = []
    n = cache["H"].shape[0]
    with torch.no_grad():
        for s in range(0, n, batch):
            sl = slice(s, min(s + batch, n))
            logits = head(cache["H"][sl], cache["mask"][sl]) / temperature
            p = masked_log_softmax(logits, cache["mask"][sl]).exp().cpu().numpy()
            m = cache["mask"][sl].cpu().numpy()
            for i in range(p.shape[0]):
                out.append([float(x) for x in p[i][m[i]]])
    return out


def fit_temperature(head, cache, lo=0.25, hi=8.0, steps=60):
    """One scalar, chosen to minimise NLL on the calibration split.

    A grid search rather than gradient descent: the objective is
    one-dimensional and smooth, the range is known, and this cannot fail to
    converge or overshoot.
    """
    head.eval()
    with torch.no_grad():
        logits = head(cache["H"], cache["mask"])
    best, best_nll = 1.0, float("inf")
    for t in np.linspace(lo, hi, steps):
        lp = masked_log_softmax(logits / float(t), cache["mask"])
        nll = -lp.gather(1, cache["outcome"].unsqueeze(1)).mean().item()
        if nll < best_nll:
            best, best_nll = float(t), nll
    return best, best_nll


def check_invariance(head, cache, gen, trials=4):
    """Permute the candidate axis and see whether anything moves.

    Under per-candidate encoding this is not an approximation of reordering the
    options -- it IS reordering them, because a candidate's vector was computed
    without reference to any other. So a non-zero result here is a real defect,
    not measurement noise.
    """
    head.eval()
    base = predict(head, cache)
    flips = 0
    total = 0
    max_tv = 0.0
    with torch.no_grad():
        for _ in range(trials):
            Hs, ms, _ = shuffled(cache["H"], cache["mask"], cache["outcome"], gen)
            # Recover the permutation by matching rows is unnecessary: compare
            # SORTED distributions, which are permutation-invariant summaries,
            # plus the max probability which must be identical.
            logits = head(Hs, ms)
            p = masked_log_softmax(logits, ms).exp().cpu().numpy()
            m = ms.cpu().numpy()
            for i in range(p.shape[0]):
                got = sorted(float(x) for x in p[i][m[i]])
                want = sorted(base[i])
                tv = 0.5 * sum(abs(a - b) for a, b in zip(got, want))
                max_tv = max(max_tv, tv)
                total += 1
                if tv > 1e-5:
                    flips += 1
    return {"rows": total, "moved": flips, "max_tv": max_tv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cal-frac", type=float, default=0.15)
    ap.add_argument("--brier-weight", type=float, default=1.0)
    ap.add_argument("--shuffle", action="store_true", help="augment with option shuffling")
    ap.add_argument("--no-set-head", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    tr_all = load(a.train_cache, a.device)
    te = load(a.test_cache, a.device)
    test_rows = [json.loads(x) for x in open(a.test_data)]
    d_model = tr_all["H"].shape[2]

    results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        torch.manual_seed(seed)
        gen = torch.Generator(device=a.device).manual_seed(seed)

        n = tr_all["H"].shape[0]
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(a.device)
        n_cal = int(n * a.cal_frac)
        cal_i, tr_i = order[:n_cal], order[n_cal:]
        cal = {k: (v[cal_i] if torch.is_tensor(v) else v) for k, v in tr_all.items()}
        tr = {k: (v[tr_i] if torch.is_tensor(v) else v) for k, v in tr_all.items()}

        head = DecisionHeadV2(d_model, set_head=not a.no_set_head).to(a.device)
        opt = torch.optim.Adam(head.parameters(), lr=a.lr)
        m = tr["H"].shape[0]
        for _ in range(a.epochs):
            head.train()
            perm = torch.randperm(m, device=a.device)
            for s in range(0, m, a.batch):
                idx = perm[s : s + a.batch]
                H, mk, oc = tr["H"][idx], tr["mask"][idx], tr["outcome"][idx]
                if a.shuffle:
                    H, mk, oc = shuffled(H, mk, oc, gen)
                logp = masked_log_softmax(head(H, mk), mk)
                loss = composite_loss(logp, oc, mk, a.brier_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()

        t, cal_nll = fit_temperature(head, cal)
        pre = evaluate_by_kind(predict(head, te, 1.0), test_rows)
        post = evaluate_by_kind(predict(head, te, t), test_rows)
        inv = check_invariance(head, te, gen)
        results.append({"seed": seed, "T": t, "pre": pre, "post": post, "inv": inv})
        print(f"seed {seed}: T={t:.3f} (cal NLL {cal_nll:.4f})  "
              f"sq-L2 {post['overall']['sq_l2']:.5f}  ECE {pre['overall']['ece']:.4f} -> "
              f"{post['overall']['ece']:.4f}  invariance moved {inv['moved']}/{inv['rows']} "
              f"(max TV {inv['max_tv']:.2e})", flush=True)

    print()
    print(format_report(results[0]["post"], f"v2 head, temperature-scaled (seed {results[0]['seed']})"))
    print()
    print(format_report(reference_points(test_rows)["oracle"], "oracle"))

    print("\n=== across seeds ===")
    print(f"{'seed':>5} {'T':>7} {'sq-L2':>9} {'ECE pre':>9} {'ECE post':>9} {'moved':>12}")
    for r in results:
        print(f"{r['seed']:>5} {r['T']:>7.3f} {r['post']['overall']['sq_l2']:>9.5f} "
              f"{r['pre']['overall']['ece']:>9.4f} {r['post']['overall']['ece']:>9.4f} "
              f"{r['inv']['moved']:>6}/{r['inv']['rows']:<5}")
    sq = [r["post"]["overall"]["sq_l2"] for r in results]
    print(f"\nsq-L2 median {np.median(sq):.5f}  min {min(sq):.5f}  max {max(sq):.5f}")
    moved = sum(r["inv"]["moved"] for r in results)
    print(f"option-order sensitivity: {moved} rows moved out of "
          f"{sum(r['inv']['rows'] for r in results)} -- v1's untuned read flipped 78% of answers.")


if __name__ == "__main__":
    main()
