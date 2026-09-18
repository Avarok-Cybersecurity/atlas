# SPDX-License-Identifier: AGPL-3.0-only

"""Train the decision head on cached features and score every arm.

WHAT IS BEING COMPARED
----------------------
Five predictors over the same test set:

  uniform   no-information floor.
  untuned   the backbone's own LM head restricted to the label tokens -- the
            zero-shot read, no training. NanoJev's "Untuned Qwen3-0.6B" row.
  ce        decision head trained with observed cross-entropy.
  brier     decision head trained with direct Brier.
  paired    decision head trained with the paired proper-reward estimator.
  oracle    the true distribution itself.

The oracle is not decoration. Outcome metrics have a floor set by the world's
entropy, and without the oracle printed beside them a Brier of 0.30 reads as
bad when it may be within a whisker of optimal. A model that BEATS the oracle
on outcome metrics has memorised sampled labels, which the train/test split
should prevent and which this makes visible if it does not.

WHAT TRAINING MAY SEE
---------------------
`outcome` only. `truth` lives in the same cache file for convenience but is
read exclusively by the evaluator. The whole claim -- that a proper scoring
rule recovers a distribution from samples of it -- is void if the
distribution was an input.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from evaluate import evaluate_by_kind, format_report, reference_points
from heads import DecisionHead, masked_log_softmax
from objectives import OBJECTIVES, compute_loss


def load(path: str, device: str):
    z = np.load(path, allow_pickle=True)
    d = {
        "H": torch.tensor(z["H"], device=device),
        "base_logits": torch.tensor(z["base_logits"], device=device),
        "label_idx": torch.tensor(z["label_idx"].astype(np.int64), device=device),
        "mask": torch.tensor(z["mask"], device=device),
        "outcome": torch.tensor(z["outcome"], device=device),
        "ordered": torch.tensor(z["ordered"], device=device),
        "emb": torch.tensor(z["emb"], device=device),
        "truth": z["truth"],
        "kind": z["kind"],
        "kinds": [str(x) for x in z["kinds"]],
    }
    return d


def rows_for_eval(cache, data_path):
    """The evaluator wants the original records; re-read them for `truth`."""
    return [json.loads(line) for line in open(data_path)]


def predict(head, cache, batch=1024):
    """Head predictions as a list of per-example distributions."""
    head.eval()
    out = []
    n = cache["H"].shape[0]
    with torch.no_grad():
        for s in range(0, n, batch):
            sl = slice(s, min(s + batch, n))
            emb = cache["emb"][cache["label_idx"][sl]]
            logits = head(cache["H"][sl], emb, cache["ordered"][sl])
            logp = masked_log_softmax(logits, cache["mask"][sl])
            p = logp.exp().cpu().numpy()
            m = cache["mask"][sl].cpu().numpy()
            for i in range(p.shape[0]):
                out.append([float(x) for x in p[i][m[i]]])
    return out


def untuned_predictions(cache):
    """Softmax of the LM head over the candidate ids: the zero-shot read."""
    logp = masked_log_softmax(cache["base_logits"], cache["mask"])
    p = logp.exp().cpu().numpy()
    m = cache["mask"].cpu().numpy()
    return [[float(x) for x in p[i][m[i]]] for i in range(p.shape[0])]


def train_head(cache, objective, d_model, epochs, lr, batch, m, seed, device, log_every=0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    head = DecisionHead(d_model).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = cache["H"].shape[0]

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        steps = 0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            emb = cache["emb"][cache["label_idx"][idx]]
            logits = head(cache["H"][idx], emb, cache["ordered"][idx])
            logp = masked_log_softmax(logits, cache["mask"][idx])
            loss = compute_loss(objective, logp, cache["outcome"][idx], cache["mask"][idx],
                                m=m, generator=gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            steps += 1
        if log_every and (ep + 1) % log_every == 0:
            print(f"    [{objective}] epoch {ep+1}/{epochs} loss {total/steps:.4f}", flush=True)
    return head


def separation_report(per_seed_sq: dict, seeds: list) -> str:
    """Is any gap between objectives bigger than the noise between seeds?

    The arms share seeds, so they can be compared PAIRWISE per seed, which
    removes the seed-to-seed variation common to all of them and is far more
    powerful than comparing two spreads. Reported as the mean paired
    difference and a sign count.

    This block exists because the medians here order the same way NanoJev's do
    while the per-seed spread is an order of magnitude larger than the gap --
    the exact shape of a result that looks like a finding and is not one. A
    ranking that changes between the median and the mean is noise, and saying
    so in the output is cheaper than remembering to check.
    """
    import statistics as st

    names = list(per_seed_sq)
    if len(names) < 2:
        return ""
    lines = ["=== is the objective ranking real? ==="]
    lines.append(f"{'arm':8} {'mean':>9} {'median':>9} {'std':>9} {'min':>9} {'max':>9}")
    for n in names:
        v = per_seed_sq[n]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        lines.append(f"{n:8} {st.mean(v):>9.5f} {st.median(v):>9.5f} {sd:>9.5f} {min(v):>9.5f} {max(v):>9.5f}")

    by_mean = sorted(names, key=lambda n: st.mean(per_seed_sq[n]))
    by_med = sorted(names, key=lambda n: st.median(per_seed_sq[n]))
    lines.append("")
    lines.append(f"order by mean  : {' < '.join(by_mean)}")
    lines.append(f"order by median: {' < '.join(by_med)}")
    if by_mean != by_med:
        lines.append("  -> mean and median DISAGREE: the ranking is not supported by this data.")

    lines.append("")
    lines.append("paired per-seed differences (negative = row better than column):")
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            d = [a_ - b_ for a_, b_ in zip(per_seed_sq[x], per_seed_sq[y])]
            wins = sum(1 for v in d if v < 0)
            sd = st.stdev(d) if len(d) > 1 else 0.0
            md = st.mean(d)
            # A difference smaller than the standard error of the difference
            # is not a difference.
            se = sd / (len(d) ** 0.5) if len(d) > 1 else float("inf")
            verdict = "separated" if abs(md) > 2 * se else "NOT separated"
            lines.append(
                f"  {x:7} vs {y:7}  mean diff {md:+.5f}  se {se:.5f}  "
                f"{x} better in {wins}/{len(d)} seeds  -> {verdict}"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--m", type=int, default=32, help="samples for the paired estimator")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--objectives", default="ce,brier,paired")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    tr = load(a.train_cache, a.device)
    te = load(a.test_cache, a.device)
    test_rows = rows_for_eval(te, a.test_data)
    d_model = tr["H"].shape[1]
    seeds = [int(s) for s in a.seeds.split(",")]

    print(f"train n={tr['H'].shape[0]}  test n={te['H'].shape[0]}  d={d_model}")
    print(f"objectives: {a.objectives}   seeds: {seeds}   epochs: {a.epochs}\n")

    refs = reference_points(test_rows)
    results = {}
    per_seed_sq = {}

    print(format_report(refs["uniform"], "uniform (no-information floor)"))
    print()
    untuned = evaluate_by_kind(untuned_predictions(te), test_rows)
    results["untuned"] = untuned
    print(format_report(untuned, "untuned read (LM head over label tokens, no training)"))
    print()

    for obj in a.objectives.split(","):
        per_seed = []
        for sd in seeds:
            head = train_head(tr, obj, d_model, a.epochs, a.lr, a.batch, a.m, sd, a.device)
            rep = evaluate_by_kind(predict(head, te), test_rows)
            per_seed.append(rep)
        # Median across seeds, so one lucky initialisation cannot carry an arm.
        med = {}
        for slice_name in per_seed[0]:
            med[slice_name] = {
                k: float(np.median([r[slice_name][k] for r in per_seed]))
                for k in per_seed[0][slice_name]
            }
        results[obj] = med
        per_seed_sq[obj] = [r["overall"]["sq_l2"] for r in per_seed]
        print(format_report(med, f"head trained with {obj} -- {OBJECTIVES[obj]} (median of {len(seeds)} seeds)"))
        print(f"per-seed overall sq-L2: {[round(x, 5) for x in per_seed_sq[obj]]}")
        print()

    results["oracle"] = refs["oracle"]
    print(format_report(refs["oracle"], "oracle (the true distribution -- the achievable floor)"))
    print()

    print("=== headline: overall distribution error vs the truth (lower is better) ===")
    print(f"{'arm':10} {'sq-L2':>9} {'MAE':>9} {'Brier':>8} {'ECE':>8} {'true-acc':>9}")
    order = ["untuned"] + a.objectives.split(",") + ["oracle"]
    base = results["untuned"]["overall"]["sq_l2"]
    for name in order:
        m_ = results[name]["overall"]
        rel = "" if name == "untuned" else f"  ({(m_['sq_l2']/base - 1)*100:+.1f}% vs untuned)"
        print(
            f"{name:10} {m_['sq_l2']:>9.5f} {m_['mean_abs_err']:>9.5f} {m_['brier']:>8.4f} "
            f"{m_['ece']:>8.4f} {m_['truth_acc']:>9.4f}{rel}"
        )

    print()
    print(separation_report(per_seed_sq, seeds))

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
