# SPDX-License-Identifier: AGPL-3.0-only

"""Scoring a decision model, separating what you can measure in the wild from
what only a simulator can tell you.

TWO CLASSES OF METRIC
---------------------
Everything here divides into metrics computed against the SAMPLED OUTCOME and
metrics computed against the TRUE DISTRIBUTION.

Outcome metrics (`nll`, `brier`, `ece`, `outcome_acc`) are what a real
deployment can compute: you observe what happened, not what was likely. They
are honest but noisy, and crucially they are *floored* by the world's own
entropy -- a perfectly calibrated model predicting 0.7 on a genuine 70/30
event still eats loss on every trial. So a better Brier is evidence of a
better model, but a Brier of 0 is not the target and chasing it means
overfitting the noise.

Truth metrics (`sq_l2`, `truth_acc`, `mean_abs_err`) compare the predicted
distribution to the real one. They are the direct measurement of calibration
and they are only available because the generator computes the answer in
closed form. `sq_l2` is the number NanoJev reports as "distribution error".

Reporting only the first class is how a miscalibrated model passes review: it
can look fine on accuracy while placing its probability mass anywhere.

WHY ECE IS REPORTED BUT NOT TRUSTED
-----------------------------------
Binned ECE is the conventional calibration number, so it is here for
comparability, but it is a biased estimator: it depends on bin count, it can
be driven to zero by a model that is wrong in compensating directions, and it
says nothing per-example. When `sq_l2` and ECE disagree, `sq_l2` is the one
that means something -- it is a proper scoring rule against the actual target.
"""

from __future__ import annotations

import math


def _safe_log(x: float) -> float:
    return math.log(max(x, 1e-12))


def evaluate(preds: list[list[float]], rows: list[dict], n_bins: int = 10) -> dict:
    """Score predictions against both the outcomes and the true distributions.

    `preds[i]` and `rows[i]["truth"]` must be distributions over the same
    label order.
    """
    assert len(preds) == len(rows), f"{len(preds)} preds vs {len(rows)} rows"

    n = len(preds)
    nll = brier = sq_l2 = mae = 0.0
    outcome_hits = truth_hits = 0
    # (confidence, correct) per example, for the ECE bins.
    conf_correct = []

    for p, r in zip(preds, rows):
        q, y = r["truth"], r["outcome"]
        assert len(p) == len(q), f"width mismatch: pred {len(p)} vs truth {len(q)}"

        nll += -_safe_log(p[y])
        brier += sum((p[k] - (1.0 if k == y else 0.0)) ** 2 for k in range(len(p)))
        sq_l2 += sum((p[k] - q[k]) ** 2 for k in range(len(p)))
        mae += sum(abs(p[k] - q[k]) for k in range(len(p))) / len(p)

        am = max(range(len(p)), key=lambda k: p[k])
        outcome_hits += int(am == y)
        truth_hits += int(am == max(range(len(q)), key=lambda k: q[k]))
        conf_correct.append((p[am], am == y))

    # Equal-width reliability bins over the top-label confidence.
    bins = [[] for _ in range(n_bins)]
    for c, ok in conf_correct:
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, ok))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(1 for _, ok in b if ok) / len(b)
        conf = sum(c for c, _ in b) / len(b)
        ece += (len(b) / n) * abs(acc - conf)

    return {
        "n": n,
        # --- measurable in deployment ---
        "nll": nll / n,
        "brier": brier / n,
        "ece": ece,
        "outcome_acc": outcome_hits / n,
        # --- requires the simulator ---
        "sq_l2": sq_l2 / n,
        "mean_abs_err": mae / n,
        "truth_acc": truth_hits / n,
    }


def evaluate_by_kind(preds: list[list[float]], rows: list[dict]) -> dict:
    """Overall plus a breakdown per question type.

    The three types have different label counts and different intrinsic
    entropy, so a single pooled number hides which primitive is failing --
    `choice` is near-uniform by construction and will always score worse on
    outcome metrics than `noul`.
    """
    out = {"overall": evaluate(preds, rows)}
    for kind in sorted({r["kind"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["kind"] == kind]
        out[kind] = evaluate([preds[i] for i in idx], [rows[i] for i in idx])
    return out


def format_report(report: dict, title: str) -> str:
    lines = [
        f"=== {title} ===",
        f"{'slice':8} {'n':>5} {'NLL':>7} {'Brier':>7} {'ECE':>7} "
        f"{'out-acc':>8} | {'sq-L2':>8} {'MAE':>7} {'true-acc':>8}",
    ]
    order = ["overall"] + [k for k in report if k != "overall"]
    for k in order:
        m = report[k]
        lines.append(
            f"{k:8} {m['n']:>5} {m['nll']:>7.4f} {m['brier']:>7.4f} {m['ece']:>7.4f} "
            f"{m['outcome_acc']:>8.4f} | {m['sq_l2']:>8.5f} {m['mean_abs_err']:>7.5f} "
            f"{m['truth_acc']:>8.4f}"
        )
    lines.append("left of the bar: measurable in deployment.  right: needs the simulator's truth.")
    return "\n".join(lines)


def reference_points(rows: list[dict]) -> dict:
    """Two reference predictors, because a metric without a scale is a number.

    `uniform` is the no-information floor. `oracle` predicts the true
    distribution exactly -- its outcome metrics are NOT zero, and that residual
    is the world's entropy, the part no model can remove. A model whose Brier
    approaches the oracle's is done; a model that beats it has memorised the
    sampled outcomes.
    """
    uni = [[1.0 / len(r["truth"])] * len(r["truth"]) for r in rows]
    orc = [list(r["truth"]) for r in rows]
    return {
        "uniform": evaluate_by_kind(uni, rows),
        "oracle": evaluate_by_kind(orc, rows),
    }
