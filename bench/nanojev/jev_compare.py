# SPDX-License-Identifier: AGPL-3.0-only

"""Jev against the local read and the oracle, on identical examples.

All arms are scored by the same `evaluate.py` against the same closed-form
truth, so the columns are directly comparable. The `wall` row is the control
that tells you whether a poor score anywhere else means "cannot see the grid"
or "cannot do the arithmetic" -- see `jev_client.py` for why that distinction
decides whether the rest of this table is a fair reading of Jev or not.
"""

from __future__ import annotations

import argparse
import json

from evaluate import evaluate_by_kind, format_report, reference_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--jev", required=True, help="jsonl from jev_client.py")
    ap.add_argument("--model", default="/home/ms/models/qwen3-0.6b")
    ap.add_argument("--skip-local", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    rows = [json.loads(x) for x in open(a.data)]
    jev = {}
    for line in open(a.jev):
        r = json.loads(line)
        jev[r["idx"]] = r["probs"]

    # Only examples Jev actually answered, so every arm sees one identical set.
    idx = sorted(jev)
    rows = [rows[i] for i in idx]
    jev_preds = [jev[i] for i in idx]
    print(f"examples scored: {len(rows)}\n")

    refs = reference_points(rows)
    print(format_report(refs["uniform"], "uniform (no-information floor)"))
    print()

    if not a.skip_local:
        from reader import Reader

        reader = Reader(a.model)
        local = []
        for s in range(0, len(rows), a.batch):
            local.extend(reader.baseline_probs(rows[s : s + a.batch]))
        local_rep = evaluate_by_kind(local, rows)
        print(format_report(local_rep, "untuned local read (Qwen3-0.6B LM head over label tokens)"))
        print()
    else:
        local_rep = None

    jev_rep = evaluate_by_kind(jev_preds, rows)
    print(format_report(jev_rep, "TypeSafe Jev (jev-latest, live API)"))
    print()
    print(format_report(refs["oracle"], "oracle (the achievable floor)"))
    print()

    print("=== distribution error against the truth, by question type (lower better) ===")
    kinds = ["overall"] + sorted({r["kind"] for r in rows})
    header = f"{'slice':8} {'uniform':>9} "
    if local_rep:
        header += f"{'local':>9} "
    header += f"{'JEV':>9} {'oracle':>8}"
    print(header)
    for k in kinds:
        line = f"{k:8} {refs['uniform'][k]['sq_l2']:>9.5f} "
        if local_rep:
            line += f"{local_rep[k]['sq_l2']:>9.5f} "
        line += f"{jev_rep[k]['sq_l2']:>9.5f} {refs['oracle'][k]['sq_l2']:>8.5f}"
        print(line)

    print()
    print("=== calibration (ECE) ===")
    for k in kinds:
        line = f"{k:8} {refs['uniform'][k]['ece']:>9.4f} "
        if local_rep:
            line += f"{local_rep[k]['ece']:>9.4f} "
        line += f"{jev_rep[k]['ece']:>9.4f} {refs['oracle'][k]['ece']:>8.4f}"
        print(line)

    print()
    w = jev_rep.get("wall")
    if w:
        print(f"CONTROL -- Jev on pure perception ('wall'): "
              f"true-acc {w['truth_acc']:.4f}  sq-L2 {w['sq_l2']:.5f}  ECE {w['ece']:.4f}")
        print("If this row is strong, Jev can read the observation, and a weak `noul`/`choice`/")
        print("`score` row is a failure of the probability computation rather than of perception.")
        print("Those three require deriving rho*wall + (1-rho)*mean(wall) from a noise model given")
        print("in prose -- deliberation, which is not what a System One model is sold to do.")


if __name__ == "__main__":
    main()
