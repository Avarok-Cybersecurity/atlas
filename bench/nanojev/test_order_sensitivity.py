# SPDX-License-Identifier: AGPL-3.0-only

"""Does the answer change when the options are listed in a different order?

WHY THIS MATTERS
----------------
A decision read is supposed to be a function of the state and the candidate
SET. The order the candidates happen to be written in is not part of the
question, so any dependence on it is pure error -- and unlike ordinary error
it is invisible to accuracy on a fixed dataset, because the dataset always
presents the options the same way.

Verdict, the ModernBERT open-Jev implementation, measures 3-4.5% of answers
flipping under option reordering, concentrated at low confidence. That is the
number to compare against, and it is a number nobody gets by accident: you
only find it if you test for it.

WHAT IS AND IS NOT EXPECTED TO BE INVARIANT
-------------------------------------------
Two parts of this pipeline have different guarantees:

  the head   scores each candidate from its own embedding, so
             score_k depends on (h, e_k) and NOT on k's position.
             Permutation-equivariant by construction.

  the prompt lists the options in order, so the hidden state h that the head
             consumes is itself a function of the order. Nothing about that is
             invariant.

So the head being order-blind does not make the system order-blind, and the
composition is what gets measured here. This is the experiment that tells you
which of the two dominates.

ORDERED LABELS ARE EXCLUDED
---------------------------
`score` questions have genuinely ordered labels (none < low < ... < certain).
Permuting them changes the question rather than its presentation, so they are
skipped. Including them would manufacture a large "flip rate" that means
nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics

from reader import Reader


def permute(ex: dict, perm: list[int]) -> dict:
    """Re-order an example's labels; truth follows so it stays aligned."""
    out = dict(ex)
    out["labels"] = [ex["labels"][i] for i in perm]
    out["truth"] = [ex["truth"][i] for i in perm]
    out["outcome"] = perm.index(ex["outcome"])
    return out


def unpermute(probs: list[float], perm: list[int]) -> list[float]:
    """Map a prediction made under `perm` back to canonical label order."""
    back = [0.0] * len(probs)
    for pos, orig in enumerate(perm):
        back[orig] = probs[pos]
    return back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="/home/ms/models/qwen3-0.6b")
    ap.add_argument("--n", type=int, default=400, help="examples to test")
    ap.add_argument("--perms", type=int, default=4, help="random permutations per example")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = [json.loads(x) for x in open(a.data)]
    # Ordered label sets are excluded: see the module docstring.
    rows = [r for r in rows if r["kind"] != "score"][: a.n]
    if not rows:
        raise SystemExit("no unordered examples in this file")

    reader = Reader(a.model, device=a.device)

    def predict(batch_rows):
        out = []
        for s in range(0, len(batch_rows), a.batch):
            out.extend(reader.baseline_probs(batch_rows[s : s + a.batch]))
        return out

    canonical = predict(rows)

    flips = 0
    total = 0
    tvs = []
    flip_conf, keep_conf = [], []
    per_kind: dict[str, list[int]] = {}
    # Probability mass landing on each SLOT, regardless of which label sits
    # there. If the read were content-driven this would follow the labels and
    # average out across random permutations; if it is position-driven it will
    # not. This is the direct measurement, not an inference from flip rate.
    slot_mass: dict[tuple, list[float]] = {}

    for p_i in range(a.perms):
        permuted_rows, perms = [], []
        for r in rows:
            k = len(r["labels"])
            perm = list(range(k))
            # A permutation that changes nothing tests nothing.
            while True:
                rng.shuffle(perm)
                if perm != list(range(k)) or k == 1:
                    break
            perms.append(perm)
            permuted_rows.append(permute(r, perm))

        got = predict(permuted_rows)
        for r, base, g, perm in zip(rows, canonical, got, perms):
            back = unpermute(g, perm)
            am_b = max(range(len(base)), key=lambda i: base[i])
            am_g = max(range(len(back)), key=lambda i: back[i])
            total += 1
            for slot, pv in enumerate(g):
                slot_mass.setdefault((r["kind"], slot), []).append(pv)
            kk = per_kind.setdefault(r["kind"], [0, 0])
            kk[1] += 1
            if am_b != am_g:
                kk[0] += 1
            # Total variation distance: half the L1 between the two
            # distributions, so it reads as "probability mass moved".
            tv = 0.5 * sum(abs(base[i] - back[i]) for i in range(len(base)))
            tvs.append(tv)
            if am_b != am_g:
                flips += 1
                flip_conf.append(max(base))
            else:
                keep_conf.append(max(base))

    print(f"model      : {a.model}")
    print(f"examples   : {len(rows)} (ordered 'score' questions excluded)")
    print(f"permutations per example: {a.perms}   comparisons: {total}\n")

    print(f"argmax flip rate      : {flips/total*100:.2f}%  ({flips}/{total})")
    print(f"mean TV distance      : {statistics.mean(tvs):.4f}")
    print(f"median TV distance    : {statistics.median(tvs):.4f}")
    print(f"max TV distance       : {max(tvs):.4f}")
    if flip_conf:
        print(f"\nmean confidence when the answer FLIPPED : {statistics.mean(flip_conf):.4f}")
    if keep_conf:
        print(f"mean confidence when it HELD            : {statistics.mean(keep_conf):.4f}")
    if flip_conf and keep_conf:
        if statistics.mean(flip_conf) < statistics.mean(keep_conf):
            print("  -> flips concentrate at LOW confidence, as Verdict reports. A confidence")
            print("     threshold would suppress most of them, which is the usable mitigation.")
        else:
            print("  -> flips are NOT confined to low confidence: thresholding will not save this,")
            print("     and the order dependence is reaching confident answers.")

    # A read that ignored content entirely and answered at random would still
    # flip (K-1)/K of the time. Quoting a flip rate without that baseline makes
    # a 4-option task look worse than a 2-option one for purely arithmetic
    # reasons.
    k_of = {}
    for r in rows:
        k_of[r["kind"]] = len(r["labels"])
    print("\nper question type (chance = what a content-blind random answer gives):")
    for k in sorted(per_kind):
        f, t = per_kind[k]
        kk = k_of[k]
        chance = (kk - 1) / kk * 100
        verdict = "WORSE than chance" if f / t * 100 > chance else "better than chance"
        print(f"  {k:7} K={kk}  flip {f/t*100:5.2f}%  (chance {chance:.1f}%)  {verdict}")

    print("\nprobability mass by SLOT, averaged over random permutations:")
    print("  (per kind: K=2 and K=4 have different uniform baselines, so pooling them lies)")
    for kind in sorted({kk for kk, _ in slot_mass}):
        kk = k_of[kind]
        uni = 1.0 / kk
        parts = []
        for slot in range(kk):
            vals = slot_mass[(kind, slot)]
            parts.append(f"slot{slot} {statistics.mean(vals):.3f}")
        print(f"  {kind:7} K={kk} uniform={uni:.3f} : " + "  ".join(parts))
    print("  Under random permutations a content-driven read would sit at the uniform")
    print("  value in every slot. A skew IS the positional preference, measured")
    print("  directly rather than inferred from the flip rate.")

    print("\nreference: Verdict (ModernBERT 151M) reports 3-4.5% flips under reordering.")
    print("A read whose distribution moves with option order is reporting presentation,")
    print("not belief -- and no amount of calibration fixes that, because the model is")
    print("calibrated to the wrong thing.")


if __name__ == "__main__":
    main()
