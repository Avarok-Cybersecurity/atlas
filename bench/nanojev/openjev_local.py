# SPDX-License-Identifier: AGPL-3.0-only

"""Run the open Apache-2.0 decision encoder on the simulator.

`com-kotobalabs/open-jev-deberta-v3-large` is a DeBERTa-v3-large encoder with a
typed-decision head, trained on banking77 / sst5 / BoolQ with option shuffling
and a cross-entropy + Brier composite loss, and temperature-scaled afterwards.
It is the only open checkpoint in this family that is BOTH permissively
licensed and calibrated -- the MIT alternative's own card says to treat its
probabilities as rankings rather than confidences, which disqualifies it for
anything that thresholds on confidence.

Two reasons it is worth measuring here beyond the licence:

  * It is an ENCODER. Bidirectional attention has no notion of "the option
    listed first", so it should not carry the causal-LM position bias that
    made our untuned read flip 78% of answers under option reordering.
  * It trains with option shuffling deliberately, which is the training-time
    fix for that same failure.

Whether either actually holds is measurable with `test_order_sensitivity.py`
and with the same truth-based metrics as everything else here. Note this is
strictly OUT OF DOMAIN for it: gridworld navigation is nothing like the
support-ticket and review text it was trained on, and its own card reports a
16-point in-domain to OOD accuracy drop. Read the result as "how does a small
open decision encoder transfer", not as its headline quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

LEVEL_DESCRIPTIONS = {
    "none": "no realistic chance of collision",
    "low": "a small chance of collision",
    "medium": "a moderate chance of collision",
    "high": "a large chance of collision",
    "certain": "collision is essentially certain",
}


def build_state(ex: dict) -> str:
    """Identical wording to the Jev client, so the arms are comparable."""
    s = (
        "A gridworld observation, 5x5, centred on the agent.\n"
        "'#' is a wall, '.' is open, 'A' is the agent.\n"
    )
    if ex["kind"] != "wall":
        s += (
            f"The agent's actuator is unreliable: it follows the requested direction with "
            f"probability {ex['rho']:.2f}; otherwise it moves in a uniformly random one of the "
            f"four directions (which may coincidentally be the requested one).\n"
        )
    return s + "\n" + ex["view"]


def build_question(ex: dict) -> dict:
    kind = ex["kind"]
    if kind in ("noul", "wall"):
        # This model fixes noul options to ("no", "yes") and returns P(yes),
        # which is the simulator's convention too.
        return {"type": "noul", "instructions": ex["question"]}
    if kind == "choice":
        return {"type": "choice", "instructions": ex["question"], "options": list(ex["labels"])}
    if kind == "score":
        return {
            "type": "score",
            "instructions": ex["question"],
            "options": [LEVEL_DESCRIPTIONS[x] for x in ex["labels"]],
        }
    raise ValueError(kind)


def parse_answer(ex: dict, q: dict, ans: dict) -> list[float]:
    kind = ex["kind"]
    if kind in ("noul", "wall"):
        p = float(ans["noul"])
        return [1.0 - p, p]
    # For choice and score the probabilities are keyed by the option strings
    # exactly as passed, so map back through the list we sent.
    probs = ans["probabilities"]
    return [float(probs[o]) for o in q["options"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", default="/home/ms/models/open-jev-deberta")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    sys.path.insert(0, a.model_dir)
    from typed_decisions.open_jev import OpenJev

    rows = [json.loads(x) for x in open(a.data)]
    model = OpenJev.from_pretrained(a.model_dir, device=a.device)

    t0 = time.time()
    with open(a.out, "w") as fh:
        for i, ex in enumerate(rows):
            q = build_question(ex)
            ans = model.decide(build_state(ex), [q])[0]
            fh.write(
                json.dumps({"idx": i, "kind": ex["kind"], "probs": parse_answer(ex, q, ans)}) + "\n"
            )
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    print(f"wrote {a.out}: {len(rows)} decisions in {dt:.1f}s ({dt/len(rows)*1000:.0f} ms/decision)")


if __name__ == "__main__":
    main()
