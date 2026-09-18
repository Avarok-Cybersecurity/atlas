# SPDX-License-Identifier: AGPL-3.0-only

"""A gridworld whose answers have EXACTLY KNOWN probabilities.

WHY A SIMULATOR AND NOT A REAL DATASET
--------------------------------------
The claim a decision model makes is not "I am usually right", it is "when I
say 0.85 I am right 85% of the time". Testing that needs the true probability
of each event, and a labelled dataset does not have one -- it has a single
sampled outcome per item. From outcomes alone you can measure Brier and a
binned ECE, but both conflate the model's error with the sampler's noise, and
neither can tell an honest 0.7 from a lucky 1.0.

So the world here is built so the answer is computable in closed form. The
agent's actuator is unreliable: with probability `rho` it executes the
requested direction, otherwise it moves in a uniformly random one (which may
by chance be the requested one). For the question "does moving <d> collide?":

    P(collide | d) = rho * wall[d] + (1 - rho) * (1/4) * sum_d' wall[d']

Every term is known to the generator, so each example carries both a sampled
outcome Y (what training sees) and the exact probability q (what evaluation
compares against, and which training never sees). That split is the whole
point: a model can only reach q by learning the structure, never by
memorising labels.

This mirrors NanoJev's protocol, where a frozen simulator supplies exact
transition probabilities and the observed outcomes come from separate draws.

THE THREE QUESTION TYPES
------------------------
All three are derived from the same underlying wall geometry, so a single
world state yields a consistent set of questions with consistent truths:

  noul    "does moving <d> collide?"      -> P(collide|d), 2 labels
  choice  "which direction is safest?"    -> distribution over 4 directions,
                                             derived from the per-direction
                                             collision probabilities
  score   "how risky is moving <d>?"      -> an ordered bucket of P(collide|d)

`choice` needs care. The safest direction is a deterministic function of the
wall layout, so a one-hot target would make it a pure classification task with
no calibration content. Instead the target is the softmax of the negated
collision probabilities at a fixed temperature: a genuine distribution that is
flat when directions tie and peaked when one is clearly better. Ties are
common on an open grid, which is exactly where a miscalibrated model shows
itself.
"""

from __future__ import annotations

import json
import math
import random

# North, South, West, East. Order is fixed and load-bearing: it indexes the
# wall vector, the label lists and the cached targets alike.
DIRS = ("north", "south", "west", "east")
DELTA = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}

# The observation the model is shown. 5x5 centred on the agent, matching the
# local window NanoJev uses, so the question is answerable from what is shown
# and nothing else. A larger window would leak the whole map and make the task
# memorisable; a smaller one would make some questions unanswerable.
VIEW = 5

SCORE_LEVELS = ("none", "low", "medium", "high", "certain")
# Bucket edges over P(collide). Chosen so an open cell (p = (1-rho)/4 * walls)
# and a walled one land in different buckets at the default rho.
SCORE_EDGES = (0.02, 0.20, 0.45, 0.80)


def _bucket(p: float) -> int:
    for i, e in enumerate(SCORE_EDGES):
        if p < e:
            return i
    return len(SCORE_EDGES)


class Grid:
    """A rectangular grid of walls with an agent somewhere open."""

    def __init__(self, h: int, w: int, wall_prob: float, rng: random.Random):
        self.h, self.w = h, w
        # Border is always wall, so the edge cases (literally) are represented.
        self.cells = [
            [
                1 if (r == 0 or c == 0 or r == h - 1 or c == w - 1) else int(rng.random() < wall_prob)
                for c in range(w)
            ]
            for r in range(h)
        ]
        open_cells = [(r, c) for r in range(h) for c in range(w) if not self.cells[r][c]]
        if not open_cells:
            # Degenerate draw: carve one cell so the world is always usable.
            self.cells[h // 2][w // 2] = 0
            open_cells = [(h // 2, w // 2)]
        self.ar, self.ac = rng.choice(open_cells)

    def is_wall(self, r: int, c: int) -> int:
        if r < 0 or c < 0 or r >= self.h or c >= self.w:
            return 1
        return self.cells[r][c]

    def wall_vector(self) -> list[int]:
        """1 if the neighbour in that direction is a wall, in DIRS order."""
        return [self.is_wall(self.ar + DELTA[d][0], self.ac + DELTA[d][1]) for d in DIRS]

    def collide_probs(self, rho: float) -> list[float]:
        """Exact P(collide | requested direction), in DIRS order.

        With probability `rho` the requested direction executes and collides
        iff that neighbour is a wall. Otherwise a uniformly random direction
        executes -- including, one time in four, the requested one -- so the
        slip term is the mean of the wall vector and does not depend on the
        request.
        """
        walls = self.wall_vector()
        slip = sum(walls) / len(walls)
        return [rho * w + (1.0 - rho) * slip for w in walls]

    def render(self) -> str:
        """The 5x5 local view as text. '#' wall, '.' open, 'A' the agent."""
        half = VIEW // 2
        rows = []
        for dr in range(-half, half + 1):
            row = []
            for dc in range(-half, half + 1):
                r, c = self.ar + dr, self.ac + dc
                if dr == 0 and dc == 0:
                    row.append("A")
                else:
                    row.append("#" if self.is_wall(r, c) else ".")
            rows.append("".join(row))
        return "\n".join(rows)


def make_example(rng: random.Random, rho: float, kind: str, size: int, wall_prob: float) -> dict:
    """One (observation, question, true distribution, sampled outcome) tuple."""
    g = Grid(size, size, wall_prob, rng)
    probs = g.collide_probs(rho)
    view = g.render()

    if kind == "noul":
        di = rng.randrange(len(DIRS))
        p = probs[di]
        # labels[0] is "no" so index 1 == "collides", keeping the positive
        # class at index 1 for the boolean head.
        truth = [1.0 - p, p]
        outcome = 1 if rng.random() < p else 0
        return {
            "kind": "noul",
            "view": view,
            "question": f"Does moving {DIRS[di]} collide with a wall?",
            "labels": ["no", "yes"],
            "truth": truth,
            "outcome": outcome,
            "rho": rho,
        }

    if kind == "choice":
        # A distribution, not a one-hot: see the module docstring.
        temp = 0.25
        neg = [-p / temp for p in probs]
        m = max(neg)
        ex = [math.exp(x - m) for x in neg]
        s = sum(ex)
        truth = [e / s for e in ex]
        outcome = _sample(truth, rng)
        return {
            "kind": "choice",
            "view": view,
            "question": "Which direction is the safest to move?",
            "labels": list(DIRS),
            "truth": truth,
            "outcome": outcome,
            "rho": rho,
        }

    if kind == "score":
        di = rng.randrange(len(DIRS))
        p = probs[di]
        # The true level is a deterministic bucket of p, but the OUTCOME is
        # drawn from a distribution that respects bucket adjacency: a p near an
        # edge genuinely belongs to both neighbours. Without that the ordered
        # levels would carry no more information than a hard classification.
        truth = _level_distribution(p)
        outcome = _sample(truth, rng)
        return {
            "kind": "score",
            "view": view,
            "question": f"How risky is moving {DIRS[di]}?",
            "labels": list(SCORE_LEVELS),
            "truth": truth,
            "outcome": outcome,
            "rho": rho,
        }

    raise ValueError(f"unknown kind {kind!r}")


def _level_distribution(p: float) -> list[float]:
    """Soft membership of `p` in the ordered buckets.

    Mass sits on the containing bucket and leaks to a neighbour in proportion
    to how close `p` sits to the shared edge, so the target is a real ordered
    distribution rather than a relabelled one-hot.
    """
    k = len(SCORE_LEVELS)
    centres = []
    lo = 0.0
    for e in SCORE_EDGES:
        centres.append((lo + e) / 2.0)
        lo = e
    centres.append((lo + 1.0) / 2.0)
    # Distance-weighted membership with a width tied to the bucket spacing.
    width = 0.18
    w = [math.exp(-((p - c) ** 2) / (2 * width * width)) for c in centres]
    s = sum(w)
    return [x / s for x in w]


def _sample(dist: list[float], rng: random.Random) -> int:
    x = rng.random()
    acc = 0.0
    for i, p in enumerate(dist):
        acc += p
        if x < acc:
            return i
    return len(dist) - 1


def generate(n: int, seed: int, rho: float, kinds: tuple[str, ...], size: int, wall_prob: float) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(make_example(rng, rho, kinds[i % len(kinds)], size, wall_prob))
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=0.8, help="actuator reliability")
    ap.add_argument("--size", type=int, default=8, help="grid side")
    ap.add_argument("--wall-prob", type=float, default=0.3)
    ap.add_argument("--kinds", default="noul,choice,score")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = generate(a.n, a.seed, a.rho, tuple(a.kinds.split(",")), a.size, a.wall_prob)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # A generator that silently produced a degenerate set (every answer the
    # same, or every distribution flat) would look fine downstream and make
    # every model score identically. Report the shape so that is visible.
    from collections import Counter

    per_kind = Counter(r["kind"] for r in rows)
    print(f"wrote {len(rows)} examples to {a.out}")
    for k, c in sorted(per_kind.items()):
        sub = [r for r in rows if r["kind"] == k]
        outs = Counter(r["outcome"] for r in sub)
        ent = sum(-sum(p * math.log(p) for p in r["truth"] if p > 0) for r in sub) / len(sub)
        conf = sum(max(r["truth"]) for r in sub) / len(sub)
        print(
            f"  {k:6} n={c:5}  outcome spread={dict(sorted(outs.items()))}  "
            f"mean true entropy={ent:.3f}  mean max-prob={conf:.3f}"
        )


if __name__ == "__main__":
    main()
