# SPDX-License-Identifier: AGPL-3.0-only

"""Score TypeSafe's Jev on the simulator, against closed-form truth.

WHY THIS IS WORTH DOING
-----------------------
Jev's central claim is calibration: "when the model reports an 85% confidence
score, the probabilities are statistically aligned so that it is correct 85% of
the time." Every public comparison of it -- the open reimplementations, the
published evals -- measures ACCURACY against labelled data. None of them can
measure calibration directly, because labelled data has one sampled outcome per
item and no true probability.

This testbed does have the true probability, in closed form. So the same
metrics already used for the local models apply unchanged to Jev, and the
answer is a direct measurement of the claim rather than a proxy for it.

READ THE RESULT CAREFULLY
-------------------------
Two of the four question types ask for something Jev is not obviously built
for, and the comparison is only fair if that is stated up front:

  wall    "is the cell north of the agent a wall?"  Pure perception. The answer
          is visible in the observation, the true distribution is 0 or 1, and
          no arithmetic is involved. A System One model should be good at this.

  noul    "does moving north collide?"  The answer requires COMPUTING
          rho*wall[d] + (1-rho)*mean(wall) from a noise model stated in prose.
          That is System Two work, and Jev is explicitly a System One model --
          fast structured judgement, not deliberation.

  choice  "which direction is safest?"  Ranking, plus the same arithmetic to
          break near-ties honestly.

  score   "how risky is moving north?"  Bucketing the same computed quantity.

So `wall` is the control that makes the rest interpretable. A model that fails
`wall` cannot read the grid, and nothing else it does here means anything. A
model that passes `wall` and misses `noul` is failing the probability
computation specifically -- which is a real limitation to know about, but not
the same as being miscalibrated on the tasks it is sold for.

Nothing here is a verdict on Jev for routing, moderation or classification,
which is what it is actually for. It is a measurement of one property, on one
synthetic task, chosen because the property is otherwise unmeasurable.

DATA
----
The states sent are synthetic gridworld renderings. No real or sensitive
content leaves the machine.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

URL = "https://api.typesafe.ai/v1/systemone"

LEVEL_DESCRIPTIONS = {
    "none": "no realistic chance of collision",
    "low": "a small chance of collision",
    "medium": "a moderate chance of collision",
    "high": "a large chance of collision",
    "certain": "collision is essentially certain",
}


def build_state(ex: dict) -> str:
    """The observation, plus the actuator model the truth depends on.

    `rho` has to be in the prompt: without it the collision probability is not
    determined by the observation, and no model -- Jev, ours, or a perfect one
    -- could be expected to reach the true value.
    """
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
        if kind == "wall":
            crit = {"true": "that cell is a wall", "false": "that cell is open"}
        else:
            crit = {
                "true": "the move ends in a collision with a wall",
                "false": "the move does not collide",
            }
        return {"type": "noul", "instructions": ex["question"], "criteria": crit}
    if kind == "choice":
        return {
            "type": "choice",
            "instructions": ex["question"],
            "criteria": {d: f"moving {d} is the safest available move" for d in ex["labels"]},
        }
    if kind == "score":
        return {
            "type": "score",
            "instructions": ex["question"],
            "criteria": [LEVEL_DESCRIPTIONS[l] for l in ex["labels"]],
        }
    raise ValueError(kind)


def parse_answer(ex: dict, ans: dict) -> list[float]:
    """Jev's answer, mapped onto this example's canonical label order."""
    kind = ex["kind"]
    if kind in ("noul", "wall"):
        # `noul` is P(true); labels are ["no", "yes"] so index 1 is the
        # positive class, matching the simulator's convention.
        p = float(ans["noul"])
        return [1.0 - p, p]
    if kind == "choice":
        probs = ans["probabilities"]
        # Keyed by the criteria keys, which are our labels.
        return [float(probs[d]) for d in ex["labels"]]
    if kind == "score":
        probs = ans["probabilities"]
        # Keyed by level index as a string, in the order criteria were given.
        return [float(probs[str(i)]) for i in range(len(ex["labels"]))]
    raise ValueError(kind)


def call_one(key: str, ex: dict, timeout: int, max_retries: int = 5) -> dict:
    body = {
        "model": "jev-latest",
        "state": build_state(ex),
        "questions": {"q": build_question(ex)},
    }
    data = json.dumps(body).encode()
    delay = 1.0
    for attempt in range(max_retries):
        req = urllib.request.Request(
            URL,
            data=data,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 429 and 529 are the documented backoff cases; anything else is a
            # real error and retrying it just wastes the budget.
            if e.code in (429, 529) and attempt < max_retries - 1:
                time.sleep(delay + random.random())
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf8','replace')[:300]}") from e
        except Exception as e:  # noqa: BLE001 - network flake
            if attempt < max_retries - 1:
                time.sleep(delay + random.random())
                delay *= 2
                continue
            raise RuntimeError(repr(e)) from e
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--key-file", default="/home/ms/typesafe.key")
    ap.add_argument("--out", required=True, help="jsonl cache of raw responses")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    key = open(a.key_file).read().strip()
    rows = [json.loads(x) for x in open(a.data)]
    if a.limit:
        rows = rows[: a.limit]

    # Resume: an interrupted run must not re-pay for answers already held.
    done = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done[r["idx"]] = r
        print(f"resuming: {len(done)} cached")

    todo = [i for i in range(len(rows)) if i not in done]
    lock = threading.Lock()
    fh = open(a.out, "a")
    errors = []
    tok_in = tok_out = 0
    t0 = time.time()

    def worker(idxs):
        nonlocal tok_in, tok_out
        for i in idxs:
            try:
                resp = call_one(key, rows[i], a.timeout)
                probs = parse_answer(rows[i], resp["answers"]["q"])
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append((i, repr(e)[:200]))
                continue
            rec = {
                "idx": i,
                "kind": rows[i]["kind"],
                "probs": probs,
                "model": resp.get("model"),
                "usage": resp.get("usage", {}),
            }
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                tok_in += resp.get("usage", {}).get("input_tokens", 0)
                tok_out += resp.get("usage", {}).get("output_tokens", 0)
            if (i % 20) == 0:
                print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    chunks = [todo[k :: a.concurrency] for k in range(a.concurrency)]
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fh.close()

    print(f"\ncalls attempted: {len(todo)}   errors: {len(errors)}")
    for i, e in errors[:5]:
        print(f"  idx {i}: {e}")
    print(f"tokens in={tok_in} out={tok_out}   elapsed {time.time()-t0:.0f}s")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
