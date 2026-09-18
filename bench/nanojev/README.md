# NanoJev-style decision reads on Qwen3-0.6B

A testbed for **zero-decode structured decisions**: state and question in, a
full probability distribution out, from one forward pass with no token
generation. This is the shape of TypeSafe's Jev and of
[NanoJev](https://github.com/TianyuCodings/NanoJev), reproduced small enough to
run and argue with.

Nothing here touches the Atlas engine. It answers one question first: **does a
proper scoring rule actually buy calibrated decisions on a small backbone, and
by how much?** If it does, the serving path is worth building; if it does not,
no amount of engine work rescues it.

## The mechanism

The prompt is arranged so the answer occupies exactly one token position. The
model runs forward once. The distribution is read off the logits at that
position, restricted to the candidate label ids, and normalised. There is no
sampling loop and no output to parse.

Two predictors share that forward pass:

- **untuned read** — score the candidates with the backbone's own LM head.
  Zero training. This is the floor.
- **decision head** — a bilinear scorer over the hidden state at the answer
  position and each candidate's embedding. Because it scores candidate
  *embeddings* rather than indices, one trained head serves a 2-option boolean
  and a 200-option choice without retraining, which is what "dynamic
  candidates" has to mean.

Labels must be single, distinct tokens or the read is silently wrong —
`resolve_labels` enforces that against the real tokenizer and raises rather
than degrading. A client with multi-token options maps them to stand-ins
(`"moderation_spam"` → `"A"`).

## Why a simulator, not a dataset

The claim a decision model makes is not "I am usually right" but "when I say
0.85 I am right 85% of the time". Testing that needs the *true* probability of
each event. A labelled dataset does not have one — it has a single sampled
outcome, from which you can compute Brier and a binned ECE, but neither can
tell an honest 0.7 from a lucky 1.0.

So `sim.py` builds a gridworld with an unreliable actuator: with probability
`rho` the requested move executes, otherwise a uniformly random one does.
Then

    P(collide | d) = rho * wall[d] + (1 - rho) * mean(wall)

is exact. Every example carries both a **sampled outcome** (all training ever
sees) and the **true distribution** (evaluation only). A model can reach the
truth only by learning the structure, never by memorising labels.

Three question types, all from the same wall geometry:

| type | question | labels |
|---|---|---|
| `noul` | does moving `<d>` collide? | no / yes |
| `choice` | which direction is safest? | the four directions |
| `score` | how risky is moving `<d>`? | none / low / medium / high / certain |

`choice` targets are a softmax over negated collision probabilities, not a
one-hot — a one-hot would make it pure classification with no calibration
content, and ties are common on an open grid, which is exactly where a
miscalibrated model shows itself.

## The three objectives

All are **proper** scoring rules: uniquely minimised, in expectation over
outcomes, by reporting the true distribution. A loss that only wants the right
argmax is free to output 1.0 on a genuine 70/30 event — it scores *better* for
doing so — which is the miscalibration the whole exercise is against.

- `ce` — `-log p[Y]`. Unbounded; punishes confident misses infinitely hard.
- `brier` — `sum_k (p_k - 1[Y=k])^2`. Bounded, gentler on noisy labels.
- `paired` — REINFORCE estimator of the same quadratic objective, with a
  leave-one-out baseline. The RLCD-shaped arm: what you must use when the
  reward is only observable by acting.

`test_objectives.py` is the positive control and should be run first. A wrong
policy gradient does not crash — it trains, converges, and lands on the wrong
distribution. The test points each objective at a free logit vector with a
known target, where anything but recovery is a bug in the estimator rather
than a hard task.

## Running it

```bash
VENV=/home/ms/nanojev-venv/bin/python   # torch 2.14+cu130, transformers
T=/tmp/nj

# 0. the control -- all three objectives must recover a known distribution
$VENV test_objectives.py

# 1. data (train and test from different seeds)
$VENV sim.py --n 6000 --seed 1   --rho 0.75 --out $T/train.jsonl
$VENV sim.py --n 1500 --seed 777 --rho 0.75 --out $T/test.jsonl

# 2. run the frozen backbone once, keep the hidden states
$VENV cache_features.py --data $T/train.jsonl --out $T/train.npz --batch 64
$VENV cache_features.py --data $T/test.jsonl  --out $T/test.npz  --batch 64

# 3. train every arm and score them against the truth
$VENV train.py --train-cache $T/train.npz --test-cache $T/test.npz \
               --train-data $T/train.jsonl --test-data $T/test.jsonl \
               --epochs 200 --seeds 0,1,2,3,4,5,6,7
```

Caching matters for fairness as much as for speed: every arm then trains on
byte-identical features, so a difference between objectives cannot be a
difference in what the backbone happened to produce.

## Reading the output

Metrics are split by what they need:

- **left of the bar** (`NLL`, `Brier`, `ECE`, `out-acc`) — computable in a real
  deployment, against observed outcomes. Floored by the world's entropy: a
  perfectly calibrated model still eats loss on every trial.
- **right of the bar** (`sq-L2`, `MAE`, `true-acc`) — against the true
  distribution. The direct measurement of calibration, available only because
  the generator computes the answer in closed form. `sq-L2` is what NanoJev
  reports as distribution error.

Two reference rows frame the scale. `uniform` is the no-information floor.
`oracle` predicts the truth exactly — and its outcome metrics are **not** zero;
that residual is the world's entropy, the part no model can remove. An arm
that beats the oracle on outcome metrics has memorised sampled labels.

Report medians over several seeds and look at the per-seed spread before
believing any gap between objectives. The arms sit close together and seed
noise swamps small differences — NanoJev say the same of their own comparison.

## Option-order sensitivity: the failure accuracy cannot see

A decision read should be a function of the state and the candidate **set**.
The order the options happen to be written in is not part of the question, so
any dependence on it is pure error — and it is invisible to accuracy on a
fixed dataset, because the dataset always presents the options the same way.
You only find it if you test for it.

`test_order_sensitivity.py` permutes the candidate list, maps the prediction
back to canonical order, and measures what moved. On the **untuned read** with
Qwen3-0.6B (300 examples, 4 permutations each):

| kind | K | flip rate | chance | slot-0 mass | uniform |
|---|---|---|---|---|---|
| `noul` | 2 | 93.3% | 50.0% | 0.707 | 0.500 |
| `choice` | 4 | 63.3% | 75.0% | 0.412 | 0.250 |

Mean total-variation distance 0.308 — roughly a third of the probability mass
moves when the options are merely reordered.

Read carefully, because the two rows say different things. `choice` beats
chance, so there **is** content signal in it. `noul` is *worse* than chance,
which a content-blind random answerer would not manage: the model is
systematically picking whatever is listed first. The slot-mass column is the
direct measurement — under random permutations a content-driven read would sit
at the uniform value in every slot, and slot 0 draws 1.4x (noul) to 1.65x
(choice) its share instead.

Two consequences worth stating plainly:

- Part of what training buys is **learning to ignore position**, not just
  learning the task. Any "training helped by X%" number on this testbed is
  partly that.
- Flips here occur at *higher* confidence than holds (0.565 vs 0.452), the
  opposite of what Verdict reports. So a confidence threshold does not rescue
  it. A read whose distribution tracks presentation is calibrated to the wrong
  thing, and post-hoc calibration cannot fix that.

Ordered `score` questions are excluded: permuting them changes the question
rather than its presentation, and including them would manufacture a large
flip rate that means nothing.

## Related work, and where this sits

Several independent reimplementations of Jev appeared in September 2026, and
they are worth reading before extending this:

- **[NanoJev](https://github.com/TianyuCodings/NanoJev)** — Qwen3-0.6B with
  decision heads and a published RLCD derivation. The design this testbed
  follows most closely. Reports 95%/90% on 4x4/6x6 maze navigation against an
  untuned Qwen3-0.6B's 35%/15%.
- **[SemIf](https://github.com/TheoLeeCJ/SemIf)** (openjev.com) — direct logit
  readout on open models in the browser. Measures direct typed logits at
  **5.21x** the speed of generating JSON (1.023s vs 5.332s on the same work),
  and gets 0.845 modal agreement against Jev's 0.883 on TypeSafe's published
  102-row subset. Its "parallel suffix reuse" (reusing the state prefix across
  criteria) takes 2.33 -> 20.03 decisions/s.
- **[Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev)** — ModernBERT
  151M, RLCD as a *composite* of cross-entropy and Brier, post-hoc L-BFGS
  temperature scaling, and an explicit `__insufficient_evidence__` abstention
  candidate. The most rigorous of the three on robustness: it reports the
  3-4.5% order-flip figure this testbed compares against, and that abstention
  recall collapses from 75.5% to 18% against confusable siblings.

Sobering context for any small-model result here, including this one: on
TypeSafe's own published evaluation, OpenJev scores **48.07%** against
DiffusionGemma-26B's 88.43% and Jev's 90.80%. A clean number on a synthetic
testbed is not evidence that a 0.6B model does this job.

Three things from that work are missing here and would be the next additions:
post-hoc temperature scaling, an abstention candidate, and a composite
CE+Brier objective (a reasonable response to the finding below that the
individual rules are statistically indistinguishable).

## Files

| file | what it is |
|---|---|
| `sim.py` | gridworld with exact answer probabilities |
| `reader.py` | the zero-decode read; single-token label enforcement |
| `heads.py` | bilinear candidate scorer, dynamic K, optional ordinal term |
| `objectives.py` | the three proper scoring rules |
| `test_objectives.py` | positive control for the estimators |
| `cache_features.py` | run the backbone once, keep hidden states |
| `train.py` | train every arm, score against truth, print the comparison |
| `finetune.py` | full fine-tune (backbone + head) with validation-based epoch selection |
| `test_order_sensitivity.py` | permute the options; measure flips, TV distance, slot bias |

## Limits of this testbed

- The backbone is **frozen**; only the head trains. NanoJev fine-tunes the
  backbone too (100 full-model steps, lr 2e-5 / 2e-4), which is the difference
  between a probe and a decision model. A frozen probe's ceiling is whatever
  the hidden state already encodes linearly.
- The task is synthetic and small. It is built to make calibration
  *measurable*, not to be hard.
- `rho` is fixed and stated in the prompt. It has to be: if it varied
  unseen, the true probability would not be determined by the observation and
  no model could reach it.
