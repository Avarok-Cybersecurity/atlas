# Deriving a control vector, end to end

A worked example: build a **verbosity** vector from scratch and measure what it
does. Nothing here is specific to verbosity — swap the corpus and the same four
steps produce any behavioural direction.

## What this actually is

A control vector is a **contrastive mean difference**. There is no training, no
gradient, no optimiser, no fine-tune:

```
for each layer L:
    v[L] = mean(activations over POSITIVE corpus) - mean(over NEGATIVE corpus)
    v[L] = v[L] / |v[L]|
```

That's the whole method. The expensive part is one forward pass per prompt —
and only the *prefill*, since that's where the activations are measured.

## Why verbosity, for an example

Because **the effect is a number.** Steering verbosity moves
`completion_tokens`, so the demo can be checked rather than admired. A style
vector judged by reading prose can only be argued about; this one either moves
the count or it doesn't.

It also separates the two operators, and the separation turned out to be the
main lesson. `add` with a signed scale *displaces* along the axis and gives a
real dial in both directions. `project` does not, and structurally cannot:
`h -= s·(h·v)v` is quadratic in `v`, so negating the vector changes nothing.
Projection removes an axis; it does not travel along one.

Measured, same direction and layers on one serve:

| operator | scale | median Δ `completion_tokens` |
|---|---|---|
| `add` | −0.10 | −90.0% *(too far — answers end mid-sentence)* |
| `add` | −0.05 | **−54.9%** |
| `add` | −0.02 | −20.8% |
| `add` | +0.02 | +63.5% |
| `add` | +0.05 | +225.4% |
| `add` | +0.10 | saturates and degrades |
| `project` | 0.5 / 1.0 / 2.0 | +49.5% / +25.8% / +40.2% — all *longer*, not ordered by dose |
| `project` | 4.0 | output destroyed |

The `add` ladder is monotone through zero with 6/6 sign agreement in every arm.
The `project` ladder never shortens at any scale, including `s=2`, which
reflects the component and in theory should. What it measures instead is that
**perturbation lengthens output** — an unrelated refusal vector moved length
+8.6% the same way.

So: **`project` to ablate a feature, `add` to move along an axis.** Refusal is
the former (there is no useful "more refusal"); verbosity is the latter.

## The corpus is the whole ballgame

`positive.txt` and `negative.txt` are paired **line by line**: line N of each is
the *same underlying request*, differing only in verbosity framing.

```
positive.txt:  Explain in thorough detail, with background and trade-offs, how binary search works.
negative.txt:  In one sentence, how does binary search work?
```

That pairing is not a nicety. The derivation subtracts one mean from the other,
so **anything that differs between a pair other than the property under test
ends up in the vector.** If the positive set happened to be all about databases
and the negative all about networking, you would derive a database-vs-networking
direction and call it verbosity.

Two rules that follow:

- **Vary the framing wording.** Use a dozen rotating phrasings, not one prefix
  200 times, or you capture that literal phrase rather than the concept.
- **Keep the sets balanced in size.** Wildly different token counts skew the
  contrast; the derive script warns above 1.5×.

## Steps

**1. Boot with capture armed.**

```sh
spark serve <preset> --model-from-path <path> --control-vector-capture
```

A derivation mode, not a serving one — it adds a reduction per layer per
forward.

**2. Run both passes.**

```sh
uv run scripts/run_control_vector_capture.py \
    --positive examples/control-vectors/verbosity/positive.txt \
    --negative examples/control-vectors/verbosity/negative.txt \
    --out-dir /tmp/verbosity
```

Sends each prompt with `max_tokens: 1` (prefill is where capture happens) and
**sequentially** — the accumulator is global, so concurrent requests would
interleave two corpora into one sum with no way to separate them afterwards.
Order is reset → positive → dump → reset → negative → dump.

**3. Derive the vector.**

```sh
uv run scripts/derive_control_vector.py \
    /tmp/verbosity/positive.bin /tmp/verbosity/negative.bin \
    /tmp/verbosity/vector.gguf --layers 4-44
```

Read the two diagnostics it prints before trusting the output:

- **raw |mean-diff| per layer** — if these are near zero, the contrast found
  nothing and the normalisation just amplified noise into a unit vector.
- **adjacent-layer cosine** — a real feature is carried coherently across
  depth. The shipped refusal vector reads 0.968 adjacent / 0.722 all-pairs.
  Values near zero mean each layer found a *different* direction, which is what
  noise looks like after normalisation.

**4. Serve it and measure.**

```sh
spark serve <preset> ... --control-vector verbosity=/tmp/verbosity/vector.gguf \
                         --control-vector-layers verbosity=4-44
uv run scripts/measure_verbosity_vector.py --vector verbosity
```

Same prompts, same temperature, same `max_tokens`, same serve — only the
`control_vector` field differs, which is what makes it a comparison rather than
two observations.

## Reading the result honestly

- A delta smaller than the spread **between prompts** is not an effect. The
  measure script prints both and says so.
- If requests hit `finish_reason: length`, the ceiling is clipping the effect —
  raise `--max-tokens` before quoting anything.
- One prompt set on one serve is "observed under this configuration", not a
  settled number. Repeat before treating the magnitude as real.

## What this vector actually captures

Worth being precise, because it is a step removed from what the name suggests.

Within a pair the topic wording is **identical**, so the subtraction cancels it
and what survives is the *framing*. The direction therefore encodes how the
model represents **"a request for elaboration" vs "a request for brevity"**
during prefill — not output length directly.

Steering with it pushes the residual stream toward the elaborate-request
representation, and longer output follows from that. It is the mechanism
`repeng`-style vectors rely on and it works in practice, but the causal chain
has a link in it: the vector is about how the *request* is encoded, and length
is a consequence. If a measured length effect fails to appear, that link is the
first place to look — not the arithmetic.

## Choosing your own contrast

Good candidates share one property: **you can check the result without
squinting.** Verbosity → token count. Code-comment density → comment lines over
total. Sentiment → a classifier.

Worth avoiding: hedging or confidence. It looks like a harmless style knob, but
ablating hedging makes a model assert uncertain things flatly — a calibration
regression wearing a style costume.
