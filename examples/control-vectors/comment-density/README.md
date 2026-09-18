# Comment density — a length-invariant control vector

The second worked example. Same four steps as
[`../verbosity/`](../verbosity/README.md), but chosen to test something the
first example could not.

## Why this one, after verbosity

Verbosity is measured on `completion_tokens`, and the sweeps turned up an
awkward fact: **almost any sufficiently strong steering lengthens output.** An
*unrelated* refusal vector moved length +8.6%. Every projection arm came out
longer regardless of scale, including the one that should have pushed the other
way.

So a length metric is partly forgeable. A vector that merely perturbs the
residual stream scores as a weak success, and only a dose–response through zero
distinguishes it from a real axis.

Comment density is measured as a **ratio**:

```
comment lines ÷ total lines, inside fenced code blocks
```

A perturbation that makes the model ramble adds comment lines *and* code lines
and moves the ratio very little. The metric cannot be reached by accident,
which makes this the stronger test of whether the derivation pipeline produces
real directions.

It is also a knob worth having: "explain every step" and "just give me the
code" are both legitimate modes for a coding assistant.

## The corpus

Generated, not hand-written:

```sh
uv run examples/control-vectors/comment-density/generate.py
```

`positive.txt` and `negative.txt` are emitted from one task list, so line N of
each is the same task **by construction**. The verbosity corpus was paired by
hand and verified afterwards; generating it removes the error class instead of
detecting it.

Two properties the generator enforces, both of which would otherwise quietly
ruin the vector:

**Length-matched framings.** Asking for comments naturally produces longer
output. If the positive framing were *also* the wordier prompt, the derived
direction would carry a length component and we would be measuring verbosity
again under a new name. The generator reports the mean-character ratio (1.14
here) and warns outside 0.8–1.25.

**Balanced languages.** 200 tasks across Python, Rust, bash, CUDA, JavaScript,
Go, SQL, C, Java and TypeScript, selected round-robin rather than by slicing
the list in written order. Comment *syntax* differs (`#`, `//`, `--`,
`/* */`) and so does comment *culture* — Rust doc comments, bash header
blocks — and a direction derived almost entirely from Python `#` would be a
narrower thing than the one we mean.

Also enforced: no duplicate tasks (they would double-weight the contrast), 16
rotating phrasings per side (one repeated prefix captures that literal string
rather than the concept).

## Steps

Identical to the verbosity example, with one change that matters:

```sh
# 1. boot with capture armed, same topology you will serve under
spark serve <preset> --model-from-path <path> --control-vector-capture

# 2. both passes, sequentially
uv run scripts/run_control_vector_capture.py \
    --positive examples/control-vectors/comment-density/positive.txt \
    --negative examples/control-vectors/comment-density/negative.txt \
    --out-dir /tmp/comment-density

# 3. size the dose BEFORE choosing scales
uv run scripts/inspect_capture_dump.py \
    /tmp/comment-density/positive.bin /tmp/comment-density/negative.bin \
    --layers 4-44

# 4. derive BOTH magnitude conventions
uv run scripts/derive_control_vector.py ... vector.gguf     --layers 4-44
uv run scripts/derive_control_vector.py ... vector-add.gguf --layers 4-44 \
    --magnitude raw
```

Step 4 emits two files because `project` wants unit rows and `add` wants raw
per-layer magnitudes — see
[`docs/design/control-vector-tooling.md`](../../../docs/design/control-vector-tooling.md).
For a ratio metric you want `add`, since `project` has no direction to travel
in.

## Reading the result

Start the `add` ladder inside the band the verbosity example established
(roughly ±0.02 to ±0.05 for this model at layers 4–44) rather than guessing —
and check the text, not just the ratio. A vector driven too hard produces
beautifully commented code that does not compile, and the ratio will happily
report that as a success.
