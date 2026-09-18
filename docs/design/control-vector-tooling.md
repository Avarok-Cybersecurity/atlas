# Control-vector tooling

Reference for the four scripts that take you from a pair of corpora to a
validated, served control vector — and for the operating knowledge that is not
obvious from the code. Most of what follows was learned by getting it wrong
first; each caveat marks a result that was briefly believed and was false.

For what a control vector *is* and how the apply path works, see
[`qwen4exp-control-vectors.md`](qwen4exp-control-vectors.md). For a worked
end-to-end example, see [`examples/control-vectors/`](../../examples/control-vectors/).

---

## The pipeline

```
   boot with --control-vector-capture
            │
            ▼
   run_control_vector_capture.py      reset → positive → dump → reset → negative → dump
            │                          (two .bin FP64 accumulators)
            ▼
   inspect_capture_dump.py            how big is the stream? what dose is sane?
            │
            ▼
   derive_control_vector.py           mean-difference → .gguf   (--magnitude unit | raw)
            │
            ▼
   inspect_control_vector.py          is it one coherent feature, or 41 noise fits?
            │
            ▼
   boot with --control-vector name=file.gguf ... and measure
```

---

## Choosing the operator

**This is the first decision and it follows from the behaviour, not from taste.**

| | `project` | `add` |
|---|---|---|
| arithmetic | `h -= s·(h·v)v` | `h += s·v` |
| has a direction? | **no** — quadratic in `v`, so `v` and `−v` are identical | yes, signed |
| bounded? | yes — can only remove what is present | no — displaces as far as told |
| use for | a feature to **ablate** | an axis to **traverse** |

Projection is sign-blind. Negating the vector file changes nothing: both `h·v`
and `v` flip, and their product does not. The only knob is `scale` — `0<s<1`
partial removal, `s=1` full removal, `s=2` reflection, `s<0` amplification.

So:

- **Refusal** is a feature to delete. There is no useful "more refusal", and
  projection's sign-blindness costs nothing. This is why the published
  refusal-projection vector works well and why projection was chosen for it.
- **Verbosity, formality, comment density** are axes you want to move along in
  both directions. Projection structurally cannot express that. Use `add`.

**Measured consequence.** A verbosity direction under `project` at scales
0.5/1.0/2.0 gave +49.5% / +25.8% / +40.2% — all *longer*, not ordered by dose,
and never shorter at any scale including `s=2` reflection. The same direction
under `add` gave a clean monotone ladder from −90% to +225% through zero. Same
vector, same layers, same serve.

---

## Sizing the scale

`project` and `add` do **not** share a scale range. They are not comparable
numbers.

| operator | usable range (measured, qwen3.8-flash-next, layers 4–44) |
|---|---|
| `project` | `0.5 – 2.0` usable; `4.0` destroys the model |
| `add` (raw-magnitude file) | `−0.05 – +0.05` useful; `±0.10` past the edge |

Why `add` needs a scale ~40× smaller: the displacement is applied at **every**
active layer and compounds, and the raw mean-difference is already ~30% of the
stream norm at each one. Projection is self-limiting; addition is not.

`control_vector.rs` documents `Add` as wanting "a much smaller scale (~0.1)".
That figure is right and was ignored once here, at the cost of a full sweep:
scales of ±1/2/4 were chosen by reasoning that `|mean-diff|` is the "natural"
dose. Every arm produced `useruseruser…` to the token cap.

**Measure before guessing.** `inspect_capture_dump.py` reports the stream
magnitude. A scale three orders of magnitude too small leaves the model
untouched and publishes as *"add mode does nothing"* — a dose of zero reported
as a null result.

---

## `--magnitude unit` vs `--magnitude raw`

The two operators consume magnitude differently, so the file must be built for
the operator.

- **`unit`** (default) normalises each layer to length 1. Correct for
  `project`: `ControlVector::load` folds a stored row's norm into the per-layer
  scale, so the magnitude cancels and unit rows at scale 1.0 reproduce
  llama.cpp exactly.
- **`raw`** keeps `|mean-diff|` per layer. Required for `add`, which applies the
  row verbatim under **one global scalar**. `|mean-diff|` varies ~11× across a
  typical active range, so a unit file forces one number to dose every layer at
  once — wrong at nearly all of them.

The same derivation can emit both; they are the same direction in two
magnitude conventions, and both can be registered on one serve.

---

## Measurement traps

Each of these produced a wrong number that was briefly believed.

**The ceiling artifact.** When every response hits `max_tokens`, the
"delta" becomes a function of the *baseline alone* and is identical for any arm
that saturates. Three different scales reported `+119.2%` with the same
per-prompt range — that is not a finding, it is the cap erasing the difference
between them. Always report truncation count; treat a fully-truncated arm as
unusable, not as a number.

**`finish_reason: stop` does not mean the answer finished.** Over-driven
brevity makes the model emit EOS mid-sentence — an answer ending at a bare
backtick, reported as a clean `stop`. The usual truncation check cannot see
this. Read the text.

**Degeneracy has several shapes, and a detector built for one misses the
others.** Repetition (low unique-word ratio), multilingual token salad
(*high* unique-word ratio — sails past a repetition test), and
whitespace-free repetition (`useruseruser…` splits into one enormous "word",
so every test gated on a word count silently skips). Check script mixing and
function-word density too.

**Perturbation lengthens output.** Any sufficiently strong steering makes the
model less crisp and therefore wordier — an *unrelated* vector moved length
+8.6%. So a length metric is partly forgeable, and a length result needs a
dose–response through zero, not a single arm. Prefer a metric that is a
**ratio** (e.g. comment lines ÷ total lines) where perturbation moves numerator
and denominator together.

**Always run an unrelated-vector control.** Same operator, same scale, same
layers, different direction. It measures the perturbation floor for that
configuration and tells you how much of your effect is the direction.

**Pair per prompt.** Between-prompt spread here is ~110% of the median, which
swamps arm differences. Same prompt, both arms, temperature 0.

---

## Determinism

At temperature 0 and C=1 this engine is deterministic: the unsteered baseline
reproduced byte-identically across three independent runs and two reboots. The
noise floor is **zero tokens**, so all observed scatter is attributable to
steering. Verify this per model rather than assuming it — it is what makes a
paired comparison meaningful.

---

## Serving

Registration is **boot-time**. Mode, scale, layer range and file are bound to a
name by the operator; callers send only the name.

```sh
--control-vector      "terse=/path/vector-add.gguf"
--control-vector-layers terse=4-44
--control-vector-mode   terse=add
--control-vector-scale  terse=-0.05
```

```json
{"model": "...", "messages": [...], "control_vector": "terse"}
```

The field is a **three-state directive**, not an optional string:

| request | meaning |
|---|---|
| field omitted | take the server default (`--default-control-vector`); no steering if none is set |
| `null`, `false`, `""` | explicitly **no** steering, overriding any default |
| `"name"` | that vector |
| `true` | **rejected** — it does not say *which* |

Omitted and `null` have to differ, or a server default could not exist without
silently changing what every request that omits the field means. An unknown
name is a 400 listing what is registered. The same file may be registered under
many names with different settings — that is how a dose ladder is built.

Available on Chat Completions, Completions, Responses and Anthropic Messages.

**Server policy:**

```sh
--default-control-vector refusal   # what an omitted field gets
--disable-control-vectors          # load none, whatever --control-vector says
```

The default is validated at boot, so a typo stops the serve rather than
surfacing later as a 500 on a caller's traffic. `--disable-control-vectors`
loads nothing at all rather than loading and refusing to use it, so no device
memory is held, no per-layer hook runs, and decode-graph eligibility answers
itself; a request naming a vector then gets a 400 rather than being quietly
served unsteered.

`--control-vector-layers` is **required**; there is no default. It used to
default to every layer that can carry a direction (`1..n_layer-1`), which is
not the configuration any published vector is characterised at — the refusal
projection is tuned over 4..44, so the default silently served 1..47 while the
documentation described 4..44. "Every tensor present in the file" is not the
same claim as "every layer should be steered": a vector carries a direction for
each layer it was *derived* over, and which of those to *apply* is a separate,
tuned decision. Scale and mode do still default (1.0, `project`).

**Every rank must register identically, and this is now enforced rather than
merely logged.** The id a request carries is hashed from the name *and* the
loaded configuration — file SHA-256, mode, scale bits, layer range — so two
ranks configured differently compute *different* ids and the worker's lookup
fails closed. Previously the id came from the name alone: both ranks could
register `"refusal"` from different files or at different scales, the ids
matched, the check passed, and the halves of the model steered differently with
nothing to report it. Each rank logs its full identity line at boot
(`control vector 'x' identity: id=0x… sha256=… mode=… scale=… layers=…`), so a
divergence can be diffed directly.

**Under EP/TP, every rank must register identically.** The selection is not
token-derivable: it travels as an id in the prefill preamble and resolves
against each rank's own registry. A name missing on one rank steers nothing
there while the other rank steers, and the halves disagree silently rather than
erroring. A *mode or scale* that differs is the same hazard wearing a
disguise — the id resolves on both sides, no guard fires, and half the model
runs at a different dose. Diff the load lines of both ranks; they should match
exactly, hashes included.

**Mixed-vector traffic serialises.** A batch is restricted to one composed
variant (`adapter_slot ⊕ cvec_id`) because the kernel applies one vector to the
whole highway per launch. Requests naming different vectors cannot share a
batch and run one cohort at a time. Traffic dominated by a single vector
batches normally and pays nothing.

---

## Scripts

| script | reads | does |
|---|---|---|
| `run_control_vector_capture.py` | corpora | drives both passes, `max_tokens=1`, strictly sequential |
| `inspect_capture_dump.py` | `.bin` ×2 | stream magnitude, contrast fraction, sane `add` scales |
| `derive_control_vector.py` | `.bin` ×2 | mean-difference → `.gguf`, `--magnitude`, coherence diagnostics |
| `inspect_control_vector.py` | `.gguf` | per-layer norms, cross-layer cosine, compare two vectors |

Capture sends `max_tokens=1` because capture happens during **prefill**, and
runs **sequentially** because the accumulator is global — concurrent requests
would interleave two corpora into one sum with nothing able to separate them
afterwards.

Read two numbers from `derive` before trusting output:

- **raw `|mean-diff|` per layer** — near zero means the contrast found nothing
  and normalisation amplified noise into a unit vector.
- **adjacent-layer cosine** — a real feature is carried coherently across
  depth. The published refusal vector reads 0.968 adjacent / 0.722 all-pairs;
  values near zero mean each layer found a different direction, which is what
  noise looks like after normalisation.
