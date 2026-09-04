# A better seam for weight encodings

Written 2026-09-03 after landing EXL3 (QTIP trellis) for Qwen3.8-Flash-Next and
scoping it for Qwen3.8-27B. This is a design proposal, not a plan of record.

Formats in tree today: **modelopt** (NVFP4), **compressed-tensors**, **fp8
block-scaled**, **EXL3**, and a limited **GGUF** experiment we do not expect to
keep. The EXL3 work shipped, but it went in *around* the abstraction rather than
through it, and this records why and what a second attempt should look like.

## The core defect

`QuantFormat::base_variant() -> Nvfp4Variant` (`quant_format/mod.rs:61`).

The trait's only output is another format's enum. Every format must be
expressible as an NVFP4 variant, so a format whose weights are not NVFP4-shaped
cannot be added as a format at all. EXL3 therefore became a **pre-pass that
rewrites the weight store** (`weight_map::materialize_exl3`, called at
`factory/build.rs:108`, *before* `loader_for_config` at `:112`) into
NVFP4-triplet + BF16-dense shapes, after which `detect_quant_format` returns a
`ModeloptFormat`. The comment says it plainly: *"exl3 (materialized to
modelopt-style NVFP4 + BF16 dense)"*. A format is impersonating another format
so the loaders downstream do not have to know it exists.

Everything below follows from that one decision.

## What it cost

**1. Three parallel mechanisms, no single owner.**

| phase | mechanism | knows about EXL3? |
|---|---|---|
| detect | `QuantFormat` trait | only enough to lie |
| load | `materialize_exl3` store rewrite | yes, imperatively |
| serve | `ATLAS_EXL3_NATIVE*` + `install_native_{gdn,attn}` + `Exl3DenseStage` | yes, separately |

Nothing answers "how does format F serve tensor family T" in one place. Adding a
format means editing all three, plus each model loader's layer-construction
sites (8 in `qwen35_dense.rs` alone), plus the factory, plus new env gates.

**2. Two sources of truth about tensor identity.** `exl3_dense_prefix_family`
(`exl3_materialize_dense.rs:248`) string-matches `.self_attn.q_proj` to decide a
family — while the loader constructing that layer already knows it is a q_proj.
That duplication is why `Exl3DenseFamily` is a hardcoded `{Gdn, Attn}` and why
"the dense FFN has no native family" is a *missing enum variant* rather than
data. On a dense 27B that leaves 54% of packed weights with no native path.

**3. No memory contract.** `exl3_materialize.rs:25` reasons "these tensors are
small enough (~6 GB on Qwen3.8-Flash-Next) that BF16 residency until
construction is fine." True there only because ~90% of that model is routed
experts taking a separate NVFP4 path. On a dense 27B every linear falls to the
BF16 branch: **47.7 GiB resident, ~50-55 GiB peak** — fits the box, blows the
util pledge. The pass is not wrong; nothing ever declared what it would cost.

**4. State lifetime discovered, not declared.** `NativeExl3` had to be hoisted
to a scope spanning two load phases (`61e99abfc`) because the MTP module loads
in `serve_phases/weights.rs` step 5 while the layers load earlier. Nothing said
"this format needs exactly one model-shared state"; each loader owned a local
until that broke. The refusal added in the meantime cited a stale rationale —
`Exl3LaunchState::shared()` was *already* a process-wide anchor — which is what
happens when a lifetime is folklore rather than a declaration.

**5. Gate proliferation.** `ATLAS_EXL3_NATIVE`, `_MOE`, `_DENSE`, `_GDN`,
`_ATTN`, `_WEIGHT_POOL`, plus per-fix switches. Each is an independent env read
with its own default. There is no registry, so "what is actually armed?" is
answered by grepping.

## The proposed shape

Make the trait answer the question the system actually asks, per tensor, and let
its answer be format-neutral.

```rust
pub trait WeightFormat: Send + Sync {
    fn name(&self) -> &'static str;

    /// How should this tensor be served, given what the hardware can run?
    /// Called once per tensor during planning — never on a hot path.
    fn plan(&self, t: &TensorDesc, hw: &KernelEnvelope) -> ServePlan;
}

pub enum ServePlan {
    /// Serve the packed bytes directly.
    Native  { kernel: KernelId, state: StateReq },
    /// Convert at load. `resident_bytes` is a PLEDGE, not a hint.
    Materialize { to: DenseForm, resident_bytes: u64 },
    /// Refuse loudly, with a reason a user can act on.
    Unsupported { why: String },
}
```

`TensorDesc` carries the **role the loader already knows** (`AttnQProj`,
`GdnInProjQkv`, `FfnGateProj`, `LmHead`, `MoeExpertDown`, …), dims, dtype and
per-tensor quant metadata — not a name to be re-parsed. `KernelEnvelope` is the
existing K/codebook/geometry admissibility check.

### What each property buys

- **No format's enum in the trait.** EXL3 becomes a first-class format instead
  of a pre-pass wearing ModelOpt's clothes. GGUF, if ever kept, is one more impl.
- **`resident_bytes` is a pledge.** Planning sums it before a byte is read, so
  the 47.7 GiB case is a refusal or a warning at plan time, next to the existing
  util pledge, rather than a surprise 40 minutes into a boot. This is the
  `atlas-alloc-ledger` pattern extended to load time — the ledger already taught
  us to read the table before theorising.
- **Role comes from the loader.** "Add FFN support" becomes one match arm, not a
  new enum variant plus arms plus call sites. The string-matching predicate dies.
- **`StateReq` declares lifetime.** `ModelShared(Exl3Launch)` tells the factory
  to create it once at a scope spanning every load phase. The hoist becomes the
  default rather than a bug fix, and the one-dispatch-section-at-a-time invariant
  is expressed rather than remembered.
- **One decision point.** Native vs materialise stops being two subsystems.

### The plan phase

Before loading, walk the checkpoint and emit a table: per-role decision, per-role
byte totals, peak resident, states required. Log it at INFO. That artefact alone
would have answered "does 27B fit" without a boot, and would have caught the
dense-FFN gap as data (`FfnGateProj -> Materialize`) rather than as a discovery
three days later.

## Migration, without a big bang

1. Add `plan()` alongside `base_variant()`; implement it for EXL3 only. Rewrite
   `materialize_exl3` as a *consumer* of the plan instead of the decider. Nothing
   else moves. This is where the memory pledge lands.
2. Thread `TensorRole` from the loader construction sites into `plan()`, deleting
   `exl3_dense_prefix_family`. Add `Ffn` as a role — which is also what unlocks
   dense 27B.
3. Port the other three formats to `plan()`, then delete `base_variant()` and the
   `Nvfp4Variant` coupling.
4. Fold the `ATLAS_EXL3_*` gates into one registry keyed by role, so "what is
   armed" is a single logged table.

## What is already right and should be kept

- `detect_quant_format`'s decision order — config first, then on-disk heuristic,
  then a *loud* fallback — is correct and worth preserving verbatim.
- Vendored kernels isolated under `kernels/gb10/common/exl3_vendor/` with no
  per-target gating, so they compile everywhere without per-model plumbing.
- The keep-predicates are pure functions and therefore CPU-testable. Keep that
  property when they move behind `plan()`; it is why the K-ladder work could be
  validated without a GPU.

## Cost estimate

Steps 1-2 are the ones that pay: they remove the pre-pass lie, give load-time a
memory pledge, and unlock dense models. Roughly a week including tests, and it
can land incrementally behind the existing gates. Steps 3-4 are cleanup that can
follow whenever.
