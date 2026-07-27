# Strix Halo (gfx1151): the serve recipe should be the DEFAULT

Running Atlas at its measured best on gfx1151 required **nine** `ATLAS_*` environment
variables. A user who did not have the benchmark script got a materially slower engine
and no indication anything was wrong. This page is the audit that removed them and the
reasoning behind each decision.

## The audit

Every variable in the MLPerf-edge serve recipe, checked against its read site:

| variable | recipe set | code default before | verdict |
|---|---|---|---|
| `ATLAS_FORCE_GLOBAL_GDN` | `=1` | **no read site in any `.rs`/`.cu`/`.hip`** | **DEAD** — deleted |
| `ATLAS_SSM_TAIL_MIDCHUNK` | `=1` | already ON (`!matches!(Ok("0"))`) | redundant — dropped from recipes |
| `ATLAS_SSM_TAIL_PROTECT` | `=1` | already ON (`!matches!(Ok("0")\|Ok("off"))`) | redundant — dropped from recipes |
| `ATLAS_W4A16_DP4A` | `=1` | OFF | **now default ON** under `atlas_scale` |
| `ATLAS_MTP_DRAFTER_PREFILL` | `=1` | OFF | **now default ON** under `atlas_scale` |
| `ATLAS_MTP_GATE_REPROBE` | `64` | `256` | **now defaults to 64** under `atlas_scale` |
| `ATLAS_W4A16_VARIANT` | `v1` | auto → v2 | **auto now resolves to v1** under `atlas_scale` |
| `ATLAS_MTP_CARRY_DRAFTER` | `=1` | OFF | unchanged — see "Not changed" |
| `ATLAS_KV_EXTERNAL_RESERVE_GB` | `6` | auto-measured | unchanged — a genuine operator knob |

Four of the nine did nothing at all. `ATLAS_FORCE_GLOBAL_GDN` was the worst case: it has
had no read site since GDN prefill routing became unconditional under `cfg!(atlas_scale)`
(`layers/qwen3_ssm/trait_prefill_recur.rs` — *"every H-in-shared-memory GDN prefill kernel
exceeds RDNA3.5's 64KB LDS cap … route there for all sizes"*). It survived only in docs
and shell scripts, and shipped in the MLPerf submission README, where it misled anyone
trying to reproduce the result into thinking it mattered.

## Before / after

```diff
-export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1 \
-       ATLAS_KV_EXTERNAL_RESERVE_GB=6 ATLAS_SSM_TAIL_MIDCHUNK=1 ATLAS_SSM_TAIL_PROTECT=1 \
-       ATLAS_MTP_GATE_REPROBE=64 ATLAS_MTP_DRAFTER_PREFILL=1 ATLAS_MTP_CARRY_DRAFTER=1
+export ATLAS_KV_EXTERNAL_RESERVE_GB=6   # co-tenant headroom; box-specific, stays a knob
 spark serve $SNAP --model-name nvidia/Qwen3.6-27B-NVFP4 ...
```

## On PCND

`dp4a.rs` justified its opt-in default in as many words: *"OFF by default (PCND: no
implicit production default)"*. That reading is too literal.

PCND exists so a production path never silently picks a value the operator did not choose
and cannot see. It is not a rule that every shipped win must be rediscovered from a
benchmark script. **A hardware-gated default with a documented kill switch is an explicit
choice** — it is declared in code, documented here, and reversible with one variable. The
status quo was strictly worse by PCND's own standard: the real default was "whatever the
benchmark script exports", which is invisible to users and recorded nowhere in the source.

Each flipped default therefore ships as: default-on **for the target it was measured on**,
kill switch `=0`, and a comment naming the measurement.

## Scoping

Defaults are scoped to `atlas_scale` (`ATLAS_TARGET_HW=strix*`, i.e. gfx1151 via SCALE or
native HIP) — **not** flipped globally:

- `ATLAS_W4A16_DP4A` — the DP4A kernels only exist in gfx1151 builds. Elsewhere the kernel
  handle misses and callers fall back to the float E2M1-LUT path regardless, so a global
  flip would be a no-op that merely looks risky.
- `ATLAS_MTP_DRAFTER_PREFILL`, `ATLAS_MTP_GATE_REPROBE` — these are cross-target codepaths.
  Both were measured on gfx1151; GB10 has its own MTP tuning and neither has been A/B'd
  there. Flipping them globally would change GB10's shipping behaviour on the strength of
  a measurement from different hardware.
- `ATLAS_W4A16_VARIANT` — v2 remains the auto pick on NVIDIA; only gfx1151's auto moves
  to v1.

`spark-server` has no `build.rs` and so never receives the `atlas_scale` cfg. Rather than
duplicate the cfg plumbing, the predicate is exported once from `spark-runtime` as
`atlas_scale_target()` and used there.

## Not changed

- **`ATLAS_KV_EXTERNAL_RESERVE_GB`** stays a flag. It reserves headroom for co-tenants
  that the auto-measurement cannot see because they have not started yet. That is
  genuinely operator knowledge, not a hardware property, and `6` is specific to this box.
- **`ATLAS_MTP_CARRY_DRAFTER`** stays default-off. It is in the recipe, but it was measured
  **inert** on strix, and "inert" is not a reason to flip a default — it is a reason to run
  a proper A/B and then either promote it or delete it from the recipe. Left alone pending
  that measurement rather than promoted on the strength of it being in a script.

## Validation

The defaults are only correct if they reproduce the recipe. `bench/strix_mlperf/` includes
the A/B: one arm exports all nine variables, the other exports none, same binary.

The gate is **byte-identical emitted text** across the two arms plus latency within noise.
Anything else means a default does not match what the recipe was setting, and the
difference is the bug.
