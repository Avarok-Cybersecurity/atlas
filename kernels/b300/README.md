# B300 Kimi K3 bring-up target

Build with `AVAROK_TARGET_HW=b300 AVAROK_TARGET_MODEL=kimi-k3` and select
`AVAROK_TARGET_QUANT=bf16` for the small twin or `mxfp4` for official packed
weights. The `nvfp4` directory retains the registry's default quant alias; it
does not convert MXFP4 weights into NVFP4.

This target uses `sm_103a` (CUDA 12.9 or newer compiler support; the repository
requires CUDA 13.0+). B200 `sm_100a` and Spark `sm_121f` binaries must still
fail the B300 architecture preflight. Runtime device memory and SM count must
be checked on the actual rental. All 182 selected Kimi MXFP4 kernels compiled with CUDA 13.0.88 for
`sm_103a` on an ARM64 Spark host. B300 module-load, inference and performance
remain unverified; cross-compilation does not execute on B300.

The common kernels, Kimi kernels and DeepSeek E8M0 GEMM dependency are real,
B300-owned snapshots from the commit recorded in `SOURCE_SNAPSHOT.json`.
Its hashes record initial provenance, not a requirement to track later GB10
changes. All aliases stay within B300, so later B300 tuning cannot modify
Spark, Hopper, B200 or AMD inputs through a link. Keep future optimized
sources here; do not edit another hardware target via a symlink.

Conservative serving defaults are explicit in `HARDWARE.toml`. The existing
W4A16 E8M0 path is the initial numerical baseline. Native datacentre FP4
block-scaled MMA needs a separate tcgen05 implementation and correctness and
performance receipts before enabling it. This target does not prove packed
TP8 loading or full Kimi inference; those are separate integration gates.

## Changes after the source snapshot

`SOURCE_SNAPSHOT.json` records the **original upstream source hashes**, not
hashes of the current B300 files. Subsequent B300 changes are recorded by Git:

- `common/moe_shared_expert_fused.cu`: removed the inherited hardcoded
  DeepSeek-only activation clamp from generic SiLU decode. Routed and shared
  experts now use the same plain `silu(gate) * up` math. Kimi's LatentMoE calls
  the separate E8M0 GEMM and `situ_glu_vec` path; it does not dispatch this
  generic SiLU kernel. No GB10 source or clamp-scope exception was changed.

## Focused extraction

This target is extracted from draft #1150 at `bba13ef3a`. Historical compiler
and device observations above apply to the original integration workspace;
this extraction runs CPU-side structural/registration checks only. It does not
add the model runtime integration or qualify a complete serving build.
Existing Hopper/B200 scaffolding from merged #1045 is reused, and the GB10
expert alias changes are reviewed separately in #1165. No GPU or rental was
started for this extraction. Keep the review draft until its dependent runtime
work and required hardware qualification are complete.

The owned snapshot deliberately preserves source bytes, including inherited
extra blank lines at EOF in `common/gated_delta_rule_wy.cu` and
`common/gated_delta_rule_wy_f16.cu`. `git diff --check` reports those two
whitespace-only findings; no kernel arithmetic was changed to clean them up.
