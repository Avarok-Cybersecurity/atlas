# SCALE / gfx1151 Phase-0 probes (Atlas → Strix Halo port)

Minimal CUDA repros used to characterise SCALE 1.7.0 codegen for AMD Strix
Halo (`gfx1151`). Compile-only signal; run on the Strix box with
`scale-1.7.0` installed.

```bash
export SCALE_HOME=~/scale17/scale-1.7.0-Linux
NV="$SCALE_HOME/targets/gfx1151/bin/nvcc"
"$NV" --cuda-device-only -c -O3 <probe>.cu -o /tmp/x.o   # device object
"$NV" <probe>.cu -o /tmp/x.out                            # full host link
```

| Probe | Construct | gfx1151 result |
|---|---|---|
| `bf16_mma_shfl_probe.cu` | BF16 `mma.sync m16n8k16` + `__shfl_xor_sync` | ✅ compiles → AMD GPU ELF |
| `cpasync_only_probe.cu` | `cp.async.cg` double-buffer | ✅ compiles |
| `e4m3_mma_only_probe.cu` | FP8 `mma.sync m16n8k32 .e4m3` (pre-packed, no cvt) | ❌ "does not know how to codegen the PTX type: e4m3" |
| `e4m3_mma_cpasync_probe.cu` | `cvt.rn.satfinite.e4m3x2.f32` + e4m3 MMA + cp.async | ❌ `__nv_cvt_floatraw_to_fp8` undefined |

First two are positive controls (the BF16 compute path works on gfx1151).
Last two are the Spectral bug repros: the FP8 **e4m3 tensor-core path** has
no gfx1151 codegen in SCALE 1.7.0 — both the `cvt` pack helper and the
`m16n8k32 .e4m3` MMA itself. Until SCALE adds it, Atlas bridges under
`#if defined(__SCALE__)` (the macro SCALE actually defines — **not**
`__HIP_PLATFORM_AMD__`): the `cvt` via SCALE's exact `__nv_cvt_float_to_fp8`
intrinsic, the MMA via dequant-e4m3→BF16 + BF16 MMA. See
`docs/porting/amd-strix-halo-scale.md` §4.
