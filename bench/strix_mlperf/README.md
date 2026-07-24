# Strix Halo (gfx1151) MLPerf-edge replication harness

Probes used to produce and gate the Qwen3.6-27B-NVFP4 numbers in
`docs/porting/STRIX_NVFP4_MLPERF.md`. All assume a serve on `localhost:8081`
(see that doc for the exact serve invocation) and `temperature = 0`.

| script | answers |
|---|---|
| `cold_sweep.py` | Cold-prefill throughput: TTFT vs prompt length, fits `ttft = slope*ntok + intercept` → separates fixed overhead from per-token prefill cost. |
| `deep_probe.py` | Warm multi-turn TTFT across 12 growing turns → does TTFT scale with conversation depth? |
| `tpot_probe.py` | Warm decode speed / TPOT, for K (draft-count) A/Bs. |
| `logit_gate.py` + `logit_diff.py` | **Prefill numerics drift gate.** The strongest check we have. |
| `ktrace_agg.py` | Aggregates a `rocprofv3 --kernel-trace` CSV into a per-kernel time breakdown. |

## The drift gate (use this for any prefill-path change)

`logit_gate.py` calls `/v1/completions` with `echo=true, logprobs=5`. That returns
**prompt logprobs** — scored entirely by the prefill path, with no decode, MTP or
sampling in the loop. A 6k-token prompt yields ~4800 independently scored positions,
so it is far sharper than eyeballing a few generated tokens, and unlike argmax
agreement it cannot hide sub-threshold logit drift.

```bash
python3 logit_gate.py /tmp/lp_base.json      # baseline binary/config
# ... rebuild or flip the flag, restart serve ...
python3 logit_gate.py /tmp/lp_cand.json      # candidate
python3 logit_diff.py /tmp/lp_base.json /tmp/lp_cand.json
```

Reports per-prompt `max|dlogprob|`, mean, and max KL over the top-k distributions.
A kernel retiling that only changes *which CTA owns which rows* must come back
**exactly** `0.0` — that is the pass bar used for the M_TILE=64 prefill default.

## Cold-prefill measurement — two traps

1. **Prefix cache.** Every prompt needs a unique *leading* marker, or the radix cache
   shares blocks and you measure a warm path. `cold_sweep.py` does this.
2. **Read the slope, not one point.** A flat tok/s across 1k→8k means a throughput
   problem; a rate that degrades with length means an occupancy/depth problem. They
   have completely different fixes.

## Kernel attribution

`rocprofv3 --kernel-trace` is safe on a live gfx1151 serve (unlike `ncu` on GB10,
which has frozen a box). Wrap the serve, drive one cold prefill, SIGINT, then:

```bash
rocprofv3 --kernel-trace -d /tmp/prof_out -o ktrace --output-format csv -- <serve cmd>
python3 ktrace_agg.py            # last-30s window, per-kernel totals
```

Static occupancy/LDS/VGPR/spill facts come from the compiler, not arithmetic — the
HIP-mirrored sources are dropped in the build dir:

```bash
hipcc -x hip --offload-arch=gfx1151 -O3 \
  -I crates/atlas-kernels/hip/compat -include hip/hip_runtime.h \
  -c hip-target/release/build/atlas-kernels-*/out/hip_mirror/*/w4a16_gemm.cu \
  -o /tmp/probe.o -Rpass-analysis=kernel-resource-usage
```

Note gfx11 LDS is **128 KB per WGP** (64 KB max per workgroup) — occupancy is
`floor(131072 / LDS_per_CTA)` CTAs × waves-per-CTA ÷ 4 SIMDs.

## Box traps (cost real time if unknown)

- `pkill -f "<pattern>"` / `pgrep -f` run over ssh **match their own command line**.
  They will kill your shell before the command runs, and report phantom processes.
  Use `pgrep -x spark`, or collect PIDs and kill by PID.
- `--gpu-memory-utilization 0.40` OOMs on the desktop (`32 MB free / 62.5 GB`) even
  with zero spark processes: the APU GTT reports ~93% allocated and `drop_caches`
  does not release it. **0.35 is the working ceiling.** Allow >60 s drain between
  serve restarts — overlapping restarts wedge the pool.
- `cargo build` defaults to `ATLAS_TARGET_MODEL=qwen3-next-80b`. Without
  `ATLAS_TARGET_MODEL=qwen3.6-27b` a kernel edit produces an **md5-identical binary**
  and you will "measure" a change that never compiled.
- The gb10 `w4a16_bf16_v2_bench` example does **not** run here — it resolves
  `*_bf16` / `*_v2` / `dense_gemm_*` kernels that do not exist in `strix-hip`, via
  `gpu.kernel()`, which hard-errors. Use `examples/strix_ffn_bench` instead.
