<!-- provenance-id: 526f6e616c6420522e205374657369616b -->
# DeepSeek-V4.1 decode, phase 2 prototypes (S2)

The two pieces that make a device-side, bit-exact routing selection possible, each with its
exhaustive or randomized check built in (see `docs/perf/DS41_DECODE_RETUNE_2026-09.md`):

* `glibc_softplus_exact.cu`: glibc 2.39's `expf` (table + three double FMAs, round-half-away
  `frinta` / `fcvtas`) and `log1pf` (the Sun algorithm with the exact FMA pattern of the
  installed aarch64 `libm.so.6`, read off `objdump`) as device functions, verified bit-identical
  to the host over all 2^32 `f32` inputs, and so is `sqrtf(softplus(x))`, the router's score.
  CUDA's own `expf` / `log1pf` differ from glibc on 58.4 M inputs, which is why a naive device
  top-k cannot reproduce the host's selection.
* `route_select_dev.cu`: one block of 384 threads per token: score, `score + bias` key, rank by
  (key descending, index ascending) = Rust's stable `sort_by(partial_cmp(b, a))`, top-6, the
  weights in pick order (`__fadd_rn` / `__fdiv_rn` / `__fmul_rn`, so nvcc cannot fuse), the plan
  in ascending expert id with a device slot lookup and a miss flag. 20,000 random tokens with
  forced ties: 0 mismatches in picks, weight bits, plan or flags.

Not wired into the engine on one Spark: with 65% of decode steps missing an expert, the host
must act inside the layer anyway (fetch), so the round trip stays; on a B200 (everything
resident) this is what a whole-step CUDA graph needs.
