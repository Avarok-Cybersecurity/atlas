<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# What MLX's float32 arithmetic actually does

Measured, not assumed. Re-run with `mlx_oracle.py` in this directory against an
MLX install on the target Mac; the numbers below are from **mlx 0.29.3 /
mlx-metal 0.29.3, Device(gpu, 0), Apple M4 Pro (apple-48gb-metal), 2026-09-18**.

A `tier = "T2"` row claims bit-identity with MLX. Nothing can honestly claim it
until these three questions have answers, because each one decides whether a
Metal kernel *can* reproduce MLX's bits at all.

| question | measured answer | what it forces on our kernels |
|---|---|---|
| Does MLX contract `a*b+c` into an FMA? | **No.** Over 64 inputs chosen *because* the two behaviours differ there, MLX matched two-roundings 64/64 and contracted-FMA 0/64. | Compile Metal with `-ffp-contract=off`. An explicit `fma()` in a kernel that mirrors an MLX expression is a bit-level divergence, not an optimisation. |
| Are its transcendentals bit-identical to numpy float32? | **No, but within 1 ULP.** `exp` 1563/2500 identical, `rsqrt` 1842/2500, max delta 1 ULP either way. | numpy is not a usable oracle for T2. Match MLX itself, or take T1 with a stated ULP bound. |
| Is `mx.sum` reduction order-dependent? | **Yes.** Permutation-invariant in only 10/32 random permutations of the same 4096 values. Run-to-run on *identical* input: deterministic. | A reduction whose partition differs from MLX's differs in the low bits. Bit-identity needs MLX's exact partition, not merely a correct sum. Determinism means our own byte-stability tests remain meaningful. |

## Two traps this measurement walked into first

**A non-discriminating probe answered confidently.** The first FMA attempt used
`a = b = 1+1ulp, c = -1`, where two-roundings and contracted-FMA produce the
*same* bits (`0x34800000`). The probe printed "does NOT contract" — the right
answer by luck, from an input that could not have told the difference. The
version here SEARCHES for inputs where the two behaviours provably diverge and
reports how many were found, so an inconclusive run cannot masquerade as a
verdict.

**One permutation is not a sample.** The first reduction probe permuted once,
got identical bits, and concluded order-independence. Thirty-two permutations
show that agreement happens 10 times in 32 — the single draw landed in the
agreeing third. The order-dependence is real and would have silently justified
a wrong partition in `mlx_int8_gemv`.

## Consequence for the map

No row may carry `tier = "T2"` on the strength of a numpy comparison. The
reachable claims today are T1 (FP32 CPU reference within a stated bound, plus
registered mutations) for everything, and T2 only where a kernel reproduces
MLX's own partition and rounding — which, per the table above, means no
contraction, MLX's transcendental implementation, and MLX's reduction tree.
