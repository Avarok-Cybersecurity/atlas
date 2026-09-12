# Atlas PTX gate — `hopper` @ `sm_90a`

* generated: 2026-09-05T02:17:00Z on `spark1`
* toolchain: Build cuda_13.0.r13.0/compiler.36424714_0
* strict (`--Werror all-warnings`, as build.rs): True
* self-test: known_good passed=True, `known_bad_post_hopper.cu` failed=True
* **171/171 kernels compiled** (0 failed)

| model | kernels | pass | fail |
|---|---:|---:|---:|
| nemotron-super-120b-a12b | 171 | 171 | 0 |

No failures: every kernel in this hardware set emitted PTX and assembled for the target architecture.

## Highest register pressure

| model | kernel | max registers | spill bytes |
|---|---|---:|---:|
| nemotron-super-120b-a12b | `gated_delta_rule_fla` | 255 | 324 |
| nemotron-super-120b-a12b | `gated_delta_rule_persistent` | 255 | 124 |
| nemotron-super-120b-a12b | `gated_delta_rule_wy2_resident` | 255 | 120 |
| nemotron-super-120b-a12b | `gated_delta_rule_wy2_resident_f16` | 255 | 40 |
| nemotron-super-120b-a12b | `gated_delta_rule_wy3_resident` | 255 | 204 |
| nemotron-super-120b-a12b | `gated_delta_rule_wy3_resident_f16` | 255 | 64 |
| nemotron-super-120b-a12b | `inferspark_prefill_v47` | 253 | 0 |
| nemotron-super-120b-a12b | `moe_w4a16_grouped_gemm` | 168 | 0 |
| nemotron-super-120b-a12b | `w4a16_gemm` | 168 | 624 |
| nemotron-super-120b-a12b | `fp8_gemm_t_blockscaled` | 167 | 0 |

Compilation is not correctness. Nothing here has run on hopper silicon; these kernels are known to EXIST for the architecture, not to produce the right numbers or to be fast.
