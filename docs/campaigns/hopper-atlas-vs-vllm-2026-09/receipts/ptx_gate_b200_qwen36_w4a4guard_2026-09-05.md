# Atlas PTX gate — `b200` @ `sm_100a`

* generated: 2026-09-05T03:29:20Z on `spark1`
* toolchain: Build cuda_13.0.r13.0/compiler.36424714_0
* strict (`--Werror all-warnings`, as build.rs): True
* HARDWARE.toml `[build] extra_nvcc_flags`: `-DATLAS_NO_WARP_BLOCKSCALE_MMA`
* self-test: known_good passed=True, `known_bad_post_blackwell_dc.cu` failed=True
* **173/173 kernels compiled** (0 failed)

| model | kernels | pass | fail |
|---|---:|---:|---:|
| qwen3.6-35b-a3b | 173 | 173 | 0 |

No failures: every kernel in this hardware set emitted PTX and assembled for the target architecture.

## Highest register pressure

| model | kernel | max registers | spill bytes |
|---|---|---:|---:|
| qwen3.6-35b-a3b | `gated_delta_rule` | 255 | 1396 |
| qwen3.6-35b-a3b | `gated_delta_rule_persistent` | 255 | 2256 |
| qwen3.6-35b-a3b | `gated_delta_rule_wy2_resident` | 255 | 100 |
| qwen3.6-35b-a3b | `gated_delta_rule_wy2_resident_f16` | 255 | 12 |
| qwen3.6-35b-a3b | `gated_delta_rule_wy3_resident` | 255 | 16 |
| qwen3.6-35b-a3b | `gated_delta_rule_wy3_resident_f16` | 255 | 40 |
| qwen3.6-35b-a3b | `inferspark_prefill_v47` | 255 | 0 |
| qwen3.6-35b-a3b | `gated_delta_rule_fla` | 254 | 0 |
| qwen3.6-35b-a3b | `fp8_gemm_t_blockscaled` | 168 | 0 |
| qwen3.6-35b-a3b | `w4a16_gemm` | 168 | 492 |

Compilation is not correctness. Nothing here has run on b200 silicon; these kernels are known to EXIST for the architecture, not to produce the right numbers or to be fast.
