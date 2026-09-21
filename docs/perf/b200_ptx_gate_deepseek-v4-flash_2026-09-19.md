<!-- provenance-id: 526f6e616c6420522e205374657369616b -->
# Atlas PTX gate — `b200` @ `sm_100a`

* generated: 2026-09-19T22:38:16Z on `blackbird`
* toolchain: Build cuda_13.0.r13.0/compiler.36424714_0
* strict (`--Werror all-warnings`, as build.rs): True
* HARDWARE.toml `[build] extra_nvcc_flags`: `-DAVAROK_NO_WARP_BLOCKSCALE_MMA`
* self-test: known_good passed=True, `known_bad_post_blackwell_dc.cu` failed=True
* **199/199 kernels compiled** (0 failed, 0 rejected entry function(s))

| model | kernels | pass | fail |
|---|---:|---:|---:|
| deepseek-v4-flash | 199 | 199 | 0 |

No failures: every kernel in this hardware set emitted PTX and assembled for the target architecture.

## Highest register pressure

| model | kernel | max registers | spill bytes |
|---|---|---:|---:|
| deepseek-v4-flash | `gated_delta_rule_persistent` | 255 | 2256 |
| deepseek-v4-flash | `gated_delta_rule_wy2_resident` | 255 | 100 |
| deepseek-v4-flash | `gated_delta_rule_wy2_resident_f16` | 255 | 12 |
| deepseek-v4-flash | `gated_delta_rule_wy3_resident` | 255 | 16 |
| deepseek-v4-flash | `gated_delta_rule_wy3_resident_f16` | 255 | 40 |
| deepseek-v4-flash | `inferspark_prefill_v47` | 255 | 0 |
| deepseek-v4-flash | `kquant_moe` | 255 | 0 |
| deepseek-v4-flash | `gated_delta_rule_fla` | 254 | 0 |
| deepseek-v4-flash | `fp8_gemm_t_blockscaled` | 168 | 0 |
| deepseek-v4-flash | `w4a16_gemm` | 168 | 492 |

Compilation is not correctness. Nothing here has run on b200 silicon; these kernels are known to EXIST for the architecture, not to produce the right numbers or to be fast.
