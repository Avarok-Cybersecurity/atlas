# Atlas PTX gate — `b200` @ `sm_100a`

* generated: 2026-09-05T02:07:20Z on `spark1`
* toolchain: Build cuda_13.0.r13.0/compiler.36424714_0
* strict (`--Werror all-warnings`, as build.rs): True
* self-test: known_good passed=True, `known_bad_post_blackwell_dc.cu` failed=True
* **172/173 kernels compiled** (1 failed)

| model | kernels | pass | fail |
|---|---:|---:|---:|
| qwen3.6-35b-a3b | 173 | 172 | 1 |

## Failures

| model | kernel | stage | first error |
|---|---|---|---|
| qwen3.6-35b-a3b | `moe_w4a16_grouped_gemm` | nvcc --ptx | `ptxas /tmp/tmp.85yqNEDTZz/ptx/468e64e5686b1911.ptx, line 18924; error   : Instruction 'mma with block scale' not supported on .target 'sm_100a'` |

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
