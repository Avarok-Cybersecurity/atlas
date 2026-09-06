# Atlas PTX gate — `gb10` @ `sm_121f`

* generated: 2026-09-05T03:29:23Z on `spark1`
* toolchain: Build cuda_13.0.r13.0/compiler.36424714_0
* strict (`--Werror all-warnings`, as build.rs): True
* self-test: known_good passed=True, `known_bad_post_hopper.cu` failed=True
* **151/173 kernels compiled** (22 failed)

| model | kernels | pass | fail |
|---|---:|---:|---:|
| qwen3.6-35b-a3b | 173 | 151 | 22 |

## Failures

| model | kernel | stage | first error |
|---|---|---|---|
| qwen3.6-35b-a3b | `inferspark_prefill` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_64' uses too much shared data (0x16000 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_fp8kv` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_fp8kv_64' uses too much shared data (0x16000 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_64' uses too much shared data (0x16000 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_batched` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_batched_64' uses too much shared data (0x16000 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_bf16k_turbo2v` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_bf16k_turbo2v_64' uses too much shared data (0x16010 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_bf16k_turbo3v` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_bf16k_turbo3v_64' uses too much shared data (0x16020 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_bf16k_turbo4v` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_bf16k_turbo4v_64' uses too much shared data (0x16040 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_fp8` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_fp8_64' uses too much shared data (0x10400 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_fp8_batched` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_fp8_batched_64' uses too much shared data (0x16000 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_fp8k_turbo2v` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_fp8k_turbo2v_64' uses too much shared data (0x16010 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_fp8k_turbo3v` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_fp8k_turbo3v_64' uses too much shared data (0x16020 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_fp8k_turbo4v` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_fp8k_turbo4v_64' uses too much shared data (0x16040 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_indirect` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_indirect_64' uses too much shared data (0x1600c bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_nvfp4` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_nvfp4_64' uses too much shared data (0x16040 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_nvfp4_batched` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_nvfp4_batched_64' uses too much shared data (0x16040 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_turbo2` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_turbo2_64' uses too much shared data (0x16010 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_turbo3` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_turbo3_64' uses too much shared data (0x16020 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_turbo3k_turbo8v` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_turbo3k_turbo8v_64' uses too much shared data (0x16020 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_turbo4` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_turbo4_64' uses too much shared data (0x16040 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_turbo4k_turbo3v` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_turbo4k_turbo3v_64' uses too much shared data (0x16060 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_turbo4k_turbo8v` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_turbo4k_turbo8v_64' uses too much shared data (0x16040 bytes, 0xc000 max)` |
| qwen3.6-35b-a3b | `inferspark_prefill_paged_turbo8` | ptxas | `ptxas error   : Entry function 'inferspark_prefill_paged_turbo8_64' uses too much shared data (0x16000 bytes, 0xc000 max)` |

## Highest register pressure

| model | kernel | max registers | spill bytes |
|---|---|---:|---:|
| qwen3.6-35b-a3b | `gated_delta_rule` | 255 | 1412 |
| qwen3.6-35b-a3b | `gated_delta_rule_persistent` | 255 | 2240 |
| qwen3.6-35b-a3b | `gated_delta_rule_wy2_resident` | 255 | 112 |
| qwen3.6-35b-a3b | `gated_delta_rule_wy2_resident_f16` | 255 | 24 |
| qwen3.6-35b-a3b | `gated_delta_rule_wy3_resident` | 255 | 16 |
| qwen3.6-35b-a3b | `gated_delta_rule_wy3_resident_f16` | 255 | 148 |
| qwen3.6-35b-a3b | `inferspark_prefill_v47` | 252 | 0 |
| qwen3.6-35b-a3b | `gated_delta_rule_fla` | 250 | 0 |
| qwen3.6-35b-a3b | `w4a16_gemm` | 168 | 0 |
| qwen3.6-35b-a3b | `fp8_gemm_t_blockscaled` | 162 | 0 |

Compilation is not correctness. Nothing here has run on gb10 silicon; these kernels are known to EXIST for the architecture, not to produce the right numbers or to be fast.
