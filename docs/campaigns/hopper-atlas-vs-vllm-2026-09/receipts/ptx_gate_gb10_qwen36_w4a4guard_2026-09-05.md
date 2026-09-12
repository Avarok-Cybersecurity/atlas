# Atlas PTX gate — `gb10` @ `sm_121f`

* generated: 2026-09-05T03:27:36Z on `spark1`
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

---

## Reading this receipt (added by hand; everything above is gate output)

This is the GB10 **control** for the W4A4 guard
(`#ifndef ATLAS_NO_WARP_BLOCKSCALE_MMA` in
`kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu`). GB10 does not
define the macro, so it compiles the guarded region in, and the question this
run answers is whether adding the guard moved anything on the target that ships.

**It did not.** `moe_w4a16_grouped_gemm` is not among the failures, and the
file's PTX is byte-identical before and after the guard:
`sha256 137b44c2762d1996c9a1551a906a692cb067edae0b4ee4beee9098d303de4b3a`
(`nvcc --ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS`, both trees).

**The 22 failures are pre-existing and unrelated.** They are the whole
`inferspark_prefill*` family, all rejected at the `_64` (BR=64) entry point for
shared-memory size (`0x16000 bytes, 0xc000 max`), and all of them pass
`nvcc --ptx` — only the gate's separate `ptxas` stage refuses them. The same
gate run against the tree at `cd82c53~1` (this PR's parent) gives **151/173 with
the identical 22 stems**: receipt
`ptx_gate_gb10_qwen36_preguard_control_2026-09-05.*`.

This was the first time the gate had been pointed at `--hw gb10` at all, so the
22 are a finding of that, not of this change. GB10 is the shipping target, so
either the gate's `ptxas` stage is stricter than the shipped pipeline — which
emits PTX and lets the driver JIT it, where a kernel may opt into >48 KB shared
memory at runtime — or those BR=64 entry points are dead on GB10 today. That is
open, tracked separately, and deliberately not resolved here.
