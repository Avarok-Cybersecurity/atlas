<!-- provenance-id: 526f6e616c6420522e205374657369616b -->
# DeepSeek-V4 / V4.1 Flash kernels on the b200 target (sm_100a)

The kernel set for `deepseek-v4-flash` and, through its `kernel_source`, `deepseek-v4.1-flash` on datacentre Blackwell (B200 / GB200, SM 10.0). Every `.cu` and the `KERNEL.toml` in `nvfp4/` is a symlink into `kernels/gb10/deepseek-v4-flash/nvfp4/`: one source, two targets, the mechanism every b200 model directory in this tree uses (`scripts/check_cross_hardware.py` rule S1 forbids NEW cross-hardware symlinks and leaves these standing). `SOURCE_SNAPSHOT.json` pins the exact bytes behind each link (git blob at cc06bf548 and sha256), so a build on rented hardware can be checked against what the gate below compiled.

**Nothing in this directory has run on a B200.** What is established here is that the kernels exist for the architecture and what they cost in registers and shared memory; correctness and speed on the part need the part.

## The gate (2026-09-19, CUDA 13.0.88, this box)

```
scripts/hopper_ptx_gate.sh --hw b200 --model deepseek-v4-flash --strict --jobs 12
```

`nvcc --ptx -arch=sm_100a -O3 --fmad=false -DAVAROK_NO_WARP_BLOCKSCALE_MMA --Werror all-warnings`, then `ptxas -arch=sm_100a -v`, over the 199 kernel sources this model directory reaches (23 in `nvfp4/`, the rest in `common/`), with the gate's own self-test (a known-good fixture and the sm_100a negative fixture) run first: **199/199 compiled, 0 rejected entry functions, 0 source changes needed.** Because no source was touched, GB10 byte identity is the byte identity already proven on `main` (the serial GPU tests and the 6/6 oracle standard of #1155); nothing had to be re-proven.

For the record: the same gate against the gb10 set at `sm_121f` reports 22 failures, all `inferspark_prefill*` entries in `common/` exceeding the 48 KB static shared-memory cap of the gate's compile mode; they predate this work, are not on the V4.1 path, and are not touched here.

## The ptxas resource table

`PTXAS_RESOURCES.md` has every entry function of the 13 modules the V4.1 runtime loads (`attn_v41`, `moe_v41`, `engram_v41`, `hc_v41`, `kquant_moe`, `csa_compress`, `hyper_connection`, `dense_gemm_bf16`, `dense_gemv_bf16`, `dequant_gguf_bf16`, `rms_norm`, `argmax_bf16`, `embed_from_argmax`): registers, static shared memory and spill bytes on sm_100a against sm_121f. 105 entries, no spills on either. What it says for the decode hot path:

| entry | regs sm_100a / sm_121f | note |
|---|---|---|
| `kquant_mmvq_q2_k_experts_w8`, `q3_k_experts_w8` | 40 / 40 | the routed experts (17 of 44 ms a token on GB10); 8 warps a block, 256 threads; at 40 registers the SM holds 6 blocks either way |
| `kquant_mmvq_q2_k_w` | 48 / 48 | the attention projections (276 launches a token) |
| `kquant_mmvq_q6_k_w` | 40 / 48 | the LM head |
| `moe_v41_router_gemv_f32out_products` | 40 / 44 | |
| `attn_v41_sparse_attn` | 32 / 40 | 9,216 B smem |
| `hc_v41_mixes_dot` | 32 / 48 | |
| `attn_v41_index_score` | 164 / 164 | the one high-pressure decode kernel; runs on the eight index-source layers only |
| `atlas_q2_k_mmq128_{nc,wc}` | 255 / 255 | the prefill MMQ tiles, at the register ceiling on both parts, no spills |
| `atlas_q3_k_mmq128_{nc,wc}` | 230-234 / 254 | |

sm_100a uses the same or fewer registers on every entry (the largest drops: the hyper-connection kernels 48 -> 32, `quantize_mmq_nvfp4` 40 -> 24). None of the decode kernels is occupancy-limited on either part; the grid shapes were chosen for 48 SMs and are the retune target of #1141 (148 SMs: the 320-block `wq_a` and 128-block `wkv` launches fill 0.67 and 0.27 waves on GB10 and less than a quarter of a wave on B200).

## What a B200 changes and what it does not

The V4.1 decode step on one GB10 after #1155 is 42.4 ms of wall for 42.8 ms of GPU work: 1,865 launches and 186 host waits a token, and 6.09 GB of weights read (2.93 GB routed experts from the 100 GiB arena, the attention projections, the engram rows, the Q6_K head). At 8 TB/s the same bytes cost ~0.8 ms; what remains is the launch and wait overhead, ~8-12 ms of it, which is why the levers for this part are the device-side routing and the whole-step CUDA graph (S2/S3 of the same PR) and not the kernels above. The first hour on a rented B200 is written up in `docs/perf/B200_FIRST_HOUR_2026-09.md`.
