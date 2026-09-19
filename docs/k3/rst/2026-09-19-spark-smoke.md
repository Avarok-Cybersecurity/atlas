# K3 branch smoke checks, 2026-09-19

These are development checks, not benchmark certification or proof of full K3 on B300.
Source: main `4c00f65a2` plus notebook commits `6d8f91211` and `2f5db77ba`.
The runtime build preceded two test-only compatibility fixes in the latter commit;
the serving code was identical. The source was copied into isolated lab directories.

## Reproduction and identity

Hardware: two idle DGX Sparks, GB10/ARM64, CUDA compiler 13.0.88, Rust 1.93.1.
Existing Kimi-K3-0.40B checkpoint used; no full K3 download.
SHA256:

- Server binary on both ranks: `3d20126799d7d8c5f9400da8033291f73df64a1efba834f58d1e362208c5d473`
- Checkpoint config: `503bd5d1589ea8fd21b82528eb2ab130a593b171e1b61bbb75a90014581599d2`
- `model.safetensors`: `a39b7ed2769ee2f9891a19cef6f1ca7986a3f295c09759b5b3135285e5d7a678`

Built with the GB10/BF16 command in [README](../README.md). Single-rank serving:

```bash
spark serve --model-from-path /models/Kimi-K3-0.40B --model-name kimi-k3-twin \
  --bind 127.0.0.1 --port 18888 --max-seq-len 512 --kv-cache-dtype bf16
```

TP2 used the same binary and checkpoint on both machines, adding
`--tp-size 2 --world-size 2 --rank <0-or-1> --master-addr <rank-0-link-IP>`
`--master-port 29619 --max-batch-size 1`, with HTTP port 18889.
Lab NIC was `enp1s0f1np1`, HCA `rocep1s0f1`; rediscover these on another machine.
Lab NCCL settings: `NCCL_SOCKET_IFNAME=enp1s0f1np1`, `NCCL_IB_HCA=rocep1s0f1`,
`NCCL_NVLS_ENABLE=0`, `NCCL_NET_GDR_LEVEL=0`, `NCCL_NET_GDR_C2C=0`,
`NCCL_DMABUF_ENABLE=0`, `NCCL_PROTO=Simple`, `NCCL_ALGO=Ring`.
These settings are **not** a recommended B300 NVSwitch configuration.

## Generation and negative control

Prompt: `According to all known laws of aviation,`, temperature zero, 16 output tokens.
TP1 and TP2, including repeat requests in the same process, returned exactly:

> there is no way a bee should be able to fly. Its wings are too

Both reported 8 input / 16 output tokens and `finish_reason=length`.
The first probe retained its failed receipt after an incorrect expected
leading space; the strict comparison caught that fixture mismatch. The corrected
prefix matched the response. No throughput claim is made from this tiny canary.

For the negative control, the owned worker was terminated, then generation was
requested from the head. The probe returned failure at its 8-second deadline
(8.011 seconds observed), rather than claiming healthy inference. This validates
client containment, **not** automatic server recovery: the head was then stopped.
Both GPUs were checked free of compute processes afterward.

## Build and test checks

- Full Linux `spark-server` compile and GB10 release CUDA build passed.
- B300 MXFP4 target: 182 unique CUDA kernel invocations passed for `sm_103a`;
  host `cargo check -p avarok-kernels` finished in 29.80 seconds. No B300 executed.
- K3 weight-loader tests: 22 passed, including TP2 sharding and packed TP8 refusal.
- Core tests without a checkpoint: 90 passed / 11 explicitly ignored. Missing-checkpoint
  negative control fails when an ignored checkpoint test is explicitly selected.
- Kernel crate: 114 passed / 7 GPU tests ignored; architecture tests: 16 passed.
- Python checkpoint/probe suite: 13 passed, including fake endpoint timeout and
  corruption/path/identity/deadline failures.
- Formatting and cross-hardware reach check passed; shadow checker passed for seven
  hardware trees. Two trailing blank-line warnings are inherited verbatim in the
  provenance-pinned B300 snapshots; no other hardware source was edited.

The long CPU golden suite was first stopped during an unoptimized run after the
first-token test passed, then rerun with `cargo test --release` under a 600-second
limit. Final result is recorded below. Bulk build/server logs remain lab artifacts,
not benchmark records committed into this notebook.

Final checkpoint suite: **11 passed / 0 failed / 0 ignored**, 71.65 seconds in
release mode. This includes the eight-prompt golden comparison through EOS,
prefill/decode agreement, prefix/hybrid state checks, and the deliberately wrong
math/router controls. Core and model Clippy checks passed with warnings denied.

License check: the repository's Docker-based checker ran on Spark2 and failed on
14 unchanged vendored Hopper `q4k_vendor` headers. No reported path belongs to
this change; every changed Rust/CUDA source has the required first-line SPDX.
The same checker could not start locally because the Mac Docker daemon was off.
