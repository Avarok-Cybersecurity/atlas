# B200 Kimi K3 rehearsal target

Build with `AVAROK_TARGET_HW=b200 AVAROK_TARGET_MODEL=kimi-k3
AVAROK_TARGET_QUANT=mxfp4 CARGO_TARGET_DIR=target/k3-b200 cargo build
--locked --release -p spark-server --features nccl` (one shell command).
The B200 hardware declaration selects `sm_100a` and disables unsupported
warp-level blockscaled MMA. Do not reuse B300 or Spark binaries.

KDA and MLA sources are B200-owned copies of the B300 bring-up sources at
`32ebc80d5`; edits here do not change B300 or GB10. Quant aliases stay within
B200. E8M0 is also a B200-owned byte-identical copy of the existing DeepSeek
dependency; quant aliases resolve to that local copy. Common sources retain
the existing B200 mirror.

Start with the packed 0.40B twin on one GPU, compare two-GPU output with that
control, then reserve all four GPUs for TP4. Its eight attention/KDA heads,
1024 hidden width, 256 expert intermediate and 163840 vocabulary divide by
four; the local expert reduction width remains aligned to 32-value MXFP4
blocks. Divisibility is admission evidence, not proof of correct execution.
Run production-shape numerical fixtures separately. EP remains one.

On the rehearsal B200, all 182 selected kernels compiled with CUDA 13.0.48.
The production TP8-shape KDA/MLA numerical tests passed, including nonzero
history, restored continuation and negative controls. E8M0 expert GEMMs
matched independent CPU references exactly after BF16 rounding for w1/w2/w3
and irregular shapes. These are correctness tests, not throughput certification
or full-model inference. Small-model serving results are recorded separately.
The dimensions in MODEL.toml are descriptive metadata; runtime admission uses
the actual checkpoint config and device inventory.
