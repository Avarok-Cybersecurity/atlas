# Hopper Kimi K3 target

Keep-packed Unsloth UD-Q2_K_XL at TP=8 is the path this tree serves on
H200. Official `moonshotai/Kimi-K3` MXFP4 is a different file (below).

Build with `AVAROK_TARGET_HW=hopper AVAROK_TARGET_MODEL=kimi-k3
AVAROK_TARGET_QUANT=bf16 CARGO_TARGET_DIR=target/k3-hopper cargo build
--locked --release -p spark-server --features nccl`. Selects `sm_90a`.
Do not reuse Spark (`sm_121`) or B200 (`sm_100a`) binaries.

No `nvfp4/` tree. Hopper has no NVFP4 datapath. The 0.40B twin on GB10
was runtime-quantized to NVFP4; on H200 it stays BF16.

KDA/MLA/E8M0 sources are Hopper-owned copies of the B200 stems. mxfp4
aliases stay inside this directory. Common sources keep the Hopper
mirror.

First cell: packed 0.40B twin on one GPU, then TP across the eight
local devices. Official `moonshotai/Kimi-K3` MXFP4 is about 214.6 GiB
per TP8 rank and does not fit 141 GB H200 resident. Do not start that
download until a reviewed non-resident expert path exists.
`K3_EXPERT_BACKEND=mmap` is dummy `K3E1` packs, not 96-shard
safetensors.
