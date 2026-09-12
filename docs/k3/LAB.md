# Lab inventory — Atlas K3

Filled 2026-09-11 from the live boxes. Do not copy IPs or iface names from other Spark labs or from `scripts/start-ep2.sh` comments (`enp1s0f0np0` is **Down** here).

## Hosts

| hostname | IP | role | API | notes |
| --- | --- | --- | --- | --- |
| spark1 | 192.168.50.125 (enP7s7 Ethernet) | head / rank 0 | :8888 | NCCL master-addr. Wi-Fi 192.168.50.80 (`wlP9s9`). Tailscale 100.110.52.54. User `pidtom`. GB10 SM121, 121 GiB UMA. |
| spark2 | 192.168.50.36 (enP7s7 Ethernet) | worker / rank 1 | :8889 | No public client API. Wi-Fi 192.168.50.23. User `pidtom`. Same class box. |
| train (5090 workstation) | 192.168.50.122 | correctness GPU | local | Windows host + WSL Ubuntu (`tturn`). RTX 5090 SM120, 32607 MiB. SSH `train.local`. |

SSH from the Mac:

```
Host gx10 spark spark1   # 192.168.50.125 pidtom
Host spark2              # 192.168.50.36  pidtom
```

Passwordless SSH spark1 ↔ spark2: **yes** (verified 2026-09-11: `ssh spark1` → `spark2`; `ssh spark2` → `spark1`).

## Fabric

- QSFP cable: spark1 `enp1s0f1np1` ↔ spark2 `enp1s0f1np1` (primary RoCE rail)
- Second rail also Up (unused for NCCL pin): spark1 `enP2p1s0f1np1` = 192.168.101.10, spark2 `enP2p1s0f1np1` = 192.168.101.11

`ibdev2netdev` (spark1), 2026-09-11:

```
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

`ibdev2netdev` (spark2): identical Up/Down pattern.

- Up RoCE iface (pin `NCCL_SOCKET_IFNAME`): **`enp1s0f1np1`**
- `NCCL_IB_HCA` pin: **`rocep1s0f1`**
- RoCE IPs: spark1 `192.168.100.10/24`, spark2 `192.168.100.11/24`
- ICMP over RoCE (2026-09-11): spark1→spark2 rtt 0.8–1.4 ms; spark2→spark1 rtt 0.4–1.1 ms; 0% loss
- NCCL 2-rank all-reduce **re-run 2026-09-11** (this campaign): `docs/k3/logs/nccl-2rank-*-2026-09-11.log`
  - Transport: `NET/IB` RoCE, `Connected all rings, use ring PXN 0 GDR 0`
  - 16 KiB: 30.39 µs, 0.539 GB/s
  - 64 KiB: 35.86 µs, 1.827 GB/s
  - 1 MiB: 121.43 µs, **8.635 GB/s**
  - 2026-09-03 prior (not C7): 16 KiB 25.28 µs / 1 MiB 6.060 GB/s

K3-LAB: `scripts/start-ep2.sh` still documents `enp1s0f0np0`. That iface is Down on these boxes. Unpinned NCCL will pick a dead HCA. Pin the Up twin.

Working NCCL env (confirmed against the 2026-09-03 RoCE all-reduce on this pair; freeze unless a later run disagrees):

```
NCCL_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_HCA=rocep1s0f1
NCCL_NVLS_ENABLE=0
NCCL_NET_GDR_LEVEL=0
NCCL_NET_GDR_C2C=0
NCCL_DMABUF_ENABLE=0
NCCL_PROTO=Simple
NCCL_ALGO=Ring
```

Dual-node launcher pattern:

```bash
HEAD_IP=192.168.50.125 WORKER_IP=192.168.50.36 \
  NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1 \
  bash scripts/start-ep2.sh <already-shipping-Atlas-MoE>
```

Use Ethernet mgmt IPs for SSH/HTTP. Use `192.168.100.10` as NCCL master-addr if the runtime binds the RoCE iface; do not mix mgmt and RoCE in the same pin.

Prove the fabric with a shipped Atlas MoE before any K3 dummy TP.

## Disk / mem snapshot (2026-09-11, after reclaim)

| host | disk before → after | notes |
| --- | --- | --- |
| spark1 | 43 GiB free (96%) → 104 GiB free (89%) | `/dev/shm` was 57/61 GiB of leftover Iron packets (Sep 7–8); swap was 16/16 GiB. Cleared shm + `/tmp/iron-wt-*` + cargo targets. Models kept (`DeepSeek-V4-Flash-0731` 156G, `dsv4-flash-vision-exl3-mixedk` 95G, `gguf/` 268G). |
| spark2 | 64 GiB free (93%) → 223 GiB free (75%) | Removed cargo `target/` under `work/`, butter, fable. Models kept (DSV4 159G, `gguf/vision-mxfp4` 152G). No Atlas serve at probe time. |
| train WSL | 5.8 GiB free (97%) → 76 GiB free (60%) | Deleted dated `iron-*-20260907/08/10` experiment trees only. Models kept. llama-swap on `:8080` was down. |

Reclaimable later (not touched): spark1 `gguf/vision-exp` 88G + `gguf/vision-mxfp4-tp2` 74G; spark2 `gguf/vision-mxfp4` 152G; train leftover `iron-*` ~9.3G.

## 5090 SM121 launch test (S1 hour 1)

- Date: 2026-09-11
- Host: train WSL, nvcc 13.0.88 (`/home/tturn/cuda/bin/nvcc`), device `sm_120`
- Source: trivial `__global__ void k()`
- Command:
  - cubin: `nvcc -arch=sm_121` → run
  - PTX: `nvcc -gencode=arch=compute_121,code=compute_121` → run
  - control: `nvcc -arch=sm_120` → run
- Result: **CANNOT** launch Atlas-class SM121 images on the 5090.
  - `sm_121` cubin: `launch: no kernel image is available for execution on the device`
  - `compute_121` PTX: same `no kernel image` (no JIT to sm_120)
  - `sm_120` cubin control: `launch: no error` / `sync: no error`
- Consequence: **5090 is PyTorch-and-shape-debug only.** All GPU kernel milestones stay on spark1/spark2. Do not plan `kernels/rtx5090` for this PRD.

## Constraint

spark1 + spark2 ≈ 240 GB UMA (121 GiB + 121 GiB reported).
Official `moonshotai/Kimi-K3` MXFP4 ≈ 1.561 TB across 96 shards.
Two Sparks cannot load official K3. There is no full-model-on-lab milestone.
