# TP4 full-shape composition preflight

This is a bounded synthetic integration test, not a full Kimi K3 deployment.
It exercises `forward_one_layer_with_cores` with the current CUDA callbacks;
it does not exercise checkpoint loading, `K3BoundLayer` allocation, NCCL,
tokenization, all 93 layers, or answer quality.

## Shape provenance

Dimensions come from the pinned official text configuration in
`fixtures/moonshotai-Kimi-K3-config.json`, also audited against the official
checkpoint headers in `evidence/official-header-audit-20260919/`:

| Dimension | Full model | TP4 rank | TP8 rank |
|---|---:|---:|---:|
| Hidden width | 7,168 | 7,168 | 7,168 |
| KDA and MLA heads | 96 | 24 | 12 |
| KDA head width / convolution taps | 128 / 4 | 128 / 4 | 128 / 4 |
| MLA QK / value width | 192 / 128 | 192 / 128 | 192 / 128 |
| Query / KV latent rank | 1,536 / 512 | 1,536 / 512 | 1,536 / 512 |
| First dense intermediate | 33,792 | 8,448 | 4,224 |
| Routed latent width | 3,584 | 3,584 | 3,584 |
| Expert intermediate | 3,072 | 768 | 384 |
| Shared intermediate (replicated) | 6,144 | 6,144 | 6,144 |
| Router census / selected experts | 896 / 16 | 896 / 16 | 896 / 16 |

The fixture asserts these values before allocating. It constructs one
KDA+dense layer and one MLA+LatentMoE/shared layer. Matrices have full physical
sizes, with deterministic sparse values to avoid unstable random depth.
The real 896-way router selects 16 explicitly provisioned packed experts;
there are no weights for unselected experts. It runs two cached tokens,
repeats them after reset, then runs the second token with cleared history.
Exact reset replay, nonzero finite outputs, MLA cache advancement, and a
nonzero cached-versus-cleared difference are required. This is composition
and lifecycle coverage; independent kernel arithmetic oracles are separate.

The calculated host matrix payload is 2,210,299,904 bytes (2.059 GiB), with a
3 GiB assertion before allocation. Persistent uploaded BF16 dense/shared and
selected MXFP4 expert weights total 697,761,792 bytes (0.650 GiB), bounded by
1 GiB in the allocation helper. KDA state, small vectors, upload temporaries,
wrapper scratch, context and allocator overhead are additional. These are
payload estimates, not measured process RSS or GPU high-water. The test prints
actual tracked persistent allocation bytes, build time, graph time and
aggregate callback wall times. No performance extrapolation is implied.

Run only with an explicitly reserved GPU, matching B200 K3 build variables,
and an external deadline:

```sh
K3_ORACLE_GPU_ORDINAL=2 timeout 180s cargo test --release -p spark-model \
  --test k3_tp4_composed_cuda -- --ignored --nocapture --test-threads=1
```

## Remaining production host work

At the audited source, `host_decode.rs` uses resident KDA recurrence but still
calls the host-KV MLA wrapper. The existing resident MLA helper is not wired
into this path. Each MLA token therefore uploads the entire expanded history.
`K3_CUDA_DENSE=1` explicitly enables dense/shared CUDA; its default remains off.
Even when enabled, lazy `host_layer` binding still materializes non-packed
layer weights as host FP32.

KDA Q/K/V, gate projections and output projection remain CPU matrix-vector
operations. MLA projections and KV expansion remain on CPU. AttnRes, norms,
router/top-k, latent down/up projections, expert SiTU and weighted reduction
also remain on CPU. Packed expert gate/up are batched, but their output is
downloaded for SiTU, uploaded for expert down, then downloaded for weighted
mixing. Buffers and pointer tables are allocated per call. BF16 TP all-reduce
also round-trips through the host. This path refuses graph capture,
multi-sequence decode and speculative rollback.

## Estimates from current layouts

These are arithmetic estimates, **not observed transfer counters or timings**.
Let P be TP size, h=96/P, D=128, L=3584, I=3072/P, K=16, and T the live context.
There are 69 KDA and 24 MLA layers, with 92 routed MoE layers.

- Resident KDA state: `69*4*(h*D*D + 3*h*D*4)` bytes: 113.20 MiB/rank
  at TP4, 56.60 MiB at TP8. Host oracle state is retained separately.
- KDA per-layer H2D is `4*(3*h*D + 12*h*D + h*D + h)` bytes: 192.09 KiB
  at TP4, 96.05 KiB at TP8. This includes repeated convolution weights.
  Output D2H is `4*h*D`: 12 or 6 KiB. Recurrence is no longer copied each step.
- MLA full-history H2D across 24 layers is `24*4*h*(192+128)*T` bytes/token.
  At context 4,096 this is **2.8125 GiB/rank/token at TP4**, 1.40625 GiB at
  TP8. At 32,768 it is 22.5 and 11.25 GiB, respectively. Additional projected
  activation transfers are excluded. Resident MLA integration removes this
  history-dependent host upload without requiring a new attention kernel.
- Packed expert activation transfers, excluding pointer tables, are
  `2*(2*K*L + K*I)` H2D and `2*(2*K*I + K*L)` D2H bytes/layer. Across 92
  layers: 36.66 MiB/rank/token at TP4, 33.42 MiB at TP8. Allocation and
  synchronization latency are additional.
- With CUDA dense/shared enabled, remaining CPU projection matrices contain
  approximately 14.68 billion MACs/rank/token at TP4 and 10.21 billion at TP8
  (KDA + MLA + router/latent down/up only). Reading their FP32 weights once
  means 58.73 GB or 40.85 GB per rank/token before other CPU work. This is a
  traffic model, not a throughput forecast.

Official-header text weight payload is S=1,559,965,606,912 bytes, of which
R=40,505,956,864 bytes is replicated. Verified storage-only A_log padding is
69*32*4=8,832 bytes. The current rank payload is
`R + (S-R-8832)/P`: **391.50 GiB at TP4**, **214.61 GiB at TP8**, before
staging, caches, context or runtime reserves. The whole official checkpoint
therefore does not fit on four B200s. The TP8 header audit additionally reports
83.69 GiB of lazily materialized host FP32 layer weights per rank. Passing this
small selected-expert fixture does not establish complete-model admission.


## B200 execution result

The bounded test passed on B200: five tokens through the two composed layers,
exact reset replay, cleared-history maximum difference 0.06444836, and all
five CUDA callback categories executed. The test took 4.74 seconds total;
fixture construction was 0.771 seconds and the five-token graph was 0.369
seconds. These synthetic sparse-weight times are not a full-model throughput
prediction. The [compact receipt](evidence/tp4-composition-20260919.json)
records source/binary hashes, allocation payloads and callback times.
