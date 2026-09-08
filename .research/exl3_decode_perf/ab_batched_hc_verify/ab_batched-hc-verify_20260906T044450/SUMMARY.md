# A/B — cross-sequence batched mHC verify — /home/ms/atlas/.claude/worktrees/wf_a8fa8242-651-7/.research/exl3_decode_perf/ab_batched-hc-verify_20260906T044450

FINGERPRINT ab_batched-hc-verify bin=/home/ms/atlas/.claude/worktrees/wf_a8fa8242-651-7/target/release/spark sha256=aefac27e8c49bb3b git=9e81c4d06 branch=perf/batched-hc-verify dirty=14 date=2026-09-06T08:44:51Z host=dgx-00 port=8899 preset=qwen3.8-flash-next-exl3 arms=[serial hcbatch hcbatch serial] salt=4242

## Liveness (an arm without its lines is inert and is NOT a measurement)

```
serial_1 live=yes boot_lines=1 gate_lines=1 multi_lines=0 declined=0 verify_errors=0 rc=0
hcbatch_1 live=yes boot_lines=1 gate_lines=1 multi_lines=1 declined=0 verify_errors=0 rc=0
hcbatch_2 live=yes boot_lines=1 gate_lines=1 multi_lines=1 declined=0 verify_errors=0 rc=0
serial_2 live=yes boot_lines=1 gate_lines=1 multi_lines=0 declined=0 verify_errors=0 rc=0
```

## Short arm (2000-token prompts, 300 tokens, 3 repeats; medians over repeats)

| boot | C | median per-stream tok/s (reps) | aggregate decode tok/s (reps) | TPOT p50 proxy ms | TTFT s (rep0) | ok | min avail GB |
|---|---:|---|---|---:|---|---|---:|
| serial_1 | 1 | 27.13 [27.13, 27.37, 26.74] | 27.22 [27.22, 27.46, 26.83] | 36.9 | [3.7] | 1/1/1/1/1/1 | 22.1 |
| serial_1 | 2 | 13.09 [12.78, 13.09, 13.15] | 24.07 [23.65, 24.07, 24.32] | 76.4 | [6.9, 3.5] | 2/2/2/2/2/2 | 21.4 |
| serial_1 | 4 | 6.14 [6.14, 6.14, 6.3] | 22.15 [21.82, 22.15, 22.43] | 162.9 | [3.5, 13.9, 7.0, 10.5] | 4/4/4/4/4/4 | 20.1 |
| hcbatch_1 | 1 | 27.15 [27.15, 27.48, 26.73] | 27.24 [27.24, 27.57, 26.82] | 36.8 | [3.7] | 1/1/1/1/1/1 | 22.2 |
| hcbatch_1 | 2 | 9.46 [9.46, 9.0, 9.66] | 16.90 [16.9, 16.25, 17.29] | 105.7 | [3.5, 7.0] | 2/2/2/2/2/2 | 21.4 |
| hcbatch_1 | 4 | 4.27 [4.24, 4.27, 4.42] | 14.67 [14.61, 14.67, 15.05] | 234.2 | [7.0, 3.5, 10.4, 13.8] | 4/4/4/4/4/4 | 20.1 |
| hcbatch_2 | 1 | 27.16 [27.16, 27.44, 26.75] | 27.26 [27.26, 27.53, 26.84] | 36.8 | [3.7] | 1/1/1/1/1/1 | 22.2 |
| hcbatch_2 | 2 | 10.50 [9.47, 10.5, 10.61] | 18.20 [16.89, 18.2, 18.88] | 95.2 | [3.5, 7.0] | 2/2/2/2/2/2 | 21.5 |
| hcbatch_2 | 4 | 4.30 [4.3, 4.2, 4.32] | 14.79 [14.79, 14.62, 14.92] | 232.6 | [10.4, 13.9, 3.4, 6.9] | 4/4/4/4/4/4 | 20.2 |
| serial_2 | 1 | 27.15 [27.15, 27.5, 26.72] | 27.24 [27.24, 27.59, 26.81] | 36.8 | [3.6] | 1/1/1/1/1/1 | 23.1 |
| serial_2 | 2 | 13.07 [12.89, 13.07, 13.08] | 23.88 [23.7, 23.96, 23.88] | 76.5 | [3.5, 6.9] | 2/2/2/2/2/2 | 22.4 |
| serial_2 | 4 | 6.20 [6.14, 6.23, 6.2] | 22.34 [21.86, 22.34, 22.38] | 161.3 | [3.4, 13.7, 10.3, 6.9] | 4/4/4/4/4/4 | 21.1 |

## Warm cell (512-token prompts, 128 tokens; pass 2 = prefix-cached)

| boot | pass | C | median per-stream tok/s | aggregate decode tok/s | TTFT s | prompt tok |
|---|---:|---:|---:|---:|---|---|
| serial_1 | 1 | 1 | 28.6 | 28.83 | [1.1] | [548] |
| serial_1 | 1 | 2 | 14.12 | 26.09 | [1.1, 2.2] | [552] |
| serial_1 | 1 | 4 | 6.84 | 24.96 | [1.1, 2.3, 3.4, 4.5] | [553] |
| serial_1 | 2 | 1 | 28.71 | 28.93 | [0.2] | [548] |
| serial_1 | 2 | 2 | 14.83 | 28.59 | [0.2, 0.5] | [552] |
| serial_1 | 2 | 4 | 7.38 | 28.76 | [0.2, 0.5, 0.7, 0.9] | [553] |
| hcbatch_1 | 1 | 1 | 28.54 | 28.76 | [1.1] | [548] |
| hcbatch_1 | 1 | 2 | 11.54 | 20.31 | [2.2, 1.1] | [552] |
| hcbatch_1 | 1 | 4 | 4.22 | 15.32 | [1.1, 2.3, 3.4, 4.5] | [553] |
| hcbatch_1 | 2 | 1 | 28.63 | 28.86 | [0.2] | [548] |
| hcbatch_1 | 2 | 2 | 11.21 | 20.9 | [0.2, 0.5] | [552] |
| hcbatch_1 | 2 | 4 | 4.3 | 16.75 | [0.2, 0.5, 0.7, 0.9] | [553] |
| hcbatch_2 | 1 | 1 | 28.07 | 28.29 | [1.1] | [548] |
| hcbatch_2 | 1 | 2 | 11.37 | 20.01 | [2.3, 1.1] | [552] |
| hcbatch_2 | 1 | 4 | 4.55 | 16.04 | [2.3, 1.1, 3.6, 4.8] | [553] |
| hcbatch_2 | 2 | 1 | 28.71 | 28.94 | [0.2] | [548] |
| hcbatch_2 | 2 | 2 | 11.22 | 20.91 | [0.2, 0.5] | [552] |
| hcbatch_2 | 2 | 4 | 5.1 | 18.49 | [0.4, 0.7, 0.2, 1.0] | [553] |
| serial_2 | 1 | 1 | 28.52 | 28.74 | [1.1] | [548] |
| serial_2 | 1 | 2 | 13.93 | 25.89 | [2.2, 1.1] | [552] |
| serial_2 | 1 | 4 | 6.81 | 24.95 | [2.3, 1.1, 3.4, 4.5] | [553] |
| serial_2 | 2 | 1 | 28.61 | 28.83 | [0.2] | [548] |
| serial_2 | 2 | 2 | 14.68 | 28.51 | [0.2, 0.5] | [552] |
| serial_2 | 2 | 4 | 7.38 | 28.59 | [0.2, 0.7, 1.0, 0.4] | [553] |

## Acceptance (ATLAS_MTP_ACCEPT_DEBUG flushes; serial records n=1 only, batched arms n=chunk width)

Buckets are POOLED over the whole boot: a serial boot's n=1 row mixes its C=1, C=2 and C=4 cells (every
serial finish records batch_n=1), while a batched boot's n=2 / n=4 rows come from the C=2 / C=4 cells only.
Criterion 3 is therefore serial n=1 (pooled) vs batched n=2 / n=4 — valid because a serial verify is a
single-sequence forward whose acceptance does not depend on C; per-cell windowing is a follow-up.

| boot | bucket n | k_drafts | verifies | mean_na (verify-weighted) | p1 | flushes |
|---|---:|---:|---:|---:|---:|---:|
| serial_1 | 1 | 2 | 3328 | 1.363 | 0.809 | 26 |
| hcbatch_1 | 1 | 2 | 640 | 1.319 | 0.794 | 5 |
| hcbatch_1 | 2 | 2 | 1408 | 0.413 | 0.276 | 11 |
| hcbatch_1 | 3 | 2 | 768 | 0.291 | 0.204 | 6 |
| hcbatch_1 | 4 | 2 | 2432 | 0.260 | 0.169 | 19 |
| hcbatch_2 | 1 | 2 | 640 | 1.312 | 0.789 | 5 |
| hcbatch_2 | 2 | 2 | 1280 | 0.505 | 0.318 | 10 |
| hcbatch_2 | 3 | 2 | 640 | 0.362 | 0.253 | 5 |
| hcbatch_2 | 4 | 2 | 2560 | 0.263 | 0.175 | 20 |
| serial_2 | 1 | 2 | 3328 | 1.365 | 0.812 | 26 |

## Text identity (per-stream content_sha256, same salt => same prompt)

Lists are in PROMPT order (measure_concurrency.py sorts by stream index; older JSONs without a `streams`
key are compared as sorted multisets). The serial-vs-serial spread is the CONTROL: criterion 4 is only
evaluated where the control is itself IDENTICAL, otherwise the cell is nondeterministic on the serial
path too and a DIFFERS is not attributable to the lever.

- C=1 rep=0: serial-vs-serial IDENTICAL; batched-vs-serial IDENTICAL to serial [prompt-ordered]
- C=1 rep=1: serial-vs-serial IDENTICAL; batched-vs-serial IDENTICAL to serial [prompt-ordered]
- C=1 rep=2: serial-vs-serial IDENTICAL; batched-vs-serial IDENTICAL to serial [prompt-ordered]
- C=2 rep=0: serial-vs-serial DIFFERS (2 variants); batched-vs-serial NOT EVALUATED (serial control not identical) [prompt-ordered]
    - serial_1: ['d870ac11a5957b27', '2e9bd00d5e9f4e85']
    - hcbatch_1: ['d870ac11a5957b27', '0c8190875799ebda']
    - hcbatch_2: ['d870ac11a5957b27', '0c8190875799ebda']
    - serial_2: ['d870ac11a5957b27', 'b77a386863a5ad91']
- C=2 rep=1: serial-vs-serial DIFFERS (2 variants); batched-vs-serial NOT EVALUATED (serial control not identical) [prompt-ordered]
    - serial_1: ['6c9fd9fde7604364', 'f3758d6abfe78627']
    - hcbatch_1: ['b4abbe2cb6c942c9', '65f427b3d55dfd66']
    - hcbatch_2: ['cbde4552609fa92a', '58a1631b53050d11']
    - serial_2: ['baa02ccb31a678d6', '25ea5715d7c0c844']
- C=2 rep=2: serial-vs-serial DIFFERS (2 variants); batched-vs-serial NOT EVALUATED (serial control not identical) [prompt-ordered]
    - serial_1: ['b8d965885ba101ab', '3d73c8bf05874dfd']
    - hcbatch_1: ['b51989400f1f4135', '79371524af6f7979']
    - hcbatch_2: ['67f8a301cd383f5f', '51a35174ee677b94']
    - serial_2: ['b6331d37da69e8bb', 'ffcc0f500f8f21b1']
- C=4 rep=0: serial-vs-serial IDENTICAL; batched-vs-serial DIFFERS from serial (3 variants) [prompt-ordered]
    - serial_1: ['449b01318330dbdf', '886800f042d393c5', '61f65ff02b7112ad', 'fbb3a11fbd7f558d']
    - hcbatch_1: ['e5a5d5be3c0175fd', '991a1e25bf6ce66a', '8c8ba8f87d99456b', '8265fa10b69d8fff']
    - hcbatch_2: ['e90712a36f385248', 'd0f25ea6ad88bf85', 'cb67bf6e047825b3', '3f4b6da64a09a72e']
    - serial_2: ['449b01318330dbdf', '886800f042d393c5', '61f65ff02b7112ad', 'fbb3a11fbd7f558d']
- C=4 rep=1: serial-vs-serial DIFFERS (2 variants); batched-vs-serial NOT EVALUATED (serial control not identical) [prompt-ordered]
    - serial_1: ['ece0943e4290fc4d', 'b54a05364226b684', 'da1f0a14c24a03e2', '84966e98e3312580']
    - hcbatch_1: ['25598f49325153e9', '57cd12c975c5855b', '40a7f41e6416a35b', '850e9f18854e621e']
    - hcbatch_2: ['3ec31d1d2492ac4d', '9ef73c25a739fb60', 'da1f0a14c24a03e2', 'dfed3d14cc89471e']
    - serial_2: ['293a1a49ab21781a', '7cd6d37942a197cb', '733c460f644bb1a0', '2fbe842b4446e724']
- C=4 rep=2: serial-vs-serial DIFFERS (2 variants); batched-vs-serial NOT EVALUATED (serial control not identical) [prompt-ordered]
    - serial_1: ['32f600299e7e7ba9', '704b2f56f0b3649b', '7fb128893c76b6e5', 'c29a8b2693057e01']
    - hcbatch_1: ['77e88bebee72e527', '31b03dfdfd8a0914', '1248b293dc64ac53', '56dda5d5f38f0693']
    - hcbatch_2: ['4376d34344601790', 'e02456d599199839', '08eb4053aced5d96', 'eef5f1ee203a0bd8']
    - serial_2: ['2aafad355ce3a8d0', '923e4e3cd50d36e0', '50d97169f60e0312', 'd4831c95146d76f6']

Pass criteria live in EXL3_DECODE_PERF.md ('Cross-sequence batched mHC verify'): C=1 identical + within 3%;
C=2/4 aggregate >= 1.15x serial with acceptance within 0.05; no verify errors; watchdog never tripped.
