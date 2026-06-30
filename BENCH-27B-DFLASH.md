# DFlash Benchmark — Qwen3.6-27B NVFP4

**Железо:** NVIDIA GB10 (Grace Blackwell), 121 GB unified RAM  
**Модель:** Qwen3.6-27B-NVFP4 (target) + z-lab/Qwen3.6-27B-DFlash (drafter, BF16)  
**Промпт:** MinHeap (Write a Python implementation of a min-heap...)  
**Метод:** 1 warmup + 2 runs, median  
**† — высокая дисперсия total_tok между runs (>20%)**

---

## K (ATLAS_DFLASH_DRAFT_CAP = K−1)

| Config | T | K | window | SSM path | tok/s | output tok/s | TTFT_ms | total_tok | think_tok | out_tok | accept/step | accept_% | verify_ms | propose_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No-MTP baseline | 0 | — | — | — | 13.6 | 13.6 | 237 | 1351 | 0 | 1351 | — | — | — | — |
| MTP fp8 head | 0 | 2 | — | — | 15.6 | 15.6 | 245 | 1351 | 0 | 1351 | 0.48 | 48.4% | 79.8 | N/A |
| DFlash | 0 | 2 | 512 | WY2 | 8.6 | 6.3 | 326 | 1395 | 694 | 701 | 0.06 | 5.8% | 77 | 95 |
| DFlash | 0.6 | 2 | 512 | WY2 | 8.6 | 6.4 | 327 | 1368 | 668 | 701 | 0.08 | 7.9% | 77 | 113 |
| DFlash | 1.0 | 2 | 512 | WY2 | 8.7 | 6.5 | 709† | 1364 | 663 | 701 | 0.10 | 10.2% | 77 | 113 |
| DFlash | 0 | 4 | 512 | WY4 | 4.9 | 3.0 | 324 | 1395 | 694 | 701 | 0.05 | 4.8% | 257 | 98 |
| DFlash | 0.6 | 4 | 512 | WY4 | 6.1 | 6.1 | 328 | 1138† | 0* | 1138 | 0.06 | 2.0% | 257 | 79 |
| DFlash | 1.0 | 4 | 512 | WY4 | 7.1 | 7.1 | 327 | 961 | 0* | 961 | 0.10 | 3.3% | 257 | 79 |
| DFlash | 0 | 8 | 512 | prefill_ssm | 4.3 | 4.3 | 329 | 1395 | 694 | 701 | 0.17 | 2.4% | 320 | 112 |
| DFlash | 0.6 | 8 | 512 | prefill_ssm | 4.1 | 4.1 | 331 | 1376 | 675 | 701 | 0.12 | 1.7% | 320 | 112 |
| DFlash | 1.0 | 8 | 512 | prefill_ssm | 6.1 | 6.1 | 328 | 1118† | 694* | 424 | 0.14 | 2.0% | 320 | 112 |
| DFlash | 0 | 16 | 512 | prefill_ssm | 2.9 | 2.9 | 335 | 1395 | 694 | 701 | 0.17 | 1.1% | 585 | 113 |
| DFlash | 0.6 | 16 | 512 | prefill_ssm | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DFlash | 1.0 | 16 | 512 | prefill_ssm | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## TODO: будущие бенчи

- [ ] **DFlash с включённым thinking** — сейчас DFlash отключается при `inside_thinking` (`mod.rs:340`). Нужно снять этот guard и померять acceptance и tok/s на thinking фазе отдельно. Если drafter умеет предсказывать thinking токены — потенциально большой выигрыш (thinking chain может быть 1000+ токенов).

---

## vLLM aeon reference (2026-06-29)

| Config | T | tok/s | accept_% | τ | Notes |
|---|---|---|---|---|---|
| vLLM DFlash, no thinking | 0 | 43 | 27.2% | 4.09 | 3 req × 139 tok, NVFP4 |
| vLLM DFlash, with thinking | 0 | 43 | 28.5% | 4.28 | 5 req × 1500 tok |
