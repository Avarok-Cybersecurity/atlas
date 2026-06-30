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

## vLLM aeon reference (2026-06-29) — ⚠️ AGGREGATE, не per-request

| Config | T | tok/s | accept_% | τ | Notes |
|---|---|---|---|---|---|
| vLLM DFlash, no thinking | 0 | 43 | 27.2% | 4.09 | 3 req × 139 tok, NVFP4 — **43 = СУММА 3 concurrent req** (`serve.py:584 output_throughput`) |
| vLLM DFlash, with thinking | 0 | 43 | 28.5% | 4.28 | 5 req × 1500 tok |

Эти 43 tok/s — **aggregate output throughput на несколько concurrent запросов**, НЕ single-request
latency. Прямое сравнение нашего single-request 22 против их aggregate 43 было неверным.
Чистый single-request замер см. ниже.

---

## ✅ Чистый single-request A/B (validated, Jun 30) — STRICT no-think vs no-think

**Условия (идентичны для обоих движков):**
- Дословный промпт: `Write a Python implementation of a min-heap with insert, extract_min, heapify (O(n) build from list), and peek. Show time complexity for each operation. Include a complete working demo with at least 10 elements.`
- T=0 (greedy), max_tokens=700, single request (concurrency=1), warm-run
- drafter `z-lab/Qwen3.6-27B-DFlash`, γ=15 (draft cap, K=16)
- **thinking OFF** на обоих (no reasoning токенов)

**Модель-таргет — сверена побайтово (гипотеза «ocicek легче» ОПРОВЕРГНУТА):**
- Atlas `/isolorg/models/Qwen3.6-27B-NVFP4`: 20 559 272 552 B (2 шарда, llm+MTP)
- vLLM `ocicek/Qwen3.6-27B-NVFP4` (HF API): 20 559 284 232 B (llm 19.71G + mtp 0.85G)
- Разница **11 680 B (0.00006%)**, тот же формат NVFP4 group-16, 2672 llm + 15 MTP тензора, 64 слоя.
  Видимость «легче» = на HF-странице виден только файл llm 19.7G без MTP.

| Движок | mode | tok/s (total) | tok/s (decode) | τ | prompt_tok | completion_tok | finish | per-step |
|---|---|---|---|---|---|---|---|---|
| **vLLM DFlash** | no-think | **64.56** | 65.92 | **7.34–7.73** | 60 | 700 | length | — |
| **Atlas DFlash** | no-think | **16.4** | — | **4.80** | 60 | 237 | length | propose 71.7ms + verify 269.3ms |
| Atlas DFlash | thinking ON (best) | 22 | — | 6.56 | 60 | 701 | length | propose ~113ms + verify 269ms |

**Per-position acceptance (no-think):**
- vLLM: 0.880, 0.795, 0.711, 0.627, 0.542, 0.530, 0.458, 0.410, 0.361, 0.313, 0.253, 0.181, 0.096, 0.096, 0.084
- Atlas pos0 (с thinking, EAGLE K=γ) ≈ 0.94 — по приёмке мы конкурентны/выше.

### Выводы
1. **vLLM ~2.9–3.9× быстрее single-request** (64.56 vs Atlas best 22 = 2.9×; vs Atlas no-think 16.4 = 3.9×).
   Разрыв РЕАЛЬНЫЙ, не артефакт aggregate-сравнения.
2. **Это НЕ приёмка.** τ сопоставимы (7.3 vs 4.8–6.56), per-position близки. Узкое место Atlas —
   **сырая decode-loop efficiency**: наш verify (269ms) > весь step vLLM (~15ms/ток × 7τ ≈ 110ms).
3. **Сюрприз: thinking ПОМОГАЕТ Atlas-приёмке.** No-think уронил τ 6.56→4.80 и tok/s 22→16.4 —
   drafter лучше предсказывает код после reasoning-цепочки в контексте. Прежняя гипотеза
   «thinking занижает Atlas» неверна (в output-фазе DFlash и так выключен внутри thinking, `mod.rs:340`).
4. **Главный подозреваемый разрыва — CUDA-graphs + torch.compile (C5).** vLLM warmup 52.6s
   (компиляция + graph capture) → рантайм ~15ms/ток. Atlas форсит eager (graphed-K=γ corruption guard).
   Eager платит launch overhead на десятках ядер × 64 слоя × step.
