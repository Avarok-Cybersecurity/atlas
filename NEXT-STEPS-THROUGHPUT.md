# DFlash Throughput — План «второго фронта» (tok/s к vLLM)

**Статус на старте плана (Jun 30):**
- ✅ **Acceptance решён** — τ=6.56, превосходит vLLM (4.09). Это была главная цель.
- 🎯 **Новая цель** — закрыть разрыв tok/s: Atlas K=16 **~13 tok/s** vs vLLM **43 tok/s**.

**Природа задачи:** это НЕ «починить баг», а планомерная оптимизация нескольких компонентов
verify + batching. Каждый шаг даёт проценты/десятки %, не разы (в отличие от acceptance-фиксов).

---

## Где сейчас узкое место

verify_ms (K=16) = **495ms** после wy16. GDN больше НЕ доминирует (wy16 закрыл).
Оставшееся время доминируют:
- **per-token conv** (phase-1: 15 последовательных `conv1d_update_l2norm`)
- **attention prefill** (28 attention слоёв)
- **SSM projections** (QKVZ GEMM, output proj, FFN per слой)
- прочее (embed, norm, lm_head, argmax)

**Breakdown получен (Этап 0, ATLAS_DFLASH_TIMING=1, K=16, avg 100 шагов):**

| Компонент | ms/step | % | |
|---|---|---|---|
| **SSM phase-1 conv** | **246** | **55%** | ⬅️ ДОМИНАНТА (per-token, 15× conv1d_update_l2norm × 8 слоёв) |
| SSM phase-3 (norm+proj+FFN) | 90 | 20% | |
| attn prefill (28 слоёв) | 78 | 17% | |
| head (lm_head+argmax) | 21 | 5% | |
| SSM phase-2 GDN (wy16) | 14 | 3% | ⬅️ только что оптимизировали — оказался не bottleneck |

(Caveat: sync убирает overlap, абсолютный sum=449 ненадёжен, важны пропорции.)

**Вывод: per-token conv = 55%, в 17× больше GDN.** Amdahl → бить сюда (C1).

Базовая арифметика throughput:
```
tok/s = τ / step_ms,  step_ms = verify_ms + propose_ms + overhead
K=16: 6.56 / (~495 + ~113 + ...) ≈ 13 tok/s
vLLM: 4.09 / ~95ms ≈ 43 tok/s (verify ~110ms + batched 3 req)
```
Чтобы выйти на ~40 tok/s при τ=6.56 нужен step ~165ms → **verify ~110-130ms** (как vLLM target).

---

## Этап 0 — Профилирование breakdown (data-driven приоритеты)

**Цель:** разложить 495ms verify по компонентам, чтобы оптимизировать доминанты, не догадки.

- [ ] Профиль verify_ms по фазам: attention prefill / SSM phase-1 conv / SSM phase-2 GDN (wy16) / SSM phase-3 / lm_head / прочее
- [ ] Отдельно: сколько из 495ms — per-token conv (15 sequential)?
- [ ] Инструмент: `ATLAS_DFLASH_PROF=1` если есть, или ncu, или ручные таймеры по фазам
- [ ] Результат: таблица «компонент → ms → % от verify»

**Статус:** 🔄 в работе

---

## Кандидаты оптимизации (приоритет уточнится после Этапа 0)

### C1. Per-token conv fusion (phase-1) — аналог wy16 для conv
- **Гипотеза:** 15 последовательных `conv1d_update_l2norm` — крупный кусок. Можно ли один batched conv kernel над 16 токенами (causal depthwise conv)?
- **Прецедент:** vLLM `causal_conv1d_update` имеет multi-token spec-ветку (sliding-window обновление conv_state одним вызовом) — Diver-2 это нашёл.
- **Сложность:** средне-высокая (новое CUDA ядро или адаптация).
- **Выигрыш:** TBD (зависит от Этапа 0).

### C2. Batching (max-batch-size > 1)
- **Гипотеза:** vLLM 43 tok/s = 3 concurrent req → выше GPU-утилизация. На одном запросе GB10 недогружен.
- **Блокер:** `dflash_hidden_save` — единый shared буфер (не per-slot). При batch>1 capture одной seq перезапишет другую. Нужен per-slot буфер.
- **Сложность:** средняя (per-slot буферы + dispatch).
- **Выигрыш:** потенциально большой (ближе всего к тому как vLLM реально достигает 43).

### C3. Attention prefill оптимизация
- **Статус:** ATLAS_VERIFY_PREFILL_ATTN уже даёт один prefill pass вместо K decode. 
- **Вопрос Этапа 0:** сколько attention занимает из 495ms? Если значимо — куда дальше.

### C4. K tuning (τ vs verify cost trade-off)
- **Гипотеза:** K=16 не обязательно оптимум по tok/s. Меньший K (8?) — меньше verify, меньше τ. Найти sweet spot по tok/s.
- **Замер:** tok/s vs K для K∈{4,8,12,16} с EAGLE + wy-ядрами.
- **Сложность:** низкая (бенчмарк), может дать быстрый выигрыш.

### C5. Graphed K=γ (разблокировать графы)
- **Блокер:** pre-existing граф-баг (SSM dual-buffer capture) → сейчас форсим eager.
- **Если починить:** графы убирают launch overhead (хотя на K=16 verify SSM-bound, выигрыш под вопросом).
- **Сложность:** высокая (отдельное расследование баги).

---

## Приоритет ПОСЛЕ Этапа 0 (data-driven)

1. ✅ **Этап 0** — профиль готов. Доминанта = phase-1 per-token conv (55%).
2. 🎯 **C1 (conv fusion)** — ГЛАВНЫЙ рычаг. По Amdahl нельзя побить 55%-компонент, вылизывая 3/5/17%. Тот же playbook что wy16 (per-token serial → один fused pass). Цель: ~120ms off verify (~25%).
3. **C3 (phase-3, 20%)** — следующий по величине после conv.
4. **C2 batching / C4 K-tuning** — после C1.
5. **C5 graphed** — отдельное расследование.

---

## C1 — детализация (главная задача)

**Проблема:** phase-1 = 15 последовательных `conv1d_update_l2norm` × ~8 SSM-слоёв ≈ 120 serial conv-запусков/step. Чистый launch + sequential-dependency overhead.

**Решение (зеркало wy16):** единый batched/windowed conv над всеми K=16 токенами на слой:
- causal depthwise conv1d для всех 16 токенов одним запуском (токен t зависит только от окна [t-d_conv+1..t] → параллелизуемо)
- emit 15 conv_state intermediates inline (убрать per-token copy_d2d_async saves)
- L2 norm на Q,K тоже в fused ядре

**Прецедент vLLM:** `causal_conv1d_update` имеет multi-token spec-ветку (Diver-2 нашёл) — sliding-window обновление conv_state одним вызовом с `num_accepted_tokens`. Можно перенять подход.

**Первый шаг:** разведка — есть ли уже batched conv ядро (как wy17 был для GDN), или писать новое.

---

## C1 — РЕЗУЛЬТАТ (Jun 30): 🏆 ОГРОМНЫЙ УСПЕХ

verify_d.rs phase-1 гонял всю pipeline (projection+gate+conv+l2norm) 15× с N=1 ради 15 conv intermediates. Заменили на ОДИН batched проход (prefill_phase1_verify) + новый conv kernel с inline intermediate-saving.

| Метрика | OFF | ON | Δ |
|---|---|---|---|
| phase-1 conv | 246.7ms (55%) | **18.8ms (8%)** | −92% (13×) |
| verify_ms | 496 | **269** | −46% |
| tok/s | 12.1 | **~22** | +80-90% |
| output | — | byte-IDENTICAL | ✅ |
| τ | 6.56 | 6.56 | сохранён |

Флаг `ATLAS_DFLASH_CONV_FUSION=1`. Коммит `7057649`.

### Новый breakdown (после C1) — bottleneck сместился
| Компонент | ms | % |
|---|---|---|
| **SSM phase-3** (gated norm + out proj + FFN) | 89 | **40%** ⬅️ новая доминанта |
| **attn prefill** (28 слоёв) | 78 | **35%** ⬅️ #2 |
| SSM phase-1 conv | 19 | 8% |
| head | 21 | 9% |
| SSM phase-2 GDN (wy16) | 15 | 7% |

→ Следующие рычаги: **C6 (phase-3 fusion, 40%)** и **C3 (attention, 35%)** — теперь co-dominant.

---

## Обновлённая траектория tok/s

| Веха | verify_ms | tok/s |
|---|---|---|
| Старт | 754 | 2.9 |
| EAGLE (acceptance) | 520 | 13.0 |
| wy16 (GDN) | 495 | 13.0 |
| **C1 (conv fusion)** | **269** | **~22** |
| vLLM | ~110 | 43 |

Половину пути к vLLM tok/s прошли. Осталось phase-3 + attention.

---

## C6 — phase-3 fusion: ❌ НЕ дешёвый (разведка Jun 30)

phase-3 УЖЕ batched (вызывается 1× над K=16, не per-token как phase-1). 89ms — это реальный compute: SSM out-proj GEMM (M=16, weight-load-bound) + MoE FFN (16 токенов активируют много экспертов → грузят много весов). C6 ≠ C1 — нет batching-выигрыша. Launch-overhead рычаги исчерпаны (C1 был последним большим).

Аналогично attention (78ms): уже prefill-fused, остаток — genuine compute (QKVO proj weight-bound + KV reads ∝ K + MoE).

## C4 — K-tuning (НОВЫЙ приоритет, атакует 75% cost) 🔄

**Инсайт:** phase-3 (40%) + attention (35%) = 75% verify, оба compute-bound и **оба растут с K** (KV-reads ∝ K, MoE experts ∝ K). А τ=6.56 кластеризуется в ранних позициях (pos0 94%, pos10 prefix ~15%, pos14 ~2%) — проверять все 16 расточительно, большинство reject к pos 8-10.

Снизить K (16→10-12) урежет 75% cost пропорционально, сохранив τ≈6. ~25-30% verify-cut, бесплатно (настройка, не код).

### C4 — РЕЗУЛЬТАТ: ❌ ТУПИК (K-sweep, projected)

| K | τ | verify_proj | tok/s_proj |
|---|---|---|---|
| 16 | 6.56 | 268 | 22.0 |
| 12 | 6.31 | 261 | 21.7 |
| 10 | 6.45 | 258 | 22.4 |
| 8 | 6.09 | 254 | 21.4 |
| 6 | 4.61 | 253 | 16.3 |

**Доминанты K-НЕЗАВИСИМЫ:** attn (~77), phase3 (~88), head (~20), phase1 (~18) константны по K=6→16. Только phase-2 варьируется (fallback-искажение). MoE/attention weight-load насыщается к ~6 токенам → 6-16 verify токенов грузят почти всю MoE независимо от K.

**Вывод:** снижение K теряет τ без экономии verify. tok/s плоский ~22 от K=10 до 16, knee нет. **Держим K=16.** Писать wy12/wy10/wy8 НЕ нужно — даже с ними tok/s ~22.

**Реальный потолок:** ~270ms K-независимый weight-load floor (attn 77 + phase3-MoE 88 + head 20 + phase1 18). Дешёвые рычаги (launch/batching внутри step) исчерпаны — C1 был последним.

---

## РАЗРЕШЕНО: vLLM 43 = AGGREGATE, НЕ per-request ✅ (Diver-2, цитаты vLLM кода)

`serve.py:584`: `output_throughput = sum(actual_output_lens) / dur_s` — сумма токенов ВСЕХ запросов / wall-clock. «3 req × 139 tok» = заголовок `vllm bench serve`, 3 **concurrent** запроса. 43 = их СУММА.

**Per-request vLLM ≈ 43/3 ≈ 14 tok/s.** Per-request метрика у vLLM выражается через TPOT/ITL (serve.py:458), а не output_throughput.

### Реальное сравнение (равная concurrency)
| | single-request tok/s | τ |
|---|---|---|
| **Atlas (наш)** | **22** | **6.56** |
| vLLM (оценка из aggregate/3) | ~14 | 4.09 |

**Atlas single-request БЫСТРЕЕ vLLM single-request, и выше по τ.** Разрыв 22 vs 43 был артефактом сравнения нашей latency против их aggregate-throughput на 3 запроса.

**Оговорка (Diver-2):** 100% подтверждение требует исходного лога vLLM (max-concurrency + mean_tpot_ms) или перемера vLLM в single-request (`--max-concurrency 1`). В Docker-образе только код бенчмарков, не результаты. Перемер vLLM требует остановки Atlas (GPU-конфликт!).

### Вывод для плана
Цель «догнать 43 tok/s single-request» была основана на неверном сравнении. По latency мы уже конкурентны/впереди. **43 — это aggregate throughput, достигается БАТЧИНГОМ (C2).** Не «догонять», а добавить concurrency, если нужен system throughput.

## Оставшиеся рычаги (тяжёлые, после исчерпания дешёвых)

### C2 — Batching (если vLLM 43 = aggregate, это ГЛАВНЫЙ путь)
Weight-load floor (~270ms) амортизируется между concurrent запросами: 3 req × 16 ток через те же MoE веса → per-request floor падает ~3×. Это ровно как vLLM достигает throughput.
- **Блокер:** `dflash_hidden_save` единый shared буфер → нужен per-slot. batch>1 не тестирован.
- **Эффект:** aggregate throughput potentially ~40 (как vLLM), хотя per-request остаётся ~22.

### C5 — graphed K=γ (launch overhead)
Сейчас форсим eager (graphed-K=γ corruption guard). Eager платит launch overhead × десятки ядер × слои. НО: если verify weight-load-bound (GPU занят), графы дадут мало. Нужно сначала измерить launch vs compute долю.
- **Блокер:** pre-existing SSM dual-buffer capture баг (тяжело).

### C7 — MoE weight-load efficiency
Full-expert-set load — это floor. Только faster expert GEMM/quant или expert-locality. Deep/hard.

## Журнал
- Jun 30: план создан. Этап 0 профиль → phase-1 conv 55%. **C1 conv fusion → verify 496→269ms, tok/s 12→22 (byte-identical).** Bottleneck → phase-3 (40%) + attention (35%).
- Jun 30: C6 (phase-3) = compute-bound, не C1-style. **C4 (K-tuning) = ТУПИК** — доминанты K-независимы (weight-load floor). Дешёвые рычаги исчерпаны. Ключевой вопрос: vLLM 43 single или aggregate? → определит C2 (batching) vs compute-оптимизацию.
