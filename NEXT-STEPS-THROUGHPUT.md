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

## Журнал
- Jun 30: план создан. Профилирование (Этап 0) → phase-1 conv доминирует (55%). Приоритет C1 (conv fusion). Запущена разведка conv-пути.
