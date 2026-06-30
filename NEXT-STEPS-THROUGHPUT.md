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

**Точный breakdown — TBD (профилирование, Этап 0).**

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

## Рекомендованный порядок (черновой, до Этапа 0)

1. **Этап 0** — профиль (обязательно первым, иначе оптимизируем вслепую)
2. **C4 (K tuning)** — дёшево, может дать быстрый tok/s выигрыш
3. **C1 или C2** — по результатам Этапа 0 (что доминирует: conv → C1; недогрузка GPU → C2)
4. Остальное по данным

---

## Журнал
- Jun 30: план создан. Запущено профилирование (Этап 0).
