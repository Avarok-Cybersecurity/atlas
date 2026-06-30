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

## ✅ VALIDATED: чистый single-request A/B (Jun 30) — разрыв РЕАЛЬНЫЙ, ~2.9–3.9×

Перемеряли ОБА движка в одинаковых условиях (дословный промпт, T=0, max_tokens=700, γ=15,
single request, warm-run, **thinking OFF на обоих**). Кросс-проверка двумя Дайверами.

**Сначала сняли все красные флаги прошлого замера:**
1. ✅ **Модель — идентична.** Сверено побайтово (я лично + Diver-2): Atlas 20 559 272 552 B vs
   ocicek `20 559 284 232 B` (HF API) — разница 11 680 B (0.00006%), тот же NVFP4 group-16,
   2672 llm + 15 MTP тензора, 64 слоя. Гипотеза «ocicek легче → нечестно» ОПРОВЕРГНУТА
   (видимость «легче» = на HF видно только файл llm 19.7G без MTP-головы).
2. ✅ **Промпт — дословный Atlas-bench** (не «тот же класс»).
3. ✅ **thinking выровнен** — оба no-think.

### Финальная таблица

| Движок | mode | tok/s (total) | τ | completion_tok | per-step |
|---|---|---|---|---|---|
| **vLLM DFlash** | no-think | **64.56** | 7.34–7.73 | 700 | ~15ms/ток |
| **Atlas DFlash** | no-think | **16.4** | 4.80 | 237 | propose 71.7 + verify 269.3ms |
| Atlas DFlash | thinking ON (best) | 22 | 6.56 | 701 | propose ~113 + verify 269ms |

→ **vLLM в 2.9× быстрее Atlas-best (22), в 3.9× быстрее Atlas-no-think (16.4).** Разрыв РЕАЛЬНЫЙ.

### Три ключевых вывода
1. **Это НЕ приёмка.** τ сопоставимы (7.3 vs 4.8–6.56), per-position близки (наш pos0 94% vs vLLM 88%).
   Узкое место Atlas — **сырая decode-loop efficiency**: наш verify (269ms) > весь step vLLM (~110ms).
2. **Сюрприз — thinking ПОМОГАЕТ Atlas-приёмке** (предыдущая гипотеза неверна). No-think уронил
   τ 6.56→4.80 и tok/s 22→16.4: drafter лучше предсказывает код после reasoning-цепочки в контексте.
   (В output-фазе DFlash и так выключен внутри thinking, `mod.rs:340` — речь про код ПОСЛЕ.)
3. **Старый «43 = aggregate, Atlas впереди» — ОТМЕНЁН.** Тот вывод опирался на сравнение нашего
   single-request против vLLM-aggregate/3. Прямой single-request замер vLLM = 64.56 → мы позади.

---

## ⭐ ГЛАВНЫЙ ВЫВОД (Jun 30, профиль Diver-2): разрыв в ЯДРАХ, не в графах

**Измерено прямым профилем vLLM (in-process torch.profiler, custom-scopes, eager):**

| фаза | vLLM ms/шаг | Atlas ms/шаг | отрыв |
|---|---|---|---|
| target forward (**verify**) | **95.2** | 269.3 | **2.8×** |
| drafter (**propose**) | **27.7** | 71.7 | 2.6× |
| postprocess (sample+reject) | 11.5 | — | — |
| **итого/шаг** | **~134** | ~341 | **2.9×** |

**C5 (графы) ОПРОВЕРГНУТ как главный рычаг:** within-session A/B vLLM **graphs vs `--enforce-eager` = +4.9%**
(60.3 vs 57.5 tok/s decode, τ=7.4 одинаков). Launch-overhead — НЕ узкое место. Decode-шаг compute/memory-bound,
большие GEMM доминируют. Идеальные графы купят Atlas ~5%, не 2.9×. **C5 демотирован** (средний фикс ради 5%).

**Per-kernel доминирование vLLM (Self-CUDA, eager):**
| ядро | доля | ↔ Atlas фаза |
|---|---|---|
| **`flashinfer_mm_fp4`** (NVFP4 GEMM: FFN+проекции) | **54.4%** | ↔ **phase-3** (norm+proj+FFN) — ГЛАВНЫЙ отрыв |
| `aten::mm` cutlass bf16 (drafter 5сл + lm_head 248K vocab) | 30% | ↔ propose + head |
| `qwen_gdn_attention_core` | 8.6% | ↔ phase-2 GDN |
| `fused_sigmoid_gating_delta_rule` (fla SSM single-launch) | 8.4% | ↔ phase-2/phase-1 |
| conv1d / attn / norms / fp4-quant | ~4% | ↔ phase-1 conv / attn / head |

### Пересмотр «weight-load floor» — ОПРОВЕРГНУТ
Прежний вывод «floor ~270ms неустраним» НЕВЕРЕН. vLLM делает тот же verify за 95ms на той же
модели/формате/железе. Floor не физический — это **2.8× менее эффективные ядра**. Отрыв размазан
по всем фазам (verify И propose 2.6-2.8×), не в одной точке.

## ⭐ НОВЫЙ ПРИОРИТЕТ: эффективность ядер (FP4 GEMM + fused SSM)

1. **K1 — FP4-GEMM эффективность (phase-3 / FFN / проекции)** — 54% коста vLLM, главный аналог нашей
   доминанты. Цель: догнать фьюзед-cutlass/flashinfer FP4-GEMM. Эталон — форк aeon (`flashinfer_mm_fp4`,
   vllm-cutlass FP4). **Это рычаг на 2.9×, не 5%.**
2. **K2 — fused SSM/GDN ядра** — vLLM `fused_sigmoid_gating_delta_rule` (fla, single-launch) + GDN core ~17%.
   Сравнить с нашими wy16 + phase-1 conv: единый ли launch, насколько эффективнее.
3. **K3 — lm_head над vocab 248K** — часть bf16-GEMM 30%, оба платят; вторично.
4. **C5 (графы)** — отложен, ~5%, дешёвый pickup потом. Разбор готов (порт K=2 in-place на K=γ).

**Критический путь — ЗАКРЫТ (Diver-2, Jun 30):** форк aeon склонирован, изучен. Результат: оба ядра
(`flashinfer_mm_fp4`, `fused_sigmoid_gating_delta_rule`) — СТОРОННИЕ библиотеки (flashinfer-ai, fla),
aeon их не писал/не патчил (только 3 Python-патча, ноль CUDA). FP4-GEMM = CUTLASS NVFP4 W4A4
block-scaled, group_size=16, специализация под sm_120a уже есть в `flashinfer/data/csrc/fp4_gemm_cutlass_sm120.cu`.
CUTLASS — header-only C++ (BSD-3), линкуется напрямую в Rust+CUDA Atlas, не Triton. fused-SSM — Triton
(`fla`), из Rust не линкуется → нужен ручной CUDA-порт. Полная схема и две референс-точки конфигурации
(flashinfer и vLLM `cutlass_scaled_fp4_mm`) — см. THROUGHPUT-GAP-FINDINGS.md, Шаг 1.
**Решение пересмотрено (Jun 30, явное указание):** flashinfer/CUTLASS как зависимость НЕ тащим
(в идеале вообще без внешних GEMM-либ). Найденная схема (W4A4 NVFP4, group_size=16, block-scaled
scale-handling, tile/cluster-конфиг под sm_120a) — это РЕФЕРЕНС для понимания слабостей нашей
СОБСТВЕННОЙ phase-3-реализации, не код для вендоринга. Новый критический путь K1: профилировать
наш текущий phase-3 GEMM-kernel точечно, найти конкретные неэффективности относительно референсной
схемы (tiling/occupancy/fusion активации-квантования/scale-handling), переписать/докопипастить
нужные куски в существующий Atlas CUDA. Двигаемся в сторону vLLM как baseline по числам, не по коду.

### Прочие тяжёлые рычаги (после C5)
- **C2 batching** — aggregate throughput через concurrency (per-slot `dflash_hidden_save`). Блокер: shared буфер.
- **C7 MoE weight-load efficiency** — full-expert-set load = floor. Faster expert GEMM/quant. Deep/hard.

---

## Факты по vLLM-замеру (Diver-2, подтверждено логами/кодом/HF)

Чем именно vLLM достигает 64.56 — все красные флаги сняты:
1. **Образ КАСТОМНЫЙ** — `ghcr.io/aeon-7/aeon-vllm-ultimate:latest`, vLLM `0.23.0+aeon.sm121a.dflash`,
   собран из исходников под **sm_121a (GB10)**. Фичи: **DFlash PR #40898** (без него DFlash в stock
   vLLM не работает), NVFP4/FP8 KV-cache на Triton (#44389), **flashinfer-cutlass / vllm-cutlass GEMM**,
   TurboQuant K8V4, DFlash high-conc/prefix-cache фиксы. Не stock vLLM.
2. **CUDA-graphs ВКЛ** — `enforce_eager=False`, `cudagraph_mode=FULL_AND_PIECEWISE`, `VLLM_COMPILE`/inductor.
   Warmup 52.6s (compile+capture) → тёплый шаг ~15ms/ток. ← главное отличие от нашего eager.
   Команда: `serve ocicek/Qwen3.6-27B-NVFP4 --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}' --gpu-memory-utilization 0.8 --max-model-len 8192 --max-num-seqs 4`.
3. **Драфтер НЕ устарел** — обе стороны на одном HF-snapshot `0919688` (2026-04-27, latest), идентичные веса.
4. **vLLM спекулирует и в thinking** — reasoning-гейта в spec_decode НЕТ (propose гейтится только длиной
   `input_fits_in_drafter`). У нас guard `mod.rs:340` глушит DFlash внутри think → мы тут теряем (см. Workstream B).

→ Разрыв сужается до двух осей: **(A) CUDA-graphs** (наш eager vs их графы) + **(B) эффективность ядер**
(их cutlass/flashinfer/Triton vs наши). Пропорцию измерит профиль Diver-2 (graphs vs `--enforce-eager`).

---

## C5 — детальный разбор (Diver-1, read-only) — порт K=2 in-place дизайна на K=γ

**Guard:** `verify_d.rs:186-213`. По умолчанию `force_eager=true`; разблок флагом
`ATLAS_DFLASH_UNSAFE_KGAMMA_GRAPH=1`. Форсит eager на **весь verify-шаг** (единый граф = 64 слоя +
SSM 3 фазы + norm + lm_head + argmax). На K=γ мы не графим вообще ничего.

**Механика capture исправна, ломается только K=γ:**
- K=2/K=3/K=4 графятся успешно (verify_b/c/c2). Метаданные (seq_len/positions) заливаются вне графа
  по фикс-адресу → растущий seq_len граф НЕ ломает (гипотеза «stale seq_len» опровергнута).

**Корень — dual-buffer SSM (структурно подтверждено):**
- Улика: `verify_b.rs:51-58` — *«K=2 убрал dual-buffer pre-verify copy, h_state канонический, фикс-указатель
  → граф работает. K=3/K=4/DFlash всё ещё гоняют `pre_verify_copy_async`.»*
- K=γ висит на legacy: `pre_verify_copy_async` (`verify_d.rs:56`, seed checkpoint→live) +
  `commit_verify_state_async` (`verify_dflash_step.rs:239`).
- **Исключены** (статикой): address-mismatch (интермедиаты алиасят `ssm_pool`, одна память),
  dynamic shape, cross-stream hazard, EAGLE. Точную ломающую операцию даст только `compute-sanitizer`
  (GPU) — узкий подозреваемый: взаимодействие seed-копии/scratch-canonical split с кернелами под capture.

**Полфикса УЖЕ в коде:** in-place commit стиля K=2 (`commit_accepted_prefix_dispatch`,
`async_chkpt.rs:229-278`) реализован, но K=γ его не зовёт.

**План фикса (не реализован):**
1. Убрать `pre_verify_copy_async` для K=γ (`verify_d.rs:56`) — h_state/conv_state каноничны, как в K=2.
2. `commit_verify_state_async` → `commit_accepted_prefix` на dflash-пути (`verify_dflash_step.rs:239`).
3. Выпилить scratch/canonical split (WY4-fallback save/restore, `verify_d.rs:480-519`).
4. Снять guard, A/B: byte-identical T=0 + τ-preserved.

**Сложность: средняя.** Риск: 3-фазный prefill-SSM писался под dual-buffer — проверить, что phase-1/wy16
мутируют canonical h_state корректно без pre-seed.

**Окупаемость — ОТКРЫТО.** Зависит от доли launch-overhead. Меряем с двух сторон:
(а) дёшево у нас: ATLAS_DFLASH_TIMING sum vs wall; (б) Diver-2: vLLM graphs vs eager (верхняя граница).
Если floor (attn 77 + phase3 88 + head 20) доминирует — C5 даст мало; если launch-overhead большой — окупится.

---

## Workstream B — DFlash на thinking (снять guard mod.rs:340)

**Мотивация:** vLLM спекулирует на reasoning-токенах и держит 64 tok/s. У нас guard `mod.rs:340`
глушит DFlash внутри think → 694 think-токена идут без спекуляции на базовой скорости = большая доля
end-to-end латентности. Логики ограничивать драфтер в thinking, если vLLM этого не делает, нет.

**Ключевой вопрос ДО снятия guard — зачем он стоит:**
- (a) консервативный/исторический (без замеров) → снять и сразу выиграть.
- (b) измеренная причина: драфтер (тренирован на коде) плохо предсказывает reasoning → приёмка
  околонулевая → verify-стоимость спекуляции > экономии → было net-negative на eager. vLLM переживает
  за счёт дешёвого графового verify. Тогда B завязан на C5 (порядок: C5 → B).
- Плюс корректность: не ломают ли think-токены состояние (как было с позиционными багами).

**Эмпирика, которую надо снять:** реальная приёмка DFlash на think-токенах (τ на reasoning vs на коде).

**Порядок (предв.):** сначала C5 (разблокирует дешёвый verify, от которого зависит окупаемость B),
потом снять thinking-guard и перемерить end-to-end латентность полного thinking-ответа.

---

## СОСТОЯНИЕ СЕССИИ (для возобновления после компакта)

**Дайверы (Monet subsessions) — resume, НЕ респавнить:**
- **Diver-1** `claude_2a97ebc5-39b0-4508-a132-74b416a0b317` — Atlas код/GPU/сервер. opus. СТОИТ (Atlas погашен, GPU освобождён).
- **Diver-2** `claude_2c5e18e6-7288-4b8a-8c4d-f2981d627179` — vLLM reference reader. opus. СТОИТ (vLLM погашен).
- Резюме результата: `source $MONET_ENV_FILE; curl $MONET_URL/subsession/result?sessionId=...`

**GPU СЕЙЧАС:** свободна (vLLM погашен Diver-2, Atlas лежит). ⚠️ GPU-мьютекс:
НИКОГДА Atlas server + vLLM Docker одновременно — проверять `nvidia-smi --query-compute-apps` перед стартом.

**Pending / следующий шаг (порядок пересмотрен после форк-находки):**
1. ✅ Профиль vLLM ГОТОВ: графы дают +5% (C5 демотирован), разрыв 2.9× в ЯДРАХ (FP4-GEMM 54%, fused-SSM 17%).
2. ✅ **Форк aeon склонирован и изучен (Diver-2).** Вывод: оба доминирующих ядра — сторонние библиотеки
   (CUTLASS NVFP4 GEMM через flashinfer, Triton fused-SSM через fla), aeon их не писал. CUTLASS линкуется
   в Rust+CUDA напрямую (header-only, BSD-3) — путь K1 это вендоринг/линковка, не reverse-engineering.
3. 🎯 **K1 — следующий шаг (БЕЗ flashinfer/CUTLASS как зависимости).** Сначала разведка (read-only,
   Diver-1): прочитать наш текущий phase-3 GEMM-kernel (out-proj+FFN, crates/spark-model) построчно,
   сопоставить с референсной схемой (W4A4 NVFP4 group-16 block-scaled, tile/cluster/scale-handling под
   sm_120a) и найти КОНКРЕТНЫЕ точки неэффективности: tiling/block-size, occupancy, fusion
   квантования-активации, scale-layout, redundant memory traffic. Затем точечно переписать/докопипастить
   в нашем существующем CUDA — не вендорить чужую библиотеку.
4. **K2 — аналогично для fused SSM/GDN** (~17%, схема fla как референс, не Triton-код напрямую —
   найти неэффективности в нашем wy16+conv пути и устранить в своём коде). **K3 — lm_head 248K** (вторично).
5. **C5 (графы)** — отложен (~5%, дешёвый pickup потом). Разбор Diver-1 готов: порт K=2 in-place commit на
   K=γ (guard verify_d.rs:186-213, корень legacy dual-buffer, полфикса в `commit_accepted_prefix_dispatch`).
6. **Workstream B** — снять thinking-guard `mod.rs:340`. Сначала приёмка DFlash на think-токенах. Завязка на K1/C5.

**Все фиксы закоммичены** на ветке `optimizations`. Env-флаги: ATLAS_DFLASH_EAGLE_FIX,
ATLAS_DFLASH_CONV_FUSION, ATLAS_VERIFY_PREFILL_SSM, ATLAS_VERIFY_PREFILL_ATTN, ATLAS_DFLASH_TIMING
(все off=byte-identical). Default форсит eager для K=γ (graphed corruption guard).

**Документы (внешняя память):** DEBUG-ACCEPTANCE.md, OPTIMIZATIONS-CATALOG.md, BENCH-27B-DFLASH.md
(числа + чистый A/B), SSM_VERIFY_PLAN.md / PREFILL_VERIFY_PLAN.md, этот файл (throughput план).

## Журнал
- Jun 30: план создан. Этап 0 профиль → phase-1 conv 55%. **C1 conv fusion → verify 496→269ms, tok/s 12→22 (byte-identical).** Bottleneck → phase-3 (40%) + attention (35%).
- Jun 30: C6 (phase-3) = compute-bound, не C1-style. **C4 (K-tuning) = ТУПИК** — доминанты K-независимы (weight-load floor). Дешёвые рычаги исчерпаны.
- Jun 30: **ОТМЕНЁН прежний вывод «vLLM 43 = aggregate, Atlas впереди».** Чистый single-request A/B (оба no-think, дословный промпт, модель сверена побайтово): **vLLM 64.56 vs Atlas 16.4 (no-think) / 22 (best)** → vLLM 2.9–3.9× быстрее, разрыв РЕАЛЬНЫЙ. Узкое место = decode-loop efficiency, не приёмка. Открытие: thinking ПОМОГАЕТ Atlas-приёмке (τ 4.80→6.56). **Поворот на C5 (CUDA graphs).**
- Jun 30: vLLM-конфиг вскрыт (Diver-2): кастомный образ aeon `0.23.0+aeon.sm121a.dflash` (DFlash PR #40898 + cutlass/Triton ядра), **CUDA-graphs ВКЛ**, драфтер не устарел (snapshot `0919688`), vLLM спекулирует и в thinking. Разрыв → 2 оси: графы + ядра.
- Jun 30: **C5 разбор (Diver-1, read-only):** guard `verify_d.rs:186-213` форсит eager весь шаг; корень — legacy dual-buffer SSM (`pre_verify_copy_async` + `commit_verify_state_async`), которого нет в рабочем K=2; полфикса уже в коде (`commit_accepted_prefix_dispatch`); сложность средняя. Окупаемость ждёт профиль Diver-2.
- Jun 30: **Workstream B заведён** — снять thinking-guard `mod.rs:340` (vLLM спекулирует в think, мы нет). Завязка на приёмку DFlash на think-токенах; порядок предв. C5 → B.
- Jun 30: **⭐ ПРОФИЛЬ vLLM (Diver-2) развернул приоритеты.** Графы vs eager = **+4.9%** → **C5 демотирован** (не главный). Разрыв 2.9× **в ЯДРАХ**: vLLM verify 95ms vs наш 269ms (2.8×), propose 28 vs 71.7 (2.6×). Доминанта — **`flashinfer_mm_fp4` 54%** (↔ наша phase-3), fused-SSM 17%. **«Weight-load floor неустраним» ОПРОВЕРГНУТ** — vLLM делает тот же verify за 95ms лучшими ядрами. **Новый приоритет K1: FP4-GEMM эффективность.** Критический путь — склонить форк aeon (эталон ядер).
- Jun 30: **Форк aeon склонирован и изучен (Diver-2).** `github.com/AEON-7/vllm-ultimate-dgx-spark` — публичный build-репо (3 Python-патча, ноль CUDA). Оба доминирующих ядра (`flashinfer_mm_fp4` 54.4%, `fused_sigmoid_gating_delta_rule` 8.4%) — СТОРОННИЕ библиотеки (flashinfer-ai CUTLASS NVFP4 GEMM, fla Triton fused-SSM), aeon их не трогал. FP4-GEMM = CUTLASS NVFP4 W4A4 block-scaled, group_size=16, sm_120a-специализация уже в `flashinfer/data/csrc/fp4_gemm_cutlass_sm120.cu`. CUTLASS header-only C++ (BSD-3) → линкуется в Rust+CUDA Atlas напрямую, не нужно переписывать с нуля. fused-SSM — Triton, из Rust не линкуется, нужен ручной CUDA-порт. **K1 переопределён: не reverse-engineering aeon-ядер, а вендоринг/линковка CUTLASS NVFP4-GEMM.**
