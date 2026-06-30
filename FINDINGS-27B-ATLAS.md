# Findings: Qwen3.6-27B NVFP4 на Atlas / GB10

**Железо:** NVIDIA GB10 (Grace Blackwell, SM121a), 121 GB unified RAM, 178 GB/s  
**Модель:** Qwen3.6-27B-NVFP4 (19.2 GB, modelopt format)  
**Сборка:** `ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4 ATLAS_TARGET_HW=gb10`

---

## Текущие лучшие числа (2026-06-28)

| Конфигурация | tok/s | Примечание |
|---|---|---|
| No-MTP baseline | 12.0 | decode=83ms/step |
| **MTP K=2 fp8 head** | **17.8** | текущий лучший (T=1) |
| MTP K=2 bf16 head | 16.2 | хуже fp8 по accept rate |
| DFlash γ=16 (K=16 verify) | ~11.7 | T=0 accept≈1% → хуже baseline |
| SGLang FP8 + DFlash SM121 | ~21.0 | для сравнения |
| vLLM aeon DFlash K=16 | 43 | NVFP4+DFlash, accept=27-28%, τ≈4.1 — reference (139 tok/запрос, ctx gap не накапливается) |

**Оптимальный рабочий конфиг (MTP):**
```bash
./spark serve Qwen3.6-27B-NVFP4 --port 8888 \
    --kv-cache-dtype nvfp4 --kv-high-precision-layers 4 \
    --speculative --mtp-quantization fp8 --num-drafts 1 \
    --scheduling-policy slai
```

**Оптимальный рабочий конфиг (DFlash, cap=1):**
```bash
ATLAS_DFLASH_CTX_WINDOW=64 \
./spark serve Qwen3.6-27B-NVFP4 --port 8888 \
    --dflash --draft-model <path> \
    --kv-cache-dtype nvfp4 --kv-high-precision-layers 4 \
    --scheduling-policy slai --gpu-memory-utilization 0.75
# ATLAS_DFLASH_DRAFT_CAP не задан → default=1 → K=2 verify
```

---

## Профилирование decode шага

### Breakdown SSM слоя (~1150μs/layer)

| Операция | Время | Доля |
|---|---|---|
| FFN gate_up (NVFP4, fused) | 462μs | 40% |
| FFN silu_down (NVFP4, fused) | 272μs | 24% |
| qkvz GEMV (NVFP4) | 218μs | 19% |
| out_proj GEMV (NVFP4) | 94μs | 8% |
| gdn_decode | 43μs | 4% |
| overhead (launch/norm) | ~61μs | 5% |

**FFN = 64% SSM слоя.** Bandwidth efficiency: 81% — уже хорошо, запас ~20%.  
**GDN decode = 4%.** Latency-bound при 15% occupancy (48 блоков / 20 SM). Оптимизировать бесполезно.  
**Attention = ~5% decode time.** NOT bottleneck. split-K не нужен.

### MTP gate timing (измерено)
```
decode = 72-83ms, verify K=2 = ~120ms → verify_multiplier = 1.6-1.7
```

---

## Что исследовано и закрыто

### ✅ Оптимизации реализованы

| Задача | Результат | Коммит |
|---|---|---|
| fc GEMV loop → batched pipelined GEMM | propose 110ms → 43ms | `cfc3012` |
| rep_pen/DRY bypass в K=2/K=3/K=4/Kγ verify | accept/step 1.69 → **4.116** | `066f84c`, `ba803ca` |
| norm каждые 16 токенов вместо каждого | <1% прироста — kernel latency-bound | — |

### ✅ Баги найдены и исправлены

| Баг | Эффект до фикса | Коммит |
|---|---|---|
| YaRN вместо standard RoPE в DFlash drafter | 0% acceptance, Paris（巴黎）баг | `27aa5e2` |
| BF16/FP32 mismatch в batched verify (sequential else-branch) | h_state corruption → NaN → EOS | `6c6dee6` |
| embed stride `fp32=4` вместо `2` в verify_d.rs | garbled output K=5..16 | `2e3a26a` |
| **Poisoned context (verify_b.rs):** `try_dflash_capture(layer_idx, k-1=1)` — захватывал hidden драфта (отвергнутого токена) вместо last_token. На REJECT (97% шагов) dflash_hidden_save = hidden(draft) → в ctx_hidden_acc попадал garbage → самоусиливающийся цикл: плохой ctx → плохие черновики → почти всегда reject → снова плохой ctx. **Фикс:** `try_dflash_capture(layer_idx, 0)` — всегда захватывает row 0 = last_token. 2.9% → 6.9%. | незакоммичен |
| **Accept ctx gap (verify_b.rs + verify_k2_step.rs):** На ACCEPT `seq_len += 2` но `ctx_len += 1` — один слот вместо двух. После N accept'ов накапливается gap = N между `noise0_pos` и последним ctx slot'ом. При 6.9% acceptance за 700 токенов: gap ≈ 48, noise0_pos = 757, last ctx slot = 709 → RoPE позиции драфтера разъезжаются на 48 → acceptance коллапсирует по мере роста sequence. **Фикс:** (1) `dflash_hidden_save` удвоен (2×51200 байт): row0=last_token, row1=draft. (2) В CUDA graph loop добавлен `try_dflash_capture_draft(layer_idx)` — row 1 → вторая половина буфера. (3) В ACCEPT ветке после propose: `dflash_accept_append()` — appends row1 к `ctx[N+1]`, `ctx_len += 2`, теперь sync с seq_len. Бенч в процессе. | незакоммичен |
| WY17 bail check `<17` вместо `<16`, conv loop OOB | WY17 unreachable при γ=16 | `2e3a26a` |
| norm_token_count не откатывался при reject | CPU counter drift | `6c6dee6` |
| rep_pen/DRY pipeline применялся в K=2/K=3 verify | занижал acceptance в 2-3× | `066f84c` |

### ✅ Гипотезы опровергнуты (не тратить время повторно)

- **Attention — не bottleneck.** 5% decode time. split-K не даёт прироста (байт-идентичный A/B).
- **GDN decode — не bottleneck.** 4% SSM слоя. Kernel latency-bound при 15% occupancy. Любые оптимизации дадут <1%.
- **Norm оптимизация.** Реализована (каждые 16 токенов), прирост <1%. Закрыто.
- **T=2 denoising.** Propose удваивается (43ms→92ms), acceptance не улучшается. Модель конвергирует при T=1. Не использовать.
- **enable_thinking=False / reasoning_effort=none / /no_think текст.** Atlas игнорирует все эти параметры — модель думает всегда.
- **K2 drift gauge warnings (<30%).** Норма для DFlash на thinking-модели. Не pathology логитов target model.
- **Три-фазный SSM verify path — не помогает (ATLAS_VERIFY_PREFILL_SSM=1):** Стеш заменяет sequential GDN на WY4 для K≥5, но не трогает реальный bottleneck: (1) Phase1 по-прежнему K serial conv1d вызовов, (2) 4 attention слоя по-прежнему K×paged-decode. Verify масштабируется O(K) по attention независимо от GDN. Benchmark (2026-06-29): F4=7.0 tok/s, F8=4.2 tok/s — деградация относительно K=2 (10.4 tok/s). Причина: при acceptance ~3% per-token прирост accepted/шаг ничтожен (~1.03→~1.21 за K=2→8), а verify time растёт пропорционально K. **Оптимум при текущем acceptance — K=2.** Единственный путь к реальному speedup при высоком K — prefill-mode causal attention (один проход с causal mask вместо K decode passes), что стеш НЕ реализует.
- **ATLAS_DFLASH_DRAFT_CAP механика:** env var работает корректно — обрезает `drafts.len()` до cap перед dispatch. Default=1 (не 15). Scheduler показывает `num_drafts=15` в startup log, но это scheduler-level max, не фактический cap. Drafter всегда выдаёт `γ-1=15` реальных токенов (noise0 — untrained, пропускается). `ATLAS_DFLASH_DRAFT_CAP=15` → k_verify=16 (max). Настоящий K=16 = CAP=15.
- **DFlash K=γ (CAP≥15) — sequential bottleneck:** verify=912ms, verify_mult=12.47, ~2.2 tok/s без оптимизаций. Sequential per-token loop K=16. **WY17 недостижим при γ=16:** drafter выдаёт только `γ-1=15` новых токенов — noise0 не обучен (logits мусор), SGLang его пропускает (`draft_next = greedy_sample(draft_hidden[:, 1:, :])`). `drafts.len()` max=15, `k_verify` max=16. WY17 требует k_verify=17 → нужен γ=17. Даже с CAP=16: `drafts=15`, `k_verify=16`. **Обходной путь: `ATLAS_VERIFY_PREFILL_SSM=1`** — три-фазный verify (Phase1: serial conv1d; Phase2: WY4 batch GDN; Phase3: output). Применимо для любого K≥5. Тест в процессе.
- **MTP/DFlash отключается во время thinking:** `scheduler/mod.rs:340` — `!active[0].inside_thinking`. Thinking токены генерируются обычным decode. Стрипаются из стрима клиенту (`emit_step.rs:85 — return` на `<think>` токене). K2 gauge меряет acceptance только на output фазе.
- **ATLAS_DFLASH_CAPTURE_LAYER_OFFSET — не причина низкого acceptance:** Протестированы все три значения. offset=-1 (capture [0,15,30,45,60]) и offset=0 (capture [1,16,31,46,61]) дают одинаковые ~5.8%. offset=+1 (capture [2,17,32,47,62], соответствует vLLM post-PR #40898) даёт 3.3% — хуже. Оптимальный offset=-1 (default). vLLM PR #40898 посвящён SWA поддержке в KV-кеше и нормализации layer IDs — не применим к нашей схеме захвата.
- **vLLM PR #40898 ([Spec Decode] Add SWA support to DFlash):** Основное содержание — поддержка Sliding Window Attention в KV-кеше драфтера (не наш bottleneck) + нормализация layer IDs (+1 shift). Тест показал: shift +1 только ухудшает. Не применять.
- **DFlash raw_argmax расходился с baseline (ПОФИКШЕНО):** DFlash пропускал rep_pen/DRY при emission токена. При rejection эмитился `argmax(raw_logits)` вместо `argmax(after_rep_pen)` → sequence drift → acceptance 1%, total_tok=979 вместо 1351. **Фикс (`verify_k2_step.rs`):** acceptance check остался по raw argmax (drafter судит так же), emission теперь через pipeline как у baseline. После фикса: total_tok=1395 (baseline), acceptance 6% (было 1%).
- **SSM state mismatch для K=5..16.** Опровергнут анализом — intermediates совместимы между WY и sequential. Root cause был embed stride bug в verify_d.rs.
- **lm_head_shared dangling pointer в NVFP4.** Не баг — target и drafter имеют одинаковый vocab_size=248320, sharing корректен.

### ✅ DFlash dispatch — как работает

```
ATLAS_DFLASH_DRAFT_CAP (default=1) — обрезает drafts.len() в propose.rs перед dispatch.
Scheduler показывает num_drafts=15 в startup — это scheduler-level max, не фактический cap.
num_drafts из API игнорируется (let _ = num_drafts).
Drafter (γ=16) всегда выдаёт γ-1=15 токенов (noise0 untrained, пропускается).

Dispatch по drafts.len() после cap:
cap=1  → drafts.len()=1  → step_verify_k2        (k_verify=2,  WY2 graphed)  ← default
cap=2  → drafts.len()=2  → step_verify_k3        (k_verify=3,  WY3)
cap=3  → drafts.len()=3  → step_verify_k4        (k_verify=4,  WY4)
cap≥4  → drafts.len()≥4 → step_verify_dflash     (k_verify=5..16, sequential decode_batched)
cap=15 → drafts.len()=15 → step_verify_dflash    (k_verify=16, sequential — MAX)
```

GDN dispatch для K verify (decode_batched):
```
K=2  → gdn_wy2
K=3  → gdn_wy3
K=4  → gdn_wy4
K=17 → gdn_wy17  ← НЕДОСТИЖИМ при γ=16 (k_verify max=16)
K=5..16 → sequential per-token loop  ← путь DFlash при cap≥4
```

**ATLAS_VERIFY_PREFILL_SSM=1** (стеш, 2026-06-29): заменяет sequential K≥5 в step_verify_dflash на:
- Phase1: K serial single-token calls (conv1d + attention) → заполняет gdn_bufs
- Phase2: WY4 batch GDN recurrence over K tokens
- Phase3: gated RMS norm + output projection + FFN

Применимо для cap=3 (K=4 уже WY4), cap=7 (K=8), cap=11 (K=12), cap=15 (K=16).

**k_verify = drafts.len() + 1** (`verify_dflash_step.rs:182`). WY17 скомпилирован, но неактивен. K=2 через DFlash недостижим без изменений scheduler/dispatch.

### ✅ DFlash acceptance — почему низкий

| Сценарий | Acceptance K=2 | Причина |
|---|---|---|
| Thinking chain | 0% (отключён) | `!inside_thinking` → decode без spec |
| Code output (T=0), window=64 | ~7% | Requires exact argmax match; ctx_window ограничивает качество |
| Code output (T=0), window=512 (short seq) | ~18-25% | При полном ctx покрытии, early steps ≈25% ≈ vLLM |
| Code output (T=1) | ~3.5% | T=1 sampling diversity |
| Paper baseline | ~87%/токен | T=0, 4B target, B200 — принципиально другой кейс |

**Исследованные причины и их статус:**

| Гипотеза | Тест | Результат |
|---|---|---|
| poisoned ctx (k-1→0 fix) | fix1 | 2.9% → 6.9% ✅ помогло |
| accept ctx gap (dflash_accept_append) | в незакоммиченном коде | 6.9% → ??? (не финализировано) |
| capture layer offset (-1/0/+1) | A/B/C bench | одинаково, offset+1 хуже |
| ctx_window 64 vs 512 | bench (2026-06-29) | **64→7%, 512→18%** (НЕ одинаково — более ранний тест был до фикса ctx gap) |
| ctx_window 1 / 512 / 2048 | bench window sweep | 1→4.4%, 64→~7%, 512→17.6%, 2048→17.6% (плато) — ctx ИСПОЛЬЗУЕТСЯ |
| cap=1 vs cap=15 | K=2 vs K=16 | K=16 медленнее в 5×, acceptance не лучше |
| vLLM PR #40898 layer shift | offset+1 | 3.3% — хуже |
| RoPE на ctx K | code review | Корректно: абсолютные позиции, единый проход ✅ |
| k_norm асимметрия ctx vs noise | code review | Симметрично — один ops::rms_norm на весь k_buf ✅ |
| FC concat order Atlas vs vLLM | code review | Оба ascending [L0,L15,L30,L45,L60] ✅ |
| K=2 = 2 слоя target (не полный) | code review | K=2 = 2 токена в батче, все 94 слоя target запускаются ✅ |
| Thinking contamination | code review | НЕТ: propose гейтован !inside_thinking, ctx_hidden_acc не трогается во время thinking ✅ |
| Long bench 5.2% vs short 18% | per-batch trace | Сложность контента (complex code/edge cases), не баг системы |
| Early-seq acceptance (window=512) | 20-step trace | 5/20=**25%** — вплотную к vLLM 27.2% |

**Итоговый вывод (2026-06-29):** Gap Atlas 7% vs vLLM 27% объясняется двумя независимыми факторами:
1. **ctx_window=64 слишком мало.** При window=512 и коротких последовательностях acceptance=18-25%, что вплотную к vLLM 27.2%. Разница 25% vs 27.2% — в пределах погрешности измерений (разные промпты).
2. **ctx gap fixes незакоммичены.** Без `dflash_accept_append` + row-0 fix acceptance деградирует до 2.9%. С фиксами в working tree: 6.9% (window=64). Нужно закоммитить.
3. **Propose время растёт с window** (45ms @ window=64, 200ms+ @ eff_ctx=500). Поэтому window=512 неприменимо для длинных последовательностей. Решение — квантование drafter (П1).

**⚠️ vLLM aeon baseline на тех же моделях (2026-06-29):**

Запуск: `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (vLLM 0.23.0+aeon, SM121a, DFlash+SWA)  
Target: `Qwen3.6-27B-NVFP4` (локальная), drafter: `z-lab/Qwen3.6-27B-DFlash` (локальный)  
Метрики из `/metrics` endpoint.

**Бенч 1** (bench_findings.py, без enable_thinking, max_tokens=700): 3 запроса × 139 токенов, T=0  
→ `drafts=414, draft_tokens=6210, accepted=1692` → **per-token acceptance: 27.2%, τ=4.09**

**Бенч 2** (прямые запросы, enable_thinking=True, max_tokens=1500): 5 запросов × 1500 токенов, T=0  
→ delta: `drafts=1425, accepted=6100` → **per-token acceptance: 28.5%, τ=4.28**

| Метрика | vLLM aeon | Atlas |
|---|---|---|
| per-token acceptance | **~28%** | ~6% |
| τ (avg accepted/step) | **~4.1** | ~0.06 |
| tok/s (бенч 2) | **43** | ~12 |

**Анализ расхождения условий (27% vs 6% не апples-to-apples):**

| Параметр | vLLM бенч 1 | Atlas |
|---|---|---|
| Длина генерации/запрос | **139 токенов** | **1395 токенов** |
| Thinking в acceptance | нет (стрим без thinking) | нет (DFlash отключён при thinking, токены стриптятся) |
| accept ctx gap bug | сбрасывается каждые 139 токенов | накапливается весь запрос (незакоммичен!) |
| enable_thinking | нет | не применимо |

**Ключевая гипотеза — accept ctx gap bug объясняет разрыв:**  
В vLLM каждый запрос = 139 токенов → при 27% acceptance за запрос накапливается gap ≈ 37 → сбрасывается на следующем запросе. В Atlas = 1395 токенов в одном запросе → gap накапливается до ~42+ без сброса → RoPE позиции драфтера расходятся → acceptance коллапсирует с 6.9% (начало) до ~3% (конец последовательности).

**Что подтверждено:** чекпоинт рабочий, NVFP4 не мешает (активации остаются BF16). Главный подозреваемый — незакоммиченный `dflash_accept_append` fix (accept ctx gap).

**Что нужно проверить:** запустить Atlas bench с коротким промптом (`max_tokens=200`, ~139 токенов) — если acceptance в начале последовательности ≈ 27%, ctx_gap — корень проблемы.

### ✅ Propose timing

При `ATLAS_DFLASH_CTX_WINDOW=512` (default): 43ms (eff_ctx=25) → 200ms+ (eff_ctx=500+).  
При `ATLAS_DFLASH_CTX_WINDOW=64`: **45ms flat** на любом seq_len.  
**Всегда использовать `ATLAS_DFLASH_CTX_WINDOW=64`.**

Breakdown propose (ctx_window=64, BF16 drafter, 5.94 GB весов):

| компонент | время | % |
|---|---|---|
| fc GEMM (step0) | ~1ms | 2% |
| layers attention × 5 | ~8ms | 18% |
| layers FFN × 5 | ~18ms | 40% |
| lm_head + argmax | ~18ms | 40% |

Propose = 74% от bandwidth предела (178 GB/s). **Единственный путь к speedup — квантование весов драфтера.**

---

## Benchmark Plan

### Промпт

```
Write a Python implementation of a min-heap with insert, extract_min, heapify (O(n) build from list), and peek.
Show time complexity for each operation. Include a complete working demo with at least 10 elements.
```

**Параметры:** `temperature=0, max_tokens=700`  
**Ожидаемый output:** ~150-250 thinking tokens + ~300-450 output tokens = ~450-700 total  
**Почему:** детерминированный код + немного математики (сложности). T=0 → воспроизводимые прогоны. Короткий, но достаточный для стабильного tok/s.

### Метрики

| Метрика | Источник | Описание |
|---|---|---|
| `tok/s` | wall_time / tokens | общий throughput включая thinking |
| `TTFT_ms` | streaming: время до первого chunk | время до первого токена |
| `total_tok` | из response | все токены (thinking + output) |
| `thinking_tok` | символы до `</think>` / 3.7 | токены в thinking chain |
| `output_tok` | total_tok - thinking_tok | токены в output |
| `accept/step` | K2 gauge / step log | средний accept per verify call |
| `accept_rate_%` | из K2 gauge | per-token acceptance % |
| `verify_mult` | mtp_gate log | verify_time / decode_time |
| `propose_ms` | DFlash step log | только для DFlash конфигов |

### Конфигурации

| # | Config | Key flags | Notes |
|---|---|---|---|
| A | No-MTP baseline | (без --speculative) | pure decode |
| B | MTP K=2 fp8 | `--speculative --mtp-quantization fp8 --num-drafts 1` | текущий лучший |
| C | MTP K=2 bf16 | `--speculative --num-drafts 1` | сравнение квантования |
| D | DFlash cap=1 | `--dflash CTX_WINDOW=64 CAP=1 --max-batch-size 1` | K=2 verify |
| E | DFlash cap=15 | `--dflash CTX_WINDOW=64 CAP=15 --max-batch-size 1` | K=16 verify |

**Прогоны:** 1 warmup (cache fill) + 2 измерения, median tok/s.

### Таблица результатов (2026-06-28, T=0, min-heap prompt)

Промпт: "Write a Python implementation of a min-heap with insert, extract_min, heapify (O(n) build from list), and peek. Show time complexity for each operation. Include a complete working demo with at least 10 elements."

Условия: T=0 (argmax), 1 warmup + 2 runs, median. max_tokens=700 (сервер не ограничивает — выдаёт до stop).

| Config | T | cap | k_verify | tok/s | TTFT_ms | total_tok | accept/step | accept_% | Notes |
|---|---|---|---|---|---|---|---|---|---|
| A: No-MTP | 0 | — | — | 13.6 | 241 | 1351 | — | — | baseline |
| B: MTP fp8 | 0 | 1 | 2 | **15.7** | 247 | 1351 | 0.28 | 28% | лучший config |
| C: MTP bf16 | 0 | 1 | 2 | 15.3 | 245 | 1351 | 0.23 | 22.5% | fp8 head лучше |
| D: DFlash K=2 T=0 (bug) | 0 | 1 | 2 | 10.4 | 332 | 1395 | 0.03 | 2.9% (14 батч) | poisoned ctx bug |
| D: DFlash K=2 T=0 (fix1) | 0 | 1 | 2 | 11.9 | 323 | 1067 | 0.07 | 6.9% (7 батч) | после k-1→0 фикса |
| D: DFlash K=2 T=0.6 | 0.6 | 1 | 2 | 11.2† | 332 | 1074† | 0.04 | 3.5% (8 батч) | †дисперсия 803–1346 |
| D: DFlash K=2 T=1.0 | 1.0 | 1 | 2 | 11.7† | 331 | 914† | 0.03 | 3.2% (4 батч) | †дисперсия |
| E: DFlash K=16 seq T=0 | 0 | 15 | 16 | ~2.2 | 331 | 1395 | — | — | sequential 912ms, UNUSABLE |
| F4: DFlash K=4 prefill_ssm | 0 | 3 | 4 | 7.0 | 324 | 1006 | — | — | хуже K=2, acceptance не масштабируется |
| F8: DFlash K=8 prefill_ssm | 0 | 7 | 8 | 4.2 | 325 | 1395 | 0.03 | — | деградация × |
| F12: DFlash K=12 prefill_ssm | 0 | 11 | 12 | — | — | — | — | — | не запускали |
| F16: DFlash K=16 prefill_ssm | 0 | 15 | 16 | — | — | — | — | — | не запускали |
| G: DFlash K=2 offset+1 | 0 | 1 | 2 | 10.4 | 330 | 1395 | 0.03 | 3.3% | capture=[2,17,32,47,62] — хуже offset=-1 |

accept_% — среднее по K2 summary батчам (каждый = 100 verify-шагов) в окне timed runs. DFlash отключается во время thinking, поэтому K2 gauge меряет только output-phase токены.

**MTP/DFlash на thinking:** оба ОТКЛЮЧАЮТСЯ пока модель думает (`!inside_thinking`, `mod.rs:340`). Thinking токены стрипаются из стрима (`emit_step.rs:85`). K2 gauge меряет acceptance только на output фазе.

**DFlash поддержка температуры T>0 (РЕАЛИЗОВАНО и протестировано):** `verify_k2_step.rs` поддерживает T>0 через rejection sampling. Drafter всегда argmax (T=0 greedy) → `p_draft(d)=1` → acceptance probability = `p_target(draft_token, T)`. Реализация: при T>0 и `dflash_verify_raw_argmax=true` — D2H copy position-0 target logits (vocab × BF16 ≈ 496KB) → numerically stable softmax at T → xorshift64 sample u → accept if `u < p_target`. При T=0 — прежний argmax equality check. При rejection emitted token = pipeline argmax (приближение, не точный spec decode).

**DFlash T>0 rejection sampling:** при T>0 acceptance немного выше (3.2–3.5% vs 2.9% при T=0), но общий tok/s аналогичен. Высокая дисперсия total_tok при T>0 (модель иногда думает дольше).

**accept_%**: среднее по K2 summary батчам (каждый = 100 verify-шагов) в окне timed runs. DFlash отключён во время thinking → K2 gauge меряет только output-phase.

**E (sequential K=16):** verify_mult=12.47, verify=912ms. `decode_batched(16)` = sequential 16 per-token kernel launches. Ожидаемое ускорение от `ATLAS_VERIFY_PREFILL_SSM=1` — в процессе тестирования.

---

## Следующие шаги (приоритизировано)

### 🔴 П1: Квантование весов DFlash drafter

**Почему:** propose = 74% от bandwidth предела. Лm_head (248k vocab, 2.54 GB) — самый дорогой компонент.

| Квантование | Propose | Δ tok/s (cap=1) |
|---|---|---|
| BF16 (текущий) | 45ms | baseline 9.4 |
| FP8 / INT8 | ~28ms | ~10.4 |
| FP4 / INT4 | ~19ms | ~11.0 |

**При высоком acceptance (если thinking disable работает):** cap=15 + INT8 → propose 28ms + verify 120ms = 148ms, ~10 accept/step → **~67 tok/s**. Пока гипотетично.

---

### 🟢 П2: CUDA graph для propose

При фиксированном ctx_window=64 shape статичен. Ожидаемый выигрыш: ~2-5ms (45ms→40ms). Небольшой.

---

## Важные замечания

### Сборка (обязательно)
```bash
RUSTFLAGS="-L /tmp/nccl-stubs" \
ATLAS_TARGET_MODEL=qwen3.6-27b \
ATLAS_TARGET_QUANT=nvfp4 \
ATLAS_TARGET_HW=gb10 \
LD_LIBRARY_PATH="/home/isolo/.cache/uv/archive-v0/V0RWp7iPS0kW3pWE/nvidia/nccl/lib" \
~/.cargo/bin/cargo build --release -p spark-server
```
Без `ATLAS_TARGET_MODEL=qwen3.6-27b` → компилируется `qwen3-next-80b-a3b` → краш при старте.

### DFlash — подводные камни
- `ATLAS_DFLASH_DRAFT_CAP=1` (default) намеренно консервативен
- `ATLAS_DFLASH_CTX_WINDOW=64` — всегда ставить
- `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` — для отладки в eager mode
- `--max-batch-size 1` при тестировании — иначе OOM (SSM MTP pools: 5.4 GB при batch=1, 24 GB при batch=8)
- WY17 (K=17) недостижим при γ=16 — это нормально, sequential K=16 корректен
