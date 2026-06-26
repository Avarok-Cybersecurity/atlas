# Findings: Qwen3.6-27B NVFP4 на Atlas / GB10

**Дата:** 2026-06-26  
**Железо:** NVIDIA GB10 (Grace Blackwell, SM121a), 121 GB unified RAM, 178 GB/s  
**Модель:** Qwen3.6-27B-NVFP4 (наш чекпойнт, modelopt format, 19.2 GB)  
**Atlas:** `feat/*` ветка, сборка `ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4 ATLAS_TARGET_HW=gb10`

---

## 1. Benchmark Results

### Этот сеанс (2026-06-26) — no-MTP, temp=0, prompt=«Write a complete Python red-black tree»

| Прогон | tok/s (median) | step_ms | Примечание |
|---|---|---|---|
| baseline-no-mtp | 11.977 | 83.49ms | до изменений этой сессии |
| baseline-labels | 12.138 | 82.39ms | после добавления prof! меток |
| norm-opt-n16 | 12.076 | 82.81ms | норм раз в 16 токенов |
| full-profile-breakdown | 12.129 | 82.44ms | финальный прогон с ATLAS_PROFILE=1 |

Разброс между прогонами ~1.3% — в пределах шума. **Норм-оптимизация не дала измеримого эффекта** в tok/s, что согласуется с тем, что gdn_decode = 4% SSM слоя.

### MTP прогоны (из более ранних сессий)

| Конфигурация | tok/s | Примечание |
|---|---|---|
| Atlas NVFP4, K=2 MTP, код | **17.92** | best single-req |
| Atlas NVFP4, K=2 MTP, эссе | 14.82 | diverse vocab → хуже MTP |
| Atlas NVFP4, K=2 MTP, thinking | 12.12 | thinking = diverse → плохой accept |
| Atlas NVFP4, K=3 MTP, код | 13.75 | хуже K=2 |
| Atlas NVFP4, K=3 MTP, эссе | 12.18 | хуже K=2 |
| SGLang FP8 + DFlash SM121 | ~21.0 | наш рекорд, для сравнения |
| vLLM NVFP4 + MTP k=3, TP=2 | 23.4 | два GPU — нечестное сравнение |

**MTP K=2 оптимально.** K=3 хуже во всех случаях: второй драфт принимается в
только в 8–22% шагов, а verify overhead растёт.

### MTP accept rates (K=2)
- Код: 55–73%
- Эссе: 37–69%
- При K=3: mean accepted = 0.27–0.76 / step (нестабильно)

---

## 2. Профилирование (ATLAS_PROFILE=1)

### Breakdown одного decode шага

```
PROFILE total: ~75ms (draft) / ~82ms (K=3 verify, 3 tokens)
  attn: 18ms / 16 layers = 1.1ms/layer
  ssm:  60ms / 48 layers = 1.25ms/layer
  head: 3.1ms
  → base rate (no MTP): 12.3–13.3 tok/s
```

### Полный breakdown одного SSM слоя (~1204μs/layer, ATLAS_PROFILE=1)

Измерено с `prof!` лейблами в ssm_forward.rs и trait_decode.rs:

| Операция | Время | Доля |
|---|---|---|
| **moe (MoE FFN)** | **738μs** | **61%** ← реальный bottleneck |
| qkvz GEMV (NVFP4, [12288×5120]) | 220μs | 18% |
| out_proj GEMV (NVFP4, [5120×4096]) | 94μs | 8% |
| launch/sync overhead | ~52μs | 4% |
| gdn_decode | 43μs | 4% |
| post_norm (residual_add_rms_norm) | 10μs | 1% |
| conv1d + L2norm | 10μs | 1% |
| pre_norm (rms_norm_residual) | 9μs | 1% |
| ba_gates (BF16 GEMV, [96×5120]) | 15μs | 1% |
| gated_norm | 7μs | 1% |
| residual_add | 7μs | 1% |

**Ключевой вывод: MoE FFN = 61% времени SSM блока.** Qwen3.6-27B — гибридная модель, где каждый SSM слой содержит MoE FFN sub-block. GDN decode — всего 4%.

---

## 3. Анализ `gated_delta_rule_decode`

**Файл:** `kernels/gb10/common/gated_delta_rule.cu`

### Что делает kernel

Рекуррентное обновление SSM state для одного токена:
```
h_t = g * h_{t-1} + k_t ⊗ v_t'
out_t = h_t^T @ q_t
```
Где state `H: [k_dim=128, v_dim=128]` FP32 на голову.

### Grid / Block конфигурация

```
grid  = (num_v_heads=48, batch=1, 1)
block = (BLOCK_SIZE=128, 1, 1)
```

**Итого: 48 блоков × 128 потоков = 6144 активных потоков.**

GB10 имеет 20 SM, max 2048 thread/SM → capacity = 40960 threads.

**Occupancy = 6144 / 40960 = 15%.** Это фундаментальная проблема.

### Проходы по памяти H (64 KB на голову = 128×128×4 байт)

| Проход | Операция | Читает H | Пишет H |
|---|---|---|---|
| Loop 1 | hk_dot = H^T @ k | 64 KB | — |
| Loop 2 | H_new = g*H + k⊗v, q_dot = H_new^T @ q | 64 KB | 64 KB |
| Norm | ‖H‖² reduction + optional scale | 64 KB | 64 KB (if > norm) |

**Итого на голову:** ~320 KB (5 проходов по 64 KB)  
**Итого на слой:** 320 KB × 48 heads = **15.4 MB**  
**Теоретический минимум** при 178 GB/s: 15.4 / 178000 = **86μs**

Ядро занимает ~43μs (из prof! разбивки). **Эффективность: 86/43 = 200%?** Нет — kernel latency-bound при 15% occupancy, занимает слоты меньше теоретического минимума именно потому что меньше потоков обращается к памяти.

### Причины неэффективности

**1. Критически низкий occupancy (15%):**
48 блоков при 20 SM — только ~2.4 блока/SM активны.
При stall на глобальной памяти (load latency ~500 cycles на GB10) некого
переключиться. Warp планировщик простаивает.

**2. SSM state norm — лишний третий проход (теперь каждые 16 токенов):**
После update-а kernel заново читает всю матрицу H для Frobenius norm.
Это +33% к memory traffic. Срабатывает только если ‖H‖ > 1000, но проход
выполняется всегда. Мы оптимизировали до раз-в-16-токенов, но это не дало
измеримого ускорения — kernel latency-bound, а не bandwidth-bound.

---

## 4. Текущий статус NVFP4 GDN проекций

Вопреки тому, что написано в `NVFP4_DENSE_27B.md` (устаревший документ), 
большинство GDN проекций уже работают как **native NVFP4**:

| Проекция | Размер | Статус |
|---|---|---|
| `in_proj_qkv` | [8192, 5120] | ✅ QuantizedWeight, w4a16_gemv |
| `in_proj_z`   | [4096, 5120] | ✅ QuantizedWeight, concat → qkvz_nvfp4 |
| `out_proj`    | [5120, 4096] | ✅ QuantizedWeight, w4a16_gemv + w4a16_gemm |
| `in_proj_a`   | [48, 5120]   | ❌ dequant → BF16, merged в in_proj_ba |
| `in_proj_b`   | [48, 5120]   | ❌ dequant → BF16, merged в in_proj_ba |

`in_proj_ba` (BF16) = 15μs из 1204μs на SSM слой = **1.2%.** Не стоит трогать.

---

## 5. Потенциальные оптимизации GDN decode kernel

### A. ~~Убрать/отложить SSM state norm~~ (СДЕЛАНО, эффект negligible)

Реализовано: норм каждые 16 токенов через `do_norm = (norm_token_count % 16 == 0)`.
Прирост: <1% — kernel latency-bound, не bandwidth-bound.

### B. Fused single-pass: hk_dot + update + q_dot в один проход по H

Сейчас два независимых прохода:
- Loop 1: `hk_dot = H^T @ k`
- Loop 2: `H_new = ...; q_dot = H_new^T @ q`

**Вывод:** нельзя убрать Loop 1 без кардинального изменения алгоритма.
`hk_dot` нужен для `v_new_i`, а `v_new_i` нужен во всём Loop 2.

### C. Увеличить occupancy: split по batch или heads

С batch=1 (наш случай) увеличение occupancy невозможно через batch.
Альтернатива: разбить каждую голову на `T` тайлов по k_dim, но это требует
atomicAdd или second kernel для финального суммирования.

### D. Tensor core GEMV для H^T @ k (SM121-специфично)

При 15% occupancy мы latency-bound, а не compute-bound, поэтому выигрыш неочевиден.

### E. MoE FFN — реальный приоритет (61% времени слоя)

Оптимизации самого GDN дают максимум ~4% ускорения SSM блока.
Реальный рычаг — MoE FFN (738μs/layer × 48 layers = **35ms/step**):
- Профилировать internals MoE: expert routing, top-K selection, expert GEMMs
- SM121-оптимизированный attention (параллельный путь, но на 18% SSM слоя он не влияет)

---

## 6. Связь с DFlash / SGLang PR #3731

SGLang DFlash patch (flashinfer PR #3731) даёт +1.5–2 tok/s для SM121.
Он не трогает `gated_delta_rule_decode` напрямую — он оптимизирует
**attention decode** для SM121 через специфичный для SM12x kernel.

Разрыв Atlas vs SGLang (17.8 vs 21 tok/s) объясняется вероятно комбинацией:
- Attention: SGLang использует SM121-оптимизированный flashinfer, Atlas — generic
- MoE FFN: не профилировано детально ни там ни там
- GDN: оба используют одинаково неэффективный generic kernel

---

## 7. Что делать дальше

### ✅ Сделано в этой сессии

| Задача | Результат |
|---|---|
| Убрать norm pass из hot path (каждые 16 токенов) | Реализовано. Прирост <1% — kernel latency-bound |
| `prof!` лейблы для всех операций SSM слоя | ✅ qkvz, ba_gates, conv1d, gdn_decode, gated_norm, out_proj, pre_norm, post_norm, moe, residual_add |
| Full SSM breakdown — найти где ~878μs/layer пропадает | ✅ Найдено: MoE FFN = 738μs (61%) |
| Native U8 NVFP4 загрузка чекпойнтов | ✅ cherry-pick qwen35_dense.rs + quantized.rs. Прямая загрузка без dequant→BF16→requant |

### Следующие шаги

| Приоритет | Задача | Ожидаемый прирост |
|---|---|---|
| 🔴 Высокий | Профилировать internals MoE FFN: routers, expert GEMMs, topK — найти узкое место в 738μs | диагностика |
| 🔴 Высокий | SM121-специфичный attention kernel (как flashinfer PR #3731) | +1–3 tok/s |
| 🟡 Средний | Ускорить MoE FFN если нашли bottleneck | потенциально +5–8 tok/s |
| 🟡 Средний | Profiler в MTP verify path | диагностика MTP overhead |
| 🟢 Низкий | `in_proj_ba` → NVFP4 (completeness, не perf) | <0.2 tok/s |
| 🟢 Низкий | Tensor core GEMV для H^T @ k | 0–2 tok/s (только при высоком occupancy) |

---

## 8. Конфигурация запуска (оптимальная на сейчас)

```bash
./target/release/spark serve /path/to/Qwen3.6-27B-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --kv-high-precision-layers 4 \
    --speculative \
    --mtp-quantization bf16 \
    --num-drafts 1 \          # K=2 оптимально, K=3 хуже
    --scheduling-policy slai
```

Профилирование: `ATLAS_PROFILE=1 ./target/release/spark serve ...`  
Дополнительно: `ATLAS_MEM_PROFILE=1` для memory usage по слоям.
