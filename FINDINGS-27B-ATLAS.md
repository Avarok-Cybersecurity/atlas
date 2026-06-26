# Findings: Qwen3.6-27B NVFP4 на Atlas / GB10

**Дата:** 2026-06-26  
**Железо:** NVIDIA GB10 (Grace Blackwell, SM121a), 121 GB unified RAM, 178 GB/s  
**Модель:** Qwen3.6-27B-NVFP4 (наш чекпойнт, modelopt format, 19.2 GB)  
**Atlas:** `feat/*` ветка, сборка `ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4 ATLAS_TARGET_HW=gb10`

---

## 1. Benchmark Results

| Конфигурация | tok/s | Примечание |
|---|---|---|
| Atlas NVFP4, K=2 MTP, код | **17.92** | best single-req |
| Atlas NVFP4, K=2 MTP, эссе | 14.82 | diverse vocab → хуже MTP |
| Atlas NVFP4, K=2 MTP, thinking | 12.12 | thinking = diverse → плохой accept |
| Atlas NVFP4, K=3 MTP, код | 13.75 | хуже K=2 |
| Atlas NVFP4, K=3 MTP, эссе | 12.18 | хуже K=2 |
| Atlas NVFP4 без MTP (baseline) | ~12.3 | из PROFILE (1000/81ms) |
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

### Breakdown одного SSM слоя (~0.73ms)

| Операция | Время | Доля |
|---|---|---|
| qkvz GEMV (NVFP4, [12288×5120]) | 230μs | 31% |
| ba_gates GEMV (BF16, [64×5120]) | 25μs | 3% |
| **gdn + conv1d + norms + out_proj** | **~475μs** | **65%** |

**Вывод:** bottleneck — `gated_delta_rule_decode`, не проекционные GEMV-ы.

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

Ядро занимает ~300–400μs (грубая оценка из 475μs "остатка" вместе с conv/norms).
**Эффективность использования bandwidth: ~22–29%.**

### Причины неэффективности

**1. Критически низкий occupancy (15%):**
48 блоков при 20 SM — только ~2.4 блока/SM активны.
При stall на глобальной памяти (load latency ~500 cycles на GB10) некого
переключиться. Warp планировщик простаивает.

**2. SSM state norm — лишний третий проход:**
После update-а kernel заново читает всю матрицу H для Frobenius norm.
Это +33% к memory traffic. Срабатывает только если ‖H‖ > 1000, но проход
выполняется всегда.

**3. Два независимых цикла вместо одного:**
Loop 1 (`hk_dot`) и Loop 2 (`update + q_dot`) читают H из global memory дважды.
H помещается в L1/L2 (64KB ≪ 64MB L2 GB10), но при 48 независимых блоках
каждый читает свой кусок H, и L2 может быть вытолкнут.

**4. Нет использования SM121 tensor cores:**
GEMV вида `H^T @ k` (матрица 128×128 на вектор 128) реализован скалярно через
`#pragma unroll 4`. На SM121 можно использовать `wgmma.mma_async` (16×16×16 FP32
accumulate), что даст 4-8× compute throughput для этой операции.

### Сравнение с Qwen3-Next (числа из комментария в коде)

Комментарий в `.cu` описывает `num_value_heads: 32` — это MoE вариант.
Для Qwen3.6-27B dense у нас `num_v_heads = 48` → ещё больше памяти на слой
(48 vs 32 heads), но те же 48 блоков запускаются → occupancy ещё ниже относительно.

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

`in_proj_ba` (BF16) = 25μs из 730μs на SSM слой = **3.4%.** Не стоит трогать.

---

## 5. Потенциальные оптимизации GDN decode kernel

### A. Убрать/отложить SSM state norm (быстрая победа, ~33% экономия traffic)

Проблема: третий проход по H выполняется при каждом токене, даже если ‖H‖ < 1000.

Решение — считать norm только каждые N токенов (например N=16):
```c
// В caller-е передаём флаг:
if (do_norm_check) { /* третий проход */ }
```
Или убрать совсем для коротких контекстов (проблема Stuffed Mamba актуальна только
при 6K+ токенов). Ожидаемый прирост: ~+15% к GDN throughput.

### B. Fused single-pass: hk_dot + update + q_dot в один проход по H

Сейчас два независимых прохода:
- Loop 1: `hk_dot = H^T @ k`
- Loop 2: `H_new = ...; q_dot = H_new^T @ q`

Можно сделать один проход используя `h_{t-1}` для hk_dot и `h_t` для q_dot:
```c
// Fused:
// hk_dot += H[j][tid] * k[j]          // используем старый H
// h_new = g*H[j][tid] + k[j]*v_new    // update in-place
// q_dot += h_new * q[j]               // используем новый H
// H[j][tid] = h_new                   // write-back
```
**Это уже реализовано в текущем коде** — Loop 2 делает именно это. Loop 1 избыточен
и нужен только для вычисления `hk_dot` перед `v_new_i`. Объединить нельзя: `hk_dot`
нужен для вычисления `v_new_i`, а `v_new_i` нужен во всём Loop 2.

**Вывод:** нельзя убрать Loop 1 без кардинального изменения алгоритма.
Альтернатива — предвычислять `hk_dot` через ланцет редукции (warp shuffle) за
первые 4 warp steps, освобождая warp bandwidth. Сложно.

### C. Увеличить occupancy: split по batch или heads

С batch=1 (наш случай) увеличение occupancy невозможно через batch.
Альтернатива: разбить каждую голову на `T` тайлов по k_dim:

```
grid = (num_v_heads * T, batch, 1)
// Каждый блок обрабатывает k_dim/T строк H
```
Проблема: `hk_dot` и `q_dot` требуют редукции по всем k_dim строкам →
нужна atomicAdd или second kernel для финального суммирования.

### D. Tensor core GEMV для H^T @ k (SM121-специфично)

H[128×128] @ k[128] можно вычислить через серию 16×16 matmul:
8 тайла по 16×16 = H[128×16] @ k[16] × 8, суммируя результаты.

SM121 поддерживает `wgmma.mma_async` для FP32 accumulate. Это потенциально
4-8× быстрее для compute-bound части. Но при 15% occupancy мы bandwidth-bound,
а не compute-bound, поэтому выигрыш неочевиден.

### E. Persistent kernel: все 48 SSM слоёв в одном запуске

Запуск 48 kernel-ов (по одному на SSM слой) генерирует 48 × launch overhead.
Persistent kernel обрабатывает все слои последовательно, держа H в L2:
```
grid = (num_v_heads, batch, num_ssm_layers)
// z-dim итерирует по слоям
```
**Проблема:** SSM state разный для каждого слоя и живёт в глобальной памяти.
Preload в shared memory невозможен (каждый слой 3.1 MB, smem 64 KB/SM).

---

## 6. Связь с DFlash / SGLang PR #3731

SGLang DFlash patch (flashinfer PR #3731) даёт +1.5–2 tok/s для SM121.
Он не трогает `gated_delta_rule_decode` напрямую — он оптимизирует
**attention decode** для SM121 через специфичный для SM12x kernel.

Разрыв Atlas vs SGLang (17.8 vs 21 tok/s) объясняется вероятно комбинацией:
- Attention: SGLang использует SM121-оптимизированный flashinfer, Atlas — generic
- GDN: оба используют одинаково неэффективный generic kernel

Потенциал оптимизации GDN decode для SM121: **+2–4 tok/s** при устранении
third norm pass и улучшении occupancy.

---

## 7. Что делать дальше

| Приоритет | Задача | Ожидаемый прирост |
|---|---|---|
| 🔴 Высокий | Убрать norm pass из hot path (каждые 16 токенов) | +1–2 tok/s |
| 🔴 Высокий | SM121-специфичный attention kernel (как flashinfer PR #3731) | +1–3 tok/s |
| 🟡 Средний | `in_proj_ba` → NVFP4 (upstream completeness, не perf) | <0.2 tok/s |
| 🟡 Средний | Profiler в MTP verify path (сейчас decode_profiled не покрывает batch K=3) | diag |
| 🟢 Низкий | Tensor core GEMV для H^T @ k в GDN decode | 0–2 tok/s (неопределённо) |

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
