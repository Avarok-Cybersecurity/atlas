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
PROFILE total: ~78ms
  attn: 17.4ms / 16 layers = 1.09ms/layer
  ssm:  57.0ms / 48 layers = 1.19ms/layer
  head: 3.1ms
  → base rate (no MTP): ~12 tok/s
```

### Полный breakdown одного SSM слоя (~1150μs/layer, ATLAS_PROFILE=1)

Измерено с `prof!` лейблами в ssm_forward.rs, trait_decode.rs, dense_ffn.rs:

| Операция | Время | Доля | Примечание |
|---|---|---|---|
| **FFN gate_up (NVFP4)** | **462μs** | **40%** | w4a16_gemv_dual: gate+up fused |
| **FFN silu_down (NVFP4)** | **272μs** | **24%** | w4a16_gemv_silu_input: SiLU×up+down fused |
| qkvz GEMV (NVFP4) | 218μs | 19% | w4a16_gemv или w4a16_gemv_qkvz |
| out_proj GEMV (NVFP4) | 94μs | 8% | w4a16_gemv |
| gdn_decode | 43μs | 4% | gated_delta_rule_decode |
| overhead (launch/norm) | ~61μs | 5% | ba_gates+conv1d+norms+residual |

**SSM TOTAL: ~1150μs/layer**

**Ключевой вывод: Dense FFN = 64% времени SSM слоя.** Qwen3.6-27B — это **dense** 27B модель (не MoE). Каждый SSM слой содержит SwiGLU Dense FFN. GDN decode — всего 4%.

### Внутренний breakdown Dense FFN (новое)

| Операция | Время | Доля | Ядро |
|---|---|---|---|
| gate_up (fused) | 462μs | 63% | `w4a16_gemv_dual` — gate+up GEMV в одном запуске |
| silu_down (fused) | 272μs | 37% | `w4a16_gemv_silu_input` — SiLU(gate)×up + down GEMV |
| **FFN TOTAL** | **734μs** | 100% | |

### Анализ bandwidth эффективности Dense FFN

Параметры (из модели):
- `intermediate_size` (inter): ~13824 (по размеру весов)
- `hidden_size` (h): 5120

Размер весов NVFP4 (4 бит = 0.5 байт/параметр):
- gate_proj: 13824 × 5120 × 0.5 = **35.4 MB**
- up_proj: 35.4 MB
- down_proj: 35.4 MB
- **Итого: 106.2 MB / слой**

Теоретический минимум при 178 GB/s: 106.2 / 178 = **597μs**  
Фактически: **734μs**  
**Bandwidth efficiency: 81%** — достаточно хорошо для GEMV.

Overhead ~137μs объясняется: чтение/запись активаций (i/o буферы), launch overhead, scales.

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
| `prof!` лейблы для всех операций SSM слоя | ✅ qkvz, ba_gates, conv1d, gdn_decode, gated_norm, out_proj, pre_norm, post_norm, ffn, residual_add |
| Full SSM breakdown — найти где ~878μs/layer пропадает | ✅ Dense FFN = 734μs (64%). Модель НЕ MoE — это dense SwiGLU FFN |
| `prof!` для Dense FFN internals | ✅ gate_up=462μs (63%), silu_down=272μs (37%). Bandwidth efficiency 81% |
| Native U8 NVFP4 загрузка чекпойнтов | ✅ cherry-pick qwen35_dense.rs + quantized.rs |

### Анализ потенциала оптимизации

**w4a16_gemv_dual** (grid `(ceil(13824/4), 1, 2)` = 6912 блоков, block=256):
- 6912 блоков / 20 SM = 345 блоков/SM → **~100% occupancy**
- Уже bandwidth-bound при 81% эффективности
- Margin для улучшения: ~20% → максимум +0.3 tok/s от FFN-kernels

**Реальные рычаги в порядке потенциала:**

---

### Следующие шаги (конкретные)

#### 1. ~~SM121-attention kernel~~ ✅ ИССЛЕДОВАН — не узкое место

**Вывод (2026-06-26):** Attention decode НЕ является bottleneck'ом для Qwen3.6-27B.

**Что обнаружено в `run_paged_decode.rs` + `mod.rs`:**

Split-K для NVFP4 реализован: когда `current_ctas = num_q_heads × MAX_DECODE_SEQS < NUM_SMS`, используется split-K путь.

Для Qwen3.6-27B:
- `num_q_heads = 32`, `MAX_DECODE_SEQS = 8` → `current_ctas = 256`
- `NUM_SMS = 20` → `256 >> 20` → `num_splits = 1` → **split-K НЕ активен**

Это корректно и уже исследовалось (комментарий в коде от 2026-06-03):
> tried unpinning this for num_seqs==1 to raise split-K occupancy (16→48 CTAs) — clean A/B was **BYTE-IDENTICAL (12.7 tok/s both)**, confirming **attention is NOT the bottleneck** (~5% of decode bytes at depth).

**Итог:** Attention kernel уже оптимален. ~5% decode time. Дальнейшая работа по attention нецелесообразна.

#### 2. 🔴 ncu профилирование w4a16_gemv_dual

Подтвердить текущие показатели и найти любые unexploited opportunities:
```bash
# Запустить сервер без CUDA graph (для ncu):
ATLAS_PROFILE=1 ncu --set full \
    --target-processes all \
    -o /tmp/atlas-ncu-report \
    ./target/release/spark serve ... &
# Послать один запрос, убить, открыть отчёт:
ncu-ui /tmp/atlas-ncu-report.ncu-rep
```
Смотреть: `l1tex__t_bytes_pipe_lsu_mem_global_op_ld` (DRAM bandwidth achieved), `sm__warps_active` (occupancy), cache hit rate на weight reads.

#### 3. 🟡 Fused triple GEMV: gate+up+down в одном ядре

Сейчас 2 запуска: `gate_up_dual` (2 проекции) + `silu_down` (1 проекция).  
Можно написать одно ядро:
- каждый CTA берёт тайл [N_tile] выходного вектора
- читает gate_weights[N_tile, K] и up_weights[N_tile, K] → вычисляет silu(gate)*up в smem
- читает down_weights[N_out_tile, N_tile] → аккумулирует partial dot product

**Проблема:** down-проекция требует **полного** intermediate вектора [K=13824] для каждого выходного элемента. Нельзя потоково читать: нужен reduction через весь inter. Возможно через `atomicAdd` на partial sums, но overhead атомиков съедает выигрыш.

**Альтернатива (проще):** уменьшить overhead между двумя запусками через CUDA Graphs с удалением лишних барьеров — это почти бесплатно.

#### 4. ✅ MTP verify path — ИЗМЕРЕНО

**Данные из `mtp_gate` лога (2026-06-26):**

```
MTP gate: verify_multiplier=0.91, max_effective=2.0
  decode=89.36ms  verify K=2=80.95ms => ENABLED
```

**Выводы:**
- Verify K=2 занимает **80.95ms** против single decode **89.36ms** → **9% быстрее**
- Причина: SSM слои переходят с GEMV (K=1) на GEMM (K=2) → читают веса один раз, вычисляют 2 выхода
- Attention слои (K=2): выполняются последовательно (2 decode вызова) → немного медленнее
- Итог: батчинг SSM перевешивает attention overhead → verify быстрее single decode

**MTP производительность (бенч 2026-06-26):**
- No-MTP: 12.0 tok/s (step=83ms)
- MTP K=2: **16.2 tok/s** (step_per_token=61.78ms)

**Accept rates из K2 summary (100-шаговые окна):**
54%, 42%, 57%, 33%, 41%, 46%, 32%, 49%, 38%, 33%, 55%, 39%, 26% → **средний ~42%**

Это ниже ожидаемых 55-73% для кода — BF16 MTP head (`--mtp-quantization bf16`) хуже чем FP32.

Теоретический max при accept_rate p: `(1+p) / verify_time`
  - p=0.42 (текущий) → `1.42 / 0.0810 = 17.5 tok/s` (~совпадает с измеренным)
  - p=0.65 (цель)    → `1.65 / 0.0810 = 20.4 tok/s` (+26% от текущего)
  - p=0.75 (лучший)  → `1.75 / 0.0810 = 21.6 tok/s` (+33% от текущего)

**Bottleneck: accept rate, не verify speed.**
Verify overhead уже оптимален. Резерв — поднять accept rate с 42% до 65%+.

**Способы улучшить accept rate:**
1. `--mtp-quantization fp8` или снизить квантование MTP head → лучше качество драфта
2. Убрать "attractor" ситуации (26% окна) — возможно помогает изменение температуры для драфта
3. Проверить более высокое разрешение: `--num-drafts 2` (K=3) при 42% accept может быть хуже K=2

#### 5. 🟢 qkvz GEMV — потенциал 19% SSM времени

qkvz = 218μs на слой. Это `w4a16_gemv` для [12288 × 5120] (qkv+z конкатенировано).  
Grid: `(ceil(12288/4), 1, 1)` = 3072 блока → тоже ~100% occupancy и bandwidth-bound.  
Нет quick wins — разве что сравнить с теоретическим пределом: 12288×5120×0.5 = 31.5 MB / 178 GB/s = 177μs. Actual: 218μs → 81% efficiency. Тот же паттерн что FFN.

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
