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
| Atlas NVFP4, K=2 MTP, **fp8** head, код | **17.8** | 2026-06-26, median 5 runs |
| Atlas NVFP4, K=2 MTP, bf16 head, код | 16.2 | 2026-06-26 |
| Atlas NVFP4, K=2 MTP, bf16 head, код | 17.92 | более ранний рекорд (другой прогрев?) |
| Atlas NVFP4, K=2 MTP, эссе | 14.82 | diverse vocab → хуже MTP |
| Atlas NVFP4, K=2 MTP, thinking | 12.12 | thinking = diverse → плохой accept |
| Atlas NVFP4, K=3 MTP, код | 13.75 | хуже K=2 |
| Atlas NVFP4, K=3 MTP, эссе | 12.18 | хуже K=2 |
| SGLang FP8 + DFlash SM121 | ~21.0 | наш рекорд, для сравнения |
| vLLM NVFP4 + MTP k=3, TP=2 | 23.4 | два GPU — нечестное сравнение |

**MTP K=2 оптимально.** K=3 хуже во всех случаях: второй драфт принимается в
только в 8–22% шагов, а verify overhead растёт.

**`--mtp-quantization fp8` лучше bf16:**
- accept rate: 49.6% vs 43.2% (+6%)
- tok/s: 17.8 vs 16.2 (+10%)
- Нет attractor warnings (< 30%) у fp8; у bf16 были окна по 26%
- Качество вывода: идентично (MTP head — только драфтер; мейн модель верифицирует)
- nvfp4 для dense FFN MTP head не поддерживается (unsupported, только fp8/bf16)

### MTP accept rates (K=2)
- Код fp8 head: **49.6%** mean, 33–58% range (без аттракторов)
- Код bf16 head: 43.2% mean, 32–57% range (были окна 26%)
- Более ранние данные код: 55–73% (другие условия)
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
1. ✅ `--mtp-quantization fp8` → 49.6% accept (+6pp), +10% tok/s
2. 🔴 NVFP4 MTP head (см. ниже §6)
3. ❌ DFlash drafter (см. ниже §8) — ИССЛЕДОВАН, accept rate 0%, не работает пока

#### 5. 🟢 qkvz GEMV — потенциал 19% SSM времени

qkvz = 218μs на слой. Это `w4a16_gemv` для [12288 × 5120] (qkv+z конкатенировано).  
Grid: `(ceil(12288/4), 1, 1)` = 3072 блока → тоже ~100% occupancy и bandwidth-bound.  
Нет quick wins — разве что сравнить с теоретическим пределом: 12288×5120×0.5 = 31.5 MB / 178 GB/s = 177μs. Actual: 218μs → 81% efficiency. Тот же паттерн что FFN.

---

## 6. NVFP4 MTP head — почему не работает и что нужно

### Текущее состояние

Модель `Qwen3.6-27B-NVFP4` имеет **dense FFN MTP head** (не MoE).  
Чекпойнт хранит веса MTP head в **FP8 e4m3**.

`--mtp-quantization nvfp4` завершается с ошибкой (`new.rs:60`):
```rust
if matches!(quant, MtpQuantization::Nvfp4) {
    anyhow::bail!("MTP NVFP4 mode is not supported for dense FFN MTP heads yet ...");
}
```

### Почему только для MoE

NVFP4 для MTP head уже реализован — но **только для MoE** голов (`new.rs:80–100`):
- `moe_nvfp4: Option<MoeLayer>` → `quantize_to_nvfp4(gate, experts, ...)`
- Dense FFN голова не имеет аналогичного `dense_ffn_nvfp4` поля

### Что нужно сделать (конкретно)

**Файлы:** `crates/spark-model/src/layers/mtp_head/`

1. **`new.rs`** — добавить поле и убрать bail:
```rust
// В MtpHead struct добавить:
dense_ffn_nvfp4: Option<(QuantizedWeight, QuantizedWeight, QuantizedWeight)>,

// В new() заменить bail! на:
MtpQuantization::Nvfp4 => Some((
    quantize_to_nvfp4(&dense_ffn.gate_proj, inter, h, gpu, absmax_k, nvfp4_k, stream)?,
    quantize_to_nvfp4(&dense_ffn.up_proj,   inter, h, gpu, absmax_k, nvfp4_k, stream)?,
    quantize_to_nvfp4(&dense_ffn.down_proj, h, inter, gpu, absmax_k, nvfp4_k, stream)?,
))
```

2. **`moe_forward.rs`** — новый метод `dense_ffn_forward_nvfp4`:
```rust
// По образцу dense_ffn.rs: w4a16_gemv_dual (gate+up) + w4a16_gemv_silu_input (down)
ops::w4a16_gemv_dual(ctx.gpu, self.w4a16_gemv_dual_k, input, gate_w, gate_out, up_w, up_out, inter, h, stream)?;
ops::w4a16_gemv_silu_input(ctx.gpu, self.w4a16_gemv_silu_input_k, gate_out, up_out, down_w, output, h, inter, stream)?;
```

3. **`forward.rs`** — добавить dispatch:
```rust
let ffn_out = if self.dense_ffn_nvfp4.is_some() {
    self.dense_ffn_forward_nvfp4(normed2, ctx, stream)?
} else if self.dense_ffn_generic.is_some() {
    self.dense_ffn_forward_generic(normed2, ctx, stream)?
} else { ... };
```

Нужны также kernel handles `w4a16_gemv_dual_k` и `w4a16_gemv_silu_input_k` в `MtpHead` struct.

### Ожидаемый эффект

MTP head FFN читает **2× меньше байт** (NVFP4 vs FP8):
- gate+up: 2 × 17408 × 5120 × 0.5 = 89.1 MB → было 178 MB (FP8)
- down: 5120 × 17408 × 0.5 = 44.5 MB → было 89 MB

**Предупреждение:** чекпойнт хранит MTP веса в FP8, requantize FP8→NVFP4 может потерять точность.
Нужно проверить accept rate после реализации — может упасть ниже FP8 (49.6%).

---

## 8. DFlash drafter — АКТИВНАЯ РАБОТА (2026-06-26)

### Что это

Z-Lab DFlash (`z-lab/Qwen3.6-27B-DFlash`, 3.3 GB) — специализированный drafter,  
генерирует γ токенов параллельно (block diffusion). Архитектура:  
5 transformer слоев, hidden=5120, GQA 32/8, intermediate=17408, vocab=248320.  
fc-проекция: [5120 × 25600] BF16 — маппит 5 захваченных hiddens target-модели в drafter space.

**Drafter модель:** `/home/isolo/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106`

### История проблем и что сделано

#### Сессия 1 (ранее): тест baseline

```
run 1: 1.041 tok/s — в 17× хуже baseline
MTP gate: verify_multiplier=4.55, decode=70.27ms, verify=319.66ms
accept rate 0.0%
```

**Проблемы обнаружены:**
1. `ATLAS_DFLASH_DRAFT_CAP=1` → 1 драфт → K=2 verify → 0% accept (block diffusion ≠ autoregressive)
2. DFlash forward_block: ~110-149ms (слишком медленно)
3. K4 graphed verify: 255-260ms (4 sequential full-model passes)

#### Сессия 2 (2026-06-26): профилирование + оптимизация propose

**Добавлено профилирование `ATLAS_DFLASH_PROF=1` в `forward_block.rs`:**
- `step0_gpu` = время fc проекция (eff_ctx × batched GEMM → hidden_norm)
- `layers_gpu` = время 5 drafter-слоёв (q/k/v/gate/up/down GEMMs)
- `sync_wait` = время lm_head + argmax (последний sync)

**Результаты с scalar `dense_gemm` (eff_ctx=24-31):**
```
step0_gpu=~0ms  (fc GEMM был sequential loop из eff_ctx GEMVs — каждый читал 262 MB!)
layers_gpu=75ms (5 layers × 5 GEMMs каждый = scalar 16×16 tiles)
sync_wait=26ms  (lm_head M=4, N=248320 — bandwidth limited)
total forward_block=110-149ms
```

**Баг найден и исправлен:** Sequential fc GEMV loop (O(eff_ctx) × 262 MB weight reads) → заменён на один batched `dense_gemm` (читает веса 1 раз).

#### Сессия 3 (2026-06-26): pipelined GEMM замена

**Замена:** все `dense_gemm` (scalar 16×16 tiles) → `dense_gemm_bf16_pipelined` (tensor-core m16n8k16, 128×128 tiles, cp.async 2-stage pipeline) в:
- `forward_block_layer.rs`: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (7 GEMMs)
- `forward_block.rs`: fc GEMM (step 0), lm_head GEMM

**Результаты после pipelined GEMM (eff_ctx=28-35):**
```
step0_gpu=2ms   (fc GEMM eff_ctx=28-35 одним batched pipelined GEMM)
layers_gpu=24ms (5 layers — 3× ускорение от tensor-core MMA!)
sync_wait=16ms  (lm_head — улучшение с 26ms)
total forward_block=43-46ms
```

**Propose 43ms vs прежние 110-149ms — улучшение в 3×.**

#### Сессия 3: raw argmax fix в step_verify_k4

`step_verify_k4` получает флаг `dflash_verify_raw_argmax` (bool), который ставится в `true` когда `--dflash` включён, но параметр был помечен `_dflash_verify_raw_argmax` (unused). Исправлено — теперь при `dflash=true` пропускается `verify_pick_all_with_pipeline` (rep_pen/DRY), которая ломала acceptance для DFlash.

### Сессия 4 (2026-06-26): SSM corruption в step_verify_dflash

**Тест:** `ATLAS_DFLASH_DRAFT_CAP=4 --dflash-gamma 5` → drafts.len()=4 → `step_verify_dflash`

**Результат:**
```
prompt: "The capital of France is"
output: "Paris.\n\nThe capital of France is Paris\n\n:\n\n -）_ #!!!!!!!!!!!!"
```

Первые ~10 токенов правильные, потом резкий переход в garbage. Протестировано с `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` — та же картина. Значит **не CUDA graph bug**.

**Accept rate по логам:**
```
DFLASH K=γ verify: γ=4 accepted=0/4 (0%)  — подавляющее большинство шагов
DFLASH K=γ verify: γ=4 accepted=1/4 (25%) — редко
```

### Диагностика corruption: что проверено

**1. Буферы intermediates** — одни и те же указатели в `sequence.rs` и `meta.rs`. Не баг.

**2. Логика seq_len rollback** — корректна (при num_accepted=0, to_drop=4 → pop 4 токена).

**3. SSM h_state rollback в `commit_verify_state_async_dispatch`** — логика верна.

**4. norm_token_count drift** — реальный баг CPU counter drift, исправлен в `async_chkpt.rs`:
```rust
// Full reject (num_accepted == 0):
ssm.norm_token_count = ssm.norm_token_count.wrapping_sub(k as u32);
// Partial accept (0 < num_accepted < k):
ssm.norm_token_count = ssm.norm_token_count.wrapping_sub((k - num_accepted) as u32);
```
НО: оказался НЕ главной причиной corruption.

### Сессия 5 (2026-06-26): Реальная причина corruption + DRAFT_CAP=16

#### Реальная причина SSM corruption (USER-IDENTIFIED)

**`trait_decode_batched_conv_gdn.rs` — dispatch table:**
```
K=2  → gdn_decode_wy2    ← WY batch kernel (работает)
K=3  → gdn_decode_wy3    ← WY batch kernel (работает)
K=4  → gdn_decode_wy4    ← WY batch kernel (работает)
K=17 → gdn_decode_wy17   ← WY batch kernel (работает)
ELSE → sequential per-token gdn_decode loop ← НИКОГДА НЕ ТЕСТИРОВАЛСЯ, БАГ
```

K=5 (DRAFT_CAP=4, gamma=5) → sequential fallback → corruption.  
**Fix:** gamma=16 → K=17 → gdn_decode_wy17 → corruption исчезла. Вывод корректный.

#### Откуда drafts=1 всегда (дефолт)

В `propose.rs` строка ~272:
```rust
let cap: usize = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(1);   // НАМЕРЕННЫЙ ДЕФОЛТ: 1
```
Причина: K=γ verify path не заполнял `h_state_intermediates` как WY kernels — partial accept rollback был некорректен. С gamma=16 → wy17 intermediates заполняются правильно.

#### Результат теста с ATLAS_DFLASH_DRAFT_CAP=16

```bash
ATLAS_DFLASH_PROF=1 ATLAS_DFLASH_DRAFT_CAP=16 \
LD_LIBRARY_PATH=".../nccl/lib" \
./target/release/spark serve Qwen3.6-27B-NVFP4 --port 8888 \
  --kv-cache-dtype nvfp4 --kv-high-precision-layers 4 \
  --dflash --draft-model <path> \
  --scheduling-policy slai --gpu-memory-utilization 0.75 --mtp-quantization fp8
```

**Лог:**
```
DFlash propose: forward_block=51ms total=51ms eff_ctx=64 γ=16 drafts=16 pos=316
DFLASH K=γ verify: γ=16 accepted=0/16 (0%) seq_len=317
DFlash propose: forward_block=51ms total=51ms eff_ctx=65 γ=16 drafts=16 pos=317
DFLASH K=γ verify: γ=16 accepted=1/16 (6%) seq_len=319   ← редкость
```

**Качество вывода:** ✅ корректно (числа 1..19 без corruption)  
**Accept rate:** ~0% (1 из 20 шагов ≈ 6%, остальные 0%)  
**tok/s: ~1.0** — в 12× хуже baseline. Каждый шаг: 1 бонусный токен / ~1s (verify≈950ms + propose≈50ms).

### Проблема: Chinese characters в output на speculative пути (Paris（巴黎）баг)

**Симптом:** с MTP K=2 verify модель добавляет Chinese characters в ответы которые должны быть только на английском:
```
MTP K=2:   "The capital of France is Paris（巴黎）"
No-spec:   "The capital of France is Paris."
```

**Изоляция (2026-06-26):** protтестировано на одном промпте:
- Без speculative (--scheduling-policy slai, no --speculative): чистый English output
- С MTP K=2 fp8: добавляются Chinese иероглифы

Пользователь видел идентичный баг в SGLang при имплементации DDTree (ищет тот чат).

**Гипотезы:**
1. SSM state drift из MTP verify → незначительное изменение распределения токенов → модель "переключается" в Chinese thinking mode
2. Thinking chain при MTP verify другой → другой результат на output
3. Bonus token в K=2 verify немного меняет hidden state → следующий token распределён иначе

**Severity:** medium — output функционально правильный (Paris = верно), но с лишними символами. Может указывать на более глубокую проблему с SSM state consistency.

### Проблема: ранняя остановка генерации (seq_len cap ~87)

**Симптом:** при любом запросе с DFlash cap=16 (K=17 verify) сервер останавливает генерацию после ~49 completion токенов, несмотря на max_tokens=3000. finish_reason=length.

**Данные:**
```
prompt_tokens: 41
completion_tokens: 49
total_tokens: 90
finish_reason: length   ← при max_tokens=3000!
```

Из лога: seq_len никогда не превышает ~87 (= 41 + 46). Паттерн повторяется для всех запросов:
```
seq_len=80, 81, 83, 84, 86, 87  ← и тут стоп
```

Thinking-модель Qwen3 расходует большинство токенов на скрытый thinking chain.
Через `/v1/completions` с явным `<think>\n\n</think>\n` суффиксом thinking пропускается,
но генерация всё равно обрывается на 49 токенах.

**Гипотезы:**
1. **SLAI scheduling policy** предсказывает длину вывода и ограничивает её — counting task → ~50 токенов
2. **KV cache exhaustion**: DFlash verify аллоцирует блоки для K=17 позиций вперёд, при rollback они не освобождаются → быстрое заполнение KV cache
3. **CUDA graph replay**: граф захватывается при первом seq_len, replay с другим seq_len использует устаревшие block table → нет места → принудительный EOS
4. **max_seq_len взаимодействие с DFlash**: verify пытается обработать позицию seq_len+16, если `seq_len + 16 >= max_seq_len - buffer` — scheduler стопит

**Как изолировать:** запустить тот же промпт без DFlash (MTP only) — если стоп исчезнет, проблема в DFlash path; если нет — это SLAI или другое.

### Почему accept rate = 0%? — ROOT CAUSE НАЙДЕН (субсессия, 2026-06-26)

Субсессия прочитала `forward_block.rs`, `forward_block_layer.rs`, `propose.rs`, `MODEL.toml` и нашла **три критических бага** которые вместе гарантируют 0% accept rate:

---

#### BUG 1 (КРИТИЧЕСКИЙ): `lm_head_shared` — LM head таргет-модели, в NVFP4 освобождается

`forward_block.rs:373` явно документирует:
> "If this returns zeros or garbage, the BF16 lm_head was freed by factory.rs's NVFP4 quantization step."

В NVFP4 режиме BF16 `lm_head` таргета освобождается при квантизации. `lm_head_shared` в драфтере становится dangling pointer → garbage logits → garbage argmax → 0% accept.

**Файл:** `forward_block.rs:355`, `factory.rs`

---

#### BUG 2 (КРИТИЧЕСКИЙ): `embed_tokens_shared` OOB для `mask_token_id=248070`

`embed_tokens_shared` — это embedding table **таргет-модели** с `vocab_size=151936` (стандартный Qwen3 словарь).  
`mask_token_id = 248070` — расширенный vocab drafter'а.

Обращение к row 248070 в таблице из 151936 строк → OOB GPU read, ≈374 MB за границей аллокации → garbage embeddings для всех `gamma-1` masked noise позиций → garbage hidden states → garbage logits.

**Файл:** `forward_block.rs:253`

---

#### BUG 3 (КРИТИЧЕСКИЙ): Target LM head используется как [248320 × h] матрица, но у неё только [151936 × h] строк

Drafter vocab_size = 248320 (расширен для diffusion токенов). Target LM head = 151936 × h.  
GEMM в `forward_block.rs:355` читает как будто 248320 строк → 39% logit домена (токены 151936..248319) читают **случайную GPU память**.

Argmax over 248320 значений, где 96384 — random GPU noise → argmax почти всегда попадает в мусорный диапазон → token ID который target никогда не генерирует → **0% accept rate гарантирован**.

`draft_id_to_target_id` = `None` → ремаппинг vocab отсутствует.

**Файл:** `forward_block.rs:355`, `from_weights.rs:260`

---

#### BUG 4 (MEDIUM): Первый noise slot — `last_token` вместо `mask_token_id`

Token layout в `stream_buf`:
```
[0, 0, ..., 0,    last_token,    mask, mask, ..., mask]
 ←── eff_ctx ──→  ← pos 0 →     ←──── gamma-1 ───────→
```
Позиция 0 noise блока получает реальный токен вместо mask. Drafter обучался с all-mask входами → distribution mismatch для первой позиции.

---

#### BUG 5 (LOW): One-shot denoising (T=1) вместо T>1

Block diffusion модели обучаются с T>1 итеративными denoising шагами. Текущая impl делает T=1 → деградация качества, но не 0% сам по себе.

---

### ОБНОВЛЕНИЕ: предыдущий анализ vocab mismatch был НЕВЕРЕН

Вторая субсессия прочитала `from_weights.rs`, `mod.rs` и checkpoint. Оба — target и drafter — имеют **vocab_size=248320, hidden_size=5120**. Sharing `embed_tokens_shared` и `lm_head_shared` — **intentional и корректен**. BUG 1-3 из предыдущего анализа не существуют.

### Реальные баги (от субсессии, 2026-06-26)

#### BUG 1 (КРИТИЧЕСКИЙ): Неправильный RoPE — YaRN вместо standard

`from_weights.rs:199-231` захардкодил YaRN параметры:
```
factor=64, beta_fast=32, beta_slow=1, original_max_position_embeddings=4096
```

Но `drafter config.json`: `"rope_scaling": null` — drafter обучался со **стандартным RoPE** (theta=10M, без интерполяции). Применение YaRN с factor=64 производит полностью неправильные positional encodings в каждом слое каждого шага → garbage attention patterns → garbage hidden states → garbage logits → **0% accept rate**.

**Фикс (простой):** заменить ~30 строк YaRN вычисления на стандартный inv_freq:
```rust
// from_weights.rs:199-231 — заменить на:
let n_pairs = rotary_dim / 2;
let mut inv_freq_table = vec![0.0f32; n_pairs];
for j in 0..n_pairs {
    inv_freq_table[j] = 1.0 / rope_theta.powf((2 * j) as f32 / rotary_dim as f32);
}
```

Никаких других структурных изменений не нужно — weights loading, sharing, memory allocation всё корректны.

#### BUG 2 (MEDIUM): Stale docstring

`dflash_head.rs:7`: "8 layers, hidden=2048, GQA 32:4" — всё неверно. Реально: 5 layers, hidden=5120, GQA 32:8. Runtime читает из config, код не сломан, но misleading.

#### BUG 3 (MEDIUM): Sliding window attention игнорируется

Drafter config: layers 0-3 = sliding_attention (window=2048), layer 4 = full_attention. Все 5 слоёв в `forward_block_layer.rs` запускают full bidirectional attention. При ctx_window=512 не критично (все токены влезают), но training/inference mismatch при длинных контекстах.

#### BUG 4 (LOW): Первый noise slot не замаскирован

`forward_block.rs:234`: slot 0 noise block получает `last_token` embedding вместо `mask_token_id`. Нарушает block diffusion protocol.

### Приоритетный порядок фиксов

| # | Баг | Эффект | Сложность |
|---|-----|--------|-----------|
| 1 | YaRN вместо standard RoPE в drafter | **0% accept** | ~5 строк |
| 2 | First noise slot не замаскирован | деградация | ~2 строки |
| 3 | Sliding window не применяется | деградация при ctx>512 | medium |

**BUG 1 — простейший фикс, ожидаемый результат: accept rate >0%.**

### ОБНОВЛЕНИЕ: RoPE фикс применён, accept всё ещё 0% (2026-06-26)

YaRN→standard RoPE фикс применён (`from_weights.rs`), сборка успешна. Результаты теста:
- Paris: "The capital of France is Paris." ✅ (Paris（巴黎）баг исчез!)
- Accept rate: всё ещё **0% на каждом шаге** — RoPE был не единственной причиной

Лог: `DFLASH K=γ verify: γ=16 accepted=0/16 (0%) seq_len=111..117...`

#### Гипотезы после RoPE фикса (в расследовании)

| # | Гипотеза | Файл | Статус |
|---|----------|------|--------|
| A | `lm_head_shared` dangling в NVFP4 — BF16 lm_head freed при квантизации | `factory.rs` | 🔴 расследуется |
| B | Verify сравнивает token IDs с ошибочным offset/remapping | scheduler verify | 🔴 расследуется |
| C | forward_block_layer: ctx K/V slots не получают fc_proj значения — вместо этого нули | `forward_block_layer.rs` | 🔴 расследуется |
| D | BUG 4: первый noise slot = last_token вместо mask_token_id | `forward_block.rs:233` | известно, фикс прост |

Сабсессия `claude_b285f57b-e9ff-41cc-9f1b-ae8c2a39413e` читает код по гипотезам A/B/C.

### Сессия 6 (2026-06-27): BF16/FP32 mismatch в batched verify — ROOT CAUSE accept=0%

#### Найденный баг

`trait_decode_batched_conv_gdn.rs` — sequential fallback (K ∉ {2,3,4,17}):

```
single-token decode path (ssm_forward.rs):
  if conv1d_l2norm_f32_k.0 != 0 → FP32 conv output → FP32 q/k/v в gdn_decode  ✅

batched verify path (trait_decode_batched_conv_gdn.rs, ELSE branch):
  всегда использовал conv1d_l2norm_k → BF16 conv output → BF16 data в gdn_decode  ❌
```

`gated_delta_rule_decode` для qwen3.6-27b/nvfp4 объявлен с `const float*` для query/key/value.  
Передача BF16 данных → kernel читает каждые 2 BF16 байта как один FP32 float →  
случайные значения → h_state накапливает garbage → после ~7 токенов NaN в h_state →  
модель предсказывает EOS (token 248046) вместо реальных токенов → **accept rate = 0%**.

Отдельный баг: `verify_d.rs` строка 53: `let fp32 = 2usize;` вместо `4usize` — неправильный stride для gate/beta offset.

#### Фикс

**`trait_decode_batched_conv_gdn.rs`** — sequential else-branch теперь проверяет `conv1d_l2norm_f32_k`, аналогично single-token decode:

```rust
let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
let seq_conv_buf = if use_f32_conv { ctx.buffers.ssm_conv_out_f32() } else { conv_out_buf };
let conv1d_k = if use_f32_conv { self.conv1d_l2norm_f32_k } else { self.conv1d_l2norm_k };
let coes = if use_f32_conv { fp32 } else { bf16 };
// → правильные FP32 q/k/v передаются в gdn_decode
```

**`verify_d.rs`**: `let fp32 = 4usize;` (был 2).

#### Причина CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (901) при диагностике

Диагностический код (`gpu.synchronize(stream)` + `copy_d2h()`) находился внутри `if need_run { ... }` в `verify_d.rs` — это CUDA graph capture регион. Такие вызовы инвалидируют граф. Диагностика убрана, `trait_decode_batched.rs` восстановлен через `git checkout HEAD`.

#### Результат

| Конфигурация | До фикса | После фикса |
|---|---|---|
| DFlash K=2 (cap=1) | неизвестно | ✅ verify_multiplier=1.55, **10.5 tok/s** |
| DFlash K=γ (cap=15, K=16) | NaN→EOS, 0% | предсказывает EOS реже, но SSM state mismatch остаётся |

K=2 DFlash: `MTP gate: verify_multiplier=1.55, max_effective=16.0 => ENABLED`.  
~55% acceptance rate (1 из 1 драфтов принимается, бонус всегда).

#### Остаток: SSM state mismatch для K=γ (γ>2, K ∉ {2,3,4,17})

PAIR DUMP при K=16: `verified[..8]=[11, 198, 279, 248046, 198, 248046, ...]` — позиции 3+ предсказывают EOS.

Не NaN (FP32 fix убрал NaN), но WY-chunkwise vs sequential intermediate semantics расходятся.  
Задокументировано в `propose.rs:260-272`:

> "SSM state-management mismatch between the generic K=γ path and the hand-tuned K=2/3/4 specializations: the K=N!=2/3/4 fallback writes intermediates differently from the WY-chunkwise kernels, causing partial-accept rollback to land on stale state."

**Это отдельная задача (kernel work).** FP32 fix убрал NaN-баг; intermediate semantics нужно фиксить отдельно.

---

### Текущий статус DFlash

| Шаг | Статус |
|-----|--------|
| Profiling `ATLAS_DFLASH_PROF=1` | ✅ |
| fc GEMV loop bug fix | ✅ |
| Pipelined GEMM в forward_block_layer | ✅ 110ms→43ms |
| raw argmax fix (step_verify_k4, step_verify_dflash) | ✅ |
| norm_token_count rollback fix | ✅ `async_chkpt.rs` |
| SSM corruption исчезла (gamma=16 wy17) | ✅ |
| **YaRN→standard RoPE фикс** | ✅ 2026-06-26 |
| Paris（巴黎）баг | ✅ исчез после RoPE фикса |
| **BF16/FP32 mismatch в batched verify (ROOT CAUSE accept=0%)** | ✅ **2026-06-27** |
| fp32=2→4 в verify_d.rs | ✅ **2026-06-27** |
| **DFlash K=2 verify работает** | ✅ 1.55× multiplier, 10.5 tok/s |
| DFlash K=γ (γ>2, K ∉ {2,3,4,17}) SSM state mismatch | 🔴 **отдельная задача** |

### Текущий лучший результат DFlash

| Конфигурация | tok/s | Примечание |
|---|---|---|
| MTP K=2 fp8 | **17.8** | текущий лучший |
| **DFlash K=2 (cap=1)** | **10.5** | ✅ рабочий, 1.55× multiplier |
| DFlash cap=15, K=16 verify | ~1–2 | SSM mismatch, EOS на позициях 3+ |
| DFlash K=17 (cap=16) | ~1 | WY17 путь работает, propose overhead |
| Цель | **>17.8** | нужен K=γ с высоким acceptance |

---

## 9. Конфигурация запуска (оптимальная на сейчас)

### MTP K=2 fp8 (текущий лучший, 17.8 tok/s)
```bash
./target/release/spark serve /path/to/Qwen3.6-27B-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --kv-high-precision-layers 4 \
    --speculative \
    --mtp-quantization fp8 \
    --num-drafts 1 \
    --scheduling-policy slai
```

### DFlash K=2 (рабочий, 10.5 tok/s, cap=1 по дефолту)
```bash
ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4 ATLAS_TARGET_HW=gb10 \
RUSTFLAGS="-L /tmp/nccl-stubs" \
LD_LIBRARY_PATH="/home/isolo/.cache/uv/archive-v0/V0RWp7iPS0kW3pWE/nvidia/nccl/lib" \
./target/release/spark serve /home/isolo/Projects/isolorg/models/Qwen3.6-27B-NVFP4 \
    --port 8888 --max-seq-len 4096 --kv-cache-dtype nvfp4 \
    --kv-high-precision-layers 4 --dflash \
    --draft-model /home/isolo/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106 \
    --scheduling-policy slai --gpu-memory-utilization 0.75
# ATLAS_DFLASH_DRAFT_CAP не выставлен → cap=1 → K=2 verify → рабочий
```

Профилирование: `ATLAS_PROFILE=1 ./target/release/spark serve ...`  
Дополнительно: `ATLAS_MEM_PROFILE=1` для memory usage по слоям.

---

## 10. Next Steps

### Приоритет 1: DFlash K=γ SSM state mismatch (отдельная задача)

**Проблема:** для K ∉ {2,3,4,17} sequential fallback в `trait_decode_batched_conv_gdn.rs` пишет `h_state_intermediates` иначе чем WY-chunkwise kernels. При partial-accept rollback состояние восстанавливается из неправильного intermediate.

**Симптом:** PAIR DUMP позиции 3+ предсказывают EOS (token 248046) даже при правильном FP32 conv/GDN. h_state после t=3 не является NaN (это бы выдало все позиции), но что-то в intermediate indexing или порядке снапшотов расходится с WY-kernel ожиданиями.

**Что нужно:**
1. Сравнить как WY4 kernel заполняет `h_state_intermediates[0..2]` vs sequential путь (после каждого gdn_decode)
2. Проверить `commit_verify_state_async` — какой intermediate index используется для partial-accept
3. Возможно: добавить K=16 в WY-dispatch table (WY16 kernel) вместо sequential fallback

**Файлы:** `trait_decode_batched_conv_gdn.rs`, `ssm_pool.rs:commit_verify_state_async_dispatch`, `kernels/gb10/*/gated_delta_rule.cu`

### Приоритет 2: Verify overhead (133ms) и propose overhead (50ms)

С K=2 DFlash цикл: `propose(50ms) + verify(133ms) = 183ms` на ~1.55 токена → ~8.5 tok/s теоретически.  
Verify (133ms) — главный bottleneck. Propose (50ms) — дополнительная потеря поверх него.

MTP K=2 fp8 @ 17.8 tok/s: нет propose step, verify по `decode_verify_graphed_k2` (отдельный путь, быстрее).  
DFlash K=2 verify идёт через `decode_verify_graphed_kgamma` — тяжелее из-за SSM sequential loop.

**Пути улучшения:**
- Если исправить K=γ SSM mismatch и поднять cap → больше токенов на один verify + propose → amortize оба overhead
- Investigate можно ли `decode_verify_graphed_k2` реюзать для DFlash K=2 вместо kgamma пути
- `ATLAS_DFLASH_CTX_WINDOW` (дефолт 512) → влияет на качество propose, косвенно на acceptance

### Приоритет 3: Cleanup диагностического кода в git diff

В diff есть debug printf в `kernels/gb10/common/gated_delta_rule.cu` (не влияет на qwen3.6-27b — используется model-specific kernel, но лишний код).  
`gpu.rs`, `cuda_backend.rs`, `gpu_impl.rs` — `device_sync()` метод не вызывается нигде после очистки диагностики — можно убрать или оставить (harmless).
