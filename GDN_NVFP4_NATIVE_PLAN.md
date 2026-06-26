# Plan: Native NVFP4 GDN Decode Kernels

## Цель

Убрать оставшийся BF16 dequant для SSM/GDN проекций в пути decode, чтобы все
веса GDN-слоёв оставались в 4-бит на GPU во время инференса.

---

## Текущее состояние (после нашего NVFP4_DENSE_27B патча)

> **Важно:** Документ `NVFP4_DENSE_27B.md` устарел. Большинство GDN проекций уже
> работают как native NVFP4 — это было добавлено в ходе того же патча.

В `qwen35_dense.rs` (lines 313–370) уже существует **native NVFP4 path**:

| Проекция | Размер | Статус |
|---|---|---|
| `in_proj_qkv` | [8192, 5120] | ✅ `QuantizedWeight` (NVFP4 GEMV + GEMM) |
| `in_proj_z` | [4096, 5120] | ✅ `QuantizedWeight` (concat в qkvz_nvfp4) |
| `out_proj`   | [5120, 4096] | ✅ `QuantizedWeight` (NVFP4 GEMV + GEMM) |
| `in_proj_a`  | [48, 5120]   | ❌ dequant → BF16 → `in_proj_ba` |
| `in_proj_b`  | [48, 5120]   | ❌ dequant → BF16 → `in_proj_ba` |

Итого в BF16 остаётся только `in_proj_ba` (merged alpha+beta) — **640 KB** из ~400+ MB
суммарных весов GDN слоёв (~0.16% bandwidth).

### Benchmark (наш чекпойнт, GB10, Atlas с MTP)

| Режим | tok/s |
|---|---|
| Atlas NVFP4 + MTP (текущий) | **17.8** |
| Atlas NVFP4 без MTP (baseline) | ~13.5 |
| SGLang FP8 + DFlash (наш рекорд) | 21.0 |
| vLLM NVFP4 + MTP k=3 (TP=2, два GPU) | 23.4 |

---

## Что реально остаётся сделать

### Шаг 1: `in_proj_a` / `in_proj_b` → native NVFP4

**Файл:** `crates/spark-model/src/weight_loader/qwen35_dense.rs`

Проблема: `load_ssm_proj` вызывает `dequant_nvfp4_to_bf16` для `in_proj_a/b`,
затем `interleave_ba` сливает их в BF16 буфер `ba_dense`.

Решение: Добавить ветку `native_nvfp4 == true` для A/B проекций:

```rust
// Вместо:
let in_proj_a = load_ssm_proj(&format!("{la}.in_proj_a"), nv, h)?; // dequant → BF16
let in_proj_b = load_ssm_proj(&format!("{la}.in_proj_b"), nv, h)?; // dequant → BF16
let ba_dense  = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;

// Для native_nvfp4 path:
let in_proj_a_qw = quantized_auto(store, &format!("{la}.in_proj_a"), gpu, Nvfp4Variant::Standard)?;
let in_proj_b_qw = quantized_auto(store, &format!("{la}.in_proj_b"), gpu, Nvfp4Variant::Standard)?;
```

**Проблема с interleave:** `interleave_ba` работает с BF16 буферами через GPU kernel.
Interleaving в NVFP4-домене нетривиален — packed bits, группы по 16 элементов.

**Варианта два:**
- A) Держать `in_proj_a` и `in_proj_b` как **раздельные** `QuantizedWeight`,
  вызывать два отдельных `w4a16_gemv` в decode вместо одного.
- B) Написать CUDA kernel `nvfp4_interleave_ba` — сложнее, но единый буфер.

Рекомендуется **Вариант A** (проще, не требует нового CUDA кода).

### Шаг 2: Добавить поля в `SsmWeights` / `Qwen3SsmLayer`

**Файл:** `crates/spark-model/src/weight_map/expert.rs`

```rust
pub struct SsmWeights {
    pub in_proj_qkvz: DenseWeight,
    pub in_proj_ba:   DenseWeight,        // остаётся для BF16/FP8 моделей
    // Новые поля:
    pub in_proj_a_nvfp4: Option<QuantizedWeight>,
    pub in_proj_b_nvfp4: Option<QuantizedWeight>,
    // ...
}
```

**Файл:** `crates/spark-model/src/layers/qwen3_ssm/mod.rs`

Не требует изменений в `Qwen3SsmLayer` — достаточно добавить поля в `SsmWeights`.

### Шаг 3: Dispatch в decode path

**Файл:** `crates/spark-model/src/layers/qwen3_ssm/trait_decode_batched.rs`  
Строки 220–252 (BA projection loop)

```rust
// Сейчас (всегда BF16):
ops::dense_gemv(ctx.gpu, self.dense_gemv_k, normed_t,
    &self.ssm.in_proj_ba, ba_out, ba_size as u32, h as u32, stream)?;

// Стать должно (Вариант A, два раздельных GEMV):
if let (Some(ref a_qw), Some(ref b_qw)) =
    (&self.ssm.in_proj_a_nvfp4, &self.ssm.in_proj_b_nvfp4)
{
    // a: [nk, h] → alpha output [nk]
    ops::w4a16_gemv(ctx.gpu, self.w4a16_gemv_k, normed_t,
        a_qw, ba_out, nk as u32, h as u32, stream)?;
    // b: [nv, h] → beta output [nv]
    ops::w4a16_gemv(ctx.gpu, self.w4a16_gemv_k, normed_t,
        b_qw, ba_out.offset(nk * /* element size */), nv as u32, h as u32, stream)?;
} else {
    ops::dense_gemv(ctx.gpu, self.dense_gemv_k, normed_t,
        &self.ssm.in_proj_ba, ba_out, ba_size as u32, h as u32, stream)?;
}
```

**Внимание:** нужно проверить layout ожидаемый `compute_gdn_gates` — alpha/beta
порядок в `ba_out` буфере должен совпадать с тем, что сейчас делает `interleave_ba`.

### Шаг 4: Тестирование

```bash
# Перед изменениями: сохранить baseline
curl -s http://localhost:8888/v1/chat/completions ... → save tok/s

# Пересобрать
ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4 ATLAS_TARGET_HW=gb10 \
cargo build --release -p spark-server --no-default-features --features cuda

# Запустить и сравнить
./target/release/spark serve /path/to/Qwen3.6-27B-NVFP4 --port 8888 \
    --speculative --mtp-quantization bf16 --scheduling-policy slai
```

Ожидаемый прирост: **~0.1–0.3 tok/s** (in_proj_ba всего 640 KB из 400+ MB).

---

## Профилирование (ATLAS_PROFILE=1, реальные данные)

Запуск: одна декодирующая последовательность, 80 токенов, модель Qwen3.6-27B-NVFP4.

```
PROFILE tok=26: total=81.2ms  attn=18.1ms(16)  ssm=60.0ms(48)  head=3.1ms
  SSM qkvz:     220μs/слой  (w4a16_gemv NVFP4 [12288×5120])
  SSM ba_gates:  16μs/слой  (dense_gemv BF16  [64×5120])
  SSM остаток:  ~460μs/слой (gated_delta_rule_decode + conv1d + norms + out_proj)
```

### Breakdown по времени на один SSM слой (~0.7ms)

| Операция | Время | Доля |
|---|---|---|
| qkvz GEMV (уже NVFP4) | 220μs | 31% |
| ba_gates GEMV (BF16) | **16μs** | **2.3%** |
| gated_delta_rule_decode + conv1d + norms + out_proj | ~460μs | **66%** |

**Ключевой вывод:** `in_proj_ba` в NVFP4 сэкономит 10–12μs из 700μs (<2%). Это шум.

Реальный bottleneck — `gated_delta_rule_decode` kernel (сам SSM state update,
рекуррентное вычисление, не GEMV). Именно туда нужно смотреть для прироста.

### Почему Atlas 17.8 < SGLang FP8 21.0

81.2ms/token = ~12.3 tok/s без MTP. С MTP (accept rate 54–73%) → ×1.45 → ~17.8 tok/s.

SGLang FP8 DFlash: 21 tok/s с MTP на single GPU. Разница объясняется
`gated_delta_rule_decode` — у SGLang с DFlash патчем более эффективный GDN kernel
(PR #3731 flashinfer, SM121-specific). Atlas использует стандартный generic kernel.

---

## Риски

| Риск | Вероятность | Митигация |
|---|---|---|
| `interleave_ba` ожидает определённый layout, раздельные GEMV дадут неверный | Средняя | Проверить `compute_gdn_gates` ожидаемый layout; запустить unit test |
| `w4a16_gemv` для N=16/48 (малые N) неэффективен, kernel overhead > savings | Высокая | Замерить vs dense_gemv; если хуже — откатить |
| CUDA graph invalidation (фиксированные адреса SSM buffers) | Низкая | `ba_out` буфер не меняется, только откуда читаем weights |
| `quantized_auto` для `in_proj_a` вернёт неверный вариант | Низкая | Явно передавать `Nvfp4Variant::Standard` |

---

## Решение: стоит ли делать?

**Ответ: нет, не сейчас. in_proj_ba NVFP4 не стоит усилий.**

Профилирование показало: ba_gates = 16μs из 700μs на слой = 2.3%. Экономия
от NVFP4: -10μs/слой × 48 слоёв = -480μs на decode шаг из 81.2ms = **<0.6%**.

**Реальные приоритеты по убыванию impact:**

1. **`gated_delta_rule_decode` kernel для SM121** — занимает ~65% SSM времени.
   Нужен профильный CUDA kernel под SM12x MMA (аналог DFlash патча для SGLang).
   Потенциал: +3–5 tok/s (сопоставимо с SGLang gap).

2. **MTP tuning** — текущий accept rate 54–73%. Попробовать `--num-drafts 2` (K=3)
   vs текущий K=2, посмотреть на Atlas issue #195.

3. **in_proj_ba NVFP4** — реализовать только если нужно для upstream PR
   (completeness), не для perf.

---

## Связанные PR в upstream

- **PR #198** (open): `w4a16_gemm_t_m128_bf16` — BF16 tensor-core prefill (+30% prefill),
  не decode. Не наш случай.
- Никаких open PR по BA projection NVFP4 — поле свободно если всё же решим делать.
