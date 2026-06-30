# SSM Параллельный Verify в DFlash

**Цель**: Заменить `decode_batched(k)` для SSM-слоёв в K=γ verify-пути на существующий
трёхфазный параллельный GDN, сократив время SSM verify с ~K× последовательных
обновлений h_state до одного `prefill_gdn_full` на слой.

**Скоп**: только SSM-слои. Attention остаётся как `decode_multi_seq` (отдельный PR).

**Ожидаемый результат**: SSM-компонента с ~730ms до ~100-250ms (K=16). При неизменном
attention ~222ms новый verify_multiplier ~4-6× (вместо 12.44×). Следующий шаг —
attention fix (PREFILL_VERIFY_PLAN.md) для достижения ~1.4-1.5×.

---

## Как работает трёхфазный SSM Prefill

Уже есть в продакшне для обычного prefill (`prefill_c.rs`). На каждый SSM-слой:

```
Фаза 1 — prefill_phase1:
  RMS norm + residual
  → QKVZ GEMM (NVFP4 w4a16)
  → Deinterleave + BA GEMM + GDN gates
  → Conv1d (обновляет conv_state)
  → L2 norm на Q,K
  → Копирование в gdn_bufs[qkv, gate_beta, z] по token_offset

Фаза 2 — prefill_gdn_full:
  Один GDN-kernel на все K токенов из gdn_bufs
  Читает h_state из ssm_state как начальное состояние
  Пишет output в gdn_bufs.output
  Обновляет h_state на месте (параллельная WY-рекуррентность)

Фаза 3 — prefill_phase3:
  Читает gdn_bufs[output, z] по token_offset
  → Gated RMS norm
  → Output projection GEMM
  → Dense SwiGLU FFN
  → Residual add → hidden
```

Сейчас в verify `decode_batched(k)` выполняет все три фазы K раз последовательно
(K отдельных шагов GDN-рекуррентности, без параллелизма). Фаза 2 — узкое место.

---

## Выбор GDN-ядра для K=16

Лесенка диспетча в `trait_prefill_gdn.rs`:

| Ядро | Условие | Хранение H-state |
|------|---------|------------------|
| `gdn_prefill_wy32_k` | total_len > 32 | shared memory (~84 KB) — ~30× быстрее |
| `gdn_prefill_persistent_wy4_k` | средние | shared memory |
| `gdn_prefill_split4_k` | fallback | global memory |

При K=16: **split4** (16 ≤ 32 порог для WY32). H-state в глобальной памяти — медленнее
WY32, но всё равно обрабатывает 16 токенов параллельно вместо 16 последовательных
decode-вызовов.

При увеличении γ до 32+ WY32 активируется автоматически.

---

## Ключевые файлы

| Файл | Роль |
|------|------|
| `crates/spark-model/src/model/trait_impl/verify_d.rs` | Основное изменение: SSM-ветка, construct gdn_bufs |
| `crates/spark-model/src/model/types.rs` | Добавить поле `verify_kgamma_ssm_graph` (новый кеш графа) |
| `crates/spark-model/src/model/impl_a1.rs` | Инициализировать новое поле кеша |
| `crates/spark-model/src/layers/qwen3_ssm/trait_prefill_phase1.rs` | Без изменений — проверить сигнатуру |
| `crates/spark-model/src/layers/qwen3_ssm/trait_prefill_gdn.rs` | Без изменений — проверить диспатч |
| `crates/spark-model/src/layers/qwen3_ssm/trait_prefill_phase3.rs` | Без изменений — проверить сигнатуру |
| `crates/spark-model/src/layer/transformer_layer.rs` | Без изменений — проверить сигнатуры трейта |

---

## План реализации

### Итерация 1 — Eager-путь (env-gate, без CUDA графов)

Gate через `ATLAS_VERIFY_PREFILL_SSM=1`. Запускать с `ATLAS_DFLASH_DEBUG_NO_GRAPH=1`,
чтобы при `CUDA_LAUNCH_BLOCKING=1` точно видеть ошибки ядер.

#### Шаг 1: Создать `gdn_bufs` в `verify_d.rs`

До цикла по слоям (после строки 163 `let ctx = ForwardContext { ... }`):

```rust
let gdn_bufs = GdnPrefillBuffers {
    qkv:       self.gdn_buf_qkv,
    gate_beta: self.gdn_buf_gate_beta,
    output:    self.gdn_buf_out,
    z:         self.gdn_buf_z,
    total_len: k,
};
```

`gdn_buf_*` — предаллоцированные device-указатели уровня модели (размер `max_batch_tokens`,
всегда ≫ K=16). Новые GPU-аллокации не нужны. Фиксированные адреса на каждый вызов →
безопасно для CUDA графов.

#### Шаг 2: Добавить SSM-ветку в цикл по слоям

Текущая структура (строки 191-246):
```
if layer_type == FullAttention {
    decode_multi_seq(k)  или  decode_batched(k)  [HSS-путь]
} else {
    decode_batched(k)   ← покрывает ВСЕ не-attention слои, включая SSM
}
```

Новая структура:
```rust
if layer_type == LayerType::FullAttention {
    // без изменений: decode_multi_seq или decode_batched (HSS)
} else if layer.is_ssm_layer() && use_prefill_ssm {
    // НОВОЕ: трёхфазный параллельный SSM
    layer.prefill_phase1(
        hidden, residual,
        k,
        seq.layer_states[layer_idx].as_mut(),
        &mut kv_cache,
        seq.seq_len,     // seq_len_start: абсолютная позиция первого verify-токена
        &mut seq.block_table,
        &mut seq.disk_block_ids,
        &mut seq.disk_last_offloaded_per_layer,
        0,               // kv_write_start
        &gdn_bufs,
        0,               // token_offset: пишем в gdn_bufs[0..k]
        &ctx, stream,
    )?;
    layer.prefill_gdn_full(
        seq.layer_states[layer_idx].as_mut(),
        &gdn_bufs,
        &ctx, stream,
    )?;
    layer.prefill_phase3(
        hidden, residual,
        k,
        &gdn_bufs,
        0,               // token_offset: читаем из gdn_bufs[0..k]
        &ctx, stream,
    )?;
} else {
    // существующий путь: decode_batched (не-SSM не-attention слои, или SSM fallback)
    layer.decode_batched(...)?;
}
```

Где:
```rust
let use_prefill_ssm = std::env::var("ATLAS_VERIFY_PREFILL_SSM")
    .ok().as_deref() == Some("1");
```

#### Шаг 2б: Заполнение h_state/conv_state intermediates (R7)

После `prefill_gdn_full` (h_state = итог всех k токенов) нужно заполнить
`h_state_intermediates[0..k-1]` и `conv_state_intermediates[0..k-1]`
для корректного rollback при partial accept.

**Схема для h_state** (добавить внутрь `else if layer.is_ssm_layer()` после phase2):

```rust
// 1. Сохранить итоговый h_state
let ssm_state = seq.layer_states[layer_idx].as_any_mut()
    .downcast_mut::<SsmLayerState>()?;
ctx.gpu.copy_d2d_async(ssm_state.h_state, h_state_final_tmp, h_bytes, stream)?;

// 2. Откат к checkpoint (pre-verify h_state)
ctx.gpu.copy_d2d_async(ssm_state.h_state_checkpoint.unwrap(), ssm_state.h_state, h_bytes, stream)?;

// 3. k sequential gdn_prefill_split4(total=1) шагов → заполнение intermediates
for t in 0..k {
    let q_off = t * conv_dim * 2; // BF16
    let gb_off = t * gate_stride * 4; // FP32
    let out_tmp = scratch_ptr; // временный буфер для output (не используется)
    ops::gdn_prefill_split4(ctx.gpu, self.gdn_prefill_split4_k,
        ssm_state.h_state,
        gdn_bufs.qkv.offset(q_off),
        gdn_bufs.qkv.offset(q_off + key_dim * 2),
        gdn_bufs.qkv.offset(q_off + key_dim * 4), // V
        gdn_bufs.gate_beta.offset(gb_off),
        gdn_bufs.gate_beta.offset(gb_off + nv * 4),
        out_tmp,
        1, total=1, ...
        stream,
    )?;
    ctx.gpu.copy_d2d_async(ssm_state.h_state, ssm_state.h_state_intermediates[t], h_bytes, stream)?;
}

// 4. Восстановить итоговый h_state
ctx.gpu.copy_d2d_async(h_state_final_tmp, ssm_state.h_state, h_bytes, stream)?;
```

`h_state_final_tmp` — временный буфер на уровне модели (аналогично gdn_buf_*, можно
использовать часть scratch или выделить отдельно). Для k=16 и 48 SSM-слоёв: 48 × 16 = 768
доп. kernel-вызовов в CUDA графе — приемлемо.

**Схема для conv_state** (заполнение conv_state_intermediates): Phase1 (`conv1d_update_prefill`)
обрабатывает k токенов БАТЧЕВО и оставляет conv_state в финальном состоянии. Per-token
снэпшоты не сохраняются. Аналогичный подход:
1. copy(conv_state → conv_state_final_tmp)
2. Restore conv_state ← checkpoint (conv_state_checkpoint)
3. for t in 0..k: conv1d_update_f32(single token) → copy(conv_state → conv_state_intermediates[t])
4. Restore conv_state ← conv_state_final_tmp

Но `conv1d_l2norm_f32_k` принимает FP32, а данные в `deinterleaved` (из phase1) — BF16.
**Альтернатива**: использовать `conv1d_l2norm_k` (BF16) вместо f32-версии — с учётом
того что для intermediates нам важна СТРУКТУРА снэпшота (правильная conv window), не
FP32/BF16 точность операции. Сравнить с BF16-accept rate.

Уточнение реализации conv_state intermediates — отдельный sub-task в рамках Итерации 1.

#### Шаг 3: Проверка корректности (eager)

```bash
ATLAS_DFLASH_DEBUG_NO_GRAPH=1 \
ATLAS_VERIFY_PREFILL_SSM=1 \
ATLAS_DFLASH_DRAFT_CAP=15 \
  spark serve --model /home/isolo/Projects/isolorg/models/Qwen3.6-27B-NVFP4 \
              --draft-model /home/isolo/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106
```

Проверить:
- Acceptance rate ≥ baseline (консистентность h_state)
- Токены когерентны на 1K+ токенов
- Нет NaN / мусора в первом токене после каждого verify-шага

---

### Итерация 2 — Поддержка CUDA графов

CUDA граф для verify кешируется по ключу `(seq.slot_idx, k)` в `verify_kgamma_graph`.
После добавления трёхфазных SSM-вызовов тело захваченного графа меняется — старые
записи в том же кеше устарели и воспроизведут неправильные ядра.

**Решение**: добавить новое поле кеша `verify_kgamma_ssm_graph`. Старый
`verify_kgamma_graph` сохраняем для fallback / отката. Без версионирования ключей.

В `model/types.rs` добавить:
```rust
pub(super) verify_kgamma_ssm_graph: Mutex<HashMap<(u32, usize), GraphHandle>>,
```

В `model/impl_a1.rs` инициализировать рядом с `verify_kgamma_graph`:
```rust
verify_kgamma_ssm_graph: Mutex::new(HashMap::new()),
```

В `verify_d.rs` при `use_prefill_ssm` использовать `self.verify_kgamma_ssm_graph`
вместо `self.verify_kgamma_graph` для поиска/сохранения.

#### Шаг 2б: Убрать диагностический sync из `prefill_phase1_inner` (R4)

В `crates/spark-model/src/layers/qwen3_ssm/trait_prefill_phase1.rs`, строки 53-56:

```rust
// было:
tracing::info!("ssm phase1 ENTRY: k={k} h={h} qkvz={qkvz_size}");
ctx.gpu.synchronize(stream).map_err(|e| {
    anyhow::anyhow!("ssm phase1 ENTRY: stream broken BEFORE we start (M={k}): {e}")
})?;

// стало:
if !ctx.graph_capture {
    tracing::info!("ssm phase1 ENTRY: k={k} h={h} qkvz={qkvz_size}");
    ctx.gpu.synchronize(stream).map_err(|e| {
        anyhow::anyhow!("ssm phase1 ENTRY: stream broken BEFORE we start (M={k}): {e}")
    })?;
}
```

Аналогичный guard для `tracing::info!` в `prefill_gdn_full_inner` (строки 46-52) —
не блокирует граф (tracing — CPU-only), но устраняет verbose-логи на каждый verify.

Убрать `ATLAS_DFLASH_DEBUG_NO_GRAPH=1`, протестировать захват/воспроизведение графа
с `ATLAS_VERIFY_PREFILL_SSM=1`. Проверить промах кеша на первом шаге и попадания после.

---

### Итерация 3 — Замеры производительности

Профилирование до/после через `ATLAS_DFLASH_PROF=1` (таймер на слой):

```bash
# Baseline
ATLAS_DFLASH_PROF=1 ATLAS_DFLASH_DRAFT_CAP=15 spark serve ...

# С параллельным SSM verify
ATLAS_DFLASH_PROF=1 ATLAS_VERIFY_PREFILL_SSM=1 ATLAS_DFLASH_DRAFT_CAP=15 spark serve ...
```

Ожидания:

| Метрика | Baseline | После SSM fix |
|---------|----------|---------------|
| SSM verify время K=16 | ~730ms | ~100-250ms |
| Attn verify время K=16 | ~222ms (без изменений) | ~222ms |
| verify_multiplier K=16 | ~12.44× | ~4-6× |
| пропускная способность (accept_len≈6) | ~12.5 tok/s | ~18-25 tok/s |

Если SSM-время не улучшилось — профилировать таймер per-kernel, убедиться что
split4 GDN выбирается и узкое место не сместилось.

---

### Итерация 4 — Очистка

1. **Убрать env gate**: сделать трёхфазный SSM дефолтным путём после подтверждения корректности.
   Оставить `decode_batched` только для HSS-пути (если применимо) или не-SSM слоёв.

2. **Проверка γ > 32**: при `ATLAS_DFLASH_DRAFT_CAP=32+` WY32-ядро активируется
   автоматически (~30× для SSM). Изменений кода не требует.

3. **Long-context regression**: тест на сессиях 4K+ токенов — убедиться что h_state
   не дрейфует относительно baseline за много verify-циклов.

---

## Риски

### R0 — Переиспользование `gdn_bufs` между SSM-слоями — ✅ ЗАКРЫТ (до имплементации)

**Что**: `gdn_bufs.qkv/gate_beta/output/z` — единые фиксированные device-буферы,
переиспользуемые каждым SSM-слоем. В обычном prefill (`prefill_c.rs`) это безопасно,
потому что каждый слой выполняет все три фазы полностью, прежде чем перейти к
следующему (фазы НЕ батчируются по слоям — структура per-layer: p1→p2→p3→следующий слой).

**Для verify**: та же per-layer последовательная структура. SSM-слой `i` запускает
p1→p2→p3, затем SSM-слой `i+1` перезаписывает `gdn_bufs` своей фазой p1, и т.д.
`gdn_bufs.output` слоя `i` потребляется в p3 до начала слоя `i+1`. Перекрытий нет.

**Статус**: Закрыт. Подтверждено чтением `prefill_c.rs` — структура per-layer
идентична тому, что мы планируем в verify. Цикл в Шаге 2 вызывает все три фазы
до перехода к следующему слою.

---

### R1 — Семантика `seq_len_start` в `prefill_phase1` — ✅ ЗАКРЫТ (до имплементации)

**Что**: Фаза 1 принимает `seq_len_start: usize`. Предполагалось, что conv1d использует
это значение для инициализации скользящего окна.

**Результат**: `prefill_phase1_inner` (строка 21) принимает параметр как
`_seq_len_start: usize` — underscore-префикс означает намеренный игнор. Аналогично
игнорируются `_kv_cache`, `_block_table`, `_disk_block_ids`, `_disk_last_offloaded_per_layer`,
`_kv_write_start`. conv1d (строка 219) читает `ssm_state.conv_state` напрямую, без
seq_len_start — conv-окно уже содержит правильное состояние.

**Статус**: Закрыт. Параметр передаётся для совместимости сигнатуры трейта (дефолтный
fallback в `transformer_layer.rs` вызывает `self.prefill(seq_len_start, ...)`), но
SSM-override полностью его игнорирует.

---

### R2 — Консистентность h_state при частичном accept/reject — ✅ ЗАКРЫТ (pre-existing)

**Что**: `prefill_gdn_full` обновляет `ssm_state.h_state` на месте после обработки
K токенов. При частичном отклонении h_state продвинулся на K, а seq_len — только на j.

**Статус**: Закрыт как pre-existing limitation. Тот же риск есть у `decode_batched(k)`
сегодня — h_state не откатывается при partial reject в обоих случаях. Не новая
регрессия, не в скопе этого PR.

---

### R3 — Выбор GDN-ядра для K=16 — ✅ ЗАКРЫТ (до имплементации, лучше ожидаемого)

**Что**: Предполагалось, что для K=16 выбирается split4 (global-mem H-state, медленнее).

**Результат**: Диспатч-лесенка `prefill_gdn_full_inner` для K=16 на GB10:
- `total > 32 && wy32_k != 0` → не проходит (16 ≤ 32)
- `total > 4096` → не проходит
- **`wy4_k != 0 && !atlas_scale`** → **WY4 выбирается** (`atlas_scale` — только для AMD)
- split4: не достигается

WY4 (`gdn_prefill_persistent_wy4_k`) использует shared-memory H-state
(smem ≈ `kd*vd*4 + 8*kd*4 + 56` байт, вписывается в GB10). Быстрее split4,
минус 30× vs WY32, но принципиально лучше ожиданий.

**Статус**: Закрыт. split4 используется только как fallback если `wy4_k == 0`
(ядро не загружено) — на GB10 build этого не происходит.

---

### R4 — Захват CUDA графа: безусловный `synchronize` в phase1 — ✅ ЗАКРЫТ (фикс в плане Итерации 2)

**Что**: `prefill_phase1_inner` содержит безусловный host-sync на входе
(`trait_prefill_phase1.rs`, строки 53-56):

```rust
// Diagnostic: always sync at entry to catch prior-layer errors
tracing::info!("ssm phase1 ENTRY: k={k} h={h} qkvz={qkvz_size}");
ctx.gpu.synchronize(stream)?;
```

`ctx.gpu.synchronize(stream)` вызывает `cudaStreamSynchronize` — CPU-CUDA вызов,
запрещённый внутри CUDA graph capture-окна. Без фикса `begin_capture` → `phase1_inner`
→ `synchronize` вернёт `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.

Это чисто диагностический sync (комментарий явный). Синхронизации при k ≤ 4096
(строки 73, 165, 185, 212, 234) при K=16 не срабатывают — только строка 54.

**Статус**: Закрыт. Фикс включён в Итерацию 2 (см. ниже). Итерация 1 (eager) не затронута.

---

### R5 — `ctx.gdn_exact_replay` должен быть false в verify — ✅ ЗАКРЫТ (до имплементации)

**Что**: `ForwardContext.gdn_exact_replay = true` принудительно выбирает WY4
(bit-faithful) вместо FLA chunked-ядра (для корректности Marconi warm-hit replay).
В verify мы не воспроизводим кешированный prefix — вычисляем новые позиции.
Должно быть false.

**Статус**: Закрыт. Уже false в текущей конструкции ctx в `verify_d.rs` (строка 161:
`gdn_exact_replay: false`). Изменений не требует.

---

### R7 — h_state_intermediates не заполняются → тихое повреждение h_state — 🔴 КРИТИЧНО

**Что**: `rollback_ssm_states_dispatch` (`verify_a.rs`, строка 350) всегда вызывается
при partial accept (j < k токенов принято). Он восстанавливает h_state и conv_state
из `h_state_intermediates[j-1]` и `conv_state_intermediates[j-1]`.

Текущий `decode_batched_inner` (sequential path, строки 441-452) заполняет эти
буферы после каждого токена:
```rust
copy(h_state → h_state_intermediates[t]);
copy(conv_state → conv_state_intermediates[t]);
```

Три-фазный путь этого НЕ делает. После три-фазного verify:
- `h_state_intermediates` содержат УСТАРЕВШИЕ данные от предыдущего вызова
- `rollback_ssm_states` копирует стале данные в h_state → тихое повреждение
- Нет ни паники, ни ошибки — результат: h_state в неправильном состоянии

Код защиты (`verify_a.rs`, строка 365) срабатывает только если `is_empty()` —
но поинтеры АЛЛОЦИРОВАНЫ (len=K), просто данные устаревшие.

**Решение**: После phase2 (WY4 записал итоговый h_state), заполнить intermediates
через sequential BF16 GDN-шаги:

```
// После phase2:
1. copy(h_state → h_state_final)         // сохранить итог
2. copy(h_state_checkpoint → h_state)    // откат к pre-verify состоянию
3. for t in 0..k:
     gdn_prefill_split4(total=1, h_state, gdn_bufs.qkv[t], ...)
     copy(h_state → h_state_intermediates[t])
4. copy(h_state_final → h_state)         // восстановить итоговый
```

`gdn_prefill_split4` с total=1 принимает BF16 QKV из gdn_bufs — правильный путь
для BF16 данных (в отличие от `gdn_decode`, который ожидает FP32 на Qwen3.6-27B-NVFP4).
Это k+3 GPU-операций per SSM layer, все async — захватываются в CUDA граф.

**Conv_state intermediates**: `conv1d_update_prefill` (batched) обновляет conv_state
ДО финального состояния, не сохраняя per-token снэпшоты. Нужно аналогичное решение:
после phase1, запустить k sequential `conv1d_update` для заполнения conv_state_intermediates.
Или альтернатива: сохранить conv_state до phase1, затем запустить k sequential conv для
intermediates (аналогично схеме с h_state выше).

**Статус**: Открытый критический риск. Должен быть решён в Итерации 1 — без этого
три-фазный verify ломает speculative decoding при partial accept.

---

### R8 — FP32 vs BF16 точность в SSM

**Что**: Текущий `decode_batched_inner` для Qwen3.6-27B-NVFP4 использует
`conv1d_l2norm_f32_k` (FP32 conv output) → `gdn_decode` (принимает FP32 QKV).
Причина — предупреждение в коде (строки 371-374):
> "passing BF16 data to it misinterprets every two BF16 elements as one FP32,
> causing h_state corruption and NaN after ~7 recurrent steps."

Три-фазный путь (phase1): `conv1d_update_prefill` → BF16 output → `gdn_prefill_wy4`
(принимает BF16). Это ДРУГИЕ ядра с ДРУГОЙ точностью. Не баг — стандартный режим
prefill-а. Но h_state после verify будет иметь BF16-точность вместо FP32.

Дополнительно: `decode_batched_inner` для k=16 (sequential fallback) эмитит
`tracing::warn!("sequential K=16 use_f32_conv=true")` — наш патч эту ветку
обходит и предупреждение исчезает (хорошая negative validation).

**Статус**: Поведенческое изменение, не баг. Нормальный prefill тоже использует
BF16 и модель к этому устойчива. Требует валидации acceptance rate и long-context
теста в Итерации 3-4.

---

### R9 — norm_token_count рассинхронизация

**Что**: В sequential path (строки 419-420):
```rust
let do_norm_t = (ssm_state.norm_token_count % 16 == 0) as u32;
ssm_state.norm_token_count = ssm_state.norm_token_count.wrapping_add(1);
```
Каждый токен инкрементирует счётчик → периодическая ренормализация h_state
каждые 16 токенов в `gdn_decode`.

Три-фазный WY4 путь: `norm_token_count` НЕ обновляется. После verify с k=16
токенами счётчик отстаёт на 16. Следующий decode шаг ренормализует h_state в
неправильный момент.

**Статус**: Умеренный риск. WY-пути (k=2/3/4/17) тоже не обновляют счётчик —
модель к этому устойчива. Финальная валидация: long-context тест в Итерации 4.
В R7-решении (sequential для intermediates) norm_token_count тоже должен обновляться.

---

### R6 — Улучшение пропускной способности меньше ожидаемого

**Что**: WY4 при K=16 быстрее split4 (shared-mem H-state), но всё равно значительно
медленнее WY32 (~30×). Если overhead kernel launch + smem load при K=16 всё ещё велик
относительно полезной работы, прирост может оказаться скромным.

**Вероятность**: Неизвестна. R3 уже улучшил ожидания (WY4 вместо split4).

**Митигация**: Профилировать через `ATLAS_DFLASH_PROF=1` после Итерации 1. При
неудовлетворительном результате рассмотреть увеличение γ до 32+ для активации WY32,
либо переход к attention fix (PREFILL_VERIFY_PLAN.md) параллельно.

---

## Команда сборки (всегда)

```bash
ATLAS_TARGET_MODEL=qwen3.6-27b \
ATLAS_TARGET_QUANT=nvfp4 \
ATLAS_TARGET_HW=gb10 \
cargo build --release -p spark-server
```
