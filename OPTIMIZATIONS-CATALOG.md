# Каталог оптимизаций ветки `optimizations` (atlas)

**Диапазон:** `2e3a26a..HEAD` (13 коммитов) · **Файлов изменено:** 15 (+845 / −87) · **Дата:** Jun 28–30 2026
**Модель:** Qwen3.6-27B-NVFP4 (target) + Qwen3.6-27B-DFlash (drafter, BF16) на GB10

**Главный итог:** DFlash speculative-decode acceptance подняли с ~1–5% до **94% pos0 / τ=6.67** — превосходит vLLM (τ=4.09) на K=16. tok/s 2.9→13.0 (+93%). Остаточный разрыв с vLLM (13 vs 43 batched) теперь чисто verify-kernel cost (SSM decode), не acceptance.

---

## A. Acceptance correctness (ctx conditioning, EAGLE, позиции)

Это ядро работы — пять последовательно вскрытых багов в том, как drafter получает target hidden states для conditioning.

### A1. ctx_gap fix — захват accepted-draft hidden + трекинг реальных позиций
- **Что делает:** На ACCEPT-пути K=2 добавляет в ctx-аккумулятор hidden принятого драфта (row 1), чтобы `ctx_len` не отставал от `seq_len`. Вводит `ctx_slot_positions: Vec<i32>` — реальную абсолютную позицию каждого ctx-слота.
- **Файлы:** `impl_a1.rs` (удвоение `dflash_hidden_save` под row0+row1), `verify_b.rs` (захват обоих rows), `trait_impl/mod.rs::dflash_accept_append`, `forward_block.rs` (использует записанные позиции вместо `ctx_start+i`), `dflash_head.rs` (поле `ctx_slot_positions`).
- **Зачем:** Каждый accept создавал 1-позиционный gap между `ctx_len` и `seq_len`, накапливавшийся в растущий RoPE-mismatch. Плюс в thinking-режиме токены создают разрыв позиций (промпт 0..P, output P+T..N) — индекс слота ≠ реальная позиция.
- **Эффект:** acceptance ~1–2% (thinking) / ~3% (no-think) → **~62% (thinking) / ~70% (no-think)**. На K=16: pos0 ~0%→53%, τ ~1.0→2.97, tok/s 2.9→5.4.
- **Флаг:** дефолтное поведение (всегда вкл).
- **Коммит:** `8917168`

### A2. forward_block использует ctx_slot_positions для RoPE
- **Что делает:** `forward_block` принимает срез `ctx_positions` и берёт записанную абсолютную позицию каждого ctx-слота для RoPE-ротации вместо вычисления `ctx_start + i`. Fallback на индекс слота (корректен для no-think).
- **Файлы:** `forward_block.rs` (новый параметр + map по позициям), `propose.rs` (push реальной позиции при decode-append, прокидка `&dstate.ctx_slot_positions` в оба вызова T=1/T=2).
- **Зачем:** Неправильные position IDs ломают RoPE для всех ctx K-векторов → коллапс предсказаний в "." .
- **Флаг:** дефолт.
- **Коммит:** `85119e7`

### A3. Монотонные позиционные метки — N = position−1−num_accepted (Bug 2)
- **Что делает:** Исправляет метку row-0 ctx-слота. `after_verify` сохраняет `last_num_accepted`; decode-append в `propose` маркирует row-0 позицией `position - 1 - last_num_accepted = N` вместо ошибочного `position - 1`.
- **Файлы:** `dflash_head.rs` (поле `last_num_accepted`, запись в `after_verify`), `propose.rs` (`row0_pos`).
- **Зачем:** Оба append писали одну метку N+1 → `ctx_slot_positions` шла `N+1, N+1, N+3, N+3...` (дублируя нечётные, пропуская чётные). Row-0 hidden `h(last_token@N)` получал неверную RoPE для свежайшего conditioning-слота. Это был баг в самом фиксе A1.
- **Эффект:** pos0 raw_match 53%→**66%**, mean accepted/step 1.97→2.44 (+24%), tok/s +17–20% на K=2 и K=16 (K=16: τ 2.97→3.44, tok/s 5.4→6.4).
- **Флаг:** дефолт.
- **Коммит:** `b767d4b`

### A4. EAGLE conditioning K=2 — drafter видит bonus generator hidden (Bug 1)
- **Что делает:** Перед `propose` добавляет row 0 @ N, затем row 1 @ N+1 — так что свежайший ctx-слот для `forward_block` = row 1 = hidden, чей argmax породил bonus (как в vLLM/EAGLE). `propose` пропускает собственный decode-append через one-shot флаг `skip_next_decode_append`.
- **Файлы:** `trait_impl/mod.rs::dflash_eagle_accept_append`, `dflash_head.rs` (поле `skip_next_decode_append`), `propose.rs` (потребление флага `eagle_skip`), `verify_k2_step.rs` (вызов под флагом + skip legacy accept-append), `speculative.rs`/`traits/model.rs` (trait defaults).
- **Зачем:** При draft-gen свежайший слот был row 0 = `h(last_token@N)` — hidden, прогнанный target С bonus на входе, т.е. на шаг ВПЕРЁД от EAGLE-пары. Atlas добавлял правильный row 1, но ПОСЛЕ propose → помогало только следующему шагу.
- **Эффект:** pos0 raw_match 66.7%→**79.7%** (+13pp), tok/s 12.3→13.5. Вышли на vLLM pos0 regime. Output byte-valid.
- **Флаг:** `ATLAS_DFLASH_EAGLE_FIX=1` (off = байт-идентичный legacy).
- **Коммит:** `8aa56b8`

### A5. EAGLE K=γ + ctx-undercount — per-position capture (главная победа)
- **Что делает:** На K=γ пути `try_dflash_capture_all` захватывает ВСЕ K verify-rows в row-major K-row буфер; после того как `num_accepted` известен, `dflash_eagle_kgamma_append` добавляет rows `0..=num_accepted` на позиции `N..N+num_accepted`, причём bonus generator (row `num_accepted`) — самый свежий слот. Reject (num_accepted=0) добавляет только row 0 @ N (уже EAGLE-correct).
- **Файлы:** `impl_b3.rs::try_dflash_capture_all`, `trait_impl/mod.rs::dflash_eagle_kgamma_append`, `verify_d.rs` (выбор capture_all под флагом), `verify_dflash_step.rs` (вызов append + `skip_next_decode_append`), `impl_a1.rs` (буфер KMAX=`dflash_kgamma`.max(2)≈17 rows).
- **Зачем — два корня:**
  1. **ctx undercount:** каждый K=γ accept коммитит `num_accepted+1` позиций, но добавлялся 1 ctx-слот (row 0) → ctx отставал от seq_len, держал ~35% позиций в дырявом буфере.
  2. **EAGLE shift (scaled):** свежайший слот = `h(last_token@N)`, устаревший на `num_accepted` относительно bonus generator `h@(N+num_accepted)`.
- **Эффект (K=16, оба prefill-флага, no-think, T=0):** pos0 raw_match 66%→**94%** (выше vLLM); хвост поднят (pos8 prefix 5%→25%, pos14 0%→2%, pos4 raw 47%→87%); mean accepted/step 2.44→**5.67** (+132%); **τ 3.44→6.67** (+94%, превосходит vLLM τ=4.09 на 63%); tok/s 6.4→**13.0** (+93%) при том же verify (~520ms).
- **Флаг:** `ATLAS_DFLASH_EAGLE_FIX=1` (off = байт-идентичный legacy).
- **Коммит:** `cc02368`

### A6. (известный остаток) combine_hidden_states — НЕ реализовано
- **Что:** TODO-комментарий в `forward_block.rs:310-317`. vLLM добавляет `fc(h(bonus_N))` к `embed(bonus_N)` на noise0; Atlas захватывает `h(bonus_{N-1})`, т.е. target_hidden_stack на 1 шаг позади. Правильный фикс требует захвата hidden ВЫХОДНОГО токена (bonus_N) в verify, что текущая архитектура для K=γ не поддерживает.
- **Файлы:** `forward_block.rs` (только комментарий).
- **Эффект:** потенциальный +acceptance, не сделано.
- **Коммит:** `85119e7` (комментарий).

---

## B. Verify speed (prefill attn, prefill ssm, two-phase capture)

### B1. ATLAS_VERIFY_PREFILL_ATTN — prefill-mode attention в K=γ verify
- **Что делает:** Заменяет `decode_multi_seq(K)` на единый `prefill_attention_paged` проход для всех K verify-токенов (q_offset=seq_len, causal mask). Одно чтение KV-cache вместо K. Метадата получает два layout'а: decode (num_seqs=K, K копий block_table) и prefill (num_seqs=1, одна строка block_table); positions/slots одинаковы.
- **Файлы:** `verify_d.rs` (ветвление layout метадаты + вызов `layer.prefill(...)`).
- **Зачем:** На высоком K attention verify читал KV-cache K раз. HSS-путь остаётся fallback (decode_batched) т.к. paged-decode не видит long-context history на диске.
- **Эффект:** срезает attention verify cost. Цель плана: verify K=16 ~957ms → ~110–130ms (attn часть ~222ms). SSM `decode_batched` всё ещё доминирует на высоком K (комбинировать с B2).
- **Флаг:** `ATLAS_VERIFY_PREFILL_ATTN=1` (off = decode path fallback).
- **Коммит:** `9c5f6cd`

### B2. ATLAS_VERIFY_PREFILL_SSM — параллельный SSM verify (three-phase WY4)
- **Что делает:** Для SSM-слоёв при K≥5 заменяет последовательный `decode_batched` на трёхфазную WY4 batch GDN recurrence: Phase 1 (K serial single-token заполняют `gdn_bufs`, сохраняя `conv_state_intermediates`), Phase 2 (одна WY4 batch GDN над всеми K + replay из checkpoint для `h_state_intermediates`, через `ssm_verify_h_tmp`), Phase 3 (gated RMS norm + output proj + FFN).
- **Файлы:** `verify_d.rs` (большой блок three-phase), `impl_a1.rs`+`types.rs` (буфер `ssm_verify_h_tmp` = `ssm_pool.h_bytes`).
- **Зачем:** GDN-работа K serial h_state updates → одна batch recurrence. SSM decode — главный bottleneck verify на высоком K (8 SSM-слоёв × K sequential GDN).
- **Эффект:** prefill_ssm частично снизил verify 755→513ms (из DEBUG-doc). Цель плана: SSM K=16 ~730ms → ~100–250ms; throughput ~12.5 → ~18–25 tok/s.
- **Флаг:** `ATLAS_VERIFY_PREFILL_SSM=1`.
- **Коммит:** `9f2d5f9`

### B3. Захват ctx hiddens во время two-phase prefill
- **Что делает:** Хук `try_dflash_prefill_capture_layer` в two-phase prefill путь (`prefill_c`) для SSM и attention слоёв; после всех слоёв `update_dflash_ctx_len_after_prefill` обновляет `ctx_len`, чтобы `propose()` знал сколько prefill-позиций доступно.
- **Файлы:** `prefill_c.rs` (два хука + update), `impl_b3.rs` (`try_dflash_prefill_capture_layer` пишет `ctx_slot_positions.push(abs_pos)` только на slot_idx==0; логирование ctx_len update).
- **Зачем:** Без этого ctx-аккумулятор не содержал prefill-позиций — drafter стартовал без conditioning-контекста промпта.
- **Флаг:** дефолт (часть DFlash pipeline).
- **Коммит:** `30f2319`

### B4. try_dflash_capture_draft — захват row 1 (draft hidden)
- **Что делает:** Захватывает hidden draft-токена (row 1) в каждом capture-слое во вторую половину `dflash_hidden_save`, внутри CUDA-графа рядом с `try_dflash_capture(layer_idx, 0)`.
- **Файлы:** `impl_b3.rs::try_dflash_capture_draft`, `verify_b.rs` (вызов).
- **Зачем:** Нужен для ACCEPT-case ctx append (A1/A4) — оба hidden (last_token row0 + draft row1) должны быть доступны post-graph.
- **Флаг:** дефолт.
- **Коммит:** `8917168`

---

## C. K=2 verify pipeline

### C1. Split accept-check от emit + stochastic accept
- **Что делает:** Разделяет проверку acceptance (raw argmax — на той же базе, что и drafter) от эмиссии токена (всегда pipeline-processed, чтобы output совпадал с baseline decode). Для T>0 добавляет `dflash_stochastic_accept`: при point-mass drafter (p_draft=1 на argmax) вероятность acceptance = `p_target(draft_token, T)` (softmax по pos0 logits, детерминированный xorshift64 sample от seed+seq_len+draft_tok).
- **Файлы:** `verify_k2_step.rs` (`v0_emit`/`v1_emit` vs `v0_check`, функция `dflash_stochastic_accept`).
- **Зачем:** Раньше и check, и emit использовали raw argmax → drift последовательности на reject, коллапс accept rate ~2.5%→~1%. Pipeline (rep_pen/DRY) модифицировал target-токен до сравнения с greedy-argmax драфтера.
- **Эффект:** фикс drift; восстановление accept rate. Fallback на argmax-equality при D2H-ошибке.
- **Флаг:** `dflash_verify_raw_argmax` (внутренний, не env) + `a.temperature > 0.0`.
- **Коммит:** `7f4682c`

### C2. Default draft cap = γ−1 (был 1)
- **Что делает:** Поднимает дефолтный `ATLAS_DFLASH_DRAFT_CAP` с 1 до `gamma - 1`, разблокируя K=γ sequential verify путь по умолчанию.
- **Файлы:** `propose.rs` (`.unwrap_or(self.gamma - 1)`).
- **Зачем:** BF16 embed stride bug в `verify_d.rs` был исправлен (`2e3a26a`) — K=γ путь теперь даёт корректный output, незачем форсить K=2.
- **Флаг:** `ATLAS_DFLASH_DRAFT_CAP=N` (N=1 чтобы форсить K=2 путь).
- **Коммит:** `85119e7`

---

## D. Прочее / cleanup / диагностика

### D1. ATLAS_DFLASH_POS_DIAG — per-position acceptance диагностика
- **Что делает:** Env-gated raw per-position match rate (`drafts[i]==verified[i]` независимо от prefix) vs prefix-accept, агрегируется каждые 100 шагов. Показывает ГДЕ multi-token draft-цепь расходится с target.
- **Файлы:** `verify_dflash_step.rs` (статические AtomicU64 массивы PROPOSED/MATCHED/PREFIX_ACC).
- **Флаг:** `ATLAS_DFLASH_POS_DIAG=1`.
- **Коммит:** `ce0f4aa`

### D2. Удаление debug sync/tracing из SSM prefill phase1
- **Что делает:** Убирает диагностический `tracing::info!` + безусловный `ctx.gpu.synchronize(stream)` на входе SSM phase1.
- **Файлы:** `trait_prefill_phase1.rs` (−6 строк).
- **Зачем:** Synchronize на каждом слое убивал производительность; был временной диагностикой.
- **Флаг:** дефолт.
- **Коммит:** `f557621`

### D3. append_ctx_slot — generic trait method
- **Что делает:** Новый метод `DraftProposer::append_ctx_slot` (default no-op для MTP) + реализация в `BlockDiffusionDraftHead`: копирует один hidden-слот в ctx-аккумулятор с реальной позицией, инкрементит `ctx_len`, push в `ctx_slot_positions`. С guard `ctx_len >= max_ctx_len`.
- **Файлы:** `speculative.rs` (trait default), `dflash_head.rs` (impl).
- **Зачем:** Общая точка для всех accept-append путей (A1/A4/A5).
- **Коммит:** `8917168` / `8aa56b8`

### D4. Три trait-метода на Model + ctx_len update логирование
- **Что делает:** `dflash_accept_append`, `dflash_eagle_accept_append`, `dflash_eagle_kgamma_append` как default-no-op методы трейта `Model` (реальные impl в `trait_impl/mod.rs`). Плюс info/warn логирование при обновлении/пропуске ctx_len.
- **Файлы:** `traits/model.rs`, `trait_impl/mod.rs`, `impl_b3.rs`.
- **Коммит:** `8aa56b8`, `cc02368`

### D5. Debug tracing в propose (pos≤1)
- **Что делает:** При `position <= 1` логирует `target_layer_ids` и `gamma` (диагностика инициализации DFlash).
- **Файлы:** `propose.rs`.
- **Флаг:** дефолт (срабатывает только на первых 2 шагах).
- **Коммит:** `85119e7`

### D6. Документация и бенчмарки
- `FINDINGS-27B-ATLAS.md` (root cause ctx_gap/position ID), `BENCH-27B-DFLASH.md` (таблица K=2..16, T=0/0.6/1.0), `DEBUG-ACCEPTANCE.md` (лог расследования 27% vLLM vs 5% Atlas → fixed), `PREFILL_VERIFY_PLAN.md`, `SSM_VERIFY_PLAN.md`, `bench/` скрипты.
- **Коммиты:** `86bc664`, `bb09afe`, `2777ef8`

---

## Сводка всех env-флагов (ATLAS_*)

| Флаг | Введён | Что делает | Дефолт |
|---|---|---|---|
| **ATLAS_DFLASH_EAGLE_FIX** | `8aa56b8`/`cc02368` | EAGLE conditioning: drafter видит bonus-generator hidden как свежайший ctx-слот. K=2: row0@N→row1@N+1 перед propose. K=γ: per-position capture rows 0..=num_accepted, bonus generator свежайший. Главный acceptance-фикс. | **off** (off = байт-идентичный legacy) |
| **ATLAS_VERIFY_PREFILL_ATTN** | `9c5f6cd` | Единый `prefill_attention_paged` для всех K verify-токенов вместо `decode_multi_seq(K)`. Одно чтение KV-cache. | off (decode fallback) |
| **ATLAS_VERIFY_PREFILL_SSM** | `9f2d5f9` | Three-phase WY4 batch GDN recurrence для SSM verify при K≥5 вместо sequential `decode_batched`. | off |
| **ATLAS_DFLASH_POS_DIAG** | `ce0f4aa` | Per-position raw match-rate диагностика, агрегация каждые 100 шагов. | off |
| **ATLAS_DFLASH_DRAFT_CAP** | (ранее) | Кол-во драфтов. **Дефолт изменён** `1 → γ−1` (`85119e7`). N=1 форсит K=2 путь. | γ−1 |
| **ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN** | (ранее, упомянут) | Перезапись noise-rows детерминированным паттерном для сверки с PyTorch reference. | off |

**Внутренний (не env):** `dflash_verify_raw_argmax` — флаг в `verify_k2_step.rs`, переключает acceptance-check на raw GPU argmax (DFlash) vs processed (MTP).

---

## Траектория acceptance (итоговая)

| Этап | Коммит | K=2 pos0 | K=2 tok/s | K=16 pos0 | K=16 τ | K=16 tok/s |
|---|---|---|---|---|---|---|
| Старт (баги) | — | ~0–1% | 8.6 | 1.1% | ~1.0 | 2.9 |
| ctx_gap + positions | `8917168`/`85119e7` | 53% | 10.5 | 53% | 2.97 | 5.4 |
| monotonic labels (Bug 2) | `b767d4b` | 66.7% | 12.3 | 66% | 3.44 | 6.4 |
| EAGLE K=2 | `8aa56b8` | **79.7%** | **13.5** | — | — | — |
| EAGLE K=γ per-pos | `cc02368` | — | — | **94%** | **6.67** | **13.0** |
| vLLM референс | — | ~80% | — | ~80% | 4.09 | 43 (batched) |

**Остаток (НЕ acceptance):** разрыв tok/s 13 vs vLLM 43 — чисто verify-kernel cost. K=16 verify_ms≈520ms доминирует SSM `decode_batched` (8 SSM-слоёв × K sequential GDN). prefill-attn включён, SSM GDN replay — bottleneck. Дальнейшие цели: ускорить SSM verify, K≥4 accept-append, combine_hidden_states (A6), снизить K=2 propose overhead.

---

Каталог покрывает все 15 изменённых файлов и все 13 коммитов, включая мелочи (cap=γ−1, удаление debug sync, KMAX-буфер, логирование ctx_len, debug tracing pos≤1). Готов.
---

## Известные ограничения / открытые баги

### ⚠️ Graphed K=γ verify выдаёт corrupt output (pre-existing, не EAGLE-регрессия)
- **Симптом:** При CUDA-графах (use_graphs=true, дефолт) K=γ verify path даёт degenerate/divergent output (детерминированно, не FP-шум). При T=0 spec-decode lossless — graphed и eager обязаны совпадать побайтно, но не совпадают.
- **Изоляция:** EAGLE OFF на графах ТОЖЕ расходится → баг в pre-existing графовом K=γ пути, независим от per-position capture. Подтверждается существованием флага `ATLAS_DFLASH_DEBUG_NO_GRAPH` (его комментарий: "localize K=γ illegal-address crashes downstream of SSM"). Вероятная причина — SSM checkpoint / prefill-state capture под графом.
- **Текущий обход:** DFlash K=γ требует `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` для корректного output. Все A/B-замеры этой ветки были на eager-пути (NO_GRAPH=1) — EAGLE-фикс на нём валиден и корректен.
- **Производительность:** графовый K=γ даже медленнее eager (11.5 vs 13.0 tok/s) — SSM decode доминирует независимо от графа.
- **Статус:** отдельное расследование (вне scope EAGLE). До фикса — форсить eager для K=γ verify либо документировать NO_GRAPH-требование.
- **batch>1:** не тестирован. `dflash_hidden_save` — единый shared буфер (не per-slot); при max-batch-size>1 capture одной seq может перезаписать rows другой до её append. Требует валидации перед multi-seq.
