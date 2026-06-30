# DFlash Acceptance Debug — Qwen3.6-27B

**Цель:** понять почему vLLM DFlash даёт 27% acceptance, а Atlas — 1–10%.

**Железо:** NVIDIA GB10 (Grace Blackwell), 121 GB unified RAM  
**Модель:** Qwen3.6-27B-NVFP4 (target) + z-lab/Qwen3.6-27B-DFlash (drafter, BF16)  
**Промпт:** MinHeap (Write a Python implementation of a min-heap...)

---

## Референс

| System | T | tok/s | accept_% | τ | Notes |
|---|---|---|---|---|---|
| vLLM aeon | 0 | 43 | 27.2% | 4.09 | 3 req × 139 tok, NVFP4, no thinking |
| vLLM aeon | 0 | 43 | 28.5% | 4.28 | 5 req × 1500 tok, with thinking |
| Atlas K=2 w/ thinking | 0 | 8.6 | 5.8% | — | window=512, WY2 |
| Atlas K=2 w/o thinking | 0 | 11.9 | 6.9% | — | window=512, WY2 |

---

## Архитектура ctx_hidden_acc

- 5 слоёв target модели [L0, L15, L30, L45, L60], конкатенированы (25600 dims)
- FC-проекция → 5120 + RMSNorm
- Используется как K/V для drafter attention

**ctx_window** — размер sliding window буфера. Влияет на acceptance:
- window=1 → 4.4%
- window=64 → 7%
- window=512 → 18% (early-seq) / 5-10% (full-seq average)

---

## Хронология фиксов

### Fix 1: poisoned_ctx (коммит 2e3a26a, В HEAD)
- **Баг:** `try_dflash_capture(layer_idx, k-1)` захватывал hidden state draft-токена вместо last_token
- **Фикс:** изменено на `try_dflash_capture(layer_idx, 0)` — захватывает row 0 = last target token
- **Статус:** скомпилирован в бинарник (Jun 29 22:49)

### Fix 2: ctx_gap (uncommitted, В бинарнике)
- **Баг:** на ACCEPT seq_len += 2 (last_token + draft_token), но ctx_len += 1 → ctx отстаёт на N после N принятых токенов
- **Фикс:** добавлен `dflash_accept_append` / `append_ctx_slot` / `try_dflash_capture_draft`
- **Статус:** uncommitted, но скомпилирован (бинарник актуален)

---

## Гипотезы (ранжированы по приоритету)

### ❌ H1: ctx_hidden_acc содержит DRAFT hidden states вместо TARGET hidden states — ОПРОВЕРГНУТО

Stage 2 показал: `self.buffers.hidden_states()` в `try_dflash_capture_draft` — это буфер TARGET модели.
`offset(1 * h)` = строка 1 батча [last_token, draft_token] → hidden state draft-позиции, посчитанный TARGET моделью.
На ACCEPT: `dflash_accept_append` копирует этот target hidden state в ctx_hidden_acc через `append_ctx_slot`.
ctx_hidden_acc содержит **TARGET hidden states** для всех токенов (включая принятые draft). Механизм правильный.

---

### 🔴 H13: Acceptance деградирует с ростом seq_len — vLLM тестировал на 139 tok (early-seq)

**Описание:** vLLM benchmark: 3 req × 139 tok. Atlas benchmark: MinHeap → ~700 out_tok.
Мы знаем: early-seq (первые ~20 шагов) acceptance ≈ 25% при window=512.
vLLM никогда не выходит из early-seq зоны → 27%.
Atlas average = early(25%) + late-seq(1-5%) → итого 5-10%.

**Prediction:** если запустить Atlas DFlash на 150-токенном выводе — получим ≈ 25%.

**Как проверить:** запустить бенч с OSL=150 (short output). Если acceptance ≈ 25% → H13 подтверждена.

**Почему деградирует?** Открытый вопрос. Варианты:
- ctx_window заполняется и старые записи вытесняются → drafter теряет важный ранний контекст
- Compound ошибка в позиционном кодировании на длинных последовательностях
- Качество drafter падает на длинных prefix (training distribution mismatch)

### 🟡 H2: ctx_gap fix корректирует len, но кладёт wrong hidden state

**Описание:** Fix 2 решает проблему ctx_len ≠ seq_len, но если append_ctx_slot кладёт
drafter hidden state (а не target), то len совпадает, но контент ctx всё равно плохой.

---

### 🟡 H3: ctx только на prefill/первый токен, не обновляется инкрементально

**Описание:** возможно ctx_hidden_acc обновляется только при try_dflash_capture (во время verify),
но не добавляет позиции за newly-accepted draft tokens → drafter не видит последние принятые токены.

---

### 🟢 H4: vLLM использует другой механизм ctx (не ctx_hidden_acc)

**Описание:** vLLM DFlash может использовать полный KV-cache или другой способ передавать
контекст дрaфтеру. Если так — наш ctx_hidden_acc принципиально хуже.

**Как проверить:** посмотреть vLLM DFlash implementation (z-lab fork).

---

### ✅ H5: ctx_window=64 слишком мало — ПОДТВЕРЖДЕНО, НО НЕДОСТАТОЧНО

**Описание:** window=64 → 7%, window=512 → 18% (early-seq). Но vLLM 27% — значит это
не единственная причина.

---

### ❌ H6: бинарник не содержит фиксы — ОПРОВЕРГНУТО

Дайвер проверил: бинарник Jun 29 22:49, все исходники старше или ровесники. Оба фикса включены.

---

## Результаты расследования

### Stage 1 (Jun 30): проверка бинарника
- Бинарник: Jun 29 22:49 — актуален
- `find .rs -newer spark` — пусто (все исходники скомпилированы)
- Fix 1 (poisoned_ctx): ✅ в HEAD, скомпилирован
- Fix 2 (ctx_gap): ✅ uncommitted, но в бинарнике
- **Вывод:** бинарник не виноват → смотрим что именно кладётся в ctx

### Stage 2 (Jun 30): что в ctx_hidden_acc?
- `try_dflash_capture_draft` вызывается внутри layer loop TARGET verify forward pass
- `self.buffers.hidden_states()` = GPU буфер TARGET модели, row 1 = позиция draft токена
- На ACCEPT: `dflash_accept_append` (verify_k2_step.rs:208) → `append_ctx_slot` копирует в ctx_hidden_acc
- **Вывод: ctx содержит TARGET hidden states. H1 опровергнута.**

---

## ROOT CAUSE FOUND (Stage 3, Jun 30)

### 🔴 Bug #1: ctx Position ID Bug — Think-Token Gap

**Файл:** `crates/spark-model/src/layers/dflash_head/forward_block.rs:214`

**Баг:** формула `position_id[i] = ctx_start + i` предполагает, что ctx слоты непрерывны по абсолютной позиции. Это не так при thinking mode.

**Механизм:**
- Thinking фаза (694-776 токенов): propose заблокирован (`inside_thinking` gate), ctx НЕ обновляется
- После thinking: ctx слоты 0..57 = prompt (позиции 0..57, правильно), слоты 58..N = output (позиции 752+, НЕПРАВИЛЬНО)
- Ошибка: +694 позиций для каждого output слота
- На pos=1193: 87% ctx K-векторов имеют неправильный RoPE поворот → acceptance → 0%

**Эмпирическое подтверждение:**
- no-think early-seq (pos≈42): **33% acceptance** на отдельных шагах
- with-thinking early-seq: **1.4% acceptance**
- no-think aggregate на 400 tok: **2.6%** (лучше, но см. Bug #2)

**Предсказание когда ctx полностью корруптирован:**
- ctx_window=512, prompt_len=58
- После ctx_len=570 (pos≈1264): ВСЕ 512 слотов — output, всё неправильно → ~0%
- Подтверждено логами: late-seq (pos≥1300) acceptance = 0.7%

**Фикс:** добавить `ctx_slot_positions: Vec<i32>` в `DflashProposerState`, писать реальную abs позицию при каждом append. Передавать в `forward_block` вместо вычисления `ctx_start + i`.

---

### 🔴 Bug #2: Missing `combine_hidden_states`

**Файл:** `propose.rs:131-133` (TODO comment, не реализовано), `forward_block.rs` (step 2)

**Баг:** Atlas не реализует `combine_hidden_states` из архитектуры drafter'а.

**vLLM (правильно, drafter обучался так):**
```python
fc_out = self.fc(concat_5_layer_target_hiddens)   # [h]
noise0_input = embed(bonus_token) + fc_out          # combine_hidden_states
```

**Atlas (неправильно):**
- `fc_proj[eff_ctx-1]` вычисляется, но используется только как K/V слот в ctx attention
- НЕ добавляется в residual stream noise0
- Drafter обучался с прямым добавлением fc_out в noise0 — без этого conditioning signal ослаблен

**Эффект:**
- Без Bug #2: no-think acceptance было бы ≈ 27% (как vLLM)
- С Bug #2: no-think acceptance = 2.6-7% — conditioning идёт только через attention, а не прямым add

**Фикс (наивный — не работает):** `stream_buf[noise0_pos] += fc_proj[eff_ctx-1]` → ДЕГРАДАЦИЯ.

**Почему наивный фикс не работает — 1-step lag:**
- vLLM: `combine_hidden_states` добавляет `fc(h(bonus_N))` к `embed(bonus_N)` — hidden и токен одного шага
- Atlas: `try_dflash_capture` захватывает `h(bonus_{N-1})`, т.е. hidden прошлого шага
- Добавление `fc(h(bonus_{N-1}))` к `embed(bonus_N)` — мисматч с обучением → acceptance падает

**КОРРЕКЦИЯ (Diver-2, чтение vLLM кода):** Bug #2 — ЛОЖНЫЙ. vLLM НЕ добавляет FC output в noise0 residual stream.

Доказано цитатами vLLM (qwen3_dflash.py, dflash.py, utils.py):
- noise0 input = `embed(bonus_token)` (чистый token embedding), noise1+ = `embed(mask_token_id)`
- `DFlashQwen3Model.forward`: `hidden_states = input_embeds`, residual=None на старте — никакого `+ fc_output`
- DFlash явно: `self.parallel_drafting_hidden_state_tensor = None` ("DFlash embeds mask tokens directly")
- FC output идёт ТОЛЬКО в `precompute_and_store_context_kv` → context K/V
- Target conditioning доходит до noise токенов ЕДИНСТВЕННЫМ путём: cross-attention noise-Q → context-KV

**Atlas архитектура (FC→ctx K/V→attention) СОВПАДАЕТ с vLLM. Прямого add быть не должно.**
Наивный fix (noise0 += fc_proj) был бы лишним слагаемым → деградация (подтверждено тестом). ОТКАЗ от Bug #2.

**Новое направление:** почему K=2 даёт 60%, а K=16 только 12.5%? Не combine_hidden_states.
Подозрение: инициализация noise1+ позиций. vLLM кладёт `embed(mask_token_id)`. Что кладёт Atlas?
Если Atlas инициализирует noise1+ неправильным токеном — дальние позиции draft разваливаются.

---

## Итоговая картина разрыва 27% vs 5-10%

| Причина | Эффект на acceptance | Конфиг |
|---|---|---|
| Bug #2: нет combine_hidden_states | 27% → ~7% (early-seq) | все конфиги |
| Bug #1: position ID gap от thinking | 7% → 1-2% (thinking mode) | конфиги с thinking |
| Длинные последовательности | 7% → 1-3% (late-seq, ctx corruption) | 700+ tok |

**vLLM 27% = никаких думающих токенов + combine_hidden_states реализован + короткие выводы (139 tok).**

---

## Bug #1 Fix — Результаты (Jun 30)

**Реализовано:** `ctx_slot_positions: Vec<i32>` в DflashProposerState. При каждом append (prefill/decode/accept) пишется реальная abs позиция. В forward_block.rs вместо `ctx_start + i` используется `ctx_slot_positions[start_slot + i]`.

**7 файлов изменено:** dflash_head.rs, speculative.rs, propose.rs, impl_b3.rs, mod.rs, forward_block.rs

**Тест (CAP=1, K=2, window=512, rolling 100 steps, seq_len ~168-402):**

| Режим | До фикса | После фикса | Улучшение |
|---|---|---|---|
| С thinking | ~1.4% | **53%** | **37×** |
| Без thinking | ~2.6% | **70%** | **27×** |

No-think 70% превышает vLLM 27% — drafter работает корректно с правильными position IDs.

**Caveat:** rolling 100 steps на коротких seq (168-402 tok). Нужен полный бенч для tok/s.

## Полный бенч после Bug #1 fix (CAP=1, K=2, window=512)

| Конфиг | tok/s | accept_% | total_tok | think_tok | out_tok |
|---|---|---|---|---|---|
| K=2, с thinking | **11.5** | **~62%** | 1395 | 694 | 701 |
| K=2, без thinking | **10.5** | **~60%** | 498 | 0 | 498 |
| Baseline (до фиксов) | 13.6 | — | 1351 | 0 | 1351 |

### Почему tok/s хуже baseline несмотря на 62% acceptance

K=2 физически ограничен:
- propose(95ms) + verify(77ms) = **172ms на шаг**
- Ожидаемых токенов при 62%: 0.38×1 + 0.62×2 = **1.62 tok/step**
- Output tok/s = 1.62 / 0.172 = **9.4 tok/s**
- Baseline = 1 / 0.073 = **13.7 tok/s**

Даже при 100% acceptance K=2 даёт максимум 2/0.172 = **11.6 tok/s < 13.7**. K=2 не может побить baseline при текущих overhead.

### Что нужно для выигрыша в tok/s

Нужно больше черновиков при том же overhead. Теоретически с Bug #1 fix:

| K | γ | verify_ms (old) | expected_tok при 62% per-tok | step_ms | tok/s |
|---|---|---|---|---|---|
| 2 | 1 | 77 | 1.62 | 172 | **9.4** |
| 4 | 3 | 257 | 2.24 | 355 | **6.3** |
| 16 | 15 | 585 | ~2.6 | 698 | **3.7** |

Проблема: verify_ms растёт быстрее чем expected_tok → большие K при текущих verify временах только хуже.

**Корень проблемы:** verify_ms слишком высокий. vLLM 43 tok/s при τ=4.09 работает потому что:
1. Batched (3 concurrent req) — GPU утилизация выше
2. Possible faster target kernel

## K=4 с Bug #1 fix (CAP=3, no-think)

| metric | значение |
|---|---|
| tok/s | **3.22** (было 4.9 до фикса) |
| accept first-slot | **3.5%** |
| accept 2nd/3rd slot | 0% |
| verify_ms | 255ms |
| propose_ms | 113ms |

**K=4 стал хуже после Bug #1 fix.** Причина: `dflash_accept_append` (и запись в `ctx_slot_positions`) вызывается **ТОЛЬКО в verify_k2_step.rs**. В K=4 пути accept-append отсутствует → ctx_slot_positions не обновляется при принятии → позиции в ctx некорректны для K=4 accepted токенов.

Кроме того: K=4 has pre-existing issues (SSM layers, CUDA graph в decode_batched). Per-slot acceptance 3.5% на K=4 vs 60% на K=2 — разрыв слишком большой для одной только позиционной проблемы.

## Итоговая картина после Bug #1 fix

| Конфиг | tok/s | accept_% | vs baseline |
|---|---|---|---|
| Baseline | 13.6 | — | — |
| K=2 no-think (с фиксом) | **10.5** | **60%** | −23% |
| K=2 с thinking (с фиксом) | **11.5** | **62%** | −15% |
| K=4 no-think (с фиксом) | 3.22 | 3.5% | −76% |

K=2 acceptance исправлен (1% → 60%), но tok/s всё равно хуже baseline из-за структурного overhead:
- Max possible K=2 tok/s = 2 tok / 172ms = **11.6** (при 100% acceptance) < baseline 13.6

## Per-position acceptance breakdown (K=16, 100 verify steps) — РЕШАЮЩИЙ ЗАМЕР

| position | raw_match% | prefix_acc% |
|---|---|---|
| pos0 | 53% | 53% |
| pos1 | 55% | 42% |
| pos2 | 49% | 28% |
| pos3 | 47% | 20% |
| pos4 | 53% | 18% |
| pos5 | 45% | 13% |
| pos6 | 38% | 8% |
| pos7 | 34% | 6% |
| pos8 | 28% | 5% |
| pos9 | 29% | 4% |
| pos10-14 | 19-31% | 0% |

**raw_match%** = drafts[i]==verified[i] независимо. Деградирует плавно 53%→27%. Drafter НЕ плохой на хвосте — vLLM-уровень (27%) даже на pos14.

**prefix_acc%** = мультипликативная цепочка (нужны ВСЕ pos0..pos_k). Рушится к 0 на pos10.

**Текущий τ Atlas K=16 = Σ prefix_acc ≈ 1.97** (≈ measured 1.74).

### Главный вывод

Узкое место — НЕ качество drafter на дальних позициях. Это:
1. **pos0/pos1 только ~53-55%** — гейтят всю цепочку (каждый reject на pos0 убивает весь draft)
2. **Prefix-serial verification мультипликативна** — несмежные совпадения не используются

### Критическое противоречие с vLLM

vLLM: τ=4.09 при "27% per-position". Но если verification prefix-serial (мультипликативная):
- τ при 27% = 0.27/(1-0.27) ≈ **0.37**, НЕ 4.09
- Для τ=4 при prefix-serial нужен pos0 ≈ 80%

→ Значит ЛИБО (a) vLLM pos0 acceptance ~80% (намного выше нашего 53%), ЛИБО (b) vLLM использует tree/token verification (не linear prefix). Нужно разрешить через Diver-2.

Два рычага:
1. **Поднять pos0/pos1** — но почему наш 53%, а не выше? Conditioning на pos0?
2. **Tree verification** — принимать несмежные совпадения (vLLM-style)

## Следующие шаги

1. ✅ ~~Fix Bug #1 (position ID)~~ — **DONE: K=2 acceptance 1% → 62%**
2. ❌ ~~Bug #2 combine_hidden_states~~ — ЛОЖНЫЙ, vLLM не делает прямой add
3. **Разрешить vLLM определения (Diver-2):** что такое "27% acceptance" и "τ=4.09"? prefix-serial или tree? pos0 rate?
4. **Prefill verify** для K≥4 wiring (verify_c2.rs не подключён)

## vLLM эталон propose (Diver-2)

- **Один forward** на все γ позиций (parallel_drafting=True, никаких diffusion итераций)
- **Position ids noise:** монотонные `last_pos + 1 + query_off` (noise0=last_pos+1, noise14=last_pos+15)
- **Attention mask:** non-causal **full bidirectional** для full_attention слоёв (assert causal==False); causal только для sliding_attention слоёв
- **mask_token_id:** из dflash_config, эмбеддится обычной таблицей drafter
- **embed_normalizer:** None для Qwen (только gemma4) — НЕ масштабировать embeddings

## КЛЮЧЕВОЙ ПЕРЕСЧЁТ

Средний raw_match Atlas = 38% — ВЫШЕ vLLM "27%". Drafter не хуже vLLM.
Но τ_Atlas=1.97 vs τ_vLLM=4.09. → **Проблема не в drafter, а в verify acceptance методе.**

Гипотеза: vLLM verify = **tree/token verification** (несмежные совпадения), Atlas = **linear prefix** (reject на первом несовпадении). Это бы объяснило весь разрыв при равном raw_match.

→ Diver-2 проверяет как vLLM target VERIFY принимает токены и как считается τ=4.09.
→ Diver-1 проверяет 3 точки Atlas propose: position ids noise, mask causal/bidir, mask_token_id.

## РАЗРЕШЕНИЕ ЗАГАДКИ (Diver-2, чтение vLLM verify кода)

**Tree verification НЕТ ни у кого.** Оба — linear prefix с остановкой при первом reject. Моя гипотеза про tree опровергнута.

### Причина 1: метрики несравнимы (артефакт сравнения)
- vLLM "27.2% acceptance" = `num_accepted/num_draft_tokens` (знаменатель = num_drafts × γ) — accepted/drafted по всему γ-окну
- Наши "38% per-position" = независимый argmax-match по позиции — ДРУГАЯ величина
- vLLM per-position rate (metrics.py) = префиксная кривая выживания P[accept_len>i], тоже не независимая
- τ=4.09 ⇒ num_accepted/num_drafts=3.09 (+1 bonus). Самосогласовано при **γ≈11.4, НЕ 16**
- **Наш реальный τ = 1.74 mean accepted + 1 bonus = 2.74** (не 1.97)

### Причина 2: stochastic acceptance при T>0
- vLLM greedy (T=0): argmax-match — ИДЕНТИЧНО Atlas
- vLLM random (T>0): `rejection_random_sample_kernel` — принять с вероятностью min(1, target_p(draft)/draft_p(draft)). НЕ требует argmax. Принимает в разы длиннее.
- Atlas K=2: уже есть `dflash_stochastic_accept` (коммит 7f4682c) ✓
- Atlas K=γ (verify_d.rs): argmax-match, stochastic НЕ реализован ✗

### Остаток при T=0 (главное)
vLLM референс τ=4.09 был при **T=0** (greedy, no thinking). При T=0 оба argmax-match.
Значит при T=0 vLLM prefix выживает дольше: их accepted=3.09 vs наш 1.74.
→ **Наш pos0 argmax-match (53%) слишком низкий. vLLM pos0 должен быть ~80%.**
→ Это про КАЧЕСТВО propose pos0, не про метод verify. Diver-1 три точки могут поднять.

## ОБНОВЛЁННАЯ СТРАТЕГИЯ К vLLM-СКОРОСТЯМ

1. **Поднять pos0/early argmax-match** (53%→80%) — propose quality. Зависит от Diver-1 (position ids/bidir mask/mask token). ГЛАВНЫЙ рычаг при T=0.
2. **Stochastic acceptance в K=γ verify** (verify_d.rs) — для T>0, копия логики из K=2.
3. **Честный замер** — мерять τ=mean accept_len на γ≈11-12 (как vLLM), а не accepted% на γ=16.
4. **Prefill verify** для скорости (verify_ms) — отдельно от acceptance.

## Atlas propose 3 точки vs vLLM (Diver-1) — ВСЕ СОВПАДАЮТ

| Точка | vLLM | Atlas | Verdict |
|---|---|---|---|
| noise position IDs | монотонные last_pos+1..+15 | `[position-1, position+0..+14]` монотонные | ✅ MATCH |
| attention mask | non-causal bidirectional | `causal=false`, окно n_attn=eff_ctx+γ | ✅ MATCH |
| mask_token_id | embed(mask_token_id) noise1+ | 248070 из config, noise1+; noise0=bonus | ✅ MATCH |

Propose структурно идентичен vLLM. Разрыв НЕ в этих точках.

## Последнее подозрение: 1-step lag в context KV hidden

Diver-1: Atlas захватывает h(bonus_{N-1}) на verify (try_dflash_capture), а noise0=embed(bonus_N).
Т.е. последний context slot отстаёт на 1 токен от noise0.
vLLM (по Diver-1): context KV = h(bonus_N) синхронно с noise0=embed(bonus_N).

ПРОТИВОРЕЧИЕ Diver-1 vs Diver-2 про combine_hidden_states. Diver-2 (читала vLLM): noise0=чистый embed, FC→context KV. Diver-1: 1-step lag в захвате.

→ Решающий вопрос Diver-2: какой target hidden vLLM кладёт в context KV — h текущего bonus токена (только что засемпленного) синхронно, или предыдущего? На каком шаге захват? Если синхронно h(bonus_N), а Atlas h(bonus_{N-1}) — это финальный фикс pos0.

## РАЗРЕШЕНИЕ context KV sync (Diver-2 vLLM код) — EAGLE-style pairing

vLLM эталон:
- Самый свежий валидный слот context KV = target hidden позиции `last_pos` (та, что ЗАСЕМПЛИЛА bonus) — hidden с токеном ЭТОЙ позиции на входе
- bonus_N подаётся ТОЛЬКО как embed в noise0 на позиции `last_pos+1`
- **h(bonus_N) (прогон target ПО bonus_N) НЕ существует на момент propose и НЕ в context**
- Последний слот context = noise0_position − 1
- Семантика EAGLE: embed(новый токен на pos P+1) спаривается с target hidden позиции P (которая токен породила)

**Нет "1-step lag bug" в vLLM — это by design.** Diver-1 ошибся: и про add в noise0 (Bug #2), и про lag.

### Потенциальный РЕАЛЬНЫЙ сдвиг +1 в Atlas
Diver-2: "сдвиг +1 будет, если прогоняете target по bonus_N и кладёте h(bonus_N) в context".
Atlas verify batch = [last_token, draft]. try_dflash_capture(layer_idx, 0) = row 0 = hidden С last_token(=bonus) на входе.
→ Это h(bonus), а НЕ hidden позиции-породившей-bonus. Возможен сдвиг +1 относительно vLLM.
→ Diver-1: сверить точную позиционную семантику Atlas capture с эталоном EAGLE.

## ФИНАЛЬНЫЙ ДИАГНОЗ (Diver-1, цитаты кода verify_b.rs/propose.rs/mod.rs)

K=2 verify batch = [last_token@N, draft@N+1]. Два захвата (verify_b.rs:299-300):
- row 0 = h(last_token@N) → dflash_hidden_save 1st half
- row 1 = h(draft@N+1) → 2nd half

На ACCEPT два append в порядке:
1. propose decode-append (propose.rs:232): row 0 = h(last_token@N) → ctx. Потом forward_block генерит draft.
2. dflash_accept_append (mod.rs:303, ПОСЛЕ propose): row 1 = h(draft@N+1) → ctx.

### Баг 1: EAGLE shift (ГЛАВНЫЙ — объясняет pos0 53% vs vLLM ~80%)
При draft-gen свежайший ctx slot = row 0 = **h(last_token как ВХОД)** = hidden прогнанный target С bonus на входе. Это на ШАГ ВПЕРЁД от EAGLE.
vLLM: свежайший = h(позиции что ПОРОДИЛА bonus) = row 1 предыдущего verify (h(draft@N+1) чей argmax дал bonus).
Atlas добавляет EAGLE-правильный row 1, но ПОСЛЕ propose → помогает только следующему шагу, не текущему draft.
Подтверждение в самом коде Atlas (forward_block.rs:310-317): "target_hidden_stack 1 step behind last_token".

### Баг 2: позиционные метки (баг в нашем position fix, коммит 8917168)
Оба append пишут метку N+1 (propose.rs:247 position-1=N+1; mod.rs:326 seq_len-1=N+1), хотя row 0 hidden принадлежит позиции N.
ctx_slot_positions: N+1, N+1, N+3, N+3... — дублирует нечётные, пропускает чётные. row-0 слот (позиция N) помечен N+1.
verify_b.rs:296-298 комментарий заявляет правильное намерение (row 0@N, row 1@N+1), но код пишет обе N+1.

## ПЛАН ФИКСА (к vLLM-уровню pos0)

**Fix Баг 2 (простой):** propose decode-append должен писать позицию N (= position-2, не position-1) для row 0; dflash_accept_append пишет N+1 для row 1. Развести метки.

**Fix Баг 1 (EAGLE, главный):** drafter должен кондиционироваться на row 1 (h что породил bonus), а не row 0 (h с bonus на входе), как свежайший ctx slot ПЕРЕД forward_block. Требует переупорядочить: EAGLE-correct hidden доступен до propose, не после. Нетривиально — затрагивает порядок propose/accept append.
2. **Extend Bug #1 fix to K=4+ paths** — добавить accept-append с позицией в verify для K≥4
3. **Reduce verify+propose overhead** — для K=2 нужно (verify+propose) < 118ms для выигрыша; сейчас 172ms
4. **Bug #2 (combine_hidden_states)** — правильный фикс требует захвата h(target_greedy) в verify path; даст ещё +acceptance
