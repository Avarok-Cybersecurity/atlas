# DFlash Throughput Gap — Findings & Next Steps (consolidated, Jun 30)

Это сводный файл для продолжения работы "с чистого листа". Полная история и сырые
данные — в `NEXT-STEPS-THROUGHPUT.md` (журнал) и `BENCH-27B-DFLASH.md` (числа).
Acceptance-расследование (отдельная, уже решённая задача) — `DEBUG-ACCEPTANCE.md`.

## TL;DR

- **Acceptance решена и закрыта:** τ=6.56 (best, thinking ON), конкурентно/выше vLLM. Не трогать.
- **Throughput gap реальный: vLLM 64.56 vs Atlas 22 tok/s single-request = 2.9× медленнее.**
- **Причина НЕ в приёмке, НЕ в CUDA-графах (графы дают только +5%), а в эффективности ядер
  больших FP4-GEMM** (наша phase-3: norm+proj+FFN). Это главная и единственная цель сейчас.
- **Следующий шаг:** склонить форк-референс `github.com/AEON-7/vllm-ultimate-dgx-spark` (source
  кастомного vLLM-образа aeon) и портировать их FP4-GEMM/fused-SSM подход на Atlas phase-3/SSM.

---

## Как был получен валидный A/B (методология, чтобы не повторять путь)

Прошли через интеррогацию валидности (пользователь справедливо сомневался) — все красные флаги сняты:

1. **Модель-таргет идентична** — сверена побайтово (двумя независимыми сторонами, включая
   ручную проверку Eксперта): Atlas `/isolorg/models/Qwen3.6-27B-NVFP4` = 20 559 272 552 B,
   vLLM `ocicek/Qwen3.6-27B-NVFP4` (HF API) = 20 559 284 232 B. Разница 11 680 B (0.00006%) —
   только упаковка (ocicek вынес MTP-голову в отдельный файл), тот же формат NVFP4 group-16,
   2672 llm + 15 MTP тензора, 64 слоя.
2. **Промпт дословный** Atlas-bench: `Write a Python implementation of a min-heap with insert,
   extract_min, heapify (O(n) build from list), and peek. Show time complexity for each operation.
   Include a complete working demo with at least 10 elements.` T=0, max_tokens=700.
3. **Thinking выровнен** — оба прогона no-think (`enable_thinking=false` / `--disable-thinking`).
   Важный нюанс: vLLM спекулирует и ВНУТРИ thinking (нет reasoning-гейта в коде spec_decode),
   Atlas — нет (`mod.rs:340`). Так что no-think A/B честен и даже немного в пользу Atlas.
4. **Драфтер не устарел** — обе стороны грузят один и тот же HF-snapshot `0919688`
   (z-lab/Qwen3.6-27B-DFlash, 2026-04-27, latest на момент проверки), идентичные веса.
5. **Образ vLLM кастомный**, не stock: `ghcr.io/aeon-7/aeon-vllm-ultimate:latest`,
   vLLM `0.23.0+aeon.sm121a.dflash`, собран из исходников под sm_121a (GB10) с DFlash-патчем
   (PR #40898, без которого DFlash в stock vLLM не работает) + NVFP4/FP8 KV-cache на Triton
   (#44389) + flashinfer-cutlass/vllm-cutlass GEMM + TurboQuant. Source: репозиторий
   `github.com/AEON-7/vllm-ultimate-dgx-spark` — **это и есть форк, который предстоит склонить**.

## Финальная валидированная таблица

| Движок | mode | tok/s (total) | τ | per-step |
|---|---|---|---|---|
| **vLLM DFlash** | no-think | **64.56** | 7.34–7.73 | verify 95.2ms + propose 27.7ms + post 11.5ms ≈ 134ms |
| **Atlas DFlash** | no-think | 16.4 | 4.80 | propose 71.7ms + verify 269.3ms ≈ 341ms |
| Atlas DFlash | thinking ON (best) | **22** | **6.56** | propose ~113ms + verify 269ms |

Per-position acceptance близки (vLLM pos0 88% vs Atlas pos0 94%) — **приёмка не виновата**.

## Решающий профиль (Diver-2, in-process torch.profiler, custom-scopes, eager)

**(A) CUDA-graphs vs `--enforce-eager` на vLLM — within-session:**

| режим | tok/s decode | τ |
|---|---|---|
| graphs ON | 60.33 | 7.3 |
| `--enforce-eager` | 57.52 | 7.4 |
| **дельта** | **+4.9%** | — |

→ **launch-overhead НЕ узкое место.** Decode-шаг compute/memory-bound. C5 (разблокировать наши
графы для K=γ) демотирован — потенциальный выигрыш ~5%, не 2.9×.

**(B) Per-kernel доминирование vLLM (Self-CUDA %, eager):**

| ядро | доля | ↔ Atlas-фаза |
|---|---|---|
| **`flashinfer_mm_fp4`** (NVFP4 GEMM: FFN + проекции таргета) | **54.4%** | ↔ **phase-3** (norm+proj+FFN, 89ms у нас) — ⭐ ГЛАВНАЯ ЦЕЛЬ |
| `aten::mm` → cutlass bf16 (drafter 5 слоёв + lm_head vocab 248K) | 30% | ↔ propose + head |
| `qwen_gdn_attention_core` | 8.6% | ↔ phase-2 GDN |
| `fused_sigmoid_gating_delta_rule_update_kernel` (fla, single-launch) | 8.4% | ↔ phase-2/phase-1 |
| conv1d / softmax-attn / norms / fp4-quant | ~4% | ↔ phase-1 conv / attn / head |

**Прежний вывод «weight-load floor ~270ms физически неустраним» (из C4 K-tuning исследования) —
ОПРОВЕРГНУТ.** vLLM делает тот же verify (та же модель, тот же формат) за 95ms против наших 269ms.
Это не floor — это ядра. Отрыв пропорционален и в verify (2.8×), и в propose (2.6×) — размазан,
не локализован в одной фазе.

## Прочие закрытые вопросы (не возвращаться)

- **K-tuning (C4) — тупик.** Доминанты K-независимы (weight-load floor saturates ~6 токенов).
  Снижение K теряет τ без экономии verify. Держать K=16.
- **C1 conv fusion — уже сделано и в коде.** verify 496→269ms, tok/s 12→22, byte-identical
  (флаг `ATLAS_DFLASH_CONV_FUSION=1`, коммит `7057649`).
- **wy16 (GDN single-launch) — уже в коде.** Закрыл GDN-replay, 520→495ms.
- **Thinking ПОМОГАЕТ Atlas-приёмке, не мешает.** No-think уронил τ 6.56→4.80, tok/s 22→16.4 —
  drafter лучше предсказывает код после reasoning-цепочки в контексте. Снимать guard `mod.rs:340`
  потенциально даёт выигрыш на end-to-end latency (694 think-токена сейчас без спекуляции), но
  это Workstream B — отдельная задача, после кернел-работы (см. ниже), нужен сначала замер
  приёмки DFlash на think-токенах (может быть net-negative на eager).

## C5 — разбор готов, но отложен (read-only, Diver-1)

Если/когда вернёмся: guard `verify_d.rs:186-213` (`force_eager` по умолчанию для K=γ). Корень —
legacy dual-buffer SSM (`pre_verify_copy_async` + `commit_verify_state_async`), которого нет в
рабочем K=2-пути. Полфикса уже в коде (`commit_accepted_prefix_dispatch`, async_chkpt.rs:229-278) —
K=γ его просто не использует. Сложность средняя. **Выигрыш ~5% (см. профиль выше) — низкий приоритет.**

---

## NEXT STEPS (актуальный план)

### Шаг 1 — ЗАКРЫТ (Jun 30, Diver-2). Форк aeon склонирован и изучен.
`github.com/AEON-7/vllm-ultimate-dgx-spark` — публичный, 93 KB, **build-репо** (Dockerfile +
3 Python-патча + бенчи), сам vLLM-source НЕ вендорнут (тянется при сборке из
`lesj0610/vllm@e8c77b85`, PR #44389). Клон в `/tmp/vllm-ultimate-dgx-spark` (эфемерный, Diver-2).

**Главная находка: оба доминирующих ядра — ЧУЖИЕ библиотеки, aeon их не писал и не патчил.**

| ядро | доля cost | источник | aeon трогал? |
|---|---|---|---|
| `flashinfer_mm_fp4` | 54.4% | **flashinfer 0.6.8.post1** (flashinfer-ai, сторонняя) | нет |
| `fused_sigmoid_gating_delta_rule_update` | 8.4% | **fla** (flash-linear-attention, вендорнут в vLLM, Triton) | нет |

Все правки aeon — 3 чисто-Python патча, ноль CUDA: `patch_cudagraph_align.py` (выравнивание
capture-size графов под 1+K spec-токенов), `patch_cuda_optional_import.py` (RTLD_LAZY dlopen,
чтобы SM100-only символы не ломали загрузку на sm_121), `patch_kv_cache_utils.py` (None-safe
`min(block_size)` для гибридных моделей). Плюс build-флаги (`TORCH_CUDA_ARCH_LIST=12.1a`) и
KV-cache PR #44389. **Скорость vLLM на phase-3 — заслуга flashinfer/CUTLASS, не aeon.**

**FP4-GEMM схема (для порта в Atlas):**
- `vllm::flashinfer_mm_fp4` → flashinfer → **CUTLASS NVFP4 block-scaled GEMM**, JIT-исходники
  в `flashinfer/data/csrc/fp4_gemm_cutlass_sm120.cu` (187 строк) — **специализирован ровно под
  SM120/SM121 = наш GB10**, плюс `include/flashinfer/gemm/CutlassFp4GemmRunner`.
- Тип `FP4GemmType::W4A4_NVFP4_NVFP4` — веса И активации оба 4-bit NVFP4 (E2M1, group_size=16,
  per-block FP8-E4M3 scale + global scale) — совпадает с нашим форматом. `ClusterShape_1x1x1`,
  out dtype bf16, arch sm120. Активации квантуются на лету (`scaled_fp4_quant`, мелкое ядро ~0.7%).
- **CUTLASS — header-only C++ template-библиотека (BSD-3, не Triton)** → инстанцируется прямо
  из `.cu`, линкуется в Rust+CUDA Atlas напрямую — **не нужно переписывать GEMM с нуля**.
- Два готовых референс-конфига: (a) flashinfer `fp4_gemm_cutlass_sm120.cu`, (b) проще —
  vLLM свой `cutlass_scaled_fp4_mm` (`vllm/_custom_ops.py` → `csrc/`).

**Fused SSM (GDN) схема:** `fused_sigmoid_gating_delta_rule_update_kernel` — **Triton**, вендорнут
в `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`. Single-launch fused recurrent
gated-delta-rule, per-token snapshot состояния, выбор initial state по `num_accepted_tokens`
(IS_SPEC_DECODING). **Triton из Rust не линкуется** — здесь только ручной CUDA-порт алгоритма
(one block per batch×head, state [head_dim×head_dim] в shared/regs). Вторично, после GEMM.

### Шаг 2 — K1: улучшить СОБСТВЕННЫЙ phase-3 GEMM Atlas (главный рычаг, 54% коста vLLM)
**Решение пересмотрено (Jun 30): НЕ тащить flashinfer/CUTLASS как зависимость.** Цель —
улучшить нашу СОБСТВЕННУЮ реализацию, используя вскрытую схему (NVFP4 W4A4, group_size=16,
block-scaled, per-block FP8-E4M3 scale + global scale, tile/cluster-конфиг под sm_120a) как
референс для понимания, ЧТО именно делает чужой GEMM эффективнее — без вендоринга чужого кода.

**Read-only разведка ЗАВЕРШЕНА (Diver-1, Jun 30) — нашла реальные точки, гипотеза скорректирована:**

Исходная гипотеза «у нас тупой W4A16/BF16-MMA» — **неточна.** Фактический model-specific kernel
(`kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:w4a16_gemm_t`) уже делает **FP8 MMA**: веса
FP4→E4M3, активации BF16→E4M3 в регистрах (cast встроен в `COMPUTE_MMA`, без отдельного
прохода/барьера — здесь мы НЕ хуже vLLM), `mma.sync.m16n8k32.e4m3`, 2-stage cp.async,
scale-handling без лишних round-trip. Разрыв по этой оси — FP8(m16n8k32) vs FP4(m16n8k64) у
vLLM, т.е. ~2×, не 4×, и значим только если compute-bound (вряд ли при M=16).

**⚠️ ИСПРАВЛЕНИЕ (Jun 30, вторая read-only итерация, Diver-1 поймал свою же ошибку):**
Правки #1 (MoE grouped-GEMM threshold) **НЕ СУЩЕСТВУЕТ для нашего деплоя.** Первая разведка
прочитала generic `MoeLayer`-код, не проверив, какой `FfnComponent` реально инстанциируется.
**Qwen3.6-27B-NVFP4 — dense FFN модель, НЕ MoE** (подтверждено логом сервера —
`weight_loader::qwen35_dense: ... dense FFN`, и `config.json` без единого `*expert*/*moe*` поля).
Весь MoE-путь (`forward_batched`, `forward_prefill.rs:49,59` threshold) — мёртвый код для этого
деплоя, никогда не исполняется. `self.ffn` = `DenseFfnLayer`, не `MoeLayer`.

**Что реально происходит в phase-3 (исправленная картина):** `DenseFfnLayer::forward_prefill`
(`dense_ffn.rs:456-538`) уже идёт через батч-GEMM (`w4a16_gemm_n128_m128_v2`, M_TILE=128,
8 варпов/CTA) — НЕ per-token цикл, никакого redundant expert-weight reload (экспертов нет).
Один проход gate/up/down GEMM с M=16 (padded в M_TILE=128 тайл).

**Реальная доминанта неэффективности — пункт occupancy (бывший #2), хуже чем оценивалось:**
M_TILE=128 при verify M=16 ⇒ **7/8 варпов CTA простаивают на мусорных строках** (хуже, чем
out-proj M_TILE=64 → 3/4 простоя). Это главный подтверждённый кандидат сейчас.

Текущий список правок (актуализирован):
- **#2 occupancy (ГЛАВНЫЙ кандидат сейчас):** verify-специализированный kernel-вариант с
  M_TILE=16/32 (1-2 варпа/CTA) для `DenseFfnLayer::forward_prefill` (gate/up/down GEMM, M_TILE=128
  сейчас → 7/8 простой) И для SSM out-proj (`w4a16_gemm_t`, M_TILE=64 → 3/4 простой). Нужен ncu
  для подтверждения occupancy-vs-bandwidth перед оценкой выигрыша.
- **#3 pipeline depth:** только 2 cp.async-стадии (CUTLASS обычно 3-6) — на weight-bound GEMV
  более глубокий pipeline лучше прячет HBM-латентность.
- **#4 FP4-MMA вместо FP8** (m16n8k64 vs m16n8k32) — дорогая правка (новый MMA-путь + per-token
  A-scale), ~2× tensor-core throughput, но только если compute-bound — под вопросом при M=16.

Цель: phase-3 89ms → приблизиться к vLLM (95ms на весь verify-шаг, у нас сейчас 269ms всего).
Следующий шаг: read-only оценка реальной occupancy-потери по коду (#2), затем A/B с флагом —
требует GPU, согласовать перед запуском. `ATLAS_DFLASH_TIMING` даёт только агрегат phase-3,
без out_proj/dense-FFN split — для точного diagnosis в итоге нужен ncu.

### Шаг 3 — K2: улучшить fused SSM/GDN в нашем коде (~17% коста vLLM, после K1)
Аналогично — НЕ портировать Triton-код fla напрямую, а использовать его алгоритмическую схему
(single-launch fused recurrent gated-delta-rule, state-snapshot по num_accepted_tokens) как
референс, чтобы найти и устранить конкретные неэффективности в нашем текущем wy16 + phase-1 conv
пути. Вторично после K1.

### Шаг 4 (опционально, низкий приоритет) — C5, графы
Только если K1/K2 исчерпаны и нужны последние ~5%.

### Шаг 5 (отдельный фронт, после K1) — Workstream B: DFlash на thinking
Замерить приёмку DFlash на think-токенах. Если хорошая — снять guard `mod.rs:340`,
получить выигрыш на end-to-end latency для thinking-ответов.

---

## Дайверы (Monet subsessions) — resume, не респавнить

- **Diver-1** `claude_2a97ebc5-39b0-4508-a132-74b416a0b317` — Atlas код/GPU/сервер. opus.
- **Diver-2** `claude_2c5e18e6-7288-4b8a-8c4d-f2981d627179` — vLLM/внешний референс reader. opus.
- Резюме: `source $MONET_ENV_FILE; curl $MONET_URL/subsession/result?sessionId=...`
- **GPU-мьютекс (КРИТИЧНО):** никогда Atlas server + vLLM Docker одновременно.
  Проверять `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` перед стартом.
- **GPU сейчас:** свободна, оба сервиса погашены.

## Документы (внешняя память)
- `THROUGHPUT-GAP-FINDINGS.md` — этот файл, отправная точка.
- `NEXT-STEPS-THROUGHPUT.md` — полный журнал/история этого расследования (детальнее).
- `BENCH-27B-DFLASH.md` — все числа бенчей, K-sweep таблица.
- `DEBUG-ACCEPTANCE.md` — acceptance-расследование (решено).
- `OPTIMIZATIONS-CATALOG.md` — полный список всех изменений/коммитов в коде.
- `SSM_VERIFY_PLAN.md` / `PREFILL_VERIFY_PLAN.md` — более старые планы verify-ускорения (C1 уже выполнен оттуда).

## Env-флаги (все off = byte-identical к pre-optimization baseline)
`ATLAS_DFLASH_EAGLE_FIX`, `ATLAS_DFLASH_CONV_FUSION`, `ATLAS_VERIFY_PREFILL_SSM`,
`ATLAS_VERIFY_PREFILL_ATTN`, `ATLAS_DFLASH_TIMING`, `ATLAS_DFLASH_DRAFT_CAP=15` (K=16).
Default форсит eager для K=γ (`ATLAS_DFLASH_UNSAFE_KGAMMA_GRAPH=1` снимает, но не нужно сейчас).
