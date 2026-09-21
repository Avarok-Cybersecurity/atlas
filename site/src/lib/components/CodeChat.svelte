<script>
  // "Ask the codebase" — a lab instrument that prints its telemetry onto
  // receipt stock. Shell cloned from BenchmarkDashboard (backdrop, dialog,
  // aria, scroll lock, Escape) plus a real focus trap and focus return since
  // this dialog holds form controls. All engine state comes from
  // chat/state.svelte.js (the contract SSOT), this component never fetches.
  import {
    chat, ensureReady, abortLoad, setKey, clearKey, ask, setChatModel, resetChatModel
  } from '../chat/state.svelte.js';
  import ChatMessage from './ChatMessage.svelte';
  import { codeChat } from '$lib/data.js';

  let { onclose, onready } = $props();

  let dialogEl = $state(null);
  let inputEl = $state(null);
  let logEl = $state(null);

  let question = $state('');
  let messages = $state([]);
  // Every entry carries a stable id so the streaming card and the settled card
  // are the SAME keyed node. Without that the list swapped one component for
  // another at completion, remounting the DOM: the card blinked, its print
  // animation replayed, and the sources snapped in.
  let seq = $state(0);
  let asking = $state(false);
  let askError = $state(null);
  let lastQuestion = '';
  let keyDraft = $state('');
  let keyVisible = $state(false);

  const LOADING = ['wasm-init', 'manifest', 'loading-cached', 'downloading', 'caching', 'indexing'];
  const loading = $derived(LOADING.includes(chat.status) || chat.status === 'idle');
  const ready = $derived(chat.status === 'ready');
  const loadError = $derived(chat.status === 'error' ? chat.error : null);
  const canAsk = $derived(ready && chat.keyState === 'set' && !asking);
  const shortCommit = $derived(chat.corpus ? String(chat.corpus.commit).slice(0, 7) : '');

  // Sources open by default on desktop, collapsed on the phone sheet. This
  // component only ever mounts in the browser (lazy import on click).
  const sourcesOpen =
    typeof window === 'undefined' ? true : !window.matchMedia('(max-width: 860px)').matches;

  // A spent daily allowance names a real reset time, and the same model without
  // the ":free" suffix is usually available on credits. Both are shown only
  // when they actually apply.
  const resetLabel = $derived(
    askError?.resetAt
      ? new Date(askError.resetAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
      : ''
  );
  const FREE = ':free';
  const paidModel = $derived(
    chat.chatModel.endsWith(FREE) ? chat.chatModel.slice(0, -FREE.length) : ''
  );
  const modelIsDefault = $derived(chat.chatModel.endsWith(FREE));

  // The in-flight streamed card takes over from the pending line as soon as
  // the first thinking or answer token lands.
  const streamLive = $derived(
    asking && chat.stream && (chat.stream.reasoningText || chat.stream.answerText)
      ? chat.stream
      : null
  );

  // The in-flight card takes the id the finished message will be pushed with,
  // so the keyed each block updates props in place instead of remounting.
  const rendered = $derived(
    streamLive ? [...messages, { id: seq, role: 'assistant', live: streamLive }] : messages
  );

  const mb = (bytes) => (bytes / 1048576).toFixed(1);

  const statusTone = $derived(
    chat.status === 'error'
      ? 'error'
      : ready
        ? asking
          ? 'working'
          : 'ready'
        : chat.status === 'idle'
          ? 'idle'
          : 'working'
  );

  const statusText = $derived.by(() => {
    if (asking) return codeChat.phase[chat.msgPhase] ?? codeChat.phase.writing;
    if (ready && chat.corpus) {
      const c = chat.corpus;
      // The pill has limited width and ellipsizes from the right, so it carries
      // only the segments not shown elsewhere. The file count lives in the
      // welcome line (and the title tooltip) — including it here pushed
      // "dim 2048" past the ellipsis at 1280px.
      const parts = [codeChat.status.ready, shortCommit, `${c.chunks} chunks`, `dim ${c.dim}`];
      if (chat.offline) parts.push(codeChat.offlineBadge);
      return parts.filter(Boolean).join(' · ');
    }
    if (chat.status === 'downloading' && chat.progress.totalBytes > 0)
      return `${codeChat.status.downloading} ${mb(chat.progress.loadedBytes)} of ${mb(chat.progress.totalBytes)} MB`;
    if (chat.status === 'indexing' && chat.progress.totalPoints > 0)
      return `${codeChat.status.indexing} ${chat.progress.indexed} of ${chat.progress.totalPoints}`;
    return codeChat.status[chat.status] ?? chat.status;
  });

  const STAGE_ORDER = ['wasm-init', 'manifest', 'corpus', 'indexing'];
  const stages = $derived.by(() => {
    const cur = ['loading-cached', 'downloading', 'caching'].includes(chat.status)
      ? 'corpus'
      : chat.status;
    const idx = STAGE_ORDER.indexOf(cur);
    return STAGE_ORDER.map((id, i) => ({
      id,
      label: codeChat.loader.stages[id],
      state: idx === -1 ? 'pending' : i < idx ? 'done' : i === idx ? 'active' : 'pending',
      detail: i === idx ? stageDetail() : ''
    }));
  });

  function stageDetail() {
    const p = chat.progress;
    if (chat.status === 'downloading' || chat.status === 'caching')
      return p.totalBytes > 0 ? `${mb(p.loadedBytes)} of ${mb(p.totalBytes)} MB` : '';
    if (chat.status === 'indexing' || chat.status === 'loading-cached')
      return p.totalPoints > 0 ? `${p.indexed} of ${p.totalPoints}` : '';
    return '';
  }

  const barPct = $derived.by(() => {
    const p = chat.progress;
    if ((chat.status === 'downloading' || chat.status === 'caching') && p.totalBytes > 0)
      return Math.min(100, (p.loadedBytes / p.totalBytes) * 100);
    if ((chat.status === 'indexing' || chat.status === 'loading-cached') && p.totalPoints > 0)
      return Math.min(100, (p.indexed / p.totalPoints) * 100);
    return null;
  });

  // Scroll lock + initial focus + focus return, and kick the corpus load.
  // ensureReady is idempotent and reports failures through chat.status/error.
  $effect(() => {
    const prev = document.activeElement;
    document.body.style.overflow = 'hidden';
    dialogEl?.focus();
    ensureReady().catch(() => {});
    return () => {
      document.body.style.overflow = '';
      if (prev instanceof HTMLElement) {
        prev.focus();
        if (document.activeElement !== prev) {
          // The opener can refuse focus after close — on a phone it lives in
          // the nav drawer, which is display none again by now. Land on the
          // drawer toggle instead of dropping focus to body.
          document.querySelector('.nav-toggle')?.focus();
        }
      }
    };
  });

  $effect(() => {
    if (chat.status === 'ready') onready?.();
  });

  // Keep the newest print in view. While tokens stream we jump instantly so the
  // text never lags behind itself; once the answer settles and the sources
  // print, we glide instead, which reads as the receipt finishing rather than
  // the pane snapping.
  const reducedMotion = () =>
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  $effect(() => {
    void messages.length;
    void asking;
    void chat.msgPhase;
    void chat.stream?.reasoningText;
    void chat.stream?.answerText;
    if (!logEl) return;
    const settling = !streamLive && !asking;
    if (settling && !reducedMotion()) {
      logEl.scrollTo({ top: logEl.scrollHeight, behavior: 'smooth' });
    } else {
      logEl.scrollTop = logEl.scrollHeight;
    }
  });

  function close() {
    // Cancel-safe: closing mid load aborts the fetch and cleans partial cache.
    if (LOADING.includes(chat.status)) abortLoad();
    onclose();
  }

  function onDialogKeydown(e) {
    if (e.key !== 'Tab' || !dialogEl) return;
    const els = [
      ...dialogEl.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), summary, [tabindex]:not([tabindex="-1"])'
      )
    ].filter((el) => el.offsetParent !== null);
    if (els.length === 0) return;
    const first = els[0];
    const last = els[els.length - 1];
    if (e.shiftKey && (document.activeElement === first || document.activeElement === dialogEl)) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  async function runAsk(q, history) {
    asking = true;
    askError = null;
    try {
      const res = await ask(q, history);
      messages.push({
        id: seq++,
        role: 'assistant',
        text: res.answer,
        sources: res.sources,
        reasoning: res.reasoning,
        reasoningMs: res.reasoningMs
      });
    } catch {
      const k = chat.error?.kind;
      askError = {
        kind: ['rate', 'quota', 'key'].includes(k) ? k : 'generic',
        // A per-day cap knows when it lifts; the card names that time.
        resetAt: k === 'quota' ? (chat.error?.resetAt ?? null) : null,
        // Engine-crafted diagnostics (e.g. the embedding-dim mismatch) are
        // written for humans — surface them instead of the generic copy.
        detail: k === 'corpus' ? chat.error?.message : null
      };
    } finally {
      asking = false;
    }
  }

  function submit(text) {
    const q = (typeof text === 'string' ? text : question).trim();
    if (!q || !canAsk) return;
    question = '';
    lastQuestion = q;
    const history = messages.map((m) => ({ role: m.role, content: m.text }));
    messages.push({ id: seq++, role: 'user', text: q });
    runAsk(q, history);
  }

  function retryAsk() {
    if (!lastQuestion || asking) return;
    if (askError?.kind === 'key') {
      clearKey();
      askError = null;
      return;
    }
    runAsk(
      lastQuestion,
      messages.slice(0, -1).map((m) => ({ role: m.role, content: m.text }))
    );
  }

  // Spending the visitor's credits is their call, so this only ever runs from
  // an explicit click on the quota card.
  function usePaidModel() {
    if (!paidModel) return;
    setChatModel(paidModel);
    askError = null;
    retryAsk();
  }

  function chipPick(q) {
    if (canAsk) submit(q);
    else question = q;
  }

  function saveKey(e) {
    e.preventDefault();
    const k = keyDraft.trim();
    if (!k) return;
    setKey(k);
    keyDraft = '';
    keyVisible = false;
    inputEl?.focus();
  }

  function swapKey() {
    clearKey();
    askError = null;
  }

  // The full deep-learning glyph, printed faintly onto the instrument body.
  // It ships only in this lazy chunk, never in the initial bundle.
  const MOTIF_D = 'M292.47 53.64C296.66 47.78 297.37 42.86 304.7 39.32C316.91 33.44 331.45 39.91 335.78 52.6C337.35 57.18 336.37 61.12 336 65.69C357.49 79.46 378.99 93.22 400.48 106.98C405.51 104.56 409.08 101.41 415.06 101.22C428.97 100.77 439.45 113.41 437.92 126.85C437.44 131 434.99 134.05 433.37 137.7C448.36 158.28 463.35 178.85 478.33 199.42C483.54 198.34 488.23 197.44 493.46 199.27C506.47 203.82 512.45 219.45 505.72 231.52C503.02 236.35 498.3 240.09 493.07 241.82C491.02 242.5 488.59 242.35 486.72 243.38C485.17 246.94 484.74 251.19 483.77 254.96C481.73 262.92 479.74 270.9 477.79 278.89C476.86 282.64 473.9 288.7 474.99 292.47C498.27 308.42 479.5 343.25 453.92 331.81C448.39 329.34 445.64 324.7 442.56 319.85C426.51 321.35 410.46 322.84 394.4 324.34C390.38 333.77 386.61 341.16 375.23 342.83C371.78 343.34 368.75 342.22 365.41 342.16C356.12 356.93 346.83 371.69 337.54 386.45C341.98 393.74 345.03 400.66 341.79 409.34C336.04 424.73 315.28 428.62 304.25 416.53C300.62 412.56 298.53 407.26 298.38 401.88C298.33 400.23 298.81 398.64 298.55 397.04C286.86 391.02 275.17 385.01 263.48 379C260.02 381.31 257.53 384.71 253.78 386.82C246.18 391.12 236.75 391.69 228.54 388.85C226.37 388.1 224.47 386.65 222.33 385.93C204.41 404.18 186.49 422.44 168.57 440.69C170.12 446.26 172.11 450.79 170.89 456.83C168.52 468.57 155.43 476.97 143.73 473.78C130.75 470.24 122.89 456.03 128.36 443.26C131.21 436.61 135.57 435.97 139.26 431.34C135.01 407.34 130.75 383.34 126.49 359.34C117.62 356.97 110.08 352.78 107.23 343.33C106.65 341.39 106.66 339.31 106.07 337.36C94.48 334.32 82.89 331.29 71.31 328.25C67.23 333.18 63.45 337.03 57.03 338.74C43.18 342.44 29.49 330.61 29.65 316.8C29.78 306.07 34.37 304.66 40.15 297.61C35.26 280.21 30.37 262.8 25.48 245.39C17.76 242.51 13.1 243.77 7.33 236.05C-0.98 224.92 2.99 207.98 15.85 202.21C20.95 199.93 25.32 200.65 30.63 200.69C42.06 180.32 53.49 159.95 64.92 139.57C62.43 133.69 58.85 131.45 58.41 124.14C57.54 109.89 71.13 98.87 84.84 101.15C89.49 101.92 92.44 105.09 96.56 106.77C118.89 93.08 141.23 79.39 163.57 65.69C163.38 60.8 162.45 56.58 164.21 51.73C168.59 39.62 183.19 33.73 194.78 39.22C198.49 40.98 201.73 43.97 203.96 47.39C205.01 49.02 205.85 53.25 207.86 53.63C214.84 54.95 228.24 53.66 235.89 53.66C254.75 53.66 273.62 53.88 292.47 53.64ZM182.63 49.2C169.78 52.34 174.36 72.66 187.46 69.88C200.96 67.02 196.19 45.89 182.63 49.2ZM311.68 49.2C298.96 52.39 303.56 72.57 316.49 69.88C330.14 67.04 325.39 45.76 311.68 49.2ZM292.47 65.28C272.68 65.07 252.88 65.36 233.09 65.36C228.26 65.36 210.86 64.27 207.44 65.5C205.78 66.11 204.68 70.54 203.67 72.06C200.94 76.13 197.03 77.63 193.53 80.66C195.06 95 196.58 109.34 198.11 123.67C201.93 124.65 205.92 124.83 209.71 126.2C218.56 129.41 226.33 136.1 230.86 144.36C232.36 147.09 233.1 150.16 234.49 152.88C248.21 151.24 261.92 149.6 275.64 147.95C278.79 134.12 282.26 123.95 296.06 117.2C299.58 115.49 303.47 115.05 306.97 113.58C307.26 102.76 307.55 91.94 307.85 81.13C305.65 79.32 302.65 78.62 300.36 76.75C296.38 73.52 295.13 69.35 292.47 65.28ZM169.5 76.02C147.1 89.88 124.7 103.74 102.3 117.6C102.6 119.93 102.9 122.27 103.2 124.61C122.4 131.31 141.6 138.01 160.81 144.72C163.71 142.7 167.01 136.31 170.52 133.32C175.87 128.75 181.24 128.13 185.86 124.92C186.89 117.47 183.5 102.56 182.98 94.22C182.85 92.04 182.49 83.41 181.5 81.96C180.82 80.96 179.46 81.11 178.44 80.79C175.14 79.77 172.64 76.98 169.5 76.02ZM329.41 76.13C326.29 77.86 323.18 79.59 320.06 81.32C319.81 92.17 319.56 103.01 319.31 113.86C326.5 116.59 333.09 118.75 338.61 124.53C341.57 127.63 343.12 131.92 345.94 134.9C350.77 135.43 357.32 132.52 362.01 131.22C369.79 129.06 377.72 127.44 385.5 125.23C387.59 124.64 390.28 124.57 392.23 123.62C394.04 122.73 394.19 118.78 393.94 117.08C372.43 103.43 350.92 89.78 329.41 76.13ZM78.93 112.9C65.56 114.6 69.21 136.29 82.52 133.71C95.9 131.12 92.55 111.17 78.93 112.9ZM413.61 113.26C399.78 116.33 405.05 137.96 418.74 133.84C431.11 130.13 426.38 110.43 413.61 113.26ZM308.47 125.47C298.92 127.49 290.6 134.03 288.19 143.87C284.14 160.4 298.25 177.04 315.37 174.77C340.97 171.39 345.05 136.02 320.98 126.78C317.09 125.29 312.6 124.6 308.47 125.47ZM396.27 135.02C380.58 139.03 364.88 143.04 349.18 147.05C348.63 149.43 349.11 152.07 348.79 154.54C348.13 159.6 346.14 164.01 344.01 168.56C355.51 178.37 367.01 188.17 378.51 197.97C382.7 196.17 386.03 193.93 390.73 193.29C405.58 191.27 412.89 201.21 418.25 212.88C433.68 213.26 449.11 213.65 464.54 214.03C465.9 211.5 467.25 208.97 468.6 206.44C453.53 185.9 438.46 165.37 423.39 144.84C408.44 147.37 406.16 144.9 396.27 135.02ZM193.9 135.8C163.02 138.15 159.44 181.95 189.16 190.21C192.87 191.24 197 191.46 200.83 190.85C234.91 185.41 229.09 133.13 193.9 135.8ZM99.36 136.08C90.27 144.51 88.21 145.66 75.51 145.07C64.03 165.53 52.55 185.98 41.07 206.44C42.65 208.97 44.71 211.1 45.84 213.93C46.82 216.37 46.84 219 47.77 221.4C58.42 223.82 69.07 226.24 79.72 228.66C88.51 211.19 110.29 204.87 126.48 216.61C138.58 205.89 150.68 195.18 162.79 184.46C156.72 176.14 156.69 166.23 157.07 156.41C137.84 149.63 118.6 142.86 99.36 136.08ZM276.57 160.09C263.01 161.82 249.45 163.55 235.89 165.28C235.05 168.84 234.93 172.49 233.72 175.98C232.42 179.7 230.31 182.84 228.57 186.33C241.92 199.47 255.27 212.62 268.62 225.76C271.81 223.95 274.7 221.8 278.07 220.3C291.37 214.39 307.9 215.9 319.87 224.18C323.24 226.52 326.23 229.3 328.94 232.39C330.39 234.05 331.4 236.32 333.15 237.62C344.91 232.52 356.68 227.43 368.44 222.33C368.17 216.88 368.82 212.46 370.82 207.37C359.35 197.62 347.89 187.86 336.42 178.11C332.08 179.59 328.5 183.34 323.91 184.85C310.42 189.27 295.02 185.43 285.44 174.9C282.95 172.17 280.76 169.04 279.19 165.7C278.33 163.85 278 161.58 276.57 160.09ZM170.43 194.07C158.63 204.58 146.83 215.1 135.03 225.61C135.52 228.06 137.09 230.26 137.81 232.69C138.76 235.91 138.55 239.24 139.29 242.44C147.02 244.24 154.75 246.04 162.48 247.84C164.66 244.52 166.42 241.13 169.34 238.33C180.35 227.75 197.92 227.03 209.86 236.45C213.66 239.45 216.9 243.63 218.81 248.07C219.74 250.23 220.03 252.63 221.4 254.54C231.53 254.54 241.66 254.54 251.79 254.54C253.22 250.89 253.65 246.97 255.22 243.33C256.56 240.21 258.7 237.53 260.1 234.49C246.58 221.41 233.05 208.32 219.53 195.24C209.33 202.96 194.98 205.24 182.92 200.75C178.68 199.18 174.59 195.06 170.43 194.07ZM393.11 205.03C376.09 205.36 376.47 231.86 393.47 231.8C410.79 231.74 410.46 204.68 393.11 205.03ZM483.84 210.15C470.55 213 475.34 234.1 488.76 230.82C501.84 227.63 497.09 207.3 483.84 210.15ZM24.62 212.03C10.35 212.51 12.17 235.5 27.35 232.96C39.98 230.84 37.15 211.61 24.62 212.03ZM105.97 222.28C81.02 225.45 85.28 263.2 109.55 260.71C135.33 258.06 130.8 219.14 105.97 222.28ZM464.54 226.21C448.95 225.88 433.37 225.55 417.78 225.22C416.19 228.31 414.59 231.4 412.99 234.49C427.21 253.21 441.44 271.93 455.66 290.64C457.99 290.11 460.33 289.57 462.67 289.03C466.72 272.64 470.78 256.25 474.83 239.86C470.15 235.4 466.98 232.39 464.54 226.21ZM291.61 228.83C256.8 232.64 254.11 281.47 287.33 291.26C291.73 292.56 296.81 292.65 301.36 291.9C333.28 286.57 336.19 240.75 305.91 230.22C301.36 228.64 296.38 228.31 291.61 228.83ZM45.12 233.36C42.44 236.08 39.76 238.79 37.08 241.51C37.06 246.17 39.59 251.39 40.85 255.88C43.13 264.03 45.29 272.2 47.77 280.3C48.95 284.14 49.28 290.45 51.62 293.7C52.4 294.77 60.42 296.01 62.62 297.19C71.75 302.08 71.85 307.93 74.58 316.55C86.22 319.59 97.86 322.63 109.5 325.67C112.08 322.42 115.44 320.23 118.08 317.25C114.91 302.45 111.74 287.64 108.57 272.83C104.95 271.79 101.13 272.09 97.54 270.69C88.44 267.16 81.03 259.77 78.22 250.3C77.27 247.1 77.47 243.77 76.76 240.57C66.22 238.17 55.67 235.77 45.12 233.36ZM373.36 233.79C361.66 238.7 349.96 243.61 338.26 248.52C337.55 251.61 339.35 261.03 338.92 265.82C338.37 271.98 335.26 276.95 333.44 282.65C341.76 289.21 350.08 295.76 358.4 302.31C362.26 301.6 365.52 298.9 369.62 298.3C377.18 297.18 385.27 301.15 389.8 307.11C391.13 308.87 391.82 311.12 393.45 312.58C409.43 310.86 425.4 309.15 441.37 307.43C442.73 304.16 444.09 300.89 445.45 297.61C431.4 279.02 417.34 260.43 403.29 241.84C400.12 242.22 397.21 243.45 393.94 243.54C384.94 243.79 379.73 239.1 373.36 233.79ZM186.91 241.94C162.69 245.38 170.01 285.3 195.33 278.85C218.36 272.99 211.56 238.44 186.91 241.94ZM136.3 254.35C133.75 259.35 130.39 263.48 125.87 266.82C124.11 268.12 121.99 268.85 120.4 270.34C123.67 285.12 126.95 299.9 130.22 314.68C132.71 315.2 135.21 315.71 137.7 316.23C147.43 304.26 157.15 292.29 166.88 280.31C165.67 277.39 163.35 274.97 162.15 271.95C160.51 267.84 160.66 263.71 159.68 259.53C151.89 257.8 144.09 256.07 136.3 254.35ZM251.79 266.26C241.66 266.26 231.53 266.26 221.4 266.26C218.67 270.41 217.97 275.42 214.76 279.52C208.01 288.15 196.69 292.5 185.86 290.84C182.55 290.34 179.66 288.78 176.51 287.87C166.73 300 156.94 312.13 147.16 324.27C149.23 328.03 150.12 331.53 151.26 335.62C171.06 339.68 190.85 343.74 210.64 347.8C217.82 333.86 232.84 325.46 248.52 330.68C255.25 318.88 261.98 307.08 268.71 295.28C264.87 290.57 260.22 287.14 257.24 281.63C254.55 276.66 253.79 271.43 251.79 266.26ZM325.67 292.15C309.18 304.48 299.61 306.36 279.38 301.19C272.63 312.94 265.89 324.68 259.14 336.42C264.16 343.24 268.53 348.17 269.62 357.02C270.09 360.9 268.46 364.79 269.09 368.5C280.78 374.36 292.47 380.22 304.16 386.08C308.97 383.76 311.95 380.09 317.72 379.22C321.11 378.71 324.36 380.24 327.54 379.79C336.78 365.02 346.01 350.26 355.25 335.49C352.24 330.51 349.73 326.63 349.75 320.53C349.76 317.47 351.05 314.57 350.95 311.64C342.52 305.14 334.1 298.65 325.67 292.15ZM460.87 301.25C447.79 304.63 452.77 325.37 466.03 321.85C479.1 318.38 474.09 297.83 460.87 301.25ZM50.39 306.45C37.77 308.94 39.42 328.35 52.6 327.41C66.87 326.39 64.17 303.73 50.39 306.45ZM369.76 310.22C356.22 313.42 361.45 334.84 375.01 330.8C387.53 327.08 382.66 307.16 369.76 310.22ZM128.01 326.57C114.64 327.72 115.84 348.46 129.26 347.69C142.92 346.9 141.76 325.4 128.01 326.57ZM236.5 341.09C212.44 343.58 216.56 381.71 241.04 378.8C265.74 375.86 261.74 338.47 236.5 341.09ZM148.46 347.62C145.61 351.88 142.53 354.47 138.43 357.46C142.71 381.54 146.98 405.61 151.26 429.69C154.07 430.51 156.87 431.32 159.68 432.14C177.57 413.95 195.47 395.76 213.37 377.57C212.44 374.26 210.31 371.3 209.28 367.95C208.46 365.3 208.68 362.35 207.84 359.73C188.05 355.69 168.25 351.65 148.46 347.62ZM318.34 391.12C305.07 393.99 309.93 414.99 323.26 411.79C336.37 408.64 331.7 388.22 318.34 391.12ZM147.57 441.52C134.88 443.58 136.32 463.24 149.39 462.54C163.62 461.77 161.56 439.25 147.57 441.52Z';
</script>

<svelte:window onkeydown={(e) => { if (e.key === 'Escape') close(); }} />

<div class="cc-backdrop" onclick={close} role="presentation">
  <div
    class="cc"
    role="dialog"
    aria-modal="true"
    aria-label={codeChat.navLabel}
    tabindex="-1"
    bind:this={dialogEl}
    onclick={(e) => e.stopPropagation()}
    onkeydown={onDialogKeydown}
  >
    <svg class="cc-motif" viewBox="0 0 512 512" aria-hidden="true" focusable="false">
      <path d={MOTIF_D} fill="currentColor" fill-rule="evenodd" />
    </svg>

    <header class="cc-head">
      <div class="cc-head-titles">
        <span class="slabel cc-slabel">{codeChat.label}</span>
        <h2 class="cc-title">{codeChat.title}</h2>
      </div>
      <p class="cc-status" data-tone={statusTone} aria-live="polite" title={statusText}>
        <span class="cc-status-dot" aria-hidden="true"></span>
        <span class="cc-status-text">{statusText}</span>
      </p>
      <button type="button" class="bd-close cc-close" onclick={close} aria-label={codeChat.closeLabel}>✕</button>
    </header>

    {#if loadError}
      <div class="cc-panel">
        <div class="cc-error" role="alert">
          <span class="cc-error-tag">{(codeChat.errors[loadError.kind] ?? codeChat.errors.generic).tag}</span>
          <p class="cc-error-body">{(codeChat.errors[loadError.kind] ?? codeChat.errors.generic).body}</p>
          <button type="button" class="cc-error-retry" onclick={() => ensureReady().catch(() => {})}>
            {(codeChat.errors[loadError.kind] ?? codeChat.errors.generic).retry}
          </button>
        </div>
      </div>
    {:else if loading}
      <div class="cc-panel cc-load" aria-busy="true">
        <p class="cc-load-title">{codeChat.loader.title}</p>
        {#if chat.corpus || chat.progress.totalBytes > 0}
          <dl class="cc-load-meta">
            {#if chat.corpus}
              <div><dt>{codeChat.loader.commitLabel}</dt><dd>{shortCommit}</dd></div>
              <div><dt>{codeChat.loader.chunksLabel}</dt><dd>{chat.corpus.chunks}</dd></div>
            {/if}
            {#if chat.progress.totalBytes > 0}
              <div><dt>{codeChat.loader.sizeLabel}</dt><dd>{mb(chat.progress.totalBytes)} MB</dd></div>
            {/if}
          </dl>
        {/if}
        <ol class="cc-stages">
          {#each stages as s (s.id)}
            <li class="cc-stage" data-state={s.state}>
              <span class="cc-stage-glyph" aria-hidden="true">
                {s.state === 'done' ? '✓' : s.state === 'active' ? '●' : '○'}
              </span>
              <span class="cc-stage-label">{s.label}</span>
              {#if s.detail}<span class="cc-stage-detail">{s.detail}</span>{/if}
            </li>
          {/each}
        </ol>
        <div
          class="cc-bar"
          role="progressbar"
          aria-label={codeChat.loader.title}
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow={barPct === null ? undefined : Math.round(barPct)}
        >
          <div class="cc-bar-fill" class:cc-shimmer={barPct === null} style="width: {barPct === null ? 100 : barPct}%"></div>
        </div>
        {#if chat.offline}<p class="cc-offline">{codeChat.offlineBadge}</p>{/if}
        <p class="cc-fine">{codeChat.loader.cancelNote}</p>
      </div>
    {:else}
      <div class="cc-log" bind:this={logEl} role="log" aria-live="polite">
        {#if chat.offline}<p class="cc-offline">{codeChat.offlineBadge}</p>{/if}
        {#if messages.length === 0}
          <p class="cc-sub">{codeChat.sub}</p>
          <div class="cc-boot">
            {#each codeChat.boot as line}
              <p class="cc-boot-line">{line}</p>
            {/each}
            {#if chat.corpus}
              <p class="cc-boot-line cc-boot-corpus">
                {chat.corpus.chunks} chunks from {chat.corpus.files} files at {shortCommit}
              </p>
            {/if}
          </div>
          <div class="cc-chips">
            {#each codeChat.starters as starter}
              <button type="button" class="cc-chip" onclick={() => chipPick(starter)}>{starter}</button>
            {/each}
          </div>
        {/if}
        {#each rendered as m (m.id)}
          <ChatMessage message={m.live ? null : m} live={m.live ?? null} {sourcesOpen} />
        {/each}
        {#if !streamLive && asking}
          <p class="cc-pending">
            <span class="cc-status-dot" aria-hidden="true"></span>
            {codeChat.phase[chat.msgPhase] ?? codeChat.phase.writing}
            <span class="cc-pending-bar cc-shimmer" aria-hidden="true"></span>
          </p>
        {/if}
        {#if askError}
          <div class="cc-error" role="alert">
            <span class="cc-error-tag">{codeChat.errors[askError.kind].tag}</span>
            <p class="cc-error-body">{askError.detail ?? codeChat.errors[askError.kind].body}</p>
            {#if askError.kind === 'quota' && resetLabel}
              <p class="cc-error-reset">{codeChat.errors.quota.reset} {resetLabel}.</p>
            {/if}
            <div class="cc-error-actions">
              {#if askError.kind === 'quota' && paidModel}
                <button type="button" class="cc-error-paid" onclick={usePaidModel}>
                  {codeChat.errors.quota.paid}
                </button>
              {/if}
              <button type="button" class="cc-error-retry" onclick={retryAsk}>
                {codeChat.errors[askError.kind].retry}
              </button>
            </div>
          </div>
        {/if}
      </div>
    {/if}

    {#if chat.keyState === 'missing'}
      <div class="cc-key">
        <span class="cc-key-tag">{codeChat.key.tag}</span>
        <p class="cc-key-lead">
          {codeChat.key.lead}
          <a class="link" href={codeChat.key.url} target="_blank" rel="noopener">{codeChat.key.linkText}</a>
        </p>
        <form class="cc-key-form" onsubmit={saveKey}>
          <input
            class="cc-key-input"
            type={keyVisible ? 'text' : 'password'}
            bind:value={keyDraft}
            placeholder={codeChat.key.placeholder}
            aria-label={codeChat.key.inputLabel}
            autocomplete="off"
            spellcheck="false"
          />
          <button type="button" class="cc-key-toggle" aria-pressed={keyVisible} onclick={() => (keyVisible = !keyVisible)}>
            {keyVisible ? codeChat.key.conceal : codeChat.key.reveal}
          </button>
          <button type="submit" class="cc-key-save" disabled={!keyDraft.trim()}>{codeChat.key.save}</button>
        </form>
      </div>
    {:else}
      <div class="cc-keyrow">
        <span class="cc-connected"><span class="dot" aria-hidden="true"></span>{codeChat.key.connectedTag}</span>
        <span class="cc-keyrow-note">{codeChat.key.connectedNote}</span>
        <button type="button" class="cc-keyrow-swap" onclick={swapKey}>{codeChat.key.change}</button>
        <span class="cc-model" title={chat.chatModel}>
          <span class="cc-model-label">{codeChat.model.label}</span>
          <span class="cc-model-id">{chat.chatModel}</span>
          {#if !modelIsDefault}
            <button type="button" class="cc-model-reset" onclick={resetChatModel}>
              {codeChat.model.reset}
            </button>
          {/if}
        </span>
      </div>
    {/if}

    <form class="cc-input-row" onsubmit={(e) => { e.preventDefault(); submit(); }}>
      <span class="cc-prompt" aria-hidden="true">❯</span>
      <input
        class="cc-input"
        bind:value={question}
        bind:this={inputEl}
        placeholder={codeChat.input.placeholder}
        aria-label={codeChat.navLabel}
        disabled={!ready}
        autocomplete="off"
        spellcheck="false"
      />
      <button type="submit" class="cc-ask" disabled={!canAsk || !question.trim()}>{codeChat.input.ask}</button>
    </form>
    <div class="cc-foot">
      <span class="cc-hint" aria-live="polite">
        {#if !ready && !loadError}{codeChat.input.hintLoading}{:else if chat.keyState === 'missing'}{codeChat.input.hintNoKey}{/if}
      </span>
      <span class="cc-fineprint">{codeChat.input.fine}</span>
    </div>
  </div>
</div>
