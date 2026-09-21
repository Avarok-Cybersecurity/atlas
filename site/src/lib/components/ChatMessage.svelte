<script>
  // One printed entry in the chat log. User turns are copper prompt lines,
  // assistant turns are receipt cards with a sources footer. Passing `live`
  // (chat.stream from state.svelte.js) instead of `message` renders the
  // in-flight card: the reasoning trace streams first like a strip-chart
  // annotation, auto-collapses to a one-line disclosure when the answer
  // starts, and the answer streams below as progressive markdown. The
  // markdown renderer escapes first and builds every tag itself
  // (chat/markdown.js), so its output is safe for {@html} by construction.
  // The reasoning trace is interpolated as plain text.
  import { renderMarkdown } from '../chat/markdown.js';
  import { codeChat } from '$lib/data.js';

  let { message = null, live = null, sourcesOpen = true } = $props();

  const reasoning = $derived(live ? live.reasoningText : (message?.reasoning ?? ''));
  const answer = $derived(live ? live.answerText : (message?.text ?? ''));
  const reasoningMs = $derived(live ? live.reasoningMs : (message?.reasoningMs ?? 0));
  const secs = $derived((reasoningMs / 1000).toFixed(1));

  // The trace streams open until the first answer token, then collapses to
  // the disclosure line. A click after that pins the visitor's choice.
  const collapsible = $derived(answer.length > 0);
  let traceChoice = $state(null); // null = automatic
  const traceOpen = $derived(traceChoice ?? !collapsible);

  // Keep the streaming trace pinned to its newest line.
  let traceEl = $state(null);
  $effect(() => {
    void reasoning;
    if (live && traceEl) traceEl.scrollTop = traceEl.scrollHeight;
  });
</script>

{#if message?.role === 'user'}
  <p class="cm-user"><span class="cm-prompt" aria-hidden="true">❯</span>{message.text}</p>
{:else}
  <article
    class="cm-card receipt-print"
    data-streaming={live ? (collapsible ? 'writing' : 'thinking') : undefined}
  >
    <header class="cm-head">
      <span class="cm-title">{codeChat.answerTag}</span>
      {#if message?.sources?.length}
        <span class="cm-count">
          {message.sources.length}
          {message.sources.length === 1 ? codeChat.sourcesOne : codeChat.sourcesMany}
        </span>
      {/if}
    </header>
    {#if reasoning}
      <div class="cm-think" data-open={traceOpen}>
        {#if collapsible}
          <button
            type="button"
            class="cm-think-toggle"
            aria-expanded={traceOpen}
            onclick={() => (traceChoice = !traceOpen)}
          >
            <span class="cm-think-time">{codeChat.trace.reasonedPrefix} {secs}s</span>
            <span class="cm-think-sep" aria-hidden="true">·</span>
            <span class="cm-think-act">{traceOpen ? codeChat.trace.hide : codeChat.trace.show}</span>
          </button>
        {:else}
          <span class="slabel cm-think-label">{codeChat.trace.label}</span>
        {/if}
        <div class="cm-think-clip">
          <p class="cm-think-text" bind:this={traceEl}>{reasoning}</p>
        </div>
      </div>
    {/if}
    {#if answer || !live}
      <div class="cm-body">{@html renderMarkdown(answer)}</div>
    {/if}
    {#if message?.sources?.length}
      <details class="cm-sources" open={sourcesOpen}>
        <summary>{codeChat.sourcesHeading}</summary>
        {#each message.sources as s (s.n)}
          <a class="cm-src" href={s.url} target="_blank" rel="noopener">
            <span class="cm-src-n">[{s.n}]</span>
            <span class="cm-src-path">{s.path}</span>
            <span class="cm-src-lines">L{s.startLine}–{s.endLine}</span>
            <span class="cm-src-pct">{Math.round(s.relevancePct)}%</span>
          </a>
        {/each}
      </details>
    {/if}
  </article>
{/if}
