<script>
  // One printed entry in the chat log. User turns are copper prompt lines,
  // assistant turns are receipt cards with a sources footer. The markdown
  // renderer escapes first and builds every tag itself (chat/markdown.js), so
  // its output is safe for {@html} by construction.
  import { renderMarkdown } from '../chat/markdown.js';
  import { codeChat } from '$lib/data.js';

  let { message, sourcesOpen = true } = $props();
</script>

{#if message.role === 'user'}
  <p class="cm-user"><span class="cm-prompt" aria-hidden="true">❯</span>{message.text}</p>
{:else}
  <article class="cm-card receipt-print">
    <header class="cm-head">
      <span class="cm-title">{codeChat.answerTag}</span>
      {#if message.sources?.length}
        <span class="cm-count">
          {message.sources.length}
          {message.sources.length === 1 ? codeChat.sourcesOne : codeChat.sourcesMany}
        </span>
      {/if}
    </header>
    <div class="cm-body">{@html renderMarkdown(message.text)}</div>
    {#if message.sources?.length}
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
