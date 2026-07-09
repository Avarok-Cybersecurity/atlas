<script>
  import { gifts } from '$lib/data.js';
</script>

<div class="gift">
  <div class="gift-inner">
    <svg class="gift-mark" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M20 7h-2.18A3 3 0 0 0 12 4.35 3 3 0 0 0 6.18 7H4a1 1 0 0 0-1 1v3a1 1 0 0 0 1 1v6a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-6a1 1 0 0 0 1-1V8a1 1 0 0 0-1-1zm-5-1a1 1 0 1 1-1 1 1 1 0 0 1 1-1zM9 6a1 1 0 1 1-1 1 1 1 0 0 1 1-1zm2 12H6v-5h5zm0-7H5V9h6zm7 7h-5v-5h5zm1-7h-6V9h6z"/>
    </svg>
    <span class="gift-line">{gifts.line}</span>
    <span class="gift-sub">{gifts.sub}</span>
  </div>
</div>
