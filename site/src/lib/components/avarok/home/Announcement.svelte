<script>
  import { announcement } from '$lib/content/home.js';
  import { fill } from '$lib/content/live.js';
</script>

<div class="av-announce">
  <div class="av-announce-in">
    <span class="av-chip av-chip-violet">{announcement.tag}</span>
    <span>{fill(announcement.text)}</span>
    <a href={announcement.href}>{announcement.cta} →</a>
  </div>
</div>
