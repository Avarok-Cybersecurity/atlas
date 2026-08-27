<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Writing-only sheet. Root layout already loads the design system. This
  // keeps blog.css off the homepage and /control.
  import '../../styles/blog.css';
  let { children } = $props();
</script>

{@render children()}
