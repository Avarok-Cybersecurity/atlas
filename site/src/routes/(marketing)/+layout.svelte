<script>
  // The marketing routes. One stylesheet, scoped by the `.av` class the
  // PageShell wraps every page in. Nothing from app.css loads here.
  import '../../styles/avarok.css';
  let { children } = $props();
</script>

{@render children()}
