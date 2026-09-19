# Avarok site

Start with @AGENTS.md. It is the contributor guide for this directory, for people and for
agents, and it points to `SITE-GUIDE.md`, the generated map of every page, button, link,
address and asset on the site. Read both before changing anything.

Tooling, in one line: this is SvelteKit on Vite, run with Bun. Use `bun install`,
`bun run <script>`, `bun test` and `bun x <tool>`, not npm, yarn or pnpm. Do not replace Vite
or add a server: the site is prerendered with `adapter-static` and served as files.
