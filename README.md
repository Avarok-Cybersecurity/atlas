# bot-cards

Generated certification images, one per pull request, written here by
`certification-bot.yml` when a PR merges.

This branch carries no source. It exists because GitHub proxies images through
camo and caches them **by URL** — a stable filename would keep serving the first
render forever — so every certificate gets a content-hashed name and therefore
its own permanent, immutable URL.

Nothing here is an input to anything. Deleting a file breaks a link in an old
comment and nothing else.
