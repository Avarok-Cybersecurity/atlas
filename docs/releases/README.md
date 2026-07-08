# Release notes & shipped-image record

One file per shipped image so `:latest`'s provenance is answerable at a glance.
When `/atlas-release publish` promotes a verified image (see
[`.claude/skills/atlas-release`](../../.claude/skills/atlas-release/SKILL.md)),
record it here as `docs/releases/<git-sha>.md` with:

- the **git SHA** the image was built from (also stamped into the image as
  `org.opencontainers.image.revision` — `docker inspect` it),
- the moving tags it received (`:latest` / `:dev` / `:nightly` / `:<semver>`),
- the **serve-matrix verdict** (`tests/gate_results.py` PASS + the results table),
- notable engine changes since the previous shipped SHA.

This closes the "is `:latest` the merged code?" gap: the answer is the newest file
here, cross-checked against the image's revision label.
