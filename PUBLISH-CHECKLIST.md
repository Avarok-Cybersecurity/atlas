# Publish checklist

Run through every box before tagging a public release. Internal-only.

## Pre-flight

- [ ] **Branch**: rename `master-rewrite` → `main`, push, set as GitHub default.
- [ ] **License headers**: `gh workflow run ci.yml` (license-header job) — all green.
- [ ] **CI**: every workflow in `.github/workflows/` is green on `main`
      (fmt, clippy, license, typos, file-size-cap, security/cargo-deny, docs).
- [ ] **mdBook**: `cd book && mdbook build` — clean build, no broken links.

## Workspace `Cargo.toml`

`Cargo.toml` ships with empty URL placeholders so `cargo publish` is
locked. Fill these before tag:

```toml
repository    = "https://github.com/<org>/atlas"
homepage      = "https://github.com/<org>/atlas"
documentation = "https://docs.rs/atlas"
```

- [ ] `<org>` resolved (the Atlas-publishing GitHub org / user).
- [ ] All three URLs filled.
- [ ] `cd crates/atlas-core && cargo publish --dry-run` succeeds (no
      missing-metadata errors).

## Docker image

The `README.md`, `QUICKSTART.md`, `docs/DEPLOYMENT.md`, and `book/` all
quickstart with:

```
docker pull avarok/atlas-gb10:latest
```

This must exist and be pullable when users hit publish. Order of ops:

- [ ] Confirm the `avarok` Docker Hub namespace owner matches the
      project's `security@avarok.net` SECURITY.md address.
- [ ] Build the production image: `docker build -f docker/gb10/Dockerfile -t avarok/atlas-gb10:v0.1.0 -t avarok/atlas-gb10:latest .`
- [ ] Sanity-test the image on a clean GB10: `docker run ... serve <model>`
      → `/v1/chat/completions` returns coherent text.
- [ ] `docker push avarok/atlas-gb10:v0.1.0 && docker push avarok/atlas-gb10:latest`

## CODEOWNERS placeholder

`.github/CODEOWNERS` uses `@<MAINTAINER>` placeholders.

- [ ] Replace with real GitHub usernames (or team slugs).

## Tag + release

- [ ] `git tag -a v0.1.0 -m "v0.1.0: pure-Rust CUDA inference engine for NVIDIA DGX Spark GB10"`
- [ ] `git push origin v0.1.0`
- [ ] Draft GitHub Release with the tag's commit summary; the docs
      already cover the feature surface, so the release notes can be
      short.

## Post-publish

- [ ] CLA workflow self-test: open a dummy PR from a non-CLA-signed
      account; confirm the bot comments correctly with the new
      `branch: 'main'` setting.
- [ ] crates.io publish (in dependency order): `atlas-core`,
      `atlas-kernels`, `atlas-{activation,norm,quant,embed,reduce}`,
      `spark-runtime`, `spark-comm`, `spark-storage`, `spark-model`,
      `spark-server`, `atlas-spark-bench`. Skip `cufile-sys` and the
      `xgrammar-rs` vendor — they aren't intended for crates.io.
- [ ] Watch the first week of issues for OSS-friction reports
      (build deps, doc gaps, examples not running) and patch fast.

## Deferred (opened as GitHub issues post-tag)

- [ ] `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) — paste from
      contributor-covenant.org.
- [ ] `CHANGELOG.md` (Keep-a-Changelog format) — first entry covers
      v0.1.0.
- [ ] TODO/FIXME triage — scan the ~12 remaining and convert to issues
      with labels.
