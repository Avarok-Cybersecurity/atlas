# references/pipeline.md — build → image → publish, exact contract

Everything `/atlas-release build`, `image`, and `publish` do, grounded in the
real files. Nothing here is aspirational unless flagged **[TODO]**.

## 1. Build

### The binary
```bash
# The shippable multi-model sweep (host-native; the "38s incremental" path):
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*' \
  cargo build --release -p spark-server
# -> target/release/spark   (PTX for every kernels/gb10/<model>/<quant>/ embedded)
```

- **Target selection** is three env vars read in `crates/atlas-kernels/build.rs`
  (defaults if unset: `gb10` / `qwen3-next-80b-a3b` / `nvfp4`). `*` = "every
  subdir with a `MODEL.toml`". `build.rs` compiles each `.cu` to PTX with
  `nvcc --ptx -arch=sm_121f` (arch from `kernels/gb10/HARDWARE.toml`) and emits
  `OUT_DIR/target_ptx.rs`; a content-addressed dedup cache turns ~1100 naive
  nvcc calls into ~170 unique compiles + copies, run across cores.
- **Weights are never compiled in.** `MODEL.toml` contributes sampling presets +
  behavior + the `(model_type, hidden_size)` match table, not weights.
- **PCND footgun:** unset env → single-model `qwen3-next-80b-a3b` binary, *not* a
  sweep. `/atlas-release build` always passes all three explicitly.
- **Incrementality:** `build.rs` emits `cargo:rerun-if-changed` on every consumed
  `.cu`/`.toml` and `rerun-if-env-changed` on the three `ATLAS_TARGET_*`. Touch
  only Rust → build.rs doesn't rerun, PTX is reused, ~38s relink. Touch a kernel
  or flip a target env → PTX recompiles for the affected targets.
- **Fast Rust-only loop (no nvcc, no kernels):** set `ATLAS_SKIP_BUILD=1
  CUDARC_CUDA_VERSION=13000` (exactly what CI uses) — `atlas-kernels/build.rs` and
  `spark-runtime/build.rs` emit a PTX stub, so `cargo check`/clippy run without
  nvcc. This is a hygiene loop, **not** a runnable binary. **[TODO]** `scripts/check.sh`
  currently exports `SKIP_ATLAS_BUILD` (no `ATLAS_` prefix) + `CUDARC_CUDA_VERSION=12000`;
  that name is honored only by `spark-storage/build.rs`, so it does *not* stub the
  kernel crates. Prefer the `ATLAS_SKIP_BUILD` form until check.sh is fixed.

### The container split (`docker/gb10/Dockerfile`, multi-stage)
- **builder:** `nvidia/cuda:13.0.0-devel-ubuntu24.04` (digest-pinned). Installs
  rust + `cmake clang libclang-dev` (vendored xgrammar-rs), sets
  `ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=* ATLAS_TARGET_QUANT=*`, runs
  `cargo build --release -p spark-server`.
- **runtime:** `nvidia/cuda:13.0.0-runtime-ubuntu24.04` (digest-pinned). Installs
  `libnccl2` (**fail-fast asserts NCCL ≥ 2.28** for symmetric-memory
  `ncclMemAlloc`) + RDMA userspace (`libibverbs1 librdmacm1 ibverbs-providers`).
  Then **`COPY --from=builder /build/target/release/spark /usr/local/bin/spark`**
  + `jinja-templates/` + LICENSE/README. `ENTRYPOINT ["spark"]`, `EXPOSE 8888`.
- **"Mount the result into a runtime container" is a multi-stage `COPY`, not a
  volume mount.** The binary is baked at build time. No nvcc in the runtime image
  — the host driver re-JITs the embedded PTX → SASS for the running SM (ADR-0009).
- **Toolchain hazard [TODO]:** the sweep Dockerfile sets `RUSTUP_TOOLCHAIN=stable`
  to override the `rust-toolchain.toml` pin (`1.93.1`); the single-model
  Dockerfiles don't. Two image families can build under different rustc. Pin both
  to the same toolchain for reproducibility.

### Model images
- `docker/gb10/Dockerfile` = the **all-models** image (`avarok/atlas-gb10:*`).
- `docker/gb10/<model>/nvfp4/Dockerfile` = per-model slim images (10 exist). Each
  hardcodes `ATLAS_TARGET_MODEL=<slug> ATLAS_TARGET_QUANT=nvfp4`. Note only a
  `nvfp4/` subdir exists per model — the nvfp4 bundle carries the FP8 + BF16 code
  paths, so FP8 models (`qwen3.6-27b`, `qwen3.6-35b-a3b`) live under `nvfp4/` too.
- **COPY-set divergence [TODO]:** the sweep Dockerfile COPYs `vendor/` (required —
  `Cargo.toml` has `[patch.crates-io] cudarc = { path = "vendor/cudarc" }`) and
  `jinja-templates/`; the single-model `qwen3.5-35b-a3b` Dockerfile COPYs neither.
  Verify single-model images still build before relying on them.

## 2. Image — tag to the source

```bash
SHA=$(git rev-parse --short=7 HEAD)
docker build -f docker/gb10/Dockerfile --build-arg ATLAS_GIT_SHA="$SHA" \
  -t avarok/atlas-gb10:"$SHA" .
```

**Always tag with the git SHA first.** It is the only identifier that answers
"is `:latest` the merged code?" The moving tags (`:dev`, `:nightly`, `:latest`)
are *pointers* moved onto a verified `:sha` in Stage 4 — never built independently.

### Provenance — now stamped into the image
`scripts/release.sh` and `/atlas-release image` pass `--build-arg ATLAS_GIT_SHA=...`.
The runtime stage of `docker/gb10/Dockerfile` now declares the matching `ARG` +
`LABEL org.opencontainers.image.revision` (added by this pass), so the SHA is
recorded in the image. Answer the staleness question directly:

```bash
docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' avarok/atlas-gb10:latest
```

(The per-model single-model Dockerfiles don't yet carry the `ARG` — add the same
two lines there if you ship a per-model image and want its provenance.)

### Offline distribution (`docker save`)
```bash
docker save avarok/atlas-gb10:"$SHA" | zstd -T0 -o atlas-gb10-"$SHA".tar.zst
# receiver:  zstd -dc atlas-gb10-<sha>.tar.zst | docker load
```
Use this when there's no registry access, or to hand a verified image to a user
directly (RoCE-copy dgx1→dgx2, or attach to a GitHub Release alongside the distro
tarballs).

## 3. Publish — human-gated (never automated)

The pipeline stops here and **prepares** the push. The maintainer runs it. All
moving tags (`:dev`, `:nightly`, `:latest`) advance together onto the same
verified `:sha` — they are never built independently.

```bash
for T in nightly dev latest; do docker tag avarok/atlas-gb10:"$SHA" avarok/atlas-gb10:"$T"; done
for T in "$SHA" nightly dev latest; do docker push avarok/atlas-gb10:"$T"; done
```

Preconditions the skill enforces before printing this:
1. Stage 2 verify **passed** for exactly this `$SHA` (not a different tree).
2. The build came from the intended ref — **build from `main` tip**, not whatever
   branch happens to be checked out on the build host (a real past footgun; that
   checkout is often a feature branch). Verify: `git rev-parse --short=7 origin/main == $SHA`.
3. Docker is logged in as `avarok` on the pushing host.

After push, **record the shipped SHA** so `:latest`'s provenance is durable — append
to `docs/releases/` or rely on the image label from the ARG fix above.

## Closing the main→:latest gap (why this pipeline exists)

Today, three things are all manual and none are chained:
1. Nothing rebuilds the image on merge to `main`.
2. Nothing verifies an image before it's tagged `:latest`.
3. The push is a human running `docker build && docker push` by hand from an
   arbitrary branch.

`release.yml` automates **only** the relocatable **tarball** distros on a `v*`
tag (via `scripts/package-distro.sh` on self-hosted `[gb10]`/`[strix]` runners) —
it never touches the Docker image. So the image path has zero automation.

**The minimum viable close** (respecting no-auto-push): a self-hosted-runner
workflow on `push: [main]` that runs Stage 1 (build) + Stage 2 (verify serve
matrix) + Stage 3 (build + tag `:sha` + `docker save` as an artifact), and
**stops** — surfacing a green "ready to publish `<sha>`" the maintainer promotes
with one `docker push`. That turns weeks of drift into a one-command publish of a
pre-verified image, while keeping the decision to ship to users in human hands.
See `references/pr-and-ci.md` for the self-hosted-runner shape (the release-distros
job already proves self-hosted GB10 runners work).
