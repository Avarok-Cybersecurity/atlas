# blog.atlasinference.io — rollout record

Append-only. One entry per wave: what was found, the evidence, what changed, and
what the negative control proved. Newest entries at the bottom.

## The gate

`/workspace/atlas-blog/.gate.sh` (untracked; mirrors what `.github/workflows/site.yml`
runs in CI) — for each of `site/` and `blog/`:

1. `bun test src/lib` — unit tests
2. `bun x --bun vite build` — the SvelteKit build, with `ATLAS_RECIPES_ROOT`
   pointed at a local `atlas-recipes` checkout
3. the chevron-field contrast check (`.contrast-check.mjs`), which re-derives the
   field's luminance budget against the ground it is actually painted on

Nothing is committed unless the gate is green.

## The baseline (before any change on this branch)

Branch `feat/blog-subdomain`, cut from `origin/main` @ `29591e0a2`.

| leg | result |
|---|---|
| `site` unit tests | **483 pass / 0 fail** across 37 files |
| `site` build | **ok**, 9.25 s, adapter-static wrote `build/` |
| `blog` | absent — not yet created |
| contrast check | not yet written |

Green. Any red after this point belongs to this work until proven otherwise.

## Wave 1 — reconnaissance and the two things the brief got wrong about the repo

**Found.** Two premises needed correcting before any code was written.

1. *The working tree was stale.* `/workspace/atlas` sits on
   `fix/ssm-rollback-hardening` and its `site/` is the old **light** "warm
   workshop" system (`--bg: #f4f0e8`, copper `#b5622f`). `origin/main` has since
   moved to the **deep violet workstation** system (`--bg: #14111f`, accent
   `#9271f4`) and already names the four chevron colours as first-class tokens —
   `--ch-violet #BE9DF8`, `--ch-cyan #49C3DB`, `--ch-green #12B981`,
   `--ch-gold #EFB338`. Those are *byte-identical* to the four chevron constants
   in the supplied scaffold. All work happens in a clean worktree at
   `/workspace/atlas-blog` cut from `origin/main`, never in `/workspace/atlas`.

2. *Consequently the brief's ambiguity dissolves.* "Use the look and feel of the
   main website" and "the same color scheme" are already 90% satisfied by the
   scaffold; the only real divergence is the ground (`#0F1216` scaffold vs
   `#14111f` main). The blog adopts **main's** tokens; both properties gain the
   WebGL field.

**Evidence.** `git show origin/main:site/src/app.css`, lines 9–60.

**Consequence for the field.** `FIELD-NOTES.md` derives the field's amplitude
from a contrast budget solved against `#0F1216`. `#14111f` is a *different*
ground (violet-tinted, and slightly lighter), so those measured ratios —
13.41:1 / 10.35:1 / 4.56:1 — **do not transfer**. They have to be re-derived, and
the tightest of them (metadata gray) is only 0.06 above AA on the original
ground, so this is not a rounding concern. That re-derivation is the
`.contrast-check.mjs` gate leg.

## Wave 2 — the origin vhost

**Changed.** `blog/deploy/nginx/blog.atlasinference.io.conf` is the SSOT for the
vhost; it is installed on the avarok origin as
`/etc/nginx/sites-available/00-blog.atlasinference.io.conf` and symlinked into
`sites-enabled`. Docroot `/var/www/blog.atlasinference.io/html`, owned
`ubuntu:ubuntu` — the same user the marketing-site deploy already rsyncs as
(established below), so the blog needs no new SSH identity.

**How the deploy identity was established** — the values live in GitHub
environment secrets and cannot be read back, so they were derived from the
origin. `sshd`'s journal shows `Accepted publickey for ubuntu from 52.162.9.240`
(Azure) at `Aug 29 00:56:38`, and `/var/www/atlasinference.io/build` has mtime
`Aug 29 00:56` and owner `ubuntu`. So `DEPLOY_SSH_USER=ubuntu` and
`DEPLOY_PATH=/var/www/atlasinference.io/build`.

**Proved** — origin-direct (`--resolve … 127.0.0.1`) and again through
Cloudflare on the public name:

| check | result |
|---|---|
| `GET /` through Cloudflare | **200**, serves the docroot |
| `/` cache-control | `public, max-age=300` |
| `/_app/immutable/probe.js` | `public, max-age=31536000, immutable` |
| security headers on **HTML** | `nosniff` + `SAMEORIGIN` + `strict-origin-when-cross-origin` all present |
| `/404` (extensionless clean URL) | 200 via `$uri.html` |
| `/.env` | 404 |
| `http://…/x` | 301 → `https://blog.atlasinference.io/x` |
| `nginx -t` | ok |

**The negative control, and what it caught.** The claim being tested is that
computing Cache-Control through a `map` — instead of in a `location` — is what
keeps the security headers on HTML. The control is the docs vhost, which still
has the location-level form:

```
$ curl -D- https://docs.atlasinference.io/index.html
cache-control: public, max-age=300
# …and nothing else. No x-frame-options, no nosniff, no referrer-policy.
```

The control is **red**, as required: the defect reproduces on a live vhost, so
the check is measuring something real. The blog vhost, same request shape,
returns all four headers.

> **Open, and deliberately not acted on:** `docs.atlasinference.io` is serving
> every HTML document with no `X-Content-Type-Options`, `X-Frame-Options` or
> `Referrer-Policy`. The fix is the same `map` used here. It is a different
> property from the one this work was authorised for, so it is reported rather
> than applied.

**One thing that looked wrong and was not.** The first `curl` after
`systemctl reload nginx` came back with a 3724-byte body, `last-modified
Aug 6`, and an HSTS header this vhost never sets — i.e. some *other* server
block answered. Cause: `reload` is asynchronous, and an old worker served that
connection before the new configuration was live. Re-running after the reload
settled gives the table above every time. Recorded because "a header set I did
not write" is exactly the shape of a real misconfiguration, and it would have
been easy to go debugging server_name matching for an hour.
