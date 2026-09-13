# Cloudflare Pages hosting

atlasinference.io and blog.atlasinference.io are served by Cloudflare Pages.
There is no origin server in the request path, which is the point: the previous
host went down and took both properties with it.

## Projects

| Project | Serves | pages.dev |
| --- | --- | --- |
| `atlas-site` | `atlasinference.io` | `atlas-site-80h.pages.dev` |
| `atlas-blog` | `blog.atlasinference.io` | `atlas-blog-3ja.pages.dev` |

Both are **Direct Upload** projects, not Pages' git integration. The build in
`.github/workflows/site.yml` needs an `atlas-recipes` checkout and a GitHub
token, and it carries four gates a Pages-native build would bypass — the
flagship-recipe check, the per-route `<title>` checks on both properties, and
the blog/site cross-link check. CI builds, CI uploads the gated output.

`--branch=main` on the upload is load-bearing: a deployment on any other branch
gets a preview URL and does not move the custom domain. That fails as "the
deploy went green and the site is stale".

## What replaced the nginx config

`../nginx/atlasinference.io.conf` is kept because the origin is still mirrored
to as a warm standby. On Pages the same behaviour comes from:

- **`static/_headers`** — the security headers and the cache policy. Read the
  note at the top of that file before editing it; Pages concatenates a
  re-declared header rather than replacing it, which silently cost the hashed
  assets their year-long cache once already.
- **`src/routes/404/+page.svelte`** — prerenders to `build/404.html`. Pages has
  no `try_files ... =404`; with no such file it answers every unmatched path
  with index.html and a **200**, so broken links return the front page and
  crawlers index unbounded soft-404s.

## www -> apex, and why `www` is deliberately NOT a custom domain

The nginx `if ($host = www...)` block has no in-repo Pages equivalent.
`_redirects` path rules work (verified: a path-only rule redirects correctly)
but the documented absolute-URL form does **not** match on these projects
(verified: `https://www.atlasinference.io/* ...` never fired). A path rule is
useless here anyway, since it would bounce the apex too.

So it is a zone-level Redirect Rule on `atlasinference.io`:

    expression: (http.host eq "www.atlasinference.io")
    action:     redirect, 301
    target:     concat("https://atlasinference.io", http.request.uri.path)
    preserve query string: yes

**`www.atlasinference.io` must stay OFF the Pages project for that rule to
run.** It was attached at first, and the rule — stored, enabled, correct
expression — did nothing: every request still returned 200 with the site.
A Pages custom domain is served by the Pages edge and never reaches the zone's
ruleset engine. The tell is in the response headers: the apex returns
`cf-cache-status: DYNAMIC` and `www` returned no `cf-cache-status` at all.
Detaching it made the redirect fire on the first request afterwards.

The `www` DNS record stays a proxied CNAME to the apex. It needs no origin and
no Pages binding, because a redirect rule is evaluated before Cloudflare
resolves one — the request never looks for something to serve.

Creating or editing the rule over the API needs a token with **Zone -> Dynamic
Redirect -> Edit**, on top of the Pages, DNS and Cache Purge permissions the
rest of this setup uses. A token holding only some of those fails with
`request is not authorized` on the ruleset write while still listing rulesets
happily, which reads like a bug and is not one.
