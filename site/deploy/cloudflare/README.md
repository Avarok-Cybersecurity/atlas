# Cloudflare Pages hosting

atlasinference.io and blog.atlasinference.io are served by Cloudflare Pages.
There is no origin server in the request path, which is the point: the previous
host went down and took both properties with it.

## Projects

| Project | Serves | pages.dev |
| --- | --- | --- |
| `atlas-site` | `atlasinference.io`, `www.atlasinference.io` | `atlas-site-80h.pages.dev` |
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

## www -> apex is NOT in this repo, and cannot be

The nginx `if ($host = www...)` block has no Pages equivalent. `_redirects`
supports path rules (verified: a path-only rule redirects correctly) but the
documented absolute-URL form does **not** match on these projects (verified:
`https://www.atlasinference.io/* ...` never fires). A path rule cannot be used
because both hostnames are the same project, so it would bounce the apex too.

It lives as a zone-level Redirect Rule on `atlasinference.io`:

    expression: (http.host eq "www.atlasinference.io")
    action:     redirect, 301
    target:     concat("https://atlasinference.io", http.request.uri.path)
    preserve query string: yes

Creating it over the API needs a token with **Zone -> Dynamic Redirect -> Edit**
(neither of the tokens used for the migration had it; both could only list
rulesets). Until it exists, `www` serves the site directly on a 200 rather than
redirecting — functional, but duplicate content for crawlers.
