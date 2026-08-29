<script module>
  export const meta = {
    title: 'The nginx directive that silently deletes your security headers',
    dek: 'add_header does not accumulate across contexts. One Cache-Control rule in a location block stripped nosniff, X-Frame-Options and Referrer-Policy from every HTML document a live site served — with no warning from nginx -t.',
    date: '2026-08-29',
    tag: 'engineering',
    author: 'thomas-braun',
    readingMinutes: 6
  };
</script>

<script>
  import H2 from '$lib/components/H2.svelte';
  import Callout from '$lib/components/Callout.svelte';
  import Code from '$lib/components/Code.svelte';
</script>

<p>
  While standing up the vhost for this blog, the obvious move was to copy the one next door — the
  config already serving our documentation site, which has the same shape: static files, Cloudflare in
  front, a self-signed origin certificate. It had been running for months. <code>nginx -t</code>
  passed. Nothing had ever complained.
</p>

<p>
  It was serving every HTML document with no security headers at all.
</p>

<H2 id="the-config" index={0}>The config</H2>

<p>Reduced to the part that matters, it looked like this — and it looks fine:</p>

<Code lang="nginx" name="00-docs.example.conf" code={`server {
    root /var/www/docs/html;

    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header X-Frame-Options "SAMEORIGIN" always;

    location / {
        try_files $uri $uri.html $uri/ $uri/index.html =404;
    }

    # Caching: HTML is short-lived; hashed assets can last forever.
    location ~* \\.html$            { add_header Cache-Control "public, max-age=300"; }
    location ~* \\.(css|js|woff2)$  { add_header Cache-Control "public, max-age=31536000, immutable"; }
}`} />

<p>
  Three security headers at server level, so they apply to everything. Then a couple of small
  location blocks that add caching rules per file type. Two independent concerns, expressed
  separately. That is the reasonable reading, and it is wrong.
</p>

<H2 id="the-rule" index={1}>What add_header actually does</H2>

<p>
  From the <code>ngx_http_headers_module</code> documentation, in a sentence most people have read and
  almost nobody has internalised:
</p>

<blockquote>
  <p>
    These directives are inherited from the previous configuration level if and only if there are no
    <code>add_header</code> directives defined on the current level.
  </p>
</blockquote>

<p>
  Not merged. Not appended. <em>Inherited if and only if the current level declares none of its own.</em>
  A single <code>add_header</code> inside a <code>location</code> discards the entire inherited set.
</p>

<p>
  So <code>location ~* \.html$</code>, whose only job was to set a cache lifetime, quietly deleted
  <code>nosniff</code>, <code>X-Frame-Options</code> and <code>Referrer-Policy</code> — from HTML
  documents specifically. The exact response class where those headers do something. CSS and JS lost
  them too, where they matter less. Images and fonts, matched by no location with an
  <code>add_header</code>, kept all three. The site therefore looked <em>partly</em> protected to any
  spot check that happened to request an asset.
</p>

<Callout label="The always flag does not save you" tone="notice">
  <code>always</code> controls whether a header is emitted on error responses as well as successful
  ones. It has nothing to do with inheritance. All three headers here carried <code>always</code>, and
  all three were still discarded.
</Callout>

<H2 id="proving-it" index={2}>Proving it, rather than believing it</H2>

<p>
  This is worth reproducing yourself before you go changing configs, because the failure is invisible
  from inside the config file. One request each, straight at the origin:
</p>

<Code lang="console" code={`$ curl -sSI https://docs.example.com/index.html | grep -iE 'cache-control|x-frame|nosniff|referrer'
cache-control: public, max-age=300
                      # ...and nothing else.

$ curl -sSI https://docs.example.com/logo.png | grep -iE 'cache-control|x-frame|nosniff|referrer'
x-content-type-options: nosniff
referrer-policy: strict-origin-when-cross-origin
x-frame-options: SAMEORIGIN`} />

<p>
  The document gets caching and no protection. The image gets protection and no caching. Neither gets
  both, and nothing in <code>nginx -t</code>, the error log, or the access log says so.
</p>

<H2 id="the-fix" index={3}>The fix: compute the value, do not move the directive</H2>

<p>
  The usual advice is to repeat the security headers inside every location that sets one. That works
  and it does not last: it is three lines duplicated across four blocks, and the fifth block someone
  adds next year will not have them. The defect regenerates.
</p>

<p>
  The better shape is to have exactly one <code>add_header</code> set in the whole vhost, and to make
  the <em>value</em> of Cache-Control vary instead of its location. A <code>map</code> does that:
</p>

<Code lang="nginx" name="00-blog.atlasinference.io.conf" code={`# Longest-lived first: map regexes are evaluated in declaration order
# and the first hit wins.
map $uri $blog_cache_control {
    "~^/_app/immutable/"        "public, max-age=31536000, immutable";
    "~^/service-worker\\.js$"    "no-cache";
    "~*\\.(?:woff2?|ttf|otf)$"   "public, max-age=31536000, immutable";
    "~*\\.(?:png|jpe?g|svg|webp|avif)$"  "public, max-age=2592000";
    default                     "public, max-age=300";
}

server {
    # One header set for the whole vhost. Nothing below overrides it,
    # because nothing below declares an add_header at all.
    add_header Cache-Control $blog_cache_control always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header X-Frame-Options "SAMEORIGIN" always;

    location / {
        try_files $uri $uri.html $uri/index.html $uri/ =404;
    }
}`} />

<p>
  Now every response carries all four headers, and the caching policy is one ordered list you can read
  top to bottom instead of a set of location blocks whose precedence you have to work out. The comment
  at the top of the real file says the thing worth saying: <em>do not add an add_header to any location
  below without repeating these.</em>
</p>

<H2 id="the-general-shape" index={4}>The general shape of this bug</H2>

<p>
  What makes this class expensive is not the rule — the rule is one sentence in the manual. It is that
  the config <em>reads</em> as declarative and composable when it is neither, and that the failure
  surfaces nowhere: not at parse time, not in a log, not in a health check. The only instrument that
  catches it is a request, and the only reason to make that request is already suspecting the answer.
</p>

<p>
  Which is the argument for pinning it. The header set for this vhost is asserted by a test that
  fetches a document, an immutable asset and a 404, and fails if any of the four headers is missing
  from any of them. It was written by first reproducing the defect on the live config next door and
  watching the assertion go red — because a header test that has never failed is not evidence that the
  headers are there.
</p>
