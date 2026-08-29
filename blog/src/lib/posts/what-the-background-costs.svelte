<script module>
  export const meta = {
    title: 'What the background costs',
    dek: 'The animated field behind this page is 2.52 KB over the wire. The library that draws the identical pixels is 151 KB. Here is the measurement, the 2 KB that were hiding in a string literal, and the contrast budget that decided how bright it is allowed to be.',
    date: '2026-08-28',
    tag: 'design',
    author: 'thomas-braun',
    readingMinutes: 11
  };
</script>

<script>
  import H2 from '$lib/components/H2.svelte';
  import Callout from '$lib/components/Callout.svelte';
  import Code from '$lib/components/Code.svelte';
  import Table from '$lib/components/Table.svelte';
</script>

<p>
  The background of this page is a field of the Atlas mark drifting left to right, in depth, behind
  the masthead. The chevron geometry is the real one — arm 320×280, stroke 76, gap 280, straight out
  of the brand guidelines — drawn as a signed distance function, so the motif in the background and
  the logo in the header are the same shape at any scale.
</p>

<p>
  It renders as <strong>one fullscreen triangle with one fragment shader</strong>. No scene graph, no
  camera, no materials, no loaders. This post is about the two decisions that took the longest: which
  renderer draws it, and how bright it is allowed to be.
</p>

<H2 id="the-three-js-question" index={0}>The three.js question, answered with numbers</H2>

<p>
  Both implementations were built, before this page existed, as a straight comparison. They run the
  <em>identical</em> shader and were verified pixel-for-pixel: <strong>0 of 960,000 pixels
  differ</strong>. So everything below is cost, not appearance. The figures in this table are from
  that comparison; the ones later in the post are measured on what this page actually ships.
</p>

<Table caption="Measured on the same scene, same shader, same output. The GPU does identical work in both columns.">
  <thead>
    <tr><th></th><th>raw WebGL2</th><th>three.js r185</th></tr>
  </thead>
  <tbody>
    <tr><td>Over the wire (brotli)</td><td class="num"><span class="win">2.5 KB</span></td><td class="num">151.1 KB</td></tr>
    <tr><td>Parsed / compiled</td><td class="num"><span class="win">5.7 KB</span></td><td class="num">733 KB</td></tr>
    <tr><td>Main thread per frame</td><td class="num"><span class="win">&lt; 0.01 ms</span></td><td class="num">0.30 ms</td></tr>
    <tr><td>GPU fill cost</td><td class="num">identical</td><td class="num">identical</td></tr>
    <tr><td>Dependencies</td><td class="num">none</td><td class="num">one, ~23 MB installed</td></tr>
  </tbody>
</Table>

<p>
  <strong>three.js costs 61× the payload to draw the same pixels.</strong> Three things make that gap
  structural rather than a tuning problem.
</p>

<p>
  <strong>One.</strong> <code>WebGLRenderer</code> alone is 129.7 KB gzipped. Adding the other twelve
  classes this scene uses costs about 1 KB. Tree-shaking takes three.js from 182 KB to ~124 KB — a 32%
  cut, not a 90% one. There is no import discipline that gets under ~120 KB while using the renderer.
</p>

<p>
  <strong>Two.</strong> Roughly 139 KB of GLSL string literals survive tree-shaking, even though this
  scene never touches a built-in material. <code>WebGLProgram</code> resolves
  <code>ShaderLib[material.type]</code> by string key at runtime, so no bundler can prove any entry
  unreachable.
</p>

<p>
  <strong>Three.</strong> There is no UMD build any more. From a CDN you fetch
  <code>three.module.min.js</code> <em>and</em> <code>three.core.min.js</code> — 151 KB,
  un-tree-shaken, because a CDN cannot shake anything.
</p>

<p>
  Note also what the library does <em>not</em> do here. DPR clamping, resize, visibility handling,
  reduced motion, intersection pausing, context-loss recovery and teardown are hand-written in both
  files. What three.js contributes to this scene is <code>renderer.render()</code>.
</p>

<Callout label="Not an argument against three.js" tone="engine">
  Use it when you have a real scene — meshes, lights, GLTF, camera motion, picking. For one quad it
  is a shader-runner you pay 151 KB for. The reference implementations agree: Stripe's gradient, the
  most-copied WebGL background on the web, is hand-rolled raw WebGL at 18.2 KB transferred; Vercel's
  is 11.6 KB. Neither uses three.js for it.
</Callout>

<p>
  Threlte is not a way out either. <code>@threlte/core</code> opens with
  <code>import * as THREE from 'three'</code> and resolves <code>&lt;T.Mesh&gt;</code> by string
  lookup at runtime, so it <em>structurally</em> cannot tree-shake three. Measured:
  <strong>+62.7 KB gzipped over vanilla</strong>, a 50% penalty, to render one mesh.
</p>

<H2 id="two-kb-in-a-string" index={1}>Two kilobytes hiding in a string literal</H2>

<p>
  When this field was wired into a real build, it measured <strong>4.25 KB brotli</strong> — not the
  2.5 KB the comparison had reported. Nothing had been added to the renderer that could account for
  1.75 KB.
</p>

<p>
  The shader is imported with <code>?raw</code>, so it ships inside a JavaScript string literal. And
  a minifier does not touch string contents — it cannot, since it has no idea what the string means.
  So every byte of the commentary explaining the SDF geometry, the falloff mask and the luma clamp
  was being sent to every visitor:
</p>

<Table caption="The same field, bundled alone and minified, with and without the shader's comments.">
  <thead>
    <tr><th></th><th>minified</th><th>gzip</th><th>brotli</th></tr>
  </thead>
  <tbody>
    <tr><td>as written</td><td class="num">11,192</td><td class="num">4.91 KB</td><td class="num">4.25 KB</td></tr>
    <tr><td>comments stripped at build</td><td class="num">6,393</td><td class="num">2.79 KB</td><td class="num"><span class="win">2.52 KB</span></td></tr>
  </tbody>
</Table>

<p>
  The shader source is 7,254 bytes and 2,481 of those are code. Comments were <strong>66% of the
  file and 43% of the shipped component</strong> — most of the gap between the number the
  comparison reported and the number this repo was actually serving.
</p>

<p>
  The fix is a fifteen-line Vite plugin that strips comments from
  <code>.glsl?raw</code> imports at build time, so they stay in the file where they are useful and
  never leave the repository. It refuses to emit a shader that lost its <code>#version</code>
  directive or its <code>main()</code>, because a shader that fails to compile does not throw — the
  field just quietly falls back to the CSS dot layer, which is precisely the kind of silent
  degradation this component is built to avoid.
</p>

<Callout label="Worth checking on your own build" tone="technical">
  Any asset imported as a string — GLSL, SQL, a worker source, an HTML template — is invisible to
  your minifier. If you have ever documented a shader properly, you have shipped that documentation.
</Callout>

<H2 id="brightness-is-derived" index={2}>Brightness is derived, not chosen</H2>

<p>
  A field behind live text is an accessibility decision wearing a visual-design costume.
  Every unit of luminance it adds to the ground comes off the contrast ratio of the text
  sitting on it, and the weakest text on this page — the metadata gray <code>#82868F</code> —
  starts at <strong>5.15:1</strong> on the bare ground. WCAG AA for body text is 4.5:1. That
  leaves less room than it sounds like.
</p>

<p>
  <strong>Each hue is normalised to unit luma before it is added.</strong> The four chevron
  colours differ in luminance by about 1.6× — gold is far brighter than green — so without
  this the worst case depends on which colour happens to land under a line of text.
  Normalising makes the ceiling one number, and costs nothing visually: chroma is what reads
  at these levels, not luma.
</p>

<Code lang="glsl" name="chevron-field.glsl" code={`// Normalize each hue to unit luma before adding it, so the worst-case
// background luminance stops depending on which colour landed there.
hue /= max(dot(hue, vec3(0.2126, 0.7152, 0.0722)), 1e-3);

col += hue * l.x * dim;`} />

<p>
  <strong>Amplitude is then set from the resulting budget</strong>, by a check that runs in CI.
  And the check does not sample. The original field notes measured 14 time samples across the
  viewport and reported the worst frame they saw — a reasonable method, and one that cannot see
  the case where all three depth layers land on the same pixel at the sweep's peak. That case is
  rare. It is also exactly the case that would put a line of metadata gray under AA.
</p>

<p>
  So the gate computes the analytic bound instead: the most luminance the shader is
  <em>capable</em> of adding to any pixel, whether or not that configuration was ever sampled.
  Which is where it got interesting, because the first version of that bound produced a
  background nobody could see.
</p>

<H2 id="the-invisible-field" index={3}>The bound that deleted the background</H2>

<p>
  Three layers, each contributing up to its depth weight, sum to 2.28 layers' worth of luma on a
  pixel where all three overlap. Solving the contrast budget against <em>that</em> gives a
  density of 0.44. The field shipped at that value, and the screenshot looked wrong — so it got
  measured rather than argued about:
</p>

<Code lang="console" code={`# brightest pixel of the field, in a text-free gutter, against ground #0F1216
brightest gutter pixel (21, 24, 28)   # ground is (15, 18, 22)
                                      # +6, +6, +6 — and neutral, not tinted`} />

<p>
  Six of 255, on every channel equally. At that amplitude the chevron hues round to grey in
  8-bit and the field is not a background, it is dither. The bound was correct and the result
  was useless, which is the signature of solving the wrong problem: the amplitude of the
  <em>whole</em> field was being set by its rarest accident.
</p>

<p>
  The fix belongs in the shader, not in the density. Clamp the accumulated luma to one layer's
  worth, and the worst pixel is bounded directly instead of by division:
</p>

<Code lang="glsl" name="chevron-field.glsl" code={`// Each hue is already unit-luma, so this is a uniform scale on the colour
// vector: hue is preserved exactly, and the only pixels it touches are the
// ones where layers overlap.
float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
col /= max(1.0, lum);`} />

<p>
  Two lines, and the bound drops from 2.28 layers to 1.00 — which buys back 2.28× the amplitude
  for the same guarantee. AA is now reached at density <strong>1.0119</strong>; the field ships
  at 0.85, and the brightest gutter pixel measures <code>(23, 25, 33)</code> — visibly violet
  rather than grey.
</p>

<Table caption="Worst ground the field can paint at density 0.85, per chevron hue. Every cell clears WCAG AA; the tightest is 4.61:1.">
  <thead>
    <tr><th>text token</th><th>bare ground</th><th>violet</th><th>cyan</th><th>green</th><th>gold</th></tr>
  </thead>
  <tbody>
    <tr><td>headings <code>#E4E7EC</code></td><td class="num">15.15</td><td class="num">13.66</td><td class="num">13.62</td><td class="num">13.58</td><td class="num">13.67</td></tr>
    <tr><td>body <code>#C9CCD4</code></td><td class="num">11.69</td><td class="num">10.54</td><td class="num">10.51</td><td class="num">10.48</td><td class="num">10.55</td></tr>
    <tr><td>metadata <code>#82868F</code></td><td class="num">5.15</td><td class="num"><span class="win">4.64</span></td><td class="num"><span class="win">4.63</span></td><td class="num"><span class="win">4.61</span></td><td class="num"><span class="win">4.65</span></td></tr>
  </tbody>
</Table>

<Callout label="The control that matters" tone="verified">
  A contrast check that always passes is worse than no contrast check, because it reports safety.
  This one was watched failing — at density 1.0 without the clamp it reports 3.56:1, and the
  build goes red — before the shipped value was chosen. A gate nobody has seen fail is a gate
  nobody has tested.
</Callout>

<H2 id="what-it-handles" index={0}>The parts that are not the shader</H2>

<p>
  A background canvas that outlives a hundred page views has to handle rather more than drawing.
  Roughly 80% of the runtime is lifecycle:
</p>

<Table>
  <thead><tr><th>Concern</th><th>Behaviour</th></tr></thead>
  <tbody>
    <tr><td>No WebGL2</td><td>Returns <code>null</code>; the CSS dot field stays visible. No error, no blank area.</td></tr>
    <tr><td><code>prefers-reduced-motion</code></td><td>Renders one frozen frame and cancels the loop. Not removed — freezing keeps the design and drops to zero ongoing cost. Re-checked live, since the OS setting can change mid-session.</td></tr>
    <tr><td>Hidden tab</td><td>rAF already stops; the handler resets the clock so it does not jump on return.</td></tr>
    <tr><td>Covered but visible</td><td><code>IntersectionObserver</code> stops the loop. This is the saving rAF does <em>not</em> give you.</td></tr>
    <tr><td>High-DPI</td><td><code>min(devicePixelRatio, 1.5)</code>. Fill cost is quadratic in DPR, so 3 → 1.5 is a 4× GPU saving on phones for no visible difference in a soft field.</td></tr>
    <tr><td>Context loss</td><td><code>preventDefault()</code> on <code>webglcontextlost</code> — required, or it never restores — and a full rebuild on restore.</td></tr>
    <tr><td>Battery</td><td><code>powerPreference: 'low-power'</code>. Never wake a discrete GPU for decoration.</td></tr>
  </tbody>
</Table>

<p>Two failure modes are worth naming, because both are silent.</p>

<p>
  <strong>Skipping <code>destroy()</code> on client-side navigation</strong> leaves the rAF loop
  rendering to a detached canvas — GPU work for pixels nobody sees, compounding on every route change.
</p>

<p>
  <strong>Skipping <code>loseContext()</code></strong> leaks a GL context per mount. Browsers cap live
  contexts at 16, and on the seventeenth the <em>oldest</em> is killed — which may be an unrelated
  canvas elsewhere on the site, which then goes black with no error thrown in your code.
</p>

<H2 id="one-more-trap" index={1}>One layout trap</H2>

<p>
  The <code>&lt;canvas&gt;</code> must not sit under a transformed ancestor. A <code>transform</code>,
  <code>filter</code>, <code>perspective</code>, <code>will-change</code> or
  <code>contain: paint</code> on any ancestor makes that element the containing block for
  fixed-position descendants, and the background silently starts scrolling with the content. Keep it a
  direct child of the layout root. And do not add <code>will-change</code> to the canvas: one driving
  a GL context is already composited, so it buys nothing and costs memory.
</p>

<p>
  The whole thing — runtime, shader, Svelte wrapper — is about 550 lines, and roughly 80% of
  that is lifecycle rather than drawing. The measurement that decided its brightness is another
  140, and it runs on every build.
</p>
