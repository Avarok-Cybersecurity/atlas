# Ambient chevron field — what it is, what it costs, how to ship it

The background is a field of the Atlas mark drifting left to right, in depth,
behind the masthead. The chevron geometry is the real one — arm 320×280,
stroke 76, gap 280 straight out of `BRAND-GUIDELINES.md` — drawn as a signed
distance function, so the motif in the background and the logo in the header
are the same shape at any scale.

It renders as **one fullscreen triangle with one fragment shader**.

---

## The three.js question, answered with numbers

Both implementations are included. They run the **identical shader** and were
verified pixel-for-pixel: **0 of 960,000 pixels differ**. So everything below
is cost, not appearance.

| | raw WebGL2 | three.js r185 |
|---|---|---|
| Over the wire (brotli) | **2.5 KB** | **151.1 KB** |
| Parsed / compiled | **5.7 KB** | **733 KB** |
| Main thread per frame | **< 0.01 ms** | **0.30 ms** |
| GPU fill cost | identical | identical |
| Dependencies | none | one, versioned, ~23 MB installed |

**three.js costs 61× the payload to draw the same pixels.**

Three things make that gap unavoidable rather than a tuning problem:

1. **`WebGLRenderer` alone is 129.7 KB gzipped.** Adding the other twelve
   classes this scene uses costs about 1 KB. Tree-shaking takes three.js from
   182 KB to ~124 KB — a 32% cut, not a 90% one. There is no import discipline
   that gets under ~120 KB while using the renderer.
2. **~139 KB of GLSL string literals survive tree-shaking**, even though this
   scene never touches a built-in material. `WebGLProgram` resolves
   `ShaderLib[material.type]` by string key at runtime, so no bundler can prove
   any entry unreachable.
3. **There is no UMD build any more.** From a CDN you fetch
   `three.module.min.js` *and* `three.core.min.js` — 151 KB, un-tree-shaken,
   because a CDN cannot shake anything.

And note what three.js does *not* do for you here. DPR clamping, resize,
visibility handling, reduced motion, intersection pausing, context-loss
recovery and teardown are hand-written in both files. What the library
contributes to this scene is `renderer.render()`.

**Recommendation: ship the raw version.** It is also what the reference
implementations do — Stripe's gradient, the most-copied WebGL background on the
web, is hand-rolled raw WebGL at 18.2 KB transferred; Vercel's is 11.6 KB.
Neither uses three.js for it. On a blog whose thesis is measured performance, a
500 KB three.js chunk in the network panel is a brand problem before it is a
performance one.

Use three.js when you get a real scene — meshes, lights, GLTF, camera motion,
picking. For one quad it is a shader-runner you pay 151 KB for.

**Don't use Threlte here either.** `@threlte/core` opens with
`import * as THREE from 'three'` and resolves `<T.Mesh>` by string lookup at
runtime, so it *structurally* cannot tree-shake three. Measured: **+62.7 KB
gzipped over vanilla**, a 50% penalty, to render one mesh.

---

## Why it is safe to put under text

Amplitude is not a taste value here; it is derived. The weakest text on the
page is the `#82868F` metadata gray at **5.15:1** on the bare `#0F1216` ground.
Holding it at or above **4.5:1** caps how much luminance the field may add.

The first pass at a comfortable-looking brightness measured **3.97:1** — below
AA. Two changes fixed it properly:

- **Each hue is normalized to unit luma before it is added.** The four chevron
  colors differ in luminance by ~1.6× (gold is far brighter than green), so
  without this the worst case depended on which color happened to land under a
  line of text. Normalizing makes the ceiling one number we control, and costs
  nothing visually — chroma is what reads at these levels, not luma.
- **Amplitude set from the resulting budget**, not by eye.

Measured across 14 time samples over the whole viewport:

| | on bare ground | worst case over the field |
|---|---|---|
| Headings `#E4E7EC` | 15.15:1 | **13.41:1** |
| Body `#C9CCD4` | 11.69:1 | **10.35:1** |
| Metadata `#82868F` | 5.15:1 | **4.56:1** |

All clear AA. The field is also masked to nothing before it reaches the 672px
reading column, both vertically and horizontally — the same rule the CSS dot
field follows, so the two agree when both are on.

**If you raise `density`, re-run the contrast check.** It is the constraint the
amplitude was solved against.

---

## What it handles

| Concern | Behavior |
|---|---|
| No WebGL2 | Returns `null`; the CSS dot field stays visible. No error, no blank area. |
| `prefers-reduced-motion` | Renders **one frozen frame**, cancels the loop. Not removed — freezing keeps the design and drops to zero ongoing cost. Re-checked live, since the OS setting can change mid-session. |
| Hidden tab | rAF already stops; the handler resets the clock so it doesn't jump on return. |
| Covered but visible | `IntersectionObserver` stops the loop. This is the saving rAF does *not* give you. |
| High-DPI | `min(devicePixelRatio, 1.5)`. Fill cost is quadratic in DPR, so 3 → 1.5 is a **4× GPU saving** on phones for no visible difference in a soft field. |
| Context loss | `preventDefault()` on `webglcontextlost` (required, or it never restores) and a full rebuild on restore. |
| Teardown | rAF cancelled first, then GL objects, then `loseContext()`. |
| Battery | `powerPreference: 'low-power'` — never wakes a discrete GPU for decoration. |
| Text selection | `pointer-events: none`; verified the page is still clickable through the canvas. |

The two failure modes worth naming, because they are silent:

- **Skipping `destroy()` on client-side navigation** leaves the rAF loop
  rendering to a detached canvas — GPU work for pixels nobody sees, compounding
  on every route change.
- **Skipping `loseContext()`** leaks a GL context per mount. Browsers cap live
  contexts at **16**, and on the 17th the *oldest* is killed — which may be an
  unrelated canvas elsewhere on the site, which then goes black with no error
  thrown in your code.

---

## Wiring it into SvelteKit

Verified against Svelte 5.57 / SvelteKit 2.70 / Vite 8.

```
src/lib/gl/chevron-field.js      the runtime
src/lib/gl/chevron-field.glsl    the shader
src/lib/components/ChevronField.svelte
```

```svelte
<!-- src/routes/+layout.svelte -->
<script>
  import ChevronField from '$lib/components/ChevronField.svelte';
  let { children } = $props();
</script>

<ChevronField />
{@render children()}
```

Four things in the component that are load-bearing:

- **No `browser` guard inside `$effect`.** Verified by compiling the component
  for both targets: effect bodies are not emitted into the SSR output at all.
  A guard there is dead code. Do *not* wrap the `<canvas>` in `{#if browser}`
  either — rendering it server-side reserves its box and avoids a pop-in on
  hydration.
- **Boot inside `requestIdleCallback`.** Shader compile and link is a
  *synchronous* main-thread stall — single-digit ms on desktop, but 50–300 ms
  on low-end mobile. Landing that on a user interaction is one catastrophic INP
  sample. Building during idle, after hydration, avoids it. The idle handle is
  cancelled in teardown, because it is a suspension point: a fast unmount
  otherwise leaves a renderer nothing holds a reference to.
- **The `<canvas>` must not sit under a transformed ancestor.** `transform`,
  `filter`, `perspective`, `will-change` or `contain: paint` on any ancestor
  makes it the containing block for fixed-position descendants, and the
  background silently starts scrolling with the content. Keep it a direct child
  of the layout root.
- **Don't add `will-change` to the canvas.** A canvas driving a GL context is
  already composited; it buys nothing and costs memory.

### If you ship the three.js version anyway

Split on a wrapper module with named static imports —
`await import('$lib/gl/chevron-field-three.js')` — **never**
`await import('three')`. A namespace dynamic import yields an object whose
property accesses can't be statically resolved, so the bundler retains every
export. Measured difference in a real Vite 8 build: **55 KB gzipped**.

Also set `renderer.outputColorSpace = LinearSRGBColorSpace` and parse hex
colors by hand rather than with `new THREE.Color()`. With ColorManagement on
(the default since r152) `Color` returns Linear-sRGB components, which
silently darkens every uniform by the transfer function. That single line was
the difference between the two renderers disagreeing and matching exactly.

---

## Tuning

Everything visual is in the shader, in one place:

| What | Where |
|---|---|
| Overall strength | `amt` at the bottom of `main()` — **re-run the contrast check if raised** |
| How crowded | `if (r1 > 0.30)` in `layer()` — higher is denser |
| Mark size | `sz` in `layer()`. Keep `2.28 * sz + jitter < 0.5` or marks clip at cell seams |
| Drift speed | `speed` in the layer loop |
| Where it fades | `vert` and `horiz` in `main()` |
| Sweep period | `u_time * 0.042` → one pass every ~24s |

Set `density={0}` on the component to disable it without removing it.
