/**
 * Atlas chevron field — raw WebGL2, no dependencies.
 *
 * Renders one fullscreen triangle with one fragment shader. That is the
 * entire scene, which is why it needs no scene graph, no camera, no
 * material system, and therefore no 3D library.
 *
 * Handles, because a background canvas that outlives a hundred page
 * views has to: devicePixelRatio clamping, resize, reduced motion,
 * tab visibility, intersection, WebGL context loss, and teardown.
 *
 *   const field = createChevronField(canvas, FRAG_SRC, {
 *     ground: '#14111f', c1: '…', c2: '…', c3u: '…', c3l: '…',
 *   });
 *   field.setScroll(0.3);
 *   field.destroy();
 */

const VERT = `#version 300 es
// One triangle that covers the viewport. Cheaper than two: no diagonal
// seam, so the GPU never rasterizes the shared edge twice.
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Only the two numbers that are properties of the RENDERER have defaults here.
// The five colours deliberately do not: they are design tokens, they live in
// web-shared/atlas-tokens.css, and a copy of them in this file is a second
// source of truth that goes stale silently — the canvas paints its own ground,
// so a drifted value shows up as a seam between the canvas and the page rather
// than as an error. The caller reads them from the cascade and passes them in.
const DEFAULTS = {
  /* Contrast-derived, not a taste value, and NOT the 1.0 that FIELD-NOTES.md
     ships. That figure was solved against ground #0F1216 by sampling 14 frames;
     this field is painted on #14111f, and the gate here bounds the field
     ANALYTICALLY — the case where all three depth layers land on one pixel at
     the sweep's peak, which sampling can miss. Under that bound the tightest
     token, --t3 #8a83af, is at AA exactly at density 0.5109. 0.45 leaves
     margin and measures 4.60:1. Raising this fails `.contrast-check.mjs`. */
  density: 0.45,
  maxDpr: 1.5,          // fill cost is quadratic in DPR; 3 -> 1.5 is a 4x saving
};

const REQUIRED_COLORS = ['ground', 'c1', 'c2', 'c3u', 'c3l'];

const HEX = /^#[0-9a-fA-F]{6}$/;

const rgb = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
};

export function createChevronField(canvas, fragmentSource, opts = {}) {
  const o = { ...DEFAULTS, ...opts };

  // Fail loudly rather than painting a field in the wrong colours over the
  // wrong ground. Returning null is the same contract as "no WebGL2": the
  // caller keeps its CSS fallback and the page is still correct.
  for (const k of REQUIRED_COLORS) {
    if (!HEX.test(o[k] ?? '')) {
      console.error(`[chevron-field] ${k} must be a #rrggbb colour, got ${JSON.stringify(o[k])}`);
      return null;
    }
  }

  const gl = canvas.getContext('webgl2', {
    alpha: false,              // opaque: the compositor skips blending this layer
    antialias: false,          // the only primitive's edges are offscreen
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false,
    powerPreference: 'low-power', // never wake a discrete GPU for decoration
    desynchronized: true,
  });
  if (!gl) return null;         // caller falls back to the CSS dot field

  const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');

  let prog = null, uniforms = {}, vao = null;
  let raf = 0, t0 = performance.now(), clock = 0;
  let scroll = 0, density = o.density;
  let visible = true, onScreen = true, dead = false;

  /* ---------- build (also the context-restore path) ---------- */

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('[chevron-field]', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  function build() {
    const vs = compile(gl.VERTEX_SHADER, VERT);
    const fs = compile(gl.FRAGMENT_SHADER, fragmentSource);
    if (!vs || !fs) return false;

    prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    // shaders are reference-counted by the program; drop our handles now
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('[chevron-field]', gl.getProgramInfoLog(prog));
      return false;
    }

    uniforms = {};
    for (const n of ['u_res','u_time','u_scroll','u_density','u_c1','u_c2','u_c3u','u_c3l','u_ground'])
      uniforms[n] = gl.getUniformLocation(prog, n);

    // gl_VertexID needs no buffer, but WebGL2 still wants a bound VAO
    vao = gl.createVertexArray();

    gl.useProgram(prog);
    gl.uniform3fv(uniforms.u_c1,     rgb(o.c1));
    gl.uniform3fv(uniforms.u_c2,     rgb(o.c2));
    gl.uniform3fv(uniforms.u_c3u,    rgb(o.c3u));
    gl.uniform3fv(uniforms.u_c3l,    rgb(o.c3l));
    gl.uniform3fv(uniforms.u_ground, rgb(o.ground));
    return true;
  }

  /* ---------- size ---------- */

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, o.maxDpr);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width === w && canvas.height === h) return;
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
  }

  const ro = new ResizeObserver(() => { resize(); if (!running()) draw(); });

  /* ---------- draw ---------- */

  function draw() {
    if (dead || !prog) return;
    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.uniform2f(uniforms.u_res, canvas.width, canvas.height);
    gl.uniform1f(uniforms.u_time, clock);
    gl.uniform1f(uniforms.u_scroll, scroll);
    gl.uniform1f(uniforms.u_density, density);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  const reduced = () => motionQuery.matches;
  const running = () => !dead && visible && onScreen && !reduced();

  function frame(now) {
    clock = (now - t0) / 1000;
    draw();
    raf = requestAnimationFrame(frame);
  }

  function start() {
    if (raf || !running()) return;
    t0 = performance.now() - clock * 1000;  // resume where we left off
    raf = requestAnimationFrame(frame);
  }

  function stop() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }

  function sync() {
    if (running()) start();
    else { stop(); draw(); }   // reduced motion / hidden: one frozen frame, not a blank canvas
  }

  /* ---------- lifecycle listeners ---------- */

  const onVisibility = () => { visible = !document.hidden; sync(); };
  const onMotion = () => { clock = 0; sync(); };

  // A fixed background can be fully covered while the tab is still visible.
  // rAF only stops for hidden *tabs*, so this is the saving rAF won't give us.
  const io = new IntersectionObserver(
    ([e]) => { onScreen = e.isIntersecting; sync(); },
    { threshold: 0 }
  );

  const onLost = (e) => { e.preventDefault(); stop(); };  // preventDefault or it never restores
  const onRestored = () => { if (dead) return; if (build()) { resize(); sync(); } };

  /* ---------- init ---------- */

  if (!build()) return null;
  resize();
  ro.observe(canvas);
  io.observe(canvas);
  document.addEventListener('visibilitychange', onVisibility);
  motionQuery.addEventListener('change', onMotion);
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);
  sync();

  return {
    setScroll(v) { scroll = v; if (!running()) draw(); },
    setDensity(v) { density = v; if (!running()) draw(); },
    get running() { return !!raf; },

    /** Draw a single frame at an explicit time. Used for the frozen
     *  reduced-motion frame, and to make the output deterministic in tests. */
    renderFrame(t = 0) { stop(); clock = t; draw(); },

    destroy() {
      dead = true;
      stop();                                   // cancel the loop first
      ro.disconnect();
      io.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
      motionQuery.removeEventListener('change', onMotion);
      canvas.removeEventListener('webglcontextlost', onLost);
      canvas.removeEventListener('webglcontextrestored', onRestored);
      if (vao) gl.deleteVertexArray(vao);
      if (prog) gl.deleteProgram(prog);
      // Release the driver-side context. Without this, SPA navigation
      // marches toward the browser's 16-context cap and an unrelated
      // canvas somewhere else on the site goes black.
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      prog = vao = null;
    },
  };
}
