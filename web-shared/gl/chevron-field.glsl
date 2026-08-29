#version 300 es
precision highp float;

/* ============================================================
   Atlas chevron field
   ------------------------------------------------------------
   A depth-stacked field of the Atlas chevron, drifting slowly
   left-to-right — the direction the mark itself points, and the
   direction a token moves through the engine.

   The chevron geometry is the real one, normalized off
   BRAND-GUIDELINES.md "Mark geometry":
       arm    320 across, 280 up   -> half-width 320/280 = 1.1429
       stroke 76                   -> radius   38/280 = 0.1357
       gap    280                  -> 1.0 in these units
   Drawn as an exact SDF of the two-segment polyline with round
   caps, so it is the same shape as the vector artwork at any
   scale — not an approximation of it.
   ============================================================ */

uniform vec2  u_res;        // backing-store resolution, px
uniform float u_time;       // seconds
uniform float u_scroll;     // 0..1, page scroll
uniform float u_density;    // 0..1, user-facing intensity
uniform vec3  u_c1;         // #BE9DF8
uniform vec3  u_c2;         // #49C3DB
uniform vec3  u_c3u;        // #12B981
uniform vec3  u_c3l;        // #EFB338
uniform vec3  u_ground;     // #0F1216

out vec4 fragColor;

const float ARM    = 1.1428571;  // 320/280
const float RADIUS = 0.1357143;  // 38/280
const float GAP    = 1.0;        // 280/280 — apex-to-apex

// distance from p to segment ab
float sdSeg(vec2 p, vec2 a, vec2 b){
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

// One chevron, apex at the origin pointing +x, half-height 1.
// Round caps and a round join fall out of the segment SDF for free —
// exactly as the artwork's stroke-linecap / stroke-linejoin do.
float sdChevron(vec2 p){
  return min(sdSeg(p, vec2(-ARM,  1.0), vec2(0.0, 0.0)),
             sdSeg(p, vec2(-ARM, -1.0), vec2(0.0, 0.0))) - RADIUS;
}

// The MARK: three chevrons, apex-to-apex gap 280 = 1.0 in these units.
// This is the artwork's own geometry, so the motif in the field and the
// logo in the header are the same shape.
float sdMark(vec2 p){
  return min(min(sdChevron(p + vec2(GAP, 0.0)),
                 sdChevron(p)),
                 sdChevron(p - vec2(GAP, 0.0)));
}

// cheap hash for per-cell jitter
float hash(vec2 c){
  return fract(sin(dot(c, vec2(127.1, 311.7))) * 43758.5453);
}

// One parallax layer: a jittered grid of chevrons drifting +x.
// Returns coverage in .x and a per-cell random in .y (picks the hue).
vec2 layer(vec2 uv, float scale, float speed, float seed){
  vec2 p = uv * scale;
  p.x -= u_time * speed;

  vec2 cell = floor(p);
  vec2 f    = fract(p) - 0.5;

  float r1 = hash(cell + seed);
  float r2 = hash(cell + seed + 7.3);
  float r3 = hash(cell + seed + 19.1);

  // Thin the field out. Most cells stay empty, so the eye reads
  // scattered marks rather than wallpaper.
  if (r1 > 0.30) return vec2(0.0);

  // Modest jitter. Too much size variance and the big ones start
  // competing with the logo instead of receding behind the page.
  vec2  off = (vec2(r2, r3) - 0.5) * 0.40;
  float sz  = 0.075 + r2 * 0.035;

  float d = sdMark((f - off) / sz) * sz;

  float aa = fwidth(d) * 1.1 + 1e-5;
  return vec2(1.0 - smoothstep(-aa, aa, d), r3);
}

void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5 * u_res) / u_res.y;

  // Parallax: deeper layers move less. Scroll pushes the field down
  // slightly, so it reads as sitting behind the page.
  float sc = u_scroll * 0.35;

  vec3 col = vec3(0.0);

  // three depth layers, back to front — mirroring the three chevrons
  for (int i = 0; i < 3; i++){
    float fi    = float(i);
    float scale = 2.6 + fi * 1.9;
    float speed = 0.020 + fi * 0.016;
    float dim   = 0.52 + fi * 0.24;         // small/far marks read crisper

    vec2 l = layer(uv + vec2(0.0, sc * (1.0 + fi * 0.4)), scale, speed, fi * 37.0);

    // Hue by the per-cell random, weighted toward the brand order.
    vec3 hue = l.y < 0.40 ? u_c1
             : l.y < 0.74 ? u_c2
             : l.y < 0.88 ? u_c3u
                          : u_c3l;

    /* Normalize each hue to unit luma before adding it.
       The four chevron colors differ in luminance by ~1.6x (gold is far
       brighter than green), so without this the worst-case background
       luminance depends on which color happens to land under a line of
       text. Normalizing makes the ceiling a single number we control,
       and keeps hue identity for free — chroma is what reads, not luma. */
    hue /= max(dot(hue, vec3(0.2126, 0.7152, 0.0722)), 1e-3);

    col += hue * l.x * dim;
  }

  /* Cap the accumulated luma at ONE layer's worth.

     Three depth layers can, in principle, put a mark on the same pixel at the
     same instant. Unclamped that triples the luminance the field may add, and
     since the amplitude has to be safe for the worst pixel rather than the
     usual one, the whole field then has to be divided by three to stay inside
     the contrast budget — paying for a rare accident by making the field
     invisible everywhere else. Measured: at the amplitude that bound allows,
     the brightest pixel the field produced was 5/255 above the ground.

     Clamping bounds the worst pixel directly instead, so the amplitude is set
     by what the field normally does. Each hue is already normalised to unit
     luma, so this is a uniform scale on the colour vector — hue is preserved
     exactly, and the only pixels it touches are the ones where layers overlap.

     `.contrast-check.mjs` reads this cap. Removing it without changing that
     file would leave the gate describing a field that no longer exists. */
  float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col /= max(1.0, lum);

  /* Falloff — the whole reason this is safe to put under text.
     The field is strongest at the top of the viewport and is gone
     well before the reading column begins. Same rule the CSS dot
     field follows, so the two agree when both are on. */
  vec2 ndc = gl_FragCoord.xy / u_res;

  // gl_FragCoord.y is 0 at the BOTTOM, so "strongest at the top of the
  // viewport" is ndc.y directly. The field is gone by mid-screen, which
  // is where the reading column starts.
  float vert = smoothstep(0.30, 0.98, ndc.y);

  // Pull it away from the horizontal centre too — the prose column is
  // 672px in the middle of the page and must sit on flat ground.
  float centre = 1.0 - abs(ndc.x - 0.5) * 2.0;
  float horiz  = 1.0 - 0.82 * smoothstep(0.22, 0.92, centre);

  float mask = vert * horiz;

  // A slow bright sweep, left to right: one pass every ~24s.
  // Amplitude is tiny — it should register as "something moved"
  // in peripheral vision and nothing more.
  float sweep = exp(-pow((ndc.x - fract(u_time * 0.042)) * 3.0, 2.0));

  /* Amplitude is not a taste value — it is derived. The weakest text on
     the page is #82868F at 5.15:1 on bare ground; holding it at or above
     4.5:1 caps the background luminance the field may add. These two
     numbers are that cap, and CONTRAST.md shows the working. */
  float amt = u_density * mask * (0.030 + sweep * 0.020);

  fragColor = vec4(u_ground + col * amt, 1.0);
}
