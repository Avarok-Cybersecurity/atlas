// =============================================================================
// The media manifest. Every video and poster the marketing pages reference is
// declared here, so media.test.js can prove each file exists and the
// scripts/media pipeline knows what to render.
//
// Files live under static/media, three per clip: <name>.mp4, <name>.webm and
// the poster <name>.webp. Where a clip comes from is in its name:
//
//   console-<scene>  a screen recording of the Avarok Console product mockup
//                    running a scripted scene on demo data. The mockup is a
//                    private project and is NOT in this repository; only its
//                    recordings are. media-brief/README.md says where it lives
//                    and how to re-record.
//   broll-<scene>    a procedural loop drawn from the brand palette by
//                    src/lib/broll/scenes.js and recorded from /broll.
//
// A slot can be replaced by dropping new files of the same name, which is how
// a generated clip from the prompt pack in media-brief/ gets onto the site
// once it is approved: scripts/media/install.mjs does the encode.
// =============================================================================

import installed from './art.json' with { type: 'json' };

const clip = (name, alt, { loop = true, w = 1280, h = 720, posterAt, maxBytes } = {}) => ({
  name,
  mp4: `/media/${name}.mp4`,
  webm: `/media/${name}.webm`,
  poster: `/media/${name}.webp`,
  alt,
  loop,
  width: w,
  height: h,
  // Where in the clip the poster frame is taken, 0 to 1. Unset, encode.mjs
  // picks: late for a console scene, early for a loop.
  posterAt,
  // A size budget of its own, for the one clip that is a film and not a loop.
  maxBytes
});

export const media = {
  // The hero. The console "ask" scene. Its poster is the largest paint on the
  // front page, so app.html preloads it by this name.
  hero: clip('console-ask', 'The Avarok Console answering a question from an air gapped fleet, with tokens per second and cost per request on screen.'),
  // The product tour on the front page and the demo page.
  tour: {
    console: clip('console-ask', 'Picking a model and a compute grid, then streaming an answer with live throughput and cost.'),
    queue: clip('console-queue', 'Bringing up encrypted links to offsite grids, then queueing a large project across them and watching it run.'),
    fleet: clip('console-fleet', 'The fleet view. Nodes, kernel versions, a canary rollout climbing from five to one hundred percent.'),
    economics: clip('console-economics', 'The economics view. Cost per million tokens, chargeback by business unit and the payback clock.'),
    governance: clip('console-governance', 'The governance view. Policies, provenance for one response and the audit log.')
  },
  // The product film. Assembled by scripts/media/reel.mjs from the cut in
  // media-brief/reel.json: the generated footage, the recorded product clips
  // and captions taken from the site's own copy. Plays once, with controls.
  reel: clip('reel', 'The Avarok product film. The problem, a machine being deployed with one command, then the console: asking a fleet, queueing a project over encrypted links, the fleet, the ledger, and governance.', { loop: false, maxBytes: 9_000_000 }),
  // Product footage that is not a tour tab. The deployment clip is the answer
  // to "how does it get deployed": one command, and a machine becomes a node.
  product: {
    deploy: clip('console-deploy', 'Enrolling a machine with one command. The signature is verified, kernels are chosen for the silicon, the recipe warms, the readiness gate passes and the node starts serving.')
  },
  // Ambient loops. The three named in scenes.js can be re-recorded from
  // /broll. The rest are generated from media-brief/takes and have no
  // procedural scene; broll.test.js allows that.
  broll: {
    field: clip('broll-field', 'The Avarok chevron field, drifting.'),
    tokens: clip('broll-tokens', 'A river of tokens moving through a grid of GPUs.'),
    rack: clip('broll-rack', 'Rack lights in a dark datacenter aisle, pulsing with load.'),
    enclave: clip('broll-enclave', 'A sealed server room behind glass, its lights pulsing.'),
    desk: clip('broll-desk-box', 'A small desktop compute box with a breathing lavender status light.'),
    hall: clip('broll-hall', 'A glide over a vast data hall with light travelling along the cable trays.'),
    power: clip('broll-power', 'Indicator lamps blinking in sequence along datacenter switchgear.')
  },
  // Stills from the prompt pack, keyed by slot name. art.json is written by
  // scripts/media/install.mjs when an image is approved, and is empty until
  // then. A page hero asks artFor(its path) and renders no picture when there
  // is none, so putting art on a page is a data change and nothing else.
  art: Object.fromEntries(
    Object.entries(installed).map(([slot, a]) => [slot, { slot, src: `/media/art/${slot}.webp`, alt: a.alt, width: a.width, height: a.height, pages: a.pages ?? [] }])
  )
};

export const allClips = [media.hero, media.reel, ...Object.values(media.tour), ...Object.values(media.product), ...Object.values(media.broll)];

// Which clip belongs to which page. Product footage where the page is about a
// part of the product, ambient footage where it is about an idea. A still
// installed for the page (art.json) keeps the hero and the clip moves to a
// band under it (PageClip.svelte). With no still the clip is the hero. With
// neither the page is text only.
const heroClips = {
  '/platform': 'queue',
  '/platform/control': 'fleet',
  '/platform/economics': 'economics',
  '/platform/security': 'governance',
  '/platform/deployment': 'deploy',
  '/platform/engine': 'tokens',
  '/solutions': 'hall'
};
const clipByKey = { ...media.tour, ...media.product, ...media.broll };

/** The clip that opens a page, or null. */
export const heroClipFor = (path) => clipByKey[heroClips[path]] ?? null;

/** The installed still for a page, or null. */
export const artFor = (path) => Object.values(media.art).find((a) => a.pages.includes(path)) ?? null;
