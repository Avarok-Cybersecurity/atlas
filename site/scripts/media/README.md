# Media scripts

The pipeline that produces every clip, poster and social card on the marketing
site. Node builtins plus the Playwright chromium the e2e suite installs, and
ffmpeg on PATH (with libx264, libvpx-vp9 and libwebp). Nothing leaves the
machine.

| Script | Does | Reads | Writes |
| --- | --- | --- | --- |
| `record.mjs` | Opens each clip's page in headless chromium and records it | `build/`, `src/lib/content/media.js` | `.media-raw/*.webm`, `.media-raw/manifest.json` |
| `encode.mjs` | Trims and encodes the raw recordings | `.media-raw/` | `static/media/<slot>.{mp4,webm,webp}` |
| `install.mjs` | Puts any approved file into a slot | the file you give it | `static/media/`, `src/lib/content/art.json` |
| `reel.mjs` | Assembles the product film | `media-brief/reel.json`, takes, raw recordings, `src/lib/content/` | `static/media/reel.{mp4,webm,webp}` |
| `og.mjs` | Renders the social card | `build/`, `assets/brand/`, `src/lib/content/` | `static/og-image.png` |
| `brief.mjs` | Renders the prompt sheet | `media-brief/shots.json` | `media-brief/PROMPTS.md` |
| `serve.mjs` | Static server over `build/` used by the two above | | |

## The usual run

```sh
bun x --bun vite build
bun run media          # record.mjs, then encode.mjs
bun run og
bun run reel
```

`.media-raw/` is ignored by git. `static/media/` is committed.

## How a clip knows what to record

By its name in `src/lib/content/media.js`:

- `broll-<scene>` opens `/broll?scene=<scene>` from this repository's build.
  The scenes are in `src/lib/broll/scenes.js`.
- `console-<scene>` opens `/console?scene=<scene>` on the product mockup, which
  is a private project outside this repository. Pass its address with
  `--console-origin`. Without it those clips are skipped and their committed
  encodes are kept. `media-brief/README.md` has the details.

Both pages stamp `window.__sceneStart` on their first frame and set
`window.__sceneDone` when the scene ends. The recorder stores the two times, and
the encoder uses them to cut the page load off the front and to end the clip
where the scene ends. A b-roll clip is cut to exactly one loop period.

## Who filled a slot

`static/media/provenance.json` says how each slot got its files: `recorded` by
this pipeline, `installed` by hand with `install.mjs`, or `assembled` by
`reel.mjs`. It exists because the generated footage replaced the procedural
loops in the `broll-*` slots while the procedural recordings were still sitting
in `.media-raw/`, and the next plain `bun run media` would have put the
placeholders back without a word. `encode.mjs` now leaves an installed slot
alone and says so. `--force` overrides it.

Only the three loops in `src/lib/broll/scenes.js` have a page to record. The
other `broll-*` slots hold generated footage and the recorder skips them.

## Why the loops loop

Every motion in `src/lib/broll/scenes.js` is a function of phase with integer
harmonics, so the frame at the end of the loop is the frame at the start.
`src/lib/broll/broll.test.js` draws both frames into a recording context and
compares them. If you add motion to a scene and that test fails, the clip would
have jumped at the seam.

## Budgets

The front page loads four posters and, as they scroll near, four videos.
`encode.mjs` prints every file's size and flags anything over 1.5 MB. The hero
poster is the largest paint on the front page, keep it under 100 KB.

`VideoClip.svelte` lists the mp4 before the webm on purpose. For flat interface
footage H.264 came out smaller than VP9 at the same legibility.

## Git LFS

The repository tracks `*.mp4` and `*.webm` with LFS, and the CI checkout does
not fetch LFS objects, so an LFS pointer would ship as a broken video.
`.gitattributes` exempts `site/static/media/*` for that reason. Keep new site
clips in that directory.

## The real thread on the company page

`capture-thread.mjs` opens the public pull request the company's story starts with and writes
"The exchange" as four WebP files under `static/media/about/`, in both themes, with the clock
set to UTC so the dates in the picture match the timeline beside it. Run it again if GitHub
changes how a thread looks. It needs `ffmpeg`, like everything else here.

## What a page costs while it sits there

`../perf/cpu.mjs` (`bun run perf:cpu -- <origin> <path>`) measures idle CPU the way Chrome's
task manager does, renderer and GPU together, and then again with one suspect switched off at
a time. The budget is 5% of one core. It exists because an animated gradient once cost 38% and
nothing in CI could see it.
