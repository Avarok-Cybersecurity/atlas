# Media brief

Everything that moves or glows on the marketing site, where it comes from, and
how to replace it. Written so that a person, another model or a team of agents
can pick up the media work without asking anyone what was meant.

## What is on the site today

| Slot | Where it shows | Source today |
| --- | --- | --- |
| `reel` | Demo page, linked from the front page hero | Assembled by `scripts/media/reel.mjs` from `reel.json` |
| `console-ask` | Front page hero, tour tab Ask | Recording of the product mockup |
| `console-queue` | Tour tab Queue, platform overview hero | Recording of the product mockup |
| `console-fleet` | Tour tab Fleet, Avarok Control hero | Recording of the product mockup |
| `console-economics` | Tour tab Economics, Avarok Economics hero | Recording of the product mockup |
| `console-governance` | Tour tab Governance, security page hero | Recording of the product mockup |
| `console-deploy` | Deployment page hero, the film | Recording of the product mockup |
| `broll-field` | Ambient, reel | Generated V03 from E03, 2026-09-18. See `takes/TAKES.md` |
| `broll-tokens` | Ambient, reel | Generated V02 from E02, 2026-09-18 |
| `broll-rack` | Ambient, reel | Generated V01 from S01, 2026-09-18 |
| `broll-enclave` | Reel, security | Generated V04 from S02, 2026-09-18 |
| `broll-desk-box` | Reel, SMB | Generated V05 from S03, 2026-09-18 |
| `broll-hall` | Reel open | Generated V06 from S08, 2026-09-18 |
| `broll-power` | Reel, economics | Generated V07 from S10, 2026-09-18 |
| `og-image.png` | Link previews | `scripts/media/og.mjs`, from the brand masters |
| `art-*` stills | Page heroes | Installed 2026-09-18 from S01–S11. Ledger in `takes/TAKES.md` |

The slots are declared in `src/lib/content/media.js`. Every slot is three files
in `static/media/`: `<slot>.mp4`, `<slot>.webm` and the poster `<slot>.webp`.
Which clip opens which page is the `heroClips` map in the same file. A page
with no clip shows its still from `art.json`, and a page with neither is text only.

`static/media/provenance.json` records how each slot was filled: recorded,
installed by hand, or assembled. The encoder never overwrites an installed slot
with a procedural recording unless it is run with `--force`.

`RATIONALE.md` explains why the media is made this way: the options that were
weighed, their downsides, and how every asset is verified end to end.

## The product mockup is private

The console clips are screen recordings of a mockup of the enterprise product,
running on demo data. **The mockup is not in this repository and must not be
added to it.** That was an instruction in the facelift brief: mock it up, keep
it out of the public repo, work locally, screen record it, make it swappable.

It lives at `C:\Scripts-Cdrive\Atlas-Inference\avarok-console-mock` on Alexi's
machine, a local git repository with no remote. Its README says how to run it.
If the team wants it shared, a private repository inside the organisation is
the way. Only the encoded recordings cross into this repository.

To re-record the console clips, start the mockup's preview server and pass its
address to the recorder here:

```sh
node scripts/media/record.mjs --console-origin http://127.0.0.1:4174
node scripts/media/encode.mjs
```

Without `--console-origin` the recorder skips the console clips and the encoder
keeps the committed files, so `bun run media` works for anyone with a clone.

When the real product exists, record the same six stories from it and install
them with `scripts/media/install.mjs`. No site code changes.

## The prompt pack

`shots.json` is the source. `PROMPTS.md` is generated from it by
`node scripts/media/brief.mjs` and is the sheet to copy and paste from. A test
fails when the sheet is stale, so edit the JSON and regenerate.

The pack is written for Grok Imagine, in four modes:

1. **Stills from text** (`S` shots). One image per industry and per platform
   idea, in one house style, so the solution pages stop being text only.
2. **Stills from our own frames** (`E` shots). The procedural b-roll frames fix
   the composition. Grok re-renders them as photographs, and the result drops
   into a slot whose layout is already proven on the page.
3. **Video from an approved still** (`V` shots). Ambient motion only. Ten
   seconds, one camera path, no cuts.
4. **Video from text** (`T` shots). For pictures we have no still of: the
   encrypted mesh between sites, the office at night, a macro of the silicon.
   The house style rides along in words, so expect more rejects.

### How the prompts are built

The research behind the structure, with sources in `shots.json`:

- For video, the model weighs the start of the prompt most. Motion verb first,
  then the camera, then atmosphere. Without a motion verb it returns a still
  that barely moves.
- When the source image is strong, the video prompt stays short. The model can
  see the picture. Describing it again adds noise and pulls stylised sources
  toward a different look.
- A locked camera is written as "camera not moving". "Stable" and "steady" get
  read as smooth motion.
- One camera path per shot. Two to three beats at most, in order, with commas.
- Resolution words do nothing. Aspect ratio, resolution and duration are
  settings, not prompt text.
- Stills get one house style, joined onto every prompt by the generator, so
  eleven images read as one set. Change the style once in `shots.json`.

### Rules for every generated asset

These are not taste. Each one is a legal or a trust problem if broken.

- **Never generate the logo.** The Avarok lockup is vector artwork in
  `assets/brand/`. Composite it in an editor.
- **Never generate product UI.** Every screen on the site is a recording of the
  mockup or, later, the product. A generated dashboard is a fabricated claim.
- **No readable text, numbers included.** Set text in the editor, in IBM Plex.
- **No vendor logos or recognisable product designs.** No military insignia,
  flags, seals or uniforms. No real buildings.
- **No faces.** Hands and silhouettes are fine when a shot needs scale.
- **Nothing that implies a named customer.** We have none to name yet.
- **Say where it came from.** Keep the prompt id in the commit message when an
  asset is installed, for example `media: art-finance from S04`.

### Accepting a result

Look at it at full size, then at the size it will render.

- It matches the "keep it if" line for its shot in `PROMPTS.md`.
- Geometry holds: racks stay straight, rows stay parallel, nothing melts. For
  video, check the last second, that is where it usually goes wrong.
- The palette holds: the ground is near black, the accents are lavender, cyan,
  emerald and amber gold, and nothing reads as pure white.
- It passes every rule above.

### Installing a result

```sh
# a still, onto the pages shots.json lists for its slot
node scripts/media/install.mjs --from ~/Downloads/vault.png --as art-finance

# a clip, into an existing slot, with its loop seam removed
node scripts/media/install.mjs --from ~/Downloads/aisle.mp4 --as broll-rack --crossfade 1
```

A still is converted to WebP, registered in `src/lib/content/art.json`, and
appears in the hero of its pages on the next build. A clip replaces the slot's
three files. Generated video does not loop by itself, and `--crossfade 1`
dissolves the last second into the first so the repeat has no jump.

In Git Bash on Windows, write `--pages` values without the leading slash
(`platform/security`). The shell rewrites arguments that start with one.

## The takes

`takes/` holds what the first Grok run produced on 2026-09-18, and
`takes/TAKES.md` is its ledger: every shot, the verdict, and why the rejects
were rejected. The stills are committed because the `V` shots start from them.
The raw video takes are not: they are large, the repository routes video
through LFS, and spending the organisation's LFS quota is a decision for the
team. They are on Alexi's machine in that folder. The film builds from them
when they are present and from the encoded slots when they are not.

## The film

`reel.json` is the cut and `scripts/media/reel.mjs` assembles it: the generated
footage, the recorded product clips, and captions that are checked against the
site's own copy, so the film cannot say something the pages do not. `REEL.md`
is the editorial note behind the cut, and the brief for a richer version with
music and sound when someone wants to make one in an editor.

```sh
bun x --bun vite build && node scripts/media/reel.mjs
```

## Regenerating everything that is procedural

```sh
bun x --bun vite build            # the recorder serves pages from build/
node scripts/media/record.mjs     # b-roll, plus console clips with --console-origin
node scripts/media/encode.mjs     # mp4, webm and poster for every recorded clip
node scripts/media/og.mjs         # static/og-image.png
node scripts/media/reel.mjs       # static/media/reel.*, from media-brief/reel.json
```

`scripts/media/README.md` explains each script.
