# Why the media is made this way

The brief asked for the media to be scoped completely, reasoned through against
the options and their downsides, and verified end to end. This is that
reasoning. `README.md` says how to do the work. This says why it is done this
way, so the next person can disagree with a decision instead of guessing at it.

## The three kinds of picture, and where each comes from

| Kind | Source | Why |
| --- | --- | --- |
| The product | Screen recordings of a coded mockup | It has text in it, and the text has to be true |
| The world the product lives in | Generated from the prompt pack | No one has photographed our datacenter, and we do not have one |
| The brand in motion | Drawn by code from the palette | It has to loop perfectly and weigh nothing |

### The product: recorded, never generated

Options considered:

1. **Generate interface footage with a video model.** Fast and free of
   engineering. Rejected. Video models garble text, and an interface is mostly
   text. Worse, a generated dashboard shows numbers nobody chose, on a site
   whose whole argument is that every number is a receipt.
2. **Animate a static design in a motion tool.** Looks polished. Rejected as
   the primary source because it cannot be re-cut when the story changes
   without the person and the tool that made it, which is the opposite of what
   the brief asks of handoff.
3. **Build a working mockup and record it.** Chosen. The interface is real
   code, so the text is sharp, the numbers come from one seeded simulation, and
   a scene is a list of timed steps anyone can edit. Re-recording is one
   command.

The downsides, and what is done about them:

- *It is a mockup, and a viewer could take it for the product.* Every frame
  carries a "Demo data" chip in the interface itself, the film frames product
  footage in a window so the chip is never cropped, and the captions under the
  clips say it was recorded from the product mockup.
- *It shows a product that does not exist yet.* The brief accepted that in so
  many words. The mitigation is that the mockup stays private, so the public
  repository never appears to contain a product it does not contain.
- *The numbers could embarrass us.* The ledger's volume, licence and payback
  now follow from the hardware on screen, so a buyer who does the sum gets the
  same answer. Only one number is measured, and the interface says which.

### The world: generated, from stills first

Options considered:

1. **Stock footage.** Cheap and instant. Rejected. It is the same footage our
   competitors use, it is never in our palette, and the licence travels badly
   into a public repository.
2. **Text to video directly.** One step. Used only where there is no still to
   start from (`T` shots). With no image to hold the look, the palette drifts
   from clip to clip and more generations are thrown away.
3. **Text to image, approve the still, then image to video.** Chosen for almost
   everything. The still is cheap to regenerate until it is right, it is useful
   by itself as a page image, and it pins the look of the video made from it.
   Eleven stills made this way read as one set because they share one style
   block, joined on by the generator.
4. **Image to image from our own procedural frames** (`E` shots). Chosen where
   a loop already sat on a page. The procedural frame fixes the composition, so
   the photographic version drops into a layout already proven.

The downsides:

- *A still can lock in a flaw the video then inherits.* That is why each shot
  has a "keep it if" line and why rejected takes are kept in `takes/` with the
  reason, not deleted.
- *Generated video does not loop.* `install.mjs --crossfade` dissolves the end
  into the start. For footage that sits behind text that is enough.
- *Generated chevrons can look like a bad copy of the logo.* The rule is to
  discard any frame that reads as the mark. The film's title card was moved off
  the chevron footage for exactly this reason, and `reel.json` records why.
- *Models render plausible text where there should be none.* Every accepted
  still was checked at full size. The bright panels in the aisle image are LED
  matrices, not words.
- *Real photographs would beat all of it.* They would. A picture of the two
  development boxes on a desk, or the team at an event, is worth more than any
  of these. The slots take a photograph exactly as they take a generated image.

### The brand in motion: drawn by code

The three procedural loops exist because they can do things footage cannot:
loop with no seam (every motion is a whole number of cycles, and a test proves
the last frame is the first), weigh a few hundred kilobytes, and be regenerated
by anyone with a clone. Generated footage has since replaced them in their
slots. They remain the fallback, and the source frames for the `E` shots.

## Why the film is assembled by a script

A film cut by hand in an editor is a file nobody else can change. This one is
`reel.json` plus `scripts/media/reel.mjs`: the cut is data, the captions are
checked against the site's copy, and the whole film rebuilds in about a minute
when a clip or a sentence changes. The cost is that it cannot do what an editor
can: no music, no sound design, no bespoke motion graphics. When the team wants
those, the master this script writes (`.media-raw/reel/reel-master.mp4`) and
the per segment files beside it are the right starting point in Premiere.

## End to end verification

Every asset, generated or recorded, goes through the same path before it is on
the site. None of the steps depends on taste alone.

1. **Generate or record.** From `PROMPTS.md`, or from the mockup.
2. **Accept.** Against the shot's "keep it if" line and the rules in
   `README.md`. Record the verdict in `takes/TAKES.md`, rejects included.
3. **Install.** `scripts/media/install.mjs`. It encodes to the site's formats,
   registers stills with their alt text and pages, and records provenance so
   the recorder never overwrites the asset with a placeholder.
4. **Unit tests.** Every slot has all three files, none is a placeholder, none
   is over budget, every still has alt text and sits on a page that exists, and
   the prompt sheet matches its source.
5. **Build and browser tests.** Every page renders with one heading, and the
   front page makes no request to a third party.
6. **Look at it.** In both themes, at desktop width and on a phone. Footage sits
   in a dark frame in the light theme too, so one asset serves both.
7. **Lighthouse, in CI.** The front page has to stay at a perfect score, which
   is why clips load nothing until they are near the viewport and the hero
   poster is kept under 100 KB.
8. **Commit with the prompt id.** So any image can be traced to the words that
   made it.

Visitors who ask for reduced motion get posters and no video. That is handled
once, in `VideoClip.svelte`, for every clip on the site.
