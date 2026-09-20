# The reel

**Built.** `reel.json` is the cut as it stands and `scripts/media/reel.mjs`
assembles it into `static/media/reel.*`, which plays on the demo page. This
file is the editorial note behind that cut, and the brief for a richer version
made in an editor, with music and sound, when someone wants one.

A storyboard for a sixty second product film. The brief asked for a video that
shows how the product gets deployed and what is in it, with an enterprise look
and feel. Everything it needs either exists already or has a prompt in
`PROMPTS.md`.

Format: 1920 by 1080 master, 30 frames a second. Cut a 1280 by 720 web version
for the site and a 1080 by 1920 vertical for social from the same timeline.

## Sources

| Kind | Where | Note |
| --- | --- | --- |
| Product footage | `static/media/console-*.mp4`, or re-record at 1920 by 1080 | The only source of UI. Never generated. |
| Ambient footage | Prompts `V01` to `V07` | Until they exist, the procedural loops `broll-*.mp4` stand in |
| Lockup | `assets/brand/logo-full-ondark.svg` | Vector. Animate a reveal, never redraw |
| Type | IBM Plex Sans 600 for cards, IBM Plex Mono for labels | The site's faces, in `static/fonts/` |
| Music | A licensed library track | No generated music. Its licence is unclear |

To re-record the product footage at full HD, change the four console clips to
`{ w: 1920, h: 1080 }` in `src/lib/content/media.js`, record, and change them
back. The mockup lays out correctly at that size.

## The cut

| Time | Picture | Card or caption |
| --- | --- | --- |
| 0:00 to 0:05 | `V06`, the glide over the data hall | Your GPUs report tokens per second. |
| 0:05 to 0:10 | `V07`, the switchgear room | Your CFO pays in dollars per workload. |
| 0:10 to 0:14 | `V03`, the glass chevrons, the lockup resolves over it | Inference economics, reimagined. |
| 0:14 to 0:25 | `console-ask`, from the grid change to the finished answer | Pick a model. Pick a grid. Start. |
| 0:25 to 0:34 | `console-fleet`, the canary climbing, the node repairing itself | Every node, every kernel, every rollout. |
| 0:34 to 0:43 | `console-economics`, the line drawing in, the payback tile | Dollars per workload, not tokens per second. |
| 0:43 to 0:50 | `console-governance`, the policy switching on, the provenance panel | Every token has a receipt. |
| 0:50 to 0:54 | `V04`, the sealed room | Nothing leaves your perimeter. |
| 0:54 to 0:58 | `V01`, the push down the aisle | Week one. Months one to six. At renewal. |
| 0:58 to 1:03 | Lockup on the ground colour | Book a working session. atlascybernetics.ai |

## Rules for the cut

- Every card is a line from `src/lib/content/home.js`: the hero kicker, the
  problem statement, the tour headings, the differences and the deliveries
  timeline. If the site copy changes, the cards change with it.
- No number appears in the film unless it is on screen in a recording, where
  the "Demo data" chip is visible in the same frame.
- Hold each product shot long enough to read one thing. Speed ramps are fine,
  jump cuts inside a product shot are not: they hide what the product does.
- The "Demo data" chip stays in frame. Do not crop it out.
- End on the call to action for at least four seconds.

## Putting it on the site

Add a slot to `src/lib/content/media.js`, for example
`reel: clip('reel', '...', { loop: false })`, install the file with
`scripts/media/install.mjs --from reel.mp4 --as reel`, and render it with
`<VideoClip clip={media.reel} controls />` on the demo page. A sixty second
file is several megabytes. `VideoClip` fetches nothing until the visitor
scrolls to it, so the page weight does not change.
