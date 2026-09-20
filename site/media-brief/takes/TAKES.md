# Takes — 2026-09-18 Grok Imagine run

Generated against `shots.json` / `PROMPTS.md`. House style and never-rules applied.
Models: stills `grok-imagine-image-quality`, video `grok-imagine-video-1.5`.

## Installed

| Shot | Slot | File | Verdict |
| --- | --- | --- | --- |
| S01 | `art-aisle` | `S01-art-aisle.jpg` | Keep. Symmetric aisle, emerald LEDs, cyan floor lines. Floor reflections are light. |
| S02 | `art-enclave` | `S02-art-enclave.jpg` | Keep. Door closed, glass wall, cyan room, isolated not abandoned. Pages: `/platform/security` only (gov page belongs to S06). |
| S03 | `art-desk-box` | `S03-art-desk-box.jpg` | Keep. Plain unbranded box, lavender status and rim. |
| S04 | `art-finance` | `S04-art-finance.jpg` | Keep. Mesh cage reads as custody. No currency, no tickers. |
| S05 | `art-health` | `S05-art-health.jpg` | Keep. Service corridor, frosted door, no patients or medical marks. |
| S06 take 2 | `art-gov` | `S06-art-gov.jpg` | Keep. Austere flush door, not the riveted bunker of take 1. |
| S07 take 2 | `art-legal` | `S07-art-legal.jpg` | Keep. Books and racks share one vanishing point. Spines are not fully blank but not readable at thumbnail. |
| S08 | `art-hyperscale` | `S08-art-hyperscale.jpg` | Keep. Scale first, rows parallel. 20:9. |
| S09 | `art-research` | `S09-art-research.jpg` | Keep. Engineer's bench, PSU display dark. |
| S10 | `art-power` | `S10-art-power.jpg` | Keep. Switchgear, copper, amber lamps, cyan doorway. |
| S11 | `art-prisms` | `S11-art-prisms.jpg` | Keep as motif, not the lockup. Four glass chevrons, brand palette, racks in the background. |
| E01 | (alt for aisle) | `E01-art-aisle.jpg` | Keep as alternate. Faithful restyle of the procedural rack frame. S01 won the slot. |
| E02 | `art-grid` | `E02-art-grid.jpg` | Keep. 3×8, locks between neighbours, no fake UI. |
| E03 take 2 | source for V03 | `E03-art-prisms.jpg` | Keep. Glass chevrons in haze, depth of field. |
| V01 | `broll-rack` | `V01-broll-rack.mp4` | Keep. Slow push, racks stay straight through the last frame. |
| V02 | `broll-tokens` | `V02-broll-tokens.mp4` | Keep. Modules do not move. Lights travel. |
| V03 | `broll-field` | `V03-broll-field.mp4` | Keep with a note: last frames gather chevrons and add god rays. Sit it behind text, not as a hero. |
| V04 | `broll-enclave` | `V04-broll-enclave.mp4` | Keep. Door stays closed. Camera crosses the glass. |
| V05 | `broll-desk-box` | `V05-broll-desk-box.mp4` | Keep. Box holds its shape through the orbit. |
| V06 | `broll-hall` | `V06-broll-hall.mp4` | Keep. Opening shot of the reel. Rows stay parallel. Source was 20:9; encoder crops to 16:9. |
| V07 | `broll-power` | `V07-broll-power.mp4` | Keep. Lamps read as load arriving. |

## Pass 2: the daylight industry scenes (`N` shots)

The industry pages moved from dark corridors to the buyer's own place of work. Sources are in
`takes/pass2/`, with the generating run's notes in `pass2/HANDOFF.md`. `RATIONALE.md` has the
reasoning for the second look.

| Shot | Slot | File | Verdict |
| --- | --- | --- | --- |
| N01 | `art-gov` | `pass2/art-gov.webp` | Installed. Airfield from directly overhead. No markings legible, nothing armed, nothing in flight. |
| N02 | `art-legal` | `pass2/art-legal.webp` | Installed. Round table, blank folders, clear day skyline, nobody in the room. |
| N03 | `art-health` | `pass2/art-health.webp` | Installed, with a note. Faces are toward the camera at mid distance. At hero size they are about twelve pixels and nobody is a portrait, which is inside the rule. If it is ever shown larger, regenerate with the staff turned away. One generic word is readable on a sign. |
| N04 | `art-finance` | `pass2/art-finance.webp` | Installed. Everyone from behind, no screen legible. |
| N05 | `art-research` | `pass2/art-research.webp` | Installed. Also the Labs page. |
| N07 | `art-smb-library` | `pass2/art-smb-library.webp` | Installed. One of the stills the SMB and edge page turns through. |
| N08 | `art-smb-autoshop` | `pass2/art-smb-autoshop.webp` | Installed. Generic cars, no plates or badges legible. |
| N09 | `art-smb-office` | `pass2/art-smb-corporate.webp` | Installed. A tote bag carries an invented wordmark, too small to read at hero size. |
| N06 | `art-smb-police` | `pass2/art-smb-police.webp` | **Held back.** A paper bag on the bench carries a mark that reads as a real fast food brand, the officers' sidearms are in view, and the notice board shows faces. Regenerate: no bag, holsters out of frame or officers seated behind the counter, a board of plain notices. The page turns through three stills until then, and takes the fourth the moment it is installed. |

`art-desk-box` now serves `/pricing` only, so the daylight set on the SMB and edge page is not
interleaved with a dark still.

## Rejected

| Shot | File | Why |
| --- | --- | --- |
| S06 take 1 | `S06-art-gov-take1.jpg` | Riveted bunker door. Menacing, not austere. |
| S07 take 1 | `S07-art-legal-take1.jpg` | Mixed aisle, not a halfway transformation. |
| E03 take 1 | `E03-art-prisms-reject.jpg` | Invented a datacenter. Composition of the field was lost. |

## Not run

- **E04** (`art-console-desk`). The keep-if requires every word on the console poster to survive. Generate-and-hope will garble the UI. Composite the poster onto S03 in an editor when someone wants that frame.
- **Reel cut.** `REEL.md` is the storyboard. Sources now exist. Cut it in Premiere (or ffmpeg concat) when the team wants the sixty-second film on `/demo`.

## Prompt ids for commits

Install messages already name the slot. Example: `media: art-finance from S04`.
