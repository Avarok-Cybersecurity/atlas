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
