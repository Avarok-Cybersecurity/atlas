# Pass 2 stills — for Fable

Grok generated these. Fable owns placement, `art.json`, and install.
Do not start a preview server from this tree; Fable is already hosting.

All files are 1280×720 WebP, lanczos, q70–78. Page-hero weight.

## Install (when you choose to)

From `site/`:

```
node scripts/media/install.mjs --from media-brief/takes/pass2/art-gov.webp --as art-gov
node scripts/media/install.mjs --from media-brief/takes/pass2/art-health.webp --as art-health
node scripts/media/install.mjs --from media-brief/takes/pass2/art-finance.webp --as art-finance
node scripts/media/install.mjs --from media-brief/takes/pass2/art-legal.webp --as art-legal
node scripts/media/install.mjs --from media-brief/takes/pass2/art-research.webp --as art-research
```

SMB/edge is four interchangeable stills. Pick one, or rotate:

```
node scripts/media/install.mjs --from media-brief/takes/pass2/art-smb-police.webp --as art-desk-box --pages solutions/smb-edge
node scripts/media/install.mjs --from media-brief/takes/pass2/art-smb-library.webp --as art-desk-box --pages solutions/smb-edge
node scripts/media/install.mjs --from media-brief/takes/pass2/art-smb-autoshop.webp --as art-desk-box --pages solutions/smb-edge
node scripts/media/install.mjs --from media-brief/takes/pass2/art-smb-corporate.webp --as art-desk-box --pages solutions/smb-edge
```

`--as art-desk-box` overwrites the current desktop-box still. If you want to keep the box for `/pricing`, register a new slot instead.

## What to keep from pass 1

- `art-aisle` / enterprise datacenter — leave it. That brief was correct.
- `art-hyperscale`, `art-enclave`, `art-power`, `art-prisms`, `art-grid` — not in this pass.

## Review notes

| File | Slot | Verdict |
| --- | --- | --- |
| `art-gov.webp` | government-defense | Keep. Nadir satellite of an airfield: fighters, transports, helos, small drones. No flags or readable tail numbers at this size. |
| `art-health.webp` | healthcare | Keep. Plain clinic reception, staff talking over a chart, visitors at the desk, patients waiting. A "Triage" sign is the only readable word. |
| `art-finance.webp` | financial-services | Keep. Triple-monitor desks, charts, operators from behind. Numerals are not legible at hero size. |
| `art-legal.webp` | legal | Keep. Round table, low stack of blank folders, clear-day skyline. |
| `art-research.webp` | research, labs | Keep. Modern open office, daylight, whiteboards, not a hardware bench. |
| `art-smb-police.webp` | smb-edge option | Keep. Precinct lobby. Generic "Employees only" / "Please wait" signs. Patches are not a named force at thumbnail. |
| `art-smb-library.webp` | smb-edge option | Keep. Working public library. "Information" sign only. |
| `art-smb-autoshop.webp` | smb-edge option | Keep. Two lifts, mechanic and customer. 94 KB after a second compress. |
| `art-smb-corporate.webp` | smb-edge option | Keep. Beige cubicle office. A tote bag has invented wordmark; crop or leave. |

## Suggested alt text

- gov: Overhead satellite view of an airfield with rows of parked aircraft, helicopters and small drones.
- health: A clinic reception desk in the morning, staff talking with visitors and with each other.
- finance: A room of triple-monitor trading desks, seen from behind the operators.
- legal: A round conference table with a neat stack of folders, city skyline in clear daylight beyond the windows.
- research: A modern open-plan research office with shared desks, plants and daylight.
- smb-police: The public lobby of a small police station.
- smb-library: A public library reading room in the afternoon.
- smb-autoshop: An independent garage, two cars on lifts, a mechanic talking to a customer.
- smb-corporate: A generic small-company cubicle office.

## Not in this drop

- Product UI, the lockup, E04 console-on-desk.
- Video. The gov still is composed so it can become a slow satellite push later.
