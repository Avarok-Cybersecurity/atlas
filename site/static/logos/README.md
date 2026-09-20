# Logos on the wall

Every mark the front page and the company page show, self hosted so the site makes no third party request. The wall and the programs row render from `logoWall` in `src/lib/content/home.js`, through `src/lib/components/avarok/LogoWall.svelte`. A test in `src/lib/content/site.test.js` fails if a file here is not listed in this document.

Each mark is a trademark of its owner. They are used to name a prior employer of the team, or a program the company takes part in. None is an endorsement, and the note under the wall says so.

## Prior roles: company wordmarks

Vector masters from Wikimedia Commons. The wall greys them, and inverts them on the dark theme so a black wordmark reads white. Hover restores full opacity but not colour.

| file | source | terms as stated at the source |
| --- | --- | --- |
| `kraken.svg` | Commons, File:K-logo-wikipedia.svg | Public domain (not copyrightable). Trademark of its owner |
| `beyondgravity.svg` | Commons, File:Beyond Gravity company logo.svg | Public domain (not copyrightable). Trademark of its owner |
| `synopsys.svg` | Commons, File:Synopsys Logo.svg | Public domain (not copyrightable). Trademark of its owner |
| `anaconda.svg` | Commons, File:Anaconda logo.svg | Public domain (not copyrightable). Trademark of its owner |
| `google.svg` | Commons, File:Google 2015 logo.svg | Public domain (not copyrightable). Trademark of its owner |

## Prior roles: the two commands

| file | source | what was done to it |
| --- | --- | --- |
| `uscybercom.webp` | Commons, File:Seal of the United States Cyber Command.svg | Rendered at 160 px, saved as WebP. The vector master is 2.3 MB |
| `nswc.webp` | Commons, File:US NSWC insignia.jpg | Masked to a circle to drop the photo's white corners, scaled to 160 px, saved as WebP |

Official names, as the organisations write them: **United States Cyber Command** and **Naval Special Warfare Command**.

**Read this before launch.** Both files are works of the United States government, so nobody holds a copyright in them. That is not the same as being free to use. Department of Defense seals, emblems and insignia are protected, and a company may not use them in a way that could suggest the Department endorses it. Permission comes from the owning service's trademark and licensing office (for the Navy, the Navy Trademark Licensing Office). The wall names prior roles, says it is not an endorsement, and carries the Department's standard disclaimer, but none of that is permission.

So the emblems are a switch. Set `logoWall.emblems` to `false` in `src/lib/content/home.js` and both entries render as set type ("U.S. Cyber Command", "Naval Special Warfare"), which needs nobody's permission. Whoever owns the launch decision should either get the permission in writing or flip the switch.

## Programs and partners

Shown in their own colours. A logo drawn in dark ink has a second file for the dark theme. All SVGs were run through `svgo --multipass`.

| file | source | note |
| --- | --- | --- |
| `amd.svg`, `amd-dark.svg` | Commons, File:AMD Logo.svg | Public domain (not copyrightable), trademarked. The dark file is the same path filled white |
| `mlcommons.svg`, `mlcommons-dark.svg` | mlcommons.org, `ML-Commons-Logo.svg` and `ML-Commons-Logo-white.svg` from the site theme | MLCommons' own light and dark logos, unaltered |
| `huggingface.svg`, `huggingface-dark.svg` | huggingface.co/datasets/huggingface/brand-assets, `hf-logo-with-title.svg` | From Hugging Face's published brand assets. The dark file changes only the title's fill, from `#000B1B` to white |
| `qwen.svg` | Commons, File:Qwen Logo.svg, from github.com/QwenLM/Qwen | Apache License 2.0, trademarked. One file serves both themes |
| `scale.svg`, `scale-dark.svg` | docs.scale-lang.com, `logo_white.svg` | Spectral Compute's mark for SCALE. The dark file is the original. The light file is the same path filled `#14171c`. The mark has no wordmark, so the name is set beside it |

The NVIDIA Inception badge is `static/nvidia-inception.webp`, the badge the program gives its members.

## Adding a mark

Drop the file here, add it to `logoWall.items` (`file` for a wordmark, `emblem` for a round WebP) or to `logoWall.programs` (`file`, and `fileDark` if it is drawn in dark ink), and add a row to the right table above with where it came from and on what terms. Keep wordmarks as SVG. Run `bun x svgo --multipass` on anything over a few kilobytes.
