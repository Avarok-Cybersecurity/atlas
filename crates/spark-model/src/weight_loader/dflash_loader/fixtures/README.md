# dflash_loader fixtures

`qwen3_5_4b_dflash_config.json` is the `config.json` of
`z-lab/Qwen3.5-4B-DFlash` (https://huggingface.co/z-lab/Qwen3.5-4B-DFlash),
copied byte for byte (1,220 bytes, sha256
`6fa9ca0d10d2c3f5c93043bdd492e35283e284e4eb6a9e0479ba663609117203`) from the
local download in `~/models/Qwen3.5-4B-DFlash/` on 2026-11-12.

License: Apache-2.0 (the model card's `license: apache-2.0`). The drafter's
weights are not bundled with Atlas; only this configuration text is used, as a
test fixture for `parse_dflash_config` and the shape check. Attribution is in
the repository `NOTICE` and `CITATIONS.md` (DFlash: Chen, Liang and Liu,
arXiv:2602.06036).

Do not edit the JSON; replace it from a fresh download if the drafter is
re-released, and update the pinned values in `loader_tests.rs`.
