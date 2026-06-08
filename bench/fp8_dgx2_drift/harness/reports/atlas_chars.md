# Harness aggregate — tier `atlas_chars` (N=7)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_atlas_chars_*.json`.
Runs: [1, 2, 3, 4, 5, 6, 7]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 7 | 1.714 ± 1.161 | 1.000 | 3.000 | [0.857, 2.571] | 6/7 |
| cargo_toml_valid | rate | 7 | 0.143 ± 0.350 | 0.000 | 1.000 | [0.000, 0.429] | 1/7 |
| cargo_toml_present | rate | 7 | 0.857 ± 0.350 | 1.000 | 1.000 | [0.571, 1.000] | 6/7 |
| tool_calls_total | count | 7 | 11.143 ± 5.276 | 10.000 | 20.000 | [7.429, 15.000] | 7/7 |
| write_calls | count | 7 | 3.857 ± 1.245 | 4.000 | 6.000 | [3.000, 4.857] | 7/7 |
| drift_empty_path | count | 7 | 0.429 ± 0.728 | 0.000 | 2.000 | [0.000, 1.000] | 2/7 |
| drift_path_outside_target | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| drift_path_literal_space | count | 7 | 0.429 ± 0.728 | 0.000 | 2.000 | [0.000, 1.000] | 2/7 |
| drift_lean_prefix | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| drift_bash_as_content | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| drift_xml_attr_leak | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| drift_toml_newlines_collapsed | count | 7 | 2.429 ± 1.178 | 2.000 | 4.000 | [1.571, 3.286] | 7/7 |
| atlas_ws1_mask_fires | count | 7 | 7.286 ± 1.979 | 7.000 | 12.000 | [6.286, 9.000] | 7/7 |
| atlas_b1_drift_fires | count | 7 | 0.429 ± 0.495 | 0.000 | 1.000 | [0.143, 0.857] | 3/7 |
| atlas_tier5c_retries | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| atlas_a2_fuzzy_fires | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| atlas_tool_call_lines | count | 7 | 10.714 ± 4.949 | 10.000 | 20.000 | [7.286, 14.571] | 7/7 |
| wall_time_s | count | 7 | 237.400 ± 98.523 | 214.540 | 360.083 | [165.351, 309.090] | 7/7 |
