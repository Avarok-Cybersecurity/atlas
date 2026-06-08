# Harness aggregate — tier `fencesalvage` (N=4)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_fencesalvage_*.json`.
Runs: [1, 2, 3, 4]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 4 | 4.500 ± 5.500 | 2.000 | 14.000 | [1.000, 10.750] | 4/4 |
| cargo_toml_valid | rate | 4 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 4/4 |
| cargo_toml_present | rate | 4 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 4/4 |
| tool_calls_total | count | 4 | 9.250 ± 1.785 | 9.000 | 12.000 | [7.500, 11.250] | 4/4 |
| write_calls | count | 4 | 2.750 ± 1.090 | 3.000 | 4.000 | [1.500, 3.750] | 4/4 |
| drift_empty_path | count | 4 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/4 |
| drift_path_outside_target | count | 4 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/4 |
| drift_path_literal_space | count | 4 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/4 |
| drift_lean_prefix | count | 4 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/4 |
| drift_bash_as_content | count | 4 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/4 |
| drift_xml_attr_leak | count | 4 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/4 |
| drift_toml_newlines_collapsed | count | 4 | 0.250 ± 0.433 | 0.000 | 1.000 | [0.000, 0.750] | 1/4 |
| atlas_ws1_mask_fires | count | 4 | 2.750 ± 1.299 | 2.000 | 5.000 | [2.000, 4.250] | 4/4 |
| atlas_b1_drift_fires | count | 4 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/4 |
| atlas_tier5c_retries | count | 4 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/4 |
| atlas_a2_fuzzy_fires | count | 4 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/4 |
| atlas_tool_call_lines | count | 4 | 9.250 ± 1.785 | 9.000 | 12.000 | [7.500, 11.250] | 4/4 |
| wall_time_s | count | 4 | 211.417 ± 46.273 | 240.743 | 266.922 | [169.001, 253.833] | 4/4 |
