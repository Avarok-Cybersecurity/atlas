# Harness aggregate — tier `atlas_sf` (N=5)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_atlas_sf_*.json`.
Runs: [1, 2, 3, 4, 5]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 5 | 1.400 ± 1.200 | 2.000 | 3.000 | [0.400, 2.400] | 3/5 |
| cargo_toml_valid | rate | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/5 |
| cargo_toml_present | rate | 5 | 0.600 ± 0.490 | 1.000 | 1.000 | [0.200, 1.000] | 3/5 |
| tool_calls_total | count | 5 | 4.800 ± 2.482 | 6.000 | 8.000 | [2.600, 6.800] | 5/5 |
| write_calls | count | 5 | 2.200 ± 2.227 | 2.000 | 6.000 | [0.400, 4.200] | 3/5 |
| drift_empty_path | count | 5 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.600] | 1/5 |
| drift_path_outside_target | count | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/5 |
| drift_path_literal_space | count | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/5 |
| drift_lean_prefix | count | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/5 |
| drift_bash_as_content | count | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/5 |
| drift_xml_attr_leak | count | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/5 |
| drift_toml_newlines_collapsed | count | 5 | 1.000 ± 1.549 | 0.000 | 4.000 | [0.000, 2.600] | 2/5 |
| atlas_ws1_mask_fires | count | 5 | 1.800 ± 0.980 | 2.000 | 3.000 | [0.800, 2.600] | 4/5 |
| atlas_b1_drift_fires | count | 5 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.600] | 1/5 |
| atlas_tier5c_retries | count | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/5 |
| atlas_a2_fuzzy_fires | count | 5 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/5 |
| atlas_tool_call_lines | count | 5 | 5.000 ± 2.191 | 6.000 | 8.000 | [3.200, 6.800] | 5/5 |
| wall_time_s | count | 5 | 334.010 ± 52.108 | 360.056 | 360.080 | [281.900, 360.073] | 5/5 |
