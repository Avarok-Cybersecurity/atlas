# Harness aggregate — tier `kvbf16ws` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_kvbf16ws_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.400 ± 1.200 | 1.000 | 4.000 | [0.700, 2.200] | 8/10 |
| cargo_toml_valid | rate | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| cargo_toml_present | rate | 10 | 0.800 ± 0.400 | 1.000 | 1.000 | [0.500, 1.000] | 8/10 |
| tool_calls_total | count | 10 | 7.300 ± 3.348 | 7.000 | 12.000 | [5.200, 9.400] | 10/10 |
| write_calls | count | 10 | 3.200 ± 1.661 | 3.000 | 7.000 | [2.300, 4.300] | 10/10 |
| drift_empty_path | count | 10 | 0.800 ± 0.980 | 1.000 | 3.000 | [0.200, 1.400] | 5/10 |
| drift_path_outside_target | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.700 ± 0.900 | 0.000 | 2.000 | [0.200, 1.300] | 4/10 |
| atlas_ws1_mask_fires | count | 10 | 1.900 ± 0.300 | 2.000 | 2.000 | [1.700, 2.000] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_tier5c_retries | count | 10 | 0.700 ± 0.781 | 1.000 | 2.000 | [0.200, 1.200] | 5/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 6.600 ± 3.499 | 6.000 | 11.000 | [4.400, 8.800] | 10/10 |
| wall_time_s | count | 10 | 312.346 ± 81.875 | 360.077 | 360.088 | [255.535, 355.964] | 10/10 |
