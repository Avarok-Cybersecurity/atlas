# Harness aggregate — tier `ssmfix` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_ssmfix_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.700 ± 1.552 | 3.000 | 4.000 | [0.700, 2.600] | 6/10 |
| cargo_toml_valid | rate | 10 | 0.400 ± 0.490 | 0.000 | 1.000 | [0.100, 0.700] | 4/10 |
| cargo_toml_present | rate | 10 | 0.500 ± 0.500 | 1.000 | 1.000 | [0.200, 0.800] | 5/10 |
| tool_calls_total | count | 10 | 7.100 ± 3.330 | 8.000 | 13.000 | [5.100, 9.200] | 10/10 |
| write_calls | count | 10 | 1.400 ± 1.200 | 2.000 | 4.000 | [0.700, 2.200] | 7/10 |
| drift_empty_path | count | 10 | 0.600 ± 1.020 | 0.000 | 3.000 | [0.000, 1.300] | 3/10 |
| drift_path_outside_target | count | 10 | 0.400 ± 0.663 | 0.000 | 2.000 | [0.000, 0.800] | 3/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.500 ± 0.806 | 0.000 | 2.000 | [0.000, 1.000] | 3/10 |
| atlas_ws1_mask_fires | count | 10 | 9.700 ± 7.058 | 9.000 | 22.000 | [5.500, 14.200] | 9/10 |
| atlas_b1_drift_fires | count | 10 | 2.700 ± 2.934 | 2.000 | 8.000 | [1.000, 4.600] | 7/10 |
| atlas_tier5c_retries | count | 10 | 0.400 ± 0.490 | 0.000 | 1.000 | [0.100, 0.700] | 4/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 6.700 ± 3.035 | 7.000 | 12.000 | [4.800, 8.600] | 10/10 |
| wall_time_s | count | 10 | 248.331 ± 105.365 | 288.630 | 360.084 | [181.374, 313.593] | 10/10 |
