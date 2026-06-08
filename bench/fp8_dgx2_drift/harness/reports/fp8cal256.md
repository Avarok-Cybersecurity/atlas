# Harness aggregate — tier `fp8cal256` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_fp8cal256_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 135.900 ± 403.703 | 2.000 | 1347.000 | [0.600, 405.400] | 5/10 |
| cargo_toml_valid | rate | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| cargo_toml_present | rate | 10 | 0.500 ± 0.500 | 1.000 | 1.000 | [0.200, 0.800] | 5/10 |
| tool_calls_total | count | 10 | 9.500 ± 4.056 | 11.000 | 17.000 | [7.000, 12.100] | 10/10 |
| write_calls | count | 10 | 2.700 ± 2.326 | 2.000 | 7.000 | [1.300, 4.200] | 8/10 |
| drift_empty_path | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_path_outside_target | count | 10 | 0.300 ± 0.640 | 0.000 | 2.000 | [0.000, 0.800] | 2/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.600 ± 0.917 | 0.000 | 2.000 | [0.000, 1.200] | 3/10 |
| atlas_ws1_mask_fires | count | 10 | 16.800 ± 9.020 | 20.000 | 29.000 | [11.100, 22.200] | 9/10 |
| atlas_b1_drift_fires | count | 10 | 5.300 ± 2.968 | 6.000 | 12.000 | [3.500, 7.200] | 9/10 |
| atlas_tier5c_retries | count | 10 | 0.500 ± 0.806 | 0.000 | 2.000 | [0.000, 1.000] | 3/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 7.400 ± 3.826 | 8.000 | 13.000 | [5.000, 9.700] | 9/10 |
| wall_time_s | count | 10 | 228.894 ± 79.982 | 210.664 | 360.088 | [181.787, 280.047] | 10/10 |
