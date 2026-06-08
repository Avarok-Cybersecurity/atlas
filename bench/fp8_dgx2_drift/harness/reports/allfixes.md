# Harness aggregate — tier `allfixes` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_allfixes_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.400 ± 1.200 | 1.000 | 3.000 | [0.700, 2.200] | 7/10 |
| cargo_toml_valid | rate | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| cargo_toml_present | rate | 10 | 0.700 ± 0.458 | 1.000 | 1.000 | [0.400, 1.000] | 7/10 |
| tool_calls_total | count | 10 | 6.500 ± 3.202 | 6.000 | 14.000 | [4.700, 8.600] | 10/10 |
| write_calls | count | 10 | 1.200 ± 1.327 | 1.000 | 4.000 | [0.500, 2.100] | 6/10 |
| drift_empty_path | count | 10 | 0.400 ± 0.917 | 0.000 | 3.000 | [0.000, 1.000] | 2/10 |
| drift_path_outside_target | count | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.600 ± 0.917 | 0.000 | 3.000 | [0.100, 1.200] | 4/10 |
| atlas_ws1_mask_fires | count | 10 | 11.800 ± 5.845 | 11.000 | 28.000 | [8.900, 15.900] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 2.400 ± 1.562 | 2.000 | 6.000 | [1.500, 3.400] | 10/10 |
| atlas_tier5c_retries | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 6.300 ± 3.378 | 6.000 | 14.000 | [4.400, 8.500] | 10/10 |
| wall_time_s | count | 10 | 256.502 ± 99.932 | 332.318 | 360.087 | [194.157, 317.754] | 10/10 |
