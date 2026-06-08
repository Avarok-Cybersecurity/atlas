# Harness aggregate — tier `baseline2` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_baseline2_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.100 ± 1.136 | 1.000 | 3.000 | [0.400, 1.800] | 6/10 |
| cargo_toml_valid | rate | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| cargo_toml_present | rate | 10 | 0.600 ± 0.490 | 1.000 | 1.000 | [0.300, 0.900] | 6/10 |
| tool_calls_total | count | 10 | 7.500 ± 4.129 | 7.000 | 16.000 | [5.100, 10.200] | 10/10 |
| write_calls | count | 10 | 2.200 ± 2.040 | 2.000 | 7.000 | [1.100, 3.600] | 8/10 |
| drift_empty_path | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| drift_path_outside_target | count | 10 | 0.200 ± 0.600 | 0.000 | 2.000 | [0.000, 0.600] | 1/10 |
| drift_path_literal_space | count | 10 | 0.200 ± 0.600 | 0.000 | 2.000 | [0.000, 0.600] | 1/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.300 ± 0.640 | 0.000 | 2.000 | [0.000, 0.800] | 2/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 1.200 ± 0.980 | 1.000 | 3.000 | [0.600, 1.800] | 7/10 |
| atlas_ws1_mask_fires | count | 10 | 16.000 ± 8.614 | 17.000 | 31.000 | [10.800, 21.400] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 4.300 ± 3.164 | 5.000 | 11.000 | [2.400, 6.300] | 8/10 |
| atlas_tier5c_retries | count | 10 | 0.400 ± 0.663 | 0.000 | 2.000 | [0.000, 0.900] | 3/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 7.100 ± 3.618 | 7.000 | 14.000 | [5.000, 9.400] | 10/10 |
| wall_time_s | count | 10 | 189.315 ± 103.001 | 200.698 | 349.134 | [127.786, 254.679] | 10/10 |
