# Harness aggregate — tier `sm1_a2ao_sc1_steps` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_sm1_a2ao_sc1_steps_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.300 ± 1.187 | 1.000 | 3.000 | [0.600, 2.100] | 7/10 |
| cargo_toml_valid | rate | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| cargo_toml_present | rate | 10 | 0.600 ± 0.490 | 1.000 | 1.000 | [0.300, 0.900] | 6/10 |
| tool_calls_total | count | 10 | 9.200 ± 5.564 | 8.000 | 18.000 | [5.900, 12.800] | 10/10 |
| write_calls | count | 10 | 2.900 ± 3.360 | 2.000 | 11.000 | [1.100, 5.100] | 7/10 |
| drift_empty_path | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_path_outside_target | count | 10 | 1.500 ± 2.377 | 0.000 | 7.000 | [0.200, 3.100] | 4/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.300 ± 0.900 | 0.000 | 3.000 | [0.000, 0.900] | 1/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 2.200 ± 3.311 | 1.000 | 11.000 | [0.500, 4.500] | 6/10 |
| atlas_ws1_mask_fires | count | 10 | 17.900 ± 9.823 | 19.000 | 34.000 | [12.000, 23.900] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 2.400 ± 2.332 | 2.000 | 8.000 | [1.100, 3.900] | 8/10 |
| atlas_tier5c_retries | count | 10 | 0.300 ± 0.640 | 0.000 | 2.000 | [0.000, 0.700] | 2/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 8.900 ± 5.338 | 7.000 | 18.000 | [5.800, 12.300] | 10/10 |
| wall_time_s | count | 10 | 205.818 ± 100.684 | 181.019 | 360.067 | [146.492, 269.830] | 10/10 |
