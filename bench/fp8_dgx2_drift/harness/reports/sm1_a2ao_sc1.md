# Harness aggregate — tier `sm1_a2ao_sc1` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_sm1_a2ao_sc1_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.000 ± 1.000 | 1.000 | 3.000 | [0.400, 1.600] | 6/10 |
| cargo_toml_valid | rate | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| cargo_toml_present | rate | 10 | 0.600 ± 0.490 | 1.000 | 1.000 | [0.300, 0.900] | 6/10 |
| tool_calls_total | count | 10 | 9.800 ± 3.572 | 11.000 | 15.000 | [7.600, 12.000] | 10/10 |
| write_calls | count | 10 | 2.200 ± 2.713 | 1.000 | 8.000 | [0.700, 4.000] | 6/10 |
| drift_empty_path | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_path_outside_target | count | 10 | 0.400 ± 1.200 | 0.000 | 4.000 | [0.000, 1.200] | 1/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.400 ± 0.663 | 0.000 | 2.000 | [0.000, 0.900] | 3/10 |
| atlas_ws1_mask_fires | count | 10 | 18.000 ± 5.138 | 21.000 | 26.000 | [14.700, 21.000] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 2.500 ± 2.500 | 2.000 | 7.000 | [1.100, 4.200] | 8/10 |
| atlas_tier5c_retries | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 9.500 ± 3.557 | 11.000 | 15.000 | [7.300, 11.700] | 10/10 |
| wall_time_s | count | 10 | 152.328 ± 55.344 | 132.580 | 293.069 | [122.806, 190.937] | 10/10 |
