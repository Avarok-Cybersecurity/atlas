# Harness aggregate — tier `nothink` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_nothink_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 826.700 ± 307.914 | 897.000 | 1096.000 | [608.100, 981.800] | 10/10 |
| cargo_toml_valid | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| cargo_toml_present | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| tool_calls_total | count | 10 | 13.300 ± 3.662 | 14.000 | 22.000 | [11.300, 15.700] | 10/10 |
| write_calls | count | 10 | 2.500 ± 1.025 | 2.000 | 4.000 | [1.900, 3.200] | 10/10 |
| drift_empty_path | count | 10 | 0.300 ± 0.900 | 0.000 | 3.000 | [0.000, 0.900] | 1/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_ws1_mask_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_b1_drift_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tier5c_retries | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| wall_time_s | count | 10 | 205.981 ± 75.401 | 201.515 | 360.030 | [162.512, 254.700] | 10/10 |
